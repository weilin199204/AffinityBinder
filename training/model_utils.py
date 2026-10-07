from __future__ import print_function
import json, time, os
import shutil
import importlib.util
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
import torch.utils
import torch.utils.checkpoint

import copy
import torch.nn as nn
import torch.nn.functional as F
import random
import itertools


# 标准氨基酸字母表（含 X = 未知残基），与 featurize 内部使用的 alphabet 一致
AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYX'


def featurize(batch, device):
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    B = len(batch)
    lengths = np.array([len(b['seq']) for b in batch], dtype=np.int32) #sum of chain seq lengths
    L_max = max([len(b['seq']) for b in batch])
    # ===== 三个「掩码」的语义区别（不要混淆）=====
    # mask             : 有效残基掩码（1=真实残基，0=尾部 padding）—— 用于排除 padding，所有损失都乘它
    # chain_M          : 设计掩码（1=要生成的残基，0=给定的残基）—— 决定哪些位置要被模型预测
    # chain_encoding_all: 链编号（0,0,...1,1,...2,2...）—— 区分不同链，用于算跨链/链内关系
    # 例：抗原链 A(给定) + 抗体链 B(设计) → mask 全 1；chain_M 在 A=0、B=1；chain_encoding 在 A=0、B=1
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100*np.ones([B, L_max], dtype=np.int32) #residue idx with jumps across chains
    chain_M = np.zeros([B, L_max], dtype=np.int32) #1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones([B, L_max, L_max], dtype=np.int32) #for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros([B, L_max], dtype=np.int32) #integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32) #sequence AAs integers
    init_alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H', 'I', 'J','K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T','U', 'V','W','X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g','h', 'i', 'j','k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't','u', 'v','w','x', 'y', 'z']
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b['masked_list']
        visible_chains = b['visible_list']
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f'seq_chain_{letter}']
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains) #randomly shuffle chain order
        num_chains = b['num_of_chains']
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.zeros(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
            elif letter in masked_chains: 
                chain_seq = b[f'seq_chain_{letter}']
                chain_length = len(chain_seq)
                chain_coords = b[f'coords_chain_{letter}'] #this is a dictionary
                chain_mask = np.ones(chain_length) #0.0 for visible chains
                x_chain = np.stack([chain_coords[c] for c in [f'N_chain_{letter}', f'CA_chain_{letter}', f'C_chain_{letter}', f'O_chain_{letter}']], 1) #[chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c*np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100*(c-1)+np.arange(l0, l1)
                l0 += chain_length
                c+=1
        x = np.concatenate(x_chain_list,0) #[L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(chain_mask_list,0) #[L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list,0)

        l = len(all_sequence)
        x_pad = np.pad(x, [[0,L_max-l], [0,0], [0,0]], 'constant', constant_values=(np.nan, ))
        X[i,:,:,:] = x_pad

        m_pad = np.pad(m, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_M[i,:] = m_pad

        chain_encoding_pad = np.pad(chain_encoding, [[0,L_max-l]], 'constant', constant_values=(0.0, ))
        chain_encoding_all[i,:] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X,(2,3))).astype(np.float32)
    X[isnan] = 0.

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long,device=device)
    S = torch.from_numpy(S).to(dtype=torch.long,device=device)
    X = torch.from_numpy(X).to(dtype=torch.float32, device=device)
    mask = torch.from_numpy(mask).to(dtype=torch.float32, device=device)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32, device=device)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32, device=device)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long, device=device)
    return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all


def loss_nll(S, log_probs, mask):
    """ Negative log probabilities """
    criterion = torch.nn.NLLLoss(reduction='none')
    loss = criterion(
        log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
    ).view(S.size())
    S_argmaxed = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_argmaxed).float()
    loss_av = torch.sum(loss * mask) / torch.sum(mask)
    return loss, loss_av, true_false


def loss_smoothed(S, log_probs, mask, weight=0.1):
    """ Negative log probabilities """
    S_onehot = torch.nn.functional.one_hot(S, 21).float()

    # Label smoothing
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)

    loss = -(S_onehot * log_probs).sum(-1)
    loss_av = torch.sum(loss * mask) / 2000.0 #fixed
    return loss, loss_av


def loss_judge_guided(
    S, log_probs, mask,
    native_judge_score=None,   # [B] 原生序列裁判分数（原始尺度）
    judge_score=None,          # [B] 生成序列裁判分数（原始尺度）
    weight=0.1,                # label smoothing
    normalization_constant=2000.0,
    lamda_judge=0.001,
):
    """
    """
    # 1. 标准 label-smoothed NLL（与 loss_smoothed 相同的平滑方式）
    S_onehot = torch.nn.functional.one_hot(S, 21).float()
    S_onehot = S_onehot + weight / float(S_onehot.size(-1))
    S_onehot = S_onehot / S_onehot.sum(-1, keepdim=True)
    loss_per_res = -(S_onehot * log_probs).sum(-1)  # [B, L]
    nll_standard = torch.sum(loss_per_res * mask) / normalization_constant

    device = mask.device
    B, L = S.shape

    ## 2. 裁判边际损失 —— 推动生成序列拿到高分
    #l_margin = torch.tensor(0.0, device=device)
    #if lamda_judge > 0 and judge_score is not None:
    #    judge_score = judge_score.to(device)
    #    l_margin = torch.nn.functional.relu(judge_target - judge_score).mean()
#
    ## 3. 裁判质量重加权 —— 玻尔兹曼因子加权原生序列
    #nll_quality = torch.tensor(0.0, device=device)
    #if lamda_quality > 0 and native_judge_score is not None:
    #    native_judge_score = native_judge_score.to(device)
    #    V0 = judge_target * 3.0
    #    quality = torch.exp((native_judge_score - judge_target) / V0)
    #    quality = torch.clamp(quality, 0.3, 3.0)
    #    quality = quality.view(-1, 1)
    #    nll_weighted = (loss_per_res * mask) * quality
    #    nll_quality = torch.sum(nll_weighted) / normalization_constant
    #    nll_quality = nll_quality - nll_standard
    S_pred = torch.argmax(log_probs,-1) #[B, L]
    true_false = (S == S_pred).float()

    # 2. 计算奖励信号
    with torch.no_grad():
         # 2.1 生成用于策略梯度的序列
        #S_pred = torch.argmax(log_probs, dim=-1)  # [B, L]
        #2.2 相对奖励：相对于 native 的提升
        if native_judge_score is not None:
            reward = judge_score - native_judge_score -15.0  
            # 标准化奖励
            #reward = (reward - reward.mean()) / (reward.std() + 1e-8)
        else:
            # 如果没有 native score，退化为绝对奖励
            reward = judge_score.clone()
            #reward = (reward - reward.mean()) / (reward.std() + 1e-8)

        
        # 2.3 使用基线（Baseline）减少方差
        #baseline = reward.mean()
        reward_centered = reward #- baseline
        #reward_stats = {
        #    'reward_mean': reward.mean().item(),
        #    'reward_std': reward.std().item(),
        #    'baseline': baseline.item(),
        #    'reward_max': reward.max().item(),
        #    'reward_min': reward.min().item(),
        #}
    
    # 2.4 计算策略梯度损失
    # 获取每个位置真实 token 的 log 概率
    log_probs_flat = log_probs.reshape(-1, log_probs.shape[-1])  # [B*L, 21]
    S_pred_flat = S_pred.reshape(-1)
    mask_flat = mask.reshape(-1)  # [B*L]
    
    # 收集真实 token 的 log 概率
    log_prob_selected = torch.gather(log_probs_flat, 1, S_pred_flat.unsqueeze(1)).squeeze(1)  # [B*L]
    log_prob_selected = log_prob_selected * mask_flat
    
    # 每个序列的奖励扩展到每个 token
    # reward_centered: [B], 扩展到 [B, L]
    reward_expanded = reward_centered.unsqueeze(1).expand(-1, L)  # [B, L]
    reward_expanded = reward_expanded.reshape(-1)  # [B*L]
    reward_expanded = reward_expanded * mask_flat
    
    # 策略梯度损失：-E[log_prob * reward]
    # 注意：这里使用负号因为我们要最小化损失，但希望高 reward 的序列概率更高
    pg_loss = -(log_prob_selected * reward_expanded).sum() / (mask_flat.sum() + 1e-8)
    
    # 添加熵正则化（鼓励探索）
    # 可选：H = -Σ p log p
    #entropy = -(torch.softmax(log_probs_flat, dim=-1) * log_probs_flat).sum(-1)
    #entropy = (entropy * mask_flat).sum() / (mask_flat.sum() + 1e-8)
    # 如果需要，可以添加 entropy_bonus = -0.01 * entropy 来鼓励探索
    
    total_loss = (
        nll_standard
        + lamda_judge *  pg_loss
    )

    loss_dict = {
        "nll_loss": nll_standard.detach(),
        "pg_loss":pg_loss.detach(),
        #"quality_loss": nll_quality.detach(),
        "total_loss": total_loss.detach(),
        "true_false":true_false
    }
    return total_loss, loss_dict


# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E



class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm1(h_V + self.dropout1(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, max_relative_feature=32):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.max_relative_feature = max_relative_feature
        self.linear = nn.Linear(2*max_relative_feature+1+1, num_embeddings)

    def forward(self, offset, mask):
        d = torch.clip(offset + self.max_relative_feature, 0, 2*self.max_relative_feature)*mask + (1-mask)*(2*self.max_relative_feature+1)
        d_onehot = torch.nn.functional.one_hot(d, 2*self.max_relative_feature+1+1)
        E = self.linear(d_onehot.float())
        return E


class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask, residue_idx, chain_labels):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
 
        D_neighbors, E_idx = self._dist(Ca, mask)

        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        offset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]

        d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx



class ProteinMPNN(nn.Module):
    def __init__(self, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        vocab=21, k_neighbors=32, augment_eps=0.1, dropout=0.1):
        super(ProteinMPNN, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        self.features = ProteinFeatures(node_features, edge_features, top_k=k_neighbors, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # 裁判打分器（可选子模块）。默认 None，训练时用 attach_judge() 注入。
        # JudgeScorer 是 nn.Module，挂上后作为子模块：随模型一起 .to(device) /
        # save / load（实现「一次加载整个模型」）。其 ESM2 参数已冻结
        # （requires_grad=False），优化器需用 [p for p in model.parameters()
        #  if p.requires_grad] 过滤，避免给冻结参数建状态、浪费显存。
        self.judge_scorer = None

    def forward(self, X, S, mask, chain_M, residue_idx, chain_encoding_all):
        """ Graph-conditioned sequence model """
        device=X.device
        # Prepare node and edge embeddings
        E, E_idx = self.features(X, mask, residue_idx, chain_encoding_all)
        h_V = torch.zeros((E.shape[0], E.shape[1], E.shape[-1]), device=E.device)
        h_E = self.W_e(E)

        # Encoder is unmasked self-attention
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        for layer in self.encoder_layers:
            #h_V, h_E = torch.utils.checkpoint.checkpoint(layer, h_V, h_E, E_idx, mask, mask_attend)
             h_V, h_E=layer(h_V, h_E, E_idx, mask, mask_attend)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(S)
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx)

        # Build encoder embeddings
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx)
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx)


        chain_M = chain_M*mask #update chain_M to include missing regions
        # 关键：自回归解码顺序。
        #   chain_M==0（给定残基） × 0.0001 → 随机值≈0，排在前面
        #   chain_M==1（设计残基） × 1.0001 → 随机值≈|N(0,1)|~0.8，排在后面
        # argsort 从小到大 → 给定链先被解码（充当上下文），设计链后解码、自回归生成。
        # 即：每个设计残基生成时，能看到「所有给定链 + 之前已生成的设计残基」。
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape, device=device)))) #[numbers will be smaller for places where chain_M = 0.0 and higher for places where chain_M = 1.0]
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size, device=device))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)

        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx)
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            #h_V = torch.utils.checkpoint.checkpoint(layer, h_V, h_ESV, mask)
            h_V =layer(h_V, h_ESV, mask)

        logits = self.W_out(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def attach_judge(self, checkpoint_path, score_file=None, device=None):
        """把裁判打分器作为子模块挂到模型上（可选，训练时用于裁判引导）。

        例：
            model = ProteinMPNN(...).to(device)
            model.attach_judge("judge_checkpoints/best_judge.pt", "judge_model/7155.txt")
            # 之后即可一次性保存/加载整个模型（生成器 + 裁判）：
            #   torch.save(model.state_dict(), "mpnn_with_judge.pt")

        JudgeScorer 是 nn.Module，挂在这里会注册成子模块，随模型一起
        .to(device) / save / load。其 ESM2 参数已冻结（requires_grad=False），
        优化器需过滤掉它们。
        （JudgeScorer/compute_judge_scores 定义在本文件靠后位置，方法体在调用时
        才解析名字，所以这里引用它们没有问题。）
        """
        self.judge_scorer = JudgeScorer(checkpoint_path, score_file, device)

      
    @torch.no_grad()
    def score_with_judge(self, S, X, mask, chain_M):
        """用内置裁判对「原生序列」和「生成序列」分别打分。

        返回 (native_judge_score, judge_score)，均为 [B] 原始 pKd 尺度；
        若未调用 attach_judge() 则返回 (None, None)。

        chain_M 是设计掩码（1=生成链，0=给定链），据此派生出裁判 A/B 双塔
        所需的 chain_ids（1=设计链=A，0=给定链=B），与 judge_model/model.py 配套。
        """
        if self.judge_scorer is None:
            return None, None

        # chain_ids 用 chain_M 直接派生：1=设计链=A(配体)，0=给定链=B(靶点)。
        # chain_M 与 X/S/mask 同源同布局，天然对齐，裁判据此拆分 A/B。
        chain_ids = (chain_M > 0).long()

        # ① 原生序列打分：整条复合物（生成链 + 原始链）都用原生序列 S
        native_seqs = tokens_to_sequences(S, mask)
        return self.judge_scorer.score(native_seqs, X, mask, chain_ids)

 
    
    def train(self, mode=True):
        """切换训练/评估模式。裁判是冻结的预训练模型，必须始终保持 eval，
        否则 model.train() 会把裁判的 dropout 打开、让打分在训练时随机抖动。
        """
        super().train(mode)
        if self.judge_scorer is not None:
            self.judge_scorer.eval()




class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

def get_std_opt(parameters, d_model, step):
    return NoamOpt(
        d_model, 2, 4000, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )


# ============================================================================
# ESM2+EGNN 裁判模型集成（复用 judge_model/model.py 的 JudgeModel）
# 用于实现「裁判引导优化」：玻尔兹曼质量加权 + 边际损失。
#
# 训练循环里的典型用法：
#     judge_scorer = JudgeScorer(
#         checkpoint_path="judge_checkpoints/best_judge.pt",
#         score_file="judge_model/7155.txt",
#     )
#     ...
#     log_probs = model(X, S, mask, chain_M, residue_idx, chain_encoding_all)
#     native_judge_score, judge_score = compute_judge_scores(
#         judge_scorer, S, log_probs, X, mask, chain_M
#     )
#     total_loss, loss_dict = loss_judge_guided(
#         S, log_probs, mask, native_judge_score, judge_score
#     )
#     total_loss.backward()
# ============================================================================

def tokens_to_sequences(S, mask):
    """把 ProteinMPNN 的 token 序列转回氨基酸字符串。

    只取 mask>0 的有效残基，丢弃尾部 padding，这样 ESM2 编码后的长度正好等于
    有效残基数，与结构坐标 X 的布局对齐（padding 都在尾部）。
    """
    S_np = S.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    seqs = []
    for b in range(S.shape[0]):
        seq = ''.join(
            AA_ALPHABET[int(S_np[b, j])]
            for j in range(S.shape[1])
            if mask_np[b, j] > 0
        )
        seqs.append(seq)
    return seqs


class JudgeScorer(nn.Module):
    """ESM2+EGNN 裁判：给定 (序列, 骨架结构) 打分，返回原始尺度分数 [-log(Kd)]。

    复用 judge_model/model.py 的 JudgeModel，加载已训练 checkpoint
    （best_judge.pt / judge_epoch_50.pt）。

    继承 nn.Module：这样把它挂到 ProteinMPNN 上（self.judge_scorer = JudgeScorer(...)）
    时它就是正规子模块，可随 ProteinMPNN 一起 .to(device) / .state_dict() / save / load，
    实现「一次把整个模型（生成器 + 裁判）都加载进去」。
    注意：内部 ESM2 参数全部冻结（requires_grad=False），优化器需过滤掉它们。
    """

    def __init__(self, checkpoint_path, score_file=None, device=None):
        super().__init__()
        self.device = device

        # 复用 judge_model/model.py 里的 JudgeModel。
        # judge_model/ 没有 __init__.py，不能作为包导入；这里按文件路径动态加载，
        # 既避免污染 sys.path，也避免 `from model import ...` 与其它 model 模块重名。
        # 本文件位于 models/mpnn/src/mpnn/trainers/，judge_model 在仓库根目录，
        # 因此向上 6 层 dirname 回到根目录再进 judge_model（与同目录 mpnn.py 一致）。
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )))
        )
        judge_dir = os.path.join("/home/weizg/wei/soft/zyk_foundry-production/", "judge_model")
        spec = importlib.util.spec_from_file_location(
            "judge_model", os.path.join(judge_dir, "model.py")
        )
        judge_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(judge_module)
        JudgeModel = judge_module.JudgeModel

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_config = ckpt.get("config", {})

        self.model = JudgeModel(
            esm_name=saved_config.get("esm_name", "facebook/esm2_t33_650M_UR50D"),
            egnn_hidden_dim=saved_config.get("egnn_hidden_dim", 128),
            egnn_num_layers=saved_config.get("egnn_num_layers", 3),
            fusion_dim=saved_config.get("fusion_dim", 256),
            fusion_heads=saved_config.get("fusion_heads", 4),
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # 归一化参数（优先取 ckpt 顶层，兼容只存 config 的情况）
        score_mean = ckpt.get("score_mean", saved_config.get("score_mean"))
        score_std = ckpt.get("score_std", saved_config.get("score_std"))
        if score_mean is None and score_file:
            scores = []
            with open(score_file) as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        try:
                            scores.append(float(parts[1].strip()))
                        except ValueError:
                            continue
            score_mean = sum(scores) / len(scores) if scores else 0.0
            score_std = (sum((s - score_mean) ** 2 for s in scores) / len(scores)) ** 0.5 if scores else 1.0
        if score_mean is None:
            score_mean = 0.0
        if score_std is None:
            score_std = 1.0

        # 归一化参数注册成 buffer：随裁判一起 .to(device) / save / load
        self.register_buffer("score_mean", torch.tensor(float(score_mean), dtype=torch.float32))
        self.register_buffer("score_std", torch.tensor(float(score_std), dtype=torch.float32))



    @torch.no_grad()
    def score(self, sequences, X, mask, chain_ids=None):
             """返回 [B] 原始尺度分数（A↔B 结合亲和力）。
     
             sequences: List[str]（长度 B，整复合物 = A 链 + B 链拼接）
             X: [B, L, 4, 3] — N/CA/C/O 坐标
             mask: [B, L] — 有效残基掩码（1.0 有效）
             chain_ids: [B, L] int — 1=A(配体/设计链)，0=B(靶点/给定链)，-1=padding；
                        为 None 时退化为「整条复合物当一条链」。
             """
             # ① 保证裁判（含归一化 buffer）和坐标在同一张卡上。
             #    用 self.to() 而不是 self.model.to()，否则 score_mean/score_std
             #    这两个 buffer 会留在旧设备，最后反归一化时 dtype/device 不匹配。
             gpu_device = X.device
             judge_device = next(self.parameters()).device
             if judge_device != gpu_device:
                 self.to(gpu_device)
     
             # ② 只取 4 个主链原子 N/CA/C/O（裁判只用主链坐标）
             coords_bb = X[:, :, :4, :] if X.shape[2] >= 4 else X
     
             # ②.5 坐标预处理：与训练端对齐（judge_model/train.py __getitem__）——
             #   中心化到 CA 质心 + 缩放到 CA RMSD ≤ 15。否则 RBF（范围 2~22Å）会
             #   对未缩放的大复合物饱和，结构分支退化、打分失真。
             ca = coords_bb[:, :, 1, :]                              # [B, L, 3]
             m3 = mask.unsqueeze(-1)                                  # [B, L, 1]
             n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1)     # [B, 1]
             center = (ca * m3).sum(dim=1, keepdim=True) / n_valid    # [B, 1, 3]
             coords_bb = coords_bb - center.unsqueeze(1)
             rmsd = torch.sqrt(((ca * m3) ** 2).sum(dim=(1, 2)) / n_valid.squeeze(1) + 1e-8)  # [B]
             scale = torch.where(rmsd > 15.0, 15.0 / rmsd.clamp(min=1e-8), torch.ones_like(rmsd))
             coords_bb = coords_bb * scale.view(-1, 1, 1, 1)
     
             # ③ 前向：sequences + coords + mask + chain_ids(A/B) → A↔B 亲和力
             # 裁判在 bf16 autocast 下训练（ESM2 权重为 bf16），推理时用同样上下文，
             # 否则 bf16 的 ESM2 输出进入 fp32 的 projection 层会 dtype 不匹配。
             if chain_ids is not None and chain_ids.device != gpu_device:
                 chain_ids = chain_ids.to(gpu_device)
             with torch.autocast(
                 device_type="cuda",
                 dtype=torch.bfloat16,
                 enabled=(gpu_device.type == "cuda"),
             ):
                 output = self.model(sequences, coords_bb, mask, chain_ids)
             # ④ 取全局分数，反归一化回原始 pKd 尺度（训练时是对归一化分数做的回归）
             norm_score = output["global_logits"].squeeze(-1).float()
             return norm_score * self.score_std + self.score_mean

