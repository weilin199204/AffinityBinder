"""ProteinMPNN + A/B 双塔裁判的多卡 Phase 2 训练入口。

本文件负责数据筛选、DDP 同步、裁判调用、优化器更新、checkpoint 和日志。
Loss 公式仍由 model_utils3.loss_judge_guided 定义，本文件不修改其中的
权重、目标值或计算方式。
"""

import argparse
import math
import os
import time
from collections import OrderedDict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from model_utils import (
    ProteinMPNN,
    featurize,
    get_std_opt,
    loss_judge_guided,
    loss_nll,
)
from utils import (
    PDB_dataset,
    build_training_clusters,
    get_pdbs,
    loader_pdb,
    worker_init_fn,
)


AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWYX")


def setup(rank, world_size, timeout_minutes):
    """Initialize one NCCL process for each visible GPU."""
    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed_all(42 + rank)
    torch.cuda.set_device(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout_minutes),
    )


def is_valid_entry(entry, max_protein_length):
    sequence = entry.get("seq", "")
    return (
        bool(sequence)
        and len(sequence) <= max_protein_length
        and set(sequence).issubset(AA_ALPHABET)
    )


def synchronized_entries(pdb, args, device):
    """Return an equal number of valid entries on every rank.

    Every rank must execute the same number of DDP forward/backward operations.
    Synchronizing each outer loader step avoids both rank divergence and storing
    a complete epoch in host memory.
    """
    entries = [
        entry
        for entry in get_pdbs(
            [pdb],
            1,
            args.max_protein_length,
            args.num_examples_per_epoch,
        )
        if is_valid_entry(entry, args.max_protein_length)
    ]
    # 防卡死：不同 rank 的 get_pdbs() 可能返回不同数量的有效 entry。
    # 取 MIN 后，每个 rank 在当前外层 batch 中执行相同次数的 backward。
    common_count = torch.tensor(len(entries), device=device, dtype=torch.long)
    dist.all_reduce(common_count, op=dist.ReduceOp.MIN)
    return entries[: int(common_count.item())]


def has_two_judge_chains(mask, chain_M, chain_encoding_all):
    """Require exactly two physical chains and non-empty A/B judge groups."""
    # A/B 裁判训练数据是双链复合物；同时排除 A 或 B 为空的无效拆分。
    valid = mask > 0
    physical_chains = torch.unique(chain_encoding_all[valid])
    has_a = torch.any(valid & (chain_M > 0))
    has_b = torch.any(valid & (chain_M <= 0))
    return len(physical_chains) == 2 and bool(has_a.item()) and bool(has_b.item())


def mpnn_state_dict(model):
    """Exclude the separately checkpointed frozen judge from MPNN checkpoints."""
    # 裁判由独立 checkpoint 加载，避免每个 MPNN checkpoint 重复保存
    # ESM2+EGNN 权重并造成大文件写盘等待。
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("judge_scorer.")
    }


def load_mpnn_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return None, 0, 0
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = OrderedDict()
    for key, value in checkpoint["model_state_dict"].items():
        clean_key = key[7:] if key.startswith("module.") else key
        if not clean_key.startswith("judge_scorer."):
            state_dict[clean_key] = value
    model.load_state_dict(state_dict, strict=False)
    return checkpoint, int(checkpoint.get("step", 0)), int(checkpoint.get("epoch", 0))


def noam_amp_step(optimizer, scaler):
    """Advance Noam scheduling while stepping its wrapped Adam with AMP."""
    optimizer._step += 1
    rate = optimizer.rate()
    for group in optimizer.optimizer.param_groups:
        group["lr"] = rate
    optimizer._rate = rate
    scaler.step(optimizer.optimizer)


def score_sequences(model, S, log_probs, X, mask, chain_M):
    """Score native complex and A-designed/B-native generated complex."""
    # 原生分数：A、B 均使用结构文件中的原生序列。
    native_score = model.module.score_with_judge(S, X, mask, chain_M)
    predicted_tokens = torch.argmax(log_probs, dim=-1)
    # 生成分数：只替换设计链 A；给定链 B 必须保持原生序列。
    generated_tokens = torch.where(chain_M > 0, predicted_tokens, S)
    generated_score = model.module.score_with_judge(
        generated_tokens, X, mask, chain_M
    )
    return native_score, generated_score


def reduce_sum(value, device):
    tensor = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def gradient_l2_norm(parameters):
    """Measure gradient norm without modifying gradients."""
    # 仅用于训练诊断，不裁剪、不缩放，也不参与 Loss。
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().norm(2).square()
        squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is None:
        return 0.0
    return squared_norm.sqrt().item()


def run_epoch(model, loader, optimizer, scaler, args, device, training_mode):
    model.train(training_mode)
    stats = {
        # 标准 perplexity 使用未重复归一化的逐残基 NLL。
        "nll_sum": 0.0,
        "correct_sum": 0.0,
        "residue_count": 0.0,
        "objective_sum": 0.0,
        "native_score_sum": 0.0,
        "generated_score_sum": 0.0,
        # 以下分项直接读取 loss_judge_guided 的返回值，只记录、不重算 Loss。
        "nll_objective_sum": 0.0,
        "margin_loss_sum": 0.0,
        "quality_loss_sum": 0.0,
        "gradient_norm_sum": 0.0,
        "learning_rate_sum": 0.0,
        "sample_count": 0.0,
        "step_count": 0.0,
    }

    context = torch.enable_grad() if training_mode else torch.no_grad()
    with context:
        for pdb in loader:
            for entry in synchronized_entries(pdb, args, device):
                (
                    X,
                    S,
                    mask,
                    _,
                    chain_M,
                    residue_idx,
                    _,
                    chain_encoding_all,
                ) = featurize([entry], device)
                mask_for_loss = mask * chain_M

                usable = torch.tensor(
                    int(has_two_judge_chains(mask, chain_M, chain_encoding_all)),
                    device=device,
                    dtype=torch.long,
                )
                dist.all_reduce(usable, op=dist.ReduceOp.MIN)
                # 只要任意 rank 的对应样本不是有效双链，所有 rank 同步跳过，
                # 以保证 DDP collective 和 backward 的调用顺序完全一致。
                if usable.item() == 0:
                    continue

                if training_mode:
                    optimizer.zero_grad()

                with torch.cuda.amp.autocast(enabled=args.mixed_precision):
                    log_probs = model(
                        X,
                        S,
                        mask,
                        chain_M,
                        residue_idx,
                        chain_encoding_all,
                    )
                    
                    native_score, generated_score = score_sequences(
                        model, S, log_probs, X, mask, chain_M
                    )
                    # Loss 公式保持原样；loss_dict 只用于记录诊断分项。
                    total_loss, loss_dict = loss_judge_guided(
                        S,
                        log_probs,
                        mask_for_loss,
                        native_score,
                        generated_score,
                    )

                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"rank={dist.get_rank()} encountered non-finite loss"
                    )

                if training_mode:
                    if args.mixed_precision:
                        scaler.scale(total_loss).backward()
                        # AMP 下必须先反缩放，再测量或裁剪真实梯度。
                        scaler.unscale_(optimizer.optimizer)
                        if args.gradient_norm > 0:
                            current_gradient_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.gradient_norm
                            ).item()
                        else:
                            current_gradient_norm = gradient_l2_norm(
                                model.parameters()
                            )
                        noam_amp_step(optimizer, scaler)
                        scaler.update()
                    else:
                        total_loss.backward()
                        if args.gradient_norm > 0:
                            current_gradient_norm = torch.nn.utils.clip_grad_norm_(
                                model.parameters(), args.gradient_norm
                            ).item()
                        else:
                            current_gradient_norm = gradient_l2_norm(
                                model.parameters()
                            )
                        optimizer.step()
                    current_learning_rate = optimizer._rate
                else:
                    current_gradient_norm = 0.0
                    current_learning_rate = optimizer._rate

                # 指标与训练目标分开：perplexity 只由逐残基 NLL 计算，
                # 不使用同时包含 margin/quality 的 total_loss。
                per_residue_nll, _, true_false = loss_nll(
                    S, log_probs.detach(), mask_for_loss
                )
                stats["nll_sum"] += (
                    per_residue_nll * mask_for_loss
                ).sum().item()
                stats["correct_sum"] += (
                    true_false * mask_for_loss
                ).sum().item()
                stats["residue_count"] += mask_for_loss.sum().item()
                stats["objective_sum"] += total_loss.detach().item()
                stats["native_score_sum"] += native_score.sum().item()
                stats["generated_score_sum"] += generated_score.sum().item()
                stats["nll_objective_sum"] += loss_dict["nll_loss"].item()
                stats["margin_loss_sum"] += loss_dict["margin_loss"].item()
                stats["quality_loss_sum"] += loss_dict["quality_loss"].item()
                stats["gradient_norm_sum"] += current_gradient_norm
                stats["learning_rate_sum"] += current_learning_rate
                stats["sample_count"] += native_score.numel()
                stats["step_count"] += 1

    return stats


def summarize_stats(stats, device):
    totals = {key: reduce_sum(value, device) for key, value in stats.items()}
    if totals["residue_count"] <= 0 or totals["step_count"] <= 0:
        raise RuntimeError("No usable two-chain entries were processed")
    mean_nll = totals["nll_sum"] / totals["residue_count"]
    return {
        "perplexity": min(mean_nll, 50.0),
        "accuracy": totals["correct_sum"] / totals["residue_count"],
        "objective": totals["objective_sum"] / totals["step_count"],
        "native_score": totals["native_score_sum"] / totals["sample_count"],
        "generated_score": totals["generated_score_sum"] / totals["sample_count"],
        "nll_objective": totals["nll_objective_sum"] / totals["step_count"],
        "margin_loss": totals["margin_loss_sum"] / totals["step_count"],
        "quality_loss": totals["quality_loss_sum"] / totals["step_count"],
        "gradient_norm": totals["gradient_norm_sum"] / totals["step_count"],
        "learning_rate": totals["learning_rate_sum"] / totals["step_count"],
    }


def save_checkpoint(model, optimizer, epoch, total_step, args, output_dir):
    # 只保存 MPNN 和优化器；裁判恢复时由 judge_checkpoint 单独加载。
    checkpoint = {
        "epoch": epoch,
        "step": total_step,
        "num_edges": args.num_neighbors,
        "noise_level": args.backbone_noise,
        "model_state_dict": mpnn_state_dict(model.module),
        "optimizer_state_dict": optimizer.optimizer.state_dict(),
        "judge_checkpoint": args.judge_checkpoint,
    }
    weights_dir = os.path.join(output_dir, "model_weights")
    torch.save(checkpoint, os.path.join(weights_dir, "epoch_last.pt"))
    if epoch % args.save_model_every_n_epochs == 0:
        torch.save(
            checkpoint,
            os.path.join(weights_dir, f"epoch{epoch}_step{total_step}.pt"),
        )


def training(
    rank,
    world_size,
    args,
    train_set,
    valid_set,
    checkpoint_path,
    output_dir,
    epochs_this_spawn,
):
    setup(rank, world_size, args.ddp_timeout_minutes)
    device = torch.device(f"cuda:{rank}")
    is_main_process = rank == 0

    model = ProteinMPNN(
        node_features=args.hidden_dim,
        edge_features=args.hidden_dim,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        k_neighbors=args.num_neighbors,
        dropout=args.dropout,
        augment_eps=args.backbone_noise,
    )
    checkpoint, total_step, start_epoch = load_mpnn_checkpoint(
        model, checkpoint_path, device
    )
    model.attach_judge(args.judge_checkpoint, args.judge_score_file)
    model.to(device)
    model = DDP(
        model,
        device_ids=[rank],
        find_unused_parameters=False,
        # 裁判被冻结且各 rank 使用同一份权重，无需每次 forward
        # 重复广播其 buffer，可减少不必要的 NCCL 通信。
        broadcast_buffers=False,
    )

    optimizer = get_std_opt(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        args.hidden_dim,
        total_step,
    )
    if checkpoint is not None and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except (ValueError, RuntimeError) as error:
            if is_main_process:
                print(f"Warning: optimizer state not restored: {error}", flush=True)

    scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
    loader_options = {
        "batch_size": 1,
        "shuffle": False,
        "pin_memory": args.pin_memory,
        "num_workers": args.num_workers,
    }
    train_sampler = DistributedSampler(
        train_set, world_size, rank, shuffle=True, drop_last=True
    )
    valid_sampler = DistributedSampler(
        valid_set, world_size, rank, shuffle=False, drop_last=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        sampler=train_sampler,
        worker_init_fn=worker_init_fn,
        **loader_options,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        sampler=valid_sampler,
        worker_init_fn=worker_init_fn,
        **loader_options,
    )

    try:
        for local_epoch in range(epochs_this_spawn):
            epoch = start_epoch + local_epoch + 1
            train_sampler.set_epoch(epoch)
            valid_sampler.set_epoch(epoch)
            start_time = time.time()

            train_stats = run_epoch(
                model, train_loader, optimizer, scaler, args, device, True
            )
            train_summary = summarize_stats(train_stats, device)
            total_step += int(train_stats["step_count"])

            valid_stats = run_epoch(
                model, valid_loader, optimizer, scaler, args, device, False
            )
            valid_summary = summarize_stats(valid_stats, device)

            if is_main_process:
                save_checkpoint(
                    model, optimizer, epoch, total_step, args, output_dir
                )
            # rank 0 写 checkpoint 时，其余 rank 在此等待，防止提前进入
            # 下一次 DDP forward 后表现为 rank 不同步或 NCCL 超时。
            dist.barrier()

            if is_main_process:
                message = (
                    f"epoch={epoch} step={total_step} "
                    f"time={time.time() - start_time:.1f}s "
                    f"train_ppl={train_summary['perplexity']:.3f} "
                    f"valid_ppl={valid_summary['perplexity']:.3f} "
                    f"train_acc={train_summary['accuracy']:.3f} "
                    f"valid_acc={valid_summary['accuracy']:.3f} "
                    f"native_pKd={train_summary['native_score']:.3f} "
                    f"generated_pKd={train_summary['generated_score']:.3f} "
                    f"nll={train_summary['nll_objective']:.5f} "
                    f"margin={train_summary['margin_loss']:.5f} "
                    f"quality={train_summary['quality_loss']:.5f} "
                    f"valid_nll={valid_summary['nll_objective']:.5f} "
                    f"valid_margin={valid_summary['margin_loss']:.5f} "
                    f"valid_quality={valid_summary['quality_loss']:.5f} "
                    f"grad_norm={train_summary['gradient_norm']:.5f} "
                    f"lr={train_summary['learning_rate']:.3e} "
                    f"train_objective={train_summary['objective']:.5f} "
                    f"valid_objective={valid_summary['objective']:.5f}"
                )
                with open(
                    os.path.join(output_dir, "log.txt"),
                    "a",
                    encoding="utf-8",
                ) as handle:
                    handle.write(message + "\n")
                print(message, flush=True)
    finally:
        dist.destroy_process_group()


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("training4.py requires CUDA/NCCL")
    visible_gpus = torch.cuda.device_count()
    world_size = args.world_size or visible_gpus
    if world_size < 1 or world_size > visible_gpus:
        raise ValueError(
            f"world_size={world_size}, but only {visible_gpus} GPUs are visible"
        )
    if args.max_protein_length > 512:
        # 当前裁判训练时 max_seq_len=512；拒绝明显超出训练分布并可能
        # 超过 ESM2 位置长度限制的复合物。
        raise ValueError(
            "max_protein_length must be <= 512 for the current judge checkpoint"
        )
    if args.reload_data_every_n_epochs <= 0:
        raise ValueError("reload_data_every_n_epochs must be positive")

    data_path = args.path_for_training_data
    params = {
        "LIST": os.path.join(data_path, "list.csv"),
        "VAL": os.path.join(data_path, "valid_clusters.txt"),
        "TEST": os.path.join(data_path, "test_clusters.txt"),
        "DIR": data_path,
        "DATCUT": "2030-Jan-01",
        "RESCUT": args.rescut,
        "HOMO": 0.70,
    }
    train, valid, _ = build_training_clusters(params, args.debug)

    output_dir = time.strftime(args.path_for_outputs, time.localtime())
    os.makedirs(os.path.join(output_dir, "model_weights"), exist_ok=True)
    logfile = os.path.join(output_dir, "log.txt")
    if not os.path.exists(logfile):
        with open(logfile, "w", encoding="utf-8") as handle:
            handle.write(
                "JudgeMPNN phase-2 training; judge="
                f"{args.judge_checkpoint}\n"
            )

    completed = 0
    while completed < args.num_epochs:
        checkpoint_path = os.path.join(
            output_dir, "model_weights", "epoch_last.pt"
        )
        if not os.path.exists(checkpoint_path):
            checkpoint_path = args.previous_checkpoint
        epochs_this_spawn = min(
            args.reload_data_every_n_epochs, args.num_epochs - completed
        )
        train_set = PDB_dataset(list(train.keys()), loader_pdb, train, params)
        valid_set = PDB_dataset(list(valid.keys()), loader_pdb, valid, params)
        mp.spawn(
            training,
            args=(
                world_size,
                args,
                train_set,
                valid_set,
                checkpoint_path,
                output_dir,
                epochs_this_spawn,
            ),
            nprocs=world_size,
            join=True,
        )
        completed += epochs_this_spawn


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--path_for_training_data",
        default="/home/weizg/wei/dataset_2026/foundry-production/"
        "pdb_2021aug02/pdb_2021aug02",
    )
    parser.add_argument("--path_for_outputs", default="./exp_020")
    parser.add_argument("--previous_checkpoint", default="")
    parser.add_argument(
        "--judge_checkpoint",
        # 服务器上的最新 A/B 双塔裁判；本地旧 checkpoint 不参与训练。
        default="/home/weizg/wei/dataset_2026/foundry-production/"
        "judge_ckpt260814/best_judge.pt",
    )
    parser.add_argument(
        "--judge_score_file",
        default="/home/weizg/wei/soft/zyk_foundry-production/judge_model/7155.txt",
    )
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--save_model_every_n_epochs", type=int, default=10)
    parser.add_argument("--reload_data_every_n_epochs", type=int, default=2)
    parser.add_argument("--num_examples_per_epoch", type=int, default=1000000)
    parser.add_argument("--max_protein_length", type=int, default=512)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--num_neighbors", type=int, default=48)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--backbone_noise", type=float, default=0.2)
    parser.add_argument("--rescut", type=float, default=3.5)
    parser.add_argument("--gradient_norm", type=float, default=-1.0)
    parser.add_argument("--world_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ddp_timeout_minutes", type=int, default=30)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--mixed_precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    main(parser.parse_args())
