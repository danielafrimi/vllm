#!/usr/bin/env python
"""Temporary profiler for repo vs batched CUDA DSA block summaries.

The measured regions intentionally exclude data preparation and correctness
checks. Each core path is wrapped in an NVTX range and a torch profiler
`record_function` range, with CUDA synchronization before entering and before
leaving the range.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from compare_repo_vs_cuda_block_summaries_jit import (
    make_fake_dsa,
    make_unique_block_table,
    run_repo_per_sequence,
)
from test_cuda_block_summaries_jit import dtype_from_name, load_extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--target-len", type=int, default=4096)
    parser.add_argument("--ragged-window", type=int, default=768)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--q-indexer-dim", type=int, default=96)
    parser.add_argument("--num-physical-blocks", type=int, default=8192)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--profile-repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def make_ragged_lengths(
    *,
    batch: int,
    target_len: int,
    ragged_window: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    values: list[int] = []
    low = max(1, target_len - ragged_window)
    high = target_len + ragged_window
    anchors = [
        target_len,
        target_len - 1,
        target_len + 1,
        target_len - block_size,
        target_len + block_size,
        low,
        high,
    ]
    for value in anchors:
        values.append(max(1, value))
    i = 0
    while len(values) < batch:
        span = max(1, high - low + 1)
        values.append(low + ((i * 251 + 97) % span))
        i += 1
    return torch.tensor(values[:batch], device=device, dtype=torch.long)


def make_case(args: argparse.Namespace):
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    seq_lens = make_ragged_lengths(
        batch=args.batch,
        target_len=args.target_len,
        ragged_window=args.ragged_window,
        block_size=args.block_size,
        device=device,
    )
    max_chunks = math.ceil(int(seq_lens.max().item()) / args.block_size)
    total_chunks = int(
        torch.div(
            seq_lens + args.block_size - 1,
            args.block_size,
            rounding_mode="floor",
        ).sum().item()
    )
    if total_chunks > args.num_physical_blocks:
        raise ValueError(
            f"num_physical_blocks={args.num_physical_blocks} is too small for "
            f"{total_chunks} used chunks"
        )

    key_cache = torch.randn(
        args.num_physical_blocks,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    block_table = make_unique_block_table(
        seq_lens=seq_lens,
        block_size=args.block_size,
        max_chunks=max_chunks,
        num_physical_blocks=args.num_physical_blocks,
        device=device,
        generator=generator,
    )
    return key_cache, block_table, seq_lens


def measured_call(label: str, fn: Callable[[], torch.Tensor]) -> torch.Tensor:
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(label)
    with record_function(label):
        result = fn()
        torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    return result


def cuda_event_time_us(event) -> float:
    for attr in ("cuda_time_total", "device_time_total", "self_cuda_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def is_cuda_event(event) -> bool:
    device_type = str(getattr(event, "device_type", "")).lower()
    return "cuda" in device_type


def parent_chain_contains(event, label: str) -> bool:
    parent = getattr(event, "cpu_parent", None)
    while parent is not None:
        if getattr(parent, "name", None) == label:
            return True
        parent = getattr(parent, "cpu_parent", None)
    return False


def summarize_profile(prof, labels: list[str]) -> None:
    events = list(prof.events())
    for label in labels:
        range_events = [event for event in events if getattr(event, "name", None) == label]
        range_cuda_time = sum(cuda_event_time_us(event) for event in range_events)
        attached_kernels = []
        for event in range_events:
            attached_kernels.extend(getattr(event, "kernels", []) or [])
        if attached_kernels:
            kernel_time = sum(float(getattr(kernel, "duration", 0.0))
                              for kernel in attached_kernels)
            kernel_count = len(attached_kernels)
            unique_names = sorted({getattr(kernel, "name", "<unknown>")
                                   for kernel in attached_kernels})
        else:
            kernel_events = [
                event
                for event in events
                if is_cuda_event(event) and parent_chain_contains(event, label)
            ]
            kernel_time = sum(cuda_event_time_us(event) for event in kernel_events)
            kernel_count = len(kernel_events)
            unique_names = sorted({event.name for event in kernel_events})
        print(
            f"PROFILE label={label} range_cuda_time_us={range_cuda_time:.3f} "
            f"kernel_time_us={kernel_time:.3f} kernel_count={kernel_count} "
            f"unique_kernel_count={len(unique_names)}",
            flush=True,
        )
        for name in unique_names[:20]:
            print(f"  KERNEL {label}: {name}", flush=True)
        if len(unique_names) > 20:
            print(f"  KERNEL {label}: ... {len(unique_names) - 20} more", flush=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA is required")

    key_cache, block_table, seq_lens = make_case(args)
    ext = load_extension()
    fake_dense = make_fake_dsa(
        block_size=args.block_size,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        q_indexer_dim=args.q_indexer_dim,
        use_summary_cache=False,
    )
    fake_summary = make_fake_dsa(
        block_size=args.block_size,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        q_indexer_dim=args.q_indexer_dim,
        use_summary_cache=True,
    )

    def repo_dense() -> torch.Tensor:
        out = None
        for _ in range(args.profile_repeat):
            out = run_repo_per_sequence(
                fake=fake_dense,
                key_cache=key_cache,
                block_table=block_table,
                seq_lens=seq_lens,
            )
        assert out is not None
        return out

    def repo_summary() -> torch.Tensor:
        out = None
        for _ in range(args.profile_repeat):
            fake_summary._reset_dsa_summary_cache()
            out = run_repo_per_sequence(
                fake=fake_summary,
                key_cache=key_cache,
                block_table=block_table,
                seq_lens=seq_lens,
            )
        assert out is not None
        return out

    def batched_cuda() -> torch.Tensor:
        out = None
        for _ in range(args.profile_repeat):
            out = ext.dsa_block_summaries(
                key_cache,
                block_table,
                seq_lens,
                args.q_indexer_dim,
            )
        assert out is not None
        return out

    # Warmup intentionally outside the measured profiler/NVTX ranges.
    for _ in range(args.warmup_iters):
        dense_out = repo_dense()
        summary_out = repo_summary()
        cuda_out = batched_cuda()
    torch.cuda.synchronize()

    labels = [
        "repo_dense_core",
        "repo_summary_core",
        "batched_cuda_core",
    ]
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        with_stack=False,
        profile_memory=False,
    ) as prof:
        dense_out = measured_call(labels[0], repo_dense)
        summary_out = measured_call(labels[1], repo_summary)
        cuda_out = measured_call(labels[2], batched_cuda)

    # Correctness verification intentionally after measured ranges.
    torch.testing.assert_close(dense_out, cuda_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(summary_out, cuda_out, atol=1e-5, rtol=1e-5)
    summarize_profile(prof, labels)
    print(
        "PASS profile repo-vs-CUDA block summaries "
        f"seq_lens={seq_lens.detach().cpu().tolist()} "
        f"block_table_shape={tuple(block_table.shape)} "
        f"key_cache_shape={tuple(key_cache.shape)} "
        f"profile_repeat={args.profile_repeat}",
        flush=True,
    )


if __name__ == "__main__":
    main()
