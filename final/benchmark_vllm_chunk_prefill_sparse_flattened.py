#!/usr/bin/env python3
"""Chunk-prefill sparse attention benchmark with flattened naive sparse path.

K/V cache length is the full context length. Q contains only the final query
chunk, as if we are adding a new prefill chunk to an existing KV cache.

The benchmark compares:

- paged dense causal FlashAttention over Q_chunk x full_KV,
- flattened naive sparse paged FlashAttention with q_len=1 requests,
- transposed sparse FlashAttention with remote KV-block groups plus local causal
  block attention, merged online by LSE.

Use Nsight/NVTX for profiling; this script does not collect in-script timings.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import time
from dataclasses import dataclass

import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

from sparse_attention_core import (
    FlashAttentionConfig,
    flattened_sparse_prefill,
    make_flattened_grouped_kv as core_make_flattened_grouped_kv,
)


START_TIME = time.perf_counter()
NVTX_ENABLED = True


@dataclass(frozen=True)
class ErrorStats:
    max_abs: float
    mean_abs: float
    max_rel: float


def log_step(message: str) -> None:
    print(f"[{time.perf_counter() - START_TIME:8.3f}s] {message}", flush=True)


@contextmanager
def nvtx_range(name: str):
    if not NVTX_ENABLED:
        yield
        return
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def run_synced_nvtx(name: str, fn):
    torch.cuda.synchronize()
    with nvtx_range(name):
        result = fn()
        torch.cuda.synchronize()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-len", type=int, default=65536)
    parser.add_argument("--query-len", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--active-blocks-per-token", type=int, default=1024)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fa-version", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--bench-iters", type=int, default=1)
    parser.add_argument("--naive-request-chunk-size", type=int, default=49152)
    parser.add_argument("--remote-sequence-chunk-size", type=int, default=64)
    parser.add_argument("--local-sequence-chunk-size", type=int, default=32768)
    parser.add_argument("--post-warmup-sleep", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=6e-2)
    parser.add_argument("--rtol", type=float, default=6e-2)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument(
        "--include-flat-dense",
        action="store_true",
        help="Also time the old flat K/V dense chunk baseline.",
    )
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-transposed", action="store_true")
    parser.add_argument(
        "--check-legacy-naive",
        action="store_true",
        help="Also compare the flattened naive path directly against the original per-KV-group naive path.",
    )
    parser.add_argument(
        "--skip-transposed-merge",
        action="store_true",
        help="Time transposed FA calls but do not build the merged output.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def core_config(args: argparse.Namespace) -> FlashAttentionConfig:
    return FlashAttentionConfig(
        block_size=args.block_size,
        heads=args.heads,
        kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        fa_version=args.fa_version,
    )


def make_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    cu = torch.empty(lengths.numel() + 1, device=lengths.device, dtype=torch.int32)
    cu[0] = 0
    cu[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    return cu


def lse_as_rows(softmax_lse: torch.Tensor, total_q: int, num_heads: int) -> torch.Tensor:
    if softmax_lse.shape == (num_heads, total_q):
        return softmax_lse.transpose(0, 1).contiguous()
    if softmax_lse.shape == (total_q, num_heads):
        return softmax_lse.contiguous()
    raise ValueError(f"unexpected LSE shape: {tuple(softmax_lse.shape)}")


def call_flash_attn(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool,
    args: argparse.Namespace,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    out, softmax_lse = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        dropout_p=0.0,
        block_table=block_table,
        return_softmax_lse=True,
        fa_version=args.fa_version,
    )
    return out, lse_as_rows(softmax_lse, q.shape[0], q.shape[1])


def make_qkv(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, ...]:
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    q = torch.randn(
        args.batch_size,
        args.query_len,
        args.heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    k = torch.randn(
        args.batch_size,
        args.context_len,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        args.batch_size,
        args.context_len,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    return q, k, v


def make_paged_kv(k: torch.Tensor, v: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, context_len, kv_heads, head_dim = k.shape
    num_blocks = context_len // block_size
    key_cache = (
        k.view(batch_size, num_blocks, block_size, kv_heads, head_dim)
        .reshape(batch_size * num_blocks, block_size, kv_heads, head_dim)
        .contiguous()
    )
    value_cache = (
        v.view(batch_size, num_blocks, block_size, kv_heads, head_dim)
        .reshape(batch_size * num_blocks, block_size, kv_heads, head_dim)
        .contiguous()
    )
    return key_cache, value_cache


def make_grouped_paged_kv(k: torch.Tensor, v: torch.Tensor, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, context_len, kv_heads, head_dim = k.shape
    num_blocks = context_len // block_size
    key_cache = (
        k.view(batch_size, num_blocks, block_size, kv_heads, head_dim)
        .permute(3, 0, 1, 2, 4)
        .reshape(kv_heads, batch_size * num_blocks, block_size, 1, head_dim)
        .contiguous()
    )
    value_cache = (
        v.view(batch_size, num_blocks, block_size, kv_heads, head_dim)
        .permute(3, 0, 1, 2, 4)
        .reshape(kv_heads, batch_size * num_blocks, block_size, 1, head_dim)
        .contiguous()
    )
    return key_cache, value_cache


def make_flattened_grouped_paged_kv(
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    return core_make_flattened_grouped_kv(
        grouped_key_cache,
        grouped_value_cache,
        batch_size=args.batch_size,
        context_len=args.context_len,
        config=core_config(args),
    )


def query_positions(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    start = args.context_len - args.query_len
    return torch.arange(start, args.context_len, device=device, dtype=torch.int32)


def make_remote_block_scores(
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    num_blocks = args.context_len // args.block_size
    if args.active_blocks_per_token >= num_blocks:
        raise ValueError("active_blocks_per_token must be smaller than num context blocks.")
    if args.active_blocks_per_token < 1:
        raise ValueError("active_blocks_per_token must be at least 1.")

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    return torch.rand(
        args.batch_size,
        args.query_len,
        args.kv_heads,
        num_blocks,
        device=device,
        generator=generator,
    )


def select_remote_block_indices(
    remote_scores: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    remote_slots = args.active_blocks_per_token - 1
    device = remote_scores.device
    q_pos = query_positions(args, device)
    q_blocks = q_pos // args.block_size
    remote_lens = torch.minimum(q_blocks, torch.full_like(q_blocks, remote_slots))
    if remote_slots == 0:
        return (
            torch.empty(args.batch_size, args.query_len, args.kv_heads, 0, device=device, dtype=torch.int32),
            remote_lens,
        )

    num_blocks = args.context_len // args.block_size
    valid_past = (
        torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, 1, 1, num_blocks)
        < q_blocks.view(1, args.query_len, 1, 1)
    )
    masked_scores = remote_scores.masked_fill(~valid_past, float("-inf"))
    selected = torch.topk(masked_scores, k=remote_slots, dim=-1, sorted=False).indices.to(torch.int32)
    slots = torch.arange(remote_slots, device=device, dtype=torch.int32).view(1, 1, 1, remote_slots)
    remote_blocks = torch.where(slots < remote_lens.view(1, args.query_len, 1, 1), selected, -1)
    return remote_blocks.contiguous(), remote_lens.contiguous()


def build_remote_edges(
    remote_blocks: torch.Tensor,
    kv_group: int,
    num_blocks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, query_len, _, remote_slots = remote_blocks.shape
    device = remote_blocks.device
    remote_group = remote_blocks[:, :, kv_group, :]
    valid = remote_group >= 0
    origins = (
        torch.arange(batch_size * query_len, device=device, dtype=torch.long)
        .view(batch_size, query_len, 1)
        .expand(batch_size, query_len, remote_slots)
    )[valid]
    batch_offsets = (
        torch.arange(batch_size, device=device, dtype=torch.int32).view(batch_size, 1, 1)
        * num_blocks
    )
    physical_blocks = (remote_group + batch_offsets).to(torch.int32)[valid]
    return origins.contiguous(), physical_blocks.contiguous()


def print_remote_group_stats(remote_blocks: torch.Tensor, num_blocks: int) -> None:
    batch_size, _, kv_heads, _ = remote_blocks.shape
    total_edges = 0
    count_chunks = []
    for kv_group in range(kv_heads):
        _, physical_blocks = build_remote_edges(remote_blocks, kv_group, num_blocks)
        total_edges += physical_blocks.numel()
        count_chunks.append(
            torch.bincount(physical_blocks.long(), minlength=batch_size * num_blocks)
        )
    counts = torch.cat(count_chunks).float()
    active = counts[counts > 0]
    q = torch.quantile(active, torch.tensor([0.5, 0.9, 0.99], device=active.device))
    print("Remote transposed group lengths:")
    print(
        f"  remote_edges={total_edges}, groups={counts.numel()}, "
        f"active_groups={active.numel()}, empty_groups={int((counts == 0).sum().item())}"
    )
    print(
        f"  min_active={active.min().item():.0f}, max={active.max().item():.0f}, "
        f"mean={active.mean().item():.2f}, std={active.std(unbiased=False).item():.2f}"
    )
    print(f"  p50={q[0].item():.0f}, p90={q[1].item():.0f}, p99={q[2].item():.0f}")


def dense_chunk_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    lengths_q = torch.full((args.batch_size,), args.query_len, device=q.device, dtype=torch.int32)
    lengths_k = torch.full((args.batch_size,), args.context_len, device=q.device, dtype=torch.int32)
    with nvtx_range("dense_chunk_causal"):
        return call_flash_attn(
            q=q.reshape(args.batch_size * args.query_len, args.heads, args.head_dim).contiguous(),
            k=k.reshape(args.batch_size * args.context_len, args.kv_heads, args.head_dim).contiguous(),
            v=v.reshape(args.batch_size * args.context_len, args.kv_heads, args.head_dim).contiguous(),
            cu_seqlens_q=make_cu_seqlens(lengths_q),
            cu_seqlens_k=make_cu_seqlens(lengths_k),
            max_seqlen_q=args.query_len,
            max_seqlen_k=args.context_len,
            causal=True,
            args=args,
        )


def paged_dense_chunk_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = args.context_len // args.block_size
    lengths_q = torch.full((args.batch_size,), args.query_len, device=q.device, dtype=torch.int32)
    seqused_k = torch.full((args.batch_size,), args.context_len, device=q.device, dtype=torch.int32)
    block_table = torch.arange(
        args.batch_size * num_blocks,
        device=q.device,
        dtype=torch.int32,
    ).view(args.batch_size, num_blocks)
    with nvtx_range("paged_dense_chunk_causal"):
        return call_flash_attn(
            q=q.reshape(args.batch_size * args.query_len, args.heads, args.head_dim).contiguous(),
            k=key_cache,
            v=value_cache,
            cu_seqlens_q=make_cu_seqlens(lengths_q),
            seqused_k=seqused_k,
            max_seqlen_q=args.query_len,
            max_seqlen_k=args.context_len,
            causal=True,
            block_table=block_table,
            args=args,
        )


def naive_sparse_attention(
    q: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    remote_blocks: torch.Tensor,
    remote_lens: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = args.context_len // args.block_size
    q_heads_per_kv = args.heads // args.kv_heads
    remote_slots = args.active_blocks_per_token - 1
    num_requests = args.batch_size * args.query_len
    device = q.device

    out = torch.empty_like(q)
    all_lse = torch.empty(args.batch_size, args.query_len, args.heads, device=device, dtype=torch.float32)
    q_pos = query_positions(args, device)
    q_blocks = q_pos // args.block_size
    local_prefix = q_pos % args.block_size + 1
    batch_offsets = torch.arange(args.batch_size, device=device, dtype=torch.int32).view(args.batch_size, 1) * num_blocks
    current_blocks = (batch_offsets + q_blocks.view(1, args.query_len)).reshape(num_requests)
    seqused_k = (
        remote_lens.view(1, args.query_len).expand(args.batch_size, args.query_len).reshape(num_requests)
        * args.block_size
        + local_prefix.view(1, args.query_len).expand(args.batch_size, args.query_len).reshape(num_requests)
    ).to(torch.int32)

    with nvtx_range("naive_simple_sparse"):
      for kv_group in range(args.kv_heads):
        q_start = kv_group * q_heads_per_kv
        q_end = q_start + q_heads_per_kv
        q_group = q[:, :, q_start:q_end, :].reshape(num_requests, q_heads_per_kv, args.head_dim)
        key_group = grouped_key_cache[kv_group]
        value_group = grouped_value_cache[kv_group]

        block_table = torch.zeros(num_requests, args.active_blocks_per_token, device=device, dtype=torch.int32)
        if remote_slots > 0:
            remote_group = remote_blocks[:, :, kv_group, :].reshape(num_requests, remote_slots)
            request_batch_offsets = (
                torch.arange(args.batch_size, device=device, dtype=torch.int32)
                .view(args.batch_size, 1, 1)
                .expand(args.batch_size, args.query_len, remote_slots)
                .reshape(num_requests, remote_slots)
                * num_blocks
            )
            block_table[:, :remote_slots] = torch.where(
                remote_group >= 0,
                remote_group + request_batch_offsets,
                torch.zeros_like(remote_group),
            )
        scatter_cols = remote_lens.view(1, args.query_len).expand(args.batch_size, args.query_len).reshape(num_requests).long()
        block_table.scatter_(1, scatter_cols.unsqueeze(1), current_blocks.unsqueeze(1))

        out_chunks = []
        lse_chunks = []
        for start in range(0, num_requests, args.naive_request_chunk_size):
            end = min(start + args.naive_request_chunk_size, num_requests)
            lengths = torch.ones(end - start, device=device, dtype=torch.int32)
            chunk_out, chunk_lse = call_flash_attn(
                q=q_group[start:end],
                k=key_group,
                v=value_group,
                cu_seqlens_q=make_cu_seqlens(lengths),
                seqused_k=seqused_k[start:end],
                max_seqlen_q=1,
                max_seqlen_k=args.active_blocks_per_token * args.block_size,
                causal=False,
                block_table=block_table[start:end],
                args=args,
            )
            out_chunks.append(chunk_out)
            lse_chunks.append(chunk_lse)

        group_out = torch.cat(out_chunks, dim=0)
        group_lse = torch.cat(lse_chunks, dim=0)
        out[:, :, q_start:q_end, :] = group_out.reshape(args.batch_size, args.query_len, q_heads_per_kv, args.head_dim)
        all_lse[:, :, q_start:q_end] = group_lse.reshape(args.batch_size, args.query_len, q_heads_per_kv)

      return out, all_lse


def flattened_naive_sparse_attention(
    q: torch.Tensor,
    flat_key_cache: torch.Tensor,
    flat_value_cache: torch.Tensor,
    remote_blocks: torch.Tensor,
    remote_lens: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    with nvtx_range("naive_simple_sparse"):
        return flattened_sparse_prefill(
            q,
            flat_key_cache,
            flat_value_cache,
            remote_blocks,
            remote_lens,
            context_len=args.context_len,
            active_blocks_per_token=args.active_blocks_per_token,
            request_chunk_size=args.naive_request_chunk_size,
            config=core_config(args),
        )


def update_merge_state(
    numerator: torch.Tensor,
    denom: torch.Tensor,
    running_lse: torch.Tensor,
    origins: torch.Tensor,
    partial_out: torch.Tensor,
    partial_lse: torch.Tensor,
) -> None:
    num_partials, num_heads, head_dim = partial_out.shape
    device = partial_out.device
    head_offsets = torch.arange(num_heads, device=device, dtype=torch.long)
    flat_indices = origins.long().unsqueeze(1) * num_heads + head_offsets
    flat_indices_1d = flat_indices.reshape(-1)
    lse_1d = partial_lse.float().reshape(-1)

    chunk_max = torch.full_like(running_lse, float("-inf"))
    chunk_max.scatter_reduce_(0, flat_indices_1d, lse_1d, reduce="amax", include_self=True)
    touched = chunk_max > float("-inf")
    new_lse = torch.maximum(running_lse[touched], chunk_max[touched])
    old_scale = torch.exp(running_lse[touched] - new_lse)
    denom[touched] *= old_scale
    numerator[touched] *= old_scale.unsqueeze(-1)
    running_lse[touched] = new_lse

    weights = torch.exp(partial_lse.float() - running_lse[flat_indices_1d].view(num_partials, num_heads))
    denom.scatter_add_(0, flat_indices_1d, weights.reshape(-1))
    numerator.scatter_add_(
        0,
        flat_indices_1d.unsqueeze(-1).expand(-1, head_dim),
        (weights.unsqueeze(-1) * partial_out.float()).reshape(-1, head_dim),
    )


def add_local_states(
    q: torch.Tensor,
    key_group: torch.Tensor,
    value_group: torch.Tensor,
    kv_group: int,
    numerator: torch.Tensor,
    denom: torch.Tensor,
    running_lse: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    q_heads_per_kv = args.heads // args.kv_heads
    q_start = kv_group * q_heads_per_kv
    q_end = q_start + q_heads_per_kv
    num_blocks = args.context_len // args.block_size
    query_start_block = (args.context_len - args.query_len) // args.block_size
    query_blocks = args.query_len // args.block_size
    num_sequences = args.batch_size * query_blocks

    q_local = (
        q.view(args.batch_size, query_blocks, args.block_size, args.heads, args.head_dim)[:, :, :, q_start:q_end, :]
        .reshape(args.batch_size * args.query_len, q_heads_per_kv, args.head_dim)
        .contiguous()
    )
    block_ids = (
        torch.arange(args.batch_size, device=q.device, dtype=torch.int32).view(args.batch_size, 1) * num_blocks
        + torch.arange(query_start_block, query_start_block + query_blocks, device=q.device, dtype=torch.int32).view(1, query_blocks)
    ).reshape(num_sequences, 1)
    origins = torch.arange(args.batch_size * args.query_len, device=q.device, dtype=torch.long)

    for start in range(0, num_sequences, args.local_sequence_chunk_size):
        end = min(start + args.local_sequence_chunk_size, num_sequences)
        row_start = start * args.block_size
        row_end = end * args.block_size
        lengths = torch.full((end - start,), args.block_size, device=q.device, dtype=torch.int32)
        out, lse = call_flash_attn(
            q=q_local[row_start:row_end],
            k=key_group,
            v=value_group,
            cu_seqlens_q=make_cu_seqlens(lengths),
            seqused_k=lengths,
            max_seqlen_q=args.block_size,
            max_seqlen_k=args.block_size,
            causal=True,
            block_table=block_ids[start:end],
            args=args,
        )
        if not args.skip_transposed_merge:
            update_merge_state(numerator, denom, running_lse, origins[row_start:row_end], out, lse)


def add_remote_states(
    q: torch.Tensor,
    key_group: torch.Tensor,
    value_group: torch.Tensor,
    remote_blocks: torch.Tensor,
    kv_group: int,
    numerator: torch.Tensor,
    denom: torch.Tensor,
    running_lse: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    num_blocks = args.context_len // args.block_size
    q_heads_per_kv = args.heads // args.kv_heads
    q_start = kv_group * q_heads_per_kv
    q_end = q_start + q_heads_per_kv

    origins, physical_blocks = build_remote_edges(remote_blocks, kv_group, num_blocks)
    order = torch.argsort(physical_blocks)
    sorted_blocks = physical_blocks[order]
    sorted_origins = origins[order]
    unique_blocks, counts = torch.unique_consecutive(sorted_blocks, return_counts=True)
    counts_i32 = counts.to(torch.int32)
    row_offsets = make_cu_seqlens(counts_i32)
    block_table = unique_blocks.to(torch.int32).view(-1, 1)
    q_group = q[:, :, q_start:q_end, :].reshape(args.batch_size * args.query_len, q_heads_per_kv, args.head_dim)

    for seq_start in range(0, unique_blocks.numel(), args.remote_sequence_chunk_size):
        seq_end = min(seq_start + args.remote_sequence_chunk_size, unique_blocks.numel())
        row_start = int(row_offsets[seq_start].item())
        row_end = int(row_offsets[seq_end].item())
        chunk_counts = counts_i32[seq_start:seq_end]
        chunk_origins = sorted_origins[row_start:row_end]
        q_remote = q_group[chunk_origins].contiguous()
        seqused_k = torch.full((seq_end - seq_start,), args.block_size, device=q.device, dtype=torch.int32)
        out, lse = call_flash_attn(
            q=q_remote,
            k=key_group,
            v=value_group,
            cu_seqlens_q=make_cu_seqlens(chunk_counts),
            seqused_k=seqused_k,
            max_seqlen_q=int(chunk_counts.max().item()),
            max_seqlen_k=args.block_size,
            causal=False,
            block_table=block_table[seq_start:seq_end],
            args=args,
        )
        if not args.skip_transposed_merge:
            update_merge_state(numerator, denom, running_lse, chunk_origins, out, lse)


def transposed_sparse_attention(
    q: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    remote_blocks: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    with nvtx_range("transposed_sparse"):
        num_tokens = args.batch_size * args.query_len
        q_heads_per_kv = args.heads // args.kv_heads
        final = None
        final_lse = None
        if not args.skip_transposed_merge:
            final = torch.empty(num_tokens, args.heads, args.head_dim, device=q.device, dtype=torch.float32)
            final_lse = torch.empty(num_tokens, args.heads, device=q.device, dtype=torch.float32)

        for kv_group in range(args.kv_heads):
            key_group = grouped_key_cache[kv_group]
            value_group = grouped_value_cache[kv_group]
            numerator = torch.zeros(num_tokens * q_heads_per_kv, args.head_dim, device=q.device, dtype=torch.float32)
            denom = torch.zeros(num_tokens * q_heads_per_kv, device=q.device, dtype=torch.float32)
            running_lse = torch.full((num_tokens * q_heads_per_kv,), float("-inf"), device=q.device, dtype=torch.float32)

            with nvtx_range("transposed_local_sparse"):
                add_local_states(q, key_group, value_group, kv_group, numerator, denom, running_lse, args)
            with nvtx_range("transposed_remote_sparse"):
                add_remote_states(q, key_group, value_group, remote_blocks, kv_group, numerator, denom, running_lse, args)

            if not args.skip_transposed_merge:
                q_start = kv_group * q_heads_per_kv
                q_end = q_start + q_heads_per_kv
                assert final is not None and final_lse is not None
                group_out = numerator / denom.clamp_min(1e-30).unsqueeze(-1)
                group_lse = running_lse + torch.log(denom.clamp_min(1e-30))
                final[:, q_start:q_end, :] = group_out.view(num_tokens, q_heads_per_kv, args.head_dim)
                final_lse[:, q_start:q_end] = group_lse.view(num_tokens, q_heads_per_kv)

        if args.skip_transposed_merge:
            return (
                torch.empty(0, device=q.device, dtype=torch.float32),
                torch.empty(0, device=q.device, dtype=torch.float32),
            )

        assert final is not None and final_lse is not None
        return (
            final.view(args.batch_size, args.query_len, args.heads, args.head_dim),
            final_lse.view(args.batch_size, args.query_len, args.heads),
        )


def error_stats(actual: torch.Tensor, expected: torch.Tensor) -> ErrorStats:
    abs_err = (actual.float() - expected.float()).abs()
    rel_err = abs_err / expected.float().abs().clamp_min(1e-6)
    return ErrorStats(abs_err.max().item(), abs_err.mean().item(), rel_err.max().item())


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor, args: argparse.Namespace) -> None:
    stats = error_stats(actual, expected)
    print(
        f"{name}: max_abs={stats.max_abs:.6g}, "
        f"mean_abs={stats.mean_abs:.6g}, max_rel={stats.max_rel:.6g}"
    )
    torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)


def check_legacy_naive(
    q: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    flat_key_cache: torch.Tensor,
    flat_value_cache: torch.Tensor,
    remote_scores: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    remote_blocks, remote_lens = select_remote_block_indices(remote_scores, args)
    legacy_out, legacy_lse = naive_sparse_attention(
        q,
        grouped_key_cache,
        grouped_value_cache,
        remote_blocks,
        remote_lens,
        args,
    )
    flat_out, flat_lse = flattened_naive_sparse_attention(
        q,
        flat_key_cache,
        flat_value_cache,
        remote_blocks,
        remote_lens,
        args,
    )
    assert_close("flattened_naive_out vs legacy_naive_out", flat_out, legacy_out, args)
    assert_close("flattened_naive_lse vs legacy_naive_lse", flat_lse, legacy_lse, args)


def run_once(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    flat_key_cache: torch.Tensor,
    flat_value_cache: torch.Tensor,
    remote_scores: torch.Tensor,
    args: argparse.Namespace,
):
    def time_method(label: str, fn):
        return run_synced_nvtx(label, fn)

    if not args.skip_dense:
        time_method(
            "baseline_full_method",
            lambda: paged_dense_chunk_attention(q, key_cache, value_cache, args),
        )
        if args.include_flat_dense:
            time_method(
                "flat_dense_full_method",
                lambda: dense_chunk_attention(q, k, v, args),
            )
    def flattened_sparse_method():
        remote_blocks, remote_lens = select_remote_block_indices(remote_scores, args)
        naive_out, naive_lse = flattened_naive_sparse_attention(
            q,
            flat_key_cache,
            flat_value_cache,
            remote_blocks,
            remote_lens,
            args,
        )
        return naive_out, naive_lse, remote_blocks

    naive_out, naive_lse, remote_blocks = time_method(
        "page_sparse_full_method",
        flattened_sparse_method,
    )
    if args.skip_transposed:
        transposed_out = torch.empty(0, device=q.device, dtype=torch.float32)
        transposed_lse = torch.empty(0, device=q.device, dtype=torch.float32)
    else:
        transposed_out, transposed_lse = time_method(
            "transpose_sparse_full_method",
            lambda: transposed_sparse_attention(q, grouped_key_cache, grouped_value_cache, remote_blocks, args),
        )
    return naive_out, naive_lse, transposed_out, transposed_lse


def main() -> None:
    global NVTX_ENABLED
    args = parse_args()
    if args.context_len % args.block_size != 0 or args.query_len % args.block_size != 0:
        raise ValueError("context_len and query_len must be divisible by block_size.")
    if args.context_len < args.query_len:
        raise ValueError("context_len must be >= query_len.")
    if (args.context_len - args.query_len) % args.block_size != 0:
        raise ValueError("query chunk must start at a block boundary.")
    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv_heads.")
    if args.skip_transposed_merge and not args.skip_compare:
        raise ValueError("--skip-transposed-merge requires --skip-compare.")
    if args.skip_transposed and not args.skip_compare:
        raise ValueError("--skip-transposed requires --skip-compare.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")
    log_step(f"using device: {torch.cuda.get_device_name(0)}")

    def prepare_data():
        q, k, v = make_qkv(args, device)
        key_cache, value_cache = make_paged_kv(k, v, args.block_size)
        grouped_key_cache, grouped_value_cache = make_grouped_paged_kv(k, v, args.block_size)
        flat_key_cache, flat_value_cache = make_flattened_grouped_paged_kv(grouped_key_cache, grouped_value_cache, args)
        remote_scores = make_remote_block_scores(args, device)
        return (
            q,
            k,
            v,
            key_cache,
            value_cache,
            grouped_key_cache,
            grouped_value_cache,
            flat_key_cache,
            flat_value_cache,
            remote_scores,
        )

    (
        q,
        k,
        v,
        key_cache,
        value_cache,
        grouped_key_cache,
        grouped_value_cache,
        flat_key_cache,
        flat_value_cache,
        remote_scores,
    ) = run_synced_nvtx("data_preparation", prepare_data)
    log_step("generated Q chunk, full K/V cache, and sparse score inputs")

    num_blocks = args.context_len // args.block_size
    print("Configuration:")
    print(
        f"  batch={args.batch_size}, context_len={args.context_len}, "
        f"query_len={args.query_len}, block_size={args.block_size}, num_blocks={num_blocks}"
    )
    print(
        f"  heads={args.heads}, kv_heads={args.kv_heads}, "
        f"q_heads_per_kv={args.heads // args.kv_heads}, head_dim={args.head_dim}"
    )
    print(
        f"  dtype={args.dtype}, active_blocks_per_token={args.active_blocks_per_token}, "
        f"query_start={args.context_len - args.query_len}"
    )
    print(f"  q={tuple(q.shape)}")
    print(f"  key_cache={tuple(key_cache.shape)}")
    print(f"  grouped_key_cache={tuple(grouped_key_cache.shape)}")
    print(f"  flat_key_cache={tuple(flat_key_cache.shape)}")
    print(f"  remote_scores={tuple(remote_scores.shape)}")

    for idx in range(args.warmup_iters):
        log_step(f"warmup {idx + 1}/{args.warmup_iters}")
        NVTX_ENABLED = False
        run_once(
            q,
            k,
            v,
            key_cache,
            value_cache,
            grouped_key_cache,
            grouped_value_cache,
            flat_key_cache,
            flat_value_cache,
            remote_scores,
            args,
        )
        torch.cuda.synchronize()
        NVTX_ENABLED = True

    if args.warmup_iters > 0 and args.post_warmup_sleep > 0:
        log_step(f"sleeping {args.post_warmup_sleep:.3f}s after warmup")
        time.sleep(args.post_warmup_sleep)

    last = None
    for idx in range(args.bench_iters):
        log_step(f"profile iteration {idx + 1}/{args.bench_iters}")
        last = run_once(
            q,
            k,
            v,
            key_cache,
            value_cache,
            grouped_key_cache,
            grouped_value_cache,
            flat_key_cache,
            flat_value_cache,
            remote_scores,
            args,
        )
        torch.cuda.synchronize()

    if last is None:
        raise ValueError("bench_iters must be at least 1.")
    naive_out, naive_lse, transposed_out, transposed_lse = last
    if not args.skip_compare:
        def correctness_check():
            assert_close("transposed_out vs naive_out", transposed_out, naive_out.float(), args)
            assert_close("transposed_lse vs naive_lse", transposed_lse, naive_lse.float(), args)
            if args.check_legacy_naive:
                check_legacy_naive(
                    q,
                    grouped_key_cache,
                    grouped_value_cache,
                    flat_key_cache,
                    flat_value_cache,
                    remote_scores,
                    args,
                )
        run_synced_nvtx("correctness_only", correctness_check)
    print("PASS")


if __name__ == "__main__":
    main()
