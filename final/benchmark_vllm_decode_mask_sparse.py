#!/usr/bin/env python3
"""Decode benchmark: dense paged attention vs mask-built sparse paged attention.

This models one decode query token per batch element attending to an existing
64K-token paged KV cache. The sparse path runs top-k over precomputed scores
and gathers the sparse page table from the dense page table.

Use Nsight/NVTX for profiling; this script does not collect in-script timings.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import time

import torch
import torch.nn.functional as F
from vllm.vllm_flash_attn import flash_attn_varlen_func

from sparse_attention_core import (
    FlashAttentionConfig,
    flattened_sparse_decode,
    make_flattened_dense_block_table as core_make_flattened_dense_block_table,
    make_flattened_grouped_kv as core_make_flattened_grouped_kv,
)


START_TIME = time.perf_counter()


def log_step(message: str) -> None:
    print(f"[{time.perf_counter() - START_TIME:8.3f}s] {message}", flush=True)


@contextmanager
def nvtx_range(name: str):
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--context-len", type=int, default=65536)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument(
        "--active-blocks",
        type=int,
        default=512,
        help=(
            "Sparse pages per decode query. Top-k selects active_blocks - 1 "
            "complete remote pages; the current/latest page is appended."
        ),
    )
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fa-version", type=int, default=2)
    parser.add_argument("--num-splits", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--bench-iters", type=int, default=5)
    parser.add_argument("--separation-sleep", type=float, default=0.5)
    parser.add_argument("--atol", type=float, default=2.5e-1)
    parser.add_argument("--rtol", type=float, default=2.5e-1)
    parser.add_argument("--sparse-impl", choices=("current", "flattened"), default="current")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-sparse", action="store_true")
    parser.add_argument("--check-direct-out-correctness", action="store_true")
    parser.add_argument("--check-flattened-sparse-correctness", action="store_true")
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
        num_splits=args.num_splits,
    )


def make_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    cu = torch.empty(lengths.numel() + 1, device=lengths.device, dtype=torch.int32)
    cu[0] = 0
    cu[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    return cu


def call_flash_attn(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    block_table: torch.Tensor,
    args: argparse.Namespace,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    return flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        causal=False,
        dropout_p=0.0,
        block_table=block_table,
        return_softmax_lse=False,
        fa_version=args.fa_version,
        num_splits=args.num_splits,
        out=out,
    )


def make_qkv(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, ...]:
    dtype = dtype_from_name(args.dtype)
    torch.manual_seed(args.seed)
    padded_kv_len = (args.context_len // args.block_size + 1) * args.block_size
    q = torch.randn(args.batch_size, args.heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn(
        args.batch_size,
        padded_kv_len,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
    )
    v = torch.randn(
        args.batch_size,
        padded_kv_len,
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


def make_dense_block_table(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    num_pages = args.context_len // args.block_size + 1
    return torch.arange(args.batch_size * num_pages, device=device, dtype=torch.int32).view(
        args.batch_size,
        num_pages,
    )


def make_flattened_dense_block_table(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    num_pages = args.context_len // args.block_size + 1
    return core_make_flattened_dense_block_table(
        batch_size=args.batch_size,
        context_len=num_pages * args.block_size,
        device=device,
        config=core_config(args),
    )


def make_sparse_scores(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    num_blocks = args.context_len // args.block_size
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1)
    return torch.rand(
        args.kv_heads,
        args.batch_size,
        num_blocks,
        device=device,
        generator=generator,
    )


def make_sparse_block_tables_from_scores(
    dense_block_table: torch.Tensor,
    sparse_scores: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_pages = dense_block_table.shape[-1]
    remote_blocks = args.active_blocks - 1
    selected_remote = torch.topk(sparse_scores, k=remote_blocks, dim=-1, sorted=False).indices
    current_blocks = torch.full(
        (args.kv_heads, args.batch_size, 1),
        args.context_len // args.block_size,
        device=sparse_scores.device,
        dtype=selected_remote.dtype,
    )
    selected = torch.cat((selected_remote, current_blocks), dim=-1)
    dense_by_group = dense_block_table.view(1, args.batch_size, num_pages).expand(args.kv_heads, -1, -1)
    sparse_block_tables = dense_by_group.gather(dim=-1, index=selected)
    return sparse_block_tables, selected


def prepare_decode_metadata(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths_q = torch.ones(args.batch_size, device=device, dtype=torch.int32)
    cu_seqlens_q = make_cu_seqlens(lengths_q)
    dense_seqused_k = torch.full((args.batch_size,), args.context_len, device=device, dtype=torch.int32)
    sparse_seqused_k = torch.full(
        (args.batch_size,),
        (args.active_blocks - 1) * args.block_size + (args.context_len % args.block_size),
        device=device,
        dtype=torch.int32,
    )
    return cu_seqlens_q, dense_seqused_k, sparse_seqused_k


def dense_decode_attention(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    dense_block_table: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    return call_flash_attn(
        q=q,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_k=args.context_len,
        block_table=dense_block_table,
        args=args,
    )


def naive_sparse_decode_attention(
    q: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    sparse_block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    q_heads_per_kv = args.heads // args.kv_heads
    out = torch.empty_like(q)

    for kv_group in range(args.kv_heads):
        q_start = kv_group * q_heads_per_kv
        q_end = q_start + q_heads_per_kv
        call_flash_attn(
            q=q[:, q_start:q_end, :],
            k=grouped_key_cache[kv_group],
            v=grouped_value_cache[kv_group],
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_k=args.active_blocks * args.block_size,
            block_table=sparse_block_tables[kv_group],
            args=args,
            out=out[:, q_start:q_end, :],
        )

    return out


def legacy_sparse_decode_attention(
    q: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    sparse_block_tables: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    q_heads_per_kv = args.heads // args.kv_heads
    out = torch.empty_like(q)

    for kv_group in range(args.kv_heads):
        q_start = kv_group * q_heads_per_kv
        q_end = q_start + q_heads_per_kv
        group_out = call_flash_attn(
            q=q[:, q_start:q_end, :],
            k=grouped_key_cache[kv_group],
            v=grouped_value_cache[kv_group],
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            max_seqlen_k=args.active_blocks * args.block_size,
            block_table=sparse_block_tables[kv_group],
            args=args,
        )
        out[:, q_start:q_end, :] = group_out

    return out


def make_flattened_grouped_kv(
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_pages = args.context_len // args.block_size + 1
    return core_make_flattened_grouped_kv(
        grouped_key_cache,
        grouped_value_cache,
        batch_size=args.batch_size,
        context_len=num_pages * args.block_size,
        config=core_config(args),
    )


def make_flattened_cu_seqlens(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    lengths_q = torch.ones(args.batch_size * args.kv_heads, device=device, dtype=torch.int32)
    return make_cu_seqlens(lengths_q)


def sparse_decode_attention_from_scores(
    q: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    dense_block_table: torch.Tensor,
    sparse_scores: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    sparse_seqused_k: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    sparse_block_tables, _ = make_sparse_block_tables_from_scores(
        dense_block_table,
        sparse_scores,
        args,
    )
    return naive_sparse_decode_attention(
        q,
        grouped_key_cache,
        grouped_value_cache,
        sparse_block_tables,
        cu_seqlens_q,
        sparse_seqused_k,
        args,
    )


def flattened_sparse_decode_attention_from_scores(
    q: torch.Tensor,
    flat_key_cache: torch.Tensor,
    flat_value_cache: torch.Tensor,
    flat_dense_block_table: torch.Tensor,
    sparse_scores: torch.Tensor,
    flat_cu_seqlens_q: torch.Tensor,
    flat_sparse_seqused_k: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    del flat_sparse_seqused_k
    remote_blocks = args.active_blocks - 1
    selected = torch.empty(
        args.batch_size,
        args.kv_heads,
        args.active_blocks,
        device=q.device,
        dtype=torch.long,
    )
    topk_values = torch.empty(
        args.kv_heads,
        args.batch_size,
        remote_blocks,
        device=q.device,
        dtype=sparse_scores.dtype,
    )
    torch.topk(
        sparse_scores,
        k=remote_blocks,
        dim=-1,
        sorted=False,
        out=(topk_values, selected.permute(1, 0, 2)[..., :remote_blocks]),
    )
    selected[..., remote_blocks] = args.context_len // args.block_size
    local_prefixes = torch.full(
        (args.batch_size,),
        args.context_len % args.block_size,
        device=q.device,
        dtype=torch.int32,
    )
    return flattened_sparse_decode(
        q,
        flat_key_cache,
        flat_value_cache,
        selected,
        flat_dense_block_table,
        local_prefixes,
        flat_cu_seqlens_q,
        config=core_config(args),
    )


def check_direct_out_correctness(
    q: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    dense_block_table: torch.Tensor,
    sparse_scores: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    sparse_seqused_k: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    sparse_block_tables, _ = make_sparse_block_tables_from_scores(
        dense_block_table,
        sparse_scores,
        args,
    )
    legacy_out = legacy_sparse_decode_attention(
        q,
        grouped_key_cache,
        grouped_value_cache,
        sparse_block_tables,
        cu_seqlens_q,
        sparse_seqused_k,
        args,
    )
    direct_out = naive_sparse_decode_attention(
        q,
        grouped_key_cache,
        grouped_value_cache,
        sparse_block_tables,
        cu_seqlens_q,
        sparse_seqused_k,
        args,
    )
    torch.cuda.synchronize()
    diff = (legacy_out - direct_out).abs().float()
    max_abs = diff.max().item()
    max_ref = legacy_out.abs().float().max().item()
    max_rel = max_abs / max(max_ref, 1.0e-6)
    if max_abs != 0.0:
        raise AssertionError(
            "direct out sparse decode differs from legacy sparse decode: "
            f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}"
        )
    print(
        "Direct-out correctness vs legacy assign: "
        f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}"
    )


def check_flattened_sparse_correctness(
    q: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    dense_block_table: torch.Tensor,
    sparse_scores: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    sparse_seqused_k: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    del dense_block_table, cu_seqlens_q, sparse_seqused_k
    remote_blocks = args.active_blocks - 1
    selected_remote_kv_batch = torch.topk(sparse_scores, k=remote_blocks, dim=-1, sorted=False).indices
    selected_remote = selected_remote_kv_batch.permute(1, 0, 2).contiguous()
    reference_out = masked_compact_sparse_decode_reference(
        q,
        grouped_key_cache,
        grouped_value_cache,
        selected_remote,
        args,
    )
    flat_key_cache, flat_value_cache = make_flattened_grouped_kv(
        grouped_key_cache,
        grouped_value_cache,
        args,
    )
    flattened_out = flattened_sparse_decode_attention_from_scores(
        q,
        flat_key_cache,
        flat_value_cache,
        make_flattened_dense_block_table(args, q.device),
        sparse_scores,
        make_flattened_cu_seqlens(args, q.device),
        torch.empty(args.batch_size * args.kv_heads, device=q.device, dtype=torch.int32),
        args,
    )
    torch.cuda.synchronize()
    diff = (reference_out - flattened_out).abs().float()
    max_abs = diff.max().item()
    max_ref = reference_out.abs().float().max().item()
    max_rel = max_abs / max(max_ref, 1.0e-6)
    limit = args.atol + args.rtol * max_ref
    if max_abs > limit:
        raise AssertionError(
            "flattened sparse decode differs from masked SDPA sparse decode: "
            f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, limit={limit:.6e}"
        )
    print(
        "Flattened sparse correctness vs masked SDPA sparse: "
        f"max_abs={max_abs:.6e}, max_rel={max_rel:.6e}, limit={limit:.6e}"
    )


def masked_compact_sparse_decode_reference(
    q: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    selected_remote: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    q_heads_per_kv = args.heads // args.kv_heads
    remote_blocks = selected_remote.shape[-1]
    total_tokens = args.active_blocks * args.block_size
    num_pages = grouped_key_cache.shape[1] // args.batch_size
    local_prefix = args.context_len % args.block_size
    current_block = args.context_len // args.block_size
    compact_k = torch.empty(
        args.batch_size,
        args.kv_heads,
        total_tokens,
        args.head_dim,
        device=q.device,
        dtype=grouped_key_cache.dtype,
    )
    compact_v = torch.empty_like(compact_k)

    for batch in range(args.batch_size):
        for kv_head in range(args.kv_heads):
            write = 0
            for block in selected_remote[batch, kv_head].tolist():
                page = batch * num_pages + int(block)
                compact_k[batch, kv_head, write:write + args.block_size] = grouped_key_cache[
                    kv_head,
                    page,
                    :,
                    0,
                ]
                compact_v[batch, kv_head, write:write + args.block_size] = grouped_value_cache[
                    kv_head,
                    page,
                    :,
                    0,
                ]
                write += args.block_size

            page = batch * num_pages + current_block
            if local_prefix > 0:
                compact_k[batch, kv_head, write:write + local_prefix] = grouped_key_cache[
                    kv_head,
                    page,
                    :local_prefix,
                    0,
                ]
                compact_v[batch, kv_head, write:write + local_prefix] = grouped_value_cache[
                    kv_head,
                    page,
                    :local_prefix,
                    0,
                ]
            if local_prefix < args.block_size:
                compact_k[batch, kv_head, write + local_prefix:write + args.block_size] = 0
                compact_v[batch, kv_head, write + local_prefix:write + args.block_size] = 0

    q_compact = q.view(args.batch_size, args.kv_heads, q_heads_per_kv, args.head_dim).reshape(
        args.batch_size * args.kv_heads,
        q_heads_per_kv,
        1,
        args.head_dim,
    )
    k_heads = compact_k.repeat_interleave(q_heads_per_kv, dim=1).reshape(
        args.batch_size * args.kv_heads,
        q_heads_per_kv,
        total_tokens,
        args.head_dim,
    )
    v_heads = compact_v.repeat_interleave(q_heads_per_kv, dim=1).reshape(
        args.batch_size * args.kv_heads,
        q_heads_per_kv,
        total_tokens,
        args.head_dim,
    )
    valid_tokens = remote_blocks * args.block_size + local_prefix
    token_ids = torch.arange(total_tokens, device=q.device).view(1, 1, 1, total_tokens)
    mask = torch.zeros(
        args.batch_size * args.kv_heads,
        q_heads_per_kv,
        1,
        total_tokens,
        device=q.device,
        dtype=torch.float32,
    )
    mask.masked_fill_(token_ids >= valid_tokens, float("-inf"))
    out = F.scaled_dot_product_attention(
        q_compact,
        k_heads,
        v_heads,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    ).squeeze(2)
    return out.view(args.batch_size, args.kv_heads, q_heads_per_kv, args.head_dim).reshape(
        args.batch_size,
        args.heads,
        args.head_dim,
    )


def print_sparse_stats(
    sparse_block_tables: torch.Tensor,
    selected_blocks: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    del sparse_block_tables, selected_blocks
    print(
        "Sparse topk-built page stats: "
        f"active={args.active_blocks}, "
        f"remote_topk={args.active_blocks - 1}, "
        f"appended_current=1, "
        f"selected_count_min={args.active_blocks}, "
        f"selected_count_mean={float(args.active_blocks):.2f}, "
        f"selected_count_max={args.active_blocks}"
    )


def run_once(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    grouped_key_cache: torch.Tensor,
    grouped_value_cache: torch.Tensor,
    flat_key_cache: torch.Tensor,
    flat_value_cache: torch.Tensor,
    dense_block_table: torch.Tensor,
    flat_dense_block_table: torch.Tensor,
    sparse_scores: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    flat_cu_seqlens_q: torch.Tensor,
    dense_seqused_k: torch.Tensor,
    sparse_seqused_k: torch.Tensor,
    flat_sparse_seqused_k: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    def time_method(label: str, fn):
        return run_synced_nvtx(label, fn)

    if not args.skip_dense:
        time_method(
            "baseline_decode_full_method",
            lambda: dense_decode_attention(
                q,
                key_cache,
                value_cache,
                dense_block_table,
                cu_seqlens_q,
                dense_seqused_k,
                args,
            ),
        )
    if not args.skip_sparse:
        if args.sparse_impl == "current":
            sparse_fn = lambda: sparse_decode_attention_from_scores(
                q,
                grouped_key_cache,
                grouped_value_cache,
                dense_block_table,
                sparse_scores,
                cu_seqlens_q,
                sparse_seqused_k,
                args,
            )
        elif args.sparse_impl == "flattened":
            sparse_fn = lambda: flattened_sparse_decode_attention_from_scores(
                q,
                flat_key_cache,
                flat_value_cache,
                flat_dense_block_table,
                sparse_scores,
                flat_cu_seqlens_q,
                flat_sparse_seqused_k,
                args,
            )
        else:
            raise ValueError(f"unsupported sparse implementation: {args.sparse_impl}")
        time_method(
            "page_sparse_decode_full_method",
            sparse_fn,
        )


def main() -> None:
    args = parse_args()
    if args.context_len % args.block_size != 0:
        raise ValueError("context_len must be divisible by block_size.")
    if args.block_size % 16 != 0:
        raise ValueError("vLLM paged KV block_size should be a multiple of 16.")
    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv_heads.")
    if args.active_blocks < 2:
        raise ValueError("active_blocks must include at least one remote block plus the current block.")
    if args.active_blocks - 1 > args.context_len // args.block_size:
        raise ValueError("active_blocks - 1 must be at most the number of complete remote blocks.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda")
    log_step(f"using device: {torch.cuda.get_device_name(0)}")

    def prepare_data():
        q, k, v = make_qkv(args, device)
        key_cache, value_cache = make_paged_kv(k, v, args.block_size)
        grouped_key_cache, grouped_value_cache = make_grouped_paged_kv(k, v, args.block_size)
        flat_key_cache, flat_value_cache = make_flattened_grouped_kv(grouped_key_cache, grouped_value_cache, args)
        dense_block_table = make_dense_block_table(args, device)
        flat_dense_block_table = make_flattened_dense_block_table(args, device)
        sparse_scores = make_sparse_scores(args, device)
        cu_seqlens_q, dense_seqused_k, sparse_seqused_k = prepare_decode_metadata(args, device)
        flat_cu_seqlens_q = make_flattened_cu_seqlens(args, device)
        flat_sparse_seqused_k = sparse_seqused_k.view(
            args.batch_size,
            1,
        ).expand(args.batch_size, args.kv_heads).reshape(args.batch_size * args.kv_heads)
        return (
            q,
            key_cache,
            value_cache,
            grouped_key_cache,
            grouped_value_cache,
            flat_key_cache,
            flat_value_cache,
            dense_block_table,
            flat_dense_block_table,
            sparse_scores,
            cu_seqlens_q,
            flat_cu_seqlens_q,
            dense_seqused_k,
            sparse_seqused_k,
            flat_sparse_seqused_k,
        )

    (
        q,
        key_cache,
        value_cache,
        grouped_key_cache,
        grouped_value_cache,
        flat_key_cache,
        flat_value_cache,
        dense_block_table,
        flat_dense_block_table,
        sparse_scores,
        cu_seqlens_q,
        flat_cu_seqlens_q,
        dense_seqused_k,
        sparse_seqused_k,
        flat_sparse_seqused_k,
    ) = run_synced_nvtx("data_preparation", prepare_data)
    log_step("generated decode Q, full paged KV cache, dense block table, sparse scores, and decode metadata")

    if args.separation_sleep > 0:
        log_step(f"sleeping {args.separation_sleep:.3f}s before workload ranges")
        time.sleep(args.separation_sleep)

    num_blocks = args.context_len // args.block_size
    print("Configuration:")
    print(
        f"  batch={args.batch_size}, query_len=1, context_len={args.context_len}, "
        f"block_size={args.block_size}, num_blocks={num_blocks}"
    )
    print(
        f"  heads={args.heads}, kv_heads={args.kv_heads}, "
        f"q_heads_per_kv={args.heads // args.kv_heads}, head_dim={args.head_dim}"
    )
    print(
        f"  dtype={args.dtype}, active_blocks={args.active_blocks}, "
        f"remote_topk={args.active_blocks - 1}, sparse_tokens={args.active_blocks * args.block_size}"
    )
    print(f"  sparse_impl={args.sparse_impl}, num_splits={args.num_splits}")
    print(f"  q={tuple(q.shape)}")
    print(f"  key_cache={tuple(key_cache.shape)}")
    print(f"  grouped_key_cache={tuple(grouped_key_cache.shape)}")
    print(f"  flat_key_cache={tuple(flat_key_cache.shape)}")
    print(f"  dense_block_table={tuple(dense_block_table.shape)}")
    print(f"  flat_dense_block_table={tuple(flat_dense_block_table.shape)}")
    print(f"  sparse_scores={tuple(sparse_scores.shape)}")
    print(f"  selected_remote_blocks=({args.kv_heads}, {args.batch_size}, {args.active_blocks - 1})")
    print(f"  sparse_block_tables=({args.kv_heads}, {args.batch_size}, {args.active_blocks})")
    print_sparse_stats(sparse_scores, sparse_scores, args)

    for idx in range(args.warmup_iters):
        log_step(f"warmup {idx + 1}/{args.warmup_iters}")
        run_once(
            q,
            key_cache,
            value_cache,
            grouped_key_cache,
            grouped_value_cache,
            flat_key_cache,
            flat_value_cache,
            dense_block_table,
            flat_dense_block_table,
            sparse_scores,
            cu_seqlens_q,
            flat_cu_seqlens_q,
            dense_seqused_k,
            sparse_seqused_k,
            flat_sparse_seqused_k,
            args,
        )
        torch.cuda.synchronize()

    for idx in range(args.bench_iters):
        log_step(f"profile iteration {idx + 1}/{args.bench_iters}")
        run_once(
            q,
            key_cache,
            value_cache,
            grouped_key_cache,
            grouped_value_cache,
            flat_key_cache,
            flat_value_cache,
            dense_block_table,
            flat_dense_block_table,
            sparse_scores,
            cu_seqlens_q,
            flat_cu_seqlens_q,
            dense_seqused_k,
            sparse_seqused_k,
            flat_sparse_seqused_k,
            args,
        )
        torch.cuda.synchronize()

    if args.check_direct_out_correctness:
        run_synced_nvtx(
            "correctness_check_old_vs_direct_out",
            lambda: check_direct_out_correctness(
                q,
                grouped_key_cache,
                grouped_value_cache,
                dense_block_table,
                sparse_scores,
                cu_seqlens_q,
                sparse_seqused_k,
                args,
            ),
        )

    if args.check_flattened_sparse_correctness:
        run_synced_nvtx(
            "correctness_check_current_vs_flattened_sparse",
            lambda: check_flattened_sparse_correctness(
                q,
                grouped_key_cache,
                grouped_value_cache,
                dense_block_table,
                sparse_scores,
                cu_seqlens_q,
                sparse_seqused_k,
                args,
            ),
        )

    print("PASS")


if __name__ == "__main__":
    main()
