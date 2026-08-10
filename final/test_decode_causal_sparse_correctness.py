#!/usr/bin/env python3
"""Decode sparse causal-block correctness probe.

This is intentionally standalone from the benchmarks, but it uses the shared
core table builder so the tested decode semantics match the production path.

The tested layout is:

    [top-k full remote blocks, excluding current partial block] + [current block]

with seqused_k truncating the appended current block to the causal prefix.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from vllm.vllm_flash_attn import flash_attn_varlen_func

from sparse_attention_core import (
    FlashAttentionConfig,
    build_decode_sparse_block_table,
    make_decode_sparse_seqused_k,
)


@dataclass(frozen=True)
class Case:
    batch_size: int
    past_lens: tuple[int, ...]
    block_size: int
    heads: int
    kv_heads: int
    head_dim: int
    active_blocks: int

    @property
    def max_past_len(self) -> int:
        return max(self.past_lens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--past-lens",
        type=int,
        nargs="+",
        default=[1024, 1024 + 7, 2048 + 1, 4096 + 15],
        help="KV-cache token counts before the decode query token.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--active-blocks", type=int, default=8)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fa-version", type=int, default=2)
    parser.add_argument("--atol", type=float, default=2.5e-1)
    parser.add_argument("--rtol", type=float, default=2.5e-1)
    parser.add_argument(
        "--skip-buggy-check",
        action="store_true",
        help="Skip the intentionally buggy current-block-as-full negative check.",
    )
    parser.add_argument(
        "--include-ragged-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also test a single batch containing different per-row past lengths.",
    )
    parser.add_argument(
        "--ragged-past-lens",
        type=int,
        nargs="+",
        default=[1024, 1031, 2049, 4111],
        help="Per-batch past lengths for the ragged-batch case.",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float16 if name == "fp16" else torch.bfloat16


def make_cu_seqlens(lengths: torch.Tensor) -> torch.Tensor:
    cu = torch.empty(lengths.numel() + 1, device=lengths.device, dtype=torch.int32)
    cu[0] = 0
    cu[1:] = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    return cu


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


def make_qkv(case: Case, dtype: torch.dtype, device: torch.device, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    q = torch.randn(
        case.batch_size,
        case.heads,
        case.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    k = torch.randn(
        case.batch_size,
        case.max_past_len,
        case.kv_heads,
        case.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    v = torch.randn(
        case.batch_size,
        case.max_past_len,
        case.kv_heads,
        case.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    for batch, past_len in enumerate(case.past_lens):
        if past_len < case.max_past_len:
            k[batch, past_len:] = 8.0 * torch.randn_like(k[batch, past_len:])
            v[batch, past_len:] = 8.0 * torch.randn_like(v[batch, past_len:])
    return q, k, v


def make_padded_paged_kv(
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int,
    *,
    min_pages: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, max_past_len, kv_heads, head_dim = k.shape
    num_pages = max((max_past_len + block_size - 1) // block_size, min_pages)
    padded_len = num_pages * block_size
    if padded_len != max_past_len:
        pad_shape = (batch_size, padded_len - max_past_len, kv_heads, head_dim)
        k_pad = 8.0 * torch.randn(pad_shape, device=k.device, dtype=k.dtype)
        v_pad = 8.0 * torch.randn(pad_shape, device=v.device, dtype=v.dtype)
        k = torch.cat([k, k_pad], dim=1)
        v = torch.cat([v, v_pad], dim=1)
    key_cache = (
        k.view(batch_size, num_pages, block_size, kv_heads, head_dim)
        .permute(3, 0, 1, 2, 4)
        .reshape(kv_heads, batch_size * num_pages, block_size, 1, head_dim)
        .contiguous()
    )
    value_cache = (
        v.view(batch_size, num_pages, block_size, kv_heads, head_dim)
        .permute(3, 0, 1, 2, 4)
        .reshape(kv_heads, batch_size * num_pages, block_size, 1, head_dim)
        .contiguous()
    )
    flat_key_cache = (
        key_cache.view(kv_heads, batch_size, num_pages, block_size, 1, head_dim)
        .permute(1, 0, 2, 3, 4, 5)
        .reshape(batch_size * kv_heads * num_pages, block_size, 1, head_dim)
    )
    flat_value_cache = (
        value_cache.view(kv_heads, batch_size, num_pages, block_size, 1, head_dim)
        .permute(1, 0, 2, 3, 4, 5)
        .reshape(batch_size * kv_heads * num_pages, block_size, 1, head_dim)
    )
    return flat_key_cache.contiguous(), flat_value_cache.contiguous()


def select_remote_blocks(
    scores: torch.Tensor,
    remote_count: int,
) -> torch.Tensor:
    selected = torch.empty(
        *scores.shape[:-1],
        remote_count,
        device=scores.device,
        dtype=torch.long,
    )
    values = torch.empty(
        *scores.shape[:-1],
        remote_count,
        device=scores.device,
        dtype=scores.dtype,
    )
    torch.topk(scores, k=remote_count, dim=-1, sorted=False, out=(values, selected))
    return selected


def make_remote_scores(
    case: Case,
    current_blocks: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    max_remote_blocks = int(current_blocks.max().item())
    scores = torch.rand(case.kv_heads, case.batch_size, max_remote_blocks, device=device, generator=generator)
    valid = torch.arange(max_remote_blocks, device=device).view(1, 1, max_remote_blocks) < current_blocks.view(
        1,
        case.batch_size,
        1,
    )
    return scores.masked_fill(~valid, float("-inf"))


def make_full_decode_block_table(
    *,
    batch_size: int,
    kv_heads: int,
    num_pages: int,
    device: torch.device,
) -> torch.Tensor:
    flat_rows = batch_size * kv_heads
    return torch.arange(flat_rows * num_pages, device=device, dtype=torch.int32).view(flat_rows, num_pages)


def make_buggy_decode_block_table_with_current_full(
    full_block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
) -> torch.Tensor:
    block_table = build_decode_sparse_block_table(
        full_block_table,
        selected_blocks,
    )
    return block_table


def sparse_decode_with_table(
    q: torch.Tensor,
    flat_key_cache: torch.Tensor,
    flat_value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
    *,
    max_seqlen_k: int,
    case: Case,
    fa_version: int,
) -> torch.Tensor:
    q_heads_per_kv = case.heads // case.kv_heads
    q_flat = q.view(case.batch_size, case.kv_heads, q_heads_per_kv, case.head_dim).reshape(
        case.batch_size * case.kv_heads,
        q_heads_per_kv,
        case.head_dim,
    )
    lengths_q = torch.ones(case.batch_size * case.kv_heads, device=q.device, dtype=torch.int32)
    out_flat = flash_attn_varlen_func(
        q=q_flat,
        k=flat_key_cache,
        v=flat_value_cache,
        cu_seqlens_q=make_cu_seqlens(lengths_q),
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        max_seqlen_q=1,
        max_seqlen_k=max_seqlen_k,
        causal=False,
        dropout_p=0.0,
        block_table=block_table,
        return_softmax_lse=False,
        fa_version=fa_version,
        num_splits=0,
    )
    return out_flat.view(case.batch_size, case.kv_heads, q_heads_per_kv, case.head_dim).reshape(
        case.batch_size,
        case.heads,
        case.head_dim,
    )


def masked_compact_sdpa_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    selected_remote: torch.Tensor,
    *,
    case: Case,
    current_blocks: torch.Tensor,
) -> torch.Tensor:
    q_heads_per_kv = case.heads // case.kv_heads
    remote_count = selected_remote.shape[-1]
    total_blocks = remote_count + 1
    total_tokens = total_blocks * case.block_size
    compact_k = torch.empty(
        case.batch_size,
        case.kv_heads,
        total_tokens,
        case.head_dim,
        device=q.device,
        dtype=k.dtype,
    )
    compact_v = torch.empty_like(compact_k)

    for batch in range(case.batch_size):
        for kv_head in range(case.kv_heads):
            write = 0
            for block in selected_remote[batch, kv_head].tolist():
                start = int(block) * case.block_size
                end = start + case.block_size
                compact_k[batch, kv_head, write:write + case.block_size] = k[batch, start:end, kv_head]
                compact_v[batch, kv_head, write:write + case.block_size] = v[batch, start:end, kv_head]
                write += case.block_size

            current_block = int(current_blocks[batch].item())
            current_start = current_block * case.block_size
            current_end = min(current_start + case.block_size, case.past_lens[batch])
            current_len = current_end - current_start
            compact_k[batch, kv_head, write:write + current_len] = k[batch, current_start:current_end, kv_head]
            compact_v[batch, kv_head, write:write + current_len] = v[batch, current_start:current_end, kv_head]
            if current_len < case.block_size:
                compact_k[batch, kv_head, write + current_len:write + case.block_size] = 0
                compact_v[batch, kv_head, write + current_len:write + case.block_size] = 0

    q_compact = q.view(case.batch_size, case.kv_heads, q_heads_per_kv, case.head_dim).reshape(
        case.batch_size * case.kv_heads,
        q_heads_per_kv,
        1,
        case.head_dim,
    )
    k_heads = compact_k.repeat_interleave(q_heads_per_kv, dim=1).reshape(
        case.batch_size * case.kv_heads,
        q_heads_per_kv,
        total_tokens,
        case.head_dim,
    )
    v_heads = compact_v.repeat_interleave(q_heads_per_kv, dim=1).reshape(
        case.batch_size * case.kv_heads,
        q_heads_per_kv,
        total_tokens,
        case.head_dim,
    )

    local_prefix = torch.tensor(
        [past_len % case.block_size for past_len in case.past_lens],
        device=q.device,
        dtype=torch.long,
    )
    valid_tokens = (
        remote_count * case.block_size
        + local_prefix.view(case.batch_size, 1).expand(case.batch_size, case.kv_heads).reshape(-1)
    )
    token_ids = torch.arange(total_tokens, device=q.device).view(1, 1, 1, total_tokens)
    mask = torch.zeros(
        case.batch_size * case.kv_heads,
        q_heads_per_kv,
        1,
        total_tokens,
        device=q.device,
        dtype=torch.float32,
    )
    mask.masked_fill_(token_ids >= valid_tokens.view(-1, 1, 1, 1), float("-inf"))
    out = F.scaled_dot_product_attention(
        q_compact,
        k_heads,
        v_heads,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
    ).squeeze(2)
    return out.view(case.batch_size, case.kv_heads, q_heads_per_kv, case.head_dim).reshape(
        case.batch_size,
        case.heads,
        case.head_dim,
    )


def run_case(case: Case, dtype: torch.dtype, seed: int, fa_version: int, atol: float, rtol: float, skip_buggy: bool) -> None:
    if case.heads % case.kv_heads != 0:
        raise ValueError("heads must be divisible by kv_heads.")
    current_blocks_host = [past_len // case.block_size for past_len in case.past_lens]
    local_prefixes_host = [past_len % case.block_size for past_len in case.past_lens]
    if case.active_blocks < 2:
        raise ValueError("active_blocks must include at least one remote block plus the current block.")
    if min(current_blocks_host) < case.active_blocks - 1:
        raise ValueError("past_len is too short for active_blocks - 1 full remote blocks.")

    device = torch.device("cuda")
    q, k, v = make_qkv(case, dtype, device, seed)
    current_blocks = torch.tensor(current_blocks_host, device=device, dtype=torch.long)
    local_prefixes = torch.tensor(local_prefixes_host, device=device, dtype=torch.int32)
    num_pages = max(current_blocks_host) + 1
    flat_key_cache, flat_value_cache = make_padded_paged_kv(k, v, case.block_size, min_pages=num_pages)
    remote_count = case.active_blocks - 1

    remote_scores = make_remote_scores(case, current_blocks, seed=seed + 17, device=device)
    selected_remote_kv_batch = select_remote_blocks(remote_scores, remote_count)
    selected_remote = selected_remote_kv_batch.permute(1, 0, 2).contiguous()
    selected_blocks = torch.empty(
        case.batch_size,
        case.kv_heads,
        case.active_blocks,
        device=device,
        dtype=selected_remote_kv_batch.dtype,
    )
    selected_blocks[..., :remote_count] = selected_remote
    selected_blocks[..., remote_count] = current_blocks.view(case.batch_size, 1)
    full_block_table = make_full_decode_block_table(
        batch_size=case.batch_size,
        kv_heads=case.kv_heads,
        num_pages=num_pages,
        device=device,
    )

    block_table = run_synced_nvtx(
        "decode_block_table_build",
        lambda: build_decode_sparse_block_table(
            full_block_table,
            selected_blocks,
        ),
    )
    seqused_k = make_decode_sparse_seqused_k(
        local_prefixes,
        remote_blocks=remote_count,
        config=FlashAttentionConfig(
            block_size=case.block_size,
            heads=case.heads,
            kv_heads=case.kv_heads,
            head_dim=case.head_dim,
            fa_version=fa_version,
        ),
    )
    actual = sparse_decode_with_table(
        q,
        flat_key_cache,
        flat_value_cache,
        block_table,
        seqused_k,
        max_seqlen_k=case.active_blocks * case.block_size,
        case=case,
        fa_version=fa_version,
    )
    expected = masked_compact_sdpa_reference(q, k, v, selected_remote, case=case, current_blocks=current_blocks)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

    if not skip_buggy:
        buggy_block_table = make_buggy_decode_block_table_with_current_full(
            full_block_table,
            selected_blocks,
        )
        buggy_seqused_k = torch.full(
            (case.batch_size * case.kv_heads,),
            case.active_blocks * case.block_size,
            device=device,
            dtype=torch.int32,
        )
        buggy = sparse_decode_with_table(
            q,
            flat_key_cache,
            flat_value_cache,
            buggy_block_table,
            buggy_seqused_k,
            max_seqlen_k=case.active_blocks * case.block_size,
            case=case,
            fa_version=fa_version,
        )
        if torch.allclose(buggy, expected, atol=atol, rtol=rtol):
            raise AssertionError("buggy full-current-block decode unexpectedly matched the causal SDPA reference")

    print(
        "PASS "
        f"past_lens={list(case.past_lens)} current_blocks={current_blocks_host} "
        f"local_prefixes={local_prefixes_host} "
        f"remote_count={remote_count}"
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    dtype = dtype_from_name(args.dtype)
    for idx, past_len in enumerate(args.past_lens):
        case = Case(
            batch_size=args.batch_size,
            past_lens=tuple([past_len] * args.batch_size),
            block_size=args.block_size,
            heads=args.heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            active_blocks=args.active_blocks,
        )
        run_case(case, dtype, args.seed + idx * 1000, args.fa_version, args.atol, args.rtol, args.skip_buggy_check)
    if args.include_ragged_batch:
        ragged_lens = tuple(args.ragged_past_lens)
        case = Case(
            batch_size=len(ragged_lens),
            past_lens=ragged_lens,
            block_size=args.block_size,
            heads=args.heads,
            kv_heads=args.kv_heads,
            head_dim=args.head_dim,
            active_blocks=args.active_blocks,
        )
        run_case(case, dtype, args.seed + 9000, args.fa_version, args.atol, args.rtol, args.skip_buggy_check)


if __name__ == "__main__":
    main()
