#!/usr/bin/env python
"""Temporary Nemotron-H DSA page-table FA comparison harness.

This file is intentionally outside the production tree. It compares the
current row-wise page-table FlashAttention helper in nemotron_h.py against a
temporary flattened candidate that removes the per-KV-head FlashAttention loop
for one sequence.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func


def load_source_nemotron_h_class():
    repo = Path(os.environ.get("VLLM_SOURCE_REPO", Path.cwd()))
    source_path = repo / "vllm/model_executor/models/nemotron_h.py"
    spec = importlib.util.spec_from_file_location(
        "_temp_source_nemotron_h",
        source_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load spec for {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.NemotronHDSASelectiveAttention


NemotronHDSASelectiveAttention = load_source_nemotron_h_class()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fa-version", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=2.5e-1)
    parser.add_argument("--rtol", type=float, default=2.5e-1)
    parser.add_argument("--skip-bench", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def make_fake_dsa(args: argparse.Namespace) -> NemotronHDSASelectiveAttention:
    fake = object.__new__(NemotronHDSASelectiveAttention)
    fake.q_indexer_chunk_size = args.block_size
    fake.q_indexer_chunk_top_k = args.top_k
    fake.num_kv_heads = args.kv_heads
    fake.num_heads = args.heads
    fake.head_dim = args.head_dim
    fake.layer_idx = -1
    fake.attn = SimpleNamespace(
        sliding_window=None,
        impl=SimpleNamespace(
            vllm_flash_attn_version=args.fa_version,
            sliding_window=None,
            alibi_slopes=None,
            logits_soft_cap=0,
            sinks=None,
        ),
    )
    return fake


def make_case(
    *,
    name: str,
    q_len: int,
    key_len: int,
    args: argparse.Namespace,
) -> dict[str, torch.Tensor | int | str]:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    generator = torch.Generator(device=device)
    stable_name_seed = sum((idx + 1) * ord(char) for idx, char in enumerate(name))
    generator.manual_seed(args.seed + stable_name_seed)

    num_blocks = math.ceil(key_len / args.block_size)
    padded_key_len = num_blocks * args.block_size
    query_start = key_len - q_len
    query_positions = torch.arange(
        query_start,
        key_len,
        device=device,
        dtype=torch.long,
    )
    current_chunks = torch.div(
        query_positions,
        args.block_size,
        rounding_mode="floor",
    ).to(torch.long)

    query_states = torch.randn(
        q_len,
        args.heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    key_cache = torch.randn(
        num_blocks,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    value_cache = torch.randn(
        num_blocks,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    if padded_key_len > key_len:
        first_pad = key_len % args.block_size
        key_cache[-1, first_pad:] = 0
        value_cache[-1, first_pad:] = 0

    block_table = torch.randperm(
        num_blocks,
        device=device,
        generator=generator,
        dtype=torch.long,
    ).to(torch.int32)
    top_chunk_indices = torch.zeros(
        q_len,
        args.kv_heads,
        args.top_k,
        device=device,
        dtype=torch.long,
    )
    top_chunk_valid = torch.zeros(
        q_len,
        args.kv_heads,
        args.top_k,
        device=device,
        dtype=torch.bool,
    )
    for row in range(q_len):
        max_prior = int(current_chunks[row].item())
        valid_count = min(args.top_k, max_prior)
        if valid_count == 0:
            continue
        for group_idx in range(args.kv_heads):
            selected = torch.randperm(
                max_prior,
                device=device,
                generator=generator,
                dtype=torch.long,
            )[:valid_count]
            top_chunk_indices[row, group_idx, :valid_count] = selected
            top_chunk_valid[row, group_idx, :valid_count] = True

    return {
        "name": name,
        "key_len": key_len,
        "query_states": query_states,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_table": block_table,
        "top_chunk_indices": top_chunk_indices,
        "top_chunk_valid": top_chunk_valid,
        "current_chunks": current_chunks,
        "query_positions": query_positions,
    }


def run_current_reference(
    fake: NemotronHDSASelectiveAttention,
    case: dict[str, torch.Tensor | int | str],
) -> torch.Tensor:
    query_states = case["query_states"]
    assert isinstance(query_states, torch.Tensor)
    q_len = query_states.shape[0]
    group_size = fake.num_heads // fake.num_kv_heads
    out = torch.empty_like(query_states)
    for group_idx in range(fake.num_kv_heads):
        head_start = group_idx * group_size
        head_end = head_start + group_size
        group_out = fake._forward_dsa_chunked_page_table_fa_rows(
            query_states=query_states[:, head_start:head_end],
            key_cache=case["key_cache"],
            value_cache=case["value_cache"],
            block_table=case["block_table"],
            attn_metadata=None,
            top_chunk_indices=case["top_chunk_indices"][:, group_idx],
            top_chunk_valid=case["top_chunk_valid"][:, group_idx],
            current_chunks=case["current_chunks"],
            query_positions=case["query_positions"],
            key_len=int(case["key_len"]),
            group_idx=group_idx,
            softmax_scale=1.0 / math.sqrt(fake.head_dim),
            require_decode_tail=q_len == 1,
            path_name="temporary_reference_rows",
            fallback_label="temporary reference rows",
        )
        if group_out is None:
            raise RuntimeError("current reference unexpectedly fell back")
        out[:, head_start:head_end] = group_out
    return out


def run_flattened_candidate(
    fake: NemotronHDSASelectiveAttention,
    case: dict[str, torch.Tensor | int | str],
) -> torch.Tensor:
    query_states = case["query_states"]
    key_cache = case["key_cache"]
    value_cache = case["value_cache"]
    block_table = case["block_table"]
    top_chunk_indices = case["top_chunk_indices"]
    top_chunk_valid = case["top_chunk_valid"]
    current_chunks = case["current_chunks"]
    query_positions = case["query_positions"]
    assert isinstance(query_states, torch.Tensor)
    assert isinstance(key_cache, torch.Tensor)
    assert isinstance(value_cache, torch.Tensor)
    assert isinstance(block_table, torch.Tensor)
    assert isinstance(top_chunk_indices, torch.Tensor)
    assert isinstance(top_chunk_valid, torch.Tensor)
    assert isinstance(current_chunks, torch.Tensor)
    assert isinstance(query_positions, torch.Tensor)

    q_len = query_states.shape[0]
    group_size = fake.num_heads // fake.num_kv_heads
    block_size = fake.q_indexer_chunk_size
    num_blocks = key_cache.shape[0]
    flat_requests = q_len * fake.num_kv_heads

    q_flat = query_states.view(
        q_len,
        fake.num_kv_heads,
        group_size,
        fake.head_dim,
    ).reshape(flat_requests, group_size, fake.head_dim)
    flat_key_cache = key_cache.permute(2, 0, 1, 3).reshape(
        fake.num_kv_heads * num_blocks,
        block_size,
        1,
        fake.head_dim,
    )
    flat_value_cache = value_cache.permute(2, 0, 1, 3).reshape(
        fake.num_kv_heads * num_blocks,
        block_size,
        1,
        fake.head_dim,
    )

    valid_top_counts = top_chunk_valid.to(torch.int32).sum(dim=-1)
    max_valid_top_chunks = (
        int(valid_top_counts.max().item()) if valid_top_counts.numel() else 0
    )
    if top_chunk_indices.shape[-1] > 0 and max_valid_top_chunks > 0:
        safe_top_chunks = top_chunk_indices.masked_fill(~top_chunk_valid, 0)
        compact_order = torch.argsort(
            (~top_chunk_valid).to(torch.int64),
            dim=-1,
            stable=True,
        )
        compact_top_chunks = safe_top_chunks.gather(
            dim=-1,
            index=compact_order,
        )[..., :max_valid_top_chunks]
        group_offsets = (
            torch.arange(
                fake.num_kv_heads,
                device=query_states.device,
                dtype=torch.int32,
            ).view(1, fake.num_kv_heads, 1)
            * num_blocks
        )
        recalled_blocks = block_table.index_select(
            0,
            compact_top_chunks.reshape(-1).to(torch.long),
        ).view(q_len, fake.num_kv_heads, max_valid_top_chunks)
        recalled_blocks = recalled_blocks + group_offsets
    else:
        recalled_blocks = block_table.new_empty(
            q_len,
            fake.num_kv_heads,
            0,
        )

    group_offsets = (
        torch.arange(
            fake.num_kv_heads,
            device=query_states.device,
            dtype=torch.int32,
        ).view(1, fake.num_kv_heads, 1)
        * num_blocks
    )
    current_blocks = block_table.index_select(
        0,
        current_chunks.to(torch.long),
    ).view(q_len, 1, 1).expand(q_len, fake.num_kv_heads, 1)
    current_blocks = current_blocks + group_offsets
    temp_block_table = torch.cat((recalled_blocks, current_blocks), dim=-1)
    if (
        max_valid_top_chunks > 0
        and not bool(valid_top_counts.eq(max_valid_top_chunks).all().item())
    ):
        temp_block_table.scatter_(
            dim=-1,
            index=valid_top_counts.to(torch.long).unsqueeze(-1),
            src=current_blocks,
        )

    current_chunk_starts = current_chunks.to(torch.long) * block_size
    tail_lens = query_positions.to(torch.long) - current_chunk_starts + 1
    seqused_k = (
        valid_top_counts.to(torch.long)
        * block_size
        + tail_lens.view(q_len, 1)
    ).to(torch.int32)

    impl = getattr(fake.attn, "impl", None)
    fa_version = getattr(impl, "vllm_flash_attn_version", None)
    flash_attn_kwargs = {}
    if fa_version is not None:
        flash_attn_kwargs["fa_version"] = fa_version

    flat_out = flash_attn_varlen_func(
        q=q_flat.contiguous(),
        k=flat_key_cache.contiguous(),
        v=flat_value_cache.contiguous(),
        cu_seqlens_q=torch.arange(
            flat_requests + 1,
            device=query_states.device,
            dtype=torch.int32,
        ),
        max_seqlen_q=1,
        seqused_k=seqused_k.reshape(-1),
        max_seqlen_k=int(seqused_k.max().item()),
        dropout_p=0.0,
        softmax_scale=1.0 / math.sqrt(fake.head_dim),
        causal=False,
        block_table=temp_block_table.reshape(
            flat_requests,
            max_valid_top_chunks + 1,
        ),
        **flash_attn_kwargs,
    )
    return flat_out.view(
        q_len,
        fake.num_kv_heads,
        group_size,
        fake.head_dim,
    ).reshape(q_len, fake.num_heads, fake.head_dim)


def time_cuda(label: str, fn, args: argparse.Namespace) -> float:
    for _ in range(args.warmup_iters):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.bench_iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / args.bench_iters
    print(f"{label}: {ms:.3f} ms")
    return ms


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this temporary FA harness")
    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv_heads")
    if args.top_k < 1:
        raise ValueError("top_k must be positive")

    fake = make_fake_dsa(args)
    cases = [
        make_case(name="decode_partial_tail", q_len=1, key_len=4103, args=args),
        make_case(name="prefill_same_chunk", q_len=8, key_len=1032, args=args),
        make_case(name="prefill_multi_chunk", q_len=37, key_len=4099, args=args),
    ]

    for case in cases:
        name = str(case["name"])
        current = run_current_reference(fake, case)
        flattened = run_flattened_candidate(fake, case)
        diff = (current.float() - flattened.float()).abs()
        max_abs = float(diff.max().item())
        max_ref = float(current.float().abs().max().item())
        limit = args.atol + args.rtol * max_ref
        print(
            f"{name}: max_abs={max_abs:.6e} max_ref={max_ref:.6e} "
            f"limit={limit:.6e}"
        )
        torch.testing.assert_close(
            flattened,
            current,
            atol=args.atol,
            rtol=args.rtol,
        )

        if not args.skip_bench:
            current_ms = time_cuda(
                f"{name} current_rows",
                lambda: run_current_reference(fake, case),
                args,
            )
            flattened_ms = time_cuda(
                f"{name} flattened_candidate",
                lambda: run_flattened_candidate(fake, case),
                args,
            )
            print(f"{name} speedup: {current_ms / flattened_ms:.3f}x")


if __name__ == "__main__":
    main()
