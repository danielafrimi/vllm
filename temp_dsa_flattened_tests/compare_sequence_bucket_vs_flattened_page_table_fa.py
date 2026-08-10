#!/usr/bin/env python
"""Temporary multi-sequence Nemotron-H DSA page-table FA harness."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

import torch

from vllm.vllm_flash_attn import flash_attn_varlen_func


def load_source_nemotron_h_class():
    repo = Path(os.environ.get("VLLM_SOURCE_REPO", Path.cwd()))
    source_path = repo / "vllm/model_executor/models/nemotron_h.py"
    spec = importlib.util.spec_from_file_location(
        "_temp_source_nemotron_h_bucket",
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
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--stress-seeds", type=int, default=3)
    parser.add_argument("--fa-version", type=int, default=2)
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--bench-iters", type=int, default=20)
    parser.add_argument("--atol", type=float, default=2.5e-1)
    parser.add_argument("--rtol", type=float, default=2.5e-1)
    parser.add_argument("--skip-bench", action="store_true")
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def make_fake_dsa(args: argparse.Namespace,
                  *,
                  top_k: int | None = None) -> NemotronHDSASelectiveAttention:
    fake = object.__new__(NemotronHDSASelectiveAttention)
    fake.q_indexer_chunk_size = args.block_size
    fake.q_indexer_chunk_top_k = args.top_k if top_k is None else top_k
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


def make_bucket_case(
    args: argparse.Namespace,
    *,
    name: str,
    seq_specs: list[tuple[int, int]],
    top_k: int,
    seed: int,
) -> dict[str, Any]:
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    for q_len, key_len in seq_specs:
        if q_len <= 0 or key_len <= 0 or q_len > key_len:
            raise ValueError(
                "invalid sequence spec, expected 0 < q_len <= key_len, "
                f"got q_len={q_len} key_len={key_len}"
            )
    query_lens = torch.tensor([q_len for q_len, _ in seq_specs],
                              device=device,
                              dtype=torch.int32)
    key_lens = torch.tensor([key_len for _, key_len in seq_specs],
                            device=device,
                            dtype=torch.int32)
    query_start_loc = torch.empty(len(seq_specs) + 1,
                                  device=device,
                                  dtype=torch.int32)
    query_start_loc[0] = 0
    query_start_loc[1:] = torch.cumsum(query_lens, dim=0, dtype=torch.int32)
    total_rows = int(query_start_loc[-1].item())
    max_blocks = int(
        max(math.ceil(key_len / args.block_size) for _, key_len in seq_specs))
    num_physical_blocks = len(seq_specs) * max_blocks

    query_states = torch.randn(
        total_rows,
        args.heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    key_cache = torch.randn(
        num_physical_blocks,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    value_cache = torch.randn(
        num_physical_blocks,
        args.block_size,
        args.kv_heads,
        args.head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    block_tables = torch.empty(
        len(seq_specs),
        max_blocks,
        device=device,
        dtype=torch.int32,
    )
    query_positions = torch.empty(total_rows, device=device, dtype=torch.long)
    current_chunks = torch.empty(total_rows, device=device, dtype=torch.long)
    top_chunk_indices = torch.zeros(
        total_rows,
        args.kv_heads,
        top_k,
        device=device,
        dtype=torch.long,
    )
    top_chunk_valid = torch.zeros(
        total_rows,
        args.kv_heads,
        top_k,
        device=device,
        dtype=torch.bool,
    )

    for seq_idx, (q_len, key_len) in enumerate(seq_specs):
        seq_blocks = math.ceil(key_len / args.block_size)
        physical_start = seq_idx * max_blocks
        permuted = physical_start + torch.randperm(
            max_blocks,
            device=device,
            generator=generator,
            dtype=torch.long,
        )
        block_tables[seq_idx] = permuted.to(torch.int32)
        if key_len % args.block_size:
            pad_start = key_len % args.block_size
            last_block = int(block_tables[seq_idx, seq_blocks - 1].item())
            key_cache[last_block, pad_start:] = 0
            value_cache[last_block, pad_start:] = 0

        row_start = int(query_start_loc[seq_idx].item())
        row_end = int(query_start_loc[seq_idx + 1].item())
        positions = torch.arange(
            key_len - q_len,
            key_len,
            device=device,
            dtype=torch.long,
        )
        chunks = torch.div(
            positions,
            args.block_size,
            rounding_mode="floor",
        )
        query_positions[row_start:row_end] = positions
        current_chunks[row_start:row_end] = chunks
        for local_row, current_chunk in enumerate(chunks.tolist()):
            row = row_start + local_row
            valid_count = min(top_k, int(current_chunk))
            if valid_count == 0:
                continue
            for group_idx in range(args.kv_heads):
                selected = torch.randperm(
                    int(current_chunk),
                    device=device,
                    generator=generator,
                    dtype=torch.long,
                )[:valid_count]
                top_chunk_indices[row, group_idx, :valid_count] = selected
                top_chunk_valid[row, group_idx, :valid_count] = True

    return {
        "name": name,
        "seq_specs": seq_specs,
        "top_k": top_k,
        "seed": seed,
        "query_states": query_states,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "top_chunk_indices": top_chunk_indices,
        "top_chunk_valid": top_chunk_valid,
        "current_chunks": current_chunks,
        "query_positions": query_positions,
        "query_start_loc": query_start_loc,
        "key_lens": key_lens,
    }


def run_bucket_reference(fake: NemotronHDSASelectiveAttention,
                         case: dict[str, Any]) -> torch.Tensor:
    out = fake._forward_dsa_chunked_page_table_fa_sequence_bucket(
        query_states=case["query_states"],
        key_cache=case["key_cache"],
        value_cache=case["value_cache"],
        block_tables=case["block_tables"],
        attn_metadata=None,
        top_chunk_indices=case["top_chunk_indices"],
        top_chunk_valid=case["top_chunk_valid"],
        current_chunks=case["current_chunks"],
        query_positions=case["query_positions"],
        query_start_loc=case["query_start_loc"],
        key_lens=case["key_lens"],
        softmax_scale=1.0 / math.sqrt(fake.head_dim),
        require_decode_tail=False,
    )
    if out is None:
        raise RuntimeError("bucket reference unexpectedly fell back")
    return out


def run_flattened_bucket_candidate(fake: NemotronHDSASelectiveAttention,
                                   case: dict[str, Any]) -> torch.Tensor:
    query_states = case["query_states"]
    key_cache = case["key_cache"]
    value_cache = case["value_cache"]
    block_tables = case["block_tables"]
    top_chunk_indices = case["top_chunk_indices"]
    top_chunk_valid = case["top_chunk_valid"]
    current_chunks = case["current_chunks"]
    query_positions = case["query_positions"]
    query_start_loc = case["query_start_loc"]

    total_rows = query_states.shape[0]
    group_size = fake.num_heads // fake.num_kv_heads
    block_size = fake.q_indexer_chunk_size
    num_physical_blocks = key_cache.shape[0]
    flat_requests = total_rows * fake.num_kv_heads

    q_flat = query_states.view(
        total_rows,
        fake.num_kv_heads,
        group_size,
        fake.head_dim,
    ).reshape(flat_requests, group_size, fake.head_dim)
    flat_key_cache = key_cache.permute(2, 0, 1, 3).reshape(
        fake.num_kv_heads * num_physical_blocks,
        block_size,
        1,
        fake.head_dim,
    )
    flat_value_cache = value_cache.permute(2, 0, 1, 3).reshape(
        fake.num_kv_heads * num_physical_blocks,
        block_size,
        1,
        fake.head_dim,
    )

    row_counts = (query_start_loc[1:] - query_start_loc[:-1]).to(torch.long)
    row_seq_ids = torch.repeat_interleave(
        torch.arange(row_counts.numel(),
                     device=query_states.device,
                     dtype=torch.long),
        row_counts,
    )
    row_block_tables = block_tables.index_select(0, row_seq_ids)
    valid_top_counts = top_chunk_valid.to(torch.int32).sum(dim=-1)
    max_valid_top_chunks = (
        int(valid_top_counts.max().item()) if valid_top_counts.numel() else 0)
    group_offsets = (
        torch.arange(fake.num_kv_heads,
                     device=query_states.device,
                     dtype=torch.int32).view(1, fake.num_kv_heads, 1)
        * num_physical_blocks)

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
        recalled_blocks = row_block_tables[:, None, :].expand(
            total_rows,
            fake.num_kv_heads,
            row_block_tables.shape[1],
        ).gather(2, compact_top_chunks)
        recalled_blocks = recalled_blocks + group_offsets
    else:
        recalled_blocks = block_tables.new_empty(
            total_rows,
            fake.num_kv_heads,
            0,
        )

    current_blocks = row_block_tables.gather(
        1,
        current_chunks.to(torch.long).view(total_rows, 1),
    ).view(total_rows, 1, 1).expand(total_rows, fake.num_kv_heads, 1)
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
        + tail_lens.view(total_rows, 1)
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
        total_rows,
        fake.num_kv_heads,
        group_size,
        fake.head_dim,
    ).reshape(total_rows, fake.num_heads, fake.head_dim)


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


def case_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_cases: list[tuple[str, list[tuple[int, int]]]] = [
        (
            "ragged_user_shape",
            [
                (1, 500),
                (7, 2012),
                (64, 4099),
                (3, 71),
                (128, 8191),
                (1, 1024),
                (19, 16385),
                (256, 32783),
            ],
        ),
        (
            "boundary_and_tails",
            [
                (1, 16),
                (2, 17),
                (7, 31),
                (8, 32),
                (15, 255),
                (16, 256),
                (17, 257),
                (33, 1025),
            ],
        ),
        (
            "many_mixed_sequences",
            [
                (1, 128 + idx * 97)
                if idx % 4 == 0
                else (2 + idx % 9, 500 + idx * 211)
                if idx % 4 == 1
                else (16 + idx % 17, 2048 + idx * 257)
                if idx % 4 == 2
                else (64 + idx % 33, 4096 + idx * 389)
                for idx in range(24)
            ],
        ),
        (
            "long_ragged_prefill",
            [
                (1, 65536),
                (16, 65543),
                (37, 65551),
                (128, 131071),
                (512, 131072),
            ],
        ),
    ]
    requested_top_ks = sorted({1, 2, min(8, args.top_k), args.top_k})
    cases: list[dict[str, Any]] = []
    for seed_offset in range(args.stress_seeds):
        for top_k in requested_top_ks:
            if top_k <= 0:
                continue
            for case_idx, (name, seq_specs) in enumerate(base_cases):
                cases.append({
                    "name": f"{name}_seed{seed_offset}_top{top_k}",
                    "seq_specs": seq_specs,
                    "top_k": top_k,
                    "seed": args.seed + seed_offset * 1009 + case_idx * 97 + top_k,
                })
    return cases


def describe_case(case: dict[str, Any]) -> str:
    seq_specs = case["seq_specs"]
    q_lens = [q_len for q_len, _ in seq_specs]
    key_lens = [key_len for _, key_len in seq_specs]
    return (
        f"name={case['name']} seed={case['seed']} top_k={case['top_k']} "
        f"seqs={len(seq_specs)} rows={sum(q_lens)} "
        f"q_lens={q_lens} key_lens={key_lens}"
    )


def first_large_diff(reference: torch.Tensor,
                     flattened: torch.Tensor,
                     *,
                     atol: float,
                     rtol: float) -> str:
    diff = (reference.float() - flattened.float()).abs()
    tolerance = atol + rtol * reference.float().abs()
    bad = diff > tolerance
    if not bool(bad.any().item()):
        return "no element exceeds tolerance"
    row, head, dim = bad.nonzero()[0].tolist()
    return (
        f"first_bad=(row={row}, head={head}, dim={dim}) "
        f"ref={float(reference[row, head, dim].float().item()):.6e} "
        f"flat={float(flattened[row, head, dim].float().item()):.6e} "
        f"abs={float(diff[row, head, dim].item()):.6e} "
        f"tol={float(tolerance[row, head, dim].item()):.6e}"
    )


def run_correctness_case(args: argparse.Namespace,
                         spec: dict[str, Any]) -> str:
    fake = make_fake_dsa(args, top_k=int(spec["top_k"]))
    case = make_bucket_case(
        args,
        name=str(spec["name"]),
        seq_specs=spec["seq_specs"],
        top_k=int(spec["top_k"]),
        seed=int(spec["seed"]),
    )
    start = time.perf_counter()
    try:
        reference = run_bucket_reference(fake, case)
    except Exception as exc:
        print(f"REFERENCE_FAILURE {describe_case(case)} error={exc!r}")
        return "reference_failure"

    try:
        flattened = run_flattened_bucket_candidate(fake, case)
    except Exception as exc:
        print(f"CANDIDATE_FAILURE {describe_case(case)} error={exc!r}")
        return "candidate_failure"

    diff = (reference.float() - flattened.float()).abs()
    max_abs = float(diff.max().item())
    max_ref = float(reference.float().abs().max().item())
    limit = args.atol + args.rtol * max_ref
    elapsed = time.perf_counter() - start
    try:
        torch.testing.assert_close(
            flattened,
            reference,
            atol=args.atol,
            rtol=args.rtol,
        )
    except AssertionError as exc:
        print(
            f"MISMATCH {describe_case(case)} max_abs={max_abs:.6e} "
            f"max_ref={max_ref:.6e} limit={limit:.6e} "
            f"{first_large_diff(reference, flattened, atol=args.atol, rtol=args.rtol)} "
            f"assertion={exc}"
        )
        return "mismatch"

    valid_counts = case["top_chunk_valid"].to(torch.int32).sum(dim=-1)
    print(
        f"MATCH {describe_case(case)} max_abs={max_abs:.6e} "
        f"max_ref={max_ref:.6e} limit={limit:.6e} "
        f"valid_top_min={int(valid_counts.min().item())} "
        f"valid_top_max={int(valid_counts.max().item())} "
        f"elapsed_s={elapsed:.3f}"
    )
    return "match"


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this temporary FA harness")
    if args.heads % args.kv_heads != 0:
        raise ValueError("heads must be divisible by kv_heads")
    if args.top_k < 1:
        raise ValueError("top_k must be positive")

    outcomes = {
        "match": 0,
        "reference_failure": 0,
        "candidate_failure": 0,
        "mismatch": 0,
    }
    for spec in case_matrix(args):
        outcome = run_correctness_case(args, spec)
        outcomes[outcome] += 1

    print(
        "SUMMARY "
        + " ".join(f"{name}={count}" for name, count in outcomes.items())
    )
    if outcomes["candidate_failure"] or outcomes["mismatch"]:
        raise SystemExit(1)

    if not args.skip_bench:
        fake = make_fake_dsa(args, top_k=args.top_k)
        case = make_bucket_case(
            args,
            name="benchmark_ragged_user_shape",
            seq_specs=[
                (1, 500),
                (7, 2012),
                (64, 4099),
                (3, 71),
                (128, 8191),
                (1, 1024),
                (19, 16385),
                (256, 32783),
            ],
            top_k=args.top_k,
            seed=args.seed + 4242,
        )
        reference_ms = time_cuda(
            "sequence_bucket reference_rows",
            lambda: run_bucket_reference(fake, case),
            args,
        )
        flattened_ms = time_cuda(
            "sequence_bucket flattened_candidate",
            lambda: run_flattened_bucket_candidate(fake, case),
            args,
        )
        print(f"sequence_bucket speedup: {reference_ms / flattened_ms:.3f}x")


if __name__ == "__main__":
    main()
