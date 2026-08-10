#!/usr/bin/env python
"""Standalone Triton correctness test for batched DSA KV-page summaries.

This is intentionally separate from the CUDA JIT prototype. It compares a
Python CPU reference loop against the production Triton helper over a ragged
batch block table.
"""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl

from vllm.model_executor.models.nemotron_h_dsa_triton_summaries import (
    dsa_block_summaries_triton,
)


@triton.jit
def _dsa_block_summaries_kernel(
    key_cache,
    block_table,
    seq_lens,
    output,
    max_chunks: tl.constexpr,
    block_size: tl.constexpr,
    kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    q_indexer_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    seq_chunk = tl.program_id(0)
    kv_head = tl.program_id(1)
    seq = seq_chunk // max_chunks
    chunk = seq_chunk - seq * max_chunks
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < q_indexer_dim

    seq_len = tl.load(seq_lens + seq)
    num_chunks = tl.cdiv(seq_len, block_size)
    active = chunk < num_chunks
    remaining = seq_len - chunk * block_size
    valid_len = tl.minimum(remaining, block_size)
    valid_len = tl.maximum(valid_len, 0)
    physical_block = tl.load(block_table + seq * max_chunks + chunk, mask=active, other=0)

    acc = tl.zeros((BLOCK_D,), tl.float32)
    for offset in tl.static_range(0, block_size):
        token_valid = active & (offset < valid_len)
        key_offsets = (
            (((physical_block * block_size + offset) * kv_heads + kv_head) * head_dim)
            + dims
        )
        values = tl.load(
            key_cache + key_offsets,
            mask=token_valid & dim_mask,
            other=0.0,
        )
        acc += values.to(tl.float32)

    denom = tl.maximum(valid_len, 1).to(tl.float32)
    acc = acc / denom
    out_offsets = (
        (((seq * max_chunks + chunk) * kv_heads + kv_head) * q_indexer_dim) + dims
    )
    tl.store(output + out_offsets, acc, mask=dim_mask)


def triton_block_summaries(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indexer_dim: int,
) -> torch.Tensor:
    output = dsa_block_summaries_triton(
        key_cache=key_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        q_indexer_dim=q_indexer_dim,
    )
    if output is None:
        raise RuntimeError("production Triton summary helper returned None")
    return output


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def make_lengths(batch: int, block_size: int, device: torch.device) -> torch.Tensor:
    max_seq_len = batch * block_size + block_size // 2
    lengths = [
        1,
        max(1, block_size - 3),
        block_size,
        block_size + 1,
        block_size + 5,
        2 * block_size - 1,
        2 * block_size,
        2 * block_size + 1,
        max_seq_len,
    ]
    i = 0
    while len(lengths) < batch:
        lengths.append(1 + ((i * 17 + 5) % max_seq_len))
        i += 1
    return torch.tensor(lengths[:batch], device=device, dtype=torch.long)


def make_block_table(
    *,
    seq_lens: torch.Tensor,
    block_size: int,
    max_chunks: int,
    num_physical_blocks: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    total_chunks = int(
        torch.div(seq_lens + block_size - 1, block_size, rounding_mode="floor")
        .sum()
        .item()
    )
    if total_chunks > num_physical_blocks:
        raise ValueError(
            f"need {total_chunks} physical blocks, got {num_physical_blocks}"
        )
    ids = torch.randperm(
        num_physical_blocks,
        device=device,
        generator=generator,
        dtype=torch.long,
    )
    table = torch.zeros(
        seq_lens.numel(),
        max_chunks,
        device=device,
        dtype=torch.long,
    )
    cursor = 0
    for seq_idx, key_len in enumerate(seq_lens.detach().cpu().tolist()):
        chunks = math.ceil(int(key_len) / block_size)
        table[seq_idx, :chunks] = ids[cursor : cursor + chunks]
        cursor += chunks
    return table


def cpu_reference(
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    q_indexer_dim: int,
) -> torch.Tensor:
    key_cpu = key_cache.detach().cpu()
    table_cpu = block_table.detach().cpu()
    lens_cpu = seq_lens.detach().cpu()
    batch, max_chunks = table_cpu.shape
    _, block_size, kv_heads, _ = key_cpu.shape
    out = torch.zeros(batch, max_chunks, kv_heads, q_indexer_dim, dtype=torch.float32)
    for seq in range(batch):
        key_len = int(lens_cpu[seq].item())
        num_chunks = math.ceil(key_len / block_size)
        for chunk in range(num_chunks):
            physical_block = int(table_cpu[seq, chunk].item())
            valid_len = min(block_size, key_len - chunk * block_size)
            for kv_head in range(kv_heads):
                acc = torch.zeros(q_indexer_dim, dtype=torch.float32)
                for offset in range(valid_len):
                    acc += key_cpu[
                        physical_block,
                        offset,
                        kv_head,
                        :q_indexer_dim,
                    ].float()
                out[seq, chunk, kv_head] = acc / valid_len
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--num-physical-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--head-dim", type=int, default=80)
    parser.add_argument("--q-indexer-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sweep", action="store_true")
    return parser.parse_args()


def cases(args: argparse.Namespace) -> list[dict[str, int | str]]:
    if not args.sweep:
        return [{
            "name": "single",
            "batch": args.batch,
            "num_physical_blocks": args.num_physical_blocks,
            "block_size": args.block_size,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "q_indexer_dim": args.q_indexer_dim,
            "dtype": args.dtype,
        }]
    return [
        {
            "name": "block1",
            "batch": 4,
            "num_physical_blocks": 32,
            "block_size": 1,
            "kv_heads": 1,
            "head_dim": 3,
            "q_indexer_dim": 1,
        },
        {
            "name": "odd_small",
            "batch": 7,
            "num_physical_blocks": 128,
            "block_size": 3,
            "kv_heads": 2,
            "head_dim": 17,
            "q_indexer_dim": 13,
        },
        {
            "name": "nemotron_like",
            "batch": 9,
            "num_physical_blocks": 256,
            "block_size": 16,
            "kv_heads": 4,
            "head_dim": 80,
            "q_indexer_dim": 64,
        },
        {
            "name": "nemotron_padded_page_stride",
            "batch": 9,
            "num_physical_blocks": 256,
            "block_size": 16,
            "padded_block_size": 32,
            "kv_heads": 4,
            "head_dim": 128,
            "q_indexer_dim": 64,
        },
        {
            "name": "large_batch",
            "batch": 33,
            "num_physical_blocks": 2048,
            "block_size": 16,
            "kv_heads": 8,
            "head_dim": 128,
            "q_indexer_dim": 96,
        },
        {
            "name": "block32",
            "batch": 17,
            "num_physical_blocks": 1024,
            "block_size": 32,
            "kv_heads": 4,
            "head_dim": 128,
            "q_indexer_dim": 80,
        },
        {
            "name": "fp16",
            "batch": 19,
            "num_physical_blocks": 1024,
            "block_size": 16,
            "kv_heads": 6,
            "head_dim": 96,
            "q_indexer_dim": 95,
            "dtype": "fp16",
        },
        {
            "name": "fp32",
            "batch": 9,
            "num_physical_blocks": 256,
            "block_size": 8,
            "kv_heads": 3,
            "head_dim": 33,
            "q_indexer_dim": 17,
            "dtype": "fp32",
        },
    ]


def run_case(case: dict[str, int | str], device: torch.device, seed: int) -> None:
    dtype = dtype_from_name(str(case.get("dtype", "bf16")))
    batch = int(case["batch"])
    block_size = int(case["block_size"])
    kv_heads = int(case["kv_heads"])
    head_dim = int(case["head_dim"])
    q_indexer_dim = int(case["q_indexer_dim"])
    num_physical_blocks = int(case["num_physical_blocks"])
    padded_block_size = int(case.get("padded_block_size", block_size))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    seq_lens = make_lengths(batch, block_size, device)
    max_chunks = int(
        torch.div(
            seq_lens.max() + block_size - 1,
            block_size,
            rounding_mode="floor",
        ).item()
    )
    key_storage = torch.randn(
        num_physical_blocks,
        padded_block_size,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    key_cache = key_storage[:, :block_size, :, :]
    block_table = make_block_table(
        seq_lens=seq_lens,
        block_size=block_size,
        max_chunks=max_chunks,
        num_physical_blocks=num_physical_blocks,
        device=device,
        generator=generator,
    )
    expected = cpu_reference(key_cache, block_table, seq_lens, q_indexer_dim)
    actual = triton_block_summaries(
        key_cache,
        block_table,
        seq_lens,
        q_indexer_dim,
    ).detach().cpu()
    max_abs = (actual - expected).abs().max().item()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    print(
        f"PASS case={case['name']} dtype={dtype} "
        f"seq_lens={seq_lens.detach().cpu().tolist()} "
        f"key_cache_stride={tuple(key_cache.stride())} "
        f"output_shape={tuple(actual.shape)} max_abs_diff={max_abs:.6g}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    for idx, case in enumerate(cases(args)):
        run_case(case, device, args.seed + idx * 1009)
    print("PASS temporary Triton block-summary prototype")


if __name__ == "__main__":
    main()
