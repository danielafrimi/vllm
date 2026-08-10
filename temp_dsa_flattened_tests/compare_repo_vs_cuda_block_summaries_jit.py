#!/usr/bin/env python
"""Temporary comparison: repo per-sequence DSA summaries vs batched CUDA JIT.

This compares the current Nemotron-H helper entry point
`_get_indexer_chunk_representatives` against the temporary batched CUDA kernel
from `test_cuda_block_summaries_jit.py`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]

from test_cuda_block_summaries_jit import dtype_from_name, load_extension


def load_repo_dsa_methods_class():
    """Load the relevant method bodies directly from nemotron_h.py.

    The temporary Slurm env has torch but not this checkout's compiled vLLM C++
    extension, so importing `vllm.model_executor.models.nemotron_h` would fail.
    This still executes the exact source text of the current helper methods.
    """

    source_path = REPO_ROOT / "vllm" / "model_executor" / "models" / "nemotron_h.py"
    lines = source_path.read_text().splitlines()
    # 1-based inclusive ranges from the current file:
    # _gather_kv_sequence / _dsa_kv_cache_layout_and_block_size / reset cache
    # summary-cache helpers and _get_indexer_chunk_representatives
    # _build_indexer_chunk_representatives
    ranges = [
        (1230, 1289),
        (1431, 1757),
        (1802, 1834),
    ]
    class_src = ["class RepoDSAMethods:"]
    for start, end in ranges:
        class_src.extend(lines[start - 1 : end])
    namespace: dict[str, object] = {"torch": torch, "math": math}
    exec("\n".join(class_src), namespace)
    return namespace["RepoDSAMethods"]


RepoDSAMethods = load_repo_dsa_methods_class()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--include-alias-diagnostic", action="store_true")
    return parser.parse_args()


def make_fake_dsa(
    *,
    block_size: int,
    kv_heads: int,
    head_dim: int,
    q_indexer_dim: int,
    use_summary_cache: bool,
    summary_cache_max_blocks: int = 65536,
) -> object:
    fake = object.__new__(RepoDSAMethods)
    fake.num_kv_heads = kv_heads
    fake.head_dim = head_dim
    fake.q_indexer_dim = q_indexer_dim
    fake.q_indexer_chunk_size = block_size
    fake.q_indexer_use_summary_cache = use_summary_cache
    fake.q_indexer_summary_cache_max_blocks = summary_cache_max_blocks
    fake._dsa_summary_cache_block_ids = None
    fake._dsa_summary_cache_values = None
    fake._dsa_summary_cache_valid = None
    fake._dsa_summary_cache_block_size = None
    return fake


def sequence_lengths(batch: int, block_size: int, max_chunks: int) -> list[int]:
    max_len = max_chunks * block_size
    candidates = [
        1,
        max(1, block_size - 1),
        block_size,
        min(max_len, block_size + 1),
        min(max_len, 2 * block_size - 1),
        min(max_len, 2 * block_size),
        min(max_len, 2 * block_size + 1),
        max_len,
    ]
    lengths: list[int] = []
    for value in candidates:
        if value not in lengths:
            lengths.append(value)
    i = 0
    while len(lengths) < batch:
        value = 1 + ((i * 13 + 7) % max_len)
        if value not in lengths:
            lengths.append(value)
        i += 1
    return lengths[:batch]


def make_unique_block_table(
    *,
    seq_lens: torch.Tensor,
    block_size: int,
    max_chunks: int,
    num_physical_blocks: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    batch = int(seq_lens.numel())
    total_used = sum(math.ceil(int(length.item()) / block_size) for length in seq_lens)
    if total_used > num_physical_blocks:
        raise ValueError(
            f"need at least {total_used} physical blocks, got {num_physical_blocks}"
        )
    physical_ids = torch.randperm(
        num_physical_blocks,
        device=device,
        generator=generator,
        dtype=torch.long,
    )
    table = torch.zeros(batch, max_chunks, device=device, dtype=torch.long)
    cursor = 0
    for seq_idx, length in enumerate(seq_lens.detach().cpu().tolist()):
        num_chunks = math.ceil(int(length) / block_size)
        table[seq_idx, :num_chunks] = physical_ids[cursor : cursor + num_chunks]
        cursor += num_chunks
    return table


def run_repo_per_sequence(
    *,
    fake,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    batch, max_chunks = block_table.shape
    out = torch.zeros(
        batch,
        max_chunks,
        fake.num_kv_heads,
        fake.q_indexer_dim,
        device=key_cache.device,
        dtype=torch.float32,
    )
    for seq_idx, key_len in enumerate(seq_lens.detach().cpu().tolist()):
        reps = fake._get_indexer_chunk_representatives(
            key_states=None,
            key_cache=key_cache,
            block_table=block_table[seq_idx],
            key_len=int(key_len),
        )
        out[seq_idx, : reps.shape[0]] = reps
    return out


def cases(args: argparse.Namespace) -> list[dict[str, int | str]]:
    if not args.sweep:
        return [{
            "name": "single",
            "batch": 8,
            "max_chunks": 7,
            "num_physical_blocks": 128,
            "block_size": 16,
            "kv_heads": 4,
            "head_dim": 80,
            "q_indexer_dim": 64,
        }]
    return [
        {
            "name": "block1",
            "batch": 4,
            "max_chunks": 5,
            "num_physical_blocks": 32,
            "block_size": 1,
            "kv_heads": 1,
            "head_dim": 3,
            "q_indexer_dim": 1,
        },
        {
            "name": "odd_small",
            "batch": 7,
            "max_chunks": 9,
            "num_physical_blocks": 128,
            "block_size": 3,
            "kv_heads": 2,
            "head_dim": 17,
            "q_indexer_dim": 13,
        },
        {
            "name": "nemotron_like",
            "batch": 9,
            "max_chunks": 12,
            "num_physical_blocks": 256,
            "block_size": 16,
            "kv_heads": 4,
            "head_dim": 80,
            "q_indexer_dim": 64,
        },
        {
            "name": "qdim_equals_head",
            "batch": 11,
            "max_chunks": 18,
            "num_physical_blocks": 512,
            "block_size": 16,
            "kv_heads": 8,
            "head_dim": 64,
            "q_indexer_dim": 64,
        },
        {
            "name": "large_batch",
            "batch": 33,
            "max_chunks": 34,
            "num_physical_blocks": 2048,
            "block_size": 16,
            "kv_heads": 8,
            "head_dim": 128,
            "q_indexer_dim": 96,
        },
        {
            "name": "block32",
            "batch": 17,
            "max_chunks": 18,
            "num_physical_blocks": 1024,
            "block_size": 32,
            "kv_heads": 4,
            "head_dim": 128,
            "q_indexer_dim": 80,
        },
    ]


def run_case(
    *,
    ext,
    case: dict[str, int | str],
    dtype: torch.dtype,
    device: torch.device,
    seed: int,
) -> None:
    batch = int(case["batch"])
    max_chunks = int(case["max_chunks"])
    block_size = int(case["block_size"])
    kv_heads = int(case["kv_heads"])
    head_dim = int(case["head_dim"])
    q_indexer_dim = int(case["q_indexer_dim"])
    num_physical_blocks = int(case["num_physical_blocks"])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    seq_lens = torch.tensor(
        sequence_lengths(batch, block_size, max_chunks),
        device=device,
        dtype=torch.long,
    )
    key_cache = torch.randn(
        num_physical_blocks,
        block_size,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    block_table = make_unique_block_table(
        seq_lens=seq_lens,
        block_size=block_size,
        max_chunks=max_chunks,
        num_physical_blocks=num_physical_blocks,
        device=device,
        generator=generator,
    )

    cuda_out = ext.dsa_block_summaries(
        key_cache,
        block_table,
        seq_lens,
        q_indexer_dim,
    )
    for use_summary_cache, label in [
        (True, "repo_summary_cache"),
        (False, "repo_dense_fallback"),
    ]:
        fake = make_fake_dsa(
            block_size=block_size,
            kv_heads=kv_heads,
            head_dim=head_dim,
            q_indexer_dim=q_indexer_dim,
            use_summary_cache=use_summary_cache,
        )
        repo_out = run_repo_per_sequence(
            fake=fake,
            key_cache=key_cache,
            block_table=block_table,
            seq_lens=seq_lens,
        )
        max_abs = (repo_out - cuda_out).abs().max().item()
        torch.testing.assert_close(repo_out, cuda_out, atol=1e-5, rtol=1e-5)
        print(
            f"PASS case={case['name']} mode={label} dtype={dtype} "
            f"seq_lens={seq_lens.detach().cpu().tolist()} "
            f"output_shape={tuple(cuda_out.shape)} max_abs_diff={max_abs:.6g}",
            flush=True,
        )


def run_alias_diagnostic(*, dtype: torch.dtype, device: torch.device, seed: int) -> None:
    block_size = 16
    kv_heads = 2
    head_dim = 32
    q_indexer_dim = 16
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    key_cache = torch.randn(
        8,
        block_size,
        kv_heads,
        head_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    seq_lens = torch.tensor([block_size, block_size + 1], device=device)
    block_table = torch.tensor(
        [
            [3, 0],
            [4, 3],
        ],
        device=device,
        dtype=torch.long,
    )
    fake = make_fake_dsa(
        block_size=block_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        q_indexer_dim=q_indexer_dim,
        use_summary_cache=True,
    )
    repo_out = run_repo_per_sequence(
        fake=fake,
        key_cache=key_cache,
        block_table=block_table,
        seq_lens=seq_lens,
    )
    fresh_fake = make_fake_dsa(
        block_size=block_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        q_indexer_dim=q_indexer_dim,
        use_summary_cache=False,
    )
    expected = run_repo_per_sequence(
        fake=fresh_fake,
        key_cache=key_cache,
        block_table=block_table,
        seq_lens=seq_lens,
    )
    max_abs = (repo_out - expected).abs().max().item()
    if max_abs > 1e-5:
        print(
            "SUSPICIOUS alias_diagnostic summary-cache output depends on "
            "physical-block reuse with different logical valid lengths; "
            f"max_abs_diff={max_abs:.6g}",
            flush=True,
        )
    else:
        print("PASS alias_diagnostic no mismatch observed", flush=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device == "cuda":
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    ext = load_extension()

    for idx, case in enumerate(cases(args)):
        run_case(
            ext=ext,
            case=case,
            dtype=dtype,
            device=device,
            seed=args.seed + 1009 * idx,
        )
    if args.include_alias_diagnostic:
        run_alias_diagnostic(dtype=dtype, device=device, seed=args.seed + 9001)
    print("PASS repo-vs-CUDA block-summary comparison")


if __name__ == "__main__":
    main()
