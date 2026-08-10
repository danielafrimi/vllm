#!/usr/bin/env python
"""Temporary Nsight Systems target for DSA block-summary profiling.

This script does not use torch.profiler. It only prepares inputs, warms up,
then runs synchronized NVTX ranges around the core workloads so Nsight Systems
can capture CUDA kernels and NVTX intervals in the system timeline.
"""

from __future__ import annotations

import argparse
import torch

from compare_repo_vs_cuda_block_summaries_jit import make_fake_dsa, run_repo_per_sequence
from profile_repo_vs_cuda_block_summaries_jit import make_case
from test_cuda_block_summaries_jit import load_extension


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
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def synchronized_nvtx_call(label: str, fn):
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(label)
    out = fn()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
    return out


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

    def repo_dense():
        out = None
        for _ in range(args.repeat):
            out = run_repo_per_sequence(
                fake=fake_dense,
                key_cache=key_cache,
                block_table=block_table,
                seq_lens=seq_lens,
            )
        return out

    def repo_summary():
        out = None
        for _ in range(args.repeat):
            fake_summary._reset_dsa_summary_cache()
            out = run_repo_per_sequence(
                fake=fake_summary,
                key_cache=key_cache,
                block_table=block_table,
                seq_lens=seq_lens,
            )
        return out

    def batched_cuda():
        out = None
        for _ in range(args.repeat):
            out = ext.dsa_block_summaries(
                key_cache,
                block_table,
                seq_lens,
                args.q_indexer_dim,
            )
        return out

    for _ in range(args.warmup_iters):
        dense_out = repo_dense()
        summary_out = repo_summary()
        cuda_out = batched_cuda()
    torch.cuda.synchronize()

    dense_out = synchronized_nvtx_call("repo_dense_core", repo_dense)
    summary_out = synchronized_nvtx_call("repo_summary_core", repo_summary)
    cuda_out = synchronized_nvtx_call("batched_cuda_core", batched_cuda)

    # Correctness check after the measured NVTX ranges.
    torch.testing.assert_close(dense_out, cuda_out, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(summary_out, cuda_out, atol=1e-5, rtol=1e-5)
    print(
        "PASS nsys target repo-vs-CUDA block summaries "
        f"seq_lens={seq_lens.detach().cpu().tolist()} "
        f"block_table_shape={tuple(block_table.shape)} "
        f"key_cache_shape={tuple(key_cache.shape)} "
        f"repeat={args.repeat}",
        flush=True,
    )


if __name__ == "__main__":
    main()
