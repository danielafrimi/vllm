# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Microbenchmark the efficient Nemotron-H Q-share sampler."""

from __future__ import annotations

import argparse
import itertools

import torch

from vllm.model_executor.models.nemotron_h_dsa_triton_qshare import (
    EfficientMeanQShareProvider,
)

CASES: dict[str, tuple[int, ...]] = {
    "single_decode": (1,),
    "single_8k_prefill": (8192,),
    "decode_and_8k_prefill": (1, 8192),
    "mixed_decode_and_prefills": (1, 1, 4, 1, 17, 64, 257, 1024, 4097),
    "multiple_varied_prefills": (7, 32, 129, 1024, 4097, 8192),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=tuple(CASES),
        default="mixed_decode_and_prefills",
    )
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_lengths = CASES[args.case]
    query_start_loc_values = (0, *itertools.accumulate(query_lengths))
    total_query_rows = query_start_loc_values[-1]
    total_sampled_rows = sum(
        (length + args.group_size - 1) // args.group_size for length in query_lengths
    )
    query_start_loc_cpu = torch.tensor(
        query_start_loc_values,
        dtype=torch.int64,
    )
    query_start_loc = query_start_loc_cpu.to(
        device="cuda",
        non_blocking=True,
    )
    projected_q = torch.randn(
        total_query_rows,
        args.heads,
        args.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    provider = EfficientMeanQShareProvider(group_size=args.group_size).cuda()

    state = None
    for _ in range(args.warmup):
        state = provider(
            projected_q=projected_q,
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            total_sampled_rows=total_sampled_rows,
        )
    torch.accelerator.synchronize()

    with torch.cuda.nvtx.range(f"qshare_sampler:{args.case}"):
        for _ in range(args.iterations):
            state = provider(
                projected_q=projected_q,
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                total_sampled_rows=total_sampled_rows,
            )
    torch.accelerator.synchronize()

    assert state is not None
    print(
        "completed",
        args.case,
        args.iterations,
        tuple(state.sampled_q.shape),
    )


if __name__ == "__main__":
    main()
