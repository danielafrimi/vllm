# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton score/top-k selector for Nemotron-H chunked DSA."""

from __future__ import annotations

import math
import os
import typing

import torch

try:
    from vllm.triton_utils import tl, triton
except ImportError:
    tl = None
    triton = None


_DSA_CUDAGRAPH_TENSOR_KEEPALIVE: list[torch.Tensor] = []
_DSA_CUDAGRAPH_TENSOR_IDS: set[int] = set()


def dsa_cudagraph_keepalive(*values: typing.Any) -> None:
    """Retain eager DSA workspaces referenced by captured CUDA graphs.

    vLLM captures this custom attention path as opaque Python. Tensor objects
    allocated inside the call are otherwise released after capture even though
    graph kernels retain their addresses. A later graph capture can then reuse
    that storage and corrupt an older graph. Keep only tensors created during
    an actual stream capture, for the lifetime of the worker process.
    """
    if (
        not torch.cuda.is_available()
        or not torch.cuda.is_current_stream_capturing()
    ):
        return

    pending = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, torch.Tensor):
            tensor_id = id(value)
            if value.is_cuda and tensor_id not in _DSA_CUDAGRAPH_TENSOR_IDS:
                _DSA_CUDAGRAPH_TENSOR_IDS.add(tensor_id)
                _DSA_CUDAGRAPH_TENSOR_KEEPALIVE.append(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, (tuple, list)):
            pending.extend(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _triton_scoring_block_chunks() -> int:
    value = _env_int("VLLM_NEMOTRON_H_DSA_TRITON_SCORING_BLOCK_CHUNKS", 128)
    if value <= 0:
        raise ValueError(
            "VLLM_NEMOTRON_H_DSA_TRITON_SCORING_BLOCK_CHUNKS must be "
            f"positive, got {value}"
        )
    return value


def _triton_scoring_decode_block_chunks() -> int:
    value = _env_int(
        "VLLM_NEMOTRON_H_DSA_TRITON_SCORING_DECODE_BLOCK_CHUNKS",
        128,
    )
    if value <= 0:
        raise ValueError(
            "VLLM_NEMOTRON_H_DSA_TRITON_SCORING_DECODE_BLOCK_CHUNKS must be "
            f"positive, got {value}"
        )
    return value


def _triton_scoring_block_rows() -> int:
    value = _env_int("VLLM_NEMOTRON_H_DSA_TRITON_SCORING_BLOCK_ROWS", 64)
    if value <= 0:
        raise ValueError(
            "VLLM_NEMOTRON_H_DSA_TRITON_SCORING_BLOCK_ROWS must be "
            f"positive, got {value}"
        )
    return value


def _triton_scoring_small_block_rows() -> int:
    value = _env_int(
        "VLLM_NEMOTRON_H_DSA_TRITON_SCORING_SMALL_BLOCK_ROWS",
        4,
    )
    if value <= 1:
        raise ValueError(
            "VLLM_NEMOTRON_H_DSA_TRITON_SCORING_SMALL_BLOCK_ROWS must be "
            f"greater than 1, got {value}"
        )
    return value


def _triton_scoring_dot_precision() -> str:
    value = os.environ.get(
        "VLLM_NEMOTRON_H_DSA_TRITON_SCORING_DOT_PRECISION",
        "ieee",
    )
    if value not in {"ieee", "tf32", "tf32x3"}:
        raise ValueError(
            "VLLM_NEMOTRON_H_DSA_TRITON_SCORING_DOT_PRECISION must be "
            f"'ieee', 'tf32', or 'tf32x3', got {value!r}"
        )
    return value


def _has_top_k_per_row_prefill() -> bool:
    try:
        return hasattr(torch.ops, "_C") and hasattr(
            torch.ops._C, "top_k_per_row_prefill"
        )
    except Exception:
        return False


def _run_top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    top_k: int,
) -> None:
    """Call the stock CUDA selector behind a testable launch boundary."""
    torch.ops._C.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )


_DSA_SCORE_TILE_MODE_DECODE = 0
_DSA_SCORE_TILE_MODE_SMALL = 1
_DSA_SCORE_TILE_MODE_LARGE = 2
_DSA_SCORE_TILE_PLAN_COLUMNS = 8
_DSA_ROW_PLAN_COLUMNS = 7
_DSA_ROW_PLAN_WITH_TILE_COLUMNS = 9

# top_k_per_row_prefill uses ``top_k * sizeof(int32_t)`` dynamic shared
# memory in addition to its static 16 KiB selection workspace and scalar
# state. CUDA rounds the static allocation up to 16.25 KiB, leaving 32,512
# bytes under the kernel's default 48 KiB launch limit: exactly 8,128 indices.
# The next dynamic-policy value (8,160) otherwise fails its kernel launch with
# cudaErrorInvalidValue. Keep this Python-side because the copy-overlay image
# intentionally reuses the base image's compiled extension.
_DSA_CUDA_PREFILL_TOP_K_MAX = 8128
_DSA_TORCH_TOP_K_BLOCK_ROWS = 256


def dsa_score_tile_plan_config() -> tuple[int, int, int, int]:
    """Return the row/chunk tile sizes used by mixed score tile plans."""
    return (
        _triton_scoring_small_block_rows(),
        _triton_scoring_block_rows(),
        _triton_scoring_block_chunks(),
        _triton_scoring_decode_block_chunks(),
    )


def dsa_build_score_tile_plan_parts(
    row_plan_parts: typing.Sequence[typing.Sequence[int]],
    *,
    small_block_rows: int,
    large_block_rows: int,
    block_chunks: int,
    decode_block_chunks: int | None = None,
) -> list[tuple[int, int, int, int, int, int, int, int]]:
    """Build CPU score tile records for the mixed Triton scoring kernel.

    Input rows use the existing row-plan schema:
    row_start, q_len, score_seq_id, block_table_seq_id, group_id,
    prior_chunks, and query_position_start.
    """
    if small_block_rows <= 1:
        raise ValueError(f"small_block_rows must be > 1, got {small_block_rows}")
    if large_block_rows < small_block_rows:
        raise ValueError(
            "large_block_rows must be >= small_block_rows, got "
            f"{large_block_rows} < {small_block_rows}"
        )
    if block_chunks <= 0:
        raise ValueError(f"block_chunks must be positive, got {block_chunks}")
    if decode_block_chunks is None:
        decode_block_chunks = block_chunks
    if decode_block_chunks <= 0:
        raise ValueError(
            f"decode_block_chunks must be positive, got {decode_block_chunks}"
        )

    tile_plan_parts: list[tuple[int, int, int, int, int, int, int, int]] = []
    for row_plan in row_plan_parts:
        if len(row_plan) != _DSA_ROW_PLAN_COLUMNS:
            raise ValueError(
                "row plan entries must have "
                f"{_DSA_ROW_PLAN_COLUMNS} columns, got {len(row_plan)}"
            )
        row_start = int(row_plan[0])
        q_len = int(row_plan[1])
        score_seq_id = int(row_plan[2])
        group_id = int(row_plan[4])
        prior_chunks = int(row_plan[5])
        query_position_start = int(row_plan[6])
        if q_len <= 0 or prior_chunks <= 0:
            continue
        if q_len == 1:
            mode = _DSA_SCORE_TILE_MODE_DECODE
            block_rows = 1
            chunk_step = decode_block_chunks
        elif q_len <= small_block_rows:
            mode = _DSA_SCORE_TILE_MODE_SMALL
            block_rows = small_block_rows
            chunk_step = block_chunks
        else:
            mode = _DSA_SCORE_TILE_MODE_LARGE
            block_rows = large_block_rows
            chunk_step = block_chunks

        for q_offset in range(0, q_len, block_rows):
            tile_rows = min(block_rows, q_len - q_offset)
            tile_row_start = row_start + q_offset
            tile_position_start = query_position_start + q_offset
            for chunk_start in range(0, prior_chunks, chunk_step):
                tile_plan_parts.append(
                    (
                        tile_row_start,
                        tile_rows,
                        score_seq_id,
                        group_id,
                        prior_chunks,
                        tile_position_start,
                        chunk_start,
                        mode,
                    )
                )
    return tile_plan_parts


def dsa_count_score_tile_plan_parts(
    row_plan_parts: typing.Sequence[typing.Sequence[int]],
    *,
    small_block_rows: int,
    large_block_rows: int,
    block_chunks: int,
    decode_block_chunks: int | None = None,
) -> tuple[int, int, int, int]:
    """Count score tiles from compact CPU row metadata.

    Returns total, decode, small-prefill, and large-prefill tile counts. This
    keeps CPU work proportional to the number of sparse metadata segments rather
    than the number of score tiles.
    """
    if small_block_rows <= 1:
        raise ValueError(f"small_block_rows must be > 1, got {small_block_rows}")
    if large_block_rows < small_block_rows:
        raise ValueError(
            "large_block_rows must be >= small_block_rows, got "
            f"{large_block_rows} < {small_block_rows}"
        )
    if block_chunks <= 0:
        raise ValueError(f"block_chunks must be positive, got {block_chunks}")
    if decode_block_chunks is None:
        decode_block_chunks = block_chunks
    if decode_block_chunks <= 0:
        raise ValueError(
            f"decode_block_chunks must be positive, got {decode_block_chunks}"
        )

    total_tiles = 0
    decode_tiles = 0
    small_tiles = 0
    large_tiles = 0
    for row_plan in row_plan_parts:
        if len(row_plan) != _DSA_ROW_PLAN_COLUMNS:
            raise ValueError(
                "row plan entries must have "
                f"{_DSA_ROW_PLAN_COLUMNS} columns, got {len(row_plan)}"
            )
        q_len = int(row_plan[1])
        prior_chunks = int(row_plan[5])
        if q_len <= 0 or prior_chunks <= 0:
            continue
        if q_len == 1:
            row_tiles = math.ceil(prior_chunks / decode_block_chunks)
            decode_tiles += row_tiles
        elif q_len <= small_block_rows:
            row_tiles = math.ceil(q_len / small_block_rows) * math.ceil(
                prior_chunks / block_chunks
            )
            small_tiles += row_tiles
        else:
            row_tiles = math.ceil(q_len / large_block_rows) * math.ceil(
                prior_chunks / block_chunks
            )
            large_tiles += row_tiles
        total_tiles += row_tiles
    return total_tiles, decode_tiles, small_tiles, large_tiles


if triton is not None and tl is not None:

    @triton.jit(
        do_not_specialize=[
            "stride_q_r",
            "stride_q_d",
            "stride_rep_c",
            "stride_rep_d",
            "stride_logits_r",
            "stride_logits_c",
            "q_indexer_dim",
            "max_prior_chunks",
            "score_scale",
        ]
    )
    def _dsa_chunk_score_kernel(
        query,
        chunk_reps,
        current_chunks,
        logits,
        stride_q_r,
        stride_q_d,
        stride_rep_c,
        stride_rep_d,
        stride_logits_r,
        stride_logits_c,
        q_indexer_dim,
        max_prior_chunks,
        score_scale,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        chunk_block = tl.program_id(1)
        chunk_offsets = chunk_block * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < q_indexer_dim

        q_vals = tl.load(
            query + row * stride_q_r + dim_offsets * stride_q_d,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        reps = tl.load(
            chunk_reps
            + chunk_offsets[:, None] * stride_rep_c
            + dim_offsets[None, :] * stride_rep_d,
            mask=(chunk_offsets[:, None] < max_prior_chunks) & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(reps * q_vals[None, :], axis=1) * score_scale
        current_chunk = tl.load(current_chunks + row).to(tl.int64)
        valid_chunk = (chunk_offsets < max_prior_chunks) & (
            chunk_offsets < current_chunk
        )
        scores = tl.where(valid_chunk, scores, -3.4028234663852886e38)
        tl.store(
            logits + row * stride_logits_r + chunk_offsets * stride_logits_c,
            scores,
            mask=chunk_offsets < max_prior_chunks,
        )

    @triton.jit(
        do_not_specialize=[
            "stride_q_r",
            "stride_q_d",
            "stride_rep_s",
            "stride_rep_c",
            "stride_rep_g",
            "stride_rep_d",
            "stride_logits_r",
            "stride_logits_c",
            "q_indexer_dim",
            "max_prior_chunks",
            "score_scale",
        ]
    )
    def _dsa_batched_chunk_score_kernel(
        query,
        chunk_reps,
        current_chunks,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
        logits,
        stride_q_r,
        stride_q_d,
        stride_rep_s,
        stride_rep_c,
        stride_rep_g,
        stride_rep_d,
        stride_logits_r,
        stride_logits_c,
        q_indexer_dim,
        max_prior_chunks,
        score_scale,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        chunk_block = tl.program_id(1)
        chunk_offsets = chunk_block * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
        dim_offsets = tl.arange(0, BLOCK_D)
        dim_mask = dim_offsets < q_indexer_dim

        seq_id = tl.load(row_seq_ids + row).to(tl.int64)
        group_id = tl.load(row_group_ids + row).to(tl.int64)
        row_prior_chunks = tl.load(row_num_prior_chunks + row).to(tl.int64)

        q_vals = tl.load(
            query + row * stride_q_r + dim_offsets * stride_q_d,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        reps = tl.load(
            chunk_reps
            + seq_id * stride_rep_s
            + chunk_offsets[:, None] * stride_rep_c
            + group_id * stride_rep_g
            + dim_offsets[None, :] * stride_rep_d,
            mask=(
                (chunk_offsets[:, None] < row_prior_chunks)
                & (chunk_offsets[:, None] < max_prior_chunks)
                & dim_mask[None, :]
            ),
            other=0.0,
        ).to(tl.float32)
        scores = tl.sum(reps * q_vals[None, :], axis=1) * score_scale
        current_chunk = tl.load(current_chunks + row).to(tl.int64)
        valid_chunk = (
            (chunk_offsets < row_prior_chunks)
            & (chunk_offsets < current_chunk)
            & (chunk_offsets < max_prior_chunks)
        )
        scores = tl.where(valid_chunk, scores, -3.4028234663852886e38)
        tl.store(
            logits + row * stride_logits_r + chunk_offsets * stride_logits_c,
            scores,
            mask=chunk_offsets < max_prior_chunks,
        )

    @triton.jit(
        do_not_specialize=[
            "stride_q_r",
            "stride_q_d",
            "stride_rep_s",
            "stride_rep_c",
            "stride_rep_g",
            "stride_rep_d",
            "stride_logits_r",
            "stride_logits_c",
            "plan_stride",
            "q_indexer_dim",
            "chunk_size",
            "max_prior_chunks",
            "score_scale",
        ]
    )
    def _dsa_batched_chunk_score_plan_kernel(
        query,
        chunk_reps,
        row_plan,
        logits,
        stride_q_r,
        stride_q_d,
        stride_rep_s,
        stride_rep_c,
        stride_rep_g,
        stride_rep_d,
        stride_logits_r,
        stride_logits_c,
        plan_stride,
        q_indexer_dim,
        chunk_size,
        max_prior_chunks,
        score_scale,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        plan_idx = tl.program_id(0)
        row_block = tl.program_id(1)
        chunk_block = tl.program_id(2)

        base = row_plan + plan_idx * plan_stride
        row_start = tl.load(base + 0)
        q_len = tl.load(base + 1)
        seq_id = tl.load(base + 2).to(tl.int64)
        group_id = tl.load(base + 4).to(tl.int64)
        row_prior_chunks = tl.load(base + 5)
        query_position_start = tl.load(base + 6)

        row_offsets = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        chunk_offsets = chunk_block * BLOCK_CHUNKS + tl.arange(0, BLOCK_CHUNKS)
        dim_offsets = tl.arange(0, BLOCK_D)

        rows = row_start + row_offsets
        row_mask = row_offsets < q_len
        dim_mask = dim_offsets < q_indexer_dim
        chunk_mask = (chunk_offsets < row_prior_chunks) & (
            chunk_offsets < max_prior_chunks
        )

        q_vals = tl.load(
            query + rows[:, None] * stride_q_r + dim_offsets[None, :] * stride_q_d,
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        rep_vals = tl.load(
            chunk_reps
            + seq_id * stride_rep_s
            + chunk_offsets[None, :] * stride_rep_c
            + group_id * stride_rep_g
            + dim_offsets[:, None] * stride_rep_d,
            mask=dim_mask[:, None] & chunk_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        scores = (
            tl.dot(
                q_vals,
                rep_vals,
                input_precision=INPUT_PRECISION,
            )
            * score_scale
        )
        current_chunks = (query_position_start + row_offsets) // chunk_size
        valid = (
            row_mask[:, None]
            & chunk_mask[None, :]
            & (chunk_offsets[None, :] < current_chunks[:, None])
        )
        scores = tl.where(valid, scores, -3.4028234663852886e38)
        tl.store(
            logits
            + rows[:, None] * stride_logits_r
            + chunk_offsets[None, :] * stride_logits_c,
            scores,
            mask=row_mask[:, None] & (chunk_offsets[None, :] < max_prior_chunks),
        )

    @triton.jit
    def _dsa_batched_chunk_score_tile_store(
        query,
        chunk_reps,
        tile_plan,
        logits,
        row_current_chunks,
        stride_q_r,
        stride_q_d,
        stride_rep_s,
        stride_rep_c,
        stride_rep_g,
        stride_rep_d,
        stride_logits_r,
        stride_logits_c,
        tile_stride,
        q_indexer_dim,
        chunk_size,
        max_prior_chunks,
        score_scale,
        tile_idx,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        base = tile_plan + tile_idx * tile_stride
        row_start = tl.load(base + 0)
        tile_rows = tl.load(base + 1)
        seq_id = tl.load(base + 2).to(tl.int64)
        group_id = tl.load(base + 3).to(tl.int64)
        row_prior_chunks = tl.load(base + 4)
        chunk_start = tl.load(base + 6)

        row_offsets = tl.arange(0, BLOCK_ROWS)
        chunk_offsets = chunk_start + tl.arange(0, BLOCK_CHUNKS)
        dim_offsets = tl.arange(0, BLOCK_D)

        rows = row_start + row_offsets
        row_mask = row_offsets < tile_rows
        dim_mask = dim_offsets < q_indexer_dim
        chunk_mask = (chunk_offsets < row_prior_chunks) & (
            chunk_offsets < max_prior_chunks
        )

        # BF16 operands select the tensor-core path while tl.dot keeps its
        # default FP32 accumulator and the logits remain FP32.
        q_vals = tl.load(
            query + rows[:, None] * stride_q_r + dim_offsets[None, :] * stride_q_d,
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        rep_vals = tl.load(
            chunk_reps
            + seq_id * stride_rep_s
            + chunk_offsets[None, :] * stride_rep_c
            + group_id * stride_rep_g
            + dim_offsets[:, None] * stride_rep_d,
            mask=dim_mask[:, None] & chunk_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        scores = (
            tl.dot(
                q_vals,
                rep_vals,
                input_precision=INPUT_PRECISION,
            )
            * score_scale
        )
        current_chunks = tl.load(
            row_current_chunks + rows,
            mask=row_mask,
            other=0,
        )
        valid = (
            row_mask[:, None]
            & chunk_mask[None, :]
            & (chunk_offsets[None, :] < current_chunks[:, None])
        )
        scores = tl.where(valid, scores, -3.4028234663852886e38)
        tl.store(
            logits
            + rows[:, None] * stride_logits_r
            + chunk_offsets[None, :] * stride_logits_c,
            scores,
            mask=row_mask[:, None] & (chunk_offsets[None, :] < max_prior_chunks),
        )

    @triton.jit
    def _dsa_batched_physical_cache_score_tile_store(
        query,
        cache_values,
        cache_valid,
        block_table,
        tile_plan,
        logits,
        row_current_chunks,
        stride_q_r,
        stride_q_d,
        stride_cache_b,
        stride_cache_g,
        stride_cache_d,
        stride_block_table_s,
        stride_block_table_c,
        stride_logits_r,
        stride_logits_c,
        tile_stride,
        q_indexer_dim,
        max_prior_chunks,
        num_cache_blocks,
        score_scale,
        select_all_threshold,
        tile_idx,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        ASSUME_VALID: tl.constexpr,
    ):
        """Score logical chunks by following their physical-page IDs."""
        base = tile_plan + tile_idx * tile_stride
        row_start = tl.load(base + 0)
        tile_rows = tl.load(base + 1)
        seq_id = tl.load(base + 2).to(tl.int64)
        group_id = tl.load(base + 3).to(tl.int64)
        row_prior_chunks = tl.load(base + 4)
        chunk_start = tl.load(base + 6)

        row_offsets = tl.arange(0, BLOCK_ROWS)
        chunk_offsets = chunk_start + tl.arange(0, BLOCK_CHUNKS)
        dim_offsets = tl.arange(0, BLOCK_D)
        rows = row_start + row_offsets
        row_mask = row_offsets < tile_rows
        dim_mask = dim_offsets < q_indexer_dim
        current_chunks = tl.load(
            row_current_chunks + rows,
            mask=row_mask,
            other=0,
        )
        max_current_chunk = tl.max(current_chunks)
        # The launch grid is capacity-sized for CUDA graph reuse. Avoid cache
        # reads and dot products for runtime padding. The count-aware CUDA top-k
        # also returns every index directly when the valid row fits in top-k.
        if (chunk_start >= max_current_chunk) | (
            max_current_chunk <= select_all_threshold
        ):
            return
        chunk_mask = (chunk_offsets < row_prior_chunks) & (
            chunk_offsets < max_prior_chunks
        )

        physical_blocks = tl.load(
            block_table
            + seq_id * stride_block_table_s
            + chunk_offsets * stride_block_table_c,
            mask=chunk_mask,
            other=-1,
        ).to(tl.int64)
        physical_valid = (
            chunk_mask
            & (physical_blocks >= 0)
            & (physical_blocks < num_cache_blocks)
        )
        if ASSUME_VALID:
            cache_ready = physical_valid
        else:
            cache_ready = (
                tl.load(
                    cache_valid + physical_blocks,
                    mask=physical_valid,
                    other=0,
                )
                != 0
            )

        q_vals = tl.load(
            query + rows[:, None] * stride_q_r + dim_offsets[None, :] * stride_q_d,
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        rep_vals = tl.load(
            cache_values
            + physical_blocks[None, :] * stride_cache_b
            + group_id * stride_cache_g
            + dim_offsets[:, None] * stride_cache_d,
            mask=dim_mask[:, None] & cache_ready[None, :],
            other=0.0,
        ).to(tl.bfloat16)
        scores = (
            tl.dot(q_vals, rep_vals, input_precision=INPUT_PRECISION) * score_scale
        )
        valid = (
            row_mask[:, None]
            & cache_ready[None, :]
            & (chunk_offsets[None, :] < current_chunks[:, None])
        )
        scores = tl.where(valid, scores, -3.4028234663852886e38)
        tl.store(
            logits
            + rows[:, None] * stride_logits_r
            + chunk_offsets[None, :] * stride_logits_c,
            scores,
            mask=row_mask[:, None] & (chunk_offsets[None, :] < max_prior_chunks),
        )

    @triton.jit(
        do_not_specialize=[
            "stride_q_r",
            "stride_q_d",
            "stride_rep_s",
            "stride_rep_c",
            "stride_rep_g",
            "stride_rep_d",
            "stride_logits_r",
            "stride_logits_c",
            "tile_stride",
            "q_indexer_dim",
            "chunk_size",
            "max_prior_chunks",
            "score_scale",
        ]
    )
    def _dsa_batched_chunk_score_tile_plan_kernel(
        query,
        chunk_reps,
        tile_plan,
        logits,
        row_current_chunks,
        stride_q_r,
        stride_q_d,
        stride_rep_s,
        stride_rep_c,
        stride_rep_g,
        stride_rep_d,
        stride_logits_r,
        stride_logits_c,
        tile_stride,
        q_indexer_dim,
        chunk_size,
        max_prior_chunks,
        score_scale,
        SMALL_BLOCK_ROWS: tl.constexpr,
        LARGE_BLOCK_ROWS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        DECODE_BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
    ):
        tile_idx = tl.program_id(0)
        mode = tl.load(tile_plan + tile_idx * tile_stride + 7)
        if mode == 0:
            _dsa_batched_chunk_score_tile_store(
                query,
                chunk_reps,
                tile_plan,
                logits,
                row_current_chunks,
                stride_q_r,
                stride_q_d,
                stride_rep_s,
                stride_rep_c,
                stride_rep_g,
                stride_rep_d,
                stride_logits_r,
                stride_logits_c,
                tile_stride,
                q_indexer_dim,
                chunk_size,
                max_prior_chunks,
                score_scale,
                tile_idx,
                BLOCK_ROWS=1,
                BLOCK_CHUNKS=DECODE_BLOCK_CHUNKS,
                BLOCK_D=BLOCK_D,
                INPUT_PRECISION=INPUT_PRECISION,
            )
        elif mode == 1:
            _dsa_batched_chunk_score_tile_store(
                query,
                chunk_reps,
                tile_plan,
                logits,
                row_current_chunks,
                stride_q_r,
                stride_q_d,
                stride_rep_s,
                stride_rep_c,
                stride_rep_g,
                stride_rep_d,
                stride_logits_r,
                stride_logits_c,
                tile_stride,
                q_indexer_dim,
                chunk_size,
                max_prior_chunks,
                score_scale,
                tile_idx,
                BLOCK_ROWS=SMALL_BLOCK_ROWS,
                BLOCK_CHUNKS=BLOCK_CHUNKS,
                BLOCK_D=BLOCK_D,
                INPUT_PRECISION=INPUT_PRECISION,
            )
        else:
            _dsa_batched_chunk_score_tile_store(
                query,
                chunk_reps,
                tile_plan,
                logits,
                row_current_chunks,
                stride_q_r,
                stride_q_d,
                stride_rep_s,
                stride_rep_c,
                stride_rep_g,
                stride_rep_d,
                stride_logits_r,
                stride_logits_c,
                tile_stride,
                q_indexer_dim,
                chunk_size,
                max_prior_chunks,
                score_scale,
                tile_idx,
                BLOCK_ROWS=LARGE_BLOCK_ROWS,
                BLOCK_CHUNKS=BLOCK_CHUNKS,
                BLOCK_D=BLOCK_D,
                INPUT_PRECISION=INPUT_PRECISION,
            )

    @triton.jit(
        do_not_specialize=[
            "stride_q_r",
            "stride_q_d",
            "stride_cache_b",
            "stride_cache_g",
            "stride_cache_d",
            "stride_block_table_s",
            "stride_block_table_c",
            "stride_logits_r",
            "stride_logits_c",
            "tile_stride",
            "q_indexer_dim",
            "max_prior_chunks",
            "num_cache_blocks",
            "score_scale",
            "select_all_threshold",
        ]
    )
    def _dsa_batched_physical_cache_score_tile_plan_kernel(
        query,
        cache_values,
        cache_valid,
        block_table,
        tile_plan,
        logits,
        row_current_chunks,
        stride_q_r,
        stride_q_d,
        stride_cache_b,
        stride_cache_g,
        stride_cache_d,
        stride_block_table_s,
        stride_block_table_c,
        stride_logits_r,
        stride_logits_c,
        tile_stride,
        q_indexer_dim,
        max_prior_chunks,
        num_cache_blocks,
        score_scale,
        select_all_threshold,
        SMALL_BLOCK_ROWS: tl.constexpr,
        LARGE_BLOCK_ROWS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        DECODE_BLOCK_CHUNKS: tl.constexpr,
        BLOCK_D: tl.constexpr,
        INPUT_PRECISION: tl.constexpr,
        ASSUME_VALID: tl.constexpr,
    ):
        tile_idx = tl.program_id(0)
        mode = tl.load(tile_plan + tile_idx * tile_stride + 7)
        if mode == 0:
            _dsa_batched_physical_cache_score_tile_store(
                query,
                cache_values,
                cache_valid,
                block_table,
                tile_plan,
                logits,
                row_current_chunks,
                stride_q_r,
                stride_q_d,
                stride_cache_b,
                stride_cache_g,
                stride_cache_d,
                stride_block_table_s,
                stride_block_table_c,
                stride_logits_r,
                stride_logits_c,
                tile_stride,
                q_indexer_dim,
                max_prior_chunks,
                num_cache_blocks,
                score_scale,
                select_all_threshold,
                tile_idx,
                BLOCK_ROWS=1,
                BLOCK_CHUNKS=DECODE_BLOCK_CHUNKS,
                BLOCK_D=BLOCK_D,
                INPUT_PRECISION=INPUT_PRECISION,
                ASSUME_VALID=ASSUME_VALID,
            )
        elif mode == 1:
            _dsa_batched_physical_cache_score_tile_store(
                query,
                cache_values,
                cache_valid,
                block_table,
                tile_plan,
                logits,
                row_current_chunks,
                stride_q_r,
                stride_q_d,
                stride_cache_b,
                stride_cache_g,
                stride_cache_d,
                stride_block_table_s,
                stride_block_table_c,
                stride_logits_r,
                stride_logits_c,
                tile_stride,
                q_indexer_dim,
                max_prior_chunks,
                num_cache_blocks,
                score_scale,
                select_all_threshold,
                tile_idx,
                BLOCK_ROWS=SMALL_BLOCK_ROWS,
                BLOCK_CHUNKS=BLOCK_CHUNKS,
                BLOCK_D=BLOCK_D,
                INPUT_PRECISION=INPUT_PRECISION,
                ASSUME_VALID=ASSUME_VALID,
            )
        else:
            _dsa_batched_physical_cache_score_tile_store(
                query,
                cache_values,
                cache_valid,
                block_table,
                tile_plan,
                logits,
                row_current_chunks,
                stride_q_r,
                stride_q_d,
                stride_cache_b,
                stride_cache_g,
                stride_cache_d,
                stride_block_table_s,
                stride_block_table_c,
                stride_logits_r,
                stride_logits_c,
                tile_stride,
                q_indexer_dim,
                max_prior_chunks,
                num_cache_blocks,
                score_scale,
                select_all_threshold,
                tile_idx,
                BLOCK_ROWS=LARGE_BLOCK_ROWS,
                BLOCK_CHUNKS=BLOCK_CHUNKS,
                BLOCK_D=BLOCK_D,
                INPUT_PRECISION=INPUT_PRECISION,
                ASSUME_VALID=ASSUME_VALID,
            )

    @triton.jit
    def _dsa_build_score_tile_plan_kernel(
        row_plan,
        tile_plan,
        plan_stride,
        tile_stride,
        SMALL_BLOCK_ROWS: tl.constexpr,
        LARGE_BLOCK_ROWS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        DECODE_BLOCK_CHUNKS: tl.constexpr,
        MAX_TILES_PER_ROW_PLAN: tl.constexpr,
    ):
        plan_idx = tl.program_id(0)
        offsets = tl.arange(0, MAX_TILES_PER_ROW_PLAN)

        base = row_plan + plan_idx * plan_stride
        row_start = tl.load(base + 0)
        q_len = tl.load(base + 1)
        score_seq_id = tl.load(base + 2)
        group_id = tl.load(base + 4)
        prior_chunks = tl.load(base + 5)
        query_position_start = tl.load(base + 6)
        tile_offset = tl.load(base + 7)
        tile_count = tl.load(base + 8)

        is_decode = q_len == 1
        is_small = (q_len > 1) & (q_len <= SMALL_BLOCK_ROWS)
        block_rows = tl.where(
            is_decode, 1, tl.where(is_small, SMALL_BLOCK_ROWS, LARGE_BLOCK_ROWS)
        )
        chunk_step = tl.where(is_decode, DECODE_BLOCK_CHUNKS, BLOCK_CHUNKS)
        chunks_per_row_tile = tl.cdiv(prior_chunks, chunk_step)
        safe_chunks_per_row_tile = tl.maximum(chunks_per_row_tile, 1)

        q_tile = offsets // safe_chunks_per_row_tile
        chunk_tile = offsets - q_tile * safe_chunks_per_row_tile
        q_offset = q_tile * block_rows
        chunk_start = chunk_tile * chunk_step
        tile_rows = tl.minimum(block_rows, q_len - q_offset)
        mode = tl.where(
            is_decode,
            0,
            tl.where(is_small, 1, 2),
        )
        valid = offsets < tile_count
        out = tile_plan + (tile_offset + offsets) * tile_stride
        tl.store(out + 0, row_start + q_offset, mask=valid)
        tl.store(out + 1, tile_rows, mask=valid)
        tl.store(out + 2, score_seq_id, mask=valid)
        tl.store(out + 3, group_id, mask=valid)
        tl.store(out + 4, prior_chunks, mask=valid)
        tl.store(out + 5, query_position_start + q_offset, mask=valid)
        tl.store(out + 6, chunk_start, mask=valid)
        tl.store(out + 7, mode, mask=valid)

    @triton.jit
    def _dsa_build_fixed_decode_score_tile_plan_kernel(
        tile_plan,
        tile_stride,
        TILES_PER_ROW: tl.constexpr,
        MAX_PRIOR_CHUNKS: tl.constexpr,
        GROUP_ID: tl.constexpr,
        DECODE_BLOCK_CHUNKS: tl.constexpr,
        MODE: tl.constexpr,
    ):
        """Build a padded decode plan whose shape is independent of seq_lens."""
        tile_idx = tl.program_id(0)
        row = tile_idx // TILES_PER_ROW
        chunk_tile = tile_idx - row * TILES_PER_ROW
        out = tile_plan + tile_idx * tile_stride
        tl.store(out + 0, row)
        tl.store(out + 1, 1)
        tl.store(out + 2, row)
        tl.store(out + 3, GROUP_ID)
        tl.store(out + 4, MAX_PRIOR_CHUNKS)
        tl.store(out + 5, 0)
        tl.store(out + 6, chunk_tile * DECODE_BLOCK_CHUNKS)
        tl.store(out + 7, MODE)

    @triton.jit
    def _dsa_build_score_metadata_kernel(
        query_start_loc,
        row_query_start_loc,
        seq_lens,
        row_plan,
        score_row_seq_ids,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
        row_current_chunks,
        row_tail_lens,
        query_start_loc_stride,
        row_query_start_loc_stride,
        seq_lens_stride,
        plan_stride,
        num_actual_tokens,
        representative_group_idx,
        chunk_size,
        dense_decode_threshold,
        dense_prefill_threshold,
        active_seq_count,
        SMALL_BLOCK_ROWS: tl.constexpr,
        LARGE_BLOCK_ROWS: tl.constexpr,
        BLOCK_CHUNKS: tl.constexpr,
        DECODE_BLOCK_CHUNKS: tl.constexpr,
        CHUNK_TOP_K: tl.constexpr,
        REPRESENTATIVES_USE_ORIGINAL_SEQ_IDS: tl.constexpr,
        MAX_ACTIVE_SEQS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        seq_offsets = tl.arange(0, MAX_ACTIVE_SEQS)
        seq_mask = seq_offsets < active_seq_count

        q_start_all = tl.load(
            query_start_loc + seq_offsets * query_start_loc_stride,
            mask=seq_mask,
            other=num_actual_tokens,
        )
        q_end_all = tl.load(
            query_start_loc + (seq_offsets + 1) * query_start_loc_stride,
            mask=seq_mask,
            other=num_actual_tokens,
        )
        q_start_all = tl.minimum(q_start_all, num_actual_tokens)
        q_end_all = tl.minimum(q_end_all, num_actual_tokens)
        q_len_all = q_end_all - q_start_all
        row_q_start_all = tl.load(
            row_query_start_loc + seq_offsets * row_query_start_loc_stride,
            mask=seq_mask,
            other=num_actual_tokens,
        )
        row_q_end_all = tl.load(
            row_query_start_loc + (seq_offsets + 1) * row_query_start_loc_stride,
            mask=seq_mask,
            other=num_actual_tokens,
        )
        row_q_start_all = tl.minimum(row_q_start_all, num_actual_tokens)
        row_q_end_all = tl.minimum(row_q_end_all, num_actual_tokens)
        row_q_len_all = row_q_end_all - row_q_start_all
        key_len_all = tl.load(
            seq_lens + seq_offsets * seq_lens_stride,
            mask=seq_mask,
            other=0,
        )
        query_position_start_all = key_len_all - q_len_all
        num_chunks_all = tl.cdiv(key_len_all, chunk_size)
        prior_chunks_all = tl.maximum(num_chunks_all - 1, 0)
        dense_threshold_all = tl.where(
            q_len_all > 1,
            dense_prefill_threshold,
            dense_decode_threshold,
        )
        fits_dense_all = (dense_threshold_all >= 0) & (
            key_len_all <= dense_threshold_all
        )
        sparse_all = (
            seq_mask
            & (q_len_all > 0)
            & (key_len_all > 0)
            & (query_position_start_all >= 0)
            & (prior_chunks_all > 0)
            & (CHUNK_TOP_K > 0)
            & ~fits_dense_all
        )

        is_decode_all = row_q_len_all == 1
        is_small_all = (row_q_len_all > 1) & (row_q_len_all <= SMALL_BLOCK_ROWS)
        block_rows_all = tl.where(
            is_decode_all,
            1,
            tl.where(is_small_all, SMALL_BLOCK_ROWS, LARGE_BLOCK_ROWS),
        )
        chunk_step_all = tl.where(
            is_decode_all,
            DECODE_BLOCK_CHUNKS,
            BLOCK_CHUNKS,
        )
        tile_count_all = tl.where(
            sparse_all,
            tl.cdiv(row_q_len_all, block_rows_all)
            * tl.cdiv(prior_chunks_all, chunk_step_all),
            0,
        )
        sparse_rank_all = tl.cumsum(sparse_all.to(tl.int32), 0) - 1
        tile_offset_all = tl.cumsum(tile_count_all, 0) - tile_count_all

        this_seq = seq_offsets == seq_idx
        q_start = tl.sum(tl.where(this_seq, row_q_start_all, 0), 0)
        q_len = tl.sum(tl.where(this_seq, row_q_len_all, 0), 0)
        query_position_start = tl.sum(
            tl.where(this_seq, query_position_start_all, 0), 0
        )
        prior_chunks = tl.sum(tl.where(this_seq, prior_chunks_all, 0), 0)
        sparse_rank = tl.sum(tl.where(this_seq, sparse_rank_all, 0), 0)
        tile_offset = tl.sum(tl.where(this_seq, tile_offset_all, 0), 0)
        tile_count = tl.sum(tl.where(this_seq, tile_count_all, 0), 0)
        is_sparse = tl.sum(tl.where(this_seq, sparse_all.to(tl.int32), 0), 0) != 0
        score_seq_id = tl.where(
            REPRESENTATIVES_USE_ORIGINAL_SEQ_IDS,
            seq_idx,
            sparse_rank,
        )

        write_plan = (block_idx == 0) & is_sparse
        plan_base = row_plan + sparse_rank * plan_stride
        tl.store(plan_base + 0, q_start, mask=write_plan)
        tl.store(plan_base + 1, q_len, mask=write_plan)
        tl.store(plan_base + 2, score_seq_id, mask=write_plan)
        tl.store(plan_base + 3, seq_idx, mask=write_plan)
        tl.store(plan_base + 4, representative_group_idx, mask=write_plan)
        tl.store(plan_base + 5, prior_chunks, mask=write_plan)
        tl.store(plan_base + 6, query_position_start, mask=write_plan)
        tl.store(plan_base + 7, tile_offset, mask=write_plan)
        tl.store(plan_base + 8, tile_count, mask=write_plan)

        row_offsets = block_idx * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        row_mask = (seq_idx < active_seq_count) & (row_offsets < q_len)
        rows = q_start + row_offsets
        positions = query_position_start + row_offsets
        current_chunks = positions // chunk_size
        tail_lens = positions - current_chunks * chunk_size + 1
        row_prior_chunks = tl.where(is_sparse, prior_chunks, 0)
        row_score_seq_id = tl.where(is_sparse, score_seq_id, 0)
        row_group_id = tl.where(is_sparse, representative_group_idx, 0)

        tl.store(score_row_seq_ids + rows, row_score_seq_id, mask=row_mask)
        tl.store(row_seq_ids + rows, seq_idx, mask=row_mask)
        tl.store(row_group_ids + rows, row_group_id, mask=row_mask)
        tl.store(row_num_prior_chunks + rows, row_prior_chunks, mask=row_mask)
        tl.store(row_current_chunks + rows, current_chunks, mask=row_mask)
        tl.store(row_tail_lens + rows, tail_lens, mask=row_mask)

    @triton.jit
    def _dsa_batched_row_metadata_kernel(
        row_plan,
        score_row_seq_ids,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
        row_current_chunks,
        row_tail_lens,
        chunk_size,
        plan_stride,
        BLOCK_ROWS: tl.constexpr,
    ):
        plan_idx = tl.program_id(0)
        block_idx = tl.program_id(1)
        offsets = block_idx * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)

        base = row_plan + plan_idx * plan_stride
        row_start = tl.load(base + 0)
        q_len = tl.load(base + 1)
        score_seq_id = tl.load(base + 2)
        block_table_seq_id = tl.load(base + 3)
        group_id = tl.load(base + 4)
        prior_chunks = tl.load(base + 5)
        query_position_start = tl.load(base + 6)

        mask = offsets < q_len
        row = row_start + offsets
        positions = query_position_start + offsets
        current_chunks = positions // chunk_size
        tail_lens = positions - current_chunks * chunk_size + 1

        tl.store(score_row_seq_ids + row, score_seq_id, mask=mask)
        tl.store(row_seq_ids + row, block_table_seq_id, mask=mask)
        tl.store(row_group_ids + row, group_id, mask=mask)
        tl.store(row_num_prior_chunks + row, prior_chunks, mask=mask)
        tl.store(row_current_chunks + row, current_chunks, mask=mask)
        tl.store(row_tail_lens + row, tail_lens, mask=mask)


def _dsa_score_logits_and_valid_torch(
    score_query_states: torch.Tensor,
    chunk_representatives: torch.Tensor,
    selectable_counts: torch.Tensor,
    score_scale: float,
    chunk_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        score_query_states.is_cuda
        and chunk_representatives.is_cuda
        and score_query_states.dtype == chunk_representatives.dtype
        and score_query_states.dtype in (torch.bfloat16, torch.float16)
    ):
        chunk_logits = torch.mm(
            score_query_states,
            chunk_representatives.transpose(0, 1),
            out_dtype=torch.float32,
        )
    else:
        chunk_logits = torch.mm(
            score_query_states.float(),
            chunk_representatives.float().transpose(0, 1),
        )
    chunk_logits.mul_(score_scale)
    if chunk_ids is None:
        chunk_ids = torch.arange(
            chunk_representatives.shape[0],
            device=score_query_states.device,
            dtype=selectable_counts.dtype,
        )
    chunk_valid = chunk_ids[None, :] < selectable_counts[:, None]
    chunk_logits = chunk_logits.masked_fill(
        ~chunk_valid,
        torch.finfo(chunk_logits.dtype).min,
    )
    return chunk_logits, chunk_valid


def dsa_score_logits_torch(
    *,
    score_query_states: torch.Tensor,
    chunk_representatives: torch.Tensor,
    current_chunks: torch.Tensor,
    logit_scale: float,
    q_indexer_dim: int,
    chunk_ids: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Compute DSA chunk score logits with eager torch matmul and masking."""
    if score_query_states.dim() != 2 or chunk_representatives.dim() != 2:
        return None
    if current_chunks.dim() != 1:
        return None
    if score_query_states.shape[0] != current_chunks.shape[0]:
        return None
    if score_query_states.shape[1] != q_indexer_dim:
        return None
    if chunk_representatives.shape[1] != q_indexer_dim:
        return None
    if chunk_ids is not None and (
        chunk_ids.dim() != 1
        or chunk_ids.shape[0] != chunk_representatives.shape[0]
        or chunk_ids.device != score_query_states.device
    ):
        return None

    num_rows = int(score_query_states.shape[0])
    max_prior_chunks = int(chunk_representatives.shape[0])
    if num_rows == 0 or max_prior_chunks == 0:
        return score_query_states.new_empty(
            num_rows, max_prior_chunks, dtype=torch.float32
        )

    selectable_counts = current_chunks.clamp(
        min=0,
        max=max_prior_chunks,
    ).to(device=score_query_states.device, dtype=torch.long)
    score_scale = logit_scale / math.sqrt(q_indexer_dim)
    chunk_logits, _ = _dsa_score_logits_and_valid_torch(
        score_query_states,
        chunk_representatives,
        selectable_counts,
        score_scale,
        chunk_ids=chunk_ids,
    )
    return chunk_logits


def dsa_score_topk_torch(
    *,
    score_query_states: torch.Tensor,
    chunk_representatives: torch.Tensor,
    current_chunks: torch.Tensor,
    chunk_top_k: int,
    logit_scale: float,
    q_indexer_dim: int,
    chunk_ids: torch.Tensor | None = None,
    return_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Compute chunk score top-k with torch matmul/topk primitives."""
    if score_query_states.dim() != 2 or chunk_representatives.dim() != 2:
        return None
    if current_chunks.dim() != 1:
        return None
    if score_query_states.shape[0] != current_chunks.shape[0]:
        return None
    if score_query_states.shape[1] != q_indexer_dim:
        return None
    if chunk_representatives.shape[1] != q_indexer_dim:
        return None
    if chunk_ids is not None and (
        chunk_ids.dim() != 1
        or chunk_ids.shape[0] != chunk_representatives.shape[0]
        or chunk_ids.device != score_query_states.device
    ):
        return None
    if chunk_top_k <= 0:
        return None

    num_rows = int(score_query_states.shape[0])
    max_prior_chunks = int(chunk_representatives.shape[0])
    if num_rows == 0 or max_prior_chunks == 0:
        empty_indices = torch.empty(
            num_rows,
            0,
            device=score_query_states.device,
            dtype=torch.long,
        )
        empty_valid = torch.empty(
            num_rows,
            0,
            device=score_query_states.device,
            dtype=torch.bool,
        )
        return empty_indices, empty_valid, None

    top_k = min(chunk_top_k, max_prior_chunks)
    selectable_counts = current_chunks.clamp(
        min=0,
        max=max_prior_chunks,
    ).to(device=score_query_states.device, dtype=torch.long)
    score_scale = logit_scale / math.sqrt(q_indexer_dim)
    chunk_logits, chunk_valid = _dsa_score_logits_and_valid_torch(
        score_query_states,
        chunk_representatives,
        selectable_counts,
        score_scale,
        chunk_ids=chunk_ids,
    )
    top_chunk_indices = chunk_logits.topk(k=top_k, dim=-1, sorted=False).indices
    top_chunk_valid = chunk_valid.gather(dim=-1, index=top_chunk_indices)
    top_chunk_indices = top_chunk_indices.masked_fill(~top_chunk_valid, 0)
    return top_chunk_indices, top_chunk_valid, chunk_logits if return_logits else None


def dsa_score_topk_triton(
    *,
    score_query_states: torch.Tensor,
    chunk_representatives: torch.Tensor,
    current_chunks: torch.Tensor,
    chunk_top_k: int,
    logit_scale: float,
    q_indexer_dim: int,
    return_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Compute chunk score top-k with Triton plus vLLM's CUDA top-k kernel.

    The returned indices are chunk ids. The validity mask is slot-based:
    ``slot < min(current_chunk, chunk_top_k)``. This avoids treating undefined
    filler indices from rows shorter than ``chunk_top_k`` as real recalls.
    """
    if triton is None or tl is None:
        return None
    if not _has_top_k_per_row_prefill():
        return None
    if not score_query_states.is_cuda or not chunk_representatives.is_cuda:
        return None
    if score_query_states.dim() != 2 or chunk_representatives.dim() != 2:
        return None
    if current_chunks.dim() != 1:
        return None
    if score_query_states.shape[0] != current_chunks.shape[0]:
        return None
    if score_query_states.shape[1] != q_indexer_dim:
        return None
    if chunk_representatives.shape[1] != q_indexer_dim:
        return None
    if chunk_representatives.dtype not in (torch.float32, torch.bfloat16):
        return None
    if chunk_top_k <= 0:
        return None

    num_rows = int(score_query_states.shape[0])
    max_prior_chunks = int(chunk_representatives.shape[0])
    if num_rows == 0 or max_prior_chunks == 0:
        empty_indices = torch.empty(
            num_rows,
            0,
            device=score_query_states.device,
            dtype=torch.long,
        )
        empty_valid = torch.empty(
            num_rows,
            0,
            device=score_query_states.device,
            dtype=torch.bool,
        )
        return empty_indices, empty_valid, None

    top_k = min(chunk_top_k, max_prior_chunks)
    score_query_states = score_query_states.contiguous()
    chunk_representatives = chunk_representatives.contiguous()
    selectable_counts = (
        current_chunks.clamp(
            min=0,
            max=max_prior_chunks,
        )
        .to(device=score_query_states.device, dtype=torch.int32)
        .contiguous()
    )
    logits = torch.empty(
        num_rows,
        max_prior_chunks,
        device=score_query_states.device,
        dtype=torch.float32,
    )
    block_chunks = _triton_scoring_block_chunks()
    block_d = triton.next_power_of_2(q_indexer_dim)
    _dsa_chunk_score_kernel[(num_rows, triton.cdiv(max_prior_chunks, block_chunks))](
        score_query_states,
        chunk_representatives,
        selectable_counts,
        logits,
        score_query_states.stride(0),
        score_query_states.stride(1),
        chunk_representatives.stride(0),
        chunk_representatives.stride(1),
        logits.stride(0),
        logits.stride(1),
        q_indexer_dim,
        max_prior_chunks,
        logit_scale / math.sqrt(q_indexer_dim),
        BLOCK_CHUNKS=block_chunks,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )

    top_chunk_indices_i32 = torch.empty(
        num_rows,
        top_k,
        device=score_query_states.device,
        dtype=torch.int32,
    )
    row_starts = torch.zeros(
        num_rows,
        device=score_query_states.device,
        dtype=torch.int32,
    )
    torch.ops._C.top_k_per_row_prefill(
        logits,
        row_starts,
        selectable_counts,
        top_chunk_indices_i32,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )
    selected_counts = selectable_counts.clamp(max=top_k).to(torch.long)
    top_chunk_valid = (
        torch.arange(top_k, device=score_query_states.device, dtype=torch.long)[
            None,
            :,
        ]
        < selected_counts[:, None]
    )
    top_chunk_indices_i32 = top_chunk_indices_i32.masked_fill(
        ~top_chunk_valid,
        0,
    )
    top_chunk_indices = top_chunk_indices_i32.to(torch.long)
    return top_chunk_indices, top_chunk_valid, logits if return_logits else None


def dsa_batched_score_logits_triton(
    *,
    score_query_states: torch.Tensor,
    chunk_representatives: torch.Tensor,
    current_chunks: torch.Tensor,
    row_seq_ids: torch.Tensor,
    row_group_ids: torch.Tensor,
    row_num_prior_chunks: torch.Tensor,
    logit_scale: float,
    q_indexer_dim: int,
    max_prior_chunks: int | None = None,
) -> torch.Tensor | None:
    """Compute DSA chunk logits for rows drawn from different sequences.

    ``chunk_representatives`` is shaped
    ``[num_sequences, max_chunks, num_groups, q_indexer_dim]``. Each row in
    ``score_query_states`` selects its sequence and group through
    ``row_seq_ids`` and ``row_group_ids``.
    """
    if triton is None or tl is None:
        return None
    if not score_query_states.is_cuda or not chunk_representatives.is_cuda:
        return None
    if score_query_states.dim() != 2 or chunk_representatives.dim() != 4:
        return None
    if score_query_states.shape[1] != q_indexer_dim:
        return None
    if chunk_representatives.shape[3] != q_indexer_dim:
        return None
    if chunk_representatives.dtype not in (torch.float32, torch.bfloat16):
        return None

    num_rows = int(score_query_states.shape[0])
    row_tensors = (
        current_chunks,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
    )
    if any(
        tensor.dim() != 1 or int(tensor.shape[0]) != num_rows for tensor in row_tensors
    ):
        return None
    if any(tensor.device != score_query_states.device for tensor in row_tensors):
        return None

    if max_prior_chunks is None:
        if num_rows == 0:
            max_prior_chunks = 0
        else:
            max_prior_chunks = int(row_num_prior_chunks.max().item())
    if max_prior_chunks < 0:
        return None
    if num_rows == 0 or max_prior_chunks == 0:
        return torch.empty(
            num_rows,
            max_prior_chunks,
            device=score_query_states.device,
            dtype=torch.float32,
        )

    current_chunks = current_chunks.to(
        device=score_query_states.device,
        dtype=torch.int32,
    ).contiguous()
    row_seq_ids = row_seq_ids.to(
        device=score_query_states.device,
        dtype=torch.int32,
    ).contiguous()
    row_group_ids = row_group_ids.to(
        device=score_query_states.device,
        dtype=torch.int32,
    ).contiguous()
    row_num_prior_chunks = row_num_prior_chunks.to(
        device=score_query_states.device,
        dtype=torch.int32,
    ).contiguous()

    logits = torch.empty(
        num_rows,
        max_prior_chunks,
        device=score_query_states.device,
        dtype=torch.float32,
    )
    block_chunks = _triton_scoring_block_chunks()
    block_d = triton.next_power_of_2(q_indexer_dim)
    _dsa_batched_chunk_score_kernel[
        (num_rows, triton.cdiv(max_prior_chunks, block_chunks))
    ](
        score_query_states,
        chunk_representatives,
        current_chunks,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
        logits,
        score_query_states.stride(0),
        score_query_states.stride(1),
        chunk_representatives.stride(0),
        chunk_representatives.stride(1),
        chunk_representatives.stride(2),
        chunk_representatives.stride(3),
        logits.stride(0),
        logits.stride(1),
        q_indexer_dim,
        max_prior_chunks,
        logit_scale / math.sqrt(q_indexer_dim),
        BLOCK_CHUNKS=block_chunks,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return logits


def dsa_batched_score_logits_plan_triton(
    *,
    score_query_states: torch.Tensor,
    chunk_representatives: torch.Tensor,
    row_plan: torch.Tensor,
    total_rows: int,
    chunk_size: int,
    max_q_len: int,
    logit_scale: float,
    q_indexer_dim: int,
    max_prior_chunks: int | None = None,
) -> torch.Tensor | None:
    """Compute DSA chunk logits from compact per-sequence row metadata.

    This path scores a tile of query rows against a tile of chunk
    representatives for each row-plan entry, reusing representative loads across
    multiple query rows from the same sequence/group.
    """
    if triton is None or tl is None:
        return None
    if not score_query_states.is_cuda or not chunk_representatives.is_cuda:
        return None
    if score_query_states.dim() != 2 or chunk_representatives.dim() != 4:
        return None
    if row_plan.dim() != 2 or int(row_plan.shape[1]) != 7:
        return None
    if row_plan.dtype != torch.int32 or not row_plan.is_cuda:
        return None
    if score_query_states.shape[1] != q_indexer_dim:
        return None
    if chunk_representatives.shape[3] != q_indexer_dim:
        return None
    if chunk_representatives.dtype not in (torch.float32, torch.bfloat16):
        return None
    if (
        total_rows < 0
        or int(score_query_states.shape[0]) != total_rows
        or chunk_size <= 0
        or max_q_len < 0
    ):
        return None

    if max_prior_chunks is None:
        if int(row_plan.shape[0]) == 0:
            max_prior_chunks = 0
        else:
            max_prior_chunks = int(row_plan[:, 5].max().item())
    if max_prior_chunks < 0:
        return None
    if total_rows == 0 or max_prior_chunks == 0:
        return torch.empty(
            total_rows,
            max_prior_chunks,
            device=score_query_states.device,
            dtype=torch.float32,
        )

    # Very small K dimensions are better left on the scalar row kernel for now;
    # this also avoids dot-shape limitations on older Triton builds.
    if q_indexer_dim < 16:
        return None
    # Pure decode has no same-sequence query-row reuse. Keep that on the row
    # kernel so the larger prefill tile does not compute many masked rows.
    if max_q_len <= 1:
        return None

    logits = torch.empty(
        total_rows,
        max_prior_chunks,
        device=score_query_states.device,
        dtype=torch.float32,
    )
    configured_block_rows = _triton_scoring_block_rows()
    if max_q_len < configured_block_rows:
        row_power = 1 << (max_q_len - 1).bit_length()
        block_rows = min(configured_block_rows, max(8, row_power))
    else:
        block_rows = configured_block_rows
    block_chunks = _triton_scoring_block_chunks()
    block_d = triton.next_power_of_2(q_indexer_dim)
    _dsa_batched_chunk_score_plan_kernel[
        (
            int(row_plan.shape[0]),
            triton.cdiv(max_q_len, block_rows),
            triton.cdiv(max_prior_chunks, block_chunks),
        )
    ](
        score_query_states,
        chunk_representatives,
        row_plan,
        logits,
        score_query_states.stride(0),
        score_query_states.stride(1),
        chunk_representatives.stride(0),
        chunk_representatives.stride(1),
        chunk_representatives.stride(2),
        chunk_representatives.stride(3),
        logits.stride(0),
        logits.stride(1),
        row_plan.stride(0),
        q_indexer_dim,
        chunk_size,
        max_prior_chunks,
        logit_scale / math.sqrt(q_indexer_dim),
        BLOCK_ROWS=block_rows,
        BLOCK_CHUNKS=block_chunks,
        BLOCK_D=block_d,
        INPUT_PRECISION=_triton_scoring_dot_precision(),
        num_warps=4,
        num_stages=2,
    )
    return logits


def _dsa_batched_score_logits_physical_cache_tile_plan_triton(
    *,
    score_query_states: torch.Tensor,
    physical_cache: typing.Any,
    tile_plan: torch.Tensor,
    current_chunks: torch.Tensor,
    total_rows: int,
    chunk_size: int,
    logit_scale: float,
    q_indexer_dim: int,
    max_prior_chunks: int,
    small_block_rows: int | None = None,
    large_block_rows: int | None = None,
    block_chunks: int | None = None,
    decode_block_chunks: int | None = None,
    initialize_invalid: bool = True,
    select_all_threshold: int = 0,
) -> torch.Tensor | None:
    """Score a logical chunk table directly from a physical-page sidecar."""
    if triton is None or tl is None or not score_query_states.is_cuda:
        return None
    try:
        cache_values = physical_cache._cache_values
        cache_valid = physical_cache._cache_valid
        block_table = physical_cache._block_table
        assume_valid = bool(physical_cache._assume_historical_valid)
        logical_shape = tuple(physical_cache.shape)
        physical_block_size = int(physical_cache._block_size)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        len(logical_shape) != 4
        or physical_block_size != chunk_size
        or score_query_states.dim() != 2
        or cache_values.dim() != 3
        or cache_valid.dim() != 1
        or block_table.dim() != 2
        or tile_plan.dim() != 2
        or int(tile_plan.shape[1]) != _DSA_SCORE_TILE_PLAN_COLUMNS
        or tile_plan.dtype != torch.int32
        or current_chunks.dim() != 1
        or int(current_chunks.shape[0]) != total_rows
        or int(score_query_states.shape[0]) != total_rows
        or int(score_query_states.shape[1]) != q_indexer_dim
        or int(cache_values.shape[0]) != int(cache_valid.shape[0])
        or int(cache_values.shape[1]) != int(logical_shape[2])
        or int(cache_values.shape[2]) != q_indexer_dim
        or int(block_table.shape[0]) < int(logical_shape[0])
        or int(block_table.shape[1]) < max_prior_chunks
        or cache_values.dtype != torch.bfloat16
        or cache_valid.dtype != torch.uint8
        or block_table.dtype not in (torch.int32, torch.int64)
        or q_indexer_dim < 16
        or chunk_size <= 0
        or max_prior_chunks < 0
        or select_all_threshold < 0
    ):
        return None
    device = score_query_states.device
    if any(
        tensor.device != device
        for tensor in (
            cache_values,
            cache_valid,
            block_table,
            tile_plan,
            current_chunks,
        )
    ):
        return None
    if not cache_values.is_contiguous() or not cache_valid.is_contiguous():
        return None

    if (
        small_block_rows is None
        or large_block_rows is None
        or block_chunks is None
        or decode_block_chunks is None
    ):
        (
            configured_small_rows,
            configured_large_rows,
            configured_block_chunks,
            configured_decode_block_chunks,
        ) = dsa_score_tile_plan_config()
        if small_block_rows is None:
            small_block_rows = configured_small_rows
        if large_block_rows is None:
            large_block_rows = configured_large_rows
        if block_chunks is None:
            block_chunks = configured_block_chunks
        if decode_block_chunks is None:
            decode_block_chunks = configured_decode_block_chunks
    if small_block_rows <= 1 or large_block_rows < small_block_rows:
        return None
    if block_chunks <= 0 or decode_block_chunks <= 0:
        return None
    if total_rows == 0 or max_prior_chunks == 0:
        return torch.empty(
            total_rows,
            max_prior_chunks,
            device=device,
            dtype=torch.float32,
        )

    logits = torch.empty(
        total_rows,
        max_prior_chunks,
        device=device,
        dtype=torch.float32,
    )
    dsa_cudagraph_keepalive(logits)
    if initialize_invalid:
        logits.fill_(torch.finfo(logits.dtype).min)
    if int(tile_plan.shape[0]) == 0:
        return logits

    block_d = triton.next_power_of_2(q_indexer_dim)
    _dsa_batched_physical_cache_score_tile_plan_kernel[
        (int(tile_plan.shape[0]),)
    ](
        score_query_states,
        cache_values,
        cache_valid,
        block_table,
        tile_plan,
        logits,
        current_chunks,
        score_query_states.stride(0),
        score_query_states.stride(1),
        cache_values.stride(0),
        cache_values.stride(1),
        cache_values.stride(2),
        block_table.stride(0),
        block_table.stride(1),
        logits.stride(0),
        logits.stride(1),
        tile_plan.stride(0),
        q_indexer_dim,
        max_prior_chunks,
        int(cache_values.shape[0]),
        logit_scale / math.sqrt(q_indexer_dim),
        select_all_threshold,
        SMALL_BLOCK_ROWS=small_block_rows,
        LARGE_BLOCK_ROWS=large_block_rows,
        BLOCK_CHUNKS=block_chunks,
        DECODE_BLOCK_CHUNKS=decode_block_chunks,
        BLOCK_D=block_d,
        INPUT_PRECISION=_triton_scoring_dot_precision(),
        ASSUME_VALID=assume_valid,
        num_warps=4,
        num_stages=2,
    )
    return logits


def dsa_batched_score_logits_tile_plan_triton(
    *,
    score_query_states: torch.Tensor,
    chunk_representatives: typing.Any,
    tile_plan: torch.Tensor,
    current_chunks: torch.Tensor,
    total_rows: int,
    chunk_size: int,
    logit_scale: float,
    q_indexer_dim: int,
    max_prior_chunks: int,
    small_block_rows: int | None = None,
    large_block_rows: int | None = None,
    block_chunks: int | None = None,
    decode_block_chunks: int | None = None,
    initialize_invalid: bool = True,
    select_all_threshold: int = 0,
) -> torch.Tensor | None:
    """Compute DSA chunk logits from an explicit compact tile plan."""
    if getattr(
        chunk_representatives,
        "_is_physical_page_rep_cache",
        False,
    ):
        return _dsa_batched_score_logits_physical_cache_tile_plan_triton(
            score_query_states=score_query_states,
            physical_cache=chunk_representatives,
            tile_plan=tile_plan,
            current_chunks=current_chunks,
            total_rows=total_rows,
            chunk_size=chunk_size,
            logit_scale=logit_scale,
            q_indexer_dim=q_indexer_dim,
            max_prior_chunks=max_prior_chunks,
            small_block_rows=small_block_rows,
            large_block_rows=large_block_rows,
            block_chunks=block_chunks,
            decode_block_chunks=decode_block_chunks,
            initialize_invalid=initialize_invalid,
            select_all_threshold=select_all_threshold,
        )
    if triton is None or tl is None:
        return None
    if not score_query_states.is_cuda or not chunk_representatives.is_cuda:
        return None
    if score_query_states.dim() != 2 or chunk_representatives.dim() != 4:
        return None
    if tile_plan.dim() != 2 or int(tile_plan.shape[1]) != _DSA_SCORE_TILE_PLAN_COLUMNS:
        return None
    if tile_plan.dtype != torch.int32 or not tile_plan.is_cuda:
        return None
    if (
        current_chunks.dim() != 1
        or int(current_chunks.shape[0]) != total_rows
        or current_chunks.device != score_query_states.device
    ):
        return None
    if score_query_states.shape[1] != q_indexer_dim:
        return None
    if chunk_representatives.shape[3] != q_indexer_dim:
        return None
    if chunk_representatives.dtype not in (torch.float32, torch.bfloat16):
        return None
    if (
        total_rows < 0
        or int(score_query_states.shape[0]) != total_rows
        or chunk_size <= 0
        or q_indexer_dim <= 0
        or max_prior_chunks < 0
    ):
        return None
    if q_indexer_dim < 16:
        return None

    if (
        small_block_rows is None
        or large_block_rows is None
        or block_chunks is None
        or decode_block_chunks is None
    ):
        (
            configured_small_rows,
            configured_large_rows,
            configured_block_chunks,
            configured_decode_block_chunks,
        ) = dsa_score_tile_plan_config()
        if small_block_rows is None:
            small_block_rows = configured_small_rows
        if large_block_rows is None:
            large_block_rows = configured_large_rows
        if block_chunks is None:
            block_chunks = configured_block_chunks
        if decode_block_chunks is None:
            decode_block_chunks = configured_decode_block_chunks
    if small_block_rows <= 1 or large_block_rows < small_block_rows:
        return None
    if block_chunks <= 0:
        return None
    if decode_block_chunks <= 0:
        return None

    if total_rows == 0 or max_prior_chunks == 0:
        return torch.empty(
            total_rows,
            max_prior_chunks,
            device=score_query_states.device,
            dtype=torch.float32,
        )

    logits = torch.empty(
        total_rows,
        max_prior_chunks,
        device=score_query_states.device,
        dtype=torch.float32,
    )
    if initialize_invalid:
        logits.fill_(torch.finfo(logits.dtype).min)
    if int(tile_plan.shape[0]) == 0:
        return logits

    block_d = triton.next_power_of_2(q_indexer_dim)
    _dsa_batched_chunk_score_tile_plan_kernel[(int(tile_plan.shape[0]),)](
        score_query_states,
        chunk_representatives,
        tile_plan,
        logits,
        current_chunks,
        score_query_states.stride(0),
        score_query_states.stride(1),
        chunk_representatives.stride(0),
        chunk_representatives.stride(1),
        chunk_representatives.stride(2),
        chunk_representatives.stride(3),
        logits.stride(0),
        logits.stride(1),
        tile_plan.stride(0),
        q_indexer_dim,
        chunk_size,
        max_prior_chunks,
        logit_scale / math.sqrt(q_indexer_dim),
        SMALL_BLOCK_ROWS=small_block_rows,
        LARGE_BLOCK_ROWS=large_block_rows,
        BLOCK_CHUNKS=block_chunks,
        DECODE_BLOCK_CHUNKS=decode_block_chunks,
        BLOCK_D=block_d,
        INPUT_PRECISION=_triton_scoring_dot_precision(),
        num_warps=4,
        num_stages=2,
    )
    return logits


def dsa_build_score_tile_plan_triton(
    *,
    row_plan_with_tiles: torch.Tensor,
    total_tiles: int,
    max_tiles_per_row_plan: int,
    small_block_rows: int | None = None,
    large_block_rows: int | None = None,
    block_chunks: int | None = None,
    decode_block_chunks: int | None = None,
) -> torch.Tensor | None:
    """Build the expanded score tile plan on GPU from compact row metadata.

    ``row_plan_with_tiles`` extends the 7-column row-plan schema with
    ``tile_offset`` and ``tile_count``. Those two scalar fields are cheap for
    the CPU to compute while preserving provider-local ownership of the plan.
    """
    if triton is None or tl is None:
        return None
    if (
        row_plan_with_tiles.dim() != 2
        or int(row_plan_with_tiles.shape[1]) != _DSA_ROW_PLAN_WITH_TILE_COLUMNS
        or row_plan_with_tiles.dtype != torch.int32
        or not row_plan_with_tiles.is_cuda
    ):
        return None
    if total_tiles < 0 or max_tiles_per_row_plan < 0:
        return None

    if (
        small_block_rows is None
        or large_block_rows is None
        or block_chunks is None
        or decode_block_chunks is None
    ):
        (
            configured_small_rows,
            configured_large_rows,
            configured_block_chunks,
            configured_decode_block_chunks,
        ) = dsa_score_tile_plan_config()
        if small_block_rows is None:
            small_block_rows = configured_small_rows
        if large_block_rows is None:
            large_block_rows = configured_large_rows
        if block_chunks is None:
            block_chunks = configured_block_chunks
        if decode_block_chunks is None:
            decode_block_chunks = configured_decode_block_chunks
    if small_block_rows <= 1 or large_block_rows < small_block_rows:
        return None
    if block_chunks <= 0 or decode_block_chunks <= 0:
        return None

    tile_plan = torch.empty(
        total_tiles,
        _DSA_SCORE_TILE_PLAN_COLUMNS,
        device=row_plan_with_tiles.device,
        dtype=torch.int32,
    )
    if total_tiles == 0 or int(row_plan_with_tiles.shape[0]) == 0:
        return tile_plan
    if max_tiles_per_row_plan <= 0:
        return None

    max_tiles_power = triton.next_power_of_2(max_tiles_per_row_plan)
    _dsa_build_score_tile_plan_kernel[(int(row_plan_with_tiles.shape[0]),)](
        row_plan_with_tiles,
        tile_plan,
        row_plan_with_tiles.stride(0),
        tile_plan.stride(0),
        SMALL_BLOCK_ROWS=small_block_rows,
        LARGE_BLOCK_ROWS=large_block_rows,
        BLOCK_CHUNKS=block_chunks,
        DECODE_BLOCK_CHUNKS=decode_block_chunks,
        MAX_TILES_PER_ROW_PLAN=max_tiles_power,
        num_warps=4,
    )
    return tile_plan


def dsa_build_fixed_decode_score_tile_plan_triton(
    *,
    total_rows: int,
    max_prior_chunks: int,
    representative_group_idx: int,
    device: torch.device,
    decode_block_chunks: int | None = None,
) -> torch.Tensor | None:
    """Build a reusable padded plan for all-single-token decode.

    Actual sequence lengths remain device-side in ``current_chunks``. Scoring
    masks every padded chunk, so this plan can be reused as contexts grow.
    """
    if triton is None or tl is None or device.type != "cuda":
        return None
    if total_rows < 0 or max_prior_chunks < 0 or representative_group_idx < 0:
        return None
    if decode_block_chunks is None:
        decode_block_chunks = dsa_score_tile_plan_config()[3]
    if decode_block_chunks <= 0:
        return None

    tiles_per_row = (
        max_prior_chunks + decode_block_chunks - 1
    ) // decode_block_chunks
    total_tiles = total_rows * tiles_per_row
    tile_plan = torch.empty(
        total_tiles,
        _DSA_SCORE_TILE_PLAN_COLUMNS,
        device=device,
        dtype=torch.int32,
    )
    if total_tiles == 0:
        return tile_plan
    _dsa_build_fixed_decode_score_tile_plan_kernel[(total_tiles,)](
        tile_plan,
        tile_plan.stride(0),
        TILES_PER_ROW=tiles_per_row,
        MAX_PRIOR_CHUNKS=max_prior_chunks,
        GROUP_ID=representative_group_idx,
        DECODE_BLOCK_CHUNKS=decode_block_chunks,
        MODE=_DSA_SCORE_TILE_MODE_DECODE,
        num_warps=1,
    )
    return tile_plan


def dsa_build_score_metadata_triton(
    *,
    query_start_loc: torch.Tensor,
    row_query_start_loc: torch.Tensor | None = None,
    seq_lens: torch.Tensor,
    num_actual_tokens: int,
    active_seq_count: int,
    num_sparse_plans: int,
    total_rows: int,
    chunk_size: int,
    representative_group_idx: int,
    dense_decode_threshold: int,
    dense_prefill_threshold: int,
    chunk_top_k: int,
    max_q_len: int,
    representatives_use_original_seq_ids: bool = False,
    small_block_rows: int | None = None,
    large_block_rows: int | None = None,
    block_chunks: int | None = None,
    decode_block_chunks: int | None = None,
) -> (
    tuple[
        torch.Tensor,
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]
    | None
):
    """Build compact row/tile metadata directly from GPU batch metadata."""
    if triton is None or tl is None:
        return None
    if row_query_start_loc is None:
        row_query_start_loc = query_start_loc
    if (
        query_start_loc.dim() != 1
        or row_query_start_loc.dim() != 1
        or seq_lens.dim() != 1
        or not query_start_loc.is_cuda
        or not row_query_start_loc.is_cuda
        or not seq_lens.is_cuda
        or query_start_loc.device != seq_lens.device
        or row_query_start_loc.device != seq_lens.device
    ):
        return None
    if (
        active_seq_count < 0
        or num_sparse_plans < 0
        or total_rows < 0
        or num_actual_tokens < 0
        or chunk_size <= 0
        or chunk_top_k < 0
        or max_q_len < 0
    ):
        return None
    if int(query_start_loc.shape[0]) < active_seq_count + 1:
        return None
    if int(row_query_start_loc.shape[0]) < active_seq_count + 1:
        return None
    if int(seq_lens.shape[0]) < active_seq_count:
        return None

    if (
        small_block_rows is None
        or large_block_rows is None
        or block_chunks is None
        or decode_block_chunks is None
    ):
        (
            configured_small_rows,
            configured_large_rows,
            configured_block_chunks,
            configured_decode_block_chunks,
        ) = dsa_score_tile_plan_config()
        if small_block_rows is None:
            small_block_rows = configured_small_rows
        if large_block_rows is None:
            large_block_rows = configured_large_rows
        if block_chunks is None:
            block_chunks = configured_block_chunks
        if decode_block_chunks is None:
            decode_block_chunks = configured_decode_block_chunks
    if small_block_rows <= 1 or large_block_rows < small_block_rows:
        return None
    if block_chunks <= 0 or decode_block_chunks <= 0:
        return None

    device = query_start_loc.device
    row_plan = torch.empty(
        num_sparse_plans,
        _DSA_ROW_PLAN_WITH_TILE_COLUMNS,
        device=device,
        dtype=torch.int32,
    )
    score_row_seq_ids = torch.empty(total_rows, device=device, dtype=torch.int32)
    row_seq_ids = torch.empty(total_rows, device=device, dtype=torch.int32)
    row_group_ids = torch.empty(total_rows, device=device, dtype=torch.int32)
    row_num_prior_chunks = torch.empty(total_rows, device=device, dtype=torch.int32)
    row_current_chunks = torch.empty(total_rows, device=device, dtype=torch.int32)
    row_tail_lens = torch.empty(total_rows, device=device, dtype=torch.int32)
    if total_rows == 0 or active_seq_count == 0:
        return row_plan, (
            score_row_seq_ids,
            row_seq_ids,
            row_group_ids,
            row_num_prior_chunks,
            row_current_chunks,
            row_tail_lens,
        )
    if max_q_len <= 0:
        return None

    block_rows = 128
    max_active_seqs = triton.next_power_of_2(max(active_seq_count, 1))
    _dsa_build_score_metadata_kernel[
        (active_seq_count, triton.cdiv(max_q_len, block_rows))
    ](
        query_start_loc,
        row_query_start_loc,
        seq_lens,
        row_plan,
        score_row_seq_ids,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
        row_current_chunks,
        row_tail_lens,
        query_start_loc.stride(0),
        row_query_start_loc.stride(0),
        seq_lens.stride(0),
        row_plan.stride(0),
        num_actual_tokens,
        representative_group_idx,
        chunk_size,
        dense_decode_threshold,
        dense_prefill_threshold,
        active_seq_count,
        SMALL_BLOCK_ROWS=small_block_rows,
        LARGE_BLOCK_ROWS=large_block_rows,
        BLOCK_CHUNKS=block_chunks,
        DECODE_BLOCK_CHUNKS=decode_block_chunks,
        CHUNK_TOP_K=chunk_top_k,
        REPRESENTATIVES_USE_ORIGINAL_SEQ_IDS=representatives_use_original_seq_ids,
        MAX_ACTIVE_SEQS=max_active_seqs,
        BLOCK_ROWS=block_rows,
        num_warps=4,
    )
    return row_plan, (
        score_row_seq_ids,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
        row_current_chunks,
        row_tail_lens,
    )


def dsa_batched_row_metadata_triton(
    *,
    row_plan: torch.Tensor,
    total_rows: int,
    chunk_size: int,
    max_q_len: int,
) -> (
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]
    | None
):
    """Expand compact per-sequence DSA scoring metadata into row tensors.

    ``row_plan`` is shaped ``[num_sequences, 7]`` with columns:
    row_start, q_len, score_seq_id, block_table_seq_id, row_group_id,
    num_prior_chunks, and query_position_start.
    """
    if triton is None or tl is None:
        return None
    if (
        row_plan.dim() != 2
        or int(row_plan.shape[1]) < _DSA_ROW_PLAN_COLUMNS
        or not row_plan.is_cuda
    ):
        return None
    if row_plan.dtype != torch.int32:
        return None
    if total_rows < 0 or chunk_size <= 0 or max_q_len < 0:
        return None

    device = row_plan.device
    score_row_seq_ids = torch.empty(total_rows, device=device, dtype=torch.int32)
    row_seq_ids = torch.empty(total_rows, device=device, dtype=torch.int32)
    row_group_ids = torch.empty(total_rows, device=device, dtype=torch.int32)
    row_num_prior_chunks = torch.empty(total_rows, device=device, dtype=torch.int32)
    row_current_chunks = torch.empty(total_rows, device=device, dtype=torch.int32)
    row_tail_lens = torch.empty(total_rows, device=device, dtype=torch.int32)
    if total_rows == 0 or int(row_plan.shape[0]) == 0:
        return (
            score_row_seq_ids,
            row_seq_ids,
            row_group_ids,
            row_num_prior_chunks,
            row_current_chunks,
            row_tail_lens,
        )

    block_rows = 128
    _dsa_batched_row_metadata_kernel[
        (int(row_plan.shape[0]), triton.cdiv(max_q_len, block_rows))
    ](
        row_plan,
        score_row_seq_ids,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
        row_current_chunks,
        row_tail_lens,
        chunk_size,
        row_plan.stride(0),
        BLOCK_ROWS=block_rows,
        num_warps=4,
    )
    return (
        score_row_seq_ids,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
        row_current_chunks,
        row_tail_lens,
    )


def _dsa_select_all_prior_chunks(
    *,
    current_chunks: torch.Tensor,
    row_num_prior_chunks: torch.Tensor,
    max_prior_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor, None] | None:
    if current_chunks.dim() != 1 or row_num_prior_chunks.dim() != 1:
        return None
    if current_chunks.shape[0] != row_num_prior_chunks.shape[0]:
        return None
    if current_chunks.device != row_num_prior_chunks.device:
        return None
    if not current_chunks.is_cuda:
        return None
    num_rows = int(current_chunks.shape[0])
    device = current_chunks.device
    if max_prior_chunks < 0:
        return None
    current_chunks_i32 = current_chunks.to(device=device, dtype=torch.int32)
    row_num_prior_chunks_i32 = row_num_prior_chunks.to(
        device=device,
        dtype=torch.int32,
    )
    selectable_counts = torch.empty_like(
        current_chunks_i32,
        memory_format=torch.contiguous_format,
    )
    torch.minimum(
        current_chunks_i32,
        row_num_prior_chunks_i32,
        out=selectable_counts,
    )
    selectable_counts.clamp_(min=0, max=max_prior_chunks)
    if max_prior_chunks == 0:
        return (
            torch.empty(num_rows, 0, device=device, dtype=torch.int32),
            selectable_counts,
            None,
        )
    chunk_indices = torch.arange(
        max_prior_chunks,
        device=device,
        dtype=torch.int32,
    ).expand(num_rows, max_prior_chunks)
    return chunk_indices.contiguous(), selectable_counts, None


def _dsa_validate_dynamic_top_k(
    *,
    row_top_k: torch.Tensor | None,
    top_k_segments: typing.Sequence[tuple[int, int, int]] | None,
    num_rows: int,
    device: torch.device,
) -> bool:
    """Validate the no-sync metadata contract for mixed per-row top-k.

    ``top_k_segments`` is CPU metadata and must contain ordered, non-overlapping
    score-row ranges. Gaps represent dense/non-scored rows. ``row_top_k`` is the
    equivalent device-side representation used to cap selected counts without
    copying GPU state back to the CPU. Their values are intentionally not
    compared here because doing so would introduce a device synchronization in
    the hot path.
    """
    if row_top_k is None and top_k_segments is None:
        return True
    if row_top_k is None or top_k_segments is None:
        return False
    if (
        row_top_k.dim() != 1
        or int(row_top_k.shape[0]) != num_rows
        or row_top_k.device != device
        or row_top_k.dtype not in (torch.int32, torch.int64)
    ):
        return False

    next_row = 0
    for segment in top_k_segments:
        if len(segment) != 3:
            return False
        row_start, row_end, segment_top_k = map(int, segment)
        if (
            row_start < next_row
            or row_start < 0
            or row_end <= row_start
            or row_end > num_rows
            or segment_top_k <= 0
        ):
            return False
        next_row = row_end
    return True


def _dsa_dynamic_top_k_from_logits(
    *,
    logits: torch.Tensor,
    selectable_counts: torch.Tensor,
    row_top_k: torch.Tensor,
    top_k_segments: typing.Sequence[tuple[int, int, int]],
    max_top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select an exact K for each contiguous row segment.

    The CUDA prefill top-k operation has a uniform-K output layout and does not
    accept an output row stride. Consequently, each constant-K segment writes
    to a contiguous temporary before its valid prefix is copied into the
    rectangular ``[rows, max_top_k]`` result. This avoids relying on a prefix
    of an unordered max-K result for rows whose requested K is smaller.
    """
    num_rows = int(logits.shape[0])
    device = logits.device
    top_chunk_indices = torch.full(
        (num_rows, max_top_k),
        -1,
        device=device,
        dtype=torch.int32,
    )
    selected_counts = (
        torch.minimum(
            selectable_counts,
            row_top_k.to(device=device, dtype=torch.int32),
        )
        .clamp(min=0, max=max_top_k)
        .contiguous()
    )

    use_cuda_top_k = _has_top_k_per_row_prefill()
    for row_start, row_end, requested_top_k in top_k_segments:
        segment_top_k = min(int(requested_top_k), max_top_k)
        if segment_top_k == 0:
            continue
        segment_rows = int(row_end) - int(row_start)
        segment_logits = logits[int(row_start) : int(row_end)].contiguous()
        if use_cuda_top_k and segment_top_k <= _DSA_CUDA_PREFILL_TOP_K_MAX:
            # The custom op advances output rows by ``segment_top_k`` rather
            # than a caller-provided stride, so a slice of the max-K output is
            # not a valid destination when segment_top_k < max_top_k.
            segment_indices = torch.empty(
                segment_rows,
                segment_top_k,
                device=device,
                dtype=torch.int32,
            )
            segment_row_starts = torch.zeros(
                segment_rows,
                device=device,
                dtype=torch.int32,
            )
            segment_row_ends = selectable_counts[
                int(row_start) : int(row_end)
            ].contiguous()
            _run_top_k_per_row_prefill(
                segment_logits,
                segment_row_starts,
                segment_row_ends,
                segment_indices,
                segment_rows,
                segment_top_k,
            )
            top_chunk_indices[int(row_start) : int(row_end), :segment_top_k].copy_(
                segment_indices
            )
            continue

        # The compiled prefill selector cannot launch above K=8,128. Bound the
        # temporary values/int64-indices allocation by processing row blocks,
        # and copy-cast directly into the persistent int32 result. ``sorted``
        # keeps real candidates before masked filler for rows shorter than K.
        for block_start in range(0, segment_rows, _DSA_TORCH_TOP_K_BLOCK_ROWS):
            block_end = min(
                block_start + _DSA_TORCH_TOP_K_BLOCK_ROWS,
                segment_rows,
            )
            block_indices = (
                segment_logits[block_start:block_end]
                .topk(
                    k=segment_top_k,
                    dim=-1,
                    sorted=True,
                )
                .indices
            )
            top_chunk_indices[
                int(row_start) + block_start : int(row_start) + block_end,
                :segment_top_k,
            ].copy_(block_indices)
    return top_chunk_indices, selected_counts


def dsa_batched_score_topk_triton(
    *,
    score_query_states: torch.Tensor,
    chunk_representatives: torch.Tensor,
    current_chunks: torch.Tensor,
    row_seq_ids: torch.Tensor,
    row_group_ids: torch.Tensor,
    row_num_prior_chunks: torch.Tensor,
    chunk_top_k: int,
    logit_scale: float,
    q_indexer_dim: int,
    max_prior_chunks: int | None = None,
    return_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Compute batched DSA chunk top-k with Triton scoring and CUDA top-k."""
    if chunk_top_k <= 0:
        return None
    if max_prior_chunks is None:
        if row_num_prior_chunks.dim() != 1:
            return None
        max_prior_chunks = (
            0
            if int(row_num_prior_chunks.shape[0]) == 0
            else int(row_num_prior_chunks.max().item())
        )
    if not return_logits and chunk_top_k >= max_prior_chunks:
        return _dsa_select_all_prior_chunks(
            current_chunks=current_chunks,
            row_num_prior_chunks=row_num_prior_chunks,
            max_prior_chunks=max_prior_chunks,
        )

    logits = dsa_batched_score_logits_triton(
        score_query_states=score_query_states,
        chunk_representatives=chunk_representatives,
        current_chunks=current_chunks,
        row_seq_ids=row_seq_ids,
        row_group_ids=row_group_ids,
        row_num_prior_chunks=row_num_prior_chunks,
        logit_scale=logit_scale,
        q_indexer_dim=q_indexer_dim,
        max_prior_chunks=max_prior_chunks,
    )
    if logits is None:
        return None

    num_rows = int(logits.shape[0])
    max_prior_chunks = int(logits.shape[1])
    if num_rows == 0 or max_prior_chunks == 0:
        empty_indices = torch.empty(
            num_rows,
            0,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        empty_counts = torch.empty(
            num_rows,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        return empty_indices, empty_counts, logits if return_logits else None

    top_k = min(chunk_top_k, max_prior_chunks)
    current_chunks_i32 = current_chunks.to(
        device=score_query_states.device,
        dtype=torch.int32,
    )
    row_num_prior_chunks_i32 = row_num_prior_chunks.to(
        device=score_query_states.device,
        dtype=torch.int32,
    )
    selectable_counts = torch.empty_like(
        current_chunks_i32,
        memory_format=torch.contiguous_format,
    )
    torch.minimum(
        current_chunks_i32,
        row_num_prior_chunks_i32,
        out=selectable_counts,
    )
    selectable_counts.clamp_(min=0, max=max_prior_chunks)
    dsa_cudagraph_keepalive(
        current_chunks_i32,
        row_num_prior_chunks_i32,
        selectable_counts,
        logits,
    )

    if _has_top_k_per_row_prefill():
        top_chunk_indices_i32 = torch.empty(
            num_rows,
            top_k,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        row_starts = torch.zeros(
            num_rows,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        dsa_cudagraph_keepalive(
            top_chunk_indices_i32,
            row_starts,
        )
        torch.ops._C.top_k_per_row_prefill(
            logits,
            row_starts,
            selectable_counts,
            top_chunk_indices_i32,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            top_k,
        )
        top_chunk_indices = top_chunk_indices_i32
    else:
        top_chunk_indices = logits.topk(k=top_k, dim=-1, sorted=False).indices
        selected_counts = selectable_counts.clamp(max=top_k)
        top_chunk_valid = (
            torch.arange(
                top_k,
                device=score_query_states.device,
                dtype=torch.int32,
            )[None, :]
            < selected_counts[:, None]
        )
        top_chunk_indices = top_chunk_indices.masked_fill(~top_chunk_valid, 0)
    return top_chunk_indices, selectable_counts, logits if return_logits else None


def dsa_batched_score_topk_plan_triton(
    *,
    score_query_states: torch.Tensor,
    chunk_representatives: torch.Tensor,
    row_plan: torch.Tensor,
    current_chunks: torch.Tensor,
    row_num_prior_chunks: torch.Tensor,
    total_rows: int,
    chunk_size: int,
    max_q_len: int,
    chunk_top_k: int,
    logit_scale: float,
    q_indexer_dim: int,
    max_prior_chunks: int | None = None,
    return_logits: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Compute batched DSA top-k using row-plan tiled scoring."""
    if chunk_top_k <= 0:
        return None
    if max_prior_chunks is None:
        if row_num_prior_chunks.dim() != 1:
            return None
        max_prior_chunks = (
            0
            if int(row_num_prior_chunks.shape[0]) == 0
            else int(row_num_prior_chunks.max().item())
        )
    if not return_logits and chunk_top_k >= max_prior_chunks:
        return _dsa_select_all_prior_chunks(
            current_chunks=current_chunks,
            row_num_prior_chunks=row_num_prior_chunks,
            max_prior_chunks=max_prior_chunks,
        )

    logits = dsa_batched_score_logits_plan_triton(
        score_query_states=score_query_states,
        chunk_representatives=chunk_representatives,
        row_plan=row_plan,
        total_rows=total_rows,
        chunk_size=chunk_size,
        max_q_len=max_q_len,
        logit_scale=logit_scale,
        q_indexer_dim=q_indexer_dim,
        max_prior_chunks=max_prior_chunks,
    )
    if logits is None:
        return None

    num_rows = int(logits.shape[0])
    max_prior_chunks = int(logits.shape[1])
    if num_rows == 0 or max_prior_chunks == 0:
        empty_indices = torch.empty(
            num_rows,
            0,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        empty_counts = torch.empty(
            num_rows,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        return empty_indices, empty_counts, logits if return_logits else None

    top_k = min(chunk_top_k, max_prior_chunks)
    current_chunks_i32 = current_chunks.to(
        device=score_query_states.device,
        dtype=torch.int32,
    )
    row_num_prior_chunks_i32 = row_num_prior_chunks.to(
        device=score_query_states.device,
        dtype=torch.int32,
    )
    selectable_counts = torch.empty_like(
        current_chunks_i32,
        memory_format=torch.contiguous_format,
    )
    torch.minimum(
        current_chunks_i32,
        row_num_prior_chunks_i32,
        out=selectable_counts,
    )
    selectable_counts.clamp_(min=0, max=max_prior_chunks)
    dsa_cudagraph_keepalive(
        current_chunks_i32,
        row_num_prior_chunks_i32,
        selectable_counts,
        logits,
    )

    if _has_top_k_per_row_prefill():
        top_chunk_indices_i32 = torch.empty(
            num_rows,
            top_k,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        row_starts = torch.zeros(
            num_rows,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        dsa_cudagraph_keepalive(
            top_chunk_indices_i32,
            row_starts,
        )
        torch.ops._C.top_k_per_row_prefill(
            logits,
            row_starts,
            selectable_counts,
            top_chunk_indices_i32,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            top_k,
        )
        top_chunk_indices = top_chunk_indices_i32
    else:
        top_chunk_indices = logits.topk(k=top_k, dim=-1, sorted=False).indices
        selected_counts = selectable_counts.clamp(max=top_k)
        top_chunk_valid = (
            torch.arange(
                top_k,
                device=score_query_states.device,
                dtype=torch.int32,
            )[None, :]
            < selected_counts[:, None]
        )
        top_chunk_indices = top_chunk_indices.masked_fill(~top_chunk_valid, 0)
        dsa_cudagraph_keepalive(
            top_chunk_indices,
            selected_counts,
            top_chunk_valid,
        )
    return top_chunk_indices, selectable_counts, logits if return_logits else None


def dsa_batched_score_topk_tile_plan_triton(
    *,
    score_query_states: torch.Tensor,
    chunk_representatives: typing.Any,
    tile_plan: torch.Tensor,
    current_chunks: torch.Tensor,
    row_num_prior_chunks: torch.Tensor,
    total_rows: int,
    chunk_size: int,
    chunk_top_k: int,
    logit_scale: float,
    q_indexer_dim: int,
    max_prior_chunks: int,
    small_block_rows: int | None = None,
    large_block_rows: int | None = None,
    block_chunks: int | None = None,
    decode_block_chunks: int | None = None,
    return_logits: bool = False,
    row_top_k: torch.Tensor | None = None,
    top_k_segments: typing.Sequence[tuple[int, int, int]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None:
    """Compute batched DSA top-k using an explicit compact tile plan.

    Supplying both ``row_top_k`` and ``top_k_segments`` enables exact mixed-K
    selection. The device tensor caps returned counts while the CPU segments
    identify ordered, non-overlapping sparse ranges as
    ``(row_start, row_end, K)`` records and drive one uniform-K selection per
    contiguous segment. Rows in gaps must have zero ``row_top_k``. Omit both
    arguments for the original fixed-K behavior.
    """
    if chunk_top_k <= 0:
        return None
    if not _dsa_validate_dynamic_top_k(
        row_top_k=row_top_k,
        top_k_segments=top_k_segments,
        num_rows=total_rows,
        device=score_query_states.device,
    ):
        return None
    dynamic_top_k = row_top_k is not None
    has_cuda_prefill_top_k = _has_top_k_per_row_prefill()
    dynamic_top_k_uses_torch = dynamic_top_k and (
        not has_cuda_prefill_top_k
        or any(
            min(int(segment[2]), chunk_top_k, max_prior_chunks)
            > _DSA_CUDA_PREFILL_TOP_K_MAX
            for segment in (top_k_segments or ())
        )
    )
    if not dynamic_top_k and not return_logits and chunk_top_k >= max_prior_chunks:
        return _dsa_select_all_prior_chunks(
            current_chunks=current_chunks,
            row_num_prior_chunks=row_num_prior_chunks,
            max_prior_chunks=max_prior_chunks,
        )

    logits = dsa_batched_score_logits_tile_plan_triton(
        score_query_states=score_query_states,
        chunk_representatives=chunk_representatives,
        tile_plan=tile_plan,
        current_chunks=current_chunks,
        total_rows=total_rows,
        chunk_size=chunk_size,
        logit_scale=logit_scale,
        q_indexer_dim=q_indexer_dim,
        max_prior_chunks=max_prior_chunks,
        small_block_rows=small_block_rows,
        large_block_rows=large_block_rows,
        block_chunks=block_chunks,
        decode_block_chunks=decode_block_chunks,
        initialize_invalid=(
            return_logits or not has_cuda_prefill_top_k or dynamic_top_k_uses_torch
        ),
        select_all_threshold=(
            chunk_top_k
            if (
                not dynamic_top_k
                and not return_logits
                and has_cuda_prefill_top_k
            )
            else 0
        ),
    )
    if logits is None:
        return None

    num_rows = int(logits.shape[0])
    max_prior_chunks = int(logits.shape[1])
    if num_rows == 0 or max_prior_chunks == 0:
        empty_indices = torch.empty(
            num_rows,
            0,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        empty_counts = torch.empty(
            num_rows,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        return empty_indices, empty_counts, logits if return_logits else None

    top_k = min(chunk_top_k, max_prior_chunks)
    current_chunks_i32 = current_chunks.to(
        device=score_query_states.device,
        dtype=torch.int32,
    )
    row_num_prior_chunks_i32 = row_num_prior_chunks.to(
        device=score_query_states.device,
        dtype=torch.int32,
    )
    selectable_counts = torch.empty_like(
        current_chunks_i32,
        memory_format=torch.contiguous_format,
    )
    torch.minimum(
        current_chunks_i32,
        row_num_prior_chunks_i32,
        out=selectable_counts,
    )
    selectable_counts.clamp_(min=0, max=max_prior_chunks)
    dsa_cudagraph_keepalive(
        current_chunks_i32,
        row_num_prior_chunks_i32,
        selectable_counts,
        logits,
    )

    if dynamic_top_k:
        assert row_top_k is not None
        assert top_k_segments is not None
        top_chunk_indices, selected_counts = _dsa_dynamic_top_k_from_logits(
            logits=logits,
            selectable_counts=selectable_counts,
            row_top_k=row_top_k,
            top_k_segments=top_k_segments,
            max_top_k=top_k,
        )
        dsa_cudagraph_keepalive(top_chunk_indices, selected_counts)
        return (
            top_chunk_indices,
            selected_counts,
            logits if return_logits else None,
        )

    if _has_top_k_per_row_prefill():
        top_chunk_indices_i32 = torch.empty(
            num_rows,
            top_k,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        row_starts = torch.zeros(
            num_rows,
            device=score_query_states.device,
            dtype=torch.int32,
        )
        dsa_cudagraph_keepalive(
            top_chunk_indices_i32,
            row_starts,
        )
        torch.ops._C.top_k_per_row_prefill(
            logits,
            row_starts,
            selectable_counts,
            top_chunk_indices_i32,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            top_k,
        )
        top_chunk_indices = top_chunk_indices_i32
    else:
        top_chunk_indices = logits.topk(k=top_k, dim=-1, sorted=False).indices
        selected_counts = selectable_counts.clamp(max=top_k)
        top_chunk_valid = (
            torch.arange(
                top_k,
                device=score_query_states.device,
                dtype=torch.int32,
            )[None, :]
            < selected_counts[:, None]
        )
        top_chunk_indices = top_chunk_indices.masked_fill(~top_chunk_valid, 0)
        dsa_cudagraph_keepalive(
            top_chunk_indices,
            selected_counts,
            top_chunk_valid,
        )
    return top_chunk_indices, selectable_counts, logits if return_logits else None
