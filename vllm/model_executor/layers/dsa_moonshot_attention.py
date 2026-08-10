# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Experimental kernels for Nemotron-H chunked DSA attention.

These kernels are intentionally not wired into serving yet. They are used by the
moonshot benchmark harness to iterate on sparse prefill apply kernels with tight
tensor-parity checks before touching the production path.
"""

import math

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _dsa_prefill_gqa_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    top_chunks_ptr,
    top_valid_ptr,
    current_chunks_ptr,
    query_positions_ptr,
    out_ptr,
    top_chunks_count,
    softmax_scale: tl.constexpr,
    group_idx: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)

    offs_h = tl.arange(0, group_size)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    d_mask = offs_d < head_dim

    q_offsets = row * group_size * head_dim + offs_h[:, None] * head_dim + offs_d[None, :]
    q = tl.load(q_ptr + q_offsets, mask=d_mask[None, :], other=0.0)

    m_i = tl.full((group_size,), -float("inf"), tl.float32)
    l_i = tl.zeros((group_size,), tl.float32)
    acc = tl.zeros((group_size, BLOCK_D), tl.float32)

    top_i = 0
    while top_i < top_chunks_count:
        chunk_valid = tl.load(top_valid_ptr + row * top_chunks_count + top_i)
        logical_chunk = tl.load(top_chunks_ptr + row * top_chunks_count + top_i)
        physical_block = tl.load(block_table_ptr + logical_chunk)

        kv_offsets = (
            ((physical_block * chunk_size + offs_n[:, None]) * num_kv_heads + group_idx)
            * head_dim
            + offs_d[None, :]
        )
        k = tl.load(
            key_cache_ptr + kv_offsets,
            mask=(offs_n[:, None] < chunk_size) & d_mask[None, :] & chunk_valid,
            other=0.0,
        )
        v = tl.load(
            value_cache_ptr + kv_offsets,
            mask=(offs_n[:, None] < chunk_size) & d_mask[None, :] & chunk_valid,
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk = tl.where(chunk_valid, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new
        top_i += 1

    current_chunk = tl.load(current_chunks_ptr + row)
    query_position = tl.load(query_positions_ptr + row)
    current_start = current_chunk * chunk_size
    tail_len = query_position - current_start + 1
    physical_block = tl.load(block_table_ptr + current_chunk)
    tail_mask = offs_n < tail_len
    kv_offsets = (
        ((physical_block * chunk_size + offs_n[:, None]) * num_kv_heads + group_idx)
        * head_dim
        + offs_d[None, :]
    )
    k = tl.load(
        key_cache_ptr + kv_offsets,
        mask=tail_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    v = tl.load(
        value_cache_ptr + kv_offsets,
        mask=tail_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    qk = tl.dot(q, tl.trans(k)) * softmax_scale
    qk = tl.where(tail_mask[None, :], qk, -float("inf"))
    m_new = tl.maximum(m_i, tl.max(qk, axis=1))
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(qk - m_new[:, None])
    l_new = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_new
    l_i = l_new

    out = acc / l_i[:, None]
    tl.store(out_ptr + q_offsets, out, mask=d_mask[None, :])


def dsa_prefill_gqa_attention(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    top_chunk_indices: torch.Tensor,
    top_chunk_valid: torch.Tensor,
    current_chunks: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    group_idx: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run the experimental GQA-group chunked DSA prefill apply kernel.

    Args follow ``NemotronHDSASelectiveAttention``'s page-table prefill helper.
    Only NHD cache layout with ``chunk_size == page_size`` is supported.
    """

    if query_states.dim() != 3:
        raise ValueError(f"query_states must be [q, heads, dim], got {query_states.shape}")
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError(
            "key_cache/value_cache must use NHD [blocks, block, kv_heads, dim] layout"
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key/value cache shape mismatch: {key_cache.shape} vs {value_cache.shape}"
        )
    if top_chunk_indices.shape != top_chunk_valid.shape:
        raise ValueError(
            "top chunk indices/valid shape mismatch: "
            f"{top_chunk_indices.shape} vs {top_chunk_valid.shape}"
        )
    q_len, group_size, head_dim = query_states.shape
    if q_len == 0:
        return torch.empty_like(query_states)
    if top_chunk_indices.dim() != 2 or top_chunk_indices.shape[0] != q_len:
        raise ValueError(
            f"top_chunk_indices must be [q, top], got {top_chunk_indices.shape}"
        )
    if current_chunks.shape != (q_len,) or query_positions.shape != (q_len,):
        raise ValueError(
            "current_chunks and query_positions must be 1D with q_len entries"
        )
    if head_dim != key_cache.shape[-1]:
        raise ValueError(f"head_dim mismatch: query={head_dim} cache={key_cache.shape[-1]}")
    if head_dim > 256:
        raise ValueError(f"head_dim too large for prototype kernel: {head_dim}")
    chunk_size = int(key_cache.shape[1])
    if chunk_size != 16:
        raise ValueError(f"prototype kernel currently expects chunk/page size 16, got {chunk_size}")

    query_states = query_states.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    block_table = block_table.to(device=query_states.device, dtype=torch.int32).contiguous()
    top_chunk_indices = top_chunk_indices.to(
        device=query_states.device, dtype=torch.int32
    ).contiguous()
    top_chunk_valid = top_chunk_valid.to(device=query_states.device).contiguous()
    current_chunks = current_chunks.to(
        device=query_states.device, dtype=torch.int32
    ).contiguous()
    query_positions = query_positions.to(
        device=query_states.device, dtype=torch.int32
    ).contiguous()

    output = torch.empty_like(query_states)
    block_d = triton.next_power_of_2(head_dim)
    top_chunks_count = int(top_chunk_indices.shape[1])
    _dsa_prefill_gqa_kernel[(q_len,)](
        query_states,
        key_cache,
        value_cache,
        block_table,
        top_chunk_indices,
        top_chunk_valid,
        current_chunks,
        query_positions,
        output,
        top_chunks_count,
        softmax_scale=softmax_scale,
        group_idx=int(group_idx),
        num_kv_heads=int(key_cache.shape[2]),
        head_dim=int(head_dim),
        chunk_size=int(chunk_size),
        group_size=int(group_size),
        BLOCK_D=int(block_d),
        BLOCK_N=int(chunk_size),
        num_warps=4 if head_dim <= 128 else 8,
    )
    return output


@triton.jit
def _dsa_prefill_gqa_splitk_partial_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    top_chunks_ptr,
    top_valid_ptr,
    current_chunks_ptr,
    query_positions_ptr,
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    top_chunks_count,
    num_splits: tl.constexpr,
    split_top_chunks: tl.constexpr,
    softmax_scale: tl.constexpr,
    group_idx: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    split_id = tl.program_id(1)

    offs_h = tl.arange(0, group_size)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    d_mask = offs_d < head_dim

    q_offsets = row * group_size * head_dim + offs_h[:, None] * head_dim + offs_d[None, :]
    q = tl.load(q_ptr + q_offsets, mask=d_mask[None, :], other=0.0)

    m_i = tl.full((group_size,), -float("inf"), tl.float32)
    l_i = tl.zeros((group_size,), tl.float32)
    acc = tl.zeros((group_size, BLOCK_D), tl.float32)

    split_start = split_id * split_top_chunks
    split_end = tl.minimum(split_start + split_top_chunks, top_chunks_count)
    top_i = split_start
    while top_i < split_end:
        chunk_valid = tl.load(top_valid_ptr + row * top_chunks_count + top_i)
        logical_chunk = tl.load(top_chunks_ptr + row * top_chunks_count + top_i)
        physical_block = tl.load(block_table_ptr + logical_chunk)

        kv_offsets = (
            ((physical_block * chunk_size + offs_n[:, None]) * num_kv_heads + group_idx)
            * head_dim
            + offs_d[None, :]
        )
        k = tl.load(
            key_cache_ptr + kv_offsets,
            mask=(offs_n[:, None] < chunk_size) & d_mask[None, :] & chunk_valid,
            other=0.0,
        )
        v = tl.load(
            value_cache_ptr + kv_offsets,
            mask=(offs_n[:, None] < chunk_size) & d_mask[None, :] & chunk_valid,
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk = tl.where(chunk_valid, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new
        top_i += 1

    if split_id == num_splits - 1:
        current_chunk = tl.load(current_chunks_ptr + row)
        query_position = tl.load(query_positions_ptr + row)
        current_start = current_chunk * chunk_size
        tail_len = query_position - current_start + 1
        physical_block = tl.load(block_table_ptr + current_chunk)
        tail_mask = offs_n < tail_len
        kv_offsets = (
            ((physical_block * chunk_size + offs_n[:, None]) * num_kv_heads + group_idx)
            * head_dim
            + offs_d[None, :]
        )
        k = tl.load(
            key_cache_ptr + kv_offsets,
            mask=tail_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            value_cache_ptr + kv_offsets,
            mask=tail_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk = tl.where(tail_mask[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_new = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new

    acc_offsets = (
        ((row * num_splits + split_id) * group_size + offs_h[:, None]) * head_dim
        + offs_d[None, :]
    )
    stat_offsets = (row * num_splits + split_id) * group_size + offs_h
    tl.store(partial_acc_ptr + acc_offsets, acc, mask=d_mask[None, :])
    tl.store(partial_m_ptr + stat_offsets, m_i)
    tl.store(partial_l_ptr + stat_offsets, l_i)


@triton.jit
def _dsa_prefill_gqa_splitk_combine_kernel(
    partial_acc_ptr,
    partial_m_ptr,
    partial_l_ptr,
    out_ptr,
    num_splits: tl.constexpr,
    head_dim: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs_h = tl.arange(0, group_size)
    offs_d = tl.arange(0, BLOCK_D)
    d_mask = offs_d < head_dim

    m_i = tl.full((group_size,), -float("inf"), tl.float32)
    split_id = 0
    while split_id < num_splits:
        stat_offsets = (row * num_splits + split_id) * group_size + offs_h
        split_m = tl.load(partial_m_ptr + stat_offsets)
        m_i = tl.maximum(m_i, split_m)
        split_id += 1

    l_i = tl.zeros((group_size,), tl.float32)
    acc = tl.zeros((group_size, BLOCK_D), tl.float32)
    split_id = 0
    while split_id < num_splits:
        stat_offsets = (row * num_splits + split_id) * group_size + offs_h
        split_m = tl.load(partial_m_ptr + stat_offsets)
        split_l = tl.load(partial_l_ptr + stat_offsets)
        scale = tl.exp(split_m - m_i)
        acc_offsets = (
            ((row * num_splits + split_id) * group_size + offs_h[:, None]) * head_dim
            + offs_d[None, :]
        )
        split_acc = tl.load(
            partial_acc_ptr + acc_offsets,
            mask=d_mask[None, :],
            other=0.0,
        )
        acc += split_acc * scale[:, None]
        l_i += split_l * scale
        split_id += 1

    out = acc / l_i[:, None]
    out_offsets = row * group_size * head_dim + offs_h[:, None] * head_dim + offs_d[None, :]
    tl.store(out_ptr + out_offsets, out, mask=d_mask[None, :])


def dsa_prefill_gqa_splitk_attention(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    top_chunk_indices: torch.Tensor,
    top_chunk_valid: torch.Tensor,
    current_chunks: torch.Tensor,
    query_positions: torch.Tensor,
    *,
    group_idx: int,
    softmax_scale: float,
    split_top_chunks: int = 64,
) -> torch.Tensor:
    if split_top_chunks <= 0:
        raise ValueError(f"split_top_chunks must be positive, got {split_top_chunks}")
    q_len, group_size, head_dim = query_states.shape
    if q_len == 0:
        return torch.empty_like(query_states)
    top_chunks_count = int(top_chunk_indices.shape[1])
    num_splits = max(1, math.ceil(top_chunks_count / split_top_chunks))

    query_states = query_states.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    block_table = block_table.to(device=query_states.device, dtype=torch.int32).contiguous()
    top_chunk_indices = top_chunk_indices.to(
        device=query_states.device, dtype=torch.int32
    ).contiguous()
    top_chunk_valid = top_chunk_valid.to(device=query_states.device).contiguous()
    current_chunks = current_chunks.to(
        device=query_states.device, dtype=torch.int32
    ).contiguous()
    query_positions = query_positions.to(
        device=query_states.device, dtype=torch.int32
    ).contiguous()

    chunk_size = int(key_cache.shape[1])
    if chunk_size != 16:
        raise ValueError(f"prototype kernel currently expects chunk/page size 16, got {chunk_size}")
    block_d = triton.next_power_of_2(head_dim)
    partial_acc = torch.empty(
        q_len,
        num_splits,
        group_size,
        head_dim,
        device=query_states.device,
        dtype=torch.float32,
    )
    partial_m = torch.empty(
        q_len,
        num_splits,
        group_size,
        device=query_states.device,
        dtype=torch.float32,
    )
    partial_l = torch.empty_like(partial_m)
    output = torch.empty_like(query_states)

    _dsa_prefill_gqa_splitk_partial_kernel[(q_len, num_splits)](
        query_states,
        key_cache,
        value_cache,
        block_table,
        top_chunk_indices,
        top_chunk_valid,
        current_chunks,
        query_positions,
        partial_acc,
        partial_m,
        partial_l,
        top_chunks_count,
        num_splits=int(num_splits),
        split_top_chunks=int(split_top_chunks),
        softmax_scale=softmax_scale,
        group_idx=int(group_idx),
        num_kv_heads=int(key_cache.shape[2]),
        head_dim=int(head_dim),
        chunk_size=int(chunk_size),
        group_size=int(group_size),
        BLOCK_D=int(block_d),
        BLOCK_N=int(chunk_size),
        num_warps=4 if head_dim <= 128 else 8,
    )
    _dsa_prefill_gqa_splitk_combine_kernel[(q_len,)](
        partial_acc,
        partial_m,
        partial_l,
        output,
        num_splits=int(num_splits),
        head_dim=int(head_dim),
        group_size=int(group_size),
        BLOCK_D=int(block_d),
        num_warps=4 if head_dim <= 128 else 8,
    )
    return output


@triton.jit
def _dsa_prefill_gqa_union_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    union_chunks_ptr,
    union_masks_ptr,
    union_counts_ptr,
    row_starts_ptr,
    row_counts_ptr,
    current_chunks_ptr,
    tail_lens_ptr,
    out_ptr,
    max_union_chunks,
    softmax_scale: tl.constexpr,
    chunks_per_iter: tl.constexpr,
    key_stride_b: tl.constexpr,
    key_stride_n: tl.constexpr,
    key_stride_h: tl.constexpr,
    key_stride_d: tl.constexpr,
    value_stride_b: tl.constexpr,
    value_stride_n: tl.constexpr,
    value_stride_h: tl.constexpr,
    value_stride_d: tl.constexpr,
    group_idx: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    head = tl.program_id(1)

    offs_r = tl.arange(0, BLOCK_ROWS)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    d_mask = offs_d < head_dim

    row_start = tl.load(row_starts_ptr + row_block)
    row_count = tl.load(row_counts_ptr + row_block)
    rows = row_start + offs_r
    row_mask = offs_r < row_count

    q_offsets = (
        (rows[:, None] * group_size + head) * head_dim
        + offs_d[None, :]
    )
    q = tl.load(q_ptr + q_offsets, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

    m_i = tl.full((BLOCK_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_ROWS,), tl.float32)
    acc = tl.zeros((BLOCK_ROWS, BLOCK_D), tl.float32)

    union_count = tl.load(union_counts_ptr + row_block)
    chunk_i = 0
    while chunk_i < union_count:
        chunk_offsets = offs_n // chunk_size
        token_offsets = offs_n - chunk_offsets * chunk_size
        chunk_slots = chunk_i + chunk_offsets
        chunk_slot_valid = chunk_slots < union_count
        logical_chunk = tl.load(
            union_chunks_ptr + row_block * max_union_chunks + chunk_slots,
            mask=chunk_slot_valid,
            other=0,
        )
        row_bits = tl.load(
            union_masks_ptr + row_block * max_union_chunks + chunk_slots,
            mask=chunk_slot_valid,
            other=0,
        )
        row_has_token = ((row_bits[None, :] >> offs_r[:, None]) & 1) != 0
        token_active = row_has_token & chunk_slot_valid[None, :]
        active_rows = tl.max(tl.where(token_active, 1, 0), axis=1) != 0

        physical_block = tl.load(block_table_ptr + logical_chunk)
        kv_offsets = (
            physical_block[:, None] * key_stride_b
            + token_offsets[:, None] * key_stride_n
            + group_idx * key_stride_h
            + offs_d[None, :] * key_stride_d
        )
        value_offsets = (
            physical_block[:, None] * value_stride_b
            + token_offsets[:, None] * value_stride_n
            + group_idx * value_stride_h
            + offs_d[None, :] * value_stride_d
        )
        k = tl.load(
            key_cache_ptr + kv_offsets,
            mask=chunk_slot_valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            value_cache_ptr + value_offsets,
            mask=chunk_slot_valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk = tl.where(token_active, qk, -float("inf"))
        qk_max = tl.max(qk, axis=1)
        m_new = tl.where(active_rows, tl.maximum(m_i, qk_max), m_i)
        alpha = tl.where(active_rows, tl.exp(m_i - m_new), 1.0)
        p = tl.where(token_active, tl.exp(qk - m_new[:, None]), 0.0)
        l_new = tl.where(active_rows, l_i * alpha + tl.sum(p, axis=1), l_i)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new
        chunk_i += chunks_per_iter

    current_chunk = tl.load(current_chunks_ptr + row_block)
    physical_block = tl.load(block_table_ptr + current_chunk)
    kv_offsets = (
        physical_block * key_stride_b
        + offs_n[:, None] * key_stride_n
        + group_idx * key_stride_h
        + offs_d[None, :] * key_stride_d
    )
    value_offsets = (
        physical_block * value_stride_b
        + offs_n[:, None] * value_stride_n
        + group_idx * value_stride_h
        + offs_d[None, :] * value_stride_d
    )
    k = tl.load(
        key_cache_ptr + kv_offsets,
        mask=d_mask[None, :],
        other=0.0,
    )
    v = tl.load(
        value_cache_ptr + value_offsets,
        mask=d_mask[None, :],
        other=0.0,
    )
    tail_lens = tl.load(
        tail_lens_ptr + row_block * BLOCK_ROWS + offs_r,
        mask=row_mask,
        other=0,
    )
    tail_mask = offs_n[None, :] < tail_lens[:, None]
    active_rows = row_mask
    qk = tl.dot(q, tl.trans(k)) * softmax_scale
    qk = tl.where(tail_mask & active_rows[:, None], qk, -float("inf"))
    qk_max = tl.max(qk, axis=1)
    m_new = tl.maximum(m_i, qk_max)
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(qk - m_new[:, None])
    p = tl.where(tail_mask & active_rows[:, None], p, 0.0)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_new

    out = acc / l_i[:, None]
    tl.store(out_ptr + q_offsets, out, mask=row_mask[:, None] & d_mask[None, :])


def dsa_prefill_gqa_union_attention(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    union_chunks: torch.Tensor,
    union_masks: torch.Tensor,
    union_counts: torch.Tensor,
    row_starts: torch.Tensor,
    row_counts: torch.Tensor,
    current_chunks: torch.Tensor,
    tail_lens: torch.Tensor,
    *,
    group_idx: int,
    softmax_scale: float,
    chunks_per_iter: int = 4,
) -> torch.Tensor:
    """Run an exact union-of-top-chunks prefill prototype kernel.

    The caller groups adjacent query rows, provides the unique recalled chunks
    for each group, and a bitmask telling which rows selected each chunk.  The
    kernel then computes the same sparse attention as the per-row top-k path,
    but it loads each unique KV chunk once per row group.
    """

    if query_states.dim() != 3:
        raise ValueError(f"query_states must be [q, heads, dim], got {query_states.shape}")
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError(
            "key_cache/value_cache must use NHD [blocks, block, kv_heads, dim] layout"
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key/value cache shape mismatch: {key_cache.shape} vs {value_cache.shape}"
        )
    if union_chunks.shape != union_masks.shape:
        raise ValueError(
            f"union chunk/mask shape mismatch: {union_chunks.shape} vs {union_masks.shape}"
        )
    if union_chunks.dim() != 2:
        raise ValueError(f"union_chunks must be [row_blocks, max_union], got {union_chunks.shape}")
    row_blocks = int(union_chunks.shape[0])
    if row_blocks == 0:
        return torch.empty_like(query_states)
    if union_counts.shape != (row_blocks,):
        raise ValueError(f"union_counts must be [{row_blocks}], got {union_counts.shape}")
    if row_starts.shape != (row_blocks,) or row_counts.shape != (row_blocks,):
        raise ValueError("row_starts and row_counts must have one entry per row block")
    if current_chunks.shape != (row_blocks,):
        raise ValueError(f"current_chunks must be [{row_blocks}], got {current_chunks.shape}")
    if tail_lens.dim() != 2 or tail_lens.shape[0] != row_blocks:
        raise ValueError(f"tail_lens must be [row_blocks, block_rows], got {tail_lens.shape}")

    q_len, group_size, head_dim = query_states.shape
    if head_dim != key_cache.shape[-1]:
        raise ValueError(f"head_dim mismatch: query={head_dim} cache={key_cache.shape[-1]}")
    if head_dim > 256:
        raise ValueError(f"head_dim too large for prototype kernel: {head_dim}")
    chunk_size = int(key_cache.shape[1])
    if chunk_size != 16:
        raise ValueError(f"prototype kernel currently expects chunk/page size 16, got {chunk_size}")

    query_states = query_states.contiguous()
    block_table = block_table.to(device=query_states.device, dtype=torch.int32).contiguous()
    union_chunks = union_chunks.to(device=query_states.device, dtype=torch.int32).contiguous()
    union_masks = union_masks.to(device=query_states.device, dtype=torch.int32).contiguous()
    union_counts = union_counts.to(device=query_states.device, dtype=torch.int32).contiguous()
    row_starts = row_starts.to(device=query_states.device, dtype=torch.int32).contiguous()
    row_counts = row_counts.to(device=query_states.device, dtype=torch.int32).contiguous()
    current_chunks = current_chunks.to(device=query_states.device, dtype=torch.int32).contiguous()
    tail_lens = tail_lens.to(device=query_states.device, dtype=torch.int32).contiguous()

    output = torch.empty_like(query_states)
    block_d = triton.next_power_of_2(head_dim)
    block_rows = int(tail_lens.shape[1])
    if chunks_per_iter <= 0:
        raise ValueError(f"chunks_per_iter must be positive, got {chunks_per_iter}")
    _dsa_prefill_gqa_union_kernel[(row_blocks, group_size)](
        query_states,
        key_cache,
        value_cache,
        block_table,
        union_chunks,
        union_masks,
        union_counts,
        row_starts,
        row_counts,
        current_chunks,
        tail_lens,
        output,
        max_union_chunks=int(union_chunks.shape[1]),
        softmax_scale=softmax_scale,
        chunks_per_iter=int(chunks_per_iter),
        key_stride_b=int(key_cache.stride(0)),
        key_stride_n=int(key_cache.stride(1)),
        key_stride_h=int(key_cache.stride(2)),
        key_stride_d=int(key_cache.stride(3)),
        value_stride_b=int(value_cache.stride(0)),
        value_stride_n=int(value_cache.stride(1)),
        value_stride_h=int(value_cache.stride(2)),
        value_stride_d=int(value_cache.stride(3)),
        group_idx=int(group_idx),
        num_kv_heads=int(key_cache.shape[2]),
        head_dim=int(head_dim),
        chunk_size=int(chunk_size),
        group_size=int(group_size),
        BLOCK_ROWS=block_rows,
        BLOCK_D=int(block_d),
        BLOCK_N=int(chunk_size * chunks_per_iter),
        num_warps=4 if head_dim <= 128 else 8,
    )
    return output


@triton.jit
def _dsa_prefill_gqa_wide_union_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    union_chunks_ptr,
    full_masks_ptr,
    current_masks_ptr,
    union_counts_ptr,
    row_starts_ptr,
    row_counts_ptr,
    tail_lens_ptr,
    out_ptr,
    max_union_chunks,
    softmax_scale: tl.constexpr,
    chunks_per_iter: tl.constexpr,
    key_stride_b: tl.constexpr,
    key_stride_n: tl.constexpr,
    key_stride_h: tl.constexpr,
    key_stride_d: tl.constexpr,
    value_stride_b: tl.constexpr,
    value_stride_n: tl.constexpr,
    value_stride_h: tl.constexpr,
    value_stride_d: tl.constexpr,
    group_idx: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)
    head = tl.program_id(1)

    offs_r = tl.arange(0, BLOCK_ROWS)
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    d_mask = offs_d < head_dim

    row_start = tl.load(row_starts_ptr + row_block)
    row_count = tl.load(row_counts_ptr + row_block)
    rows = row_start + offs_r
    row_mask = offs_r < row_count

    q_offsets = (
        (rows[:, None] * group_size + head) * head_dim
        + offs_d[None, :]
    )
    q = tl.load(q_ptr + q_offsets, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

    tail_lens = tl.load(
        tail_lens_ptr + row_block * BLOCK_ROWS + offs_r,
        mask=row_mask,
        other=0,
    )

    m_i = tl.full((BLOCK_ROWS,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_ROWS,), tl.float32)
    acc = tl.zeros((BLOCK_ROWS, BLOCK_D), tl.float32)

    union_count = tl.load(union_counts_ptr + row_block)
    chunk_i = 0
    while chunk_i < union_count:
        chunk_offsets = offs_n // chunk_size
        token_offsets = offs_n - chunk_offsets * chunk_size
        chunk_slots = chunk_i + chunk_offsets
        chunk_slot_valid = chunk_slots < union_count
        logical_chunk = tl.load(
            union_chunks_ptr + row_block * max_union_chunks + chunk_slots,
            mask=chunk_slot_valid,
            other=0,
        )
        full_bits = tl.load(
            full_masks_ptr + row_block * max_union_chunks + chunk_slots,
            mask=chunk_slot_valid,
            other=0,
        )
        current_bits = tl.load(
            current_masks_ptr + row_block * max_union_chunks + chunk_slots,
            mask=chunk_slot_valid,
            other=0,
        )
        row_has_full = ((full_bits[None, :] >> offs_r[:, None]) & 1) != 0
        row_has_current = ((current_bits[None, :] >> offs_r[:, None]) & 1) != 0
        current_visible = row_has_current & (token_offsets[None, :] < tail_lens[:, None])
        token_active = (
            (row_has_full | current_visible)
            & chunk_slot_valid[None, :]
            & row_mask[:, None]
        )
        active_rows = tl.max(tl.where(token_active, 1, 0), axis=1) != 0

        physical_block = tl.load(block_table_ptr + logical_chunk)
        kv_offsets = (
            physical_block[:, None] * key_stride_b
            + token_offsets[:, None] * key_stride_n
            + group_idx * key_stride_h
            + offs_d[None, :] * key_stride_d
        )
        value_offsets = (
            physical_block[:, None] * value_stride_b
            + token_offsets[:, None] * value_stride_n
            + group_idx * value_stride_h
            + offs_d[None, :] * value_stride_d
        )
        k = tl.load(
            key_cache_ptr + kv_offsets,
            mask=chunk_slot_valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            value_cache_ptr + value_offsets,
            mask=chunk_slot_valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk = tl.where(token_active, qk, -float("inf"))
        qk_max = tl.max(qk, axis=1)
        m_new = tl.where(active_rows, tl.maximum(m_i, qk_max), m_i)
        alpha = tl.where(active_rows, tl.exp(m_i - m_new), 1.0)
        p = tl.where(token_active, tl.exp(qk - m_new[:, None]), 0.0)
        l_new = tl.where(active_rows, l_i * alpha + tl.sum(p, axis=1), l_i)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new
        chunk_i += chunks_per_iter

    out = acc / l_i[:, None]
    tl.store(out_ptr + q_offsets, out, mask=row_mask[:, None] & d_mask[None, :])


def dsa_prefill_gqa_wide_union_attention(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    union_chunks: torch.Tensor,
    full_masks: torch.Tensor,
    current_masks: torch.Tensor,
    union_counts: torch.Tensor,
    row_starts: torch.Tensor,
    row_counts: torch.Tensor,
    tail_lens: torch.Tensor,
    *,
    group_idx: int,
    softmax_scale: float,
    chunks_per_iter: int = 4,
) -> torch.Tensor:
    """Run an exact wider union kernel with per-row current-chunk tails.

    Unlike the smaller union kernel, a row block may span multiple current
    chunks.  The caller includes recalled chunks in ``full_masks`` and current
    chunks in ``current_masks``; the kernel applies each row's tail length only
    to its current chunk, preserving causal visibility.
    """

    if query_states.dim() != 3:
        raise ValueError(f"query_states must be [q, heads, dim], got {query_states.shape}")
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError(
            "key_cache/value_cache must use NHD [blocks, block, kv_heads, dim] layout"
        )
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key/value cache shape mismatch: {key_cache.shape} vs {value_cache.shape}"
        )
    if union_chunks.shape != full_masks.shape or union_chunks.shape != current_masks.shape:
        raise ValueError("union_chunks, full_masks, and current_masks must have matching shapes")
    if union_chunks.dim() != 2:
        raise ValueError(f"union_chunks must be [row_blocks, max_union], got {union_chunks.shape}")
    row_blocks = int(union_chunks.shape[0])
    if row_blocks == 0:
        return torch.empty_like(query_states)

    q_len, group_size, head_dim = query_states.shape
    if head_dim != key_cache.shape[-1]:
        raise ValueError(f"head_dim mismatch: query={head_dim} cache={key_cache.shape[-1]}")
    if head_dim > 256:
        raise ValueError(f"head_dim too large for prototype kernel: {head_dim}")
    chunk_size = int(key_cache.shape[1])
    if chunk_size != 16:
        raise ValueError(f"prototype kernel currently expects chunk/page size 16, got {chunk_size}")

    query_states = query_states.contiguous()
    block_table = block_table.to(device=query_states.device, dtype=torch.int32).contiguous()
    union_chunks = union_chunks.to(device=query_states.device, dtype=torch.int32).contiguous()
    full_masks = full_masks.to(device=query_states.device, dtype=torch.int64).contiguous()
    current_masks = current_masks.to(device=query_states.device, dtype=torch.int64).contiguous()
    union_counts = union_counts.to(device=query_states.device, dtype=torch.int32).contiguous()
    row_starts = row_starts.to(device=query_states.device, dtype=torch.int32).contiguous()
    row_counts = row_counts.to(device=query_states.device, dtype=torch.int32).contiguous()
    tail_lens = tail_lens.to(device=query_states.device, dtype=torch.int32).contiguous()

    block_rows = int(tail_lens.shape[1])
    if block_rows > 63:
        raise ValueError(f"wide union prototype supports at most 63 rows, got {block_rows}")
    if chunks_per_iter <= 0:
        raise ValueError(f"chunks_per_iter must be positive, got {chunks_per_iter}")

    output = torch.empty_like(query_states)
    block_d = triton.next_power_of_2(head_dim)
    _dsa_prefill_gqa_wide_union_kernel[(row_blocks, group_size)](
        query_states,
        key_cache,
        value_cache,
        block_table,
        union_chunks,
        full_masks,
        current_masks,
        union_counts,
        row_starts,
        row_counts,
        tail_lens,
        output,
        max_union_chunks=int(union_chunks.shape[1]),
        softmax_scale=softmax_scale,
        chunks_per_iter=int(chunks_per_iter),
        key_stride_b=int(key_cache.stride(0)),
        key_stride_n=int(key_cache.stride(1)),
        key_stride_h=int(key_cache.stride(2)),
        key_stride_d=int(key_cache.stride(3)),
        value_stride_b=int(value_cache.stride(0)),
        value_stride_n=int(value_cache.stride(1)),
        value_stride_h=int(value_cache.stride(2)),
        value_stride_d=int(value_cache.stride(3)),
        group_idx=int(group_idx),
        head_dim=int(head_dim),
        chunk_size=int(chunk_size),
        group_size=int(group_size),
        BLOCK_ROWS=block_rows,
        BLOCK_D=int(block_d),
        BLOCK_N=int(chunk_size * chunks_per_iter),
        num_warps=8 if block_rows >= 32 or head_dim > 128 else 4,
    )
    return output


@triton.jit
def _dsa_prefill_gqa_union_qh_kernel(
    q_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    union_chunks_ptr,
    union_masks_ptr,
    union_counts_ptr,
    row_starts_ptr,
    row_counts_ptr,
    current_chunks_ptr,
    tail_lens_ptr,
    out_ptr,
    max_union_chunks,
    softmax_scale: tl.constexpr,
    group_idx: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    chunk_size: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_QH: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_block = tl.program_id(0)

    offs_qh = tl.arange(0, BLOCK_QH)
    offs_row = offs_qh // group_size
    offs_head = offs_qh - offs_row * group_size
    offs_d = tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, BLOCK_N)
    d_mask = offs_d < head_dim

    row_start = tl.load(row_starts_ptr + row_block)
    row_count = tl.load(row_counts_ptr + row_block)
    rows = row_start + offs_row
    qh_mask = offs_row < row_count

    q_offsets = (
        (rows[:, None] * group_size + offs_head[:, None]) * head_dim
        + offs_d[None, :]
    )
    q = tl.load(q_ptr + q_offsets, mask=qh_mask[:, None] & d_mask[None, :], other=0.0)

    m_i = tl.full((BLOCK_QH,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_QH,), tl.float32)
    acc = tl.zeros((BLOCK_QH, BLOCK_D), tl.float32)

    union_count = tl.load(union_counts_ptr + row_block)
    chunk_i = 0
    while chunk_i < union_count:
        logical_chunk = tl.load(
            union_chunks_ptr + row_block * max_union_chunks + chunk_i
        )
        row_bits = tl.load(
            union_masks_ptr + row_block * max_union_chunks + chunk_i
        )
        qh_has_chunk = ((row_bits >> offs_row) & 1) != 0
        active_qh = qh_mask & qh_has_chunk

        physical_block = tl.load(block_table_ptr + logical_chunk)
        kv_offsets = (
            ((physical_block * chunk_size + offs_n[:, None]) * num_kv_heads + group_idx)
            * head_dim
            + offs_d[None, :]
        )
        k = tl.load(
            key_cache_ptr + kv_offsets,
            mask=d_mask[None, :],
            other=0.0,
        )
        v = tl.load(
            value_cache_ptr + kv_offsets,
            mask=d_mask[None, :],
            other=0.0,
        )

        qk = tl.dot(q, tl.trans(k)) * softmax_scale
        qk = tl.where(active_qh[:, None], qk, -float("inf"))
        qk_max = tl.max(qk, axis=1)
        m_new = tl.where(active_qh, tl.maximum(m_i, qk_max), m_i)
        alpha = tl.where(active_qh, tl.exp(m_i - m_new), 1.0)
        p = tl.where(active_qh[:, None], tl.exp(qk - m_new[:, None]), 0.0)
        l_new = tl.where(active_qh, l_i * alpha + tl.sum(p, axis=1), l_i)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
        l_i = l_new
        chunk_i += 1

    current_chunk = tl.load(current_chunks_ptr + row_block)
    physical_block = tl.load(block_table_ptr + current_chunk)
    kv_offsets = (
        ((physical_block * chunk_size + offs_n[:, None]) * num_kv_heads + group_idx)
        * head_dim
        + offs_d[None, :]
    )
    k = tl.load(
        key_cache_ptr + kv_offsets,
        mask=d_mask[None, :],
        other=0.0,
    )
    v = tl.load(
        value_cache_ptr + kv_offsets,
        mask=d_mask[None, :],
        other=0.0,
    )
    tail_lens = tl.load(
        tail_lens_ptr + row_block * BLOCK_ROWS + offs_row,
        mask=qh_mask,
        other=0,
    )
    tail_mask = offs_n[None, :] < tail_lens[:, None]
    qk = tl.dot(q, tl.trans(k)) * softmax_scale
    qk = tl.where(tail_mask & qh_mask[:, None], qk, -float("inf"))
    qk_max = tl.max(qk, axis=1)
    m_new = tl.maximum(m_i, qk_max)
    alpha = tl.exp(m_i - m_new)
    p = tl.exp(qk - m_new[:, None])
    p = tl.where(tail_mask & qh_mask[:, None], p, 0.0)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
    m_i = m_new

    out = acc / l_i[:, None]
    tl.store(out_ptr + q_offsets, out, mask=qh_mask[:, None] & d_mask[None, :])


def dsa_prefill_gqa_union_qh_attention(
    query_states: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    union_chunks: torch.Tensor,
    union_masks: torch.Tensor,
    union_counts: torch.Tensor,
    row_starts: torch.Tensor,
    row_counts: torch.Tensor,
    current_chunks: torch.Tensor,
    tail_lens: torch.Tensor,
    *,
    group_idx: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Run an exact union prefill kernel that also shares K/V across GQA heads."""

    q_len, group_size, head_dim = query_states.shape
    if q_len == 0:
        return torch.empty_like(query_states)
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise ValueError("key_cache/value_cache must be NHD 4D tensors")
    if key_cache.shape != value_cache.shape:
        raise ValueError(
            f"key/value cache shape mismatch: {key_cache.shape} vs {value_cache.shape}"
        )
    if head_dim != key_cache.shape[-1]:
        raise ValueError(f"head_dim mismatch: query={head_dim} cache={key_cache.shape[-1]}")
    chunk_size = int(key_cache.shape[1])
    if chunk_size != 16:
        raise ValueError(f"prototype kernel currently expects chunk/page size 16, got {chunk_size}")
    if union_chunks.dim() != 2 or union_chunks.shape != union_masks.shape:
        raise ValueError("union_chunks and union_masks must be matching 2D tensors")
    row_blocks = int(union_chunks.shape[0])
    if row_blocks == 0:
        return torch.empty_like(query_states)

    query_states = query_states.contiguous()
    key_cache = key_cache.contiguous()
    value_cache = value_cache.contiguous()
    block_table = block_table.to(device=query_states.device, dtype=torch.int32).contiguous()
    union_chunks = union_chunks.to(device=query_states.device, dtype=torch.int32).contiguous()
    union_masks = union_masks.to(device=query_states.device, dtype=torch.int32).contiguous()
    union_counts = union_counts.to(device=query_states.device, dtype=torch.int32).contiguous()
    row_starts = row_starts.to(device=query_states.device, dtype=torch.int32).contiguous()
    row_counts = row_counts.to(device=query_states.device, dtype=torch.int32).contiguous()
    current_chunks = current_chunks.to(device=query_states.device, dtype=torch.int32).contiguous()
    tail_lens = tail_lens.to(device=query_states.device, dtype=torch.int32).contiguous()

    output = torch.empty_like(query_states)
    block_rows = int(tail_lens.shape[1])
    block_qh = block_rows * group_size
    block_d = triton.next_power_of_2(head_dim)
    _dsa_prefill_gqa_union_qh_kernel[(row_blocks,)](
        query_states,
        key_cache,
        value_cache,
        block_table,
        union_chunks,
        union_masks,
        union_counts,
        row_starts,
        row_counts,
        current_chunks,
        tail_lens,
        output,
        max_union_chunks=int(union_chunks.shape[1]),
        softmax_scale=softmax_scale,
        group_idx=int(group_idx),
        num_kv_heads=int(key_cache.shape[2]),
        head_dim=int(head_dim),
        chunk_size=int(chunk_size),
        group_size=int(group_size),
        BLOCK_ROWS=block_rows,
        BLOCK_QH=block_qh,
        BLOCK_D=int(block_d),
        BLOCK_N=int(chunk_size),
        num_warps=4 if block_qh <= 8 else 8,
    )
    return output
