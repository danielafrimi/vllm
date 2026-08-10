# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton decode page-table builder for Nemotron-H chunked DSA."""

from __future__ import annotations

import torch

try:
    from vllm.triton_utils import tl, triton
except ImportError:
    tl = None
    triton = None


_DECODE_PAGE_TABLE_BLOCK_WIDTH = 256
_BATCHED_PAGE_TABLE_TILE_WIDTH = 256
_UNIFIED_PAGE_TABLE_BLOCK_ROWS = 8
_UNIFIED_PAGE_TABLE_BLOCK_WIDTH = 128
_MIXED_ROW_METADATA_WIDTH = 4


if triton is not None and tl is not None:

    @triton.jit(
        do_not_specialize=[
            "num_rows",
            "top_width",
            "table_width",
            "current_chunk",
            "chunk_size",
            "tail_len",
            "recent_window_pages",
        ]
    )
    def _dsa_decode_page_table_kernel(
        block_table,
        selected_blocks,
        selected_valid,
        selected_ranks,
        selected_counts,
        page_table,
        seqused_k,
        num_rows,
        top_width,
        table_width,
        current_chunk,
        chunk_size,
        tail_len,
        recent_window_pages,
        BLOCK_WIDTH: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        selected_mask = offsets < top_width
        selected_offset = row * top_width + offsets
        valid = selected_mask & (
            tl.load(selected_valid + selected_offset, mask=selected_mask, other=0) != 0
        )
        logical_page = tl.load(
            selected_blocks + selected_offset,
            mask=selected_mask,
            other=0,
        ).to(tl.int64)
        selected_rank = tl.load(
            selected_ranks + selected_offset,
            mask=selected_mask,
            other=0,
        ).to(tl.int64)
        physical_page = tl.load(block_table + logical_page, mask=valid, other=0)
        tl.store(
            page_table + row * table_width + selected_rank,
            physical_page,
            mask=valid,
        )

        selected_count = tl.load(selected_counts + row).to(tl.int64)
        recent_count = tl.minimum(tl.maximum(current_chunk, 0), recent_window_pages).to(
            tl.int64
        )
        recent_start = current_chunk - recent_count
        recent_offset = offsets - selected_count
        recent_mask = (recent_offset >= 0) & (recent_offset < recent_count)
        recent_physical_page = tl.load(
            block_table + recent_start + recent_offset,
            mask=recent_mask,
            other=0,
        )
        tl.store(
            page_table + row * table_width + offsets,
            recent_physical_page,
            mask=recent_mask,
        )

        current_physical_page = tl.load(block_table + current_chunk)
        tl.store(
            page_table + row * table_width + selected_count + recent_count,
            current_physical_page,
            mask=(tile == 0),
        )
        tl.store(
            seqused_k + row,
            (selected_count + recent_count).to(tl.int32) * chunk_size + tail_len,
            mask=tile == 0,
        )

    @triton.jit(
        do_not_specialize=[
            "num_rows",
            "top_width",
            "table_width",
            "block_table_width",
            "chunk_size",
            "recent_window_pages",
        ]
    )
    def _dsa_batched_decode_page_table_kernel(
        block_table,
        selected_blocks,
        selected_valid,
        selected_prefix_counts,
        selected_counts,
        row_seq_ids,
        current_chunks,
        tail_lens,
        page_table,
        seqused_k,
        cu_seqlens_q,
        num_rows,
        top_width,
        table_width,
        block_table_width,
        chunk_size,
        recent_window_pages,
        BLOCK_WIDTH: tl.constexpr,
        USE_COUNTS: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        selected_mask = offsets < top_width

        seq_idx = tl.load(row_seq_ids + row).to(tl.int64)
        current_chunk = tl.load(current_chunks + row).to(tl.int64)
        tail_len = tl.load(tail_lens + row).to(tl.int32)
        recent_count = tl.minimum(tl.maximum(current_chunk, 0), recent_window_pages).to(
            tl.int64
        )
        recent_start = current_chunk - recent_count
        selected_offset = row * top_width + offsets
        logical_page = tl.load(
            selected_blocks + selected_offset,
            mask=selected_mask,
            other=0,
        ).to(tl.int64)
        if USE_COUNTS:
            selected_count = tl.load(selected_counts + row).to(tl.int64)
            selected_count = tl.minimum(tl.maximum(selected_count, 0), top_width)
            valid = (
                selected_mask
                & (offsets < selected_count)
                & (logical_page >= 0)
                & (logical_page < recent_start)
            )
            selected_rank = offsets
        else:
            valid = (
                selected_mask
                & (
                    tl.load(
                        selected_valid + selected_offset, mask=selected_mask, other=0
                    )
                    != 0
                )
                & (logical_page >= 0)
                & (logical_page < recent_start)
            )
            selected_prefix_count = tl.load(
                selected_prefix_counts + selected_offset,
                mask=selected_mask,
                other=0,
            ).to(tl.int64)
            selected_rank = selected_prefix_count - 1
            selected_count = tl.load(
                selected_prefix_counts + row * top_width + top_width - 1,
                mask=top_width > 0,
                other=0,
            ).to(tl.int64)

        block_table_base = seq_idx * block_table_width
        physical_page = tl.load(
            block_table + block_table_base + logical_page,
            mask=valid,
            other=0,
        )
        tl.store(
            page_table + row * table_width + selected_rank,
            physical_page,
            mask=valid,
        )

        tl.store(
            page_table + row * table_width + offsets,
            tl.zeros((BLOCK_WIDTH,), dtype=tl.int32),
            mask=(offsets >= selected_count)
            & (offsets != selected_count + recent_count)
            & (offsets < table_width),
        )
        recent_offset = offsets - selected_count
        recent_mask = (
            (recent_offset >= 0)
            & (recent_offset < recent_count)
            & (recent_start + recent_offset < block_table_width)
        )
        recent_physical_page = tl.load(
            block_table + block_table_base + recent_start + recent_offset,
            mask=recent_mask,
            other=0,
        )
        tl.store(
            page_table + row * table_width + offsets,
            recent_physical_page,
            mask=recent_mask,
        )
        current_in_range = (
            (current_chunk >= 0) & (current_chunk < block_table_width) & (tile == 0)
        )
        current_physical_page = tl.load(
            block_table + block_table_base + current_chunk,
            mask=current_in_range,
            other=0,
        )
        tl.store(
            page_table + row * table_width + selected_count + recent_count,
            current_physical_page,
            mask=current_in_range,
        )
        tl.store(
            seqused_k + row,
            (selected_count + recent_count) * chunk_size + tail_len,
            mask=current_in_range,
        )
        tl.store(cu_seqlens_q + row, row, mask=tile == 0)
        tl.store(
            cu_seqlens_q + num_rows,
            num_rows,
            mask=(tile == 0) & (row == num_rows - 1),
        )

    @triton.jit(
        do_not_specialize=[
            "num_rows",
            "top_width",
            "table_width",
            "block_table_width",
            "chunk_size",
            "recent_window_pages",
        ]
    )
    def _dsa_batched_mixed_page_table_kernel(
        block_table,
        selected_blocks,
        selected_valid,
        selected_prefix_counts,
        selected_counts,
        row_metadata,
        current_chunks,
        tail_lens,
        page_table,
        seqused_k,
        num_rows,
        top_width,
        table_width,
        block_table_width,
        chunk_size,
        recent_window_pages,
        ROW_METADATA_WIDTH: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        USE_COUNTS: tl.constexpr,
    ):
        row = tl.program_id(0)
        tile = tl.program_id(1)
        offsets = tile * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        selected_mask = offsets < top_width

        metadata_base = row * ROW_METADATA_WIDTH
        seq_idx = tl.load(row_metadata + metadata_base).to(tl.int64)
        sparse_row = tl.load(row_metadata + metadata_base + 1).to(tl.int64)
        is_dense = sparse_row < 0
        sparse_row_safe = tl.where(is_dense, 0, sparse_row)
        block_table_base = seq_idx * block_table_width

        dense_pages = tl.load(row_metadata + metadata_base + 2).to(tl.int64)
        dense_physical_page = tl.load(
            block_table + block_table_base + offsets,
            mask=is_dense & (offsets < dense_pages),
            other=0,
        )
        tl.store(
            page_table + row * table_width + offsets,
            dense_physical_page,
            mask=is_dense & (offsets < dense_pages),
        )
        tl.store(
            page_table + row * table_width + offsets,
            tl.zeros((BLOCK_WIDTH,), dtype=tl.int32),
            mask=is_dense & (offsets >= dense_pages) & (offsets < table_width),
        )

        current_chunk = tl.load(
            current_chunks + sparse_row_safe,
            mask=~is_dense,
            other=0,
        ).to(tl.int64)
        tail_len = tl.load(
            tail_lens + sparse_row_safe,
            mask=~is_dense,
            other=0,
        ).to(tl.int32)
        recent_count = tl.minimum(tl.maximum(current_chunk, 0), recent_window_pages).to(
            tl.int64
        )
        recent_start = current_chunk - recent_count
        selected_offset = sparse_row_safe * top_width + offsets
        logical_page = tl.load(
            selected_blocks + selected_offset,
            mask=(~is_dense) & selected_mask,
            other=0,
        ).to(tl.int64)
        if USE_COUNTS:
            selected_count = tl.load(
                selected_counts + sparse_row_safe,
                mask=~is_dense,
                other=0,
            ).to(tl.int64)
            selected_count = tl.minimum(tl.maximum(selected_count, 0), top_width)
            valid = (
                (~is_dense)
                & selected_mask
                & (offsets < selected_count)
                & (logical_page >= 0)
                & (logical_page < recent_start)
            )
            selected_rank = offsets
        else:
            valid = (
                (~is_dense)
                & selected_mask
                & (
                    tl.load(
                        selected_valid + selected_offset,
                        mask=(~is_dense) & selected_mask,
                        other=0,
                    )
                    != 0
                )
                & (logical_page >= 0)
                & (logical_page < recent_start)
            )
            selected_prefix_count = tl.load(
                selected_prefix_counts + selected_offset,
                mask=(~is_dense) & selected_mask,
                other=0,
            ).to(tl.int64)
            selected_rank = selected_prefix_count - 1
            selected_count = tl.load(
                selected_prefix_counts + sparse_row_safe * top_width + top_width - 1,
                mask=(~is_dense) & (top_width > 0),
                other=0,
            ).to(tl.int64)

        sparse_physical_page = tl.load(
            block_table + block_table_base + logical_page,
            mask=valid,
            other=0,
        )
        tl.store(
            page_table + row * table_width + selected_rank,
            sparse_physical_page,
            mask=valid,
        )

        tl.store(
            page_table + row * table_width + offsets,
            tl.zeros((BLOCK_WIDTH,), dtype=tl.int32),
            mask=(~is_dense)
            & (offsets >= selected_count)
            & (offsets != selected_count + recent_count)
            & (offsets < table_width),
        )
        recent_offset = offsets - selected_count
        recent_mask = (
            (~is_dense)
            & (recent_offset >= 0)
            & (recent_offset < recent_count)
            & (recent_start + recent_offset < block_table_width)
        )
        recent_physical_page = tl.load(
            block_table + block_table_base + recent_start + recent_offset,
            mask=recent_mask,
            other=0,
        )
        tl.store(
            page_table + row * table_width + offsets,
            recent_physical_page,
            mask=recent_mask,
        )
        current_in_range = (
            (~is_dense)
            & (current_chunk >= 0)
            & (current_chunk < block_table_width)
            & (tile == 0)
        )
        current_physical_page = tl.load(
            block_table + block_table_base + current_chunk,
            mask=current_in_range,
            other=0,
        )
        tl.store(
            page_table + row * table_width + selected_count + recent_count,
            current_physical_page,
            mask=current_in_range,
        )
        sparse_seqused = (selected_count + recent_count) * chunk_size + tail_len
        dense_seqused = tl.load(row_metadata + metadata_base + 3)
        tl.store(
            seqused_k + row,
            tl.where(is_dense, dense_seqused, sparse_seqused),
            mask=tile == 0,
        )

    @triton.jit(
        do_not_specialize=[
            "active_seq_count",
            "num_actual_tokens",
            "num_requests",
            "top_width",
            "table_width",
            "block_table_width",
            "chunk_size",
            "dense_decode_threshold",
            "dense_prefill_threshold",
            "recent_window_pages",
        ]
    )
    def _dsa_batched_unified_page_table_kernel(
        block_table,
        selected_blocks,
        selected_counts,
        query_start_loc,
        seq_lens,
        page_table,
        seqused_k,
        cu_seqlens_q,
        block_table_stride_seq,
        block_table_stride_page,
        selected_blocks_stride_row,
        selected_blocks_stride_col,
        selected_counts_stride,
        query_start_loc_stride,
        seq_lens_stride,
        page_table_stride_row,
        page_table_stride_col,
        active_seq_count,
        num_actual_tokens,
        num_requests,
        top_width,
        table_width,
        block_table_width,
        chunk_size,
        dense_decode_threshold,
        dense_prefill_threshold,
        recent_window_pages,
        MAX_ACTIVE_SEQS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        row_block = tl.program_id(1)
        tile = tl.program_id(2)

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
        key_len_all = tl.load(
            seq_lens + seq_offsets * seq_lens_stride,
            mask=seq_mask,
            other=0,
        )
        query_position_start_all = key_len_all - q_len_all
        dense_threshold_all = tl.where(
            q_len_all > 1,
            dense_prefill_threshold,
            dense_decode_threshold,
        )
        active_all = (
            seq_mask
            & (q_len_all > 0)
            & (key_len_all > 0)
            & (query_position_start_all >= 0)
        )
        dense_prefix_len_all = tl.minimum(
            tl.maximum(dense_threshold_all - query_position_start_all, 0),
            q_len_all,
        )
        dense_all = active_all & (dense_prefix_len_all > 0)
        sparse_len_all = q_len_all - dense_prefix_len_all
        sparse_all = active_all & (sparse_len_all > 0)
        request_count_all = tl.where(
            active_all,
            dense_all.to(tl.int32) + sparse_len_all,
            0,
        )
        request_start_all = tl.cumsum(request_count_all, 0) - request_count_all

        this_seq = seq_offsets == seq_idx
        q_start = tl.sum(tl.where(this_seq, q_start_all, 0), 0)
        query_position_start = tl.sum(
            tl.where(this_seq, query_position_start_all, 0), 0
        )
        dense_prefix_len = tl.sum(tl.where(this_seq, dense_prefix_len_all, 0), 0)
        sparse_len = tl.sum(tl.where(this_seq, sparse_len_all, 0), 0)
        request_start = tl.sum(tl.where(this_seq, request_start_all, 0), 0)
        is_dense = tl.sum(tl.where(this_seq, dense_all.to(tl.int32), 0), 0) != 0
        is_sparse = tl.sum(tl.where(this_seq, sparse_all.to(tl.int32), 0), 0) != 0

        row_offsets = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        page_offsets = tile * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        row_offsets_2d = row_offsets[:, None]
        page_offsets_2d = page_offsets[None, :]
        dense_request_count = is_dense.to(tl.int32)
        sparse_offsets = row_offsets - dense_request_count
        request_rows = request_start + row_offsets
        token_rows = q_start + dense_prefix_len + sparse_offsets
        in_table = page_offsets_2d < table_width
        row_in_request_range = request_rows < num_requests
        dense_row = is_dense & (row_offsets == 0) & (row_block == 0)
        sparse_row = is_sparse & (sparse_offsets >= 0) & (sparse_offsets < sparse_len)
        active_row = (dense_row | sparse_row) & row_in_request_range
        page_ptrs = (
            page_table
            + request_rows[:, None] * page_table_stride_row
            + page_offsets_2d * page_table_stride_col
        )
        tl.store(
            page_ptrs,
            tl.zeros((BLOCK_ROWS, BLOCK_WIDTH), dtype=tl.int32),
            mask=active_row[:, None] & in_table,
        )

        dense_key_len = query_position_start + dense_prefix_len
        dense_pages = tl.cdiv(dense_key_len, chunk_size)
        dense_page_offsets = page_offsets_2d + row_offsets_2d * 0
        dense_mask = (
            dense_row[:, None]
            & in_table
            & (dense_page_offsets < dense_pages)
            & (dense_page_offsets < block_table_width)
            & row_in_request_range[:, None]
        )
        dense_physical_pages = tl.load(
            block_table
            + seq_idx * block_table_stride_seq
            + dense_page_offsets * block_table_stride_page,
            mask=dense_mask,
            other=0,
        )
        tl.store(page_ptrs, dense_physical_pages, mask=dense_mask)

        positions = query_position_start + dense_prefix_len + sparse_offsets
        current_chunks = positions // chunk_size
        tail_lens = positions - current_chunks * chunk_size + 1
        selected_count = tl.load(
            selected_counts + token_rows * selected_counts_stride,
            mask=sparse_row & (token_rows < num_actual_tokens),
            other=0,
        ).to(tl.int64)
        selected_count = tl.minimum(tl.maximum(selected_count, 0), top_width)
        recent_count = tl.minimum(
            tl.maximum(current_chunks, 0), recent_window_pages
        ).to(tl.int64)
        recent_start = current_chunks - recent_count
        selected_mask = (
            sparse_row[:, None]
            & row_in_request_range[:, None]
            & in_table
            & (page_offsets_2d < top_width)
            & (page_offsets_2d < selected_count[:, None])
        )
        logical_pages = tl.load(
            selected_blocks
            + token_rows[:, None] * selected_blocks_stride_row
            + page_offsets_2d * selected_blocks_stride_col,
            mask=selected_mask,
            other=0,
        ).to(tl.int64)
        sparse_selected_mask = (
            selected_mask
            & (logical_pages >= 0)
            & (logical_pages < recent_start[:, None])
            & (logical_pages < block_table_width)
        )
        sparse_physical_pages = tl.load(
            block_table
            + seq_idx * block_table_stride_seq
            + logical_pages * block_table_stride_page,
            mask=sparse_selected_mask,
            other=0,
        )
        tl.store(page_ptrs, sparse_physical_pages, mask=sparse_selected_mask)

        recent_offsets = page_offsets_2d - selected_count[:, None]
        recent_logical_pages = recent_start[:, None] + recent_offsets
        recent_mask = (
            sparse_row[:, None]
            & row_in_request_range[:, None]
            & in_table
            & (recent_offsets >= 0)
            & (recent_offsets < recent_count[:, None])
            & (recent_logical_pages >= 0)
            & (recent_logical_pages < block_table_width)
        )
        recent_physical_pages = tl.load(
            block_table
            + seq_idx * block_table_stride_seq
            + recent_logical_pages * block_table_stride_page,
            mask=recent_mask,
            other=0,
        )
        tl.store(page_ptrs, recent_physical_pages, mask=recent_mask)

        current_mask = (
            sparse_row[:, None]
            & row_in_request_range[:, None]
            & in_table
            & (page_offsets_2d == selected_count[:, None] + recent_count[:, None])
            & (current_chunks[:, None] >= 0)
            & (current_chunks[:, None] < block_table_width)
        )
        current_physical_pages = tl.load(
            block_table
            + seq_idx * block_table_stride_seq
            + current_chunks[:, None] * block_table_stride_page,
            mask=current_mask,
            other=0,
        )
        tl.store(page_ptrs, current_physical_pages, mask=current_mask)

        metadata_tile = tile == 0
        dense_metadata = (
            metadata_tile & is_dense & (row_block == 0) & (request_start < num_requests)
        )
        tl.store(seqused_k + request_start, dense_key_len, mask=dense_metadata)
        tl.store(cu_seqlens_q + request_start, q_start, mask=dense_metadata)
        tl.store(
            cu_seqlens_q + request_start + 1,
            q_start + dense_prefix_len,
            mask=dense_metadata,
        )

        sparse_metadata = (
            metadata_tile
            & sparse_row
            & row_in_request_range
            & (token_rows < num_actual_tokens)
        )
        sparse_seqused = (selected_count + recent_count).to(
            tl.int32
        ) * chunk_size + tail_lens
        tl.store(seqused_k + request_rows, sparse_seqused, mask=sparse_metadata)
        tl.store(cu_seqlens_q + request_rows, token_rows, mask=sparse_metadata)
        tl.store(
            cu_seqlens_q + request_rows + 1,
            token_rows + 1,
            mask=sparse_metadata,
        )


def dsa_decode_page_table_triton(
    *,
    block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
    selected_valid: torch.Tensor,
    current_chunk: int,
    chunk_size: int,
    tail_len: int,
    recent_window_pages: int = 0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if triton is None or tl is None:
        return None
    if not block_table.is_cuda:
        return None
    if selected_blocks.device != block_table.device:
        selected_blocks = selected_blocks.to(device=block_table.device)
    if selected_valid.device != block_table.device:
        selected_valid = selected_valid.to(device=block_table.device)
    if selected_blocks.shape != selected_valid.shape:
        return None
    if selected_blocks.dim() != 2 or block_table.dim() != 1:
        return None
    if current_chunk < 0 or current_chunk >= int(block_table.shape[0]):
        return None
    if recent_window_pages < 0:
        return None

    selected_blocks = selected_blocks.to(dtype=torch.long).contiguous()
    recent_start = max(current_chunk - recent_window_pages, 0)
    selected_valid = (
        selected_valid.to(dtype=torch.bool)
        & (selected_blocks >= 0)
        & (selected_blocks < recent_start)
    ).contiguous()

    num_rows = int(selected_blocks.shape[0])
    top_width = int(selected_blocks.shape[1])
    table_width = top_width + recent_window_pages + 1
    selected_ranks = (
        selected_valid.to(dtype=torch.long).cumsum(dim=-1) - 1
    ).contiguous()
    selected_counts = selected_valid.sum(dim=-1).to(dtype=torch.long).contiguous()
    page_table = torch.zeros(
        num_rows,
        table_width,
        device=block_table.device,
        dtype=torch.int32,
    )
    seqused_k = torch.empty(
        num_rows,
        device=block_table.device,
        dtype=torch.int32,
    )
    _dsa_decode_page_table_kernel[
        (
            num_rows,
            triton.cdiv(table_width, _DECODE_PAGE_TABLE_BLOCK_WIDTH),
        )
    ](
        block_table.contiguous(),
        selected_blocks,
        selected_valid,
        selected_ranks,
        selected_counts,
        page_table,
        seqused_k,
        num_rows,
        top_width,
        table_width,
        current_chunk,
        chunk_size,
        tail_len,
        recent_window_pages,
        BLOCK_WIDTH=_DECODE_PAGE_TABLE_BLOCK_WIDTH,
        num_warps=1,
        num_stages=2,
    )
    return page_table, seqused_k


def dsa_batched_decode_page_table_triton(
    *,
    block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
    selected_valid: torch.Tensor | None = None,
    selected_counts: torch.Tensor | None = None,
    row_seq_ids: torch.Tensor,
    current_chunks: torch.Tensor,
    tail_lens: torch.Tensor,
    chunk_size: int,
    selected_valid_is_bounded: bool = False,
    recent_window_pages: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if triton is None or tl is None:
        return None
    if not block_table.is_cuda:
        return None
    if block_table.dim() != 2:
        return None
    if selected_valid is None and selected_counts is None:
        return None
    if selected_blocks.dim() != 2:
        return None
    if selected_valid is not None and selected_blocks.shape != selected_valid.shape:
        return None
    num_rows = int(selected_blocks.shape[0])
    top_width = int(selected_blocks.shape[1])
    if recent_window_pages < 0:
        return None
    table_width = top_width + recent_window_pages + 1
    if (
        row_seq_ids.numel() != num_rows
        or current_chunks.numel() != num_rows
        or tail_lens.numel() != num_rows
    ):
        return None
    if selected_counts is not None and (
        selected_counts.dim() != 1 or selected_counts.numel() != num_rows
    ):
        return None

    device = block_table.device
    selected_blocks = selected_blocks.to(device=device).contiguous()
    row_seq_ids = row_seq_ids.to(
        device=device,
        dtype=torch.int32,
    ).contiguous()
    current_chunks = current_chunks.to(
        device=device,
        dtype=torch.int32,
    ).contiguous()
    tail_lens = tail_lens.to(
        device=device,
        dtype=torch.int32,
    ).contiguous()
    use_counts = selected_counts is not None
    if use_counts:
        assert selected_counts is not None
        selected_counts = selected_counts.to(
            device=device,
            dtype=torch.int32,
        ).contiguous()
        selected_valid_arg = selected_blocks
        selected_prefix_counts = selected_blocks
    else:
        assert selected_valid is not None
        selected_valid = selected_valid.to(
            device=device,
            dtype=torch.bool,
        ).contiguous()
        if selected_valid_is_bounded and recent_window_pages == 0:
            selected_valid = selected_valid.contiguous()
        else:
            remote_limits = torch.clamp(
                current_chunks[:, None] - recent_window_pages,
                min=0,
            )
            selected_valid = (
                selected_valid
                & (selected_blocks >= 0)
                & (selected_blocks < remote_limits)
            ).contiguous()
        if top_width == 0:
            selected_prefix_counts = torch.empty(
                num_rows,
                0,
                device=device,
                dtype=torch.int32,
            )
        else:
            selected_prefix_counts = torch.cumsum(
                selected_valid,
                dim=-1,
                dtype=torch.int32,
            )
        selected_counts = selected_prefix_counts
        selected_valid_arg = selected_valid

    page_table = torch.empty(
        num_rows,
        table_width,
        device=device,
        dtype=torch.int32,
    )
    seqused_k = torch.empty(
        num_rows,
        device=device,
        dtype=torch.int32,
    )
    cu_seqlens_q = torch.empty(
        num_rows + 1,
        device=device,
        dtype=torch.int32,
    )
    _dsa_batched_decode_page_table_kernel[
        (
            num_rows,
            triton.cdiv(table_width, _BATCHED_PAGE_TABLE_TILE_WIDTH),
        )
    ](
        block_table.contiguous(),
        selected_blocks,
        selected_valid_arg,
        selected_prefix_counts,
        selected_counts,
        row_seq_ids,
        current_chunks,
        tail_lens,
        page_table,
        seqused_k,
        cu_seqlens_q,
        num_rows,
        top_width,
        table_width,
        int(block_table.shape[1]),
        chunk_size,
        recent_window_pages,
        BLOCK_WIDTH=_BATCHED_PAGE_TABLE_TILE_WIDTH,
        USE_COUNTS=use_counts,
        num_warps=1,
        num_stages=2,
    )
    return page_table, seqused_k, cu_seqlens_q


def dsa_batched_mixed_page_table_triton(
    *,
    block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
    selected_valid: torch.Tensor | None = None,
    selected_counts: torch.Tensor | None = None,
    row_metadata: torch.Tensor,
    current_chunks: torch.Tensor,
    tail_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    table_width: int,
    chunk_size: int,
    selected_valid_is_bounded: bool = False,
    recent_window_pages: int = 0,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if triton is None or tl is None:
        return None
    if not block_table.is_cuda:
        return None
    if block_table.dim() != 2:
        return None
    if selected_valid is None and selected_counts is None:
        return None
    if selected_blocks.dim() != 2:
        return None
    if selected_valid is not None and selected_blocks.shape != selected_valid.shape:
        return None
    if row_metadata.dim() != 2:
        return None
    if int(row_metadata.shape[1]) != _MIXED_ROW_METADATA_WIDTH:
        return None
    num_rows = int(row_metadata.shape[0])
    sparse_rows = int(selected_blocks.shape[0])
    top_width = int(selected_blocks.shape[1])
    if recent_window_pages < 0:
        return None
    if table_width <= 0:
        return None
    if table_width < top_width + recent_window_pages + 1:
        return None
    if (
        current_chunks.numel() != sparse_rows
        or tail_lens.numel() != sparse_rows
        or cu_seqlens_q.numel() != num_rows + 1
    ):
        return None
    if selected_counts is not None and (
        selected_counts.dim() != 1 or selected_counts.numel() != sparse_rows
    ):
        return None

    device = block_table.device
    selected_blocks = selected_blocks.to(device=device).contiguous()
    row_metadata = row_metadata.to(device=device, dtype=torch.int32).contiguous()
    current_chunks = current_chunks.to(
        device=device,
        dtype=torch.int32,
    ).contiguous()
    tail_lens = tail_lens.to(
        device=device,
        dtype=torch.int32,
    ).contiguous()
    cu_seqlens_q = cu_seqlens_q.to(
        device=device,
        dtype=torch.int32,
    ).contiguous()
    use_counts = selected_counts is not None
    if use_counts:
        assert selected_counts is not None
        selected_counts = selected_counts.to(
            device=device,
            dtype=torch.int32,
        ).contiguous()
        selected_valid_arg = selected_blocks
        selected_prefix_counts = selected_blocks
    else:
        assert selected_valid is not None
        selected_valid = selected_valid.to(
            device=device,
            dtype=torch.bool,
        ).contiguous()
        if selected_valid_is_bounded and recent_window_pages == 0:
            selected_valid = selected_valid.contiguous()
        else:
            remote_limits = torch.clamp(
                current_chunks[:, None] - recent_window_pages,
                min=0,
            )
            selected_valid = (
                selected_valid
                & (selected_blocks >= 0)
                & (selected_blocks < remote_limits)
            ).contiguous()
        if top_width == 0:
            selected_prefix_counts = torch.empty(
                sparse_rows,
                0,
                device=device,
                dtype=torch.int32,
            )
        else:
            selected_prefix_counts = torch.cumsum(
                selected_valid,
                dim=-1,
                dtype=torch.int32,
            )
        selected_counts = selected_prefix_counts
        selected_valid_arg = selected_valid

    page_table = torch.empty(
        num_rows,
        table_width,
        device=device,
        dtype=torch.int32,
    )
    seqused_k = torch.empty(
        num_rows,
        device=device,
        dtype=torch.int32,
    )
    _dsa_batched_mixed_page_table_kernel[
        (
            num_rows,
            triton.cdiv(table_width, _BATCHED_PAGE_TABLE_TILE_WIDTH),
        )
    ](
        block_table.contiguous(),
        selected_blocks,
        selected_valid_arg,
        selected_prefix_counts,
        selected_counts,
        row_metadata,
        current_chunks,
        tail_lens,
        page_table,
        seqused_k,
        num_rows,
        top_width,
        table_width,
        int(block_table.shape[1]),
        chunk_size,
        recent_window_pages,
        ROW_METADATA_WIDTH=_MIXED_ROW_METADATA_WIDTH,
        BLOCK_WIDTH=_BATCHED_PAGE_TABLE_TILE_WIDTH,
        USE_COUNTS=use_counts,
        num_warps=1,
        num_stages=2,
    )
    return page_table, seqused_k


def dsa_batched_unified_page_table_triton(
    *,
    block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
    selected_counts: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    num_actual_tokens: int,
    active_seq_count: int,
    num_requests: int,
    table_width: int,
    max_q_len: int,
    chunk_size: int,
    dense_decode_threshold: int,
    dense_prefill_threshold: int,
    recent_window_pages: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Build dense/sparse flattened page-table metadata from GPU batch metadata.

    The CPU supplies only allocation/launch bounds. Sequence classification,
    request row mapping, ``cu_seqlens_q``, ``seqused_k``, and page-table entries
    are derived from the GPU-resident batch metadata in one Triton kernel.
    """
    if triton is None or tl is None:
        return None
    if (
        not block_table.is_cuda
        or not selected_blocks.is_cuda
        or not selected_counts.is_cuda
        or not query_start_loc.is_cuda
        or not seq_lens.is_cuda
    ):
        return None
    device = block_table.device
    if (
        selected_blocks.device != device
        or selected_counts.device != device
        or query_start_loc.device != device
        or seq_lens.device != device
    ):
        return None
    if block_table.dim() != 2 or selected_blocks.dim() != 2:
        return None
    if selected_counts.dim() != 1 or query_start_loc.dim() != 1 or seq_lens.dim() != 1:
        return None
    if (
        num_actual_tokens < 0
        or active_seq_count < 0
        or num_requests < 0
        or table_width <= 0
        or max_q_len < 0
        or chunk_size <= 0
        or recent_window_pages < 0
    ):
        return None
    if int(selected_blocks.shape[0]) < num_actual_tokens:
        return None
    if int(selected_counts.shape[0]) < num_actual_tokens:
        return None
    if int(query_start_loc.shape[0]) < active_seq_count + 1:
        return None
    if int(seq_lens.shape[0]) < active_seq_count:
        return None

    page_table = torch.empty(
        num_requests,
        table_width,
        device=device,
        dtype=torch.int32,
    )
    seqused_k = torch.empty(num_requests, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.empty(num_requests + 1, device=device, dtype=torch.int32)
    if num_requests == 0 or active_seq_count == 0:
        return page_table, seqused_k, cu_seqlens_q
    if max_q_len <= 0:
        return None

    max_active_seqs = triton.next_power_of_2(max(active_seq_count, 1))
    _dsa_batched_unified_page_table_kernel[
        (
            active_seq_count,
            triton.cdiv(max_q_len, _UNIFIED_PAGE_TABLE_BLOCK_ROWS),
            triton.cdiv(table_width, _UNIFIED_PAGE_TABLE_BLOCK_WIDTH),
        )
    ](
        block_table,
        selected_blocks,
        selected_counts,
        query_start_loc,
        seq_lens,
        page_table,
        seqused_k,
        cu_seqlens_q,
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(selected_blocks.stride(0)),
        int(selected_blocks.stride(1)),
        int(selected_counts.stride(0)),
        int(query_start_loc.stride(0)),
        int(seq_lens.stride(0)),
        int(page_table.stride(0)),
        int(page_table.stride(1)),
        active_seq_count,
        num_actual_tokens,
        num_requests,
        int(selected_blocks.shape[1]),
        table_width,
        int(block_table.shape[1]),
        chunk_size,
        dense_decode_threshold,
        dense_prefill_threshold,
        recent_window_pages,
        MAX_ACTIVE_SEQS=max_active_seqs,
        BLOCK_ROWS=_UNIFIED_PAGE_TABLE_BLOCK_ROWS,
        BLOCK_WIDTH=_UNIFIED_PAGE_TABLE_BLOCK_WIDTH,
        num_warps=4,
        num_stages=2,
    )
    return page_table, seqused_k, cu_seqlens_q
