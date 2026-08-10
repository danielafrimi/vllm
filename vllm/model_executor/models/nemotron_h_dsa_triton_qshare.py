# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Triton mean-Q sampler for Nemotron-H Q-share."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

try:
    from vllm.triton_utils import HAS_TRITON, tl, triton
except ImportError:
    HAS_TRITON = False
    tl = None
    triton = None


@dataclass(frozen=True, slots=True)
class EfficientQShareState:
    """Sampled Q rows and reusable packed-row relationship metadata."""

    sampled_q: torch.Tensor
    original_query_start_loc: torch.Tensor
    original_query_start_loc_cpu: torch.Tensor
    sampled_query_start_loc: torch.Tensor
    sampled_query_start_loc_cpu: torch.Tensor
    sampled_query_lengths: torch.Tensor
    sampled_to_sequence: torch.Tensor
    original_to_sampled: torch.Tensor
    sampled_to_original_start: torch.Tensor
    sampled_run_lengths: torch.Tensor
    absolute_position_aligned: bool = False


@dataclass(frozen=True, slots=True)
class EfficientIdentityQShareState:
    """Identity sampling result for the ordinary one-Q-per-row path."""

    sampled_q: torch.Tensor
    original_query_start_loc: torch.Tensor
    original_query_start_loc_cpu: torch.Tensor
    sampled_query_start_loc: torch.Tensor
    sampled_query_start_loc_cpu: torch.Tensor
    metadata: None = None


if HAS_TRITON and triton is not None and tl is not None:

    @triton.jit
    def _qshare_score_metadata_kernel(
        original_query_start_loc,
        sampled_query_start_loc,
        seq_lens,
        sampled_to_sequence,
        sampled_to_original_start,
        score_row_seq_ids,
        row_seq_ids,
        row_group_ids,
        row_num_prior_chunks,
        row_current_chunks,
        row_tail_lens,
        num_original_rows,
        active_seq_count,
        representative_group_idx,
        chunk_size,
        dense_decode_threshold,
        dense_prefill_threshold,
        MAX_ACTIVE_SEQS: tl.constexpr,
    ):
        row = tl.program_id(0)
        seq_idx = tl.load(sampled_to_sequence + row).to(tl.int32)
        seq_offsets = tl.arange(0, MAX_ACTIVE_SEQS)
        seq_mask = seq_offsets < active_seq_count
        original_starts = tl.load(
            original_query_start_loc + seq_offsets,
            mask=seq_mask,
            other=num_original_rows,
        )
        original_ends = tl.load(
            original_query_start_loc + seq_offsets + 1,
            mask=seq_mask,
            other=num_original_rows,
        )
        original_starts = tl.minimum(original_starts, num_original_rows)
        original_ends = tl.minimum(original_ends, num_original_rows)
        original_lengths = original_ends - original_starts
        key_lens = tl.load(seq_lens + seq_offsets, mask=seq_mask, other=0)
        query_position_starts = key_lens - original_lengths
        prior_chunks = tl.maximum(tl.cdiv(key_lens, chunk_size) - 1, 0)
        dense_thresholds = tl.where(
            original_lengths > 1,
            dense_prefill_threshold,
            dense_decode_threshold,
        )
        sparse = (
            seq_mask
            & (original_lengths > 0)
            & (key_lens > 0)
            & (query_position_starts >= 0)
            & (prior_chunks > 0)
            & ~((dense_thresholds >= 0) & (key_lens <= dense_thresholds))
        )
        sparse_ranks = tl.cumsum(sparse.to(tl.int32), axis=0) - 1
        this_seq = seq_offsets == seq_idx
        score_seq_idx = tl.sum(tl.where(this_seq, sparse_ranks, 0), axis=0)
        original_sequence_start = tl.sum(tl.where(this_seq, original_starts, 0), axis=0)
        query_position_start = tl.sum(
            tl.where(this_seq, query_position_starts, 0), axis=0
        )
        row_prior_chunks = tl.sum(tl.where(this_seq, prior_chunks, 0), axis=0)
        row_is_sparse = tl.sum(tl.where(this_seq, sparse.to(tl.int32), 0), axis=0) != 0
        original_start = tl.load(sampled_to_original_start + row)
        position = query_position_start + original_start - original_sequence_start
        current_chunk = position // chunk_size
        tail_len = position - current_chunk * chunk_size + 1

        tl.store(score_row_seq_ids + row, tl.where(row_is_sparse, score_seq_idx, 0))
        tl.store(row_seq_ids + row, seq_idx)
        tl.store(row_group_ids + row, representative_group_idx)
        tl.store(
            row_num_prior_chunks + row,
            tl.where(row_is_sparse, row_prior_chunks, 0),
        )
        tl.store(row_current_chunks + row, current_chunk)
        tl.store(row_tail_lens + row, tail_len)

    @triton.jit
    def _qshare_batched_page_table_kernel(
        block_table,
        selected_blocks,
        selected_counts,
        original_query_start_loc,
        sampled_query_start_loc,
        seq_lens,
        sampled_to_original_start,
        sampled_run_lengths,
        page_table,
        seqused_k,
        cu_seqlens_q,
        block_table_stride_seq,
        block_table_stride_page,
        selected_blocks_stride_row,
        selected_blocks_stride_col,
        page_table_stride_row,
        page_table_stride_col,
        active_seq_count,
        num_original_rows,
        num_requests,
        top_width,
        table_width,
        block_table_width,
        chunk_size,
        dense_decode_threshold,
        dense_prefill_threshold,
        recent_window_pages,
        qshare_group_size,
        MAX_ACTIVE_SEQS: tl.constexpr,
        BLOCK_ROWS: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        row_block = tl.program_id(1)
        tile = tl.program_id(2)
        seq_offsets = tl.arange(0, MAX_ACTIVE_SEQS)
        seq_mask = seq_offsets < active_seq_count

        original_starts = tl.load(
            original_query_start_loc + seq_offsets,
            mask=seq_mask,
            other=num_original_rows,
        )
        original_ends = tl.load(
            original_query_start_loc + seq_offsets + 1,
            mask=seq_mask,
            other=num_original_rows,
        )
        original_starts = tl.minimum(original_starts, num_original_rows)
        original_ends = tl.minimum(original_ends, num_original_rows)
        original_lengths = original_ends - original_starts
        sampled_starts = tl.load(
            sampled_query_start_loc + seq_offsets,
            mask=seq_mask,
            other=0,
        )
        sampled_ends = tl.load(
            sampled_query_start_loc + seq_offsets + 1,
            mask=seq_mask,
            other=0,
        )
        sampled_lengths = sampled_ends - sampled_starts
        key_lens = tl.load(seq_lens + seq_offsets, mask=seq_mask, other=0)
        query_position_starts = key_lens - original_lengths
        dense_thresholds = tl.where(
            original_lengths > 1,
            dense_prefill_threshold,
            dense_decode_threshold,
        )
        active = (
            seq_mask
            & (original_lengths > 0)
            & (key_lens > 0)
            & (query_position_starts >= 0)
        )
        dense_prefix_lengths = tl.minimum(
            tl.maximum(dense_thresholds - query_position_starts, 0),
            original_lengths,
        )
        dense = active & (dense_prefix_lengths > 0)
        sparse_lengths = original_lengths - dense_prefix_lengths
        sparse = active & (sparse_lengths > 0)
        start_residues = query_position_starts % qshare_group_size
        dense_sampled_lengths = tl.where(
            dense_prefix_lengths == original_lengths,
            sampled_lengths,
            tl.where(
                dense,
                tl.cdiv(dense_prefix_lengths + start_residues, qshare_group_size),
                0,
            ),
        )
        sparse_sampled_lengths = sampled_lengths - dense_sampled_lengths
        request_counts = tl.where(
            active,
            dense.to(tl.int32) + sparse_sampled_lengths,
            0,
        )
        request_starts = tl.cumsum(request_counts, axis=0) - request_counts

        this_seq = seq_offsets == seq_idx
        original_start = tl.sum(tl.where(this_seq, original_starts, 0), axis=0)
        sampled_start = tl.sum(tl.where(this_seq, sampled_starts, 0), axis=0)
        query_position_start = tl.sum(
            tl.where(this_seq, query_position_starts, 0), axis=0
        )
        dense_prefix_length = tl.sum(
            tl.where(this_seq, dense_prefix_lengths, 0), axis=0
        )
        dense_sampled_length = tl.sum(
            tl.where(this_seq, dense_sampled_lengths, 0), axis=0
        )
        sparse_sampled_length = tl.sum(
            tl.where(this_seq, sparse_sampled_lengths, 0), axis=0
        )
        request_start = tl.sum(tl.where(this_seq, request_starts, 0), axis=0)
        is_dense = tl.sum(tl.where(this_seq, dense.to(tl.int32), 0), axis=0) != 0
        is_sparse = tl.sum(tl.where(this_seq, sparse.to(tl.int32), 0), axis=0) != 0

        row_offsets = row_block * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        page_offsets = tile * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        dense_request_count = is_dense.to(tl.int32)
        sparse_offsets = row_offsets - dense_request_count
        request_rows = request_start + row_offsets
        sampled_rows = sampled_start + dense_sampled_length + sparse_offsets
        dense_row = is_dense & (row_offsets == 0) & (row_block == 0)
        sparse_row = (
            is_sparse & (sparse_offsets >= 0) & (sparse_offsets < sparse_sampled_length)
        )
        active_row = (dense_row | sparse_row) & (request_rows < num_requests)
        page_ptrs = (
            page_table
            + request_rows[:, None] * page_table_stride_row
            + page_offsets[None, :] * page_table_stride_col
        )
        in_table = page_offsets[None, :] < table_width
        tl.store(
            page_ptrs,
            tl.zeros((BLOCK_ROWS, BLOCK_WIDTH), dtype=tl.int32),
            mask=active_row[:, None] & in_table,
        )

        dense_key_len = query_position_start + dense_prefix_length
        dense_pages = tl.cdiv(dense_key_len, chunk_size)
        dense_mask = (
            dense_row[:, None]
            & in_table
            & (page_offsets[None, :] < dense_pages)
            & (page_offsets[None, :] < block_table_width)
        )
        dense_physical = tl.load(
            block_table
            + seq_idx * block_table_stride_seq
            + page_offsets[None, :] * block_table_stride_page,
            mask=dense_mask,
            other=0,
        )
        tl.store(page_ptrs, dense_physical, mask=dense_mask)

        sampled_original_start = tl.load(
            sampled_to_original_start + sampled_rows,
            mask=sparse_row,
            other=original_start,
        )
        run_length = tl.load(
            sampled_run_lengths + sampled_rows,
            mask=sparse_row,
            other=1,
        )
        position_start = query_position_start + sampled_original_start - original_start
        position_end = position_start + run_length - 1
        current_chunk = position_start // chunk_size
        end_chunk = position_end // chunk_size
        local_page_count = end_chunk - current_chunk + 1
        selected_count = tl.load(
            selected_counts + sampled_rows,
            mask=sparse_row,
            other=0,
        ).to(tl.int64)
        selected_count = tl.minimum(tl.maximum(selected_count, 0), top_width)
        recent_count = tl.minimum(tl.maximum(current_chunk, 0), recent_window_pages).to(
            tl.int64
        )
        recent_start = current_chunk - recent_count

        selected_mask = (
            sparse_row[:, None]
            & in_table
            & (page_offsets[None, :] < selected_count[:, None])
        )
        logical_selected = tl.load(
            selected_blocks
            + sampled_rows[:, None] * selected_blocks_stride_row
            + page_offsets[None, :] * selected_blocks_stride_col,
            mask=selected_mask,
            other=0,
        ).to(tl.int64)
        selected_mask &= (
            (logical_selected >= 0)
            & (logical_selected < recent_start[:, None])
            & (logical_selected < block_table_width)
        )
        selected_physical = tl.load(
            block_table
            + seq_idx * block_table_stride_seq
            + logical_selected * block_table_stride_page,
            mask=selected_mask,
            other=0,
        )
        tl.store(page_ptrs, selected_physical, mask=selected_mask)

        recent_offsets = page_offsets[None, :] - selected_count[:, None]
        recent_logical = recent_start[:, None] + recent_offsets
        recent_mask = (
            sparse_row[:, None]
            & in_table
            & (recent_offsets >= 0)
            & (recent_offsets < recent_count[:, None])
            & (recent_logical >= 0)
            & (recent_logical < block_table_width)
        )
        recent_physical = tl.load(
            block_table
            + seq_idx * block_table_stride_seq
            + recent_logical * block_table_stride_page,
            mask=recent_mask,
            other=0,
        )
        tl.store(page_ptrs, recent_physical, mask=recent_mask)

        local_offsets = (
            page_offsets[None, :] - selected_count[:, None] - recent_count[:, None]
        )
        local_mask = (
            sparse_row[:, None]
            & in_table
            & (local_offsets >= 0)
            & (local_offsets < local_page_count[:, None])
            & (current_chunk[:, None] + local_offsets < block_table_width)
        )
        local_physical = tl.load(
            block_table
            + seq_idx * block_table_stride_seq
            + (current_chunk[:, None] + local_offsets) * block_table_stride_page,
            mask=local_mask,
            other=0,
        )
        tl.store(page_ptrs, local_physical, mask=local_mask)

        metadata_tile = tile == 0
        dense_metadata = metadata_tile & dense_row & (request_start < num_requests)
        tl.store(seqused_k + request_rows, dense_key_len, mask=dense_metadata)
        tl.store(cu_seqlens_q + request_rows, original_start, mask=dense_metadata)
        tl.store(
            cu_seqlens_q + request_rows + 1,
            original_start + dense_prefix_length,
            mask=dense_metadata,
        )
        sparse_metadata = metadata_tile & sparse_row & (request_rows < num_requests)
        sparse_seqused = (
            (selected_count + recent_count).to(tl.int32) * chunk_size
            + position_end
            - current_chunk * chunk_size
            + 1
        )
        tl.store(seqused_k + request_rows, sparse_seqused, mask=sparse_metadata)
        tl.store(
            cu_seqlens_q + request_rows,
            sampled_original_start,
            mask=sparse_metadata,
        )
        tl.store(
            cu_seqlens_q + request_rows + 1,
            sampled_original_start + run_length,
            mask=sparse_metadata,
        )

    @triton.jit
    def _qshare_sequence_metadata_kernel(
        query_start_loc,
        query_position_starts,
        sampled_query_start_loc,
        sampled_query_lengths,
        num_sequences,
        ABSOLUTE_POSITION_ALIGNED: tl.constexpr,
        QSHARE: tl.constexpr,
        BLOCK_SEQUENCES: tl.constexpr,
    ):
        sequences = tl.arange(0, BLOCK_SEQUENCES)
        valid = sequences < num_sequences
        starts = tl.load(
            query_start_loc + sequences,
            mask=valid,
            other=0,
        )
        ends = tl.load(
            query_start_loc + sequences + 1,
            mask=valid,
            other=0,
        )
        lengths = ends - starts
        if ABSOLUTE_POSITION_ALIGNED:
            query_position_start = tl.load(
                query_position_starts + sequences,
                mask=valid,
                other=0,
            )
            start_residue = query_position_start % QSHARE
            sampled_lengths = tl.cdiv(lengths + start_residue, QSHARE)
        else:
            sampled_lengths = tl.cdiv(lengths, QSHARE)
        sampled_starts = tl.cumsum(sampled_lengths, axis=0) - sampled_lengths
        tl.store(
            sampled_query_lengths + sequences,
            sampled_lengths,
            mask=valid,
        )
        tl.store(
            sampled_query_start_loc + sequences,
            sampled_starts,
            mask=valid,
        )
        tl.store(
            sampled_query_start_loc + num_sequences,
            tl.sum(sampled_lengths, axis=0),
        )

    @triton.jit(
        do_not_specialize=[
            "q_stride_row",
            "q_stride_head",
            "q_stride_dim",
            "out_stride_row",
            "out_stride_head",
            "out_stride_dim",
            "num_heads",
            "head_dim",
        ]
    )
    def _mean_qshare_kernel(
        q,
        query_start_loc,
        query_position_starts,
        sampled_query_start_loc,
        sampled_to_sequence,
        sampled_q,
        original_to_sampled,
        sampled_to_original_start,
        sampled_run_lengths,
        q_stride_row,
        q_stride_head,
        q_stride_dim,
        out_stride_row,
        out_stride_head,
        out_stride_dim,
        num_heads,
        head_dim,
        num_sequences,
        ABSOLUTE_POSITION_ALIGNED: tl.constexpr,
        QSHARE: tl.constexpr,
        BLOCK_FEATURES: tl.constexpr,
        BLOCK_SEQUENCES: tl.constexpr,
    ):
        sampled_row = tl.program_id(0)
        sequences = tl.arange(0, BLOCK_SEQUENCES)
        valid_sequences = sequences < num_sequences
        sampled_sequence_ends = tl.load(
            sampled_query_start_loc + sequences + 1,
            mask=valid_sequences,
            other=sampled_row + 1,
        )
        sequence = tl.sum(
            (sampled_row >= sampled_sequence_ends).to(tl.int32),
            axis=0,
        )
        tl.store(sampled_to_sequence + sampled_row, sequence)
        sampled_sequence_start = tl.load(sampled_query_start_loc + sequence)
        local_sampled_row = sampled_row - sampled_sequence_start
        original_sequence_start = tl.load(query_start_loc + sequence)
        original_sequence_end = tl.load(query_start_loc + sequence + 1)
        if ABSOLUTE_POSITION_ALIGNED:
            query_position_start = tl.load(query_position_starts + sequence)
            start_residue = query_position_start % QSHARE
            local_original_start = tl.maximum(
                local_sampled_row * QSHARE - start_residue,
                0,
            )
            original_start = original_sequence_start + local_original_start
        else:
            original_start = original_sequence_start + local_sampled_row * QSHARE
        run_length = tl.minimum(QSHARE, original_sequence_end - original_start)
        if ABSOLUTE_POSITION_ALIGNED:
            next_local_start = (local_sampled_row + 1) * QSHARE - start_residue
            next_local_start = tl.maximum(next_local_start, 0)
            run_length = tl.minimum(
                run_length,
                next_local_start - local_original_start,
            )

        features = tl.arange(0, BLOCK_FEATURES)
        feature_count = num_heads * head_dim
        feature_mask = features < feature_count
        heads = features // head_dim
        dims = features - heads * head_dim
        accumulator = tl.zeros((BLOCK_FEATURES,), dtype=tl.float32)
        for offset in tl.static_range(0, QSHARE):
            row_valid = offset < run_length
            q_offsets = (
                (original_start + offset) * q_stride_row
                + heads * q_stride_head
                + dims * q_stride_dim
            )
            values = tl.load(
                q + q_offsets,
                mask=row_valid & feature_mask,
                other=0.0,
            )
            accumulator += values.to(tl.float32)

        accumulator /= run_length.to(tl.float32)
        output_offsets = (
            sampled_row * out_stride_row
            + heads * out_stride_head
            + dims * out_stride_dim
        )
        tl.store(
            sampled_q + output_offsets,
            accumulator,
            mask=feature_mask,
        )
        tl.store(sampled_to_original_start + sampled_row, original_start)
        tl.store(sampled_run_lengths + sampled_row, run_length)
        for offset in tl.static_range(0, QSHARE):
            tl.store(
                original_to_sampled + original_start + offset,
                sampled_row,
                mask=offset < run_length,
            )


class EfficientMeanQShareProvider(nn.Module):
    """Sample packed Q rows without host reads or data-dependent launches."""

    def __init__(self, *, group_size: int) -> None:
        super().__init__()
        if group_size <= 0:
            raise ValueError(f"group_size must be positive: {group_size}")
        self.group_size = group_size

    def forward(
        self,
        *,
        projected_q: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        total_sampled_rows: int,
        query_position_starts: torch.Tensor | None = None,
        query_position_starts_cpu: torch.Tensor | None = None,
    ) -> EfficientQShareState:
        num_sequences = query_start_loc.shape[0] - 1
        absolute_position_aligned = query_position_starts is not None
        if absolute_position_aligned != (query_position_starts_cpu is not None):
            raise ValueError(
                "query position starts must be provided on both CPU and device"
            )
        if absolute_position_aligned:
            assert query_position_starts is not None
            assert query_position_starts_cpu is not None
            if (
                query_position_starts.dim() != 1
                or query_position_starts_cpu.dim() != 1
                or int(query_position_starts.shape[0]) != num_sequences
                or int(query_position_starts_cpu.shape[0]) != num_sequences
                or query_position_starts.device != query_start_loc.device
            ):
                raise ValueError(
                    "query position starts must match the packed sequence count"
                )
        else:
            # The kernels keep their original slice-relative specialization when
            # neither dynamic recall nor a recent window is enabled.
            query_position_starts = query_start_loc
            query_position_starts_cpu = query_start_loc_cpu[:-1]
        sampled_query_lengths = torch.empty(
            num_sequences,
            device=query_start_loc.device,
            dtype=query_start_loc.dtype,
        )
        sampled_query_start_loc = torch.empty_like(query_start_loc)
        block_sequences = triton.next_power_of_2(num_sequences)
        _qshare_sequence_metadata_kernel[(1,)](
            query_start_loc,
            query_position_starts,
            sampled_query_start_loc,
            sampled_query_lengths,
            num_sequences,
            ABSOLUTE_POSITION_ALIGNED=absolute_position_aligned,
            QSHARE=self.group_size,
            BLOCK_SEQUENCES=block_sequences,
            num_warps=1,
        )
        query_lengths_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        assert query_position_starts_cpu is not None
        start_residues_cpu = (
            query_position_starts_cpu % self.group_size
            if absolute_position_aligned
            else torch.zeros_like(query_lengths_cpu)
        )
        sampled_query_lengths_cpu = torch.div(
            query_lengths_cpu + start_residues_cpu + self.group_size - 1,
            self.group_size,
            rounding_mode="floor",
        )
        sampled_query_start_loc_cpu = torch.cat(
            (
                sampled_query_lengths_cpu.new_zeros(1),
                sampled_query_lengths_cpu.cumsum(dim=0),
            )
        )
        expected_sampled_rows = int(sampled_query_start_loc_cpu[-1])
        if total_sampled_rows != expected_sampled_rows:
            raise ValueError(
                "total sampled rows does not match the Q-share run layout: "
                f"expected {expected_sampled_rows}, got {total_sampled_rows}"
            )
        sampled_to_sequence = torch.empty(
            total_sampled_rows,
            device=query_start_loc.device,
            dtype=query_start_loc.dtype,
        )
        sampled_q = torch.empty(
            (
                total_sampled_rows,
                projected_q.shape[1],
                projected_q.shape[2],
            ),
            device=projected_q.device,
            dtype=projected_q.dtype,
        )
        original_to_sampled = torch.empty(
            projected_q.shape[0],
            device=projected_q.device,
            dtype=query_start_loc.dtype,
        )
        sampled_to_original_start = torch.empty(
            total_sampled_rows,
            device=projected_q.device,
            dtype=query_start_loc.dtype,
        )
        sampled_run_lengths = torch.empty_like(sampled_to_original_start)
        feature_count = projected_q.shape[1] * projected_q.shape[2]
        block_features = triton.next_power_of_2(feature_count)
        _mean_qshare_kernel[(total_sampled_rows,)](
            projected_q,
            query_start_loc,
            query_position_starts,
            sampled_query_start_loc,
            sampled_to_sequence,
            sampled_q,
            original_to_sampled,
            sampled_to_original_start,
            sampled_run_lengths,
            projected_q.stride(0),
            projected_q.stride(1),
            projected_q.stride(2),
            sampled_q.stride(0),
            sampled_q.stride(1),
            sampled_q.stride(2),
            projected_q.shape[1],
            projected_q.shape[2],
            num_sequences,
            ABSOLUTE_POSITION_ALIGNED=absolute_position_aligned,
            QSHARE=self.group_size,
            BLOCK_FEATURES=block_features,
            BLOCK_SEQUENCES=block_sequences,
            num_warps=4,
        )
        return EfficientQShareState(
            sampled_q=sampled_q,
            original_query_start_loc=query_start_loc,
            original_query_start_loc_cpu=query_start_loc_cpu,
            sampled_query_start_loc=sampled_query_start_loc,
            sampled_query_start_loc_cpu=sampled_query_start_loc_cpu,
            sampled_query_lengths=sampled_query_lengths,
            sampled_to_sequence=sampled_to_sequence,
            original_to_sampled=original_to_sampled,
            sampled_to_original_start=sampled_to_original_start,
            sampled_run_lengths=sampled_run_lengths,
            absolute_position_aligned=absolute_position_aligned,
        )


class EfficientIdentityQShareProvider(nn.Module):
    """Preserve the original query structure without allocating metadata."""

    def forward(
        self,
        *,
        projected_q: torch.Tensor,
        query_start_loc: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        total_sampled_rows: int,
    ) -> EfficientIdentityQShareState:
        del total_sampled_rows
        return EfficientIdentityQShareState(
            sampled_q=projected_q,
            original_query_start_loc=query_start_loc,
            original_query_start_loc_cpu=query_start_loc_cpu,
            sampled_query_start_loc=query_start_loc,
            sampled_query_start_loc_cpu=query_start_loc_cpu,
        )


def qshare_score_metadata_triton(
    *,
    state: EfficientQShareState,
    seq_lens: torch.Tensor,
    active_seq_count: int,
    representative_group_idx: int,
    chunk_size: int,
    dense_decode_threshold: int,
    dense_prefill_threshold: int,
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
    if not HAS_TRITON or triton is None or tl is None:
        return None
    total_rows = int(state.sampled_q.shape[0])
    device = state.sampled_q.device
    outputs = tuple(
        torch.empty(total_rows, device=device, dtype=torch.int32) for _ in range(6)
    )
    if total_rows == 0:
        return outputs
    max_active_seqs = triton.next_power_of_2(max(active_seq_count, 1))
    _qshare_score_metadata_kernel[(total_rows,)](
        state.original_query_start_loc,
        state.sampled_query_start_loc,
        seq_lens,
        state.sampled_to_sequence,
        state.sampled_to_original_start,
        *outputs,
        int(state.original_query_start_loc_cpu[-1]),
        active_seq_count,
        representative_group_idx,
        chunk_size,
        dense_decode_threshold,
        dense_prefill_threshold,
        MAX_ACTIVE_SEQS=max_active_seqs,
        num_warps=1,
    )
    return outputs


def qshare_batched_page_table_triton(
    *,
    block_table: torch.Tensor,
    selected_blocks: torch.Tensor,
    selected_counts: torch.Tensor,
    state: EfficientQShareState,
    seq_lens: torch.Tensor,
    active_seq_count: int,
    num_requests: int,
    table_width: int,
    max_sampled_q_len: int,
    chunk_size: int,
    dense_decode_threshold: int,
    dense_prefill_threshold: int,
    recent_window_pages: int = 0,
    qshare_group_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if not HAS_TRITON or triton is None or tl is None:
        return None
    if recent_window_pages < 0 or qshare_group_size <= 0:
        return None
    device = block_table.device
    page_table = torch.empty(
        num_requests, table_width, device=device, dtype=torch.int32
    )
    seqused_k = torch.empty(num_requests, device=device, dtype=torch.int32)
    cu_seqlens_q = torch.empty(num_requests + 1, device=device, dtype=torch.int32)
    if num_requests == 0 or active_seq_count == 0:
        return page_table, seqused_k, cu_seqlens_q
    max_active_seqs = triton.next_power_of_2(max(active_seq_count, 1))
    block_rows = 8
    block_width = 128
    _qshare_batched_page_table_kernel[
        (
            active_seq_count,
            triton.cdiv(max_sampled_q_len, block_rows),
            triton.cdiv(table_width, block_width),
        )
    ](
        block_table,
        selected_blocks,
        selected_counts,
        state.original_query_start_loc,
        state.sampled_query_start_loc,
        seq_lens,
        state.sampled_to_original_start,
        state.sampled_run_lengths,
        page_table,
        seqused_k,
        cu_seqlens_q,
        block_table.stride(0),
        block_table.stride(1),
        selected_blocks.stride(0),
        selected_blocks.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        active_seq_count,
        int(state.original_query_start_loc_cpu[-1]),
        num_requests,
        int(selected_blocks.shape[1]),
        table_width,
        int(block_table.shape[1]),
        chunk_size,
        dense_decode_threshold,
        dense_prefill_threshold,
        recent_window_pages,
        qshare_group_size,
        MAX_ACTIVE_SEQS=max_active_seqs,
        BLOCK_ROWS=block_rows,
        BLOCK_WIDTH=block_width,
        num_warps=4,
        num_stages=2,
    )
    return page_table, seqused_k, cu_seqlens_q
