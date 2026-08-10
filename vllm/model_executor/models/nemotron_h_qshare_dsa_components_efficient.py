# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Efficient mean-Q-share providers for Nemotron-H chunked DSA."""

from __future__ import annotations

import math
import os
import typing

import torch

from vllm.model_executor.models.nemotron_h_chunked_dsa_components_efficient import (
    EfficientChunkedDSAProviderBundle,
    _dsa_cudagraph_runtime_active,
    _dsa_log_path_marker,
    _EfficientBatchedChunkBlockSelections,
    _EfficientChunkBlockSelection,
    _PhysicalPageChunkRepresentatives,
    _TritonBatchedChunkRepresentatives,
)
from vllm.model_executor.models.nemotron_h_dsa_query_providers import (
    MeanQShareProvider,
    SelectionQueryState,
)
from vllm.model_executor.models.nemotron_h_dsa_recall_policy import (
    DynamicRecallPolicyProvider,
    RecallPolicyProvider,
    log_recall_plan,
)
from vllm.model_executor.models.nemotron_h_dsa_triton_qshare import (
    EfficientIdentityQShareProvider,
    EfficientIdentityQShareState,
    EfficientMeanQShareProvider,
    EfficientQShareState,
    qshare_batched_page_table_triton,
    qshare_score_metadata_triton,
)
from vllm.model_executor.models.nemotron_h_dsa_triton_scoring import (
    dsa_batched_score_topk_tile_plan_triton,
    dsa_build_fixed_decode_score_tile_plan_triton,
    dsa_build_score_metadata_triton,
    dsa_build_score_tile_plan_triton,
    dsa_cudagraph_keepalive,
    dsa_score_tile_plan_config,
)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _absolute_qshare_sampled_length(
    *, query_position_start: int, query_len: int, group_size: int
) -> int:
    """Return the number of absolute-position-aligned Q-share runs."""
    if query_position_start < 0 or query_len < 0 or group_size <= 0:
        raise ValueError("invalid absolute Q-share run dimensions")
    if query_len == 0:
        return 0
    return (query_len + query_position_start % group_size + group_size - 1) // (
        group_size
    )


def _absolute_qshare_run_start(
    *, query_position_start: int, sampled_row: int, group_size: int
) -> int:
    """Return a local query-row start aligned to an absolute Q-share bucket."""
    return max(sampled_row * group_size - query_position_start % group_size, 0)


def _absolute_qshare_top_k_segments(
    *,
    policy: RecallPolicyProvider,
    query_position_start: int,
    query_len: int,
    group_size: int,
    sampled_row_start: int,
    maximum_top_k: int,
) -> list[tuple[int, int, int]]:
    """Build constant-K sampled-row segments without device reads."""
    segments: list[tuple[int, int, int]] = []
    query_segments = policy.top_k_segments(
        query_position_start=query_position_start,
        query_len=query_len,
        maximum_top_k=maximum_top_k,
    )
    for query_start, query_end, top_k in query_segments:
        if query_start > 0 and (query_position_start + query_start) % group_size:
            raise ValueError("a Q-share run crosses a recall-policy boundary")
        sampled_start = _absolute_qshare_sampled_length(
            query_position_start=query_position_start,
            query_len=query_start,
            group_size=group_size,
        )
        sampled_end = _absolute_qshare_sampled_length(
            query_position_start=query_position_start,
            query_len=query_end,
            group_size=group_size,
        )
        if sampled_end <= sampled_start:
            continue
        global_start = sampled_row_start + sampled_start
        global_end = sampled_row_start + sampled_end
        if segments and segments[-1][2] == top_k:
            segment_start, _, _ = segments[-1]
            segments[-1] = (segment_start, global_end, top_k)
        else:
            segments.append((global_start, global_end, top_k))
    return segments


class _IdentityQSharePlanProvider:
    def __init__(self) -> None:
        self.query_provider = EfficientIdentityQShareProvider()

    def build_sequence_state(self, **_: typing.Any) -> None:
        return None

    def build_single_sequence_state(self, **_: typing.Any) -> None:
        return None

    def select_blocks_batched(
        self,
        bundle: EfficientQShareChunkedDSAProviderBundle,
        **kwargs: typing.Any,
    ) -> dict[int, typing.Any | None] | None:
        return EfficientChunkedDSAProviderBundle.try_select_blocks_batched(
            bundle, **kwargs
        )

    def build_page_tables_batched(
        self,
        bundle: EfficientQShareChunkedDSAProviderBundle,
        **kwargs: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
        return EfficientChunkedDSAProviderBundle.try_build_page_tables_batched(
            bundle, **kwargs
        )

    def expand_selection_state(
        self,
        *,
        selection_state: typing.Any | None,
        **_: typing.Any,
    ) -> typing.Any | None:
        return selection_state

    def selection_query_chunk_size(
        self,
        bundle: EfficientQShareChunkedDSAProviderBundle,
        q_len: int,
    ) -> int:
        return min(bundle.q_indexer_chunked_query_chunk_size, q_len)


class _MeanQSharePlanProvider:
    def __init__(self, *, group_size: int) -> None:
        self.group_size = group_size
        self.query_provider = EfficientMeanQShareProvider(group_size=group_size)
        self.single_query_provider = MeanQShareProvider(group_size=group_size)
        self._fixed_decode_score_plans: dict[
            tuple[torch.device, int, int, int, int], torch.Tensor
        ] = {}

    def build_single_sequence_state(
        self,
        *,
        score_query_states: torch.Tensor,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
        chunk_size: int,
    ) -> SelectionQueryState:
        return self.single_query_provider(
            projected_q=score_query_states,
            current_chunks=current_chunks,
            query_positions=query_positions,
            chunk_size=chunk_size,
        )

    def build_sequence_state(
        self,
        *,
        selection_query_batch: EfficientQShareState,
        seq_idx: int,
        q_start: int,
        q_end: int,
        current_chunks: torch.Tensor,
        **_: typing.Any,
    ) -> SelectionQueryState:
        sampled_start = int(selection_query_batch.sampled_query_start_loc_cpu[seq_idx])
        sampled_end = int(
            selection_query_batch.sampled_query_start_loc_cpu[seq_idx + 1]
        )
        original_start = int(
            selection_query_batch.original_query_start_loc_cpu[seq_idx]
        )
        global_run_starts = selection_query_batch.sampled_to_original_start[
            sampled_start:sampled_end
        ]
        local_run_starts = global_run_starts - original_start
        return SelectionQueryState(
            reduced_q=selection_query_batch.sampled_q[sampled_start:sampled_end, 0],
            reduced_current_chunks=current_chunks.index_select(0, local_run_starts),
            run_starts=local_run_starts,
            run_counts=selection_query_batch.sampled_run_lengths[
                sampled_start:sampled_end
            ],
            query_row_to_reduced_row=(
                selection_query_batch.original_to_sampled[q_start:q_end] - sampled_start
            ),
        )

    def _select_blocks_cached_decode(
        self,
        bundle: EfficientQShareChunkedDSAProviderBundle,
        *,
        selection_query_batch: EfficientQShareState,
        sparse_infos: list[
            tuple[int, int, int, int, int, int, torch.Tensor | None]
        ],
        batched_chunk_representatives: _TritonBatchedChunkRepresentatives,
        representatives: _PhysicalPageChunkRepresentatives,
        representative_group_idx: int,
        seq_lens: torch.Tensor,
        active_seq_count: int,
        dense_decode_threshold: int,
        dense_prefill_threshold: int,
    ) -> _EfficientBatchedChunkBlockSelections | None:
        """Select fixed-K decode pages without rebuilding host-side plans."""
        if bundle.q_indexer_dynamic_chunk_top_k or active_seq_count <= 0:
            return None
        total_rows = int(selection_query_batch.sampled_q.shape[0])
        if (
            total_rows != active_seq_count
            or len(sparse_infos) != active_seq_count
            or int(selection_query_batch.original_query_start_loc_cpu[-1])
            != active_seq_count
            or int(selection_query_batch.sampled_query_start_loc_cpu[-1])
            != active_seq_count
        ):
            return None
        for sparse_idx, (seq_idx, q_start, q_end, *_) in enumerate(sparse_infos):
            expected_local_idx = (
                seq_idx
                if batched_chunk_representatives._seq_id_layout == "original"
                else sparse_idx
            )
            if (
                q_end - q_start != 1
                or batched_chunk_representatives._local_by_seq.get(seq_idx)
                != expected_local_idx
            ):
                return None

        physical_block_table = representatives._block_table
        max_prior_chunks = int(physical_block_table.shape[1])
        chunk_top_k = min(bundle.q_indexer_chunk_top_k, max_prior_chunks)
        if max_prior_chunks <= 0 or chunk_top_k <= 0:
            return None
        (
            small_block_rows,
            large_block_rows,
            block_chunks,
            decode_block_chunks,
        ) = dsa_score_tile_plan_config()
        cache_key = (
            selection_query_batch.sampled_q.device,
            total_rows,
            max_prior_chunks,
            representative_group_idx,
            decode_block_chunks,
        )
        tile_plan = self._fixed_decode_score_plans.get(cache_key)
        if tile_plan is None:
            tile_plan = dsa_build_fixed_decode_score_tile_plan_triton(
                total_rows=total_rows,
                max_prior_chunks=max_prior_chunks,
                representative_group_idx=representative_group_idx,
                device=selection_query_batch.sampled_q.device,
                decode_block_chunks=decode_block_chunks,
            )
            if tile_plan is None:
                return None
            self._fixed_decode_score_plans[cache_key] = tile_plan
        dsa_cudagraph_keepalive(tile_plan)

        row_metadata = qshare_score_metadata_triton(
            state=selection_query_batch,
            seq_lens=seq_lens,
            active_seq_count=active_seq_count,
            representative_group_idx=representative_group_idx,
            chunk_size=bundle.chunk_size,
            dense_decode_threshold=dense_decode_threshold,
            dense_prefill_threshold=dense_prefill_threshold,
        )
        if row_metadata is None:
            return None
        dsa_cudagraph_keepalive(row_metadata)
        (
            _score_row_seq_ids,
            row_seq_ids,
            _row_group_ids,
            row_num_prior_chunks,
            row_current_chunks,
            row_tail_lens,
        ) = row_metadata
        score_current_chunks = bundle._dsa_remote_current_chunks(row_current_chunks)
        dsa_cudagraph_keepalive(score_current_chunks)
        score_topk = dsa_batched_score_topk_tile_plan_triton(
            score_query_states=selection_query_batch.sampled_q[:, 0],
            chunk_representatives=representatives,
            tile_plan=tile_plan,
            current_chunks=score_current_chunks,
            row_num_prior_chunks=row_num_prior_chunks,
            total_rows=total_rows,
            chunk_size=bundle.chunk_size,
            chunk_top_k=chunk_top_k,
            logit_scale=bundle.q_indexer_logit_scale,
            q_indexer_dim=bundle.q_indexer_dim,
            max_prior_chunks=max_prior_chunks,
            small_block_rows=small_block_rows,
            large_block_rows=large_block_rows,
            block_chunks=block_chunks,
            decode_block_chunks=decode_block_chunks,
        )
        if score_topk is None:
            return None
        dsa_cudagraph_keepalive(score_topk)
        selected_blocks, selected_counts, _ = score_topk

        selection_by_seq: dict[int, typing.Any | None] = {}
        seq_slices: dict[int, tuple[int, int, int]] = {}
        chunk_top_k_by_seq: dict[int, int] = {}
        for row, (seq_idx, *_rest) in enumerate(sparse_infos):
            seq_slices[seq_idx] = (row, row + 1, chunk_top_k)
            chunk_top_k_by_seq[seq_idx] = chunk_top_k
            selection_by_seq[seq_idx] = _EfficientChunkBlockSelection(
                selected_block_indices=selected_blocks[row : row + 1],
                selected_block_counts=selected_counts[row : row + 1],
            )
        _dsa_log_path_marker(
            "triton_cached_decode_fixed_plan",
            rows=total_rows,
            tiles=int(tile_plan.shape[0]),
            top_k=chunk_top_k,
        )
        return _EfficientBatchedChunkBlockSelections(
            selected_block_indices=selected_blocks,
            selected_block_valid=None,
            selected_block_counts=selected_counts,
            seq_slices=seq_slices,
            chunk_top_k_by_seq=chunk_top_k_by_seq,
            row_seq_ids=row_seq_ids,
            row_current_chunks=row_current_chunks,
            row_tail_lens=row_tail_lens,
            per_seq=selection_by_seq,
            fixed_decode_plan=True,
        )

    def select_blocks_batched(
        self,
        bundle: EfficientQShareChunkedDSAProviderBundle,
        *,
        selection_query_batch: EfficientQShareState,
        sparse_infos: list[tuple[int, int, int, int, int, int, torch.Tensor | None]],
        batched_chunk_representatives: typing.Any | None,
        block_table: torch.Tensor,
        representative_group_idx: int = 0,
        seq_lens: torch.Tensor | None = None,
        active_seq_count: int | None = None,
        dense_decode_threshold: int | None = None,
        dense_prefill_threshold: int | None = None,
        **_: typing.Any,
    ) -> dict[int, typing.Any | None] | None:
        if (
            not isinstance(
                batched_chunk_representatives,
                _TritonBatchedChunkRepresentatives,
            )
            or seq_lens is None
            or active_seq_count is None
            or dense_decode_threshold is None
            or dense_prefill_threshold is None
            or block_table.dim() != 2
        ):
            return None

        feature_on = (
            bundle.q_indexer_dynamic_chunk_top_k
            or bundle.q_indexer_recent_window_pages > 0
        )
        if feature_on:
            # Feature-on Q-share decisions must describe a single recency and
            # recall-policy band. The sampler aligns runs to absolute positions;
            # unsupported group/page configurations fall through to the exact
            # policy/page-tiled sequence path.
            if not selection_query_batch.absolute_position_aligned:
                log_recall_plan(
                    "batched_qshare_fallback",
                    reason="sampler_not_absolute_position_aligned",
                    group_size=self.group_size,
                )
                return None
            if (
                bundle.q_indexer_recent_window_pages > 0
                and bundle.chunk_size % self.group_size != 0
            ):
                log_recall_plan(
                    "batched_qshare_fallback",
                    reason="group_does_not_divide_page",
                    group_size=self.group_size,
                    chunk_size=bundle.chunk_size,
                )
                return None
            if bundle.q_indexer_dynamic_chunk_top_k:
                policy = bundle.recall_policy
                if not isinstance(policy, DynamicRecallPolicyProvider):
                    return None
                if (
                    policy.dynamic_step_tokens % self.group_size != 0
                    or policy.dynamic_dense_tokens % self.group_size != 0
                ):
                    log_recall_plan(
                        "batched_qshare_fallback",
                        reason="group_does_not_divide_policy_band",
                        group_size=self.group_size,
                        step_tokens=policy.dynamic_step_tokens,
                        dense_tokens=policy.dynamic_dense_tokens,
                    )
                    return None

        representatives = batched_chunk_representatives._representatives
        if (
            representatives.dim() != 4
            or representative_group_idx < 0
            or representative_group_idx >= int(representatives.shape[2])
            or int(representatives.shape[3]) != bundle.q_indexer_dim
        ):
            return None
        if isinstance(representatives, _PhysicalPageChunkRepresentatives):
            cached_decode = self._select_blocks_cached_decode(
                bundle,
                selection_query_batch=selection_query_batch,
                sparse_infos=sparse_infos,
                batched_chunk_representatives=batched_chunk_representatives,
                representatives=representatives,
                representative_group_idx=representative_group_idx,
                seq_lens=seq_lens,
                active_seq_count=active_seq_count,
                dense_decode_threshold=dense_decode_threshold,
                dense_prefill_threshold=dense_prefill_threshold,
            )
            if cached_decode is not None:
                return cached_decode

        selection_by_seq: dict[int, typing.Any | None] = {}
        seq_slices: dict[int, tuple[int, int, int]] = {}
        chunk_top_k_by_seq: dict[int, int] = {}
        top_k_segments: list[tuple[int, int, int]] = []
        min_context_len: int | None = None
        max_context_len = 0
        min_requested_top_k: int | None = None
        max_prior_chunks = 0
        max_top_k = 0
        max_sampled_q_len = 0
        sparse_rows = 0
        total_tiles = 0
        max_tiles_per_row_plan = 0
        (
            small_block_rows,
            large_block_rows,
            block_chunks,
            decode_block_chunks,
        ) = dsa_score_tile_plan_config()
        for sparse_idx, (
            seq_idx,
            _q_start,
            _q_end,
            key_len,
            num_chunks,
            query_position_start,
            _current_chunks,
        ) in enumerate(sparse_infos):
            local_idx = batched_chunk_representatives._local_by_seq.get(seq_idx)
            expected_local_idx = (
                seq_idx
                if batched_chunk_representatives._seq_id_layout == "original"
                else sparse_idx
            )
            if local_idx != expected_local_idx:
                return None
            if (
                batched_chunk_representatives._num_chunks_by_seq.get(seq_idx)
                != num_chunks
            ):
                return None
            sampled_start = int(
                selection_query_batch.sampled_query_start_loc_cpu[seq_idx]
            )
            sampled_end = int(
                selection_query_batch.sampled_query_start_loc_cpu[seq_idx + 1]
            )
            sampled_len = sampled_end - sampled_start
            prior_chunks = max(num_chunks - 1, 0)
            seq_top_k_segments: list[tuple[int, int, int]] = []
            if bundle.q_indexer_dynamic_chunk_top_k:
                try:
                    seq_top_k_segments = _absolute_qshare_top_k_segments(
                        policy=bundle.recall_policy,
                        query_position_start=query_position_start,
                        query_len=key_len - query_position_start,
                        group_size=self.group_size,
                        sampled_row_start=sampled_start,
                        maximum_top_k=prior_chunks,
                    )
                except ValueError:
                    log_recall_plan(
                        "batched_qshare_fallback",
                        reason="run_crosses_policy_boundary",
                        group_size=self.group_size,
                        seq_idx=seq_idx,
                    )
                    return None
                if not seq_top_k_segments or seq_top_k_segments[-1][1] != sampled_end:
                    return None
                chunk_top_k = max(
                    (top_k for _, _, top_k in seq_top_k_segments),
                    default=0,
                )
            else:
                chunk_top_k = min(
                    bundle.q_indexer_chunk_top_k,
                    prior_chunks,
                )
            if prior_chunks <= 0:
                selection_by_seq[seq_idx] = None
                continue
            if bundle.q_indexer_dynamic_chunk_top_k:
                for row_start, row_end, segment_top_k in seq_top_k_segments:
                    if segment_top_k <= 0:
                        continue
                    top_k_segments.append((row_start, row_end, segment_top_k))
                    min_requested_top_k = (
                        segment_top_k
                        if min_requested_top_k is None
                        else min(min_requested_top_k, segment_top_k)
                    )
            min_context_len = (
                query_position_start + 1
                if min_context_len is None
                else min(min_context_len, query_position_start + 1)
            )
            max_context_len = max(max_context_len, key_len)
            seq_slices[seq_idx] = (sampled_start, sampled_end, chunk_top_k)
            chunk_top_k_by_seq[seq_idx] = chunk_top_k
            sparse_rows += sampled_len
            if sampled_len == 1:
                tile_count = math.ceil(prior_chunks / decode_block_chunks)
            elif sampled_len <= small_block_rows:
                tile_count = math.ceil(sampled_len / small_block_rows) * math.ceil(
                    prior_chunks / block_chunks
                )
            else:
                tile_count = math.ceil(sampled_len / large_block_rows) * math.ceil(
                    prior_chunks / block_chunks
                )
            total_tiles += tile_count
            max_tiles_per_row_plan = max(max_tiles_per_row_plan, tile_count)
            max_sampled_q_len = max(max_sampled_q_len, sampled_len)
            max_prior_chunks = max(max_prior_chunks, prior_chunks)
            max_top_k = max(max_top_k, chunk_top_k)

        if max_prior_chunks <= 0:
            return selection_by_seq
        if max_top_k <= 0:
            return None if feature_on else selection_by_seq
        score_plan = dsa_build_score_metadata_triton(
            query_start_loc=selection_query_batch.original_query_start_loc,
            row_query_start_loc=selection_query_batch.sampled_query_start_loc,
            seq_lens=seq_lens,
            num_actual_tokens=int(
                selection_query_batch.original_query_start_loc_cpu[-1]
            ),
            active_seq_count=active_seq_count,
            num_sparse_plans=len(seq_slices),
            total_rows=int(selection_query_batch.sampled_q.shape[0]),
            chunk_size=bundle.chunk_size,
            representative_group_idx=representative_group_idx,
            dense_decode_threshold=dense_decode_threshold,
            dense_prefill_threshold=dense_prefill_threshold,
            chunk_top_k=(max_top_k if feature_on else bundle.q_indexer_chunk_top_k),
            max_q_len=max_sampled_q_len,
            representatives_use_original_seq_ids=(
                batched_chunk_representatives._seq_id_layout == "original"
            ),
            small_block_rows=small_block_rows,
            large_block_rows=large_block_rows,
            block_chunks=block_chunks,
            decode_block_chunks=decode_block_chunks,
        )
        if score_plan is None:
            return None
        row_plan, _ = score_plan
        dsa_cudagraph_keepalive(score_plan)
        tile_plan = dsa_build_score_tile_plan_triton(
            row_plan_with_tiles=row_plan,
            total_tiles=total_tiles,
            max_tiles_per_row_plan=max_tiles_per_row_plan,
            small_block_rows=small_block_rows,
            large_block_rows=large_block_rows,
            block_chunks=block_chunks,
            decode_block_chunks=decode_block_chunks,
        )
        if tile_plan is None:
            return None
        dsa_cudagraph_keepalive(tile_plan)
        row_metadata = qshare_score_metadata_triton(
            state=selection_query_batch,
            seq_lens=seq_lens,
            active_seq_count=active_seq_count,
            representative_group_idx=representative_group_idx,
            chunk_size=bundle.chunk_size,
            dense_decode_threshold=dense_decode_threshold,
            dense_prefill_threshold=dense_prefill_threshold,
        )
        if row_metadata is None:
            return None
        dsa_cudagraph_keepalive(row_metadata)
        (
            _score_row_seq_ids,
            row_seq_ids,
            _row_group_ids,
            row_num_prior_chunks,
            row_current_chunks,
            row_tail_lens,
        ) = row_metadata
        score_current_chunks = row_current_chunks
        if feature_on:
            score_current_chunks = bundle._dsa_remote_current_chunks(row_current_chunks)
        dsa_cudagraph_keepalive(score_current_chunks)
        row_top_k = None
        score_top_k_segments = None
        if bundle.q_indexer_dynamic_chunk_top_k:
            row_top_k = torch.zeros_like(row_current_chunks, dtype=torch.int32)
            for segment_start, segment_end, segment_top_k in top_k_segments:
                row_top_k[segment_start:segment_end].fill_(segment_top_k)
            score_top_k_segments = top_k_segments
        score_topk = dsa_batched_score_topk_tile_plan_triton(
            score_query_states=selection_query_batch.sampled_q[:, 0],
            chunk_representatives=representatives,
            tile_plan=tile_plan,
            current_chunks=score_current_chunks,
            row_num_prior_chunks=row_num_prior_chunks,
            total_rows=int(selection_query_batch.sampled_q.shape[0]),
            chunk_size=bundle.chunk_size,
            chunk_top_k=max_top_k,
            logit_scale=bundle.q_indexer_logit_scale,
            q_indexer_dim=bundle.q_indexer_dim,
            max_prior_chunks=max_prior_chunks,
            small_block_rows=small_block_rows,
            large_block_rows=large_block_rows,
            block_chunks=block_chunks,
            decode_block_chunks=decode_block_chunks,
            row_top_k=row_top_k,
            top_k_segments=score_top_k_segments,
        )
        log_recall_plan(
            "batched_qshare_selection",
            context_min=min_context_len,
            context_max=max_context_len,
            remote_top_k_min=(
                min_requested_top_k
                if bundle.q_indexer_dynamic_chunk_top_k
                else max_top_k
            ),
            remote_top_k_max=max_top_k,
            recent_window_pages=bundle.q_indexer_recent_window_pages,
            policy_segments=len(top_k_segments),
            rows=sparse_rows,
            absolute_position_aligned=selection_query_batch.absolute_position_aligned,
        )
        if score_topk is None:
            return None
        dsa_cudagraph_keepalive(score_topk)
        selected_blocks, selected_counts, _ = score_topk
        for seq_idx, (start, end, chunk_top_k) in seq_slices.items():
            selection_by_seq[seq_idx] = _EfficientChunkBlockSelection(
                selected_block_indices=selected_blocks[start:end, :chunk_top_k],
                selected_block_counts=selected_counts[start:end],
            )
        _dsa_log_path_marker(
            "triton_batched_qshare_scoring",
            rows=int(selected_blocks.shape[0]),
            sparse_rows=sparse_rows,
            sparse_seqs=len(seq_slices),
            top_k=max_top_k,
            absolute_position_aligned=selection_query_batch.absolute_position_aligned,
            recent_window_pages=bundle.q_indexer_recent_window_pages,
        )
        return _EfficientBatchedChunkBlockSelections(
            selected_block_indices=selected_blocks,
            selected_block_valid=None,
            selected_block_counts=selected_counts,
            seq_slices=seq_slices,
            chunk_top_k_by_seq=chunk_top_k_by_seq,
            row_seq_ids=row_seq_ids,
            row_current_chunks=row_current_chunks,
            row_tail_lens=row_tail_lens,
            per_seq=selection_by_seq,
        )

    def build_page_tables_batched(
        self,
        bundle: EfficientQShareChunkedDSAProviderBundle,
        *,
        block_table: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]] | None,
        block_selection_by_seq: dict[int, typing.Any | None],
        selection_query_batch: EfficientQShareState,
        seq_lens: torch.Tensor | None,
        active_seq_count: int | None,
        dense_decode_threshold: int | None,
        dense_prefill_threshold: int | None,
        **_: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
        if (
            active_seq_infos is None
            or seq_lens is None
            or active_seq_count is None
            or dense_decode_threshold is None
            or dense_prefill_threshold is None
            or not isinstance(
                block_selection_by_seq,
                _EfficientBatchedChunkBlockSelections,
            )
        ):
            return None
        selected_blocks = block_selection_by_seq._selected_block_indices
        selected_counts = block_selection_by_seq._selected_block_counts
        if selected_counts is None:
            return None

        top_width = int(selected_blocks.shape[1])
        recent_window_pages = bundle.q_indexer_recent_window_pages
        if block_selection_by_seq._fixed_decode_plan:
            if (
                active_seq_count <= 0
                or len(block_selection_by_seq._seq_slices) != active_seq_count
                or int(selection_query_batch.original_query_start_loc_cpu[-1])
                != active_seq_count
                or int(selection_query_batch.sampled_query_start_loc_cpu[-1])
                != active_seq_count
            ):
                return None
            max_local_pages = (
                self.group_size + bundle.chunk_size - 2
            ) // bundle.chunk_size + 1
            table_width = top_width + recent_window_pages + max_local_pages
            plan = qshare_batched_page_table_triton(
                block_table=block_table,
                selected_blocks=selected_blocks,
                selected_counts=selected_counts,
                state=selection_query_batch,
                seq_lens=seq_lens,
                active_seq_count=active_seq_count,
                num_requests=active_seq_count,
                table_width=table_width,
                max_sampled_q_len=1,
                chunk_size=bundle.chunk_size,
                dense_decode_threshold=dense_decode_threshold,
                dense_prefill_threshold=dense_prefill_threshold,
                recent_window_pages=recent_window_pages,
                qshare_group_size=self.group_size,
            )
            if plan is None:
                return None
            dsa_cudagraph_keepalive(plan)
            page_table, seqused_k, _ = plan
            # DP+EP may replay a decode graph captured for more requests than
            # this replica currently owns. The page-table kernel intentionally
            # skips those zero-length rows, so its generated query offsets keep
            # stale capture-time values. Feed FA4 the live GPU offsets instead:
            # repeated trailing offsets represent padded requests with q_len=0.
            cu_seqlens_q = selection_query_batch.original_query_start_loc[
                : active_seq_count + 1
            ]
            if cu_seqlens_q.dtype != torch.int32:
                cu_seqlens_q = cu_seqlens_q.to(dtype=torch.int32)
            dsa_cudagraph_keepalive(cu_seqlens_q)
            _dsa_log_path_marker(
                "triton_cached_decode_page_table",
                rows=active_seq_count,
                sparse_top_k=top_width,
                recent_window_pages=recent_window_pages,
            )
            return (
                page_table,
                cu_seqlens_q,
                seqused_k,
                1,
                table_width * bundle.chunk_size,
            )

        num_requests = 0
        max_dense_pages = 0
        max_sampled_q_len = 1
        max_seqlen_q = 0
        max_seqlen_k = 0
        dense_requests = 0
        sparse_requests = 0
        crossing_requests = 0
        max_local_pages = (
            self.group_size + bundle.chunk_size - 2
        ) // bundle.chunk_size + 1
        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            q_len = q_end - q_start
            dense_threshold = (
                dense_prefill_threshold if q_len > 1 else dense_decode_threshold
            )
            query_position_start = key_len - q_len
            dense_prefix_len = (
                min(q_len, max(dense_threshold - query_position_start, 0))
                if dense_threshold >= 0
                else 0
            )
            if 0 < dense_prefix_len < q_len:
                if (
                    not selection_query_batch.absolute_position_aligned
                    or dense_threshold % self.group_size
                ):
                    return None
                crossing_requests += 1
                log_recall_plan(
                    "dense_sparse_boundary_split",
                    dense_tokens=dense_threshold,
                    context_start=query_position_start + 1,
                    context_end=key_len,
                    dense_rows=dense_prefix_len,
                    sparse_rows=q_len - dense_prefix_len,
                    first_sparse_context=dense_threshold + 1,
                    first_sparse_top_k=bundle._dsa_chunk_top_k_for_context(
                        dense_threshold + 1
                    ),
                    recent_window_pages=recent_window_pages,
                    qshare_group_size=self.group_size,
                    backend="triton_batched_qshare_page_table",
                )
            if dense_prefix_len > 0:
                dense_key_len = query_position_start + dense_prefix_len
                dense_pages = math.ceil(dense_key_len / bundle.chunk_size)
                num_requests += 1
                dense_requests += 1
                max_dense_pages = max(max_dense_pages, dense_pages)
                max_seqlen_q = max(max_seqlen_q, dense_prefix_len)
                max_seqlen_k = max(max_seqlen_k, dense_key_len)

            sampled_start = int(
                selection_query_batch.sampled_query_start_loc_cpu[seq_idx]
            )
            sampled_end = int(
                selection_query_batch.sampled_query_start_loc_cpu[seq_idx + 1]
            )
            sampled_len = sampled_end - sampled_start
            if dense_prefix_len == q_len:
                dense_sampled_len = sampled_len
            elif dense_prefix_len > 0:
                dense_sampled_len = _absolute_qshare_sampled_length(
                    query_position_start=query_position_start,
                    query_len=dense_prefix_len,
                    group_size=self.group_size,
                )
            else:
                dense_sampled_len = 0
            sparse_sampled_len = sampled_len - dense_sampled_len
            if sparse_sampled_len < 0:
                return None
            if sparse_sampled_len > 0:
                num_requests += sparse_sampled_len
                sparse_requests += sparse_sampled_len
                max_sampled_q_len = max(max_sampled_q_len, sparse_sampled_len)
                max_seqlen_q = max(
                    max_seqlen_q,
                    min(q_len - dense_prefix_len, self.group_size),
                )
                max_seqlen_k = max(
                    max_seqlen_k,
                    (top_width + recent_window_pages + max_local_pages)
                    * bundle.chunk_size,
                )
            max_sampled_q_len = max(
                max_sampled_q_len,
                sparse_sampled_len + int(dense_prefix_len > 0),
            )
        table_width = max(
            top_width + recent_window_pages + max_local_pages,
            max_dense_pages,
        )
        if num_requests <= 0 or table_width <= 0:
            return None
        plan = qshare_batched_page_table_triton(
            block_table=block_table,
            selected_blocks=selected_blocks,
            selected_counts=selected_counts,
            state=selection_query_batch,
            seq_lens=seq_lens,
            active_seq_count=active_seq_count,
            num_requests=num_requests,
            table_width=table_width,
            max_sampled_q_len=max_sampled_q_len,
            chunk_size=bundle.chunk_size,
            dense_decode_threshold=dense_decode_threshold,
            dense_prefill_threshold=dense_prefill_threshold,
            recent_window_pages=recent_window_pages,
            qshare_group_size=self.group_size,
        )
        if plan is None:
            return None
        dsa_cudagraph_keepalive(plan)
        page_table, seqused_k, cu_seqlens_q = plan
        _dsa_log_path_marker(
            "triton_batched_qshare_page_table",
            rows=num_requests,
            seqs=len(active_seq_infos),
            sparse_top_k=top_width,
            recent_window_pages=recent_window_pages,
        )
        if dense_requests:
            _dsa_log_path_marker(
                "dense_prefill_page_table_bucket",
                dense_requests=dense_requests,
            )
        if sparse_requests:
            _dsa_log_path_marker(
                "sparse_prefill_page_table_bucket",
                sparse_requests=sparse_requests,
            )
        if crossing_requests:
            _dsa_log_path_marker(
                "dense_sparse_prefill_page_table_bucket",
                crossings=crossing_requests,
            )
        if sparse_requests and max_seqlen_q == 1:
            _dsa_log_path_marker("sparse_decode", decode_requests=sparse_requests)
        return (
            page_table,
            cu_seqlens_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
        )

    def expand_selection_state(
        self,
        *,
        bundle: EfficientQShareChunkedDSAProviderBundle,
        selection_state: typing.Any | None,
        selection_query_state: SelectionQueryState | None,
    ) -> typing.Any | None:
        if selection_query_state is None or selection_state is None:
            return selection_state
        selected_blocks, selected_valid = bundle.get_selected_blocks(
            selection_state,
            device=selection_query_state.query_row_to_reduced_row.device,
        )
        mapping = selection_query_state.query_row_to_reduced_row
        return _EfficientChunkBlockSelection(
            selected_block_indices=selected_blocks.index_select(0, mapping),
            selected_block_valid=selected_valid.index_select(0, mapping),
        )

    def selection_query_chunk_size(
        self,
        bundle: EfficientQShareChunkedDSAProviderBundle,
        q_len: int,
    ) -> int:
        del bundle
        return q_len


class EfficientQShareChunkedDSAProviderBundle(EfficientChunkedDSAProviderBundle):
    """Efficient DSA bundle with a uniform configurable query sampler."""

    def __init__(
        self,
        *,
        qshare_group_size: int | None = None,
        chunk_size: int,
        **kwargs: typing.Any,
    ) -> None:
        super().__init__(chunk_size=chunk_size, **kwargs)
        if qshare_group_size is None:
            qshare_group_size = _env_int("VLLM_NEMOTRON_H_DSA_QSHARE_GROUP_SIZE", 1)
        if qshare_group_size <= 0 or qshare_group_size & (qshare_group_size - 1):
            raise ValueError(
                "qshare_group_size must be a positive power of two: "
                f"{qshare_group_size}"
            )
        self.qshare_group_size = qshare_group_size
        self.qshare_enabled = qshare_group_size > 1
        if qshare_group_size == 1:
            self.qshare_plan_provider = _IdentityQSharePlanProvider()
        else:
            self.qshare_plan_provider = _MeanQSharePlanProvider(
                group_size=qshare_group_size
            )
        self.query_provider = self.qshare_plan_provider.query_provider
        self._decode_identity_rows: dict[
            tuple[torch.device, torch.dtype, torch.device, int],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = {}
        _dsa_log_path_marker(
            "config",
            provider=type(self).__name__,
            qshare_group_size=qshare_group_size,
            qshare_mode="mean",
        )

    def _forward_dsa_chunked_one_kv_head_page_table_fa_bucket(
        self,
        *,
        block_table: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        **kwargs: typing.Any,
    ) -> set[int] | None:
        if (
            self.batched_representative_provider.rep_cache_cudagraph
            and _dsa_cudagraph_runtime_active()
        ):
            # Full-decode capture uses one-token dummy host metadata. Size the
            # graph from the static block-table capacity instead; selector and
            # page-table kernels still read the real sequence lengths from
            # their runtime CUDA buffers, so padded pages remain inaccessible.
            graph_key_len = int(block_table.shape[1]) * self.chunk_size
            active_seq_infos = [
                (seq_idx, q_start, q_end, graph_key_len)
                for seq_idx, q_start, q_end, _ in active_seq_infos
            ]
        return super()._forward_dsa_chunked_one_kv_head_page_table_fa_bucket(
            block_table=block_table,
            active_seq_infos=active_seq_infos,
            **kwargs,
        )

    def _prepare_decode_identity_query_batch(
        self,
        *,
        score_query_states: torch.Tensor,
        gpu_starts: torch.Tensor,
        cpu_starts: torch.Tensor,
        active_seq_count: int,
    ) -> EfficientQShareState:
        """Represent single-token decode as Q-share runs of length one.

        Mean Q-sharing is the identity when every active sequence contributes
        one query row. Reusing the original Q and query offsets avoids the
        sampled-Q allocation plus the sequence-metadata and mean-Q kernels.
        The remaining fixed-decode selector and page-table kernels consume the
        same state contract as the ordinary mean-Q path.
        """
        cache_key = (
            score_query_states.device,
            gpu_starts.dtype,
            cpu_starts.device,
            active_seq_count,
        )
        identity_rows = self._decode_identity_rows.get(cache_key)
        if identity_rows is None:
            row_ids = torch.arange(
                active_seq_count,
                device=score_query_states.device,
                dtype=gpu_starts.dtype,
            )
            run_lengths = torch.ones_like(row_ids)
            sampled_cpu_starts = torch.arange(
                active_seq_count + 1,
                device=cpu_starts.device,
                dtype=torch.int64,
            )
            identity_rows = (row_ids, run_lengths, sampled_cpu_starts)
            self._decode_identity_rows[cache_key] = identity_rows
        row_ids, run_lengths, sampled_cpu_starts = identity_rows
        return EfficientQShareState(
            sampled_q=score_query_states,
            original_query_start_loc=gpu_starts,
            original_query_start_loc_cpu=cpu_starts,
            sampled_query_start_loc=gpu_starts,
            sampled_query_start_loc_cpu=sampled_cpu_starts,
            sampled_query_lengths=run_lengths,
            sampled_to_sequence=row_ids,
            original_to_sampled=row_ids,
            sampled_to_original_start=gpu_starts[:-1],
            sampled_run_lengths=run_lengths,
            absolute_position_aligned=True,
        )

    def prepare_selection_query_batch(
        self,
        *,
        score_query_states: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        query_start_loc_cpu: torch.Tensor | None,
        active_seq_count: int,
        active_seq_infos: list[tuple[int, int, int, int]] | None = None,
    ) -> EfficientQShareState | EfficientIdentityQShareState:
        gpu_starts = typing.cast(torch.Tensor, query_start_loc)[: active_seq_count + 1]
        cpu_starts = typing.cast(torch.Tensor, query_start_loc_cpu)[
            : active_seq_count + 1
        ]
        cpu_lengths = cpu_starts[1:] - cpu_starts[:-1]
        all_single_token_decode = (
            self.qshare_enabled
            and int(score_query_states.shape[0]) == active_seq_count
            and int(cpu_starts[-1]) == active_seq_count
            and (
                active_seq_infos is None
                or (
                    len(active_seq_infos) == active_seq_count
                    and all(
                        q_end - q_start == 1
                        for _, q_start, q_end, _ in active_seq_infos
                    )
                )
            )
            and bool(torch.all(cpu_lengths == 1))
        )
        if all_single_token_decode:
            _dsa_log_path_marker(
                "qshare_decode_identity",
                rows=active_seq_count,
            )
            return self._prepare_decode_identity_query_batch(
                score_query_states=score_query_states,
                gpu_starts=gpu_starts,
                cpu_starts=cpu_starts,
                active_seq_count=active_seq_count,
            )
        request_absolute_alignment = self.qshare_enabled and (
            self.q_indexer_dynamic_chunk_top_k or self.q_indexer_recent_window_pages > 0
        )
        align_absolute_positions = False
        query_position_starts_cpu = None
        query_position_starts = None
        if request_absolute_alignment:
            position_starts = []
            expected_q_start = 0
            metadata_is_compact = (
                active_seq_infos is not None
                and len(active_seq_infos) == active_seq_count
            )
            if active_seq_infos is not None:
                for active_idx, (seq_idx, q_start, q_end, key_len) in enumerate(
                    active_seq_infos
                ):
                    q_len = q_end - q_start
                    row_is_valid = (
                        seq_idx == active_idx
                        and q_start == expected_q_start
                        and q_len > 0
                        and key_len >= q_len
                    )
                    metadata_is_compact &= row_is_valid
                    expected_q_start = q_end
                    position_starts.append(key_len - q_len)
            if metadata_is_compact:
                align_absolute_positions = True
                query_position_starts_cpu = cpu_starts.new_tensor(position_starts)
                query_position_starts = query_position_starts_cpu.to(
                    device=gpu_starts.device
                )
            else:
                log_recall_plan(
                    "batched_qshare_fallback",
                    reason="active_sequence_metadata_not_compact",
                    group_size=self.qshare_group_size,
                    seqs=active_seq_count,
                )
        if align_absolute_positions:
            assert query_position_starts_cpu is not None
            sampled_numerators = (
                cpu_lengths
                + query_position_starts_cpu.remainder(self.qshare_group_size)
                + self.qshare_group_size
                - 1
            )
        else:
            sampled_numerators = cpu_lengths + self.qshare_group_size - 1
        total_sampled_rows = int(
            torch.div(
                sampled_numerators,
                self.qshare_group_size,
                rounding_mode="floor",
            ).sum()
        )
        log_recall_plan(
            "qshare_sampling",
            absolute_position_aligned=align_absolute_positions,
            group_size=self.qshare_group_size,
            rows=int(cpu_lengths.sum()),
            sampled_rows=total_sampled_rows,
            seqs=active_seq_count,
        )
        alignment_kwargs: dict[str, torch.Tensor] = {}
        if query_position_starts is not None:
            assert query_position_starts_cpu is not None
            alignment_kwargs = {
                "query_position_starts": query_position_starts,
                "query_position_starts_cpu": query_position_starts_cpu,
            }
        return self.query_provider(
            projected_q=score_query_states,
            query_start_loc=gpu_starts,
            query_start_loc_cpu=cpu_starts,
            total_sampled_rows=total_sampled_rows,
            **alignment_kwargs,
        )

    def build_selection_query_state(
        self,
        *,
        score_query_states: torch.Tensor,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> SelectionQueryState | None:
        return self.qshare_plan_provider.build_single_sequence_state(
            score_query_states=score_query_states,
            current_chunks=current_chunks,
            query_positions=query_positions,
            chunk_size=self.chunk_size,
        )

    def build_selection_query_state_from_batch(
        self,
        *,
        selection_query_batch: typing.Any,
        **kwargs: typing.Any,
    ) -> SelectionQueryState | None:
        return self.qshare_plan_provider.build_sequence_state(
            selection_query_batch=selection_query_batch,
            **kwargs,
        )

    def try_select_blocks_batched(
        self, **kwargs: typing.Any
    ) -> dict[int, typing.Any | None] | None:
        return self.qshare_plan_provider.select_blocks_batched(self, **kwargs)

    def try_build_page_tables_batched(
        self, **kwargs: typing.Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
        return self.qshare_plan_provider.build_page_tables_batched(self, **kwargs)

    def expand_selection_state(
        self,
        *,
        selection_state: typing.Any | None,
        selection_query_state: typing.Any | None,
    ) -> typing.Any | None:
        return self.qshare_plan_provider.expand_selection_state(
            bundle=self,
            selection_state=selection_state,
            selection_query_state=selection_query_state,
        )

    def selection_query_chunk_size(self, q_len: int) -> int:
        return self.qshare_plan_provider.selection_query_chunk_size(self, q_len)
