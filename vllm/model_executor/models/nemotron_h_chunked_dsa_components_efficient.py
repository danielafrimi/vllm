# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Efficient components for Nemotron-H chunked DSA."""

from __future__ import annotations

import math
import os
import typing

import torch
from torch import nn

from vllm.compilation import monitor as compilation_monitor
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.model_executor.models.nemotron_h_chunked_dsa_components_pytorch import (
    ChunkedDSAAttentionProviderMixin,
    TorchChunkedDSARepresentativeProvider,
    _get_flash_attn_varlen_func,
    normalize_packed_nhd_kv_cache,
)
from vllm.model_executor.models.nemotron_h_dsa_recall_policy import (
    log_recall_plan,
)
from vllm.model_executor.models.nemotron_h_dsa_triton_decode_page_table import (
    dsa_batched_unified_page_table_triton,
    dsa_decode_page_table_triton,
)
from vllm.model_executor.models.nemotron_h_dsa_triton_summaries import (
    dsa_block_summaries_triton,
    dsa_cached_block_summaries_triton,
    dsa_seed_block_summary_cache_triton,
    dsa_update_block_summary_cache_triton,
    dsa_update_current_block_summary_cache_triton,
    dsa_update_written_block_summary_cache_triton,
)

try:
    from vllm.triton_utils import HAS_TRITON, tl, triton
except ImportError:
    HAS_TRITON = False
    tl = None
    triton = None

try:
    from vllm.model_executor.models.nemotron_h_dsa_triton_scoring import (
        dsa_batched_row_metadata_triton,
        dsa_batched_score_topk_tile_plan_triton,
        dsa_build_score_metadata_triton,
        dsa_build_score_tile_plan_triton,
        dsa_cudagraph_keepalive,
        dsa_score_tile_plan_config,
    )
except ImportError:
    dsa_batched_row_metadata_triton = None
    dsa_batched_score_topk_tile_plan_triton = None
    dsa_build_score_metadata_triton = None
    dsa_build_score_tile_plan_triton = None
    dsa_cudagraph_keepalive = None
    dsa_score_tile_plan_config = None


_SequenceSkipFn = typing.Callable[[int, int, int, int], bool]
_DSA_PATH_DEBUG_PRINT_LIMIT_ENV = "VLLM_NEMOTRON_H_DSA_PATH_DEBUG_PRINT_LIMIT"
_DSA_REP_CACHE_ENV = "VLLM_NEMOTRON_H_DSA_USE_REP_CACHE"
_DSA_REP_CACHE_CUDAGRAPH_ENV = "VLLM_NEMOTRON_H_DSA_REP_CACHE_CUDAGRAPH"
_DSA_REP_CACHE_VERIFY_CALLS_ENV = "VLLM_NEMOTRON_H_DSA_REP_CACHE_VERIFY_CALLS"
_DSA_TRITON_SCORING_ENV = "VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING"
_DSA_PATH_DEBUG_COUNTS: dict[str, int] = {}
_DSA_SUMMARY_METADATA_BLOCK_WIDTH = 256


def _dsa_cudagraph_runtime_active() -> bool:
    """Return whether this call is preparing or capturing a CUDA graph."""
    return (
        torch.compiler.is_compiling()
        or (torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())
        or (
            compilation_monitor.cudagraph_capturing_enabled
            and is_forward_context_available()
            and get_forward_context().cudagraph_runtime_mode
            != CUDAGraphMode.NONE
        )
    )


def _nhd_fake_page_pitch(
    cache: torch.Tensor,
    *,
    num_kv_heads: int,
) -> int | None:
    """Return the physical-page stride measured in one-head fake pages."""
    if cache.dim() != 4 or num_kv_heads <= 1:
        return None
    _, block_size, cache_kv_heads, head_dim = cache.shape
    if cache_kv_heads != num_kv_heads or min(cache.shape) <= 0:
        return None
    page_stride, token_stride, head_stride, dim_stride = cache.stride()
    dense_page_size = block_size * num_kv_heads * head_dim
    if (
        (token_stride, head_stride, dim_stride)
        != (num_kv_heads * head_dim, head_dim, 1)
        or page_stride < dense_page_size
        or page_stride % head_dim != 0
    ):
        return None
    return page_stride // head_dim


def _make_nhd_fake_page_view(
    cache: torch.Tensor,
    *,
    num_kv_heads: int,
) -> torch.Tensor | None:
    """Expose NHD storage as one-head pages without moving K/V payload data."""
    if cache.dim() != 4:
        return None
    num_pages, block_size, _, head_dim = cache.shape
    fake_page_pitch = _nhd_fake_page_pitch(cache, num_kv_heads=num_kv_heads)
    if fake_page_pitch is None:
        return None

    fake_num_pages = (num_pages - 1) * fake_page_pitch + num_kv_heads
    fake_stride = (head_dim, num_kv_heads * head_dim, head_dim, 1)
    max_offset = (
        cache.storage_offset()
        + (fake_num_pages - 1) * fake_stride[0]
        + (block_size - 1) * fake_stride[1]
        + head_dim
        - 1
    )
    storage_elems = cache.untyped_storage().nbytes() // cache.element_size()
    if max_offset >= storage_elems:
        return None
    return cache.as_strided(
        (fake_num_pages, block_size, 1, head_dim),
        fake_stride,
    )


def _remap_nhd_block_table_for_kv_head(
    block_table: torch.Tensor,
    *,
    fake_page_pitch: int,
    num_kv_heads: int,
    kv_head_idx: int,
) -> torch.Tensor:
    """Map physical NHD pages to the storage-safe fake-page namespace."""
    if not 0 <= kv_head_idx < num_kv_heads:
        raise ValueError(f"kv_head_idx={kv_head_idx} outside [0, {num_kv_heads})")
    return block_table * fake_page_pitch + kv_head_idx


if HAS_TRITON and triton is not None and tl is not None:

    @triton.jit(
        do_not_specialize=[
            "seq_lens_stride",
            "query_start_loc_stride",
            "block_table_stride_seq",
            "block_table_stride_chunk",
            "out_seq_lens_stride",
            "out_query_lens_stride",
            "out_block_table_stride_seq",
            "out_block_table_stride_chunk",
            "active_row_count",
            "table_width",
            "chunk_size",
            "dense_decode_budget",
            "dense_prefill_budget",
        ]
    )
    def _dsa_summary_metadata_kernel(
        seq_lens,
        query_start_loc,
        block_table,
        out_seq_lens,
        out_query_lens,
        out_block_table,
        seq_lens_stride,
        query_start_loc_stride,
        block_table_stride_seq,
        block_table_stride_chunk,
        out_seq_lens_stride,
        out_query_lens_stride,
        out_block_table_stride_seq,
        out_block_table_stride_chunk,
        active_row_count,
        table_width,
        chunk_size,
        dense_decode_budget,
        dense_prefill_budget,
        BLOCK_SEQS: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        USE_DENSE_SKIP: tl.constexpr,
    ):
        dst_row = tl.program_id(0)
        table_tile = tl.program_id(1)

        seq_offsets = tl.arange(0, BLOCK_SEQS)
        seq_mask = seq_offsets < active_row_count
        q_start = tl.load(
            query_start_loc + seq_offsets * query_start_loc_stride,
            mask=seq_mask,
            other=0,
        ).to(tl.int64)
        q_end = tl.load(
            query_start_loc + (seq_offsets + 1) * query_start_loc_stride,
            mask=seq_mask,
            other=0,
        ).to(tl.int64)
        q_len = q_end - q_start
        key_len = tl.load(
            seq_lens + seq_offsets * seq_lens_stride,
            mask=seq_mask,
            other=0,
        ).to(tl.int64)
        num_chunks = tl.cdiv(key_len, chunk_size)
        live = seq_mask & (q_len > 0) & (key_len > 0) & (num_chunks <= table_width)
        if USE_DENSE_SKIP:
            dense_budget = tl.where(
                q_len > 1,
                dense_prefill_budget,
                dense_decode_budget,
            )
            live = live & (key_len > dense_budget)

        live_rank = tl.cumsum(live.to(tl.int32), 0) - 1
        row_match = live & (live_rank == dst_row)
        found = tl.sum(row_match.to(tl.int32), 0) != 0
        src_row = tl.sum(tl.where(row_match, seq_offsets, 0), 0).to(tl.int64)
        compact_seq_len = tl.sum(tl.where(row_match, key_len, 0), 0)
        compact_query_len = tl.sum(tl.where(row_match, q_len, 0), 0)

        tl.store(
            out_seq_lens + dst_row * out_seq_lens_stride,
            compact_seq_len,
            mask=table_tile == 0,
        )
        tl.store(
            out_query_lens + dst_row * out_query_lens_stride,
            compact_query_len,
            mask=table_tile == 0,
        )

        table_offsets = table_tile * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
        table_mask = table_offsets < table_width
        table_values = tl.load(
            block_table
            + src_row * block_table_stride_seq
            + table_offsets * block_table_stride_chunk,
            mask=found & table_mask,
            other=0,
        ).to(tl.int64)
        tl.store(
            out_block_table
            + dst_row * out_block_table_stride_seq
            + table_offsets * out_block_table_stride_chunk,
            table_values,
            mask=table_mask,
        )


class _UnavailableRepresentatives:
    pass


_UNAVAILABLE = _UnavailableRepresentatives()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value == "1"


def _dsa_log_path_marker(marker: str, **fields: typing.Any) -> None:
    limit = _env_int(_DSA_PATH_DEBUG_PRINT_LIMIT_ENV, 0)
    if limit <= 0:
        return
    count = _DSA_PATH_DEBUG_COUNTS.get(marker, 0)
    if count >= limit:
        return
    _DSA_PATH_DEBUG_COUNTS[marker] = count + 1
    details = " ".join(f"{key}={value}" for key, value in sorted(fields.items()))
    print(
        f"DSA_PATH_MARKER marker={marker} count={count + 1} {details}",
        flush=True,
    )


def _dsa_prepare_summary_metadata_triton(
    *,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    summary_batch: int,
    active_row_count: int,
    table_width: int,
    chunk_size: int,
    use_dense_skip: bool,
    dense_decode_budget: int,
    dense_prefill_budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        not HAS_TRITON
        or triton is None
        or tl is None
        or not seq_lens.is_cuda
        or not query_start_loc.is_cuda
        or not block_table.is_cuda
        or summary_batch <= 0
        or active_row_count <= 0
        or table_width <= 0
    ):
        return None
    if (
        seq_lens.device != block_table.device
        or query_start_loc.device != block_table.device
        or seq_lens.dim() != 1
        or query_start_loc.dim() != 1
        or block_table.dim() != 2
    ):
        return None

    gpu_seq_lens = torch.empty(
        summary_batch,
        device=block_table.device,
        dtype=torch.long,
    )
    gpu_query_lens = torch.empty_like(gpu_seq_lens)
    gpu_block_table = torch.empty(
        summary_batch,
        table_width,
        device=block_table.device,
        dtype=torch.long,
    )
    block_width = _DSA_SUMMARY_METADATA_BLOCK_WIDTH
    block_seqs = triton.next_power_of_2(active_row_count)
    _dsa_summary_metadata_kernel[
        (summary_batch, triton.cdiv(table_width, block_width))
    ](
        seq_lens,
        query_start_loc,
        block_table,
        gpu_seq_lens,
        gpu_query_lens,
        gpu_block_table,
        int(seq_lens.stride(0)),
        int(query_start_loc.stride(0)),
        int(block_table.stride(0)),
        int(block_table.stride(1)),
        int(gpu_seq_lens.stride(0)),
        int(gpu_query_lens.stride(0)),
        int(gpu_block_table.stride(0)),
        int(gpu_block_table.stride(1)),
        active_row_count,
        table_width,
        chunk_size,
        int(dense_decode_budget),
        int(dense_prefill_budget),
        BLOCK_SEQS=block_seqs,
        BLOCK_WIDTH=block_width,
        USE_DENSE_SKIP=bool(use_dense_skip),
        num_warps=8,
    )
    return gpu_block_table, gpu_seq_lens, gpu_query_lens


def _dsa_prepare_summary_metadata_torch(
    *,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    block_table: torch.Tensor,
    summary_batch: int,
    active_row_count: int,
    table_width: int,
    chunk_size: int,
    use_dense_skip: bool,
    dense_decode_budget: int,
    dense_prefill_budget: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gpu_all_seq_lens = seq_lens[:active_row_count]
    if gpu_all_seq_lens.dtype != torch.long:
        gpu_all_seq_lens = gpu_all_seq_lens.to(dtype=torch.long)
    gpu_q_start = query_start_loc[:active_row_count]
    gpu_q_end = query_start_loc[1 : active_row_count + 1]
    if gpu_q_start.dtype != torch.long:
        gpu_q_start = gpu_q_start.to(dtype=torch.long)
    if gpu_q_end.dtype != torch.long:
        gpu_q_end = gpu_q_end.to(dtype=torch.long)
    gpu_q_len = gpu_q_end - gpu_q_start
    gpu_num_chunks = torch.div(
        gpu_all_seq_lens + chunk_size - 1,
        chunk_size,
        rounding_mode="floor",
    )
    gpu_live_mask = (
        (gpu_q_len > 0) & (gpu_all_seq_lens > 0) & (gpu_num_chunks <= table_width)
    )
    if use_dense_skip:
        gpu_dense_budget = torch.where(
            gpu_q_len > 1,
            torch.full_like(gpu_all_seq_lens, int(dense_prefill_budget)),
            torch.full_like(gpu_all_seq_lens, int(dense_decode_budget)),
        )
        gpu_live_mask = gpu_live_mask & (gpu_all_seq_lens > gpu_dense_budget)

    gpu_live_i64 = gpu_live_mask.to(dtype=torch.long)
    gpu_dst = torch.cumsum(gpu_live_i64, dim=0) - 1
    gpu_dst = torch.clamp(gpu_dst, min=0, max=summary_batch - 1)

    gpu_seq_lens = torch.zeros(
        summary_batch,
        device=block_table.device,
        dtype=torch.long,
    )
    gpu_seq_lens.scatter_add_(0, gpu_dst, gpu_all_seq_lens * gpu_live_i64)
    gpu_query_lens = torch.zeros_like(gpu_seq_lens)
    gpu_query_lens.scatter_add_(0, gpu_dst, gpu_q_len * gpu_live_i64)

    gpu_block_src = block_table[:active_row_count]
    if gpu_block_src.dtype != torch.long:
        gpu_block_src = gpu_block_src.to(dtype=torch.long)
    if not gpu_block_src.is_contiguous():
        gpu_block_src = gpu_block_src.contiguous()
    gpu_block_table = torch.zeros(
        summary_batch,
        table_width,
        device=block_table.device,
        dtype=torch.long,
    )
    gpu_block_table.scatter_add_(
        0,
        gpu_dst[:, None].expand(-1, table_width),
        gpu_block_src * gpu_live_i64[:, None],
    )
    return gpu_block_table, gpu_seq_lens, gpu_query_lens


class _PhysicalPageChunkRepresentatives:
    """Tensor-shaped descriptor consumed only by fused Triton scoring.

    The descriptor deliberately does not implement indexing: doing so would
    recreate the logical representative tensor that this decode path removes.
    ``dim`` and ``shape`` retain compatibility with the existing efficient and
    Q-share selector validation before the scoring entry point dispatches on
    ``_is_physical_page_rep_cache``.
    """

    __slots__ = (
        "_assume_historical_valid",
        "_block_size",
        "_block_table",
        "_cache_valid",
        "_cache_values",
        "_shape",
    )
    _is_physical_page_rep_cache = True

    def __init__(
        self,
        *,
        cache_values: torch.Tensor,
        cache_valid: torch.Tensor,
        block_table: torch.Tensor,
        batch: int,
        max_chunks: int,
        block_size: int,
    ) -> None:
        self._cache_values = cache_values
        self._cache_valid = cache_valid
        self._block_table = block_table
        self._block_size = block_size
        self._assume_historical_valid = True
        self._shape = (
            batch,
            max_chunks,
            int(cache_values.shape[1]),
            int(cache_values.shape[2]),
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self._shape

    def dim(self) -> int:
        return 4

    def __getitem__(self, _: typing.Any) -> typing.NoReturn:
        raise RuntimeError(
            "physical-page representatives require batched fused scoring"
        )


class _PhysicalPageSequenceRepresentatives:
    """Metadata-only view used by the generic page-table planner.

    The planner validates each sequence's representative shape before calling
    the batched selector.  Physical-cache decode has no logical representative
    tensor to return, so this object exposes only that shape contract.  Any
    attempt to consume representative values still fails loudly instead of
    silently rebuilding or gathering them.
    """

    __slots__ = ("_shape",)

    def __init__(
        self,
        *,
        num_chunks: int,
        num_kv_heads: int,
        q_indexer_dim: int,
    ) -> None:
        self._shape = (num_chunks, num_kv_heads, q_indexer_dim)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def dim(self) -> int:
        return 3

    def __getitem__(self, _: typing.Any) -> typing.NoReturn:
        raise RuntimeError(
            "physical-page representatives require batched fused scoring"
        )


class _TritonBatchedChunkRepresentatives:
    __slots__ = (
        "_local_by_seq",
        "_num_chunks_by_seq",
        "_representatives",
        "_seq_id_layout",
    )

    def __init__(
        self,
        *,
        representatives: torch.Tensor | _PhysicalPageChunkRepresentatives,
        local_by_seq: dict[int, int],
        num_chunks_by_seq: dict[int, int],
        seq_id_layout: str = "compact",
    ) -> None:
        if seq_id_layout not in ("compact", "original"):
            raise ValueError(f"unexpected representative layout: {seq_id_layout}")
        self._representatives = representatives
        self._local_by_seq = local_by_seq
        self._num_chunks_by_seq = num_chunks_by_seq
        self._seq_id_layout = seq_id_layout

    def __contains__(self, seq_idx: int) -> bool:
        return seq_idx in self._local_by_seq

    def __getitem__(
        self, seq_idx: int
    ) -> torch.Tensor | _PhysicalPageSequenceRepresentatives:
        result = self.get(seq_idx)
        if result is None:
            raise KeyError(seq_idx)
        return result

    def get(
        self,
        seq_idx: int,
        default: torch.Tensor | None = None,
    ) -> torch.Tensor | _PhysicalPageSequenceRepresentatives | None:
        local_idx = self._local_by_seq.get(seq_idx)
        if local_idx is None:
            return default
        num_chunks = self._num_chunks_by_seq[seq_idx]
        if isinstance(self._representatives, _PhysicalPageChunkRepresentatives):
            return _PhysicalPageSequenceRepresentatives(
                num_chunks=num_chunks,
                num_kv_heads=int(self._representatives.shape[2]),
                q_indexer_dim=int(self._representatives.shape[3]),
            )
        return self._representatives[local_idx, :num_chunks]

    def items(self):
        for seq_idx in self._local_by_seq:
            yield seq_idx, self[seq_idx]


class _EfficientChunkScores:
    __slots__ = ("_chunk_logits", "_chunk_valid")

    def __init__(
        self,
        *,
        chunk_logits: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> None:
        self._chunk_logits = chunk_logits
        self._chunk_valid = chunk_valid


class _EfficientChunkBlockSelection:
    __slots__ = (
        "_selected_block_counts",
        "_selected_block_indices",
        "_selected_block_valid",
    )

    def __init__(
        self,
        *,
        selected_block_indices: torch.Tensor,
        selected_block_valid: torch.Tensor | None = None,
        selected_block_counts: torch.Tensor | None = None,
    ) -> None:
        self._selected_block_indices = selected_block_indices
        self._selected_block_valid = selected_block_valid
        self._selected_block_counts = selected_block_counts


class _EfficientDSAPlanState:
    __slots__ = (
        "_decode_tiles",
        "_large_tiles",
        "_max_q_len",
        "_max_prior_chunks",
        "_max_tiles_per_row_plan",
        "_max_top_k",
        "_row_current_chunks",
        "_row_group_ids",
        "_row_num_prior_chunks",
        "_row_plan",
        "_row_seq_ids",
        "_row_tail_lens",
        "_score_row_seq_ids",
        "_small_tiles",
        "_tile_plan",
        "_total_rows",
        "_total_tiles",
    )

    def __init__(
        self,
        *,
        row_plan: torch.Tensor,
        tile_plan: torch.Tensor,
        score_row_seq_ids: torch.Tensor,
        row_seq_ids: torch.Tensor,
        row_group_ids: torch.Tensor,
        row_num_prior_chunks: torch.Tensor,
        row_current_chunks: torch.Tensor,
        row_tail_lens: torch.Tensor,
        total_rows: int,
        total_tiles: int,
        decode_tiles: int,
        small_tiles: int,
        large_tiles: int,
        max_q_len: int,
        max_prior_chunks: int,
        max_top_k: int,
        max_tiles_per_row_plan: int,
    ) -> None:
        self._row_plan = row_plan
        self._tile_plan = tile_plan
        self._score_row_seq_ids = score_row_seq_ids
        self._row_seq_ids = row_seq_ids
        self._row_group_ids = row_group_ids
        self._row_num_prior_chunks = row_num_prior_chunks
        self._row_current_chunks = row_current_chunks
        self._row_tail_lens = row_tail_lens
        self._total_rows = total_rows
        self._total_tiles = total_tiles
        self._decode_tiles = decode_tiles
        self._small_tiles = small_tiles
        self._large_tiles = large_tiles
        self._max_q_len = max_q_len
        self._max_prior_chunks = max_prior_chunks
        self._max_top_k = max_top_k
        self._max_tiles_per_row_plan = max_tiles_per_row_plan


class _EfficientBatchedChunkBlockSelections(dict[int, typing.Any | None]):
    __slots__ = (
        "_chunk_top_k_by_seq",
        "_dsa_plan_state",
        "_fixed_decode_plan",
        "_row_current_chunks",
        "_row_seq_ids",
        "_row_tail_lens",
        "_selected_block_counts",
        "_selected_block_indices",
        "_selected_block_valid",
        "_seq_slices",
    )

    def __init__(
        self,
        *,
        selected_block_indices: torch.Tensor,
        selected_block_valid: torch.Tensor | None,
        seq_slices: dict[int, tuple[int, int, int]],
        chunk_top_k_by_seq: dict[int, int],
        row_seq_ids: torch.Tensor,
        row_current_chunks: torch.Tensor,
        row_tail_lens: torch.Tensor,
        per_seq: dict[int, typing.Any | None],
        dsa_plan_state: _EfficientDSAPlanState | None = None,
        selected_block_counts: torch.Tensor | None = None,
        fixed_decode_plan: bool = False,
    ) -> None:
        super().__init__(per_seq)
        self._selected_block_indices = selected_block_indices
        self._selected_block_valid = selected_block_valid
        self._selected_block_counts = selected_block_counts
        self._seq_slices = seq_slices
        self._chunk_top_k_by_seq = chunk_top_k_by_seq
        self._row_seq_ids = row_seq_ids
        self._row_current_chunks = row_current_chunks
        self._row_tail_lens = row_tail_lens
        self._dsa_plan_state = dsa_plan_state
        self._fixed_decode_plan = fixed_decode_plan


def _materialize_prefix_valid_from_counts(
    *,
    selected_blocks: torch.Tensor,
    selected_counts: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    top_width = int(selected_blocks.shape[1])
    selected_counts = selected_counts.to(
        device=device,
        dtype=torch.int32,
    )
    if top_width == 0:
        return torch.empty(
            int(selected_blocks.shape[0]),
            0,
            device=device,
            dtype=torch.bool,
        )
    return (
        torch.arange(top_width, device=device, dtype=torch.int32)[None, :]
        < selected_counts[:, None]
    )


def _selection_valid(
    selection_state: _EfficientChunkBlockSelection,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    selected_valid = selection_state._selected_block_valid
    if selected_valid is not None:
        if selected_valid.device != device:
            selected_valid = selected_valid.to(device=device)
        return selected_valid
    selected_counts = selection_state._selected_block_counts
    if selected_counts is None:
        return None
    return _materialize_prefix_valid_from_counts(
        selected_blocks=selection_state._selected_block_indices,
        selected_counts=selected_counts,
        device=device,
    )


def _selection_counts(
    selection_state: _EfficientChunkBlockSelection,
    *,
    device: torch.device,
) -> torch.Tensor | None:
    selected_counts = selection_state._selected_block_counts
    if selected_counts is not None:
        if selected_counts.device != device:
            selected_counts = selected_counts.to(device=device)
        return selected_counts.to(dtype=torch.int32)
    selected_valid = selection_state._selected_block_valid
    if selected_valid is None:
        return None
    if selected_valid.device != device:
        selected_valid = selected_valid.to(device=device)
    return selected_valid.sum(dim=-1, dtype=torch.int32)


class _EfficientChunkBlockTable:
    __slots__ = (
        "_block_table",
        "_max_seqlen_k",
        "_max_seqlen_q",
        "_request_lens",
        "_seqused_k",
    )

    def __init__(
        self,
        *,
        block_table: torch.Tensor,
        request_lens: torch.Tensor,
        seqused_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> None:
        self._block_table = block_table
        self._request_lens = request_lens
        self._seqused_k = seqused_k
        self._max_seqlen_q = max_seqlen_q
        self._max_seqlen_k = max_seqlen_k


class TritonBatchedChunkedDSARepresentativeProvider(nn.Module):
    """Batched Triton chunk representative provider for Nemotron-H DSA."""

    def __init__(
        self,
        *,
        q_indexer_dim: int,
        chunk_size: int,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        if q_indexer_dim <= 0:
            raise ValueError(f"q_indexer_dim must be positive: {q_indexer_dim}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive: {chunk_size}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive: {num_kv_heads}")
        self.q_indexer_dim = q_indexer_dim
        self.chunk_size = chunk_size
        self.num_kv_heads = num_kv_heads
        self.use_rep_cache = _env_bool(_DSA_REP_CACHE_ENV, False)
        self.rep_cache_cudagraph = _env_bool(
            _DSA_REP_CACHE_CUDAGRAPH_ENV,
            False,
        )
        self._rep_cache_verify_remaining = max(
            _env_int(_DSA_REP_CACHE_VERIFY_CALLS_ENV, 0),
            0,
        )
        self.register_buffer("_rep_cache_values", None, persistent=False)
        self.register_buffer("_rep_cache_valid", None, persistent=False)
        self._rep_cache_key_cache_ptr: int | None = None
        self._rep_cache_needs_repair = True
        _dsa_log_path_marker(
            "rep_cache_config",
            enabled=self.use_rep_cache,
            cudagraph=self.rep_cache_cudagraph,
        )

    @staticmethod
    def _is_all_single_token_decode(
        active_seq_infos: list[tuple[int, int, int, int]],
    ) -> bool:
        active_q_lens = [
            q_end - q_start
            for _, q_start, q_end, _ in active_seq_infos
            if q_end > q_start
        ]
        return bool(active_q_lens) and all(q_len == 1 for q_len in active_q_lens)

    def _get_rep_cache(
        self,
        *,
        key_cache: torch.Tensor,
        cache_info: tuple[str, int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.use_rep_cache:
            return None
        if cache_info != ("NHD", self.chunk_size):
            _dsa_log_path_marker(
                "rep_cache_fallback",
                reason="unsupported_cache_layout",
                cache_info=cache_info,
            )
            return None
        if (
            key_cache.dim() != 4
            or int(key_cache.shape[1]) != self.chunk_size
            or int(key_cache.shape[2]) != self.num_kv_heads
            or self.q_indexer_dim > int(key_cache.shape[3])
            or key_cache.dtype not in (torch.bfloat16, torch.float16)
            or not key_cache.is_cuda
        ):
            _dsa_log_path_marker(
                "rep_cache_fallback",
                reason="unsupported_key_cache",
                dtype=key_cache.dtype,
                shape=tuple(key_cache.shape),
            )
            return None

        num_physical_pages = int(key_cache.shape[0])
        expected_values_shape = (
            num_physical_pages,
            self.num_kv_heads,
            self.q_indexer_dim,
        )
        key_cache_ptr = int(key_cache.data_ptr())
        cache_values = self._rep_cache_values
        cache_valid = self._rep_cache_valid
        needs_allocation = (
            cache_values is None
            or cache_valid is None
            or tuple(cache_values.shape) != expected_values_shape
            or tuple(cache_valid.shape) != (num_physical_pages,)
            or cache_values.device != key_cache.device
            or cache_valid.device != key_cache.device
            or self._rep_cache_key_cache_ptr != key_cache_ptr
        )
        if needs_allocation:
            # Allocation errors are intentionally not swallowed. Falling back
            # after an OOM would hide that the requested cache was not active.
            cache_values = torch.empty(
                expected_values_shape,
                device=key_cache.device,
                dtype=torch.bfloat16,
            )
            cache_valid = torch.zeros(
                num_physical_pages,
                device=key_cache.device,
                dtype=torch.uint8,
            )
            self._rep_cache_values = cache_values
            self._rep_cache_valid = cache_valid
            self._rep_cache_key_cache_ptr = key_cache_ptr
            self._rep_cache_needs_repair = True
            _dsa_log_path_marker(
                "rep_cache_allocated",
                num_physical_pages=num_physical_pages,
                value_bytes=cache_values.numel() * cache_values.element_size(),
                valid_bytes=cache_valid.numel() * cache_valid.element_size(),
            )
        return cache_values, cache_valid

    def forward(
        self,
        *,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
        query_start_loc: torch.Tensor | None = None,
        active_seq_infos: list[tuple[int, int, int, int]] | None = None,
        cache_info: tuple[str, int] | None = None,
        should_skip_sequence: _SequenceSkipFn | None = None,
        **_: typing.Any,
    ) -> typing.Any:
        if active_seq_infos is None:
            return _UNAVAILABLE
        required_cache_info = ("NHD", self.chunk_size)
        if self.use_rep_cache and cache_info != required_cache_info:
            _dsa_log_path_marker(
                "rep_cache_fallback",
                reason="unsupported_cache_layout",
                cache_info=cache_info,
            )
        if (
            dsa_block_summaries_triton is None
            or cache_info != required_cache_info
            or block_table.dim() != 2
        ):
            return _UNAVAILABLE
        table_width = int(block_table.shape[1])
        if table_width <= 0:
            return _UNAVAILABLE

        if (
            query_start_loc is None
            and seq_lens is not None
            and active_seq_infos
            and not key_cache.is_cuda
        ):
            cpu_row_count = max(seq_idx for seq_idx, _, _, _ in active_seq_infos) + 1
            query_starts = [0] * (cpu_row_count + 1)
            for seq_idx, q_start, q_end, _ in active_seq_infos:
                query_starts[seq_idx] = q_start
                query_starts[seq_idx + 1] = q_end
            query_start_loc = torch.tensor(
                query_starts,
                device=block_table.device,
                dtype=torch.long,
            )

        if (
            seq_lens is None
            or query_start_loc is None
            or seq_lens.dim() != 1
            or query_start_loc.dim() != 1
            or block_table.device != key_cache.device
            or seq_lens.device != key_cache.device
            or query_start_loc.device != key_cache.device
        ):
            return _UNAVAILABLE

        gpu_active_row_count = min(
            int(block_table.shape[0]),
            int(seq_lens.shape[0]),
            int(query_start_loc.shape[0]) - 1,
        )
        if gpu_active_row_count <= 0:
            return _UNAVAILABLE

        # Dense attention still has to publish representatives for completed
        # pages. A later sparse call may consume the same physical prefix even
        # when CUDA graphs are disabled.
        maintain_dense_bypass_cache = self.use_rep_cache
        use_dense_skip = (
            not maintain_dense_bypass_cache
            and should_skip_sequence is not None
            and getattr(
                self,
                "use_full_attention_short_seq",
                False,
            )
        )
        dense_decode_budget = 0
        dense_prefill_budget = 0
        if use_dense_skip:
            decode_budget = self.chunk_size * int(getattr(self, "chunk_top_k", 1))
            prefill_budget = getattr(
                self,
                "dense_prefill_kv_threshold_tokens",
                decode_budget,
            )
            dense_decode_budget = int(decode_budget)
            dense_prefill_budget = int(prefill_budget)

        live_infos: list[tuple[int, int, int, int]] = []
        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            if key_len <= 0:
                continue
            if (
                not maintain_dense_bypass_cache
                and should_skip_sequence is not None
                and should_skip_sequence(
                    seq_idx,
                    q_start,
                    q_end,
                    key_len,
                )
            ):
                continue
            if seq_idx >= int(block_table.shape[0]):
                return _UNAVAILABLE
            num_chunks = math.ceil(key_len / self.chunk_size)
            if num_chunks > table_width:
                return _UNAVAILABLE
            live_infos.append((seq_idx, key_len, num_chunks, q_end - q_start))
        if not live_infos:
            return _UNAVAILABLE

        summary_batch = len(live_infos)
        all_decode = self._is_all_single_token_decode(active_seq_infos)
        # Full-decode CUDA graphs are captured with dummy sequence lengths of
        # one token, then replayed at arbitrary context lengths. Keep the
        # descriptor and maintenance launch at the physical table width so no
        # host-derived capture length is baked into the graph.
        max_summary_chunks = (
            table_width
            if self.rep_cache_cudagraph and all_decode
            else max(num_chunks for _, _, num_chunks, _ in live_infos)
        )
        max_seed_chunks = max(
            num_chunks - max((key_len - query_len) // self.chunk_size, 0)
            for _, key_len, num_chunks, query_len in live_infos
        )
        metadata = _dsa_prepare_summary_metadata_triton(
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_table=block_table,
            summary_batch=summary_batch,
            active_row_count=gpu_active_row_count,
            table_width=table_width,
            chunk_size=self.chunk_size,
            use_dense_skip=use_dense_skip,
            dense_decode_budget=dense_decode_budget,
            dense_prefill_budget=dense_prefill_budget,
        )
        if metadata is None:
            if key_cache.is_cuda:
                return _UNAVAILABLE
            metadata = _dsa_prepare_summary_metadata_torch(
                seq_lens=seq_lens,
                query_start_loc=query_start_loc,
                block_table=block_table,
                summary_batch=summary_batch,
                active_row_count=gpu_active_row_count,
                table_width=table_width,
                chunk_size=self.chunk_size,
                use_dense_skip=use_dense_skip,
                dense_decode_budget=dense_decode_budget,
                dense_prefill_budget=dense_prefill_budget,
            )
        gpu_block_table, gpu_seq_lens, gpu_query_lens = metadata
        if dsa_cudagraph_keepalive is not None:
            dsa_cudagraph_keepalive(
                gpu_block_table,
                gpu_seq_lens,
                gpu_query_lens,
            )
        # Capture-only dummy forwards use the original path. The monitor is a
        # global capability flag and defaults to true, so it is meaningful only
        # while an actual vLLM forward context is active.
        capture_in_progress = _dsa_cudagraph_runtime_active()
        rep_cache = None
        if not capture_in_progress or self.rep_cache_cudagraph:
            rep_cache = self._get_rep_cache(
                key_cache=key_cache,
                cache_info=cache_info,
            )
        batched_representatives = None
        used_physical_cache = False
        if rep_cache is not None:
            cache_values, cache_valid = rep_cache
            if all_decode and capture_in_progress and self.rep_cache_cudagraph:
                # Graph-cache mode routes every prefill slice through this
                # provider, so every historical page is seeded before decode.
                # Maintain only the current page here; a full validity scan has
                # O(batch * context) tiny CTAs and defeats cached-decode speed.
                cache_update_mode = "cudagraph_completed_page"
                cache_updated = dsa_update_current_block_summary_cache_triton(
                    key_cache=key_cache,
                    block_table=gpu_block_table,
                    seq_lens=gpu_seq_lens,
                    cache_values=cache_values,
                    cache_valid=cache_valid,
                    q_indexer_dim=self.q_indexer_dim,
                    max_chunks=max_summary_chunks,
                )
            elif self._rep_cache_needs_repair:
                # Repair once after allocation. This covers decode-only startup
                # and prefixes that existed before this sidecar was allocated.
                cache_update_mode = (
                    "decode_repair" if all_decode else "mixed_repair"
                )
                cache_updated = dsa_update_block_summary_cache_triton(
                    key_cache=key_cache,
                    block_table=gpu_block_table,
                    seq_lens=gpu_seq_lens,
                    cache_values=cache_values,
                    cache_valid=cache_valid,
                    q_indexer_dim=self.q_indexer_dim,
                    max_chunks=max_summary_chunks,
                )
            elif not all_decode:
                # Refresh every physical page touched by this prefill slice.
                # Untouched history remains in the sidecar, so work scales with
                # newly written KV rather than the full context.
                cache_update_mode = "written_pages"
                cache_updated = dsa_update_written_block_summary_cache_triton(
                    key_cache=key_cache,
                    block_table=gpu_block_table,
                    seq_lens=gpu_seq_lens,
                    query_lens=gpu_query_lens,
                    cache_values=cache_values,
                    cache_valid=cache_valid,
                    q_indexer_dim=self.q_indexer_dim,
                    max_written_chunks=max_seed_chunks,
                    max_chunks=max_summary_chunks,
                )
            else:
                # Launch this small maintenance kernel on every decode call.
                # It predicates on seq_len % chunk_size on device, so only
                # newly completed pages are published. Avoiding a CPU-side
                # sequence-length branch also makes this stage graph-friendly.
                cache_update_mode = "completed_page"
                cache_updated = dsa_update_current_block_summary_cache_triton(
                    key_cache=key_cache,
                    block_table=gpu_block_table,
                    seq_lens=gpu_seq_lens,
                    cache_values=cache_values,
                    cache_valid=cache_valid,
                    q_indexer_dim=self.q_indexer_dim,
                    max_chunks=max_summary_chunks,
                )
            if cache_updated:
                self._rep_cache_needs_repair = False
                batched_representatives = _PhysicalPageChunkRepresentatives(
                    cache_values=cache_values,
                    cache_valid=cache_valid,
                    block_table=gpu_block_table,
                    batch=summary_batch,
                    max_chunks=max_summary_chunks,
                    block_size=self.chunk_size,
                )
            else:
                self._rep_cache_needs_repair = True
            used_physical_cache = cache_updated
            if used_physical_cache:
                _dsa_log_path_marker(
                    (
                        "rep_cache_fused_decode"
                        if all_decode
                        else "rep_cache_fused_mixed"
                    ),
                    summary_batch=summary_batch,
                    max_summary_chunks=max_summary_chunks,
                    update_mode=cache_update_mode,
                )
            else:
                _dsa_log_path_marker(
                    "rep_cache_fallback",
                    reason="cache_update_kernel_unavailable",
                )

        if (
            used_physical_cache
            and self._rep_cache_verify_remaining > 0
            and not capture_in_progress
        ):
            cache_values, cache_valid = typing.cast(
                tuple[torch.Tensor, torch.Tensor], rep_cache
            )
            cached_representatives = dsa_cached_block_summaries_triton(
                key_cache=key_cache,
                block_table=gpu_block_table,
                seq_lens=gpu_seq_lens,
                cache_values=cache_values,
                cache_valid=cache_valid,
                q_indexer_dim=self.q_indexer_dim,
                max_chunks=max_summary_chunks,
            )
            reference_representatives = dsa_block_summaries_triton(
                key_cache=key_cache,
                block_table=gpu_block_table,
                seq_lens=gpu_seq_lens,
                q_indexer_dim=self.q_indexer_dim,
                max_chunks=max_summary_chunks,
            )
            if (
                cached_representatives is None
                or reference_representatives is None
            ):
                raise RuntimeError(
                    "representative-cache verification could not build reference"
                )
            active_chunks = torch.arange(
                max_summary_chunks,
                device=gpu_seq_lens.device,
            ).unsqueeze(0) < torch.div(
                gpu_seq_lens.unsqueeze(1) + self.chunk_size - 1,
                self.chunk_size,
                rounding_mode="floor",
            )
            # Both builders intentionally leave padded chunk rows
            # uninitialized. Compare only representatives that scoring may
            # consume so variable-length decode batches do not fail spuriously.
            torch.testing.assert_close(
                cached_representatives[active_chunks],
                reference_representatives[active_chunks],
                rtol=0,
                atol=0,
            )
            self._rep_cache_verify_remaining -= 1
            _dsa_log_path_marker(
                "rep_cache_verified",
                remaining=self._rep_cache_verify_remaining,
                summary_batch=summary_batch,
                max_summary_chunks=max_summary_chunks,
            )

        if batched_representatives is None:
            batched_representatives = dsa_block_summaries_triton(
                key_cache=key_cache,
                block_table=gpu_block_table,
                seq_lens=gpu_seq_lens,
                q_indexer_dim=self.q_indexer_dim,
                max_chunks=max_summary_chunks,
            )
        expected_shape = (
            summary_batch,
            max_summary_chunks,
            self.num_kv_heads,
            self.q_indexer_dim,
        )
        if (
            batched_representatives is None
            or tuple(batched_representatives.shape) != expected_shape
        ):
            return _UNAVAILABLE

        if rep_cache is not None and not used_physical_cache:
            cache_values, cache_valid = rep_cache
            seeded = dsa_seed_block_summary_cache_triton(
                representatives=batched_representatives,
                block_table=gpu_block_table,
                seq_lens=gpu_seq_lens,
                query_lens=gpu_query_lens,
                cache_values=cache_values,
                cache_valid=cache_valid,
                block_size=self.chunk_size,
                max_seed_chunks=max_seed_chunks,
            )
            _dsa_log_path_marker(
                "rep_cache_seed" if seeded else "rep_cache_fallback",
                reason="seeded" if seeded else "seed_kernel_unavailable",
                summary_batch=summary_batch,
                max_summary_chunks=max_summary_chunks,
                max_seed_chunks=max_seed_chunks,
            )

        seq_id_layout = (
            "original"
            if all(
                seq_idx == local_idx
                for local_idx, (seq_idx, _, _, _) in enumerate(live_infos)
            )
            else "compact"
        )
        return _TritonBatchedChunkRepresentatives(
            representatives=batched_representatives,
            local_by_seq={
                seq_idx: local_idx
                for local_idx, (seq_idx, _, _, _) in enumerate(live_infos)
            },
            num_chunks_by_seq={
                seq_idx: (
                    max_summary_chunks
                    if capture_in_progress and self.rep_cache_cudagraph and all_decode
                    else num_chunks
                )
                for seq_idx, _, num_chunks, _ in live_infos
            },
            seq_id_layout=seq_id_layout,
        )

    def is_available(self, result: typing.Any) -> bool:
        return result is not _UNAVAILABLE

    def get_for_sequence(
        self,
        result: typing.Any,
        *,
        seq_idx: int,
        **_: typing.Any,
    ) -> torch.Tensor | _PhysicalPageSequenceRepresentatives | None:
        if result is _UNAVAILABLE:
            return None
        if not isinstance(result, _TritonBatchedChunkRepresentatives):
            raise TypeError(f"unexpected representative result: {type(result)!r}")
        return result.get(seq_idx)


class EfficientChunkedDSAScoringProvider(nn.Module):
    """Efficient-backend chunk scoring provider placeholder.

    The class has a separate backend identity while preserving the same torch
    math for the first score-only seam. A fused/Triton scorer can replace this
    implementation without changing the attention orchestration.
    """

    def __init__(
        self,
        *,
        q_indexer_dim: int,
        logit_scale: float,
    ) -> None:
        super().__init__()
        if q_indexer_dim <= 0:
            raise ValueError(f"q_indexer_dim must be positive: {q_indexer_dim}")
        self.q_indexer_dim = q_indexer_dim
        self.logit_scale = logit_scale

    def forward(
        self,
        *,
        score_query_states: torch.Tensor,
        representative_state: typing.Any | None = None,
        chunk_representatives: torch.Tensor | None = None,
        current_chunks: torch.Tensor,
        max_prior_chunks: int,
        chunk_ids: torch.Tensor | None = None,
        seq_idx: int | None = None,
        group_idx: int | None = None,
        **_: typing.Any,
    ) -> typing.Any:
        representatives = self._materialize_representatives(
            representative_state=representative_state,
            chunk_representatives=chunk_representatives,
            seq_idx=seq_idx,
            group_idx=group_idx,
            max_prior_chunks=max_prior_chunks,
        )
        if representatives is None:
            return _UNAVAILABLE

        if max_prior_chunks <= 0:
            shape = (score_query_states.shape[0], 0)
            return _EfficientChunkScores(
                chunk_logits=torch.empty(
                    shape,
                    device=score_query_states.device,
                    dtype=torch.float32,
                ),
                chunk_valid=torch.empty(
                    shape,
                    device=score_query_states.device,
                    dtype=torch.bool,
                ),
            )

        max_prior_chunks = int(representatives.shape[0])
        if max_prior_chunks == 0:
            shape = (score_query_states.shape[0], 0)
            return _EfficientChunkScores(
                chunk_logits=torch.empty(
                    shape,
                    device=score_query_states.device,
                    dtype=torch.float32,
                ),
                chunk_valid=torch.empty(
                    shape,
                    device=score_query_states.device,
                    dtype=torch.bool,
                ),
            )
        if (
            score_query_states.dim() != 2
            or representatives.dim() != 2
            or current_chunks.dim() != 1
            or score_query_states.shape[0] != current_chunks.shape[0]
            or score_query_states.shape[1] != self.q_indexer_dim
            or representatives.shape[1] != self.q_indexer_dim
        ):
            return _UNAVAILABLE
        if chunk_ids is not None and (
            chunk_ids.dim() != 1
            or chunk_ids.shape[0] != representatives.shape[0]
            or chunk_ids.device != score_query_states.device
        ):
            return _UNAVAILABLE

        if (
            score_query_states.is_cuda
            and representatives.is_cuda
            and score_query_states.dtype == representatives.dtype
            and score_query_states.dtype in (torch.bfloat16, torch.float16)
        ):
            chunk_logits = torch.mm(
                score_query_states,
                representatives.transpose(0, 1),
                out_dtype=torch.float32,
            )
        else:
            chunk_logits = torch.mm(
                score_query_states.float(),
                representatives.float().transpose(0, 1),
            )
        chunk_logits.mul_(self.logit_scale / math.sqrt(self.q_indexer_dim))
        selectable_counts = current_chunks.clamp(
            min=0,
            max=max_prior_chunks,
        ).to(device=score_query_states.device, dtype=torch.long)
        if chunk_ids is None:
            chunk_ids = torch.arange(
                max_prior_chunks,
                device=score_query_states.device,
                dtype=selectable_counts.dtype,
            )
        chunk_valid = chunk_ids[None, :] < selectable_counts[:, None]
        chunk_logits = chunk_logits.masked_fill(
            ~chunk_valid,
            torch.finfo(chunk_logits.dtype).min,
        )
        return _EfficientChunkScores(
            chunk_logits=chunk_logits,
            chunk_valid=chunk_valid,
        )

    def is_available(self, result: typing.Any) -> bool:
        return result is not _UNAVAILABLE

    def get_scores(
        self,
        result: typing.Any,
        **_: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if result is _UNAVAILABLE:
            return None
        if not isinstance(result, _EfficientChunkScores):
            raise TypeError(f"unexpected scoring result: {type(result)!r}")
        return result._chunk_logits, result._chunk_valid

    def _materialize_representatives(
        self,
        *,
        representative_state: typing.Any | None,
        chunk_representatives: torch.Tensor | None,
        seq_idx: int | None,
        group_idx: int | None,
        max_prior_chunks: int,
    ) -> torch.Tensor | None:
        if chunk_representatives is not None:
            representatives = chunk_representatives
        elif isinstance(representative_state, _TritonBatchedChunkRepresentatives):
            if seq_idx is None:
                return None
            local_idx = representative_state._local_by_seq.get(seq_idx)
            if local_idx is None:
                return None
            num_chunks = representative_state._num_chunks_by_seq[seq_idx]
            representatives = representative_state._representatives[
                local_idx, :num_chunks
            ]
        elif isinstance(representative_state, torch.Tensor):
            representatives = representative_state
        elif hasattr(representative_state, "_by_seq"):
            by_seq = representative_state._by_seq
            if by_seq is not None:
                if seq_idx is None:
                    return None
                representatives = by_seq.get(seq_idx)
            else:
                representatives = representative_state._single
        else:
            return None

        if representatives is None:
            return None
        representatives = representatives[:max_prior_chunks]
        if representatives.dim() == 3:
            if group_idx is None:
                return None
            if group_idx < 0 or group_idx >= int(representatives.shape[1]):
                return None
            representatives = representatives[:, group_idx]
        return representatives


class EfficientTopKChunkedDSABlockSelectionProvider(nn.Module):
    """Efficient-backend top-k logical-block selector placeholder.

    The implementation intentionally mirrors the PyTorch selector while keeping
    a distinct backend identity. A fused selector can replace this class without
    changing the attention orchestration.
    """

    def forward(
        self,
        *,
        score_state: typing.Any,
        block_top_k: int | None = None,
        chunk_top_k: int | None = None,
        **_: typing.Any,
    ) -> typing.Any:
        scores = self._materialize_scores(score_state)
        if scores is None:
            return _UNAVAILABLE
        chunk_logits, chunk_valid = scores
        top_k_limit = block_top_k if block_top_k is not None else chunk_top_k
        if top_k_limit is None:
            return _UNAVAILABLE

        if top_k_limit <= 0 or chunk_logits.shape[-1] == 0:
            shape = (chunk_logits.shape[0], 0)
            return _EfficientChunkBlockSelection(
                selected_block_indices=torch.empty(
                    shape,
                    device=chunk_logits.device,
                    dtype=torch.long,
                ),
                selected_block_valid=torch.empty(
                    shape,
                    device=chunk_logits.device,
                    dtype=torch.bool,
                ),
            )

        top_k = min(int(top_k_limit), int(chunk_logits.shape[-1]))
        selected_block_indices = chunk_logits.topk(
            k=top_k, dim=-1, sorted=False
        ).indices
        selected_block_valid = chunk_valid.gather(
            dim=-1,
            index=selected_block_indices,
        )
        selected_block_indices = selected_block_indices.masked_fill(
            ~selected_block_valid,
            0,
        )
        return _EfficientChunkBlockSelection(
            selected_block_indices=selected_block_indices,
            selected_block_valid=selected_block_valid,
        )

    def is_available(self, result: typing.Any) -> bool:
        return result is not _UNAVAILABLE

    def get_selected_blocks(
        self,
        result: typing.Any,
        **_: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if result is _UNAVAILABLE:
            return None
        if not isinstance(result, _EfficientChunkBlockSelection):
            raise TypeError(f"unexpected block selection result: {type(result)!r}")
        selected_blocks = result._selected_block_indices
        selected_valid = _selection_valid(
            result,
            device=selected_blocks.device,
        )
        if selected_valid is None:
            raise TypeError("block selection result has no validity state")
        return selected_blocks, selected_valid

    def _materialize_scores(
        self,
        score_state: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not isinstance(score_state, _EfficientChunkScores):
            return None
        return score_state._chunk_logits, score_state._chunk_valid


class EfficientChunkedDSABlockTableProvider(nn.Module):
    """Efficient-backend logical-to-physical block-table builder placeholder."""

    def forward(
        self,
        *,
        block_table: torch.Tensor,
        chunk_size: int,
        key_len: int,
        q_len: int | None = None,
        dense: bool = False,
        mode: str = "prefill",
        selection_state: typing.Any | None = None,
        current_chunks: torch.Tensor | None = None,
        query_positions: torch.Tensor | None = None,
        query_position_start: int | None = None,
        recent_window_pages: int = 0,
        **_: typing.Any,
    ) -> typing.Any:
        if chunk_size <= 0 or key_len < 0 or block_table.dim() != 1:
            return _UNAVAILABLE
        if mode == "decode":
            return self._build_decode(
                block_table=block_table,
                selection_state=selection_state,
                current_chunks=current_chunks,
                query_positions=query_positions,
                key_len=key_len,
                chunk_size=chunk_size,
                recent_window_pages=recent_window_pages,
            )
        if dense:
            if q_len is None:
                return _UNAVAILABLE
            return self._build_dense_prefill(
                block_table=block_table,
                q_len=q_len,
                key_len=key_len,
                chunk_size=chunk_size,
            )
        if q_len is None or current_chunks is None or query_position_start is None:
            return _UNAVAILABLE
        return self._build_sparse_prefill(
            block_table=block_table,
            selection_state=selection_state,
            current_chunks=current_chunks,
            query_position_start=query_position_start,
            q_len=q_len,
            key_len=key_len,
            chunk_size=chunk_size,
            recent_window_pages=recent_window_pages,
        )

    def is_available(self, result: typing.Any) -> bool:
        return result is not _UNAVAILABLE

    def get_page_table(
        self,
        result: typing.Any,
        **_: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
        if result is _UNAVAILABLE:
            return None
        if not isinstance(result, _EfficientChunkBlockTable):
            raise TypeError(f"unexpected block-table result: {type(result)!r}")
        return (
            result._block_table,
            result._request_lens,
            result._seqused_k,
            result._max_seqlen_q,
            result._max_seqlen_k,
        )

    def _build_dense_prefill(
        self,
        *,
        block_table: torch.Tensor,
        q_len: int,
        key_len: int,
        chunk_size: int,
    ) -> typing.Any:
        if q_len <= 0:
            return _UNAVAILABLE
        num_pages = math.ceil(key_len / chunk_size) if key_len > 0 else 0
        if num_pages > int(block_table.shape[0]):
            return _UNAVAILABLE
        return _EfficientChunkBlockTable(
            block_table=block_table[:num_pages].to(torch.int32).view(1, -1),
            request_lens=torch.full(
                (1,),
                q_len,
                device=block_table.device,
                dtype=torch.int32,
            ),
            seqused_k=torch.full(
                (1,),
                key_len,
                device=block_table.device,
                dtype=torch.int32,
            ),
            max_seqlen_q=q_len,
            max_seqlen_k=key_len,
        )

    def _build_sparse_prefill(
        self,
        *,
        block_table: torch.Tensor,
        selection_state: typing.Any | None,
        current_chunks: torch.Tensor,
        query_position_start: int,
        q_len: int,
        key_len: int,
        chunk_size: int,
        recent_window_pages: int,
    ) -> typing.Any:
        if current_chunks.dim() != 1 or int(current_chunks.shape[0]) != q_len:
            return _UNAVAILABLE
        if q_len <= 0 or query_position_start < 0:
            return _UNAVAILABLE
        num_pages = math.ceil(key_len / chunk_size) if key_len > 0 else 0
        if num_pages > int(block_table.shape[0]):
            return _UNAVAILABLE
        device = current_chunks.device
        if block_table.device != device:
            block_table = block_table.to(device=device)

        selection = self._materialize_selection(
            selection_state=selection_state,
            rows=q_len,
            device=device,
        )
        if selection is None:
            return _UNAVAILABLE
        selected_blocks, selected_valid = selection
        if (
            selected_blocks.shape != selected_valid.shape
            or selected_blocks.dim() != 2
            or int(selected_blocks.shape[0]) != q_len
        ):
            return _UNAVAILABLE
        if recent_window_pages > 0 and selected_blocks.shape[1] > 0:
            order = torch.argsort(
                selected_valid.to(torch.int8),
                dim=-1,
                descending=True,
                stable=True,
            )
            selected_blocks = selected_blocks.gather(-1, order)
            selected_valid = selected_valid.gather(-1, order)

        valid_counts = selected_valid.sum(dim=-1).to(dtype=torch.long)
        top_width = int(selected_blocks.shape[1])
        recent_counts = current_chunks.clamp(
            min=0,
            max=recent_window_pages,
        )
        table_width = top_width + recent_window_pages + 1
        logical_pages = current_chunks[:, None].expand(q_len, table_width).clone()
        if top_width > 0:
            logical_pages[:, :top_width] = selected_blocks.to(dtype=torch.long)
        if recent_window_pages > 0:
            recent_offsets = torch.arange(
                recent_window_pages,
                device=device,
                dtype=torch.long,
            )
            safe_recent_offsets = torch.minimum(
                recent_offsets[None, :],
                (recent_counts[:, None] - 1).clamp_min(0),
            )
            logical_pages.scatter_(
                1,
                valid_counts[:, None] + recent_offsets[None, :],
                current_chunks[:, None] - recent_counts[:, None] + safe_recent_offsets,
            )
        logical_pages.scatter_(
            1,
            (valid_counts + recent_counts)[:, None],
            current_chunks[:, None],
        )
        unused_page_mask = (
            torch.arange(table_width, device=device, dtype=torch.long)[None, :]
            > (valid_counts + recent_counts)[:, None]
        )
        # The fused top-k output uses -1 beyond each row's valid-count prefix.
        # Sanitize those unused indices before gather; masking afterward is too
        # late because gather validates every index in the tensor.
        logical_pages.masked_fill_(unused_page_mask, 0)
        physical_pages = (
            block_table.to(dtype=torch.long)
            .expand(q_len, -1)
            .gather(
                1,
                logical_pages,
            )
            .to(torch.int32)
        )
        physical_pages.masked_fill_(unused_page_mask, 0)

        seq_positions = torch.arange(
            query_position_start,
            key_len,
            device=device,
            dtype=torch.long,
        )
        if int(seq_positions.shape[0]) != q_len:
            return _UNAVAILABLE
        local_prefixes = seq_positions - current_chunks * chunk_size + 1
        seqused_k = ((valid_counts + recent_counts) * chunk_size + local_prefixes).to(
            torch.int32
        )
        return _EfficientChunkBlockTable(
            block_table=physical_pages,
            request_lens=torch.ones(q_len, device=device, dtype=torch.int32),
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=self._sparse_suffix_max_seqused_k(
                query_position_start=query_position_start,
                key_len=key_len,
                chunk_size=chunk_size,
                top_width=top_width + recent_window_pages,
            ),
        )

    def _build_decode(
        self,
        *,
        block_table: torch.Tensor,
        selection_state: typing.Any | None,
        current_chunks: torch.Tensor | None,
        query_positions: torch.Tensor | None,
        key_len: int,
        chunk_size: int,
        recent_window_pages: int,
    ) -> typing.Any:
        if current_chunks is None or query_positions is None:
            return _UNAVAILABLE
        if key_len <= 0:
            return _UNAVAILABLE
        if current_chunks.numel() != 1 or query_positions.numel() != 1:
            return _UNAVAILABLE
        device = current_chunks.device
        if block_table.device != device:
            block_table = block_table.to(device=device)

        selection = self._materialize_selection(
            selection_state=selection_state,
            rows=1,
            device=device,
        )
        if selection is None:
            return _UNAVAILABLE
        selected_blocks, selected_valid = selection
        if (
            selected_blocks.shape != selected_valid.shape
            or selected_blocks.dim() != 2
            or tuple(selected_blocks.shape[:1]) != (1,)
        ):
            return _UNAVAILABLE

        num_pages = math.ceil(key_len / chunk_size)
        if num_pages > int(block_table.shape[0]):
            return _UNAVAILABLE

        current_chunk = (key_len - 1) // chunk_size
        tail_len = key_len - current_chunk * chunk_size
        top_width = int(selected_blocks.shape[1])
        page_table_plan = dsa_decode_page_table_triton(
            block_table=block_table,
            selected_blocks=selected_blocks,
            selected_valid=selected_valid,
            current_chunk=current_chunk,
            chunk_size=chunk_size,
            tail_len=tail_len,
            recent_window_pages=recent_window_pages,
        )
        if page_table_plan is None:
            return _UNAVAILABLE

        page_table, seqused_k = page_table_plan
        recent_count = min(recent_window_pages, current_chunk)
        max_seqlen_k = (top_width + recent_count) * chunk_size + tail_len
        return _EfficientChunkBlockTable(
            block_table=page_table,
            request_lens=torch.ones(1, device=device, dtype=torch.int32),
            seqused_k=seqused_k,
            max_seqlen_q=1,
            max_seqlen_k=max_seqlen_k,
        )

    def _materialize_selection(
        self,
        *,
        selection_state: typing.Any | None,
        rows: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if selection_state is None:
            shape = (rows, 0)
            return (
                torch.empty(shape, device=device, dtype=torch.long),
                torch.empty(shape, device=device, dtype=torch.bool),
            )
        if not isinstance(selection_state, _EfficientChunkBlockSelection):
            return None
        selected_blocks = selection_state._selected_block_indices
        if selected_blocks.device != device:
            selected_blocks = selected_blocks.to(device=device)
        selected_valid = _selection_valid(selection_state, device=device)
        if selected_valid is None:
            return None
        return selected_blocks, selected_valid

    @staticmethod
    def _sparse_suffix_max_seqused_k(
        *,
        query_position_start: int,
        key_len: int,
        chunk_size: int,
        top_width: int,
    ) -> int:
        start_chunk = query_position_start // chunk_size
        end_chunk = (key_len - 1) // chunk_size
        max_used = 0
        for chunk_idx in range(start_chunk, end_chunk + 1):
            chunk_query_start = max(query_position_start, chunk_idx * chunk_size)
            chunk_query_end = min(key_len, (chunk_idx + 1) * chunk_size)
            if chunk_query_end <= chunk_query_start:
                continue
            local_prefix = chunk_query_end - chunk_idx * chunk_size
            # This is an allocation/launch upper bound. Fused selection may
            # expose the full rectangular width even when early rows use less.
            valid_count = top_width
            max_used = max(max_used, valid_count * chunk_size + local_prefix)
        return max_used


class EfficientChunkedDSAProviderBundle(
    ChunkedDSAAttentionProviderMixin,
    nn.Module,
):
    """Efficient component bundle for the chunked DSA pipeline."""

    def __init__(
        self,
        *,
        q_indexer_dim: int,
        chunk_size: int,
        num_kv_heads: int,
        head_dim: int,
        logit_scale: float,
        chunk_top_k: int = 1,
        query_chunk_size: int = 1,
        num_heads: int | None = None,
        total_num_kv_heads: int | None = None,
        **_: typing.Any,
    ) -> None:
        super().__init__()
        self.q_indexer_dim = q_indexer_dim
        self.chunk_size = chunk_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self._init_common_options(
            logit_scale=logit_scale,
            chunk_top_k=chunk_top_k,
            query_chunk_size=query_chunk_size,
            num_heads=num_kv_heads if num_heads is None else num_heads,
            total_num_kv_heads=(
                num_kv_heads if total_num_kv_heads is None else total_num_kv_heads
            ),
        )
        self.q_indexer_use_triton_scoring = _env_bool(
            _DSA_TRITON_SCORING_ENV,
            True,
        )
        _dsa_log_path_marker(
            "config",
            chunk_size=self.chunk_size,
            chunk_top_k=self.q_indexer_chunk_top_k,
            triton_scoring_provider="gpu_tile_plan",
            use_triton_scoring=self.q_indexer_use_triton_scoring,
        )
        self.representative_provider = TorchChunkedDSARepresentativeProvider(
            q_indexer_dim=q_indexer_dim,
            chunk_size=chunk_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.batched_representative_provider = (
            TritonBatchedChunkedDSARepresentativeProvider(
                q_indexer_dim=q_indexer_dim,
                chunk_size=chunk_size,
                num_kv_heads=num_kv_heads,
            )
        )
        self.batched_representative_provider.chunk_top_k = self.q_indexer_chunk_top_k
        self.batched_representative_provider.use_full_attention_short_seq = (
            self.q_indexer_use_full_attention_short_seq
        )
        self.batched_representative_provider.dense_prefill_kv_threshold_tokens = (
            self.q_indexer_dense_prefill_kv_threshold_tokens
        )
        self.scoring_provider = EfficientChunkedDSAScoringProvider(
            q_indexer_dim=q_indexer_dim,
            logit_scale=logit_scale,
        )
        self.block_selection_provider = EfficientTopKChunkedDSABlockSelectionProvider()
        self.block_table_provider = EfficientChunkedDSABlockTableProvider()

    def _forward_dsa_chunked_multi_kv_head_page_table_fa_bucket(
        self,
        *,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn: typing.Any,
        attn_metadata: typing.Any | None,
        positions: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        batched_chunk_representatives: dict[int, torch.Tensor] | None,
        output: torch.Tensor,
        indexer_q_proj: typing.Callable[
            [torch.Tensor], tuple[torch.Tensor, typing.Any]
        ],
        local_kv_head_indices: torch.Tensor,
        precomputed_indexer_q: torch.Tensor | None = None,
        precomputed_indexer_q_by_head: tuple[torch.Tensor, ...] | None = None,
    ) -> set[int]:
        """Run one FA workload per KV head on zero-copy strided K/V views."""
        if self.num_kv_heads <= 1 or self.num_heads % self.num_kv_heads != 0:
            return set()
        if local_kv_head_indices.numel() < self.num_kv_heads:
            return set()
        fake_page_pitch = _nhd_fake_page_pitch(
            key_cache, num_kv_heads=self.num_kv_heads
        )
        value_fake_page_pitch = _nhd_fake_page_pitch(
            value_cache, num_kv_heads=self.num_kv_heads
        )
        if fake_page_pitch is None or value_fake_page_pitch != fake_page_pitch:
            return set()
        fake_key_cache = _make_nhd_fake_page_view(
            key_cache, num_kv_heads=self.num_kv_heads
        )
        fake_value_cache = _make_nhd_fake_page_view(
            value_cache, num_kv_heads=self.num_kv_heads
        )
        if fake_key_cache is None or fake_value_cache is None:
            return set()

        group_size = self.num_heads // self.num_kv_heads
        plans: list[dict[str, typing.Any]] = []
        handled: set[int] | None = None
        for kv_head_idx in range(self.num_kv_heads):
            head_start = kv_head_idx * group_size
            head_end = head_start + group_size
            one_head_representatives = None
            representative_group_idx = 0
            if batched_chunk_representatives is not None:
                has_direct_representatives = all(
                    hasattr(batched_chunk_representatives, attr)
                    for attr in (
                        "_representatives",
                        "_local_by_seq",
                        "_num_chunks_by_seq",
                    )
                )
                if has_direct_representatives:
                    one_head_representatives = batched_chunk_representatives
                    representative_group_idx = kv_head_idx
                else:
                    one_head_representatives = {}
                    for (
                        seq_idx,
                        representatives,
                    ) in batched_chunk_representatives.items():
                        if (
                            representatives.dim() != 3
                            or representatives.shape[1] <= kv_head_idx
                        ):
                            return set()
                        one_head_representatives[seq_idx] = representatives[
                            :, kv_head_idx : kv_head_idx + 1
                        ].contiguous()

            one_head_indexer_q = None
            if precomputed_indexer_q_by_head is not None:
                one_head_indexer_q = precomputed_indexer_q_by_head[kv_head_idx]
            elif precomputed_indexer_q is not None:
                one_head_indexer_q = precomputed_indexer_q[
                    :, kv_head_idx : kv_head_idx + 1
                ].contiguous()

            remapped_block_table = _remap_nhd_block_table_for_kv_head(
                block_table,
                fake_page_pitch=fake_page_pitch,
                num_kv_heads=self.num_kv_heads,
                kv_head_idx=kv_head_idx,
            )
            captured: list[dict[str, typing.Any]] = []

            def _capture_flash_attn(
                *,
                _captured: list[dict[str, typing.Any]] = captured,
                **kwargs: typing.Any,
            ) -> None:
                _captured.append(kwargs)

            group_query = query_states[:, head_start:head_end]
            group_output = torch.empty_like(group_query)
            group_handled = self._forward_dsa_chunked_one_kv_head_page_table_fa_bucket(
                hidden_states=hidden_states,
                query_states=group_query,
                key_cache=fake_key_cache,
                value_cache=fake_value_cache,
                block_table=remapped_block_table,
                attn=attn,
                attn_metadata=attn_metadata,
                positions=positions,
                active_seq_infos=active_seq_infos,
                batched_chunk_representatives=one_head_representatives,
                output=group_output,
                local_kv_head_indices=local_kv_head_indices[
                    kv_head_idx : kv_head_idx + 1
                ],
                indexer_q_proj=indexer_q_proj,
                precomputed_indexer_q=one_head_indexer_q,
                representative_group_idx=representative_group_idx,
                flash_attn_override=_capture_flash_attn,
            )
            if group_handled is None or len(captured) != 1:
                return set()
            group_handled_set = set(group_handled)
            handled = (
                group_handled_set
                if handled is None
                else handled.intersection(group_handled_set)
            )
            plans.append(captured[0])

        if not plans or not handled:
            return handled or set()

        flash_attn = _get_flash_attn_varlen_func()
        if flash_attn is None:
            return set()

        # A single concatenated launch for all local KV heads scales worse than
        # linearly once the flattened request-row count crosses the large-batch
        # FA scheduling threshold.  Preserve the metadata-only fake-page view,
        # but submit the independently planned head workloads separately.  The
        # original per-head output buffers also avoid a concatenate/permute.
        for kv_head_idx, plan in enumerate(plans):
            flash_attn(**plan)
            head_start = kv_head_idx * group_size
            head_end = head_start + group_size
            for seq_idx, q_start, q_end, _ in active_seq_infos:
                if seq_idx in handled:
                    output[q_start:q_end, head_start:head_end].copy_(
                        plan["out"][q_start:q_end]
                    )
        return handled

    def should_prepare_batched_representatives(self) -> bool:
        return True

    def build_representative_state(
        self,
        *,
        key_states: torch.Tensor | None,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        key_len: int,
        **kwargs: typing.Any,
    ) -> typing.Any:
        result = self.representative_provider(
            key_states=key_states,
            key_cache=key_cache,
            block_table=block_table,
            key_len=key_len,
            **kwargs,
        )
        if not self.representative_provider.is_available(result):
            raise ValueError("Efficient DSA representative provider is unavailable")
        return result

    def gather_kv_sequence(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        key_len: int,
    ) -> torch.Tensor:
        return self.representative_provider._gather_kv_sequence(
            cache,
            block_table,
            key_len,
        )

    def gather_kv_positions_for_head(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        kv_head_idx: int,
    ) -> torch.Tensor:
        cache = normalize_packed_nhd_kv_cache(
            cache,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )
        if cache.dim() != 4:
            raise NotImplementedError(
                f"DSA cache gather expects a 4D KV cache, got {cache.shape}"
            )
        if block_table.device != token_indices.device:
            block_table = block_table.to(device=token_indices.device)
        if cache.shape[2] == self.num_kv_heads:
            block_size = cache.shape[1]
            cache_layout = "NHD"
        elif cache.shape[1] == self.num_kv_heads:
            block_size = cache.shape[2]
            cache_layout = "HND"
        else:
            raise NotImplementedError(
                "DSA cache gather only supports NHD/HND KV cache layouts, "
                f"got shape={cache.shape}, num_kv_heads={self.num_kv_heads}"
            )

        flat_token_indices = token_indices.reshape(-1)
        block_indices = torch.div(
            flat_token_indices, block_size, rounding_mode="floor"
        ).to(torch.long)
        block_offsets = flat_token_indices.remainder(block_size).to(torch.long)
        block_ids = block_table.index_select(0, block_indices).to(torch.long)
        if cache_layout == "NHD":
            selected = cache[block_ids, block_offsets, kv_head_idx]
        else:
            selected = cache[block_ids, kv_head_idx, block_offsets]
        return selected.view(*token_indices.shape, self.head_dim)

    def get_batched_representatives_by_seq(
        self,
        *,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
        active_seq_infos: list[tuple[int, int, int, int]],
        cache_info: tuple[str, int] | None = None,
        should_skip_sequence: _SequenceSkipFn | None = None,
        **kwargs: typing.Any,
    ) -> typing.Any | None:
        result = self.batched_representative_provider(
            key_cache=key_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            active_seq_infos=active_seq_infos,
            cache_info=cache_info,
            should_skip_sequence=should_skip_sequence,
            **kwargs,
        )
        if not self.batched_representative_provider.is_available(result):
            return None
        return result

    def try_select_blocks_batched(
        self,
        *,
        indexer_q: torch.Tensor,
        sparse_infos: list[tuple[int, int, int, int, int, int, torch.Tensor | None]],
        batched_chunk_representatives: typing.Any | None,
        block_table: torch.Tensor,
        representative_group_idx: int = 0,
        query_start_loc: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        num_actual_tokens: int | None = None,
        active_seq_count: int | None = None,
        dense_decode_threshold: int | None = None,
        dense_prefill_threshold: int | None = None,
        **_: typing.Any,
    ) -> dict[int, typing.Any | None] | None:
        if not getattr(self, "q_indexer_use_triton_scoring", False):
            return None
        if (
            dsa_batched_score_topk_tile_plan_triton is None
            or dsa_build_score_metadata_triton is None
            or dsa_build_score_tile_plan_triton is None
            or dsa_score_tile_plan_config is None
        ):
            return None
        if batched_chunk_representatives is None or indexer_q.dim() != 3:
            return None

        device = indexer_q.device
        num_query_rows = int(indexer_q.shape[0])
        num_groups = int(indexer_q.shape[1])
        if int(indexer_q.shape[2]) != self.q_indexer_dim or num_groups <= 0:
            return None
        if block_table.dim() != 2:
            return None
        if (
            query_start_loc is None
            or seq_lens is None
            or num_actual_tokens is None
            or active_seq_count is None
            or dense_decode_threshold is None
            or dense_prefill_threshold is None
        ):
            return None
        if (
            query_start_loc.device != device
            or seq_lens.device != device
            or query_start_loc.dim() != 1
            or seq_lens.dim() != 1
            or int(query_start_loc.shape[0]) < int(active_seq_count) + 1
            or int(seq_lens.shape[0]) < int(active_seq_count)
        ):
            return None

        direct_representatives = None
        direct_num_chunks_by_seq: dict[int, int] | None = None
        direct_local_by_seq: dict[int, int] | None = None
        direct_seq_id_layout = "compact"
        if isinstance(
            batched_chunk_representatives,
            _TritonBatchedChunkRepresentatives,
        ):
            direct_representatives = batched_chunk_representatives._representatives
            direct_num_chunks_by_seq = batched_chunk_representatives._num_chunks_by_seq
            direct_local_by_seq = batched_chunk_representatives._local_by_seq
            direct_seq_id_layout = batched_chunk_representatives._seq_id_layout
            if direct_seq_id_layout not in ("compact", "original"):
                return None
            if (
                direct_representatives.dim() != 4
                or int(direct_representatives.shape[2]) <= representative_group_idx
                or int(direct_representatives.shape[3]) != self.q_indexer_dim
            ):
                return None

        selection_by_seq: dict[int, typing.Any | None] = {}
        representative_parts: list[tuple[int, torch.Tensor, int]] = []
        scored_seq_slices: dict[int, tuple[int, int, int]] = {}
        chunk_top_k_by_seq: dict[int, int] = {}
        top_k_segments: list[tuple[int, int, int]] = []
        min_context_len: int | None = None
        max_context_len = 0
        min_requested_top_k: int | None = None
        sparse_row_count = 0
        max_chunks = 0
        max_q_len = 0
        max_prior_chunks = 0
        max_top_k = 0
        total_tiles = 0
        decode_tiles = 0
        small_tiles = 0
        large_tiles = 0
        max_tiles_per_row_plan = 0
        (
            small_block_rows,
            large_block_rows,
            block_chunks,
            decode_block_chunks,
        ) = dsa_score_tile_plan_config()
        block_chunks_minus_one = block_chunks - 1
        decode_block_chunks_minus_one = decode_block_chunks - 1
        small_block_rows_minus_one = small_block_rows - 1
        large_block_rows_minus_one = large_block_rows - 1

        for (
            seq_idx,
            q_start,
            q_end,
            _,
            num_chunks,
            query_position_start,
            current_chunks,
        ) in sparse_infos:
            q_len = q_end - q_start
            if q_len <= 0 or q_start < 0 or q_end > num_query_rows:
                return None
            max_prior_chunks_for_seq = max(num_chunks - 1, 0)
            seq_top_k_segments = self.recall_policy.top_k_segments(
                query_position_start=query_position_start,
                query_len=q_len,
                maximum_top_k=max_prior_chunks_for_seq,
            )
            chunk_top_k = max(
                (segment_top_k for _, _, segment_top_k in seq_top_k_segments),
                default=0,
            )
            if max_prior_chunks_for_seq <= 0 or chunk_top_k <= 0:
                selection_by_seq[seq_idx] = None
                continue
            for segment_start, segment_end, segment_top_k in seq_top_k_segments:
                top_k_segments.append(
                    (
                        q_start + segment_start,
                        q_start + segment_end,
                        segment_top_k,
                    )
                )
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
            max_context_len = max(max_context_len, query_position_start + q_len)
            sparse_plan_idx = len(scored_seq_slices)

            if direct_representatives is not None:
                assert direct_local_by_seq is not None
                assert direct_num_chunks_by_seq is not None
                local_seq_idx = direct_local_by_seq.get(seq_idx)
                if local_seq_idx is None:
                    return None
                expected_local_seq_idx = (
                    seq_idx if direct_seq_id_layout == "original" else sparse_plan_idx
                )
                if local_seq_idx != expected_local_seq_idx:
                    return None
                if local_seq_idx >= int(direct_representatives.shape[0]):
                    return None
                if direct_num_chunks_by_seq.get(seq_idx) != num_chunks:
                    return None
            else:
                representatives = batched_chunk_representatives.get(seq_idx)
                if representatives is None or representatives.dim() != 3:
                    return None
                if (
                    int(representatives.shape[0]) != num_chunks
                    or int(representatives.shape[1]) < num_groups
                    or int(representatives.shape[2]) != self.q_indexer_dim
                ):
                    return None
                local_seq_idx = len(representative_parts)
                if local_seq_idx != sparse_plan_idx:
                    return None
                representative_parts.append(
                    (local_seq_idx, representatives, num_chunks)
                )
            if current_chunks is not None and (
                current_chunks.dim() != 1 or int(current_chunks.shape[0]) != q_len
            ):
                return None

            if q_len == 1:
                row_decode_tiles = (
                    max_prior_chunks_for_seq + decode_block_chunks_minus_one
                ) // decode_block_chunks
                row_small_tiles = 0
                row_large_tiles = 0
                tile_count = row_decode_tiles
            elif q_len <= small_block_rows:
                row_decode_tiles = 0
                row_small_tiles = (
                    (q_len + small_block_rows_minus_one) // small_block_rows
                ) * (
                    (max_prior_chunks_for_seq + block_chunks_minus_one) // block_chunks
                )
                row_large_tiles = 0
                tile_count = row_small_tiles
            else:
                row_decode_tiles = 0
                row_small_tiles = 0
                row_large_tiles = (
                    (q_len + large_block_rows_minus_one) // large_block_rows
                ) * (
                    (max_prior_chunks_for_seq + block_chunks_minus_one) // block_chunks
                )
                tile_count = row_large_tiles
            total_tiles += tile_count
            decode_tiles += row_decode_tiles
            small_tiles += row_small_tiles
            large_tiles += row_large_tiles
            max_tiles_per_row_plan = max(max_tiles_per_row_plan, tile_count)
            scored_seq_slices[seq_idx] = (q_start, q_end, chunk_top_k)
            chunk_top_k_by_seq[seq_idx] = chunk_top_k
            sparse_row_count += q_len
            max_chunks = max(max_chunks, num_chunks)
            max_q_len = max(max_q_len, q_len)
            max_prior_chunks = max(max_prior_chunks, max_prior_chunks_for_seq)
            max_top_k = max(max_top_k, chunk_top_k)

        if direct_representatives is None and not representative_parts:
            return selection_by_seq
        if max_prior_chunks <= 0 or max_top_k <= 0:
            return selection_by_seq

        if direct_representatives is None:
            padded_representatives = torch.empty(
                len(representative_parts),
                max_chunks,
                num_groups,
                self.q_indexer_dim,
                device=device,
                dtype=torch.float32,
            )
            for local_seq_idx, representatives, num_chunks in representative_parts:
                padded_representatives[local_seq_idx, :num_chunks].copy_(
                    representatives[:num_chunks, :num_groups].to(
                        device=device,
                        dtype=torch.float32,
                    )
                )
                if num_chunks < max_chunks:
                    padded_representatives[local_seq_idx, num_chunks:].zero_()
            score_representatives = padded_representatives
        else:
            score_representatives = direct_representatives
        score_query_states = indexer_q[:num_query_rows, 0]
        metadata_plan = dsa_build_score_metadata_triton(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            num_actual_tokens=int(num_actual_tokens),
            active_seq_count=int(active_seq_count),
            num_sparse_plans=len(scored_seq_slices),
            total_rows=num_query_rows,
            chunk_size=self.chunk_size,
            representative_group_idx=representative_group_idx,
            dense_decode_threshold=int(dense_decode_threshold),
            dense_prefill_threshold=int(dense_prefill_threshold),
            chunk_top_k=(
                max_top_k
                if self.q_indexer_dynamic_chunk_top_k
                else self.q_indexer_chunk_top_k
            ),
            max_q_len=max_q_len,
            representatives_use_original_seq_ids=(
                direct_representatives is not None
                and direct_seq_id_layout == "original"
            ),
            small_block_rows=small_block_rows,
            large_block_rows=large_block_rows,
            block_chunks=block_chunks,
            decode_block_chunks=decode_block_chunks,
        )
        if metadata_plan is None:
            return None
        row_plan, row_metadata = metadata_plan
        (
            score_row_seq_ids,
            row_seq_ids,
            row_group_ids,
            row_num_prior_chunks,
            row_current_chunks,
            row_tail_lens,
        ) = row_metadata

        score_current_chunks = row_current_chunks
        row_top_k = None
        score_top_k_segments = None
        if self.q_indexer_dynamic_chunk_top_k or self.q_indexer_recent_window_pages:
            score_current_chunks = torch.zeros_like(row_current_chunks)
            for (
                _seq_idx,
                q_start,
                q_end,
                _key_len,
                _num_chunks,
                _query_position_start,
                _current_chunks,
            ) in sparse_infos:
                score_current_chunks[q_start:q_end].copy_(
                    self._dsa_remote_current_chunks(row_current_chunks[q_start:q_end])
                )
        if self.q_indexer_dynamic_chunk_top_k:
            row_top_k = torch.zeros_like(row_current_chunks, dtype=torch.int32)
            for segment_start, segment_end, segment_top_k in top_k_segments:
                row_top_k[segment_start:segment_end].fill_(segment_top_k)
            score_top_k_segments = top_k_segments

        score_tile_plan = dsa_build_score_tile_plan_triton(
            row_plan_with_tiles=row_plan,
            total_tiles=total_tiles,
            max_tiles_per_row_plan=max_tiles_per_row_plan,
            small_block_rows=small_block_rows,
            large_block_rows=large_block_rows,
            block_chunks=block_chunks,
            decode_block_chunks=decode_block_chunks,
        )
        if score_tile_plan is None:
            return None
        score_topk = dsa_batched_score_topk_tile_plan_triton(
            score_query_states=score_query_states,
            chunk_representatives=score_representatives,
            tile_plan=score_tile_plan,
            current_chunks=score_current_chunks,
            row_num_prior_chunks=row_num_prior_chunks,
            total_rows=num_query_rows,
            chunk_size=self.chunk_size,
            chunk_top_k=max_top_k,
            logit_scale=self.q_indexer_logit_scale,
            q_indexer_dim=self.q_indexer_dim,
            max_prior_chunks=max_prior_chunks,
            small_block_rows=small_block_rows,
            large_block_rows=large_block_rows,
            block_chunks=block_chunks,
            decode_block_chunks=decode_block_chunks,
            row_top_k=row_top_k,
            top_k_segments=score_top_k_segments,
        )
        log_recall_plan(
            "batched_selection",
            context_min=min_context_len,
            context_max=max_context_len,
            remote_top_k_min=min_requested_top_k,
            remote_top_k_max=max_top_k,
            recent_window_pages=self.q_indexer_recent_window_pages,
            policy_segments=len(top_k_segments),
            rows=sparse_row_count,
        )
        _dsa_log_path_marker(
            "triton_batched_score_tile_plan",
            decode_tiles=decode_tiles,
            large_tiles=large_tiles,
            rows=num_query_rows,
            sparse_rows=sparse_row_count,
            sparse_seqs=len(scored_seq_slices),
            small_tiles=small_tiles,
            tiles=total_tiles,
        )
        if score_topk is None:
            return None
        selected_block_indices, selected_block_counts, _ = score_topk

        dsa_plan_state = _EfficientDSAPlanState(
            row_plan=row_plan,
            tile_plan=score_tile_plan,
            score_row_seq_ids=score_row_seq_ids,
            row_seq_ids=row_seq_ids,
            row_group_ids=row_group_ids,
            row_num_prior_chunks=row_num_prior_chunks,
            row_current_chunks=row_current_chunks,
            row_tail_lens=row_tail_lens,
            total_rows=num_query_rows,
            total_tiles=total_tiles,
            decode_tiles=decode_tiles,
            small_tiles=small_tiles,
            large_tiles=large_tiles,
            max_q_len=max_q_len,
            max_prior_chunks=max_prior_chunks,
            max_top_k=max_top_k,
            max_tiles_per_row_plan=max_tiles_per_row_plan,
        )

        for seq_idx, (start, end, chunk_top_k) in scored_seq_slices.items():
            selection_by_seq[seq_idx] = _EfficientChunkBlockSelection(
                selected_block_indices=selected_block_indices[start:end, :chunk_top_k],
                selected_block_counts=selected_block_counts[start:end],
            )

        batched_selection_by_seq = _EfficientBatchedChunkBlockSelections(
            selected_block_indices=selected_block_indices,
            selected_block_valid=None,
            selected_block_counts=selected_block_counts,
            seq_slices=scored_seq_slices,
            chunk_top_k_by_seq=chunk_top_k_by_seq,
            row_seq_ids=row_seq_ids,
            row_current_chunks=row_current_chunks,
            row_tail_lens=row_tail_lens,
            per_seq=selection_by_seq,
            dsa_plan_state=dsa_plan_state,
        )
        _dsa_log_path_marker(
            "triton_batched_scoring",
            max_prior_chunks=max_prior_chunks,
            rows=int(selected_block_indices.shape[0]),
            seqs=len(scored_seq_slices),
            top_k=max_top_k,
        )
        return batched_selection_by_seq

    def try_build_page_tables_batched(
        self,
        *,
        block_table: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]] | None = None,
        sparse_infos: list[tuple[int, int, int, int, int, int, torch.Tensor | None]],
        block_selection_by_seq: dict[int, typing.Any | None],
        total_rows: int,
        device: torch.device,
        query_start_loc: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        num_actual_tokens: int | None = None,
        active_seq_count: int | None = None,
        dense_decode_threshold: int | None = None,
        dense_prefill_threshold: int | None = None,
        **_: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
        if dsa_batched_unified_page_table_triton is None:
            return None
        if (
            block_table.dim() != 2
            or block_table.device != device
            or total_rows <= 0
            or active_seq_infos is None
            or query_start_loc is None
            or seq_lens is None
            or num_actual_tokens is None
            or active_seq_count is None
            or dense_decode_threshold is None
            or dense_prefill_threshold is None
        ):
            return None
        if (
            query_start_loc.device != device
            or seq_lens.device != device
            or query_start_loc.dim() != 1
            or seq_lens.dim() != 1
            or int(active_seq_count) != len(active_seq_infos)
            or int(num_actual_tokens) != total_rows
        ):
            return None
        if not isinstance(
            block_selection_by_seq,
            _EfficientBatchedChunkBlockSelections,
        ):
            return None
        selected_blocks_t = block_selection_by_seq._selected_block_indices
        selected_counts_t = block_selection_by_seq._selected_block_counts
        if selected_counts_t is None:
            return None
        if (
            selected_blocks_t.device != device
            or selected_counts_t.device != device
            or selected_blocks_t.dim() != 2
            or selected_counts_t.dim() != 1
            or int(selected_blocks_t.shape[0]) < total_rows
            or int(selected_counts_t.shape[0]) < total_rows
        ):
            return None

        max_top_width = int(selected_blocks_t.shape[1])
        recent_window_pages = self.q_indexer_recent_window_pages
        expected_q_start = 0
        num_requests = 0
        dense_requests = 0
        sparse_requests = 0
        crossing_requests = 0
        max_dense_pages = 0
        max_source_q_len = 0
        max_seqlen_q = 0
        max_seqlen_k = 0
        block_table_rows = int(block_table.shape[0])
        block_table_width = int(block_table.shape[1])
        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            if q_start != expected_q_start:
                return None
            expected_q_start = q_end
            if (
                seq_idx < 0
                or seq_idx >= block_table_rows
                or key_len <= 0
                or key_len - q_len < 0
            ):
                return None
            max_source_q_len = max(max_source_q_len, q_len)
            dense_threshold = (
                int(dense_prefill_threshold)
                if q_len > 1
                else int(dense_decode_threshold)
            )
            query_position_start = key_len - q_len
            dense_prefix_len = (
                min(q_len, max(dense_threshold - query_position_start, 0))
                if dense_threshold >= 0
                else 0
            )
            if dense_prefix_len > 0:
                dense_key_len = query_position_start + dense_prefix_len
                dense_pages = math.ceil(dense_key_len / self.chunk_size)
                if dense_pages > block_table_width:
                    return None
                num_requests += 1
                dense_requests += 1
                max_dense_pages = max(max_dense_pages, dense_pages)
                max_seqlen_q = max(max_seqlen_q, dense_prefix_len)
                max_seqlen_k = max(max_seqlen_k, dense_key_len)
            sparse_len = q_len - dense_prefix_len
            if sparse_len <= 0:
                continue
            if dense_prefix_len > 0:
                crossing_requests += 1
                log_recall_plan(
                    "dense_sparse_boundary_split",
                    dense_tokens=dense_threshold,
                    context_start=query_position_start + 1,
                    context_end=key_len,
                    dense_rows=dense_prefix_len,
                    sparse_rows=sparse_len,
                    first_sparse_context=dense_threshold + 1,
                    first_sparse_top_k=self._dsa_chunk_top_k_for_context(
                        dense_threshold + 1
                    ),
                    recent_window_pages=recent_window_pages,
                    qshare_group_size=1,
                    backend="triton_batched_unified_page_table",
                )
            num_requests += sparse_len
            sparse_requests += sparse_len
            max_seqlen_q = max(max_seqlen_q, 1)
            max_seqlen_k = max(
                max_seqlen_k,
                self.block_table_provider._sparse_suffix_max_seqused_k(
                    query_position_start=query_position_start + dense_prefix_len,
                    key_len=key_len,
                    chunk_size=self.chunk_size,
                    top_width=max_top_width + recent_window_pages,
                ),
            )
        if expected_q_start != total_rows or num_requests <= 0:
            return None
        table_width = max(
            max_top_width + recent_window_pages + 1,
            max_dense_pages,
        )
        table_elems = num_requests * table_width
        if table_width <= 0:
            return None

        page_table_plan = dsa_batched_unified_page_table_triton(
            block_table=block_table,
            selected_blocks=selected_blocks_t,
            selected_counts=selected_counts_t,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            num_actual_tokens=int(num_actual_tokens),
            active_seq_count=int(active_seq_count),
            num_requests=num_requests,
            table_width=table_width,
            max_q_len=max_source_q_len,
            chunk_size=self.chunk_size,
            dense_decode_threshold=int(dense_decode_threshold),
            dense_prefill_threshold=int(dense_prefill_threshold),
            recent_window_pages=recent_window_pages,
        )
        if page_table_plan is None:
            return None
        page_table, seqused_k, cu_seqlens_q = page_table_plan
        _dsa_log_path_marker(
            "triton_batched_unified_page_table",
            dense=max_dense_pages,
            rows=num_requests,
            seqs=len(active_seq_infos),
            sparse_top_k=max_top_width,
            recent_window_pages=recent_window_pages,
        )
        log_recall_plan(
            "batched_page_table",
            remote_top_k_width=max_top_width,
            recent_window_pages=recent_window_pages,
            table_width=table_width,
            rows=num_requests,
        )
        if crossing_requests:
            _dsa_log_path_marker(
                "dense_sparse_prefill_page_table_bucket",
                crossings=crossing_requests,
            )
        if dense_requests:
            _dsa_log_path_marker(
                "dense_prefill_page_table_bucket",
                dense_requests=dense_requests,
                max_seqlen_k=max_seqlen_k,
                max_seqlen_q=max_seqlen_q,
                num_requests=num_requests,
                table_elems=table_elems,
            )
        if sparse_requests:
            _dsa_log_path_marker(
                "sparse_prefill_page_table_bucket",
                max_seqlen_k=max_seqlen_k,
                max_seqlen_q=max_seqlen_q,
                num_requests=num_requests,
                sparse_requests=sparse_requests,
                table_elems=table_elems,
            )
        if sparse_requests and max_seqlen_q == 1:
            _dsa_log_path_marker(
                "sparse_decode",
                decode_requests=sparse_requests,
                max_seqlen_k=max_seqlen_k,
                max_seqlen_q=max_seqlen_q,
                num_requests=num_requests,
                table_elems=table_elems,
            )
        return page_table, cu_seqlens_q, seqused_k, max_seqlen_q, max_seqlen_k

    def select_blocks(
        self,
        *,
        score_query_states: torch.Tensor,
        representative_state: typing.Any,
        current_chunks: torch.Tensor,
        max_prior_chunks: int,
        block_top_k: int,
        block_table: torch.Tensor | None = None,
        chunk_ids: torch.Tensor | None = None,
        seq_idx: int | None = None,
        group_idx: int | None = None,
        **kwargs: typing.Any,
    ) -> typing.Any | None:
        if max_prior_chunks <= 0 or block_top_k <= 0:
            return None
        score_state = self.scoring_provider(
            score_query_states=score_query_states,
            representative_state=representative_state,
            current_chunks=current_chunks,
            max_prior_chunks=max_prior_chunks,
            chunk_ids=chunk_ids,
            seq_idx=seq_idx,
            group_idx=group_idx,
            block_table=block_table,
            **kwargs,
        )
        if not self.scoring_provider.is_available(score_state):
            raise ValueError("DSA scoring provider is unavailable")
        selection_state = self.block_selection_provider(
            score_state=score_state,
            block_top_k=block_top_k,
            block_table=block_table,
            current_chunks=current_chunks,
            max_prior_chunks=max_prior_chunks,
            chunk_ids=chunk_ids,
            seq_idx=seq_idx,
            group_idx=group_idx,
            **kwargs,
        )
        if not self.block_selection_provider.is_available(selection_state):
            raise ValueError("DSA block selection provider is unavailable")
        return selection_state

    def get_selected_blocks(
        self,
        selection_state: typing.Any,
        **kwargs: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selection = self.block_selection_provider.get_selected_blocks(
            selection_state,
            **kwargs,
        )
        if selection is None:
            raise ValueError("DSA block selection provider is unavailable")
        return selection

    def build_page_table_plan(
        self,
        *,
        block_table: torch.Tensor,
        chunk_size: int,
        key_len: int,
        **kwargs: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int] | None:
        block_table_state = self.block_table_provider(
            block_table=block_table,
            chunk_size=chunk_size,
            key_len=key_len,
            recent_window_pages=self.q_indexer_recent_window_pages,
            **kwargs,
        )
        return self.block_table_provider.get_page_table(block_table_state)
