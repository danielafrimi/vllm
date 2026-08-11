# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PyTorch components for Nemotron-H chunked DSA."""

from __future__ import annotations

import math
import os
import sys
import typing

import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.models.nemotron_h_dsa_recall_policy import (
    RecallPolicyProvider,
    log_recall_config,
    log_recall_plan,
    make_recall_policy_provider,
)

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None


_SequenceSkipFn = typing.Callable[[int, int, int, int], bool]
_DSA_PAGE_TABLE_FA_ENV = "VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA"
_DSA_PREFILL_PAGE_TABLE_FA_ENV = "VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA"
_DSA_FULL_ATTN_SHORT_SEQ_ENV = "VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ"
_DSA_FLATTENED_PREFILL_PAGE_TABLE_FA_ENV = (
    "VLLM_NEMOTRON_H_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA"
)
_DSA_FLATTENED_DECODE_PAGE_TABLE_FA_ENV = (
    "VLLM_NEMOTRON_H_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA"
)
_DSA_DENSE_PREFILL_KV_THRESHOLD_ENV = (
    "VLLM_NEMOTRON_H_DSA_DENSE_PREFILL_KV_THRESHOLD_TOKENS"
)
_DSA_TRITON_SCORING_ENV = "VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    return default if value is None else value == "1"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _resolve_dsa_symbol(name: str, default):
    parent = sys.modules.get("vllm.model_executor.models.nemotron_h")
    return getattr(parent, name, default) if parent is not None else default


def _get_flash_attn_varlen_func():
    return _resolve_dsa_symbol("flash_attn_varlen_func", flash_attn_varlen_func)


def _get_dsa_kv_cache_layout() -> str:
    parent = sys.modules.get("vllm.model_executor.models.nemotron_h")
    if parent is not None and "_get_dsa_kv_cache_layout" in vars(parent):
        return typing.cast(str, vars(parent)["_get_dsa_kv_cache_layout"]())
    try:
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        return get_kv_cache_layout()
    except AssertionError:
        return "NHD"


def normalize_packed_nhd_kv_cache(
    cache: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Restore the explicit NHD view used by the v0.22 DSA providers.

    Current vLLM's NHD cache can pack the head and head-dimension axes into a
    three-dimensional ``[num_blocks, block_size, num_kv_heads * head_dim]``
    tensor.  DSA's provider code indexes heads explicitly, so expose those
    logical axes without copying the cache payload.
    """
    if cache.dim() != 3:
        return cache
    packed_head_dim = num_kv_heads * head_dim
    if int(cache.shape[-1]) != packed_head_dim:
        raise NotImplementedError(
            "DSA 3D KV cache must use packed NHD layout, "
            f"got shape={cache.shape}, num_kv_heads={num_kv_heads}, "
            f"head_dim={head_dim}"
        )
    return cache.unflatten(-1, (num_kv_heads, head_dim))


class _UnavailableRepresentatives:
    pass


_UNAVAILABLE = _UnavailableRepresentatives()


class _TorchChunkRepresentatives:
    __slots__ = ("_by_seq", "_single")

    def __init__(
        self,
        *,
        single: torch.Tensor | None = None,
        by_seq: dict[int, torch.Tensor] | None = None,
    ) -> None:
        self._single = single
        self._by_seq = by_seq


class _TorchChunkScores:
    __slots__ = ("_chunk_logits", "_chunk_valid")

    def __init__(
        self,
        *,
        chunk_logits: torch.Tensor,
        chunk_valid: torch.Tensor,
    ) -> None:
        self._chunk_logits = chunk_logits
        self._chunk_valid = chunk_valid


class _TorchChunkBlockSelection:
    __slots__ = ("_selected_block_indices", "_selected_block_valid")

    def __init__(
        self,
        *,
        selected_block_indices: torch.Tensor,
        selected_block_valid: torch.Tensor,
    ) -> None:
        self._selected_block_indices = selected_block_indices
        self._selected_block_valid = selected_block_valid


class _TorchChunkBlockTable:
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


class TorchChunkedDSARepresentativeProvider(nn.Module):
    """Reference chunk representative provider for Nemotron-H DSA.

    Inputs are intentionally keyword-only and permissive. Callers may pass the
    whole DSA runtime envelope; this provider consumes only the fields it needs.
    The returned value is opaque and should be interpreted through
    ``get_for_sequence``.
    """

    def __init__(
        self,
        *,
        q_indexer_dim: int,
        chunk_size: int,
        num_kv_heads: int,
        head_dim: int | None = None,
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
        self.head_dim = head_dim

    def forward(
        self,
        *,
        key_states: torch.Tensor | None = None,
        key_cache: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        key_len: int | None = None,
        active_seq_infos: list[tuple[int, int, int, int]] | None = None,
        should_skip_sequence: _SequenceSkipFn | None = None,
        **_: typing.Any,
    ) -> typing.Any:
        if active_seq_infos is not None:
            if key_cache is None or block_table is None:
                return _UNAVAILABLE
            if block_table.dim() != 2:
                return _UNAVAILABLE
            by_seq: dict[int, torch.Tensor] = {}
            for seq_idx, q_start, q_end, seq_key_len in active_seq_infos:
                if should_skip_sequence is not None and should_skip_sequence(
                    seq_idx,
                    q_start,
                    q_end,
                    seq_key_len,
                ):
                    continue
                if seq_idx >= int(block_table.shape[0]):
                    return _UNAVAILABLE
                seq_key_states = self._gather_kv_sequence(
                    key_cache,
                    block_table[seq_idx],
                    seq_key_len,
                )
                by_seq[seq_idx] = self._build_chunk_representatives(
                    seq_key_states[..., : self.q_indexer_dim]
                )
            return _TorchChunkRepresentatives(by_seq=by_seq)

        if key_len is None:
            if key_states is None:
                return _UNAVAILABLE
            key_len = int(key_states.shape[0])
        if key_states is None:
            if key_cache is None or block_table is None:
                return _UNAVAILABLE
            key_states = self._gather_kv_sequence(
                key_cache,
                block_table,
                key_len,
            )
        return _TorchChunkRepresentatives(
            single=self._build_chunk_representatives(
                key_states[..., : self.q_indexer_dim]
            )
        )

    def is_available(self, result: typing.Any) -> bool:
        return result is not _UNAVAILABLE

    def get_for_sequence(
        self,
        result: typing.Any,
        *,
        seq_idx: int | None = None,
        **_: typing.Any,
    ) -> torch.Tensor | None:
        if result is _UNAVAILABLE:
            return None
        if not isinstance(result, _TorchChunkRepresentatives):
            raise TypeError(f"unexpected representative result: {type(result)!r}")
        if result._by_seq is not None:
            if seq_idx is None:
                raise ValueError("seq_idx is required for batched representatives")
            return result._by_seq.get(seq_idx)
        return result._single

    def _gather_kv_sequence(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        key_len: int,
    ) -> torch.Tensor:
        head_dim = self.head_dim if self.head_dim is not None else cache.shape[-1]
        cache = normalize_packed_nhd_kv_cache(
            cache,
            num_kv_heads=self.num_kv_heads,
            head_dim=head_dim,
        )
        if cache.dim() != 4:
            raise NotImplementedError(
                f"DSA cache gather expects a 4D KV cache, got {cache.shape}"
            )
        if key_len == 0:
            return cache.new_empty(0, self.num_kv_heads, head_dim)
        if cache.shape[2] == self.num_kv_heads:
            block_size = int(cache.shape[1])
            cache_layout = "NHD"
        elif cache.shape[1] == self.num_kv_heads:
            block_size = int(cache.shape[2])
            cache_layout = "HND"
        else:
            raise NotImplementedError(
                "DSA cache gather only supports NHD/HND KV cache layouts, "
                f"got shape={cache.shape}, num_kv_heads={self.num_kv_heads}"
            )

        if block_table.device != cache.device:
            block_table = block_table.to(device=cache.device)
        token_indices = torch.arange(key_len, device=cache.device, dtype=torch.long)
        block_indices = torch.div(
            token_indices,
            block_size,
            rounding_mode="floor",
        )
        block_offsets = token_indices.remainder(block_size)
        block_ids = block_table.index_select(0, block_indices).to(torch.long)
        if cache_layout == "NHD":
            return cache[block_ids, block_offsets]
        return cache[block_ids, :, block_offsets]

    def _build_chunk_representatives(
        self,
        indexer_key_states: torch.Tensor,
    ) -> torch.Tensor:
        key_len = int(indexer_key_states.shape[0])
        chunk_size = self.chunk_size
        num_chunks = math.ceil(key_len / chunk_size) if key_len > 0 else 0
        if num_chunks == 0:
            return indexer_key_states.new_empty(
                0,
                self.num_kv_heads,
                self.q_indexer_dim,
                dtype=torch.float32,
            )
        padded_len = num_chunks * chunk_size
        if padded_len != key_len:
            padding = indexer_key_states.new_zeros(
                padded_len - key_len,
                self.num_kv_heads,
                self.q_indexer_dim,
            )
            indexer_key_states = torch.cat((indexer_key_states, padding), dim=0)

        chunked_keys = indexer_key_states.view(
            num_chunks,
            chunk_size,
            self.num_kv_heads,
            self.q_indexer_dim,
        )
        chunk_sums = chunked_keys.float().sum(dim=1)
        # Keep this entirely on the device: assigning a Python scalar into the
        # final CUDA element is an implicit CPU-to-GPU copy, which invalidates
        # vLLM's CUDA graph capture.
        chunk_starts = torch.arange(
            num_chunks,
            device=indexer_key_states.device,
            dtype=torch.int64,
        ) * chunk_size
        chunk_lengths = (key_len - chunk_starts).clamp_(max=chunk_size).to(
            dtype=chunk_sums.dtype
        )
        return chunk_sums / chunk_lengths[:, None, None]


class TorchChunkedDSAScoringProvider(nn.Module):
    """Reference chunk scoring provider for Nemotron-H DSA.

    This component only scores chunks. It does not select top-k chunks; callers
    pass the opaque result to a selector stage.
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
            return _TorchChunkScores(
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
            return _TorchChunkScores(
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

        chunk_logits = torch.matmul(
            score_query_states.float(),
            representatives.transpose(0, 1),
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
        return _TorchChunkScores(
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
        if not isinstance(result, _TorchChunkScores):
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
        elif isinstance(representative_state, _TorchChunkRepresentatives):
            if representative_state._by_seq is not None:
                if seq_idx is None:
                    return None
                representatives = representative_state._by_seq.get(seq_idx)
            else:
                representatives = representative_state._single
        elif isinstance(representative_state, torch.Tensor):
            representatives = representative_state
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


class TorchTopKChunkedDSABlockSelectionProvider(nn.Module):
    """Reference top-k logical-block selector for Nemotron-H DSA.

    This component consumes opaque score state and returns opaque selection
    state. Today the selection algorithm is top-k over scored chunks, but the
    provider boundary is block-selection-shaped so later selectors can choose
    blocks using different algorithms or metadata.
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
            return _TorchChunkBlockSelection(
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
        return _TorchChunkBlockSelection(
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
        if not isinstance(result, _TorchChunkBlockSelection):
            raise TypeError(f"unexpected block selection result: {type(result)!r}")
        return result._selected_block_indices, result._selected_block_valid

    def _materialize_scores(
        self,
        score_state: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not isinstance(score_state, _TorchChunkScores):
            return None
        return score_state._chunk_logits, score_state._chunk_valid


class TorchChunkedDSABlockTableProvider(nn.Module):
    """Reference logical-to-physical block-table builder for DSA page FA."""

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
        selection_query_state: typing.Any | None = None,
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
                recent_window_pages=recent_window_pages,
            )
        if q_len is None or current_chunks is None or query_position_start is None:
            return _UNAVAILABLE
        if selection_query_state is not None:
            return self._build_sparse_prefill_runs(
                block_table=block_table,
                selection_state=selection_state,
                selection_query_state=selection_query_state,
                query_position_start=query_position_start,
                q_len=q_len,
                key_len=key_len,
                chunk_size=chunk_size,
                recent_window_pages=recent_window_pages,
            )
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

    def _build_sparse_prefill_runs(
        self,
        *,
        block_table: torch.Tensor,
        selection_state: typing.Any | None,
        selection_query_state: typing.Any,
        query_position_start: int,
        q_len: int,
        key_len: int,
        chunk_size: int,
        recent_window_pages: int,
    ) -> typing.Any:
        run_starts = selection_query_state.run_starts
        run_counts = selection_query_state.run_counts
        current_chunks = selection_query_state.reduced_current_chunks
        if (
            run_starts.dim() != 1
            or run_counts.dim() != 1
            or current_chunks.dim() != 1
            or run_starts.shape != run_counts.shape
            or run_starts.shape != current_chunks.shape
            or int(run_counts.sum().item()) != q_len
            or q_len <= 0
            or query_position_start < 0
        ):
            return _UNAVAILABLE
        runs = int(run_starts.numel())
        device = current_chunks.device
        if block_table.device != device:
            block_table = block_table.to(device=device)
        selection = self._materialize_selection(
            selection_state=selection_state,
            rows=runs,
            device=device,
        )
        if selection is None:
            return _UNAVAILABLE
        selected_blocks, selected_valid = selection
        if (
            selected_blocks.shape != selected_valid.shape
            or selected_blocks.dim() != 2
            or int(selected_blocks.shape[0]) != runs
        ):
            return _UNAVAILABLE
        if recent_window_pages > 0:
            selected_blocks, selected_valid = self._compact_selection_prefix(
                selected_blocks,
                selected_valid,
            )

        valid_counts = selected_valid.sum(dim=-1).to(dtype=torch.long)
        top_width = int(selected_blocks.shape[1])
        run_end_positions = query_position_start + run_starts + run_counts - 1
        run_end_chunks = torch.div(
            run_end_positions,
            chunk_size,
            rounding_mode="floor",
        )
        recent_counts = current_chunks.clamp(
            min=0,
            max=recent_window_pages,
        )
        local_page_counts = run_end_chunks - current_chunks + 1
        max_local_pages = int(local_page_counts.max().item())
        table_width = top_width + recent_window_pages + max_local_pages
        logical_pages = current_chunks[:, None].expand(runs, table_width).clone()
        if top_width > 0:
            logical_pages[:, :top_width] = selected_blocks.to(dtype=torch.long)
        recent_offsets = torch.arange(
            recent_window_pages,
            device=device,
            dtype=torch.long,
        )
        if recent_window_pages > 0:
            recent_columns = valid_counts[:, None] + recent_offsets[None, :]
            safe_recent_offsets = torch.minimum(
                recent_offsets[None, :],
                (recent_counts[:, None] - 1).clamp_min(0),
            )
            logical_pages.scatter_(
                1,
                recent_columns,
                current_chunks[:, None] - recent_counts[:, None] + safe_recent_offsets,
            )
        local_offsets = torch.arange(
            max_local_pages,
            device=device,
            dtype=torch.long,
        )
        local_columns = (
            valid_counts[:, None] + recent_counts[:, None] + local_offsets[None, :]
        )
        logical_pages.scatter_(
            1,
            local_columns,
            current_chunks[:, None]
            + torch.minimum(
                local_offsets[None, :],
                local_page_counts[:, None] - 1,
            ),
        )
        if logical_pages.numel() > 0:
            max_logical_page = int(logical_pages.max().item())
            if max_logical_page >= int(block_table.shape[0]):
                return _UNAVAILABLE

        physical_pages = (
            block_table.to(dtype=torch.long)
            .expand(runs, -1)
            .gather(
                1,
                logical_pages,
            )
            .to(torch.int32)
        )
        used_page_mask = (
            torch.arange(
                table_width,
                device=device,
                dtype=torch.long,
            )[None, :]
            < (valid_counts + recent_counts + local_page_counts)[:, None]
        )
        physical_pages.masked_fill_(~used_page_mask, 0)
        local_prefixes = run_end_positions - current_chunks * chunk_size + 1
        seqused_k = ((valid_counts + recent_counts) * chunk_size + local_prefixes).to(
            torch.int32
        )
        return _TorchChunkBlockTable(
            block_table=physical_pages,
            request_lens=run_counts.to(torch.int32),
            seqused_k=seqused_k,
            max_seqlen_q=int(run_counts.max().item()),
            max_seqlen_k=int(seqused_k.max().item()),
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
        if not isinstance(result, _TorchChunkBlockTable):
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
        recent_window_pages: int,
    ) -> typing.Any:
        if q_len <= 0:
            return _UNAVAILABLE
        num_pages = math.ceil(key_len / chunk_size) if key_len > 0 else 0
        if num_pages > int(block_table.shape[0]):
            return _UNAVAILABLE
        return _TorchChunkBlockTable(
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
        if recent_window_pages > 0:
            selected_blocks, selected_valid = self._compact_selection_prefix(
                selected_blocks,
                selected_valid,
            )

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
        if logical_pages.numel() > 0:
            max_logical_page = int(logical_pages.max().item())
            if max_logical_page >= int(block_table.shape[0]):
                return _UNAVAILABLE

        physical_pages = (
            block_table.to(dtype=torch.long)
            .expand(q_len, -1)
            .gather(
                1,
                logical_pages,
            )
            .to(torch.int32)
        )
        used_page_mask = (
            torch.arange(table_width, device=device, dtype=torch.long)[None, :]
            <= (valid_counts + recent_counts)[:, None]
        )
        physical_pages.masked_fill_(~used_page_mask, 0)

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
        return _TorchChunkBlockTable(
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
        if current_chunks.numel() != 1 or query_positions.numel() != 1:
            return _UNAVAILABLE
        device = current_chunks.device
        if block_table.device != device:
            block_table = block_table.to(device=device)
        if query_positions.device != device:
            query_positions = query_positions.to(device=device)

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

        query_position = int(query_positions[0].item())
        if query_position != key_len - 1 or query_position < 0:
            return _UNAVAILABLE
        current_chunk = int(current_chunks[0].item())
        if query_position >= key_len:
            return _UNAVAILABLE
        valid_selected_blocks = selected_blocks[0].masked_select(selected_valid[0])
        max_page_id = current_chunk
        if valid_selected_blocks.numel() > 0:
            max_page_id = max(max_page_id, int(valid_selected_blocks.max().item()))
        if max_page_id >= int(block_table.shape[0]):
            return _UNAVAILABLE

        current_chunk_t = current_chunks.to(torch.long)
        recent_count = min(recent_window_pages, current_chunk)
        recent_pages = torch.arange(
            current_chunk - recent_count,
            current_chunk,
            device=device,
            dtype=torch.long,
        )
        logical_pages = torch.cat(
            (
                valid_selected_blocks.to(device=device, dtype=torch.long),
                recent_pages,
                current_chunk_t,
            )
        )
        page_table = (
            block_table.index_select(0, logical_pages).to(torch.int32).view(1, -1)
        )
        selected_count = int(valid_selected_blocks.numel()) + recent_count
        tail_len = query_positions.to(torch.long) - current_chunk_t * chunk_size + 1
        seqused_k = (selected_count * chunk_size + tail_len).to(torch.int32)
        max_seqlen_k = int(seqused_k.max().item())
        return _TorchChunkBlockTable(
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
        if not isinstance(selection_state, _TorchChunkBlockSelection):
            return None
        selected_blocks = selection_state._selected_block_indices
        selected_valid = selection_state._selected_block_valid
        if selected_blocks.device != device:
            selected_blocks = selected_blocks.to(device=device)
        if selected_valid.device != device:
            selected_valid = selected_valid.to(device=device)
        return selected_blocks, selected_valid

    @staticmethod
    def _compact_selection_prefix(
        selected_blocks: torch.Tensor,
        selected_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move valid selected pages to the prefix without changing their set."""
        if selected_blocks.shape[-1] == 0:
            return selected_blocks, selected_valid
        order = torch.argsort(
            selected_valid.to(torch.int8),
            dim=-1,
            descending=True,
            stable=True,
        )
        return (
            selected_blocks.gather(-1, order),
            selected_valid.gather(-1, order),
        )

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
            valid_count = min(top_width, chunk_idx)
            max_used = max(max_used, valid_count * chunk_size + local_prefix)
        return max_used


class _ChunkedDSABatchRepresentatives:
    __slots__ = ("_by_seq",)

    def __init__(self, by_seq: dict[int, torch.Tensor] | None) -> None:
        self._by_seq = by_seq


class _ChunkedDSABatchScores:
    __slots__ = ("_representatives",)

    def __init__(
        self,
        representatives: _ChunkedDSABatchRepresentatives,
    ) -> None:
        self._representatives = representatives


class _ChunkedDSABatchSelection:
    __slots__ = ("_scores",)

    def __init__(self, scores: _ChunkedDSABatchScores) -> None:
        self._scores = scores


class _ChunkedDSABatchBlockTables:
    __slots__ = ("_selection",)

    def __init__(self, selection: _ChunkedDSABatchSelection) -> None:
        self._selection = selection


class ChunkedDSAAttentionProviderMixin:
    def _init_common_options(
        self,
        *,
        logit_scale: float,
        chunk_top_k: int,
        query_chunk_size: int,
        num_heads: int,
        total_num_kv_heads: int,
    ) -> None:
        self.q_indexer_logit_scale = logit_scale
        self.q_indexer_chunk_top_k = chunk_top_k
        self.q_indexer_chunked_query_chunk_size = query_chunk_size
        self.recall_policy: RecallPolicyProvider = make_recall_policy_provider(
            chunk_size=self.chunk_size,
            fixed_chunk_top_k=chunk_top_k,
        )
        self.q_indexer_dynamic_chunk_top_k = self.recall_policy.dynamic
        self.q_indexer_recent_window_pages = self.recall_policy.recent_window_pages
        self.q_indexer_use_page_table_fa = _env_bool(_DSA_PAGE_TABLE_FA_ENV)
        self.q_indexer_use_prefill_page_table_fa = _env_bool(
            _DSA_PREFILL_PAGE_TABLE_FA_ENV
        )
        self.q_indexer_use_full_attention_short_seq = _env_bool(
            _DSA_FULL_ATTN_SHORT_SEQ_ENV
        )
        self.q_indexer_use_flattened_prefill_page_table_fa = _env_bool(
            _DSA_FLATTENED_PREFILL_PAGE_TABLE_FA_ENV
        )
        self.q_indexer_use_flattened_decode_page_table_fa = _env_bool(
            _DSA_FLATTENED_DECODE_PAGE_TABLE_FA_ENV
        )
        self.q_indexer_use_triton_scoring = _env_bool(_DSA_TRITON_SCORING_ENV)
        self.num_heads = num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.q_indexer_dense_prefill_kv_threshold_tokens = _env_int(
            _DSA_DENSE_PREFILL_KV_THRESHOLD_ENV,
            self._dsa_dense_attention_budget_tokens(),
        )
        if self.q_indexer_dense_prefill_kv_threshold_tokens <= 0:
            raise ValueError(
                f"{_DSA_DENSE_PREFILL_KV_THRESHOLD_ENV} must be positive: "
                f"{self.q_indexer_dense_prefill_kv_threshold_tokens}"
            )
        log_recall_config(self.recall_policy, owner=type(self).__name__)

    def prepare_representatives(
        self,
        *,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
        active_seq_infos: list[tuple[int, int, int, int]],
        cache_info: tuple[str, int] | None,
        **kwargs: typing.Any,
    ) -> _ChunkedDSABatchRepresentatives:
        by_seq = None
        if self.should_prepare_batched_representatives():

            def should_skip_sequence(
                _seq_idx: int,
                q_start: int,
                q_end: int,
                key_len: int,
            ) -> bool:
                return self._dsa_sequence_fits_dense_attention(
                    key_len,
                    q_end - q_start,
                )

            by_seq = self.get_batched_representatives_by_seq(
                key_cache=key_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                active_seq_infos=active_seq_infos,
                cache_info=cache_info,
                should_skip_sequence=should_skip_sequence,
                **kwargs,
            )
        return _ChunkedDSABatchRepresentatives(by_seq)

    def should_prepare_batched_representatives(self) -> bool:
        return False

    def get_cache_info(self, cache: torch.Tensor) -> tuple[str, int] | None:
        return self._dsa_kv_cache_layout_and_block_size(cache)

    def prepare_scores(
        self,
        *,
        representatives: _ChunkedDSABatchRepresentatives,
    ) -> _ChunkedDSABatchScores:
        return _ChunkedDSABatchScores(representatives)

    def prepare_selection(
        self,
        *,
        scores: _ChunkedDSABatchScores,
    ) -> _ChunkedDSABatchSelection:
        return _ChunkedDSABatchSelection(scores)

    def prepare_block_tables(
        self,
        *,
        selection: _ChunkedDSABatchSelection,
    ) -> _ChunkedDSABatchBlockTables:
        return _ChunkedDSABatchBlockTables(selection)

    def forward_attention(
        self,
        *,
        block_state: _ChunkedDSABatchBlockTables,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn: typing.Any,
        attn_metadata: typing.Any | None,
        positions: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        indexer_q_proj: typing.Callable[
            [torch.Tensor], tuple[torch.Tensor, typing.Any]
        ],
        local_kv_head_indices: torch.Tensor,
        precomputed_indexer_q: torch.Tensor | None = None,
        precomputed_indexer_q_by_head: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        batched_chunk_representatives = (
            block_state._selection._scores._representatives._by_seq
        )
        output = query_states.new_zeros(query_states.shape)
        page_table_handled_seq_indices = (
            self._forward_dsa_chunked_unified_page_table_fa_bucket(
                hidden_states=hidden_states,
                query_states=query_states,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                attn=attn,
                attn_metadata=attn_metadata,
                positions=positions,
                active_seq_infos=active_seq_infos,
                batched_chunk_representatives=batched_chunk_representatives,
                output=output,
                indexer_q_proj=indexer_q_proj,
                local_kv_head_indices=local_kv_head_indices,
                precomputed_indexer_q=precomputed_indexer_q,
                precomputed_indexer_q_by_head=precomputed_indexer_q_by_head,
            )
        )

        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            if seq_idx in page_table_handled_seq_indices:
                continue

            if precomputed_indexer_q is None:
                indexer_q, _ = indexer_q_proj(hidden_states[q_start:q_end])
                indexer_q = indexer_q.view(
                    -1, self.total_num_kv_heads, self.q_indexer_dim
                )
                indexer_q = indexer_q.index_select(
                    1, local_kv_head_indices.to(indexer_q.device)
                )
            else:
                indexer_q = precomputed_indexer_q[q_start:q_end]
            precomputed = (
                batched_chunk_representatives.get(seq_idx)
                if batched_chunk_representatives is not None
                else None
            )
            seq_output = self._forward_dsa_chunked_sequence_with_dense_prefix(
                query_states=query_states[q_start:q_end],
                indexer_query_states=indexer_q,
                key_states=None,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[seq_idx],
                attn=attn,
                attn_metadata=attn_metadata,
                positions=positions[q_start:q_end],
                key_len=key_len,
                chunk_representatives=precomputed,
            )
            output[q_start:q_end] = seq_output

        return output

    def _dsa_dense_attention_budget_tokens(
        self,
        query_len: int | None = None,
    ) -> int:
        default_budget = self.recall_policy.dense_tokens
        if query_len is not None and query_len > 1 and not self.recall_policy.dynamic:
            return getattr(
                self,
                "q_indexer_dense_prefill_kv_threshold_tokens",
                default_budget,
            )
        return default_budget

    def _dsa_chunk_top_k_for_context(self, context_len: int) -> int:
        return self.recall_policy.top_k_for_context(context_len)

    def _dsa_chunk_top_k_for_context_tensor(
        self,
        context_lens: torch.Tensor,
    ) -> torch.Tensor:
        return self.recall_policy.top_k_for_context_tensor(context_lens)

    def _dsa_remote_current_chunks(
        self,
        current_chunks: torch.Tensor,
    ) -> torch.Tensor:
        return self.recall_policy.remote_chunk_counts(current_chunks)

    def _dsa_recent_page_counts(
        self,
        current_chunks: torch.Tensor,
    ) -> torch.Tensor:
        return self.recall_policy.recent_page_counts(current_chunks)

    def _dsa_chunked_query_tile_end(
        self,
        *,
        query_start: int,
        query_len: int,
        first_query_position: int,
    ) -> int:
        query_end = self.recall_policy.query_tile_end(
            query_start=query_start,
            query_len=query_len,
            first_query_position=first_query_position,
            query_chunk_size=self.q_indexer_chunked_query_chunk_size,
        )
        if self.q_indexer_recent_window_pages > 0 and getattr(
            self, "qshare_enabled", False
        ):
            query_position = first_query_position + query_start
            page_end_position = (
                (query_position // self.chunk_size) + 1
            ) * self.chunk_size
            query_end = min(
                query_end,
                page_end_position - first_query_position,
            )
        return max(query_start + 1, query_end)

    def _dsa_sequence_fits_dense_attention(
        self,
        key_len: int,
        query_len: int | None = None,
    ) -> bool:
        budget = self._dsa_dense_attention_budget_tokens(query_len=query_len)
        return (
            getattr(self, "q_indexer_use_full_attention_short_seq", False)
            and key_len <= budget
        )

    def _dsa_dense_query_prefix_len(self, *, key_len: int, query_len: int) -> int:
        if query_len < 0 or key_len < query_len:
            raise ValueError(
                f"invalid dense-prefix dimensions: {key_len=}, {query_len=}"
            )
        if not getattr(self, "q_indexer_use_full_attention_short_seq", False):
            return 0
        dense_tokens = self._dsa_dense_attention_budget_tokens(query_len=query_len)
        dense_rows = self.recall_policy.dense_query_prefix_len(
            query_position_start=key_len - query_len,
            query_len=query_len,
            dense_tokens=dense_tokens,
        )
        if 0 < dense_rows < query_len and getattr(self, "qshare_enabled", False):
            qshare_group_size = int(getattr(self, "qshare_group_size", 1))
            if dense_tokens % qshare_group_size:
                raise ValueError(
                    "dense attention threshold must align to Q-share groups: "
                    f"dense_tokens={dense_tokens} "
                    f"qshare_group_size={qshare_group_size}"
                )
        return dense_rows

    def build_selection_query_state(
        self,
        *,
        score_query_states: torch.Tensor,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> typing.Any | None:
        del score_query_states, current_chunks, query_positions
        return None

    def prepare_selection_query_batch(
        self,
        *,
        score_query_states: torch.Tensor,
        query_start_loc: torch.Tensor | None,
        query_start_loc_cpu: torch.Tensor | None,
        active_seq_count: int,
        active_seq_infos: list[tuple[int, int, int, int]] | None = None,
    ) -> typing.Any | None:
        del (
            score_query_states,
            query_start_loc,
            query_start_loc_cpu,
            active_seq_count,
            active_seq_infos,
        )
        return None

    def build_selection_query_state_from_batch(
        self,
        *,
        selection_query_batch: typing.Any | None,
        seq_idx: int,
        q_start: int,
        q_end: int,
        score_query_states: torch.Tensor,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> typing.Any | None:
        del selection_query_batch, seq_idx, q_start, q_end
        return self.build_selection_query_state(
            score_query_states=score_query_states,
            current_chunks=current_chunks,
            query_positions=query_positions,
        )

    def get_selection_query_rows(
        self,
        *,
        selection_query_state: typing.Any | None,
        score_query_states: torch.Tensor,
        current_chunks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if selection_query_state is None:
            return score_query_states, current_chunks
        return (
            selection_query_state.reduced_q,
            selection_query_state.reduced_current_chunks,
        )

    def expand_selection_state(
        self,
        *,
        selection_state: typing.Any | None,
        selection_query_state: typing.Any | None,
    ) -> typing.Any | None:
        del selection_query_state
        return selection_state

    def selection_query_chunk_size(self, q_len: int) -> int:
        return min(self.q_indexer_chunked_query_chunk_size, q_len)

    def _dsa_kv_cache_layout_and_block_size(
        self,
        cache: torch.Tensor,
    ) -> tuple[str, int] | None:
        if cache.dim() != 4:
            return None
        if cache.shape[2] == self.num_kv_heads:
            return "NHD", int(cache.shape[1])
        if cache.shape[1] == self.num_kv_heads:
            return "HND", int(cache.shape[2])
        return None

    def _forward_dsa_full_page_table_fa_sequence(
        self,
        *,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn: typing.Any,
        attn_metadata: typing.Any | None,
        positions: torch.Tensor,
        key_len: int,
        allow_long_sequence: bool = False,
    ) -> torch.Tensor | None:
        reason = self._dsa_full_page_table_fa_fallback_reason(
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn=attn,
            attn_metadata=attn_metadata,
            positions=positions,
            key_len=key_len,
            allow_long_sequence=allow_long_sequence,
        )
        if reason is not None:
            return None

        flash_attn = _get_flash_attn_varlen_func()
        assert flash_attn is not None
        device = query_states.device
        query_len = query_states.shape[0]
        block_size = int(key_cache.shape[1])
        num_blocks = math.ceil(key_len / block_size)
        if block_table.device != device:
            block_table = block_table.to(device=device)

        temp_block_table = block_table[:num_blocks].reshape(1, num_blocks)
        cu_seqlens_q = torch.tensor([0, query_len], device=device, dtype=torch.int32)
        seqused_k = torch.tensor([key_len], device=device, dtype=torch.int32)
        output = torch.empty_like(query_states)
        impl = getattr(attn, "impl", None)
        fa_version = getattr(impl, "vllm_flash_attn_version", None)
        flash_attn_kwargs: dict[str, typing.Any] = {}
        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        flash_attn(
            q=query_states.contiguous(),
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=query_len,
            seqused_k=seqused_k,
            max_seqlen_k=key_len,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=True,
            block_table=temp_block_table,
            **flash_attn_kwargs,
        )
        return output

    def _dsa_full_page_table_fa_fallback_reason(
        self,
        *,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn: typing.Any,
        attn_metadata: typing.Any | None,
        positions: torch.Tensor | None,
        key_len: int,
        allow_long_sequence: bool = False,
        positions_are_known_suffix: bool = False,
        num_kv_heads: int | None = None,
    ) -> str | None:
        if (
            not getattr(self, "q_indexer_use_full_attention_short_seq", False)
            and not allow_long_sequence
        ):
            return "short-sequence full attention is disabled"
        if _get_flash_attn_varlen_func() is None:
            return "flash_attn_varlen_func is unavailable"
        query_len = int(query_states.shape[0])
        if query_len <= 0:
            return "query sequence is empty"
        if positions is None:
            if not positions_are_known_suffix:
                return "position metadata is required when suffix layout is unknown"
        elif positions.numel() != query_len:
            return (
                "position metadata must match query length, "
                f"query_len={query_len} positions={int(positions.numel())}"
            )
        # ``torch.equal(...)->bool`` synchronizes the device, which CUDA graph
        # capture forbids. vLLM's graph warm-up uses its synthetic final
        # contiguous suffix, so retain the page-table FA path during capture
        # and perform the defensive value check on normal eager execution.
        is_cuda_graph_capturing = (
            positions is not None
            and positions.is_cuda
            and torch.cuda.is_current_stream_capturing()
        )
        if not positions_are_known_suffix and not is_cuda_graph_capturing:
            assert positions is not None
            expected_positions = torch.arange(
                key_len - query_len,
                key_len,
                device=positions.device,
                dtype=positions.dtype,
            )
            if not bool(torch.equal(positions, expected_positions)):
                return (
                    "query positions are not the final contiguous suffix "
                    "of the KV sequence"
                )
        if (
            not allow_long_sequence
            and key_len > self._dsa_dense_attention_budget_tokens(query_len=query_len)
        ):
            budget = self._dsa_dense_attention_budget_tokens(query_len=query_len)
            return (
                "sequence exceeds dense attention budget, "
                f"key_len={key_len} budget={budget}"
            )
        return self._dsa_common_page_table_fa_fallback_reason(
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn=attn,
            attn_metadata=attn_metadata,
            key_len=key_len,
            num_kv_heads=num_kv_heads,
        )

    def _dsa_common_page_table_fa_fallback_reason(
        self,
        *,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn: typing.Any,
        attn_metadata: typing.Any | None,
        key_len: int,
        num_kv_heads: int | None = None,
    ) -> str | None:
        if key_cache.dim() != 4 or value_cache.dim() != 4:
            return (
                "paged FA requires 4D NHD key/value caches, "
                f"got key={tuple(key_cache.shape)} value={tuple(value_cache.shape)}"
            )
        if key_cache.shape != value_cache.shape:
            return (
                "paged FA requires matching key/value cache shapes, "
                f"got key={tuple(key_cache.shape)} value={tuple(value_cache.shape)}"
            )
        cache_layout = _get_dsa_kv_cache_layout()
        if cache_layout != "NHD":
            return (
                f"paged FA prototype only supports NHD cache layout, got {cache_layout}"
            )
        expected_suffix = (
            self.chunk_size,
            self.num_kv_heads if num_kv_heads is None else num_kv_heads,
            self.head_dim,
        )
        if tuple(key_cache.shape[1:]) != expected_suffix:
            return (
                "paged FA prototype only supports NHD cache shape "
                "(blocks, block_size, kv_heads, head_dim), "
                f"got shape={tuple(key_cache.shape)} "
                f"expected_suffix={expected_suffix}"
            )
        if block_table.dim() != 1:
            return (
                "expected a per-sequence 1D block table, "
                f"got {tuple(block_table.shape)}"
            )
        num_blocks = math.ceil(key_len / int(key_cache.shape[1]))
        if num_blocks > int(block_table.shape[0]):
            return (
                "sequence needs more pages than block table provides, "
                f"num_blocks={num_blocks} block_table_len={int(block_table.shape[0])}"
            )
        if getattr(attn_metadata, "use_cascade", False):
            return "cascade/prefix attention metadata is not handled"
        if getattr(attn_metadata, "dcp_context_kv_lens", None) is not None:
            return "decode context parallel metadata is not handled"
        attn_sliding_window = getattr(attn, "sliding_window", None)
        impl = getattr(attn, "impl", None)
        impl_sliding_window = getattr(impl, "sliding_window", None)
        if attn_sliding_window is not None or impl_sliding_window not in (
            None,
            (-1, -1),
            [-1, -1],
        ):
            return "sliding-window attention is not handled"
        if getattr(impl, "alibi_slopes", None) is not None:
            return "ALiBi attention is not handled"
        if getattr(impl, "logits_soft_cap", 0) not in (None, 0, 0.0):
            return "attention logits soft cap is not handled"
        if getattr(impl, "sinks", None) is not None:
            return "attention sinks are not handled"
        return None

    def _forward_dsa_chunked_unified_page_table_fa_bucket(
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
        if self.num_kv_heads == 1:
            handled = self._forward_dsa_chunked_single_kv_head_page_table_fa_bucket(
                hidden_states=hidden_states,
                query_states=query_states,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                attn=attn,
                attn_metadata=attn_metadata,
                positions=positions,
                active_seq_infos=active_seq_infos,
                batched_chunk_representatives=batched_chunk_representatives,
                output=output,
                indexer_q_proj=indexer_q_proj,
                local_kv_head_indices=local_kv_head_indices,
                precomputed_indexer_q=precomputed_indexer_q,
                precomputed_indexer_q_by_head=precomputed_indexer_q_by_head,
            )
            return handled or set()
        return self._forward_dsa_chunked_multi_kv_head_page_table_fa_bucket(
            hidden_states=hidden_states,
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn=attn,
            attn_metadata=attn_metadata,
            positions=positions,
            active_seq_infos=active_seq_infos,
            batched_chunk_representatives=batched_chunk_representatives,
            output=output,
            indexer_q_proj=indexer_q_proj,
            local_kv_head_indices=local_kv_head_indices,
            precomputed_indexer_q=precomputed_indexer_q,
            precomputed_indexer_q_by_head=precomputed_indexer_q_by_head,
        )

    def _forward_dsa_chunked_single_kv_head_page_table_fa_bucket(
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
    ) -> set[int] | None:
        if self.num_kv_heads != 1:
            return None
        return self._forward_dsa_chunked_one_kv_head_page_table_fa_bucket(
            hidden_states=hidden_states,
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn=attn,
            attn_metadata=attn_metadata,
            positions=positions,
            active_seq_infos=active_seq_infos,
            batched_chunk_representatives=batched_chunk_representatives,
            output=output,
            local_kv_head_indices=local_kv_head_indices,
            indexer_q_proj=indexer_q_proj,
            precomputed_indexer_q=(
                precomputed_indexer_q_by_head[0]
                if precomputed_indexer_q_by_head is not None
                else precomputed_indexer_q
            ),
        )

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
        if self.num_kv_heads <= 1:
            return set()
        if self.num_heads % self.num_kv_heads != 0:
            return set()
        if self._dsa_kv_cache_layout_and_block_size(key_cache) != (
            "NHD",
            self.chunk_size,
        ):
            return set()
        if self._dsa_kv_cache_layout_and_block_size(value_cache) != (
            "NHD",
            self.chunk_size,
        ):
            return set()
        if local_kv_head_indices.numel() < self.num_kv_heads:
            return set()

        group_size = self.num_heads // self.num_kv_heads
        handled: set[int] | None = None
        for kv_head_idx in range(self.num_kv_heads):
            head_start = kv_head_idx * group_size
            head_end = head_start + group_size
            one_head_representatives = None
            one_head_representative_group_idx = 0
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
                    one_head_representative_group_idx = kv_head_idx
                else:
                    one_head_representatives = {}
                    for (
                        seq_idx,
                        representatives,
                    ) in batched_chunk_representatives.items():
                        if (
                            representatives.dim() != 3
                            or int(representatives.shape[1]) <= kv_head_idx
                        ):
                            return set()
                        one_head_representatives[seq_idx] = representatives[
                            :, kv_head_idx : kv_head_idx + 1
                        ].contiguous()

            group_output = torch.empty_like(query_states[:, head_start:head_end])
            group_key_cache = key_cache[
                :, :, kv_head_idx : kv_head_idx + 1, :
            ].contiguous()
            group_value_cache = value_cache[
                :, :, kv_head_idx : kv_head_idx + 1, :
            ].contiguous()
            one_head_indexer_q = None
            if precomputed_indexer_q_by_head is not None:
                one_head_indexer_q = precomputed_indexer_q_by_head[kv_head_idx]
            elif precomputed_indexer_q is not None:
                one_head_indexer_q = precomputed_indexer_q[
                    :, kv_head_idx : kv_head_idx + 1
                ].contiguous()
            group_handled = self._forward_dsa_chunked_one_kv_head_page_table_fa_bucket(
                hidden_states=hidden_states,
                query_states=query_states[:, head_start:head_end],
                key_cache=group_key_cache,
                value_cache=group_value_cache,
                block_table=block_table,
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
                representative_group_idx=one_head_representative_group_idx,
            )
            if group_handled is None:
                return set()

            group_handled_set = set(group_handled)
            handled = (
                group_handled_set
                if handled is None
                else handled.intersection(group_handled_set)
            )
            for seq_idx, q_start, q_end, _ in active_seq_infos:
                if seq_idx in group_handled_set:
                    output[q_start:q_end, head_start:head_end].copy_(
                        group_output[q_start:q_end]
                    )

        return handled or set()

    def _forward_dsa_chunked_one_kv_head_page_table_fa_bucket(
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
        local_kv_head_indices: torch.Tensor,
        indexer_q_proj: typing.Callable[
            [torch.Tensor], tuple[torch.Tensor, typing.Any]
        ],
        precomputed_indexer_q: torch.Tensor | None = None,
        representative_group_idx: int = 0,
        flash_attn_override: (typing.Callable[..., typing.Any] | None) = None,
    ) -> set[int] | None:
        if not getattr(self, "q_indexer_use_page_table_fa", False):
            return None
        if not getattr(self, "q_indexer_use_prefill_page_table_fa", False):
            return None
        if not getattr(self, "q_indexer_use_flattened_prefill_page_table_fa", False):
            return None
        if block_table.dim() != 2:
            return None
        if not active_seq_infos:
            return set()

        expected_q_start = 0
        for _, q_start, q_end, _ in active_seq_infos:
            if q_start != expected_q_start:
                return None
            expected_q_start = q_end
        total_rows = expected_q_start
        # CUDA graph decode can pass padded query tensors; the active rows still
        # form a compact prefix described by active_seq_infos/query_start_loc.
        if total_rows > int(query_states.shape[0]):
            return None
        if tuple(output.shape) != tuple(query_states.shape):
            raise ValueError(
                "one-KV unified page-table FA output must match query shape, "
                f"output={tuple(output.shape)} query={tuple(query_states.shape)}"
            )

        flash_attn = flash_attn_override or _get_flash_attn_varlen_func()
        if flash_attn is None:
            return None
        device = query_states.device
        chunk_size = self.chunk_size
        table_parts: list[torch.Tensor] = []
        request_lens_parts: list[torch.Tensor] = []
        seqused_k_parts: list[torch.Tensor] = []
        sparse_infos: list[
            tuple[int, int, int, int, int, int, torch.Tensor | None]
        ] = []
        sparse_info_by_seq: dict[int, tuple[int, int, int, torch.Tensor | None]] = {}
        max_seqlen_q = 0
        max_seqlen_k = 0

        def _make_current_chunks(
            *,
            query_position_start: int,
            key_len: int,
            num_chunks: int,
        ) -> torch.Tensor:
            seq_positions = torch.arange(
                query_position_start,
                key_len,
                device="cpu",
                dtype=torch.long,
            )
            current_chunks = torch.div(
                seq_positions,
                chunk_size,
                rounding_mode="floor",
            )
            return current_chunks.clamp(min=0, max=num_chunks - 1)

        if block_table.device != device:
            block_table = block_table.to(device=device)
        block_table_rows = int(block_table.shape[0])
        use_dense_short_seq = getattr(
            self, "q_indexer_use_full_attention_short_seq", False
        )
        dense_decode_threshold = -1
        dense_prefill_threshold = -1
        if use_dense_short_seq:
            dense_decode_threshold = self._dsa_dense_attention_budget_tokens(
                query_len=1
            )
            dense_prefill_threshold = self._dsa_dense_attention_budget_tokens(
                query_len=2
            )
        query_start_loc_gpu = (
            None
            if attn_metadata is None
            else getattr(attn_metadata, "query_start_loc", None)
        )
        query_start_loc_cpu = (
            None
            if attn_metadata is None
            else getattr(attn_metadata, "query_start_loc_cpu", None)
        )
        seq_lens_gpu = (
            None if attn_metadata is None else getattr(attn_metadata, "seq_lens", None)
        )

        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            if key_len <= 0:
                return None
            query_position_start = key_len - q_len
            if query_position_start < 0:
                return None
            if seq_idx >= block_table_rows:
                return None
            seq_block_table = block_table[seq_idx]
            reason = self._dsa_common_page_table_fa_fallback_reason(
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=seq_block_table,
                attn=attn,
                attn_metadata=attn_metadata,
                key_len=key_len,
                num_kv_heads=1,
            )
            if reason is not None:
                return None

            dense_threshold = (
                dense_decode_threshold if q_len == 1 else dense_prefill_threshold
            )
            if use_dense_short_seq and key_len <= dense_threshold:
                dense_reason = self._dsa_full_page_table_fa_fallback_reason(
                    query_states=query_states[q_start:q_end],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=seq_block_table,
                    attn=attn,
                    attn_metadata=attn_metadata,
                    positions=None,
                    key_len=key_len,
                    positions_are_known_suffix=True,
                    num_kv_heads=1,
                )
                if dense_reason is not None:
                    return None
                continue

            num_chunks = math.ceil(key_len / chunk_size)

            if (
                batched_chunk_representatives is None
                or seq_idx not in batched_chunk_representatives
            ):
                return None
            chunk_representatives = batched_chunk_representatives[seq_idx]
            if (
                chunk_representatives.dim() != 3
                or int(chunk_representatives.shape[0]) != num_chunks
                or int(chunk_representatives.shape[1]) <= representative_group_idx
                or int(chunk_representatives.shape[2]) != self.q_indexer_dim
            ):
                return None
            sparse_infos.append(
                (
                    seq_idx,
                    q_start,
                    q_end,
                    key_len,
                    num_chunks,
                    query_position_start,
                    None,
                )
            )
            sparse_info_by_seq[seq_idx] = (
                num_chunks,
                query_position_start,
                key_len,
                None,
            )

        block_selection_by_seq: dict[int, typing.Any | None] = {}
        selection_query_state_by_seq: dict[int, typing.Any | None] = {}
        if sparse_infos:
            if precomputed_indexer_q is None:
                indexer_q, _ = indexer_q_proj(hidden_states[:total_rows])
                total_num_kv_heads = getattr(
                    self,
                    "total_num_kv_heads",
                    self.num_kv_heads,
                )
                indexer_q = indexer_q.view(
                    total_rows,
                    total_num_kv_heads,
                    self.q_indexer_dim,
                ).index_select(1, local_kv_head_indices.to(indexer_q.device))
            else:
                indexer_q = precomputed_indexer_q[:total_rows]
            indexer_scale = self.q_indexer_logit_scale / math.sqrt(self.q_indexer_dim)
            selection_query_batch = self.prepare_selection_query_batch(
                score_query_states=indexer_q,
                query_start_loc=query_start_loc_gpu,
                query_start_loc_cpu=query_start_loc_cpu,
                active_seq_count=len(active_seq_infos),
                active_seq_infos=active_seq_infos,
            )
            try_select_blocks_batched = getattr(
                self,
                "try_select_blocks_batched",
                None,
            )
            batched_selection = None
            if callable(try_select_blocks_batched):
                batched_selection = try_select_blocks_batched(
                    indexer_q=indexer_q,
                    selection_query_batch=selection_query_batch,
                    sparse_infos=sparse_infos,
                    batched_chunk_representatives=batched_chunk_representatives,
                    block_table=block_table,
                    representative_group_idx=representative_group_idx,
                    query_start_loc=query_start_loc_gpu,
                    seq_lens=seq_lens_gpu,
                    num_actual_tokens=total_rows,
                    active_seq_count=len(active_seq_infos),
                    dense_decode_threshold=dense_decode_threshold,
                    dense_prefill_threshold=dense_prefill_threshold,
                )
            if batched_selection is not None:
                block_selection_by_seq = batched_selection
            elif self.q_indexer_dynamic_chunk_top_k or (
                self.q_indexer_recent_window_pages > 0
                and getattr(self, "qshare_enabled", False)
            ):
                # The request-local policy can change inside this prefill slice.
                # Recent-window Q-share runs also must not cross a page boundary.
                # Fall back to the already policy/page-tiled sequence path rather
                # than applying an inexact shared decision.
                return None
            else:
                for (
                    seq_idx,
                    q_start,
                    q_end,
                    key_len,
                    num_chunks,
                    query_position_start,
                    current_chunks,
                ) in sparse_infos:
                    max_prior_chunks = max(num_chunks - 1, 0)
                    chunk_top_k = min(self.q_indexer_chunk_top_k, max_prior_chunks)
                    if current_chunks is None:
                        current_chunks = _make_current_chunks(
                            query_position_start=query_position_start,
                            key_len=key_len,
                            num_chunks=num_chunks,
                        )
                        sparse_info_by_seq[seq_idx] = (
                            num_chunks,
                            query_position_start,
                            key_len,
                            current_chunks,
                        )
                    current_chunks = current_chunks.to(device=device)
                    query_positions = torch.arange(
                        query_position_start,
                        key_len,
                        device=device,
                        dtype=torch.long,
                    )
                    selection_query_state = self.build_selection_query_state_from_batch(
                        selection_query_batch=selection_query_batch,
                        seq_idx=seq_idx,
                        q_start=q_start,
                        q_end=q_end,
                        score_query_states=indexer_q[q_start:q_end, 0],
                        current_chunks=current_chunks,
                        query_positions=query_positions,
                    )
                    selection_query_state_by_seq[seq_idx] = selection_query_state
                    if max_prior_chunks <= 0 or chunk_top_k <= 0:
                        block_selection_by_seq[seq_idx] = None
                        continue
                    score_query_states, score_current_chunks = (
                        self.get_selection_query_rows(
                            selection_query_state=selection_query_state,
                            score_query_states=indexer_q[q_start:q_end, 0],
                            current_chunks=current_chunks,
                        )
                    )
                    score_current_chunks = self._dsa_remote_current_chunks(
                        score_current_chunks
                    )
                    chunk_ids = torch.arange(
                        max_prior_chunks,
                        device=device,
                        dtype=current_chunks.dtype,
                    )
                    chunk_representatives = batched_chunk_representatives[seq_idx]
                    block_selection_by_seq[seq_idx] = self.select_blocks(
                        score_query_states=score_query_states,
                        representative_state=chunk_representatives,
                        current_chunks=score_current_chunks,
                        max_prior_chunks=max_prior_chunks,
                        block_top_k=chunk_top_k,
                        indexer_scale=indexer_scale,
                        block_table=block_table[seq_idx],
                        chunk_ids=chunk_ids,
                        seq_idx=seq_idx,
                        group_idx=0,
                    )

        if sparse_infos:
            try_build_page_tables_batched = getattr(
                self,
                "try_build_page_tables_batched",
                None,
            )
            if callable(try_build_page_tables_batched):
                q_for_fa = query_states[:total_rows].contiguous()
                batched_page_table_plan = try_build_page_tables_batched(
                    block_table=block_table,
                    selection_query_batch=selection_query_batch,
                    active_seq_infos=active_seq_infos,
                    sparse_infos=sparse_infos,
                    block_selection_by_seq=block_selection_by_seq,
                    total_rows=total_rows,
                    device=device,
                    query_start_loc=query_start_loc_gpu,
                    seq_lens=seq_lens_gpu,
                    num_actual_tokens=total_rows,
                    active_seq_count=len(active_seq_infos),
                    dense_decode_threshold=dense_decode_threshold,
                    dense_prefill_threshold=dense_prefill_threshold,
                )
                if batched_page_table_plan is not None:
                    (
                        plan_block_table,
                        cu_seqlens_q,
                        seqused_k_t,
                        max_seqlen_q,
                        max_seqlen_k,
                    ) = batched_page_table_plan
                    impl = getattr(attn, "impl", None)
                    fa_version = getattr(impl, "vllm_flash_attn_version", None)
                    flash_attn_kwargs: dict[str, typing.Any] = {}
                    if fa_version is not None:
                        flash_attn_kwargs["fa_version"] = fa_version

                    flash_attn(
                        q=q_for_fa,
                        k=key_cache,
                        v=value_cache,
                        out=output[:total_rows],
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        seqused_k=seqused_k_t,
                        max_seqlen_k=max_seqlen_k,
                        dropout_p=0.0,
                        softmax_scale=1.0 / math.sqrt(self.head_dim),
                        causal=True,
                        block_table=plan_block_table,
                        **flash_attn_kwargs,
                    )
                    return {seq_idx for seq_idx, _, _, _ in active_seq_infos}

        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            q_len = q_end - q_start
            seq_block_table = block_table[seq_idx]
            if self._dsa_sequence_fits_dense_attention(key_len, q_len):
                page_table_plan = self.build_page_table_plan(
                    block_table=seq_block_table,
                    chunk_size=chunk_size,
                    key_len=key_len,
                    q_len=q_len,
                    dense=True,
                    mode="prefill",
                )
                if page_table_plan is None:
                    return None
                pages, request_lens, seqused_k, plan_max_q, plan_max_k = page_table_plan
                table_parts.append(pages)
                request_lens_parts.append(request_lens)
                seqused_k_parts.append(seqused_k)
                max_seqlen_q = max(max_seqlen_q, plan_max_q)
                max_seqlen_k = max(max_seqlen_k, plan_max_k)
                continue

            dense_rows = self._dsa_dense_query_prefix_len(
                key_len=key_len,
                query_len=q_len,
            )
            if dense_rows > 0:
                # The optimized batched planner may represent this as one dense
                # request followed by sparse requests. The generic per-sequence
                # planner cannot, so leave it to the exact split fallback.
                log_recall_plan(
                    "unified_dense_sparse_fallback",
                    dense_tokens=self._dsa_dense_attention_budget_tokens(
                        query_len=q_len
                    ),
                    context_start=key_len - q_len + 1,
                    context_end=key_len,
                    dense_rows=dense_rows,
                    sparse_rows=q_len - dense_rows,
                    reason="planner_did_not_split_crossing",
                )
                return None

            (
                num_chunks,
                query_position_start,
                sparse_key_len,
                current_chunks,
            ) = sparse_info_by_seq[seq_idx]
            if sparse_key_len != key_len:
                return None
            if current_chunks is None:
                current_chunks = _make_current_chunks(
                    query_position_start=query_position_start,
                    key_len=key_len,
                    num_chunks=num_chunks,
                )
                sparse_info_by_seq[seq_idx] = (
                    num_chunks,
                    query_position_start,
                    key_len,
                    current_chunks,
                )
            current_chunks = current_chunks.to(device=device)
            page_table_plan = self.build_page_table_plan(
                block_table=seq_block_table,
                chunk_size=chunk_size,
                key_len=key_len,
                q_len=q_len,
                dense=False,
                mode="prefill",
                selection_state=block_selection_by_seq[seq_idx],
                current_chunks=current_chunks,
                query_position_start=query_position_start,
                selection_query_state=selection_query_state_by_seq.get(seq_idx),
            )
            if page_table_plan is None:
                return None
            pages, request_lens, seqused_k, plan_max_q, plan_max_k = page_table_plan
            table_parts.append(pages)
            request_lens_parts.append(request_lens)
            seqused_k_parts.append(seqused_k)
            max_seqlen_q = max(max_seqlen_q, plan_max_q)
            max_seqlen_k = max(max_seqlen_k, plan_max_k)

        if not table_parts:
            return set()
        max_pages = max(int(part.shape[1]) for part in table_parts)
        num_requests = sum(int(part.shape[0]) for part in table_parts)
        plan_block_table = torch.zeros(
            num_requests,
            max_pages,
            device=device,
            dtype=torch.int32,
        )
        request_start = 0
        for pages in table_parts:
            request_end = request_start + int(pages.shape[0])
            plan_block_table[request_start:request_end, : int(pages.shape[1])] = pages
            request_start = request_end

        request_lens_t = torch.cat(request_lens_parts, dim=0)
        seqused_k_t = torch.cat(seqused_k_parts, dim=0)
        impl = getattr(attn, "impl", None)
        fa_version = getattr(impl, "vllm_flash_attn_version", None)
        flash_attn_kwargs: dict[str, typing.Any] = {}
        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        flash_attn(
            q=query_states[:total_rows].contiguous(),
            k=key_cache,
            v=value_cache,
            out=output[:total_rows],
            cu_seqlens_q=self._make_cu_seqlens_from_lengths(request_lens_t),
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k_t,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=True,
            block_table=plan_block_table,
            **flash_attn_kwargs,
        )
        return {seq_idx for seq_idx, _, _, _ in active_seq_infos}

    def _make_cu_seqlens_from_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        cu_seqlens = torch.empty(
            lengths.numel() + 1,
            device=lengths.device,
            dtype=torch.int32,
        )
        cu_seqlens[0] = 0
        cu_seqlens[1:] = torch.cumsum(lengths.to(torch.int32), dim=0)
        return cu_seqlens

    def _forward_dsa_chunked_flattened_decode_page_table_fa_sequence(
        self,
        *,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn: typing.Any,
        attn_metadata: typing.Any | None,
        selection_state: typing.Any | None,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
        key_len: int,
        softmax_scale: float,
    ) -> torch.Tensor | None:
        if self.num_kv_heads != 1:
            return None
        if int(query_states.shape[0]) != 1:
            return None
        if current_chunks.numel() != 1:
            raise ValueError("current_chunks must contain one decode row")
        if query_positions.numel() != 1:
            raise ValueError("query_positions must contain one decode row")
        if block_table.dim() != 1:
            raise ValueError(
                "block_table must be a per-sequence 1D block table, "
                f"got {tuple(block_table.shape)}"
            )

        reason = self._dsa_common_page_table_fa_fallback_reason(
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn=attn,
            attn_metadata=attn_metadata,
            key_len=key_len,
        )
        if reason is not None:
            return None

        flash_attn = _get_flash_attn_varlen_func()
        if flash_attn is None:
            return None
        device = query_states.device
        chunk_size = self.chunk_size
        if block_table.device != device:
            block_table = block_table.to(device=device)
        if current_chunks.device != device:
            current_chunks = current_chunks.to(device=device)
        if query_positions.device != device:
            query_positions = query_positions.to(device=device)
        page_table_plan = self.build_page_table_plan(
            block_table=block_table,
            chunk_size=chunk_size,
            key_len=key_len,
            mode="decode",
            selection_state=selection_state,
            current_chunks=current_chunks,
            query_positions=query_positions,
        )
        if page_table_plan is None:
            return None
        temp_block_table, request_lens, seqused_k, max_seqlen_q, max_seqlen_k = (
            page_table_plan
        )

        impl = getattr(attn, "impl", None)
        fa_version = getattr(impl, "vllm_flash_attn_version", None)
        flash_attn_kwargs: dict[str, typing.Any] = {}
        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        output = torch.empty_like(query_states)
        flash_attn(
            q=query_states.contiguous(),
            k=key_cache,
            v=value_cache,
            out=output,
            cu_seqlens_q=self._make_cu_seqlens_from_lengths(request_lens),
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            block_table=temp_block_table,
            **flash_attn_kwargs,
        )
        return output

    def _forward_dsa_chunked_sequence_with_dense_prefix(
        self,
        *,
        query_states: torch.Tensor,
        indexer_query_states: torch.Tensor,
        key_states: torch.Tensor | None,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn: typing.Any,
        attn_metadata: typing.Any | None,
        positions: torch.Tensor,
        key_len: int | None = None,
        chunk_representatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_len = int(query_states.shape[0])
        if key_len is None:
            if key_states is None:
                raise ValueError("key_len is required when key_states is omitted")
            key_len = int(key_states.shape[0])
        dense_rows = self._dsa_dense_query_prefix_len(
            key_len=key_len,
            query_len=q_len,
        )
        if dense_rows == 0 or dense_rows == q_len:
            return self._forward_dsa_chunked_sequence(
                query_states=query_states,
                indexer_query_states=indexer_query_states,
                key_states=key_states,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                attn=attn,
                attn_metadata=attn_metadata,
                positions=positions,
                key_len=key_len,
                chunk_representatives=chunk_representatives,
            )

        dense_tokens = self._dsa_dense_attention_budget_tokens(query_len=q_len)
        query_position_start = key_len - q_len
        dense_key_len = query_position_start + dense_rows
        if dense_key_len != dense_tokens:
            raise AssertionError(
                "dense/sparse crossing must end exactly at the dense threshold: "
                f"{dense_key_len=}, {dense_tokens=}"
            )
        log_recall_plan(
            "dense_sparse_boundary_split",
            dense_tokens=dense_tokens,
            context_start=query_position_start + 1,
            context_end=key_len,
            dense_rows=dense_rows,
            sparse_rows=q_len - dense_rows,
            first_sparse_context=dense_tokens + 1,
            first_sparse_top_k=self._dsa_chunk_top_k_for_context(dense_tokens + 1),
            recent_window_pages=self.q_indexer_recent_window_pages,
            qshare_group_size=int(getattr(self, "qshare_group_size", 1)),
            backend="sequence_split_fallback",
        )
        dense_output = self._forward_dsa_full_page_table_fa_sequence(
            query_states=query_states[:dense_rows],
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn=attn,
            attn_metadata=attn_metadata,
            positions=positions[:dense_rows],
            key_len=dense_key_len,
            # The caller has already classified these rows with the effective
            # prefill threshold. Avoid treating a one-row prefix as decode.
            allow_long_sequence=True,
        )
        if dense_output is None:
            reason = self._dsa_full_page_table_fa_fallback_reason(
                query_states=query_states[:dense_rows],
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                attn=attn,
                attn_metadata=attn_metadata,
                positions=positions[:dense_rows],
                key_len=dense_key_len,
                allow_long_sequence=True,
            )
            raise RuntimeError(
                "dense attention is required for a threshold-crossing prefill, "
                "but paged FlashAttention could not execute the dense prefix: "
                f"{reason}"
            )
        sparse_output = self._forward_dsa_chunked_sequence(
            query_states=query_states[dense_rows:],
            indexer_query_states=indexer_query_states[dense_rows:],
            key_states=key_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn=attn,
            attn_metadata=attn_metadata,
            positions=positions[dense_rows:],
            key_len=key_len,
            chunk_representatives=chunk_representatives,
        )
        return torch.cat((dense_output, sparse_output), dim=0)

    def _forward_dsa_chunked_sequence(
        self,
        *,
        query_states: torch.Tensor,
        indexer_query_states: torch.Tensor,
        key_states: torch.Tensor | None,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn: typing.Any,
        attn_metadata: typing.Any | None,
        positions: torch.Tensor,
        key_len: int | None = None,
        chunk_representatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_len = query_states.shape[0]
        if key_len is None:
            if key_states is None:
                raise ValueError("key_len is required when key_states is omitted")
            key_len = key_states.shape[0]
        output = query_states.new_empty(q_len, self.num_heads, self.head_dim)
        if q_len == 0 or key_len == 0:
            return output.zero_()

        if getattr(self, "q_indexer_use_full_attention_short_seq", False):
            full_page_table_output = self._forward_dsa_full_page_table_fa_sequence(
                query_states=query_states,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                attn=attn,
                attn_metadata=attn_metadata,
                positions=positions,
                key_len=key_len,
            )
            if full_page_table_output is not None:
                return full_page_table_output

        chunk_size = self.chunk_size
        num_chunks = math.ceil(key_len / chunk_size)
        query_chunk_size = self.selection_query_chunk_size(q_len)
        first_query_position = key_len - q_len
        indexer_scale = self.q_indexer_logit_scale / math.sqrt(self.q_indexer_dim)
        main_scale = 1.0 / math.sqrt(self.head_dim)
        group_size = self.num_heads // self.num_kv_heads
        if chunk_representatives is None:
            if key_states is None:
                key_states = self.gather_kv_sequence(key_cache, block_table, key_len)
            chunk_representatives = self.build_representative_state(
                key_states=key_states,
                key_cache=key_cache,
                block_table=block_table,
                key_len=key_len,
            )
        else:
            expected_shape = (num_chunks, self.num_kv_heads, self.q_indexer_dim)
            if tuple(chunk_representatives.shape) != expected_shape:
                raise ValueError(
                    "batched DSA chunk representatives have unexpected shape: "
                    f"got {tuple(chunk_representatives.shape)}, "
                    f"expected {expected_shape}"
                )
        if key_states is None:
            key_states = self.gather_kv_sequence(key_cache, block_table, key_len)
        chunk_offsets = torch.arange(
            chunk_size, device=query_states.device, dtype=torch.long
        )

        query_start = 0
        while query_start < q_len:
            query_end = min(
                self._dsa_chunked_query_tile_end(
                    query_start=query_start,
                    query_len=q_len,
                    first_query_position=first_query_position,
                ),
                query_start + query_chunk_size,
            )
            chunk_len = query_end - query_start
            query_positions = positions[query_start:query_end].to(
                device=query_states.device, dtype=torch.long
            )
            current_chunks = torch.div(
                query_positions, chunk_size, rounding_mode="floor"
            )
            current_chunks = current_chunks.clamp(min=0, max=num_chunks - 1)
            current_chunk_starts = current_chunks * chunk_size
            tail_indices = current_chunk_starts[:, None] + chunk_offsets[None, :]
            tail_valid = (tail_indices <= query_positions[:, None]) & (
                tail_indices < key_len
            )

            context_len = first_query_position + query_start + 1
            requested_chunk_top_k = self._dsa_chunk_top_k_for_context(context_len)
            max_prior_chunks = max(num_chunks - 1, 0)
            if max_prior_chunks > 0:
                chunk_top_k = min(requested_chunk_top_k, max_prior_chunks)
                chunk_ids = torch.arange(
                    max_prior_chunks, device=query_states.device, dtype=torch.long
                )
            else:
                chunk_top_k = 0
                chunk_ids = torch.empty(0, device=query_states.device, dtype=torch.long)

            log_recall_plan(
                "sequence_tile",
                context_start=context_len,
                context_end=first_query_position + query_end,
                remote_top_k=chunk_top_k,
                recent_window_pages=self.q_indexer_recent_window_pages,
                rows=chunk_len,
            )

            selected_blocks_by_group: dict[int, typing.Any] = {}
            if chunk_top_k > 0:
                for group_idx in range(self.num_kv_heads):
                    score_query_states = indexer_query_states[
                        query_start:query_end, group_idx
                    ]
                    selection_query_state = self.build_selection_query_state(
                        score_query_states=score_query_states,
                        current_chunks=current_chunks,
                        query_positions=query_positions,
                    )
                    score_query_states, score_current_chunks = (
                        self.get_selection_query_rows(
                            selection_query_state=selection_query_state,
                            score_query_states=score_query_states,
                            current_chunks=current_chunks,
                        )
                    )
                    score_remote_chunks = self._dsa_remote_current_chunks(
                        score_current_chunks
                    )
                    selection_state = self.select_blocks(
                        score_query_states=score_query_states,
                        representative_state=chunk_representatives,
                        current_chunks=score_remote_chunks,
                        max_prior_chunks=max_prior_chunks,
                        block_top_k=chunk_top_k,
                        indexer_scale=indexer_scale,
                        block_table=block_table,
                        chunk_ids=chunk_ids,
                        group_idx=group_idx,
                    )
                    selected_blocks_by_group[group_idx] = self.expand_selection_state(
                        selection_state=selection_state,
                        selection_query_state=selection_query_state,
                    )

            if (
                chunk_len == 1
                and getattr(self, "q_indexer_use_page_table_fa", False)
                and getattr(self, "q_indexer_use_flattened_decode_page_table_fa", False)
            ):
                flat_output = (
                    self._forward_dsa_chunked_flattened_decode_page_table_fa_sequence(
                        query_states=query_states[query_start:query_end],
                        key_cache=key_cache,
                        value_cache=value_cache,
                        block_table=block_table,
                        attn=attn,
                        attn_metadata=attn_metadata,
                        selection_state=selected_blocks_by_group.get(0),
                        current_chunks=current_chunks,
                        query_positions=query_positions,
                        key_len=key_len,
                        softmax_scale=main_scale,
                    )
                )
                if flat_output is not None:
                    output[query_start:query_end] = flat_output
                    query_start = query_end
                    continue

            for group_idx in range(self.num_kv_heads):
                head_start = group_idx * group_size
                head_end = head_start + group_size
                group_query_states = query_states[
                    query_start:query_end, head_start:head_end
                ]

                if chunk_top_k > 0:
                    top_chunk_indices, top_chunk_valid = self.get_selected_blocks(
                        selected_blocks_by_group[group_idx],
                        device=query_states.device,
                    )
                    chunk_token_indices = (
                        top_chunk_indices[..., None] * chunk_size
                        + chunk_offsets[None, None, :]
                    )
                    chunk_token_valid = top_chunk_valid[..., None] & (
                        chunk_token_indices < key_len
                    )
                    selected_width = int(top_chunk_indices.shape[-1])
                    chunk_token_indices = chunk_token_indices.reshape(
                        chunk_len, selected_width * chunk_size
                    )
                    chunk_token_valid = chunk_token_valid.reshape(
                        chunk_len, selected_width * chunk_size
                    )
                else:
                    chunk_token_indices = tail_indices.new_empty(chunk_len, 0)
                    chunk_token_valid = tail_valid.new_empty(chunk_len, 0)

                recent_counts = self._dsa_recent_page_counts(current_chunks)
                recent_width = self.q_indexer_recent_window_pages
                if recent_width > 0:
                    recent_offsets = torch.arange(
                        recent_width,
                        device=query_states.device,
                        dtype=torch.long,
                    )
                    recent_starts = current_chunks - recent_counts
                    recent_chunks = recent_starts[:, None] + recent_offsets[None, :]
                    recent_valid = recent_offsets[None, :] < recent_counts[:, None]
                    recent_token_indices = (
                        recent_chunks[..., None] * chunk_size
                        + chunk_offsets[None, None, :]
                    ).reshape(chunk_len, recent_width * chunk_size)
                    recent_token_valid = (
                        recent_valid[..., None]
                        .expand(-1, -1, chunk_size)
                        .reshape(chunk_len, recent_width * chunk_size)
                    )
                else:
                    recent_token_indices = tail_indices.new_empty(chunk_len, 0)
                    recent_token_valid = tail_valid.new_empty(chunk_len, 0)

                recall_indices = torch.cat(
                    (chunk_token_indices, recent_token_indices, tail_indices),
                    dim=-1,
                )
                recall_valid = torch.cat(
                    (chunk_token_valid, recent_token_valid, tail_valid),
                    dim=-1,
                )

                safe_recall_indices = recall_indices.masked_fill(~recall_valid, 0)
                selected_k = key_states[:, group_idx].index_select(
                    0, safe_recall_indices.reshape(-1)
                )
                selected_v = self.gather_kv_positions_for_head(
                    value_cache, block_table, safe_recall_indices, group_idx
                )
                recall_len = recall_indices.shape[-1]
                selected_k = selected_k.view(chunk_len, recall_len, self.head_dim)
                selected_v = selected_v.view(chunk_len, recall_len, self.head_dim)

                main_logits = torch.einsum(
                    "qhd,qkd->hqk", group_query_states.float(), selected_k.float()
                )
                main_logits.mul_(main_scale)
                main_logits = main_logits.masked_fill(
                    ~recall_valid[None, :, :],
                    torch.finfo(main_logits.dtype).min,
                )
                attn_weights = F.softmax(main_logits, dim=-1, dtype=torch.float32)
                attn_weights = attn_weights.to(query_states.dtype)
                output[query_start:query_end, head_start:head_end] = torch.einsum(
                    "hqk,qkd->qhd", attn_weights, selected_v
                )
            query_start = query_end
        return output


class TorchChunkedDSAProviderBundle(ChunkedDSAAttentionProviderMixin, nn.Module):
    """PyTorch component bundle for the chunked DSA pipeline."""

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
        self.representative_provider = TorchChunkedDSARepresentativeProvider(
            q_indexer_dim=q_indexer_dim,
            chunk_size=chunk_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.scoring_provider = TorchChunkedDSAScoringProvider(
            q_indexer_dim=q_indexer_dim,
            logit_scale=logit_scale,
        )
        self.block_selection_provider = TorchTopKChunkedDSABlockSelectionProvider()
        self.block_table_provider = TorchChunkedDSABlockTableProvider()

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
            raise ValueError("PyTorch DSA representative provider is unavailable")
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
        active_seq_infos: list[tuple[int, int, int, int]],
        cache_info: tuple[str, int] | None = None,
        should_skip_sequence: _SequenceSkipFn | None = None,
        **kwargs: typing.Any,
    ) -> dict[int, torch.Tensor] | None:
        result = self.representative_provider(
            key_cache=key_cache,
            block_table=block_table,
            active_seq_infos=active_seq_infos,
            cache_info=cache_info,
            should_skip_sequence=should_skip_sequence,
            **kwargs,
        )
        if not self.representative_provider.is_available(result):
            return None

        by_seq: dict[int, torch.Tensor] = {}
        for seq_idx, _, _, _ in active_seq_infos:
            representatives = self.representative_provider.get_for_sequence(
                result,
                seq_idx=seq_idx,
            )
            if representatives is not None:
                by_seq[seq_idx] = representatives
        return by_seq or None

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
