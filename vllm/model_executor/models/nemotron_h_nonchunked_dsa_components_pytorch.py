# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PyTorch components for token-level Nemotron-H DSA."""

from __future__ import annotations

import math
import typing

import torch
import torch.nn.functional as F
from torch import nn

_SequenceSkipFn = typing.Callable[[int, int, int, int], bool]


class _UnavailableTokenState:
    pass


_UNAVAILABLE = _UnavailableTokenState()


class _TorchTokenRepresentatives:

    __slots__ = ("_by_seq", "_single")

    def __init__(
        self,
        *,
        single: torch.Tensor | None = None,
        by_seq: dict[int, torch.Tensor] | None = None,
    ) -> None:
        self._single = single
        self._by_seq = by_seq


class _TorchTokenScores:

    __slots__ = ("_token_logits", "_token_valid")

    def __init__(
        self,
        *,
        token_logits: torch.Tensor,
        token_valid: torch.Tensor,
    ) -> None:
        self._token_logits = token_logits
        self._token_valid = token_valid


class _TorchTokenSelection:

    __slots__ = ("_selected_token_indices", "_selected_token_valid")

    def __init__(
        self,
        *,
        selected_token_indices: torch.Tensor,
        selected_token_valid: torch.Tensor,
    ) -> None:
        self._selected_token_indices = selected_token_indices
        self._selected_token_valid = selected_token_valid


class _TorchTokenGatherPlan:

    __slots__ = ("_selected_token_indices", "_selected_token_valid")

    def __init__(
        self,
        *,
        selected_token_indices: torch.Tensor,
        selected_token_valid: torch.Tensor,
    ) -> None:
        self._selected_token_indices = selected_token_indices
        self._selected_token_valid = selected_token_valid


class _NonChunkedDSABatchRepresentatives:

    __slots__ = ("_by_seq",)

    def __init__(self, by_seq: dict[int, torch.Tensor] | None) -> None:
        self._by_seq = by_seq


class _NonChunkedDSABatchScores:

    __slots__ = ("_representatives",)

    def __init__(
        self,
        representatives: _NonChunkedDSABatchRepresentatives,
    ) -> None:
        self._representatives = representatives


class _NonChunkedDSABatchSelection:

    __slots__ = ("_scores",)

    def __init__(self, scores: _NonChunkedDSABatchScores) -> None:
        self._scores = scores


class _NonChunkedDSABatchTokenTables:

    __slots__ = ("_selection",)

    def __init__(self, selection: _NonChunkedDSABatchSelection) -> None:
        self._selection = selection


class TorchNonChunkedDSARepresentativeProvider(nn.Module):
    """Token representative provider for non-chunked DSA.

    The "representative" of a token is the token's own indexer key vector.
    """

    def __init__(
        self,
        *,
        q_indexer_dim: int,
        num_kv_heads: int,
        head_dim: int | None = None,
    ) -> None:
        super().__init__()
        if q_indexer_dim <= 0:
            raise ValueError(f"q_indexer_dim must be positive: {q_indexer_dim}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive: {num_kv_heads}")
        self.q_indexer_dim = q_indexer_dim
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
            if key_cache is None or block_table is None or block_table.dim() != 2:
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
                by_seq[seq_idx] = seq_key_states[..., : self.q_indexer_dim]
            return _TorchTokenRepresentatives(by_seq=by_seq)

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
        return _TorchTokenRepresentatives(
            single=key_states[..., : self.q_indexer_dim]
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
        if not isinstance(result, _TorchTokenRepresentatives):
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
        if cache.dim() != 4:
            raise NotImplementedError(
                f"DSA cache gather expects a 4D KV cache, got {cache.shape}"
            )
        head_dim = self.head_dim if self.head_dim is not None else cache.shape[-1]
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


class TorchNonChunkedDSAScoringProvider(nn.Module):
    """Token scoring provider for non-chunked DSA."""

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
        token_representatives: torch.Tensor | None = None,
        query_positions: torch.Tensor,
        key_len: int | None = None,
        key_positions: torch.Tensor | None = None,
        seq_idx: int | None = None,
        group_idx: int | None = None,
        **_: typing.Any,
    ) -> typing.Any:
        representatives = self._materialize_representatives(
            representative_state=representative_state,
            token_representatives=token_representatives,
            seq_idx=seq_idx,
            group_idx=group_idx,
        )
        if representatives is None:
            return _UNAVAILABLE
        if key_len is None:
            key_len = int(representatives.shape[0])
        representatives = representatives[:key_len]

        if key_len <= 0:
            shape = (score_query_states.shape[0], 0)
            return _TorchTokenScores(
                token_logits=torch.empty(
                    shape,
                    device=score_query_states.device,
                    dtype=torch.float32,
                ),
                token_valid=torch.empty(
                    shape,
                    device=score_query_states.device,
                    dtype=torch.bool,
                ),
            )
        if (
            score_query_states.dim() != 2
            or representatives.dim() != 2
            or query_positions.dim() != 1
            or score_query_states.shape[0] != query_positions.shape[0]
            or score_query_states.shape[1] != self.q_indexer_dim
            or representatives.shape[1] != self.q_indexer_dim
        ):
            return _UNAVAILABLE

        device = score_query_states.device
        if representatives.device != device:
            representatives = representatives.to(device=device)
        query_positions = query_positions.to(device=device, dtype=torch.long)
        if key_positions is None:
            key_positions = torch.arange(key_len, device=device, dtype=torch.long)
        else:
            key_positions = key_positions.to(device=device, dtype=torch.long)
            if key_positions.dim() != 1 or int(key_positions.shape[0]) != key_len:
                return _UNAVAILABLE

        token_logits = torch.matmul(
            score_query_states.float(),
            representatives.float().transpose(0, 1),
        )
        token_logits.mul_(self.logit_scale / math.sqrt(self.q_indexer_dim))
        token_valid = key_positions[None, :] <= query_positions[:, None]
        token_logits = token_logits.masked_fill(
            ~token_valid,
            torch.finfo(token_logits.dtype).min,
        )
        return _TorchTokenScores(
            token_logits=token_logits,
            token_valid=token_valid,
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
        if not isinstance(result, _TorchTokenScores):
            raise TypeError(f"unexpected scoring result: {type(result)!r}")
        return result._token_logits, result._token_valid

    def _materialize_representatives(
        self,
        *,
        representative_state: typing.Any | None,
        token_representatives: torch.Tensor | None,
        seq_idx: int | None,
        group_idx: int | None,
    ) -> torch.Tensor | None:
        if token_representatives is not None:
            representatives = token_representatives
        elif isinstance(representative_state, _TorchTokenRepresentatives):
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
        if representatives.dim() == 3:
            if group_idx is None:
                return None
            if group_idx < 0 or group_idx >= int(representatives.shape[1]):
                return None
            representatives = representatives[:, group_idx]
        return representatives


class TorchTopKTokenDSASelectionProvider(nn.Module):
    """Top-k token selector for non-chunked DSA."""

    def forward(
        self,
        *,
        score_state: typing.Any,
        token_top_k: int | None = None,
        block_top_k: int | None = None,
        top_k: int | None = None,
        **_: typing.Any,
    ) -> typing.Any:
        scores = self._materialize_scores(score_state)
        if scores is None:
            return _UNAVAILABLE
        token_logits, token_valid = scores
        top_k_limit = token_top_k
        if top_k_limit is None:
            top_k_limit = top_k if top_k is not None else block_top_k
        if top_k_limit is None:
            return _UNAVAILABLE

        if top_k_limit <= 0 or token_logits.shape[-1] == 0:
            shape = (token_logits.shape[0], 0)
            return _TorchTokenSelection(
                selected_token_indices=torch.empty(
                    shape,
                    device=token_logits.device,
                    dtype=torch.long,
                ),
                selected_token_valid=torch.empty(
                    shape,
                    device=token_logits.device,
                    dtype=torch.bool,
                ),
            )

        selected_width = min(int(top_k_limit), int(token_logits.shape[-1]))
        selected_token_indices = token_logits.topk(
            k=selected_width,
            dim=-1,
        ).indices
        selected_token_valid = token_valid.gather(
            dim=-1,
            index=selected_token_indices,
        )
        selected_token_indices = selected_token_indices.masked_fill(
            ~selected_token_valid,
            0,
        )
        return _TorchTokenSelection(
            selected_token_indices=selected_token_indices,
            selected_token_valid=selected_token_valid,
        )

    def is_available(self, result: typing.Any) -> bool:
        return result is not _UNAVAILABLE

    def get_selected_tokens(
        self,
        result: typing.Any,
        **_: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if result is _UNAVAILABLE:
            return None
        if not isinstance(result, _TorchTokenSelection):
            raise TypeError(f"unexpected token selection result: {type(result)!r}")
        return result._selected_token_indices, result._selected_token_valid

    def _materialize_scores(
        self,
        score_state: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not isinstance(score_state, _TorchTokenScores):
            return None
        return score_state._token_logits, score_state._token_valid


class TorchNonChunkedDSATokenTableProvider(nn.Module):
    """Builds a token-gather plan for non-chunked DSA."""

    def forward(
        self,
        *,
        selection_state: typing.Any | None,
        rows: int | None = None,
        device: torch.device | None = None,
        **_: typing.Any,
    ) -> typing.Any:
        if selection_state is None:
            if rows is None or device is None:
                return _UNAVAILABLE
            shape = (rows, 0)
            return _TorchTokenGatherPlan(
                selected_token_indices=torch.empty(
                    shape,
                    device=device,
                    dtype=torch.long,
                ),
                selected_token_valid=torch.empty(
                    shape,
                    device=device,
                    dtype=torch.bool,
                ),
            )
        if not isinstance(selection_state, _TorchTokenSelection):
            return _UNAVAILABLE
        return _TorchTokenGatherPlan(
            selected_token_indices=selection_state._selected_token_indices,
            selected_token_valid=selection_state._selected_token_valid,
        )

    def is_available(self, result: typing.Any) -> bool:
        return result is not _UNAVAILABLE

    def get_token_gather_plan(
        self,
        result: typing.Any,
        **_: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if result is _UNAVAILABLE:
            return None
        if not isinstance(result, _TorchTokenGatherPlan):
            raise TypeError(f"unexpected token table result: {type(result)!r}")
        return result._selected_token_indices, result._selected_token_valid


class TorchNonChunkedDSAProviderBundle(nn.Module):
    """PyTorch component bundle for token-level DSA."""

    def __init__(
        self,
        *,
        q_indexer_dim: int,
        num_kv_heads: int,
        head_dim: int,
        logit_scale: float,
        chunk_size: int | None = None,
        chunk_top_k: int | None = None,
        top_k: int | None = None,
        query_chunk_size: int = 256,
        num_heads: int | None = None,
        total_num_kv_heads: int | None = None,
        **_: typing.Any,
    ) -> None:
        super().__init__()
        if top_k is None:
            if chunk_size is None or chunk_top_k is None:
                top_k = 2048
            else:
                top_k = int(chunk_size) * int(chunk_top_k)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive: {top_k}")
        if query_chunk_size <= 0:
            raise ValueError(
                f"query_chunk_size must be positive: {query_chunk_size}"
            )
        self.q_indexer_dim = q_indexer_dim
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_indexer_logit_scale = logit_scale
        self.q_indexer_top_k = top_k
        self.q_indexer_query_chunk_size = query_chunk_size
        self.num_heads = num_kv_heads if num_heads is None else num_heads
        self.total_num_kv_heads = (
            num_kv_heads if total_num_kv_heads is None else total_num_kv_heads
        )
        self.representative_provider = TorchNonChunkedDSARepresentativeProvider(
            q_indexer_dim=q_indexer_dim,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.scoring_provider = TorchNonChunkedDSAScoringProvider(
            q_indexer_dim=q_indexer_dim,
            logit_scale=logit_scale,
        )
        self.token_selection_provider = TorchTopKTokenDSASelectionProvider()
        self.token_table_provider = TorchNonChunkedDSATokenTableProvider()

    def prepare_representatives(
        self,
        *,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        cache_info: tuple[str, int] | None,
        **_: typing.Any,
    ) -> _NonChunkedDSABatchRepresentatives:
        result = self.representative_provider(
            key_cache=key_cache,
            block_table=block_table,
            active_seq_infos=active_seq_infos,
            cache_info=cache_info,
        )
        if not self.representative_provider.is_available(result):
            return _NonChunkedDSABatchRepresentatives(None)

        by_seq: dict[int, torch.Tensor] = {}
        for seq_idx, _, _, _ in active_seq_infos:
            representatives = self.representative_provider.get_for_sequence(
                result,
                seq_idx=seq_idx,
            )
            if representatives is not None:
                by_seq[seq_idx] = representatives
        return _NonChunkedDSABatchRepresentatives(by_seq or None)

    def prepare_scores(
        self,
        *,
        representatives: _NonChunkedDSABatchRepresentatives,
    ) -> _NonChunkedDSABatchScores:
        return _NonChunkedDSABatchScores(representatives)

    def prepare_selection(
        self,
        *,
        scores: _NonChunkedDSABatchScores,
    ) -> _NonChunkedDSABatchSelection:
        return _NonChunkedDSABatchSelection(scores)

    def prepare_block_tables(
        self,
        *,
        selection: _NonChunkedDSABatchSelection,
    ) -> _NonChunkedDSABatchTokenTables:
        return _NonChunkedDSABatchTokenTables(selection)

    def get_cache_info(self, cache: torch.Tensor) -> tuple[str, int] | None:
        return self._dsa_kv_cache_layout_and_block_size(cache)

    def forward_attention(
        self,
        *,
        block_state: _NonChunkedDSABatchTokenTables,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn: typing.Any,
        attn_metadata: typing.Any | None,
        positions: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        indexer_q_proj: typing.Callable[[torch.Tensor], tuple[torch.Tensor, typing.Any]],
        local_kv_head_indices: torch.Tensor,
    ) -> torch.Tensor:
        del attn, attn_metadata
        batched_token_representatives = (
            block_state._selection._scores._representatives._by_seq
        )
        output = query_states.new_zeros(query_states.shape)
        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            indexer_q, _ = indexer_q_proj(hidden_states[q_start:q_end])
            indexer_q = indexer_q.view(
                -1,
                self.total_num_kv_heads,
                self.q_indexer_dim,
            )
            indexer_q = indexer_q.index_select(
                1,
                local_kv_head_indices.to(indexer_q.device),
            )
            precomputed = (
                batched_token_representatives.get(seq_idx)
                if batched_token_representatives is not None
                else None
            )
            seq_output = self._forward_dsa_sequence(
                query_states=query_states[q_start:q_end],
                indexer_query_states=indexer_q,
                key_states=None,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[seq_idx],
                positions=positions[q_start:q_end],
                key_len=key_len,
                token_representatives=precomputed,
            )
            output[q_start:q_end] = seq_output
        return output

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
            raise ValueError(
                "PyTorch non-chunked DSA representative provider is unavailable"
            )
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
            flat_token_indices,
            block_size,
            rounding_mode="floor",
        ).to(torch.long)
        block_offsets = flat_token_indices.remainder(block_size).to(torch.long)
        block_ids = block_table.index_select(0, block_indices).to(torch.long)
        if cache_layout == "NHD":
            selected = cache[block_ids, block_offsets, kv_head_idx]
        else:
            selected = cache[block_ids, kv_head_idx, block_offsets]
        return selected.view(*token_indices.shape, self.head_dim)

    def select_tokens(
        self,
        *,
        score_query_states: torch.Tensor,
        representative_state: typing.Any,
        query_positions: torch.Tensor,
        key_len: int,
        token_top_k: int,
        key_positions: torch.Tensor | None = None,
        seq_idx: int | None = None,
        group_idx: int | None = None,
        **kwargs: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        score_state = self.scoring_provider(
            score_query_states=score_query_states,
            representative_state=representative_state,
            query_positions=query_positions,
            key_len=key_len,
            key_positions=key_positions,
            seq_idx=seq_idx,
            group_idx=group_idx,
            **kwargs,
        )
        if not self.scoring_provider.is_available(score_state):
            raise ValueError("DSA token scoring provider is unavailable")
        selection_state = self.token_selection_provider(
            score_state=score_state,
            token_top_k=token_top_k,
            seq_idx=seq_idx,
            group_idx=group_idx,
            **kwargs,
        )
        if not self.token_selection_provider.is_available(selection_state):
            raise ValueError("DSA token selection provider is unavailable")
        token_table_state = self.token_table_provider(
            selection_state=selection_state,
            rows=score_query_states.shape[0],
            device=score_query_states.device,
            **kwargs,
        )
        selection = self.token_table_provider.get_token_gather_plan(
            token_table_state
        )
        if selection is None:
            raise ValueError("DSA token table provider is unavailable")
        return selection

    def _forward_dsa_sequence(
        self,
        *,
        query_states: torch.Tensor,
        indexer_query_states: torch.Tensor,
        key_states: torch.Tensor | None,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        positions: torch.Tensor,
        key_len: int | None = None,
        token_representatives: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_len = query_states.shape[0]
        if key_len is None:
            if key_states is None:
                raise ValueError("key_len is required when key_states is omitted")
            key_len = key_states.shape[0]
        output = query_states.new_empty(q_len, self.num_heads, self.head_dim)
        if q_len == 0 or key_len == 0:
            return output.zero_()

        if key_states is None:
            key_states = self.gather_kv_sequence(key_cache, block_table, key_len)
        if token_representatives is None:
            representative_state = self.build_representative_state(
                key_states=key_states,
                key_cache=key_cache,
                block_table=block_table,
                key_len=key_len,
            )
            token_representatives = self.representative_provider.get_for_sequence(
                representative_state
            )
        if token_representatives is None:
            raise ValueError("DSA token representatives are unavailable")

        expected_shape = (key_len, self.num_kv_heads, self.q_indexer_dim)
        if tuple(token_representatives.shape) != expected_shape:
            raise ValueError(
                "DSA token representatives have unexpected shape: "
                f"got {tuple(token_representatives.shape)}, "
                f"expected {expected_shape}"
            )

        query_chunk_size = min(self.q_indexer_query_chunk_size, q_len)
        main_scale = 1.0 / math.sqrt(self.head_dim)
        group_size = self.num_heads // self.num_kv_heads
        key_positions = torch.arange(
            key_len,
            device=query_states.device,
            dtype=torch.long,
        )
        token_top_k = min(self.q_indexer_top_k, key_len)

        for query_start in range(0, q_len, query_chunk_size):
            query_end = min(query_start + query_chunk_size, q_len)
            chunk_len = query_end - query_start
            query_positions = positions[query_start:query_end].to(
                device=query_states.device,
                dtype=torch.long,
            )
            for group_idx in range(self.num_kv_heads):
                head_start = group_idx * group_size
                head_end = head_start + group_size
                group_query_states = query_states[
                    query_start:query_end,
                    head_start:head_end,
                ]
                selected_token_indices, selected_token_valid = self.select_tokens(
                    score_query_states=indexer_query_states[
                        query_start:query_end,
                        group_idx,
                    ],
                    representative_state=token_representatives,
                    query_positions=query_positions,
                    key_len=key_len,
                    key_positions=key_positions,
                    token_top_k=token_top_k,
                    group_idx=group_idx,
                )

                safe_token_indices = selected_token_indices.masked_fill(
                    ~selected_token_valid,
                    0,
                )
                selected_k = key_states[:, group_idx].index_select(
                    0,
                    safe_token_indices.reshape(-1),
                )
                selected_v = self.gather_kv_positions_for_head(
                    value_cache,
                    block_table,
                    safe_token_indices,
                    group_idx,
                )
                selected_width = selected_token_indices.shape[-1]
                selected_k = selected_k.view(
                    chunk_len,
                    selected_width,
                    self.head_dim,
                )
                selected_v = selected_v.view(
                    chunk_len,
                    selected_width,
                    self.head_dim,
                )

                main_logits = torch.einsum(
                    "qhd,qkd->hqk",
                    group_query_states.float(),
                    selected_k.float(),
                )
                main_logits.mul_(main_scale)
                main_logits = main_logits.masked_fill(
                    ~selected_token_valid[None, :, :],
                    torch.finfo(main_logits.dtype).min,
                )
                attn_weights = F.softmax(
                    main_logits,
                    dim=-1,
                    dtype=torch.float32,
                )
                attn_weights = attn_weights.to(query_states.dtype)
                output[query_start:query_end, head_start:head_end] = torch.einsum(
                    "hqk,qkd->qhd",
                    attn_weights,
                    selected_v,
                )
        return output

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
