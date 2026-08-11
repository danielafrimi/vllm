# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sparse attention implementation for Nemotron-H DSA layers."""

import math
import os
import sys
import typing

import torch
import torch.nn.functional as F

from vllm.config import CacheConfig, ModelConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
    unified_kv_cache_update,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.nemotron_h import (
    NemotronHAttention,
    _split_dsa_kv_cache,
)
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

try:
    from vllm.model_executor.models.nemotron_h_dsa_triton_scoring import (
        dsa_score_topk_torch,
    )
except ImportError:
    dsa_score_topk_torch = None

try:
    from vllm.model_executor.models.nemotron_h_dsa_triton_summaries import (
        dsa_block_summaries_triton,
    )
except ImportError:
    dsa_block_summaries_triton = None


_DSA_DENSE_PREFILL_KV_THRESHOLD_ENV = (
    "VLLM_NEMOTRON_H_DSA_DENSE_PREFILL_KV_THRESHOLD_TOKENS"
)
_DSA_CHUNK_TOP_K_ENV = "VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K"


def _coalesce(value, default):
    return default if value is None else value


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
        # Unit tests call this module without a current vLLM runtime config.
        return "NHD"


class NemotronHDSALegacyAttention(NemotronHAttention):
    """Simple PyTorch chunk-DSA attention for Nemotron-H layers."""

    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            config=config,
            layer_idx=layer_idx,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.layer_idx = layer_idx
        self.q_indexer_dim = int(getattr(config, "q_indexer_dim"))
        if not 0 < self.q_indexer_dim <= self.head_dim:
            raise ValueError(
                "q_indexer_dim must be in [1, head_dim] when reusing main K; "
                f"got q_indexer_dim={self.q_indexer_dim}, head_dim={self.head_dim}"
            )

        q_indexer_attn_mode = getattr(
            config, "q_indexer_attn_mode", "chunked_topk_sparse")
        if q_indexer_attn_mode in {"chunked_sparse", "topk_chunked"}:
            q_indexer_attn_mode = "chunked_topk_sparse"
        if q_indexer_attn_mode != "chunked_topk_sparse":
            raise ValueError(
                "Simplified Nemotron-H DSA only supports "
                f"chunked_topk_sparse, got {q_indexer_attn_mode!r}"
            )
        self.q_indexer_attn_mode = q_indexer_attn_mode
        self.q_indexer_logit_scale = float(
            _coalesce(getattr(config, "q_indexer_logit_scale", None), 1.0))
        self.q_indexer_top_k = int(
            _coalesce(getattr(config, "q_indexer_top_k", None), 2048))
        self.q_indexer_chunk_size = int(
            _coalesce(getattr(config, "q_indexer_chunk_size", None), 16))
        if self.q_indexer_top_k <= 0:
            raise ValueError(
                f"q_indexer_top_k must be positive: {self.q_indexer_top_k}")
        if self.q_indexer_chunk_size <= 0:
            raise ValueError(
                "q_indexer_chunk_size must be positive: "
                f"{self.q_indexer_chunk_size}")

        default_chunk_top_k = math.ceil(
            self.q_indexer_top_k / self.q_indexer_chunk_size)
        self.q_indexer_chunk_top_k = _env_int(
            _DSA_CHUNK_TOP_K_ENV,
            int(_coalesce(
                getattr(config, "q_indexer_chunk_top_k", None),
                default_chunk_top_k,
            )),
        )
        self.q_indexer_chunked_query_chunk_size = int(
            _coalesce(
                getattr(config, "q_indexer_chunked_query_chunk_size", None),
                min(
                    int(_coalesce(
                        getattr(config, "q_indexer_query_chunk_size", None),
                        256,
                    )),
                    16,
                ),
            ))
        if self.q_indexer_chunk_top_k <= 0:
            raise ValueError(
                "q_indexer_chunk_top_k must be positive: "
                f"{self.q_indexer_chunk_top_k}")
        if self.q_indexer_chunked_query_chunk_size <= 0:
            raise ValueError(
                "q_indexer_chunked_query_chunk_size must be positive: "
                f"{self.q_indexer_chunked_query_chunk_size}")
        self.q_indexer_use_triton_batched_summaries = (
            os.environ.get(
                "VLLM_NEMOTRON_H_DSA_USE_TRITON_BATCHED_SUMMARIES",
                "0",
            )
            == "1"
        )
        self.q_indexer_use_page_table_fa = (
            os.environ.get("VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA", "0") == "1"
        )
        self.q_indexer_use_prefill_page_table_fa = (
            os.environ.get(
                "VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA", "0"
            )
            == "1"
        )
        self.q_indexer_use_full_attention_short_seq = (
            os.environ.get(
                "VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ", "0"
            )
            == "1"
        )
        self.q_indexer_use_flattened_prefill_page_table_fa = (
            os.environ.get(
                "VLLM_NEMOTRON_H_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA",
                "0",
            )
            == "1"
        )
        self.q_indexer_use_flattened_decode_page_table_fa = (
            os.environ.get(
                "VLLM_NEMOTRON_H_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA",
                "0",
            )
            == "1"
        )
        self.q_indexer_dense_prefill_kv_threshold_tokens = _env_int(
            _DSA_DENSE_PREFILL_KV_THRESHOLD_ENV,
            self._dsa_dense_attention_budget_tokens(),
        )
        if self.q_indexer_dense_prefill_kv_threshold_tokens <= 0:
            raise ValueError(
                f"{_DSA_DENSE_PREFILL_KV_THRESHOLD_ENV} must be positive: "
                f"{self.q_indexer_dense_prefill_kv_threshold_tokens}"
            )

        self.indexer_q_proj = ReplicatedLinear(
            config.hidden_size,
            self.total_num_kv_heads * self.q_indexer_dim,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer_q_proj",
        )

        winners = getattr(config, "q_indexer_init_query_heads", None)
        if winners is None:
            winners = [-1] * self.total_num_kv_heads
        self.register_buffer(
            "dsa_winner_query_heads",
            torch.tensor([int(w) for w in winners], dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_local_kv_head_indices",
            torch.tensor(self._get_local_kv_head_indices(), dtype=torch.long),
            persistent=False,
        )

    def _get_local_kv_head_indices(self) -> list[int]:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if self.total_num_kv_heads >= tp_size:
            start = tp_rank * self.num_kv_heads
            return list(range(start, start + self.num_kv_heads))

        ranks_per_kv_head = tp_size // self.total_num_kv_heads
        return [tp_rank // ranks_per_kv_head]

    def _dsa_dense_attention_budget_tokens(
        self,
        query_len: int | None = None,
    ) -> int:
        default_budget = self.q_indexer_chunk_size * self.q_indexer_chunk_top_k
        if query_len is not None and query_len > 1:
            return getattr(
                self,
                "q_indexer_dense_prefill_kv_threshold_tokens",
                default_budget,
            )
        return default_budget

    def _dsa_sequence_fits_dense_attention(
        self,
        key_len: int,
        query_len: int | None = None,
    ) -> bool:
        budget = self._dsa_dense_attention_budget_tokens(query_len=query_len)
        fits = (
            getattr(self, "q_indexer_use_full_attention_short_seq", False)
            and key_len <= budget
        )
        return fits

    def _dsa_active_sequence_infos(
        self,
        attn_metadata: typing.Any,
    ) -> list[tuple[int, int, int, int]]:
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        query_start_loc = getattr(
            attn_metadata, "query_start_loc_cpu", None)
        if query_start_loc is None:
            query_start_loc = attn_metadata.query_start_loc
        seq_lens = getattr(attn_metadata, "seq_lens_cpu", None)
        if seq_lens is None:
            seq_lens = getattr(attn_metadata, "_seq_lens_cpu", None)
        if seq_lens is None:
            seq_lens = attn_metadata.seq_lens

        infos: list[tuple[int, int, int, int]] = []
        for seq_idx in range(query_start_loc.numel() - 1):
            q_start = int(query_start_loc[seq_idx].item())
            q_end = int(query_start_loc[seq_idx + 1].item())
            if q_start >= num_actual_tokens:
                break
            q_end = min(q_end, num_actual_tokens)
            if q_end <= q_start:
                continue
            key_len = int(seq_lens[seq_idx].item())
            infos.append((seq_idx, q_start, q_end, key_len))
        return infos

    @torch.compiler.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if positions is None:
            raise ValueError("DSA selective attention requires token positions")

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q_view = q.view(-1, self.num_heads, self.head_dim)
        k_view = k.view(-1, self.num_kv_heads, self.head_dim)
        v_view = v.view(-1, self.num_kv_heads, self.head_dim)

        attn_metadata, _, kv_cache, _ = get_attention_context(self.attn.layer_name)
        if attn_metadata is None:
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

        unified_kv_cache_update(k_view, v_view, self.attn.layer_name)
        key_cache, value_cache = self._split_kv_cache(kv_cache)
        block_table = attn_metadata.block_table
        attn_output = q.new_zeros(q.shape)
        active_seq_infos = self._dsa_active_sequence_infos(attn_metadata)
        cache_info = self._dsa_kv_cache_layout_and_block_size(key_cache)

        # Build chunk representatives across the active batch when possible.
        batched_chunk_representatives = (
            self._get_triton_batched_chunk_representatives(
                key_cache=key_cache,
                block_table=block_table,
                active_seq_infos=active_seq_infos,
                cache_info=cache_info,
            )
        )

        # Consume eligible rows through unified page-table FA before fallback.
        attn_output_view = attn_output.view(-1, self.num_heads, self.head_dim)
        page_table_handled_seq_indices = (
            self._forward_dsa_chunked_unified_page_table_fa_bucket(
                hidden_states=hidden_states,
                query_states=q_view,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                attn_metadata=attn_metadata,
                positions=positions,
                active_seq_infos=active_seq_infos,
                batched_chunk_representatives=batched_chunk_representatives,
                output=attn_output_view,
            )
        )

        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            if seq_idx in page_table_handled_seq_indices:
                continue

            indexer_q, _ = self.indexer_q_proj(hidden_states[q_start:q_end])
            indexer_q = indexer_q.view(
                -1, self.total_num_kv_heads, self.q_indexer_dim)
            indexer_q = indexer_q.index_select(
                1, self._local_kv_head_indices.to(indexer_q.device))
            precomputed = (
                batched_chunk_representatives.get(seq_idx)
                if batched_chunk_representatives is not None
                else None
            )
            seq_output = self._forward_dsa_chunked_sequence(
                query_states=q_view[q_start:q_end],
                indexer_query_states=indexer_q,
                key_states=None,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table[seq_idx],
                attn_metadata=attn_metadata,
                positions=positions[q_start:q_end],
                key_len=key_len,
                chunk_representatives=precomputed,
            )
            attn_output[q_start:q_end] = seq_output.reshape(q_end - q_start, -1)

        output, _ = self.o_proj(attn_output)
        return output

    def _split_kv_cache(
        self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _split_dsa_kv_cache(
            kv_cache,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

    def _gather_kv_sequence(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        key_len: int,
    ) -> torch.Tensor:
        if key_len == 0:
            return cache.new_empty(0, self.num_kv_heads, self.head_dim)
        if cache.dim() != 4:
            raise NotImplementedError(
                f"DSA cache gather expects a 4D KV cache, got {cache.shape}")
        if cache.shape[2] == self.num_kv_heads:
            block_size = cache.shape[1]
            cache_layout = "NHD"
        elif cache.shape[1] == self.num_kv_heads:
            block_size = cache.shape[2]
            cache_layout = "HND"
        else:
            raise NotImplementedError(
                "DSA cache gather only supports NHD/HND KV cache layouts, "
                f"got shape={cache.shape}, num_kv_heads={self.num_kv_heads}")

        if block_table.device != cache.device:
            block_table = block_table.to(device=cache.device)
        token_indices = torch.arange(key_len, device=cache.device, dtype=torch.long)
        block_indices = torch.div(token_indices, block_size, rounding_mode="floor")
        block_offsets = token_indices.remainder(block_size)
        block_ids = block_table.index_select(0, block_indices).to(torch.long)
        if cache_layout == "NHD":
            return cache[block_ids, block_offsets]
        return cache[block_ids, :, block_offsets]

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

    def _get_indexer_chunk_representatives(
        self,
        *,
        key_states: torch.Tensor | None,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        key_len: int,
    ) -> torch.Tensor:
        if key_states is None:
            key_states = self._gather_kv_sequence(key_cache, block_table, key_len)
        return self._build_indexer_chunk_representatives(
            key_states[..., : self.q_indexer_dim])

    def _get_triton_batched_chunk_representatives(
        self,
        *,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        cache_info: tuple[str, int] | None,
    ) -> dict[int, torch.Tensor] | None:
        helper = _resolve_dsa_symbol(
            "dsa_block_summaries_triton", dsa_block_summaries_triton)
        required_cache_info = ("NHD", self.q_indexer_chunk_size)
        if (
            not getattr(self, "q_indexer_use_triton_batched_summaries", False)
            or helper is None
            or cache_info != required_cache_info
            or block_table.dim() != 2
        ):
            return None

        table_width = int(block_table.shape[1])
        if table_width <= 0:
            return None

        chunk_size = self.q_indexer_chunk_size
        live_infos: list[tuple[int, int, int]] = []
        max_live_chunks = 0
        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            if key_len <= 0:
                continue
            if self._dsa_sequence_fits_dense_attention(key_len, q_end - q_start):
                continue
            if seq_idx >= int(block_table.shape[0]):
                return None
            num_chunks = math.ceil(key_len / chunk_size)
            if num_chunks > table_width:
                return None
            max_live_chunks = max(max_live_chunks, num_chunks)
            live_infos.append((seq_idx, key_len, num_chunks))
        if not live_infos:
            return None

        active_seq_indices = torch.tensor(
            [seq_idx for seq_idx, _, _ in live_infos],
            device=block_table.device,
            dtype=torch.long,
        )
        active_block_table = block_table.index_select(0, active_seq_indices)
        active_block_table = active_block_table[:, :max_live_chunks].contiguous()
        active_seq_lens = torch.tensor(
            [key_len for _, key_len, _ in live_infos],
            device=key_cache.device,
            dtype=torch.long,
        )
        if active_block_table.device != key_cache.device:
            active_block_table = active_block_table.to(device=key_cache.device)

        # Let Triton summarize all active KV page chunks in one launch.
        batched_representatives = helper(
            key_cache=key_cache,
            block_table=active_block_table,
            seq_lens=active_seq_lens,
            q_indexer_dim=self.q_indexer_dim,
        )
        expected_shape = (
            len(live_infos),
            max_live_chunks,
            self.num_kv_heads,
            self.q_indexer_dim,
        )
        if (
            batched_representatives is None
            or tuple(batched_representatives.shape) != expected_shape
        ):
            return None

        return {
            seq_idx: batched_representatives[local_idx, :num_chunks]
            for local_idx, (seq_idx, _, num_chunks) in enumerate(live_infos)
        }

    def _forward_dsa_full_page_table_fa_sequence(
        self,
        *,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
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

        # Hand FA the physical pages for the contiguous causal suffix.
        temp_block_table = block_table[:num_blocks].reshape(1, num_blocks)
        cu_seqlens_q = torch.tensor([0, query_len], device=device, dtype=torch.int32)
        seqused_k = torch.tensor([key_len], device=device, dtype=torch.int32)
        output = torch.empty_like(query_states)
        impl = getattr(self.attn, "impl", None)
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
        if not positions_are_known_suffix:
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
            and key_len
            > self._dsa_dense_attention_budget_tokens(query_len=query_len)
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
            return f"paged FA prototype only supports NHD cache layout, got {cache_layout}"
        expected_suffix = (
            self.q_indexer_chunk_size,
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
        attn_sliding_window = getattr(self.attn, "sliding_window", None)
        impl = getattr(self.attn, "impl", None)
        impl_sliding_window = getattr(impl, "sliding_window", None)
        if (
            attn_sliding_window is not None
            or impl_sliding_window not in (None, (-1, -1), [-1, -1])
        ):
            return "sliding-window attention is not handled"
        if getattr(impl, "alibi_slopes", None) is not None:
            return "ALiBi attention is not handled"
        if getattr(impl, "logits_soft_cap", 0) not in (None, 0, 0.0):
            return "attention logits soft cap is not handled"
        if getattr(impl, "sinks", None) is not None:
            return "attention sinks are not handled"
        return None

    def _dsa_score_group_top_chunks(
        self,
        *,
        score_query_states: torch.Tensor,
        chunk_representatives: torch.Tensor,
        current_chunks: torch.Tensor,
        max_prior_chunks: int,
        chunk_top_k: int,
        indexer_scale: float,
        chunk_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if max_prior_chunks <= 0 or chunk_top_k <= 0:
            shape = (score_query_states.shape[0], 0)
            return (
                torch.empty(shape, device=score_query_states.device, dtype=torch.long),
                torch.empty(shape, device=score_query_states.device, dtype=torch.bool),
            )

        score_topk_torch = _resolve_dsa_symbol(
            "dsa_score_topk_torch", dsa_score_topk_torch)
        if score_topk_torch is not None:
            torch_score_topk = score_topk_torch(
                score_query_states=score_query_states,
                chunk_representatives=chunk_representatives[:max_prior_chunks],
                current_chunks=current_chunks,
                chunk_top_k=chunk_top_k,
                logit_scale=self.q_indexer_logit_scale,
                q_indexer_dim=self.q_indexer_dim,
                chunk_ids=chunk_ids,
            )
            if torch_score_topk is not None:
                top_chunk_indices, top_chunk_valid, _ = torch_score_topk
                return top_chunk_indices, top_chunk_valid

        chunk_logits = torch.matmul(
            score_query_states.float(),
            chunk_representatives[:max_prior_chunks].transpose(0, 1),
        )
        chunk_logits.mul_(indexer_scale)
        if chunk_ids is None:
            chunk_ids = torch.arange(
                max_prior_chunks,
                device=score_query_states.device,
                dtype=current_chunks.dtype,
            )
        chunk_valid = (
            chunk_ids[None, :] < current_chunks[:, None]
        )
        chunk_logits = chunk_logits.masked_fill(
            ~chunk_valid, torch.finfo(chunk_logits.dtype).min)
        top_chunk_indices = chunk_logits.topk(k=chunk_top_k, dim=-1).indices
        top_chunk_valid = chunk_valid.gather(dim=-1, index=top_chunk_indices)
        top_chunk_indices = top_chunk_indices.masked_fill(~top_chunk_valid, 0)
        return top_chunk_indices, top_chunk_valid

    def _forward_dsa_chunked_unified_page_table_fa_bucket(
        self,
        *,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn_metadata: typing.Any | None,
        positions: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        batched_chunk_representatives: dict[int, torch.Tensor] | None,
        output: torch.Tensor,
    ) -> set[int]:
        if self.num_kv_heads == 1:
            handled = self._forward_dsa_chunked_single_kv_head_page_table_fa_bucket(
                hidden_states=hidden_states,
                query_states=query_states,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                attn_metadata=attn_metadata,
                positions=positions,
                active_seq_infos=active_seq_infos,
                batched_chunk_representatives=batched_chunk_representatives,
                output=output,
            )
            return handled or set()
        return self._forward_dsa_chunked_multi_kv_head_page_table_fa_bucket(
            hidden_states=hidden_states,
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn_metadata=attn_metadata,
            positions=positions,
            active_seq_infos=active_seq_infos,
            batched_chunk_representatives=batched_chunk_representatives,
            output=output,
        )

    def _forward_dsa_chunked_single_kv_head_page_table_fa_bucket(
        self,
        *,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn_metadata: typing.Any | None,
        positions: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        batched_chunk_representatives: dict[int, torch.Tensor] | None,
        output: torch.Tensor,
    ) -> set[int] | None:
        if self.num_kv_heads != 1:
            return None
        return self._forward_dsa_chunked_one_kv_head_page_table_fa_bucket(
            hidden_states=hidden_states,
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn_metadata=attn_metadata,
            positions=positions,
            active_seq_infos=active_seq_infos,
            batched_chunk_representatives=batched_chunk_representatives,
            output=output,
            local_kv_head_indices=self._local_kv_head_indices,
        )

    def _forward_dsa_chunked_multi_kv_head_page_table_fa_bucket(
        self,
        *,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn_metadata: typing.Any | None,
        positions: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        batched_chunk_representatives: dict[int, torch.Tensor] | None,
        output: torch.Tensor,
    ) -> set[int]:
        if self.num_kv_heads <= 1:
            return set()
        if self.num_heads % self.num_kv_heads != 0:
            return set()
        if self._dsa_kv_cache_layout_and_block_size(key_cache) != (
            "NHD",
            self.q_indexer_chunk_size,
        ):
            return set()
        if self._dsa_kv_cache_layout_and_block_size(value_cache) != (
            "NHD",
            self.q_indexer_chunk_size,
        ):
            return set()
        if self._local_kv_head_indices.numel() < self.num_kv_heads:
            return set()

        group_size = self.num_heads // self.num_kv_heads
        handled: set[int] | None = None
        for kv_head_idx in range(self.num_kv_heads):
            head_start = kv_head_idx * group_size
            head_end = head_start + group_size
            one_head_representatives = None
            if batched_chunk_representatives is not None:
                one_head_representatives = {}
                for seq_idx, representatives in batched_chunk_representatives.items():
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
            group_handled = (
                self._forward_dsa_chunked_one_kv_head_page_table_fa_bucket(
                    hidden_states=hidden_states,
                    query_states=query_states[:, head_start:head_end],
                    key_cache=group_key_cache,
                    value_cache=group_value_cache,
                    block_table=block_table,
                    attn_metadata=attn_metadata,
                    positions=positions,
                    active_seq_infos=active_seq_infos,
                    batched_chunk_representatives=one_head_representatives,
                    output=group_output,
                    local_kv_head_indices=self._local_kv_head_indices[
                        kv_head_idx : kv_head_idx + 1
                    ],
                )
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
        attn_metadata: typing.Any | None,
        positions: torch.Tensor,
        active_seq_infos: list[tuple[int, int, int, int]],
        batched_chunk_representatives: dict[int, torch.Tensor] | None,
        output: torch.Tensor,
        local_kv_head_indices: torch.Tensor,
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
        if total_rows != int(query_states.shape[0]):
            return None
        if tuple(output.shape) != tuple(query_states.shape):
            raise ValueError(
                "one-KV unified page-table FA output must match query shape, "
                f"output={tuple(output.shape)} query={tuple(query_states.shape)}"
            )

        flash_attn = _get_flash_attn_varlen_func()
        if flash_attn is None:
            return None
        device = query_states.device
        chunk_size = self.q_indexer_chunk_size
        table_parts: list[torch.Tensor] = []
        request_lens_parts: list[torch.Tensor] = []
        seqused_k_parts: list[torch.Tensor] = []
        sparse_infos: list[tuple[int, int, int, int, int, int,
                                 torch.Tensor]] = []
        sparse_info_by_seq: dict[int, tuple[int, int, torch.Tensor]] = {}
        max_seqlen_q = 0
        max_seqlen_k = 0

        if block_table.device != device:
            block_table = block_table.to(device=device)

        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            if key_len <= 0:
                return None
            query_position_start = key_len - q_len
            if query_position_start < 0:
                return None
            if seq_idx >= int(block_table.shape[0]):
                return None
            seq_block_table = block_table[seq_idx]
            reason = self._dsa_common_page_table_fa_fallback_reason(
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=seq_block_table,
                attn_metadata=attn_metadata,
                key_len=key_len,
                num_kv_heads=1,
            )
            if reason is not None:
                return None

            if self._dsa_sequence_fits_dense_attention(key_len, q_len):
                dense_reason = self._dsa_full_page_table_fa_fallback_reason(
                    query_states=query_states[q_start:q_end],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=seq_block_table,
                    attn_metadata=attn_metadata,
                    positions=None,
                    key_len=key_len,
                    positions_are_known_suffix=True,
                    num_kv_heads=1,
                )
                if dense_reason is not None:
                    return None
                continue

            seq_positions = torch.arange(
                query_position_start,
                key_len,
                device=device,
                dtype=torch.long,
            )
            current_chunks = torch.div(
                seq_positions,
                chunk_size,
                rounding_mode="floor",
            )
            num_chunks = math.ceil(key_len / chunk_size)
            current_chunks = current_chunks.clamp(min=0, max=num_chunks - 1)

            if (
                batched_chunk_representatives is None
                or seq_idx not in batched_chunk_representatives
            ):
                return None
            chunk_representatives = batched_chunk_representatives[seq_idx]
            expected_shape = (num_chunks, 1, self.q_indexer_dim)
            if tuple(chunk_representatives.shape) != expected_shape:
                return None
            sparse_infos.append(
                (
                    seq_idx,
                    q_start,
                    q_end,
                    key_len,
                    num_chunks,
                    query_position_start,
                    current_chunks,
                )
            )
            sparse_info_by_seq[seq_idx] = (
                num_chunks,
                query_position_start,
                current_chunks,
            )

        top_chunks_by_seq: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        if sparse_infos:
            indexer_q, _ = self.indexer_q_proj(hidden_states[:total_rows])
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
            indexer_scale = self.q_indexer_logit_scale / math.sqrt(
                self.q_indexer_dim)
            for (
                seq_idx,
                q_start,
                q_end,
                _,
                num_chunks,
                _,
                current_chunks,
            ) in sparse_infos:
                max_prior_chunks = max(num_chunks - 1, 0)
                chunk_top_k = min(self.q_indexer_chunk_top_k, max_prior_chunks)
                if max_prior_chunks <= 0 or chunk_top_k <= 0:
                    empty_indices = torch.empty(
                        q_end - q_start,
                        0,
                        device=device,
                        dtype=torch.long,
                    )
                    empty_valid = torch.empty_like(empty_indices, dtype=torch.bool)
                    top_chunks_by_seq[seq_idx] = (empty_indices, empty_valid)
                    continue
                chunk_ids = torch.arange(
                    max_prior_chunks,
                    device=device,
                    dtype=current_chunks.dtype,
                )
                chunk_representatives = batched_chunk_representatives[seq_idx]
                group_top_indices, group_top_valid = self._dsa_score_group_top_chunks(
                    score_query_states=indexer_q[q_start:q_end, 0],
                    chunk_representatives=chunk_representatives[:max_prior_chunks, 0],
                    current_chunks=current_chunks,
                    max_prior_chunks=max_prior_chunks,
                    chunk_top_k=chunk_top_k,
                    indexer_scale=indexer_scale,
                    chunk_ids=chunk_ids,
                )
                top_chunks_by_seq[seq_idx] = (group_top_indices, group_top_valid)

        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            q_len = q_end - q_start
            seq_block_table = block_table[seq_idx]
            if self._dsa_sequence_fits_dense_attention(key_len, q_len):
                num_pages = math.ceil(key_len / chunk_size)
                table_parts.append(
                    seq_block_table[:num_pages].to(torch.int32).view(1, -1))
                request_lens_parts.append(
                    torch.full((1, ), q_len, device=device, dtype=torch.int32))
                seqused_k_parts.append(
                    torch.full((1, ), key_len, device=device, dtype=torch.int32))
                max_seqlen_q = max(max_seqlen_q, q_len)
                max_seqlen_k = max(max_seqlen_k, key_len)
                continue

            top_indices, top_valid = top_chunks_by_seq[seq_idx]
            num_chunks, query_position_start, current_chunks = sparse_info_by_seq[
                seq_idx]
            seq_positions = torch.arange(
                query_position_start,
                key_len,
                device=device,
                dtype=torch.long,
            )

            valid_counts = top_valid.sum(dim=-1).to(dtype=torch.long)
            top_width = int(top_indices.shape[1])
            logical_pages = current_chunks[:, None].expand(
                q_len, top_width + 1).clone()
            if top_width > 0:
                logical_pages[:, :top_width] = top_indices.to(dtype=torch.long)
            logical_pages.scatter_(1, valid_counts[:, None],
                                   current_chunks[:, None])

            physical_pages = seq_block_table.to(dtype=torch.long).expand(
                q_len, -1).gather(1, logical_pages).to(torch.int32)
            used_page_mask = (
                torch.arange(
                    top_width + 1,
                    device=device,
                    dtype=torch.long,
                )[None, :] <= valid_counts[:, None]
            )
            physical_pages.masked_fill_(~used_page_mask, 0)
            table_parts.append(physical_pages)

            local_prefixes = seq_positions - current_chunks * chunk_size + 1
            seqused_k_parts.append(
                (valid_counts * chunk_size + local_prefixes).to(torch.int32))
            request_lens_parts.append(
                torch.ones(q_len, device=device, dtype=torch.int32))
            max_seqlen_q = max(max_seqlen_q, 1)
            max_seqlen_k = max(
                max_seqlen_k,
                self._dsa_sparse_suffix_max_seqused_k(
                    query_position_start=query_position_start,
                    key_len=key_len,
                    chunk_size=chunk_size,
                    top_width=top_width,
                ),
            )

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
            plan_block_table[request_start:request_end, :int(
                pages.shape[1])] = pages
            request_start = request_end

        request_lens_t = torch.cat(request_lens_parts, dim=0)
        seqused_k_t = torch.cat(seqused_k_parts, dim=0)
        impl = getattr(self.attn, "impl", None)
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

    @staticmethod
    def _dsa_sparse_suffix_max_seqused_k(
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
        attn_metadata: typing.Any | None,
        top_chunk_indices: torch.Tensor,
        top_chunk_valid: torch.Tensor,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
        key_len: int,
        softmax_scale: float,
    ) -> torch.Tensor | None:
        if self.num_kv_heads != 1:
            return None
        if int(query_states.shape[0]) != 1:
            return None
        if top_chunk_indices.shape != top_chunk_valid.shape:
            raise ValueError(
                "top chunk shape mismatch, "
                f"indices={tuple(top_chunk_indices.shape)} "
                f"valid={tuple(top_chunk_valid.shape)}"
            )
        expected_top_prefix = (1, 1)
        if (
            top_chunk_indices.dim() != 3
            or tuple(top_chunk_indices.shape[:2]) != expected_top_prefix
        ):
            raise ValueError(
                "flattened decode page-table FA expects top chunks shaped "
                "(1, 1, top_k), "
                f"got {tuple(top_chunk_indices.shape)} "
                f"expected_prefix={expected_top_prefix}"
            )
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
            attn_metadata=attn_metadata,
            key_len=key_len,
        )
        if reason is not None:
            return None

        query_position = int(query_positions[0].item())
        if query_position != key_len - 1:
            return None
        current_chunk = int(current_chunks[0].item())
        if query_position < 0 or query_position >= key_len:
            return None
        valid_top_chunks = top_chunk_indices[0, 0].masked_select(
            top_chunk_valid[0, 0])
        max_page_id = current_chunk
        if valid_top_chunks.numel() > 0:
            max_page_id = max(max_page_id, int(valid_top_chunks.max().item()))
        if max_page_id >= int(block_table.shape[0]):
            return None

        flash_attn = _get_flash_attn_varlen_func()
        if flash_attn is None:
            return None
        device = query_states.device
        chunk_size = self.q_indexer_chunk_size
        if block_table.device != device:
            block_table = block_table.to(device=device)
        if top_chunk_indices.device != device:
            top_chunk_indices = top_chunk_indices.to(device=device)
        if top_chunk_valid.device != device:
            top_chunk_valid = top_chunk_valid.to(device=device)
        if current_chunks.device != device:
            current_chunks = current_chunks.to(device=device)
        if query_positions.device != device:
            query_positions = query_positions.to(device=device)
        valid_top_chunks = valid_top_chunks.to(device=device, dtype=torch.long)

        current_chunk_t = current_chunks.to(torch.long)
        logical_pages = torch.cat((valid_top_chunks, current_chunk_t))
        temp_block_table = block_table.index_select(0, logical_pages).to(
            torch.int32).view(1, -1)
        valid_top_count = int(valid_top_chunks.numel())
        tail_len = query_positions.to(torch.long) - current_chunk_t * chunk_size + 1
        seqused_k = (valid_top_count * chunk_size + tail_len).to(torch.int32)
        max_seqlen_k = int(seqused_k.max().item())

        impl = getattr(self.attn, "impl", None)
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
            cu_seqlens_q=torch.tensor([0, 1], device=device, dtype=torch.int32),
            max_seqlen_q=1,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            block_table=temp_block_table,
            **flash_attn_kwargs,
        )
        return output

    def _gather_kv_positions_for_head(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        token_indices: torch.Tensor,
        kv_head_idx: int,
    ) -> torch.Tensor:
        if cache.dim() != 4:
            raise NotImplementedError(
                f"DSA cache gather expects a 4D KV cache, got {cache.shape}")
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
                f"got shape={cache.shape}, num_kv_heads={self.num_kv_heads}")

        flat_token_indices = token_indices.reshape(-1)
        block_indices = torch.div(
            flat_token_indices, block_size, rounding_mode="floor").to(torch.long)
        block_offsets = flat_token_indices.remainder(block_size).to(torch.long)
        block_ids = block_table.index_select(0, block_indices).to(torch.long)
        if cache_layout == "NHD":
            selected = cache[block_ids, block_offsets, kv_head_idx]
        else:
            selected = cache[block_ids, kv_head_idx, block_offsets]
        return selected.view(*token_indices.shape, self.head_dim)

    def _build_indexer_chunk_representatives(
        self, indexer_key_states: torch.Tensor) -> torch.Tensor:
        key_len = indexer_key_states.shape[0]
        chunk_size = self.q_indexer_chunk_size
        num_chunks = math.ceil(key_len / chunk_size)
        padded_len = num_chunks * chunk_size
        if padded_len != key_len:
            padding = indexer_key_states.new_zeros(
                padded_len - key_len, self.num_kv_heads, self.q_indexer_dim)
            indexer_key_states = torch.cat((indexer_key_states, padding), dim=0)

        chunked_keys = indexer_key_states.view(
            num_chunks, chunk_size, self.num_kv_heads, self.q_indexer_dim)
        chunk_sums = chunked_keys.float().sum(dim=1)
        chunk_lengths = torch.full(
            (num_chunks,), chunk_size, device=indexer_key_states.device,
            dtype=chunk_sums.dtype)
        if padded_len != key_len:
            chunk_lengths[-1] = key_len - (num_chunks - 1) * chunk_size
        return chunk_sums / chunk_lengths[:, None, None]

    def _forward_dsa_chunked_sequence(
        self,
        *,
        query_states: torch.Tensor,
        indexer_query_states: torch.Tensor,
        key_states: torch.Tensor | None,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
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
                attn_metadata=attn_metadata,
                positions=positions,
                key_len=key_len,
            )
            if full_page_table_output is not None:
                return full_page_table_output

        chunk_size = self.q_indexer_chunk_size
        num_chunks = math.ceil(key_len / chunk_size)
        query_chunk_size = min(self.q_indexer_chunked_query_chunk_size, q_len)
        indexer_scale = self.q_indexer_logit_scale / math.sqrt(self.q_indexer_dim)
        main_scale = 1.0 / math.sqrt(self.head_dim)
        group_size = self.num_heads // self.num_kv_heads
        if chunk_representatives is None:
            # Build per-chunk key representatives from gathered KV states.
            if key_states is None:
                key_states = self._gather_kv_sequence(
                    key_cache, block_table, key_len)
            chunk_representatives = self._get_indexer_chunk_representatives(
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
            key_states = self._gather_kv_sequence(key_cache, block_table, key_len)
        chunk_offsets = torch.arange(
            chunk_size, device=query_states.device, dtype=torch.long)

        for query_start in range(0, q_len, query_chunk_size):
            query_end = min(query_start + query_chunk_size, q_len)
            chunk_len = query_end - query_start
            query_positions = positions[query_start:query_end].to(
                device=query_states.device, dtype=torch.long)
            current_chunks = torch.div(
                query_positions, chunk_size, rounding_mode="floor")
            current_chunks = current_chunks.clamp(min=0, max=num_chunks - 1)
            current_chunk_starts = current_chunks * chunk_size
            tail_indices = current_chunk_starts[:, None] + chunk_offsets[None, :]
            tail_valid = (tail_indices <= query_positions[:, None]) & (
                tail_indices < key_len)

            # Score up to the sequence bound; row masks keep causality exact.
            max_prior_chunks = max(num_chunks - 1, 0)
            if max_prior_chunks > 0:
                chunk_top_k = min(self.q_indexer_chunk_top_k, max_prior_chunks)
                chunk_ids = torch.arange(
                    max_prior_chunks, device=query_states.device, dtype=torch.long)
            else:
                chunk_top_k = 0
                chunk_ids = torch.empty(0, device=query_states.device, dtype=torch.long)

            if (
                chunk_len == 1
                and getattr(self, "q_indexer_use_page_table_fa", False)
                and getattr(self, "q_indexer_use_flattened_decode_page_table_fa", False)
            ):
                top_indices_by_group = torch.empty(
                    chunk_len,
                    self.num_kv_heads,
                    chunk_top_k,
                    device=query_states.device,
                    dtype=torch.long,
                )
                top_valid_by_group = torch.empty(
                    chunk_len,
                    self.num_kv_heads,
                    chunk_top_k,
                    device=query_states.device,
                    dtype=torch.bool,
                )
                for group_idx in range(self.num_kv_heads):
                    if chunk_top_k == 0:
                        continue
                    # Score one decode row for every KV group, then flatten.
                    score_query_states = indexer_query_states[
                        query_start:query_end, group_idx]
                    group_top_indices, group_top_valid = (
                        self._dsa_score_group_top_chunks(
                            score_query_states=score_query_states,
                            chunk_representatives=chunk_representatives[
                                :max_prior_chunks, group_idx],
                            current_chunks=current_chunks,
                            max_prior_chunks=max_prior_chunks,
                            chunk_top_k=chunk_top_k,
                            indexer_scale=indexer_scale,
                            chunk_ids=chunk_ids,
                        )
                    )
                    top_indices_by_group[:, group_idx] = group_top_indices
                    top_valid_by_group[:, group_idx] = group_top_valid

                flat_output = (
                    self._forward_dsa_chunked_flattened_decode_page_table_fa_sequence(
                        query_states=query_states[query_start:query_end],
                        key_cache=key_cache,
                        value_cache=value_cache,
                        block_table=block_table,
                        attn_metadata=attn_metadata,
                        top_chunk_indices=top_indices_by_group,
                        top_chunk_valid=top_valid_by_group,
                        current_chunks=current_chunks,
                        query_positions=query_positions,
                        key_len=key_len,
                        softmax_scale=main_scale,
                    )
                )
                if flat_output is not None:
                    output[query_start:query_end] = flat_output
                    continue

            for group_idx in range(self.num_kv_heads):
                head_start = group_idx * group_size
                head_end = head_start + group_size
                group_query_states = query_states[
                    query_start:query_end, head_start:head_end]

                if chunk_top_k > 0:
                    # Score prior chunks with eager torch matmul/top-k.
                    score_query_states = indexer_query_states[
                        query_start:query_end, group_idx]
                    top_chunk_indices, top_chunk_valid = (
                        self._dsa_score_group_top_chunks(
                            score_query_states=score_query_states,
                            chunk_representatives=chunk_representatives[
                                :max_prior_chunks, group_idx],
                            current_chunks=current_chunks,
                            max_prior_chunks=max_prior_chunks,
                            chunk_top_k=chunk_top_k,
                            indexer_scale=indexer_scale,
                            chunk_ids=chunk_ids,
                        )
                    )
                    # Expand selected chunks into token rows, then append causal tail.
                    chunk_token_indices = (
                        top_chunk_indices[..., None] * chunk_size
                        + chunk_offsets[None, None, :])
                    chunk_token_valid = top_chunk_valid[..., None] & (
                        chunk_token_indices < key_len)
                    chunk_token_indices = chunk_token_indices.reshape(
                        chunk_len, chunk_top_k * chunk_size)
                    chunk_token_valid = chunk_token_valid.reshape(
                        chunk_len, chunk_top_k * chunk_size)
                    recall_indices = torch.cat(
                        (chunk_token_indices, tail_indices), dim=-1)
                    recall_valid = torch.cat(
                        (chunk_token_valid, tail_valid), dim=-1)
                else:
                    top_chunk_indices = torch.empty(
                        chunk_len,
                        0,
                        device=query_states.device,
                        dtype=torch.long,
                    )
                    top_chunk_valid = torch.empty(
                        chunk_len,
                        0,
                        device=query_states.device,
                        dtype=torch.bool,
                    )
                    recall_indices = tail_indices
                    recall_valid = tail_valid

                # Gather the sparse token set when no flattened FA bucket applies.
                safe_recall_indices = recall_indices.masked_fill(
                    ~recall_valid, 0)
                selected_k = key_states[:, group_idx].index_select(
                    0, safe_recall_indices.reshape(-1))
                selected_v = self._gather_kv_positions_for_head(
                    value_cache, block_table, safe_recall_indices, group_idx)
                recall_len = recall_indices.shape[-1]
                selected_k = selected_k.view(chunk_len, recall_len, self.head_dim)
                selected_v = selected_v.view(chunk_len, recall_len, self.head_dim)

                main_logits = torch.einsum(
                    "qhd,qkd->hqk", group_query_states.float(),
                    selected_k.float())
                main_logits.mul_(main_scale)
                main_logits = main_logits.masked_fill(
                    ~recall_valid[None, :, :],
                    torch.finfo(main_logits.dtype).min,
                )
                attn_weights = F.softmax(
                    main_logits, dim=-1, dtype=torch.float32)
                attn_weights = attn_weights.to(query_states.dtype)
                output[query_start:query_end, head_start:head_end] = torch.einsum(
                    "hqk,qkd->qhd", attn_weights, selected_v)
        return output
