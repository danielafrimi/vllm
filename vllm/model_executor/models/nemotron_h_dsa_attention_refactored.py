# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sparse attention implementation for Nemotron-H DSA layers."""

import importlib
import inspect
import math
import os
import typing

import torch

from vllm.config import CacheConfig, ModelConfig, get_current_vllm_config_or_none
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
    unified_kv_cache_update,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.nemotron_h import NemotronHAttention
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

_DSA_PROVIDER_CLASS_ENV = "VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS"
_DSA_PROVIDER_MODULE_ENV = "VLLM_NEMOTRON_H_DSA_PROVIDER_MODULE"
_DSA_CHUNK_TOP_K_ENV = "VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K"
_TORCH_PROVIDER_CLASS = (
    "vllm.model_executor.models.nemotron_h_chunked_dsa_components_pytorch."
    "TorchChunkedDSAProviderBundle"
)
_EFFICIENT_PROVIDER_CLASS = (
    "vllm.model_executor.models.nemotron_h_chunked_dsa_components_efficient."
    "EfficientChunkedDSAProviderBundle"
)
_NONCHUNKED_TORCH_PROVIDER_CLASS = (
    "vllm.model_executor.models.nemotron_h_nonchunked_dsa_components_pytorch."
    "TorchNonChunkedDSAProviderBundle"
)
_DSA_PROVIDER_CLASS_ALIASES = {
    "efficient": _EFFICIENT_PROVIDER_CLASS,
    "cuda": _EFFICIENT_PROVIDER_CLASS,
    "triton": _EFFICIENT_PROVIDER_CLASS,
    "pytorch": _TORCH_PROVIDER_CLASS,
    "torch": _TORCH_PROVIDER_CLASS,
    "nonchunked": _NONCHUNKED_TORCH_PROVIDER_CLASS,
    "nonchunked-pytorch": _NONCHUNKED_TORCH_PROVIDER_CLASS,
    "token": _NONCHUNKED_TORCH_PROVIDER_CLASS,
    "token-pytorch": _NONCHUNKED_TORCH_PROVIDER_CLASS,
}
_DSA_LAYER_REGISTRY: dict[str, "NemotronHDSARefactoredAttention"] = {}


def _coalesce(value, default):
    return default if value is None else value


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def _resolve_provider_class(path: str) -> str:
    return _DSA_PROVIDER_CLASS_ALIASES.get(path, path)


def _load_class(path: str) -> type:
    path = _resolve_provider_class(path)
    if "." in path:
        module_name, class_name = path.rsplit(".", 1)
    else:
        module_name = os.environ.get(_DSA_PROVIDER_MODULE_ENV, __name__)
        class_name = path
    try:
        return getattr(importlib.import_module(module_name), class_name)
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Could not load Nemotron-H DSA provider class "
            f"{path!r} from {_DSA_PROVIDER_CLASS_ENV}"
        ) from exc


def _make_chunked_dsa_provider(**kwargs: typing.Any) -> typing.Any:
    default_class = (
        _EFFICIENT_PROVIDER_CLASS if torch.cuda.is_available()
        else _TORCH_PROVIDER_CLASS
    )
    provider_class = _load_class(
        os.environ.get(_DSA_PROVIDER_CLASS_ENV, default_class))
    return provider_class(**kwargs)


def _get_dsa_layer(
    layer_name: LayerNameType,
) -> "NemotronHDSARefactoredAttention":
    resolved = _resolve_layer_name(layer_name)
    if is_forward_context_available():
        layer = get_forward_context().no_compile_layers.get(resolved)
        if layer is not None:
            return typing.cast("NemotronHDSARefactoredAttention", layer)
    try:
        return _DSA_LAYER_REGISTRY[resolved]
    except KeyError as exc:
        raise RuntimeError(
            f"Nemotron-H DSA attention layer {resolved!r} is not registered"
        ) from exc


def nemotron_h_dsa_attention_with_output(
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
    indexer_q: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # Preserve the dependency from the KV-cache update through this opaque op.
    del kv_cache_dummy_dep
    dsa_layer = _get_dsa_layer(layer_name)
    dsa_layer._forward_dsa_attention_with_output(
        hidden_states=hidden_states,
        query_states=query_states,
        key_states=key_states,
        value_states=value_states,
        output=output,
        positions=positions,
        indexer_q=indexer_q,
    )


def nemotron_h_dsa_attention_with_output_fake(
    hidden_states: torch.Tensor,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    output: torch.Tensor,
    positions: torch.Tensor,
    indexer_q: torch.Tensor,
    layer_name: LayerNameType,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    return


direct_register_custom_op(
    op_name="nemotron_h_dsa_attention_with_output",
    op_func=nemotron_h_dsa_attention_with_output,
    mutates_args=["output"],
    fake_impl=nemotron_h_dsa_attention_with_output_fake,
)


class NemotronHDSARefactoredAttention(NemotronHAttention):
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
        self._init_dsa_component_providers()
        self.dsa_layer_name = f"{prefix}.dsa_attention"
        current_vllm_config = get_current_vllm_config_or_none()
        if current_vllm_config is not None:
            compilation_config = current_vllm_config.compilation_config
            if self.dsa_layer_name in compilation_config.static_forward_context:
                raise ValueError(f"Duplicate layer name: {self.dsa_layer_name}")
            compilation_config.static_forward_context[self.dsa_layer_name] = self
        _DSA_LAYER_REGISTRY[self.dsa_layer_name] = self

    def _init_dsa_component_providers(self) -> None:
        self.dsa_components = _make_chunked_dsa_provider(
            q_indexer_dim=self.q_indexer_dim,
            chunk_size=self.q_indexer_chunk_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            logit_scale=self.q_indexer_logit_scale,
            chunk_top_k=self.q_indexer_chunk_top_k,
            query_chunk_size=self.q_indexer_chunked_query_chunk_size,
            num_heads=self.num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
        )
        self.rep_provider = self.dsa_components.prepare_representatives
        self.score_provider = self.dsa_components.prepare_scores
        self.selection_provider = self.dsa_components.prepare_selection
        self.block_table_provider = self.dsa_components.prepare_block_tables
        self.attention_provider = self.dsa_components.forward_attention
        self._attention_provider_accepts_precomputed_indexer = (
            "precomputed_indexer_q"
            in inspect.signature(self.attention_provider).parameters
        )

    def _get_local_kv_head_indices(self) -> list[int]:
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if self.total_num_kv_heads >= tp_size:
            start = tp_rank * self.num_kv_heads
            return list(range(start, start + self.num_kv_heads))

        ranks_per_kv_head = tp_size // self.total_num_kv_heads
        return [tp_rank // ranks_per_kv_head]

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
        k_view = k.view(-1, self.num_kv_heads, self.head_dim)
        v_view = v.view(-1, self.num_kv_heads, self.head_dim)

        indexer_q, _ = self.indexer_q_proj(hidden_states)
        indexer_q = indexer_q.view(
            -1,
            self.total_num_kv_heads,
            self.q_indexer_dim,
        ).index_select(1, self._local_kv_head_indices.to(indexer_q.device))

        if is_forward_context_available():
            kv_cache_dummy_dep = torch.ops.vllm.unified_kv_cache_update(
                k_view,
                v_view,
                _encode_layer_name(self.attn.layer_name),
            )
        else:
            kv_cache_dummy_dep = unified_kv_cache_update(
                k_view,
                v_view,
                self.attn.layer_name,
            )

        attn_output = torch.empty_like(q)
        torch.ops.vllm.nemotron_h_dsa_attention_with_output(
            hidden_states,
            q,
            k,
            v,
            attn_output,
            positions,
            indexer_q,
            _encode_layer_name(self.dsa_layer_name),
            kv_cache_dummy_dep=kv_cache_dummy_dep,
        )

        output, _ = self.o_proj(attn_output)
        return output

    def _forward_dsa_attention_with_output(
        self,
        *,
        hidden_states: torch.Tensor,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        output: torch.Tensor,
        positions: torch.Tensor,
        indexer_q: torch.Tensor,
    ) -> None:
        q_view = query_states.view(-1, self.num_heads, self.head_dim)

        attn_metadata, _, kv_cache, _ = get_attention_context(self.attn.layer_name)
        if attn_metadata is None:
            output.copy_(self.attn(query_states, key_states, value_states))
            return

        key_cache, value_cache = self._split_kv_cache(kv_cache)
        block_table = attn_metadata.block_table
        active_seq_infos = self._dsa_active_sequence_infos(attn_metadata)
        indexer_q_by_head = tuple(
            indexer_q[:, kv_head_idx : kv_head_idx + 1].contiguous()
            for kv_head_idx in range(self.num_kv_heads)
        )

        representatives = self.rep_provider(
            key_cache=key_cache,
            block_table=block_table,
            seq_lens=getattr(attn_metadata, "seq_lens", None),
            query_start_loc=getattr(attn_metadata, "query_start_loc", None),
            active_seq_infos=active_seq_infos,
            cache_info=self.dsa_components.get_cache_info(key_cache),
        )
        scores = self.score_provider(representatives=representatives)
        selection = self.selection_provider(scores=scores)
        new_blocks = self.block_table_provider(selection=selection)
        attention_kwargs = dict(
            block_state=new_blocks,
            hidden_states=hidden_states,
            query_states=q_view,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn=self.attn,
            attn_metadata=attn_metadata,
            positions=positions,
            active_seq_infos=active_seq_infos,
            indexer_q_proj=self.indexer_q_proj,
            local_kv_head_indices=self._local_kv_head_indices,
        )
        if self._attention_provider_accepts_precomputed_indexer:
            attention_kwargs.update(
                precomputed_indexer_q=indexer_q,
                precomputed_indexer_q_by_head=indexer_q_by_head,
            )
        attn_output_view = self.attention_provider(**attention_kwargs)
        output.copy_(attn_output_view.reshape_as(output))

    def _split_kv_cache(
        self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_cache.dim() < 2:
            raise NotImplementedError(
                f"DSA KV cache expects at least 2 dimensions, got {kv_cache.shape}")
        if kv_cache.shape[0] == 2:
            return kv_cache.unbind(0)
        if kv_cache.shape[1] == 2:
            return kv_cache.unbind(1)
        raise NotImplementedError(
            "DSA KV cache only supports K/V stacked on dimension 0 or 1, "
            f"got shape={kv_cache.shape}")
