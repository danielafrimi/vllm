# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/vllm-project/vllm/blob/94d8ec8d2bcb4ec55e33022b313c7e978edf05e1/vllm/model_executor/models/bamba.py
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only NemotronH model."""

import importlib
import math
import os
import time
import typing
from collections.abc import Callable, Iterable, Mapping
from itertools import islice

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed import (
    get_ep_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention.attention import (
    get_attention_context,
    unified_kv_cache_update,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEFactory,
    GateLinear,
    activation_without_mul,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMambaPrefixCaching,
    SupportsPP,
    SupportsQuant,
    SupportsReplaySSM,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.nemotron_h import NemotronHConfig

try:
    from vllm.vllm_flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

try:
    from vllm.model_executor.layers.dsa_moonshot_attention import (
        dsa_prefill_gqa_union_attention,
    )
except ImportError:
    dsa_prefill_gqa_union_attention = None


_DSA_DEBUG_FORWARD_PRINT_LIMIT = 5
_DSA_DEBUG_FORWARD_PRINT_COUNT = 0
_DSA_PAGE_TABLE_FA_DEBUG_PRINT_LIMIT = 10
_DSA_PAGE_TABLE_FA_DEBUG_PRINT_COUNT = 0
_DSA_TIMING_DEBUG_PRINT_COUNT = 0
_DSA_TOPK_STATS_PRINT_COUNT = 0

_DSA_ATTENTION_CLASS_ENV = "VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS"
_DSA_ATTENTION_MODULE_ENV = "VLLM_NEMOTRON_H_DSA_ATTENTION_MODULE"
_DSA_PROVIDER_CLASS_ENV = "VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS"
_MOONSHOT_DSA_ATTENTION_CLASS = f"{__name__}.NemotronHDSASelectiveAttention"
_LEGACY_DSA_ATTENTION_CLASS = (
    "vllm.model_executor.models.nemotron_h_dsa_attention_legacy."
    "NemotronHDSALegacyAttention"
)
_REFACTORED_DSA_ATTENTION_CLASS = (
    "vllm.model_executor.models.nemotron_h_dsa_attention_refactored."
    "NemotronHDSARefactoredAttention"
)
_DSA_ATTENTION_CLASS_ALIASES: dict[str, tuple[str, str | None]] = {
    "moonshot": (_MOONSHOT_DSA_ATTENTION_CLASS, None),
    "vanilla": (_MOONSHOT_DSA_ATTENTION_CLASS, None),
    "legacy": (_LEGACY_DSA_ATTENTION_CLASS, None),
    "refactored": (_REFACTORED_DSA_ATTENTION_CLASS, None),
    "refactored-efficient": (_REFACTORED_DSA_ATTENTION_CLASS, "efficient"),
    "refactored-pytorch": (_REFACTORED_DSA_ATTENTION_CLASS, "pytorch"),
}


def _load_dsa_attention_class(path: str) -> type[nn.Module]:
    if "." in path:
        module_name, class_name = path.rsplit(".", 1)
    else:
        module_name = os.environ.get(_DSA_ATTENTION_MODULE_ENV, __name__)
        class_name = path
    try:
        attention_cls = getattr(importlib.import_module(module_name), class_name)
    except (AttributeError, ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Could not load Nemotron-H DSA attention class "
            f"{path!r} from {_DSA_ATTENTION_CLASS_ENV}"
        ) from exc
    if not isinstance(attention_cls, type) or not issubclass(attention_cls, nn.Module):
        raise TypeError(f"Nemotron-H DSA attention class {path!r} is not an nn.Module")
    return attention_cls


def _get_dsa_attention_class() -> type[nn.Module]:
    selection = os.environ.get(_DSA_ATTENTION_CLASS_ENV, "moonshot")
    path, provider = _DSA_ATTENTION_CLASS_ALIASES.get(
        selection, (selection, None)
    )
    if provider is not None:
        # The combined aliases make the complete implementation choice a
        # single environment-variable switch. Explicit class paths retain the
        # separate provider override for experiments and third-party classes.
        os.environ[_DSA_PROVIDER_CLASS_ENV] = provider
    return _load_dsa_attention_class(path)


def _print_dsa_forward_debug(message: str) -> None:
    global _DSA_DEBUG_FORWARD_PRINT_COUNT
    if os.environ.get("RANK", "0") != "0":
        return
    if _DSA_DEBUG_FORWARD_PRINT_COUNT >= _DSA_DEBUG_FORWARD_PRINT_LIMIT:
        return
    _DSA_DEBUG_FORWARD_PRINT_COUNT += 1
    print(message, flush=True)


def _print_dsa_page_table_fa_debug(message: str) -> None:
    global _DSA_PAGE_TABLE_FA_DEBUG_PRINT_COUNT
    if os.environ.get("RANK", "0") != "0":
        return
    if _DSA_PAGE_TABLE_FA_DEBUG_PRINT_COUNT >= _DSA_PAGE_TABLE_FA_DEBUG_PRINT_LIMIT:
        return
    _DSA_PAGE_TABLE_FA_DEBUG_PRINT_COUNT += 1
    print(message, flush=True)


def _dsa_timing_enabled() -> bool:
    return os.environ.get("VLLM_NEMOTRON_H_DSA_TIMING", "0") == "1"


def _dsa_timing_sync(device: torch.device | None) -> None:
    if device is not None and device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _dsa_timing_start(device: torch.device | None) -> float:
    if not _dsa_timing_enabled():
        return 0.0
    _dsa_timing_sync(device)
    return time.perf_counter()


def _dsa_timing_elapsed(start: float, device: torch.device | None) -> float:
    if not _dsa_timing_enabled():
        return 0.0
    _dsa_timing_sync(device)
    return (time.perf_counter() - start) * 1000.0


def _print_dsa_timing_debug(message: str) -> None:
    global _DSA_TIMING_DEBUG_PRINT_COUNT
    if not _dsa_timing_enabled():
        return
    if os.environ.get("RANK", "0") != "0":
        return
    limit = int(os.environ.get("VLLM_NEMOTRON_H_DSA_TIMING_PRINT_LIMIT", "200"))
    if _DSA_TIMING_DEBUG_PRINT_COUNT >= limit:
        return
    _DSA_TIMING_DEBUG_PRINT_COUNT += 1
    print(f"DSA_TIMING {message}", flush=True)


def _print_dsa_topk_stats(message: str) -> None:
    global _DSA_TOPK_STATS_PRINT_COUNT
    if os.environ.get("VLLM_NEMOTRON_H_DSA_TOPK_STATS", "0") != "1":
        return
    if os.environ.get("RANK", "0") != "0":
        return
    limit = int(os.environ.get("VLLM_NEMOTRON_H_DSA_TOPK_STATS_LIMIT", "200"))
    if _DSA_TOPK_STATS_PRINT_COUNT >= limit:
        return
    _DSA_TOPK_STATS_PRINT_COUNT += 1
    print(f"DSA_TOPK_STATS {message}", flush=True)


def _get_dsa_kv_cache_layout() -> str:
    try:
        from vllm.v1.attention.backends.utils import get_kv_cache_layout

        return get_kv_cache_layout()
    except AssertionError:
        # Unit tests may call the module directly without a current vLLM config.
        return "NHD"


def _split_dsa_kv_cache(
    kv_cache: torch.Tensor,
    *,
    num_kv_heads: int,
    head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return explicit NHD K/V views for old and current vLLM caches.

    vLLM 0.22 stored FlashAttention caches with a leading K/V axis, while
    current vLLM packs K and V into the final dimension of the logical
    ``[blocks, kv_heads, block_size, 2 * head_dim]`` tensor.  Check the current
    packed form first: for two-KV-head models, testing ``shape[1] == 2`` first
    would incorrectly interpret the KV-head axis as the old K/V axis.
    """
    if (
        kv_cache.dim() == 4
        and int(kv_cache.shape[1]) == num_kv_heads
        and int(kv_cache.shape[-1]) == 2 * head_dim
    ):
        return kv_cache.transpose(1, 2).split(head_dim, dim=-1)

    if kv_cache.dim() >= 2:
        if int(kv_cache.shape[0]) == 2:
            return kv_cache.unbind(0)
        if int(kv_cache.shape[1]) == 2:
            return kv_cache.unbind(1)
    raise NotImplementedError(
        "DSA KV cache must use current packed-KV or legacy stacked-KV storage, "
        f"got shape={tuple(kv_cache.shape)}, num_kv_heads={num_kv_heads}, "
        f"head_dim={head_dim}"
    )


def _coalesce(value, default):
    return default if value is None else value


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


class NemotronHMLP(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        self.act_fn = ReLUSquaredActivation()

    def forward(self, x: torch.Tensor):
        x, _ = self.up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class NemotronHMoE(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts
        self.use_latent_moe: bool = getattr(config, "moe_latent_size", None) is not None
        self.moe_hidden_size: int = (
            config.moe_latent_size if self.use_latent_moe else config.hidden_size
        )

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        self.gate = GateLinear(
            config.hidden_size,
            config.n_routed_experts,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )
        # Load balancing settings.
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts  # noqa: E501
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if config.n_shared_experts is None or config.n_shared_experts == 0:
            self.shared_experts = None
        else:
            intermediate_size = (
                config.moe_shared_expert_intermediate_size * config.n_shared_experts
            )

            self.shared_experts = NemotronHMLP(
                config=config,
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                quant_config=quant_config,
                reduce_results=False,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_experts",
            )

        if self.use_latent_moe:
            self.fc1_latent_proj = ReplicatedLinear(
                input_size=config.hidden_size,
                output_size=self.moe_hidden_size,
                bias=config.mlp_bias,
                quant_config=quant_config,
                disable_tp=self.is_sequence_parallel,
                prefix=f"{prefix}.fc1_latent_proj",
            )
            self.fc2_latent_proj = ReplicatedLinear(
                input_size=self.moe_hidden_size,
                output_size=config.hidden_size,
                bias=config.mlp_bias,
                quant_config=quant_config,
                disable_tp=self.is_sequence_parallel,
                prefix=f"{prefix}.fc2_latent_proj",
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None

        self.experts = FusedMoEFactory(
            shared_experts=self.shared_experts,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            ckpt_names=("up_proj", "down_proj", ""),
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            activation=activation_without_mul(config.mlp_hidden_act),
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            routed_input_transform=self.fc1_latent_proj,
            routed_output_transform=self.fc2_latent_proj,
            routed_scaling_factor=self.routed_scaling_factor,
            apply_routed_scale_to_output=True,
            router_logits_dtype=self.gate.out_dtype,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # router_logits: (num_tokens, n_experts)
        router_logits, _ = self.gate(hidden_states)

        final_hidden_states = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(num_tokens, hidden_dim)


class NemotronHMLPDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        hybrid_override_pattern = config.hybrid_override_pattern
        mlp_index = hybrid_override_pattern[: layer_idx + 1].count("-") - 1
        # Get per-layer config for heterogeneous models if exist
        get_layer_config = getattr(config, "get_nemotron_h_config_for_layer", None)
        layer_config = get_layer_config(layer_idx) if get_layer_config else config
        config = layer_config

        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[mlp_index]
        else:
            intermediate_size = config.intermediate_size

        self.mixer = NemotronHMLP(
            config,
            hidden_size=config.hidden_size,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            bias=config.mlp_bias,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class NemotronHMoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        # Get per-layer config for heterogeneous models if exists
        get_layer_config = getattr(config, "get_nemotron_h_config_for_layer", None)
        layer_config = get_layer_config(layer_idx) if get_layer_config else config

        self.mixer = NemotronHMoE(
            layer_config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class NemotronHMambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mixer = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=config.ssm_state_size,
            conv_kernel_size=config.conv_kernel,
            intermediate_size=config.mamba_num_heads * config.mamba_head_dim,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            n_groups=config.n_groups,
            num_heads=config.mamba_num_heads,
            head_dim=config.mamba_head_dim,
            rms_norm_eps=config.layer_norm_epsilon,
            activation=config.mamba_hidden_act,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        output = self.mixer(hidden_states)
        return output, residual


class NemotronHAttention(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Get per-layer sliding window from config (for heterogeneous models)
        sliding_window = getattr(config, "sliding_window", None)

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            per_layer_sliding_window=sliding_window,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class NemotronHDSASelectiveAttention(NemotronHAttention):
    """Correctness-first DSA top-k attention for Nemotron-H Puzzle layers."""

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

        self.q_indexer_attn_mode = os.environ.get(
            "VLLM_NEMOTRON_H_DSA_ATTN_MODE",
            _coalesce(getattr(config, "q_indexer_attn_mode", None), "topk_sparse"),
        )
        if self.q_indexer_attn_mode in {"chunked_sparse", "topk_chunked"}:
            self.q_indexer_attn_mode = "chunked_topk_sparse"
        if self.q_indexer_attn_mode not in {
            "topk_sparse",
            "chunked_topk_sparse",
            "pass_through",
            "disabled",
        }:
            raise ValueError(
                "Nemotron-H DSA supports topk_sparse/chunked_topk_sparse/"
                "pass_through/disabled, "
                f"got {self.q_indexer_attn_mode!r}"
            )
        self.q_indexer_logit_scale = float(
            _coalesce(getattr(config, "q_indexer_logit_scale", None), 1.0)
        )
        self.q_indexer_top_k = int(
            _coalesce(getattr(config, "q_indexer_top_k", None), 2048)
        )
        self.q_indexer_query_chunk_size = int(
            _coalesce(getattr(config, "q_indexer_query_chunk_size", None), 256)
        )
        if self.q_indexer_top_k <= 0:
            raise ValueError(
                f"q_indexer_top_k must be positive: {self.q_indexer_top_k}"
            )
        if self.q_indexer_query_chunk_size <= 0:
            raise ValueError(
                "q_indexer_query_chunk_size must be positive: "
                f"{self.q_indexer_query_chunk_size}"
            )
        self.q_indexer_chunk_size = _env_int(
            "VLLM_NEMOTRON_H_DSA_CHUNK_SIZE",
            int(_coalesce(getattr(config, "q_indexer_chunk_size", None), 16)),
        )
        if self.q_indexer_chunk_size <= 0:
            raise ValueError(
                f"q_indexer_chunk_size must be positive: {self.q_indexer_chunk_size}"
            )
        default_chunk_top_k = math.ceil(
            self.q_indexer_top_k / self.q_indexer_chunk_size
        )
        self.q_indexer_chunk_top_k = _env_int(
            "VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K",
            int(
                _coalesce(
                    getattr(config, "q_indexer_chunk_top_k", None),
                    default_chunk_top_k,
                )
            ),
        )
        if self.q_indexer_chunk_top_k <= 0:
            raise ValueError(
                "q_indexer_chunk_top_k must be positive: "
                f"{self.q_indexer_chunk_top_k}"
            )
        self.q_indexer_chunked_query_chunk_size = _env_int(
            "VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE",
            int(
                _coalesce(
                    getattr(config, "q_indexer_chunked_query_chunk_size", None),
                    min(self.q_indexer_query_chunk_size, 16),
                )
            ),
        )
        if self.q_indexer_chunked_query_chunk_size <= 0:
            raise ValueError(
                "q_indexer_chunked_query_chunk_size must be positive: "
                f"{self.q_indexer_chunked_query_chunk_size}"
            )
        self.q_indexer_use_flash_topk = (
            os.environ.get("VLLM_NEMOTRON_H_DSA_USE_FLASH_TOPK", "0") == "1"
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
        self.q_indexer_share_chunk_topk = (
            os.environ.get("VLLM_NEMOTRON_H_DSA_SHARE_CHUNK_TOPK", "0") == "1"
        )
        self.q_indexer_share_topk_group_size = _env_int(
            "VLLM_NEMOTRON_H_DSA_SHARE_TOPK_GROUP_SIZE",
            self.q_indexer_chunk_size,
        )
        if self.q_indexer_share_topk_group_size <= 0:
            raise ValueError(
                "q_indexer_share_topk_group_size must be positive: "
                f"{self.q_indexer_share_topk_group_size}"
            )
        share_topk_mode = os.environ.get(
            "VLLM_NEMOTRON_H_DSA_SHARE_TOPK_MODE",
            "representative",
        ).strip().lower().replace("-", "_")
        share_topk_mode_aliases = {
            "avg": "mean",
            "average": "mean",
            "noncausal_mean": "mean",
            "score_sum": "strict_union_sum",
            "union_sum": "strict_union_sum",
            "vote": "histogram",
            "voting": "histogram",
            "histogram_vote": "histogram",
            "histogram_voting": "histogram",
        }
        self.q_indexer_share_topk_mode = share_topk_mode_aliases.get(
            share_topk_mode,
            share_topk_mode,
        )
        if self.q_indexer_share_topk_mode not in {
            "representative",
            "mean",
            "causal_mean",
            "histogram",
            "union",
            "strict_union_sum",
        }:
            raise ValueError(
                "VLLM_NEMOTRON_H_DSA_SHARE_TOPK_MODE must be one of "
                "representative, mean, causal_mean, histogram, union, or "
                "strict_union_sum, got "
                f"{share_topk_mode!r}"
            )
        self.q_indexer_share_topk_union_max_chunks = _env_int(
            "VLLM_NEMOTRON_H_DSA_SHARE_TOPK_UNION_MAX_CHUNKS",
            max(self.q_indexer_chunk_top_k, self.q_indexer_chunk_top_k * 2),
        )
        if self.q_indexer_share_topk_union_max_chunks <= 0:
            raise ValueError(
                "q_indexer_share_topk_union_max_chunks must be positive: "
                f"{self.q_indexer_share_topk_union_max_chunks}"
            )
        self.q_indexer_use_shared_prefill_page_table_fa = (
            os.environ.get(
                "VLLM_NEMOTRON_H_DSA_USE_SHARED_PREFILL_PAGE_TABLE_FA", "0"
            )
            == "1"
        )
        self.q_indexer_use_union_prefill_kernel = (
            os.environ.get("VLLM_NEMOTRON_H_DSA_USE_UNION_PREFILL_KERNEL", "0")
            == "1"
        )
        self.q_indexer_use_union_superset_prefill_page_table_fa = (
            os.environ.get(
                "VLLM_NEMOTRON_H_DSA_USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA",
                "0",
            )
            == "1"
        )
        self.q_indexer_union_chunks_per_iter = _env_int(
            "VLLM_NEMOTRON_H_DSA_UNION_CHUNKS_PER_ITER",
            8,
        )
        if self.q_indexer_union_chunks_per_iter <= 0:
            raise ValueError(
                "q_indexer_union_chunks_per_iter must be positive: "
                f"{self.q_indexer_union_chunks_per_iter}"
            )
        self._dsa_cache_config_block_size = (
            getattr(cache_config, "block_size", None)
            if cache_config is not None
            else None
        )
        self.q_indexer_use_summary_cache = (
            os.environ.get("VLLM_NEMOTRON_H_DSA_USE_SUMMARY_CACHE", "0") == "1"
        )
        self.q_indexer_summary_cache_max_blocks = _env_int(
            "VLLM_NEMOTRON_H_DSA_SUMMARY_CACHE_MAX_BLOCKS",
            65536,
        )
        if self.q_indexer_summary_cache_max_blocks <= 0:
            raise ValueError(
                "q_indexer_summary_cache_max_blocks must be positive: "
                f"{self.q_indexer_summary_cache_max_blocks}"
            )
        self._dsa_summary_cache_block_ids: torch.Tensor | None = None
        self._dsa_summary_cache_values: torch.Tensor | None = None
        self._dsa_summary_cache_valid: torch.Tensor | None = None
        self._dsa_summary_cache_block_size: int | None = None

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

    def _dsa_dense_attention_budget_tokens(self) -> int:
        if self.q_indexer_attn_mode == "chunked_topk_sparse":
            return self.q_indexer_chunk_size * self.q_indexer_chunk_top_k
        return self.q_indexer_top_k

    def _dsa_sequence_fits_dense_attention(self, key_len: int) -> bool:
        return (
            self.q_indexer_use_full_attention_short_seq
            and key_len <= self._dsa_dense_attention_budget_tokens()
        )

    def _dsa_active_sequence_infos(
        self,
        attn_metadata: typing.Any,
    ) -> list[tuple[int, int, int, int]]:
        num_actual_tokens = int(attn_metadata.num_actual_tokens)
        query_start_loc = attn_metadata.query_start_loc
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

    def _dsa_all_sequences_fit_dense_attention(
        self,
        active_seq_infos: list[tuple[int, int, int, int]],
    ) -> bool:
        return bool(active_seq_infos) and all(
            self._dsa_sequence_fits_dense_attention(key_len)
            for _, _, _, key_len in active_seq_infos
        )

    @torch.compiler.disable
    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.q_indexer_attn_mode in {"pass_through", "disabled"}:
            return super().forward(hidden_states=hidden_states, **kwargs)
        if positions is None:
            raise ValueError("DSA selective attention requires token positions")

        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if self.attn.kv_cache_dtype not in {"auto", "float16", "bfloat16"}:
            attn_output = self.attn(q, k, v)
            output, _ = self.o_proj(attn_output)
            return output

        q_view = q.view(-1, self.num_heads, self.head_dim)
        k_view = k.view(-1, self.num_kv_heads, self.head_dim)
        v_view = v.view(-1, self.num_kv_heads, self.head_dim)

        attn_metadata, _, kv_cache, _ = get_attention_context(self.attn.layer_name)
        attn_output = q.new_zeros(q.shape)
        if attn_metadata is None:
            output, _ = self.o_proj(attn_output)
            return output

        active_seq_infos = self._dsa_active_sequence_infos(attn_metadata)
        if self._dsa_all_sequences_fit_dense_attention(active_seq_infos):
            timing_start = _dsa_timing_start(hidden_states.device)
            attn_output = self.attn(q, k, v)
            full_attention_ms = _dsa_timing_elapsed(timing_start, hidden_states.device)
            self._invalidate_dsa_summary_cache_for_slots(
                attn_metadata,
                block_size=getattr(self, "_dsa_cache_config_block_size", None),
            )
            max_key_len = max(key_len for _, _, _, key_len in active_seq_infos)
            _print_dsa_timing_debug(
                "path=full_attention_short_batch "
                f"layer={self.layer_idx} seqs={len(active_seq_infos)} "
                f"max_key_len={max_key_len} "
                f"budget={self._dsa_dense_attention_budget_tokens()} "
                f"total_ms={full_attention_ms:.3f}"
            )
            output, _ = self.o_proj(attn_output)
            return output

        unified_kv_cache_update(k_view, v_view, self.attn.layer_name)

        key_cache, value_cache = self._split_kv_cache(kv_cache)
        block_table = attn_metadata.block_table
        cache_info = self._dsa_kv_cache_layout_and_block_size(key_cache)
        if cache_info is not None:
            _, cache_block_size = cache_info
            self._invalidate_dsa_summary_cache_for_slots(
                attn_metadata,
                block_size=cache_block_size,
            )

        for seq_idx, q_start, q_end, key_len in active_seq_infos:
            if self._dsa_sequence_fits_dense_attention(key_len):
                seq_output = self._forward_dsa_full_page_table_fa_sequence(
                    query_states=q_view[q_start:q_end],
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=block_table[seq_idx],
                    attn_metadata=attn_metadata,
                    positions=positions[q_start:q_end],
                    key_len=key_len,
                )
                if seq_output is not None:
                    attn_output[q_start:q_end] = seq_output.reshape(q_end - q_start, -1)
                    continue

            timing_device = hidden_states.device
            timing_start = _dsa_timing_start(timing_device)
            indexer_q, _ = self.indexer_q_proj(hidden_states[q_start:q_end])
            indexer_q = indexer_q.view(
                -1,
                self.total_num_kv_heads,
                self.q_indexer_dim,
            ).index_select(1, self._local_kv_head_indices.to(indexer_q.device))
            indexer_proj_ms = _dsa_timing_elapsed(timing_start, timing_device)

            if self.q_indexer_attn_mode == "chunked_topk_sparse":
                seq_output = self._forward_dsa_chunked_sequence(
                    query_states=q_view[q_start:q_end],
                    indexer_query_states=indexer_q,
                    key_states=None,
                    key_len=key_len,
                    key_cache=key_cache,
                    value_cache=value_cache,
                    block_table=block_table[seq_idx],
                    attn_metadata=attn_metadata,
                    positions=positions[q_start:q_end],
                    debug_indexer_proj_ms=indexer_proj_ms,
                    debug_key_gather_ms=0.0,
                )
            else:
                timing_start = _dsa_timing_start(key_cache.device)
                key_states = self._gather_kv_sequence(
                    key_cache,
                    block_table[seq_idx],
                    key_len,
                )
                key_gather_ms = _dsa_timing_elapsed(timing_start, key_cache.device)
                seq_output = self._forward_dsa_sequence(
                    query_states=q_view[q_start:q_end],
                    indexer_query_states=indexer_q,
                    key_states=key_states,
                    value_cache=value_cache,
                    block_table=block_table[seq_idx],
                    positions=positions[q_start:q_end],
                )
            attn_output[q_start:q_end] = seq_output.reshape(q_end - q_start, -1)

        output, _ = self.o_proj(attn_output)
        return output

    def _split_kv_cache(self, kv_cache: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _split_dsa_kv_cache(
            kv_cache,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
        )

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
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA short-seq full FlashAttention fallback: "
                f"{reason} layer={self.layer_idx}"
            )
            return None

        assert flash_attn_varlen_func is not None
        device = query_states.device
        query_len = query_states.shape[0]
        block_size = int(key_cache.shape[1])
        num_blocks = math.ceil(key_len / block_size)
        if block_table.device != device:
            block_table = block_table.to(device=device)

        full_timing_start = _dsa_timing_start(device)
        table_timing_start = _dsa_timing_start(device)
        temp_block_table = block_table[:num_blocks].reshape(1, num_blocks)
        table_build_ms = _dsa_timing_elapsed(table_timing_start, device)

        cu_seqlens_q = torch.tensor([0, query_len], device=device, dtype=torch.int32)
        seqused_k = torch.tensor([key_len], device=device, dtype=torch.int32)
        output = torch.empty_like(query_states)
        impl = getattr(self.attn, "impl", None)
        fa_version = getattr(impl, "vllm_flash_attn_version", None)

        flash_attn_kwargs: dict[str, typing.Any] = {}
        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        fa_timing_start = _dsa_timing_start(device)
        flash_attn_varlen_func(
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
        fa_ms = _dsa_timing_elapsed(fa_timing_start, device)
        total_ms = _dsa_timing_elapsed(full_timing_start, device)
        _print_dsa_timing_debug(
            f"path=full_page_table layer={self.layer_idx} "
            f"q_len={query_len} key_len={key_len} "
            f"num_blocks={num_blocks} table_build_ms={table_build_ms:.3f} "
            f"fa_ms={fa_ms:.3f} total_ms={total_ms:.3f}"
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
        positions: torch.Tensor,
        key_len: int,
        allow_long_sequence: bool = False,
    ) -> str | None:
        if not self.q_indexer_use_full_attention_short_seq and not allow_long_sequence:
            return "short-sequence full attention is disabled"
        if flash_attn_varlen_func is None:
            return "flash_attn_varlen_func is unavailable"
        query_len = int(query_states.shape[0])
        if query_len <= 0:
            return "query sequence is empty"
        if positions.numel() != query_len:
            return (
                "position metadata must match query length, "
                f"query_len={query_len} positions={int(positions.numel())}"
            )
        expected_positions = torch.arange(
            key_len - query_len,
            key_len,
            device=positions.device,
            dtype=positions.dtype,
        )
        if not bool(torch.equal(positions, expected_positions)):
            return "query positions are not the final contiguous suffix of the KV sequence"
        if (
            not allow_long_sequence
            and key_len > self._dsa_dense_attention_budget_tokens()
        ):
            return (
                "sequence exceeds dense attention budget, "
                f"key_len={key_len} budget={self._dsa_dense_attention_budget_tokens()}"
            )
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
            self.num_kv_heads,
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
                f"DSA cache gather expects a 4D KV cache, got {cache.shape}"
            )
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

    def _reset_dsa_summary_cache(self) -> None:
        self._dsa_summary_cache_block_ids = None
        self._dsa_summary_cache_values = None
        self._dsa_summary_cache_valid = None
        self._dsa_summary_cache_block_size = None

    def _dsa_shared_topk_run_starts(
        self,
        *,
        current_chunks: torch.Tensor,
        query_positions: torch.Tensor,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_len = int(current_chunks.shape[0])
        group_size = min(self.q_indexer_share_topk_group_size, chunk_size)
        new_run = torch.ones(
            chunk_len,
            device=current_chunks.device,
            dtype=torch.bool,
        )
        if chunk_len > 1:
            run_breaks = current_chunks[1:] != current_chunks[:-1]
            if group_size < chunk_size:
                group_ids = torch.div(
                    query_positions.to(torch.long).remainder(chunk_size),
                    group_size,
                    rounding_mode="floor",
                )
                run_breaks |= group_ids[1:] != group_ids[:-1]
            new_run[1:] = run_breaks
        run_starts = torch.nonzero(new_run, as_tuple=False).reshape(-1)
        run_ends = torch.cat(
            (
                run_starts[1:],
                torch.tensor(
                    [chunk_len],
                    device=current_chunks.device,
                    dtype=run_starts.dtype,
                ),
            )
        )
        return run_starts, run_ends - run_starts

    def _dsa_mean_shared_query_states(
        self,
        *,
        score_query_states: torch.Tensor,
        shared_run_starts: torch.Tensor,
        shared_run_counts: torch.Tensor,
    ) -> torch.Tensor:
        run_ends = shared_run_starts + shared_run_counts
        prefix = score_query_states.float().cumsum(dim=0)
        sums = prefix.index_select(0, run_ends - 1)
        before_start_rows = shared_run_starts - 1
        has_before = before_start_rows >= 0
        if bool(has_before.any().item()):
            sums[has_before] -= prefix.index_select(
                0,
                before_start_rows[has_before],
            )
        counts = shared_run_counts.to(prefix.dtype).unsqueeze(-1)
        return sums / counts

    def _dsa_causal_mean_shared_query_states(
        self,
        *,
        score_query_states: torch.Tensor,
        current_chunks: torch.Tensor,
        shared_run_starts: torch.Tensor,
    ) -> torch.Tensor:
        source_rows = (shared_run_starts - 1).clamp_min(0)
        target_chunks = current_chunks.index_select(0, shared_run_starts)
        source_chunks = current_chunks.index_select(0, source_rows)
        source_rows = torch.where(
            source_chunks == target_chunks,
            source_rows,
            shared_run_starts,
        )

        _, chunk_run_counts = torch.unique_consecutive(
            current_chunks,
            return_counts=True,
        )
        chunk_run_ends = chunk_run_counts.cumsum(0)
        chunk_run_starts = chunk_run_ends - chunk_run_counts
        chunk_start_for_row = torch.repeat_interleave(
            chunk_run_starts,
            chunk_run_counts,
        )
        start_rows = chunk_start_for_row.index_select(0, source_rows)

        prefix = score_query_states.float().cumsum(dim=0)
        sums = prefix.index_select(0, source_rows)
        before_start_rows = start_rows - 1
        has_before = before_start_rows >= 0
        if bool(has_before.any().item()):
            sums[has_before] -= prefix.index_select(
                0,
                before_start_rows[has_before],
            )
        counts = (source_rows - start_rows + 1).to(prefix.dtype).unsqueeze(-1)
        return sums / counts

    def _dsa_aggregate_shared_topk(
        self,
        *,
        top_chunk_indices: torch.Tensor,
        top_chunk_valid: torch.Tensor,
        chunk_logits: torch.Tensor | None = None,
        shared_run_starts: torch.Tensor,
        shared_run_counts: torch.Tensor,
        chunk_top_k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mode = self.q_indexer_share_topk_mode
        if mode == "representative":
            raise ValueError("representative sharing should not aggregate top-k")

        cap = chunk_top_k
        if mode == "union":
            cap = self.q_indexer_share_topk_union_max_chunks

        selected_chunks: list[torch.Tensor] = []
        max_width = 0
        for run_start, run_count in zip(shared_run_starts, shared_run_counts):
            start = int(run_start.item())
            end = start + int(run_count.item())
            run_indices = top_chunk_indices[start:end]
            run_valid = top_chunk_valid[start:end]
            valid_ids = run_indices.masked_select(run_valid)
            if valid_ids.numel() == 0:
                selected = valid_ids
            else:
                unique_ids, inverse = torch.unique(
                    valid_ids,
                    sorted=True,
                    return_inverse=True,
                )
                width = min(int(unique_ids.numel()), cap)
                if mode == "strict_union_sum":
                    if chunk_logits is None:
                        raise ValueError(
                            "strict_union_sum requires per-query chunk logits"
                        )
                    run_logits = chunk_logits[start:end].index_select(
                        1,
                        unique_ids,
                    )
                    scores = run_logits.sum(dim=0)
                    _, order = torch.topk(
                        scores,
                        k=width,
                        largest=True,
                        sorted=True,
                    )
                    selected = unique_ids.index_select(0, order)
                elif mode == "histogram" or int(unique_ids.numel()) > width:
                    counts = torch.bincount(
                        inverse,
                        minlength=int(unique_ids.numel()),
                    )
                    _, order = torch.topk(
                        counts,
                        k=width,
                        largest=True,
                        sorted=True,
                    )
                    selected = unique_ids.index_select(0, order)
                else:
                    selected = unique_ids
            selected_chunks.append(selected)
            max_width = max(max_width, int(selected.numel()))

        num_runs = int(shared_run_starts.numel())
        out_indices = top_chunk_indices.new_zeros((num_runs, max_width))
        out_valid = torch.zeros(
            (num_runs, max_width),
            device=top_chunk_valid.device,
            dtype=torch.bool,
        )
        for row, selected in enumerate(selected_chunks):
            width = int(selected.numel())
            if width > 0:
                out_indices[row, :width] = selected
                out_valid[row, :width] = True
        return out_indices, out_valid

    def _ensure_dsa_summary_cache(
        self,
        *,
        device: torch.device,
        block_size: int,
    ) -> None:
        cache_block_ids = getattr(self, "_dsa_summary_cache_block_ids", None)
        cache_values = getattr(self, "_dsa_summary_cache_values", None)
        cache_valid = getattr(self, "_dsa_summary_cache_valid", None)
        cache_block_size = getattr(self, "_dsa_summary_cache_block_size", None)
        cache_ready = (
            cache_block_ids is not None
            and cache_values is not None
            and cache_valid is not None
            and cache_block_size == block_size
            and cache_block_ids.device == device
            and cache_values.device == device
            and cache_valid.device == device
        )
        if cache_ready:
            return

        self._dsa_summary_cache_block_ids = torch.empty(
            0,
            device=device,
            dtype=torch.long,
        )
        self._dsa_summary_cache_values = torch.empty(
            0,
            self.num_kv_heads,
            self.q_indexer_dim,
            device=device,
            dtype=torch.float32,
        )
        self._dsa_summary_cache_valid = torch.empty(
            0,
            device=device,
            dtype=torch.bool,
        )
        self._dsa_summary_cache_block_size = block_size

    def _lookup_dsa_summary_cache(
        self,
        physical_block_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cache_block_ids = self._dsa_summary_cache_block_ids
        cache_valid = self._dsa_summary_cache_valid
        if cache_block_ids is None or cache_valid is None:
            positions = torch.zeros_like(physical_block_ids, dtype=torch.long)
            matched = torch.zeros_like(physical_block_ids, dtype=torch.bool)
            return positions, matched, matched
        if cache_block_ids.numel() == 0:
            positions = torch.zeros_like(physical_block_ids, dtype=torch.long)
            matched = torch.zeros_like(physical_block_ids, dtype=torch.bool)
            return positions, matched, matched

        positions = torch.searchsorted(cache_block_ids, physical_block_ids)
        in_bounds = positions < cache_block_ids.numel()
        safe_positions = positions.clamp_max(cache_block_ids.numel() - 1)
        matched = in_bounds & (
            cache_block_ids.index_select(0, safe_positions) == physical_block_ids
        )
        valid = matched & cache_valid.index_select(0, safe_positions)
        return safe_positions, matched, valid

    def _invalidate_dsa_summary_cache_for_slots(
        self,
        attn_metadata: typing.Any | None,
        *,
        block_size: int | None = None,
    ) -> None:
        cache_block_ids = getattr(self, "_dsa_summary_cache_block_ids", None)
        cache_valid = getattr(self, "_dsa_summary_cache_valid", None)
        if cache_block_ids is None or cache_valid is None or cache_block_ids.numel() == 0:
            return
        if attn_metadata is None:
            return
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None or slot_mapping.numel() == 0:
            return

        if block_size is None:
            block_size = getattr(self, "_dsa_summary_cache_block_size", None)
        if block_size is None:
            block_size = getattr(self, "_dsa_cache_config_block_size", None)
        if block_size is None or block_size <= 0:
            self._reset_dsa_summary_cache()
            return

        slots = slot_mapping.reshape(-1).to(device=cache_block_ids.device)
        slots = slots[slots >= 0]
        if slots.numel() == 0:
            return
        written_blocks = torch.unique(
            torch.div(slots, int(block_size), rounding_mode="floor").to(torch.long)
        )
        positions, matched, _ = self._lookup_dsa_summary_cache(written_blocks)
        if matched.any():
            cache_valid.index_fill_(0, positions[matched], False)

    def _compute_dsa_block_summaries_from_cache(
        self,
        *,
        key_cache: torch.Tensor,
        physical_block_ids: torch.Tensor,
        chunk_lengths: torch.Tensor,
        cache_layout: str,
        block_size: int,
    ) -> torch.Tensor:
        if physical_block_ids.numel() == 0:
            return key_cache.new_empty(
                0,
                self.num_kv_heads,
                self.q_indexer_dim,
                dtype=torch.float32,
            )

        physical_block_ids = physical_block_ids.to(
            device=key_cache.device,
            dtype=torch.long,
        )
        chunk_lengths = chunk_lengths.to(device=key_cache.device, dtype=torch.long)
        if cache_layout == "NHD":
            block_states = key_cache.index_select(0, physical_block_ids)[
                :, :, :, : self.q_indexer_dim
            ]
            offsets = torch.arange(block_size, device=key_cache.device)
            valid = offsets[None, :] < chunk_lengths[:, None]
            block_sums = block_states.float().masked_fill(
                ~valid[:, :, None, None],
                0.0,
            ).sum(dim=1)
        else:
            block_states = key_cache.index_select(0, physical_block_ids)[
                :, :, :, : self.q_indexer_dim
            ]
            offsets = torch.arange(block_size, device=key_cache.device)
            valid = offsets[None, :] < chunk_lengths[:, None]
            block_sums = block_states.float().masked_fill(
                ~valid[:, None, :, None],
                0.0,
            ).sum(dim=2)

        lengths = chunk_lengths.clamp_min(1).to(dtype=block_sums.dtype)
        return block_sums / lengths[:, None, None]

    def _get_indexer_chunk_representatives(
        self,
        *,
        key_states: torch.Tensor | None,
        key_cache: torch.Tensor,
        block_table: torch.Tensor,
        key_len: int,
    ) -> torch.Tensor:
        if key_states is not None:
            return self._build_indexer_chunk_representatives(
                key_states[..., : self.q_indexer_dim]
            )

        cache_info = self._dsa_kv_cache_layout_and_block_size(key_cache)
        use_summary_cache = getattr(self, "q_indexer_use_summary_cache", True)
        if (
            not use_summary_cache
            or cache_info is None
            or block_table.dim() != 1
        ):
            key_states = self._gather_kv_sequence(key_cache, block_table, key_len)
            return self._build_indexer_chunk_representatives(
                key_states[..., : self.q_indexer_dim]
            )

        cache_layout, block_size = cache_info
        chunk_size = self.q_indexer_chunk_size
        num_chunks = math.ceil(key_len / chunk_size)
        if block_size != chunk_size or num_chunks > int(block_table.shape[0]):
            key_states = self._gather_kv_sequence(key_cache, block_table, key_len)
            return self._build_indexer_chunk_representatives(
                key_states[..., : self.q_indexer_dim]
            )

        if block_table.device != key_cache.device:
            block_table = block_table.to(device=key_cache.device)

        logical_chunk_ids = torch.arange(
            num_chunks,
            device=key_cache.device,
            dtype=torch.long,
        )
        physical_block_ids = block_table.index_select(0, logical_chunk_ids).to(
            torch.long
        )
        chunk_lengths = torch.full(
            (num_chunks,),
            chunk_size,
            device=key_cache.device,
            dtype=torch.long,
        )
        last_len = key_len - (num_chunks - 1) * chunk_size
        chunk_lengths[-1] = last_len

        self._ensure_dsa_summary_cache(
            device=key_cache.device,
            block_size=block_size,
        )
        cache_block_ids = self._dsa_summary_cache_block_ids
        cache_values = self._dsa_summary_cache_values
        cache_valid = self._dsa_summary_cache_valid
        assert cache_block_ids is not None
        assert cache_values is not None
        assert cache_valid is not None

        representatives = torch.empty(
            num_chunks,
            self.num_kv_heads,
            self.q_indexer_dim,
            device=key_cache.device,
            dtype=torch.float32,
        )
        cache_positions, matched, valid = self._lookup_dsa_summary_cache(
            physical_block_ids
        )
        if valid.any():
            representatives[valid] = cache_values.index_select(
                0,
                cache_positions[valid],
            )

        missing = ~valid
        if missing.any():
            missing_positions = torch.nonzero(missing, as_tuple=False).reshape(-1)
            missing_block_ids = physical_block_ids.index_select(
                0,
                missing_positions,
            )
            missing_lengths = chunk_lengths.index_select(0, missing_positions)
            missing_summaries = self._compute_dsa_block_summaries_from_cache(
                key_cache=key_cache,
                physical_block_ids=missing_block_ids,
                chunk_lengths=missing_lengths,
                cache_layout=cache_layout,
                block_size=block_size,
            )
            representatives.index_copy_(0, missing_positions, missing_summaries)

            missing_matched = matched.index_select(0, missing_positions)
            if missing_matched.any():
                matched_cache_positions = cache_positions.index_select(
                    0,
                    missing_positions[missing_matched],
                )
                cache_values.index_copy_(
                    0,
                    matched_cache_positions,
                    missing_summaries[missing_matched],
                )
                cache_valid.index_fill_(0, matched_cache_positions, True)

            cacheable = (~missing_matched) & (missing_lengths == block_size)
            if cacheable.any():
                new_block_ids = missing_block_ids[cacheable]
                new_summaries = missing_summaries[cacheable]
                sorted_ids, order = new_block_ids.sort()
                sorted_summaries = new_summaries.index_select(0, order)
                unique = torch.ones_like(sorted_ids, dtype=torch.bool)
                unique[1:] = sorted_ids[1:] != sorted_ids[:-1]
                sorted_ids = sorted_ids[unique]
                sorted_summaries = sorted_summaries[unique]

                max_blocks = getattr(
                    self,
                    "q_indexer_summary_cache_max_blocks",
                    65536,
                )
                if sorted_ids.numel() <= max_blocks:
                    if cache_block_ids.numel() + sorted_ids.numel() > max_blocks:
                        self._ensure_dsa_summary_cache(
                            device=key_cache.device,
                            block_size=block_size,
                        )
                        self._dsa_summary_cache_block_ids = torch.empty(
                            0,
                            device=key_cache.device,
                            dtype=torch.long,
                        )
                        self._dsa_summary_cache_values = torch.empty(
                            0,
                            self.num_kv_heads,
                            self.q_indexer_dim,
                            device=key_cache.device,
                            dtype=torch.float32,
                        )
                        self._dsa_summary_cache_valid = torch.empty(
                            0,
                            device=key_cache.device,
                            dtype=torch.bool,
                        )
                        cache_block_ids = self._dsa_summary_cache_block_ids
                        cache_values = self._dsa_summary_cache_values
                        cache_valid = self._dsa_summary_cache_valid
                        assert cache_block_ids is not None
                        assert cache_values is not None
                        assert cache_valid is not None

                    merged_ids = torch.cat((cache_block_ids, sorted_ids), dim=0)
                    merged_values = torch.cat(
                        (cache_values, sorted_summaries),
                        dim=0,
                    )
                    merged_valid = torch.cat(
                        (
                            cache_valid,
                            torch.ones_like(sorted_ids, dtype=torch.bool),
                        ),
                        dim=0,
                    )
                    merged_ids, order = merged_ids.sort()
                    self._dsa_summary_cache_block_ids = merged_ids
                    self._dsa_summary_cache_values = merged_values.index_select(
                        0,
                        order,
                    )
                    self._dsa_summary_cache_valid = merged_valid.index_select(
                        0,
                        order,
                    )

        return representatives

    def _gather_kv_positions_for_head(
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

    def _build_indexer_chunk_representatives(
        self,
        indexer_key_states: torch.Tensor,
    ) -> torch.Tensor:
        key_len = indexer_key_states.shape[0]
        chunk_size = self.q_indexer_chunk_size
        num_chunks = math.ceil(key_len / chunk_size)
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
        chunk_lengths = torch.full(
            (num_chunks,),
            chunk_size,
            device=indexer_key_states.device,
            dtype=chunk_sums.dtype,
        )
        if padded_len != key_len:
            chunk_lengths[-1] = key_len - (num_chunks - 1) * chunk_size
        return chunk_sums / chunk_lengths[:, None, None]

    def _forward_dsa_sequence(
        self,
        *,
        query_states: torch.Tensor,
        indexer_query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        q_len = query_states.shape[0]
        key_len = key_states.shape[0]
        output = query_states.new_empty(q_len, self.num_heads, self.head_dim)
        if q_len == 0 or key_len == 0:
            return output.zero_()

        key_positions = torch.arange(key_len, device=query_states.device)
        query_chunk_size = min(self.q_indexer_query_chunk_size, q_len)
        indexer_scale = self.q_indexer_logit_scale / math.sqrt(self.q_indexer_dim)
        main_scale = 1.0 / math.sqrt(self.head_dim)
        group_size = self.num_heads // self.num_kv_heads
        indexer_key_states = key_states[..., : self.q_indexer_dim]

        for query_start in range(0, q_len, query_chunk_size):
            query_end = min(query_start + query_chunk_size, q_len)
            chunk_len = query_end - query_start
            query_positions = positions[query_start:query_end]
            valid = key_positions[None, :] <= positions[query_start:query_end, None]
            for group_idx in range(self.num_kv_heads):
                indexer_logits = torch.matmul(
                    indexer_query_states[query_start:query_end, group_idx].float(),
                    indexer_key_states[:, group_idx].float().transpose(0, 1),
                )
                indexer_logits.mul_(indexer_scale)
                indexer_logits = indexer_logits.masked_fill(
                    ~valid,
                    torch.finfo(indexer_logits.dtype).min,
                )
                top_k = min(self.q_indexer_top_k, key_len)
                _print_dsa_forward_debug(
                    "good news! vLLM Nemotron-H DSA selective attention ran "
                    f"topk={top_k} requested_topk={self.q_indexer_top_k} "
                    f"layer={self.layer_idx} q_len={q_len} key_len={key_len}"
                )
                topk_indices = indexer_logits.topk(k=top_k, dim=-1).indices
                selected_k = key_states[:, group_idx].index_select(
                    0,
                    topk_indices.reshape(-1),
                )
                selected_v = self._gather_kv_positions_for_head(
                    value_cache,
                    block_table,
                    topk_indices,
                    group_idx,
                )
                selected_k = selected_k.view(
                    query_end - query_start,
                    top_k,
                    self.head_dim,
                )
                selected_v = selected_v.view(
                    query_end - query_start,
                    top_k,
                    self.head_dim,
                )

                head_start = group_idx * group_size
                head_end = head_start + group_size

                min_query_position = int(query_positions.min().item())
                if (
                    self.q_indexer_use_flash_topk
                    and flash_attn_varlen_func is not None
                    and chunk_len == 1
                    and min_query_position + 1 >= top_k
                ):
                    output[query_start:query_end, head_start:head_end] = (
                        self._flash_attn_selected_topk(
                            query_states[
                                query_start:query_end,
                                head_start:head_end,
                            ],
                            selected_k,
                            selected_v,
                            chunk_len,
                            top_k,
                            main_scale,
                        )
                    )
                    continue

                selected_valid = valid.gather(dim=-1, index=topk_indices)
                main_logits = torch.einsum(
                    "qhd,qkd->hqk",
                    query_states[query_start:query_end, head_start:head_end].float(),
                    selected_k.float(),
                )
                main_logits.mul_(main_scale)
                main_logits = main_logits.masked_fill(
                    ~selected_valid[None, :, :],
                    torch.finfo(main_logits.dtype).min,
                )
                attn_weights = F.softmax(
                    main_logits,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query_states.dtype)
                output[query_start:query_end, head_start:head_end] = torch.einsum(
                    "hqk,qkd->qhd",
                    attn_weights,
                    selected_v,
                )
        return output

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
        debug_indexer_proj_ms: float = 0.0,
        debug_key_gather_ms: float = 0.0,
    ) -> torch.Tensor:
        q_len = query_states.shape[0]
        if key_len is None:
            if key_states is None:
                raise ValueError("key_len is required when key_states is omitted")
            key_len = key_states.shape[0]
        output = query_states.new_empty(q_len, self.num_heads, self.head_dim)
        if q_len == 0 or key_len == 0:
            return output.zero_()

        timing_device = query_states.device
        sequence_timing_start = _dsa_timing_start(timing_device)
        chunk_size = self.q_indexer_chunk_size
        num_chunks = math.ceil(key_len / chunk_size)
        if (
            q_len > 1
            and getattr(self, "q_indexer_use_prefill_page_table_fa", False)
            and int(positions[0].item()) // chunk_size
            < self.q_indexer_chunk_top_k
        ):
            full_output = self._forward_dsa_full_page_table_fa_sequence(
                query_states=query_states,
                key_cache=key_cache,
                value_cache=value_cache,
                block_table=block_table,
                attn_metadata=attn_metadata,
                positions=positions,
                key_len=key_len,
                allow_long_sequence=True,
            )
            if full_output is not None:
                _print_dsa_page_table_fa_debug(
                    "Nemotron-H DSA boundary prefill full FlashAttention ran "
                    f"layer={self.layer_idx} q_len={q_len} key_len={key_len} "
                    f"first_position={int(positions[0].item())} "
                    f"chunk_size={chunk_size} "
                    f"top_chunks={self.q_indexer_chunk_top_k}"
                )
                return full_output
        query_chunk_size = min(self.q_indexer_chunked_query_chunk_size, q_len)
        indexer_scale = self.q_indexer_logit_scale / math.sqrt(self.q_indexer_dim)
        main_scale = 1.0 / math.sqrt(self.head_dim)
        group_size = self.num_heads // self.num_kv_heads
        timing_start = _dsa_timing_start(timing_device)
        cache_info = self._dsa_kv_cache_layout_and_block_size(key_cache)
        can_use_summary_cache = (
            key_states is None
            and getattr(self, "q_indexer_use_summary_cache", True)
            and cache_info is not None
            and cache_info[1] == chunk_size
            and block_table.dim() == 1
            and num_chunks <= int(block_table.shape[0])
        )
        if key_states is None and not can_use_summary_cache:
            key_states = self._gather_kv_sequence(
                key_cache,
                block_table,
                key_len,
            )
        chunk_representatives = self._get_indexer_chunk_representatives(
            key_states=key_states,
            key_cache=key_cache,
            block_table=block_table,
            key_len=key_len,
        )
        summary_ms = _dsa_timing_elapsed(timing_start, timing_device)
        chunk_offsets = torch.arange(
            chunk_size,
            device=query_states.device,
            dtype=torch.long,
        )
        score_topk_ms = 0.0
        prefill_page_table_ms = 0.0
        decode_page_table_ms = 0.0
        manual_materialize_ms = 0.0
        manual_attention_ms = 0.0
        prefill_page_table_calls = 0
        decode_page_table_calls = 0
        manual_calls = 0
        max_chunk_top_k = 0
        max_recall_len = 0

        for query_start in range(0, q_len, query_chunk_size):
            query_end = min(query_start + query_chunk_size, q_len)
            chunk_len = query_end - query_start
            query_positions = positions[query_start:query_end].to(torch.long)
            current_chunks = torch.div(
                query_positions,
                chunk_size,
                rounding_mode="floor",
            ).clamp_(min=0, max=num_chunks - 1)
            # The current chunk's full representative may contain future keys
            # during prefill, so score only prior chunks and append this causal
            # local tail separately.
            current_chunk_starts = current_chunks * chunk_size
            tail_indices = current_chunk_starts[:, None] + chunk_offsets[None, :]
            tail_valid = (tail_indices <= query_positions[:, None]) & (
                tail_indices < key_len
            )
            share_chunk_topk = (
                getattr(self, "q_indexer_share_chunk_topk", False)
                and chunk_len > 1
                and self.q_indexer_share_topk_group_size > 1
            )
            if share_chunk_topk:
                share_topk_mode = self.q_indexer_share_topk_mode
                shared_run_starts, shared_run_counts = (
                    self._dsa_shared_topk_run_starts(
                        current_chunks=current_chunks,
                        query_positions=query_positions,
                        chunk_size=chunk_size,
                    )
                )
                # Use a causal representative for the whole DSA chunk.  The
                # previous token is usually a better local-context proxy than
                # the first token, but at the current scheduler batch boundary
                # we only have the current row available.
                shared_source_rows = (shared_run_starts - 1).clamp_min(0)
                shared_target_chunks = current_chunks.index_select(
                    0,
                    shared_run_starts,
                )
            else:
                share_topk_mode = "none"
            max_prior_chunks = int(current_chunks.max().item())
            chunk_ids = torch.arange(
                max_prior_chunks,
                device=query_states.device,
                dtype=torch.long,
            )

            for group_idx in range(self.num_kv_heads):
                timing_start = _dsa_timing_start(timing_device)
                if max_prior_chunks > 0:
                    score_query_states = indexer_query_states[
                        query_start:query_end,
                        group_idx,
                    ]
                    score_current_chunks = current_chunks
                    if share_chunk_topk and share_topk_mode == "representative":
                        score_query_states = score_query_states.index_select(
                            0,
                            shared_source_rows,
                        )
                        score_current_chunks = shared_target_chunks
                    elif share_chunk_topk and share_topk_mode == "mean":
                        score_query_states = self._dsa_mean_shared_query_states(
                            score_query_states=score_query_states,
                            shared_run_starts=shared_run_starts,
                            shared_run_counts=shared_run_counts,
                        )
                        score_current_chunks = shared_target_chunks
                    elif share_chunk_topk and share_topk_mode == "causal_mean":
                        score_query_states = (
                            self._dsa_causal_mean_shared_query_states(
                                score_query_states=score_query_states,
                                current_chunks=current_chunks,
                                shared_run_starts=shared_run_starts,
                            )
                        )
                        score_current_chunks = shared_target_chunks
                    chunk_logits = torch.matmul(
                        score_query_states.float(),
                        chunk_representatives[
                            :max_prior_chunks, group_idx
                        ].transpose(0, 1),
                    )
                    chunk_logits.mul_(indexer_scale)
                    chunk_valid = chunk_ids[None, :] < score_current_chunks[:, None]
                    chunk_logits = chunk_logits.masked_fill(
                        ~chunk_valid,
                        torch.finfo(chunk_logits.dtype).min,
                    )
                    chunk_top_k = min(
                        self.q_indexer_chunk_top_k,
                        max_prior_chunks,
                    )
                    top_chunk_indices = chunk_logits.topk(
                        k=chunk_top_k,
                        dim=-1,
                    ).indices
                    top_chunk_valid = chunk_valid.gather(
                        dim=-1,
                        index=top_chunk_indices,
                    )
                    if share_chunk_topk and share_topk_mode in {
                        "histogram",
                        "strict_union_sum",
                        "union",
                    }:
                        top_chunk_indices, top_chunk_valid = (
                            self._dsa_aggregate_shared_topk(
                                top_chunk_indices=top_chunk_indices,
                                top_chunk_valid=top_chunk_valid,
                                chunk_logits=chunk_logits,
                                shared_run_starts=shared_run_starts,
                                shared_run_counts=shared_run_counts,
                                chunk_top_k=chunk_top_k,
                            )
                        )
                        chunk_top_k = int(top_chunk_indices.shape[1])
                        score_current_chunks = shared_target_chunks
                    if (
                        os.environ.get("VLLM_NEMOTRON_H_DSA_TOPK_STATS", "0")
                        == "1"
                    ):
                        runs = 0
                        same_adjacent = 0
                        adjacent = 0
                        union_total = 0
                        exact_reuse_tokens = 0
                        row = 0
                        top_chunk_indices_cpu = top_chunk_indices.detach().cpu()
                        top_chunk_valid_cpu = top_chunk_valid.detach().cpu()
                        current_chunks_cpu = score_current_chunks.detach().cpu()
                        while row < top_chunk_indices_cpu.shape[0]:
                            end = row + 1
                            while (
                                end < top_chunk_indices_cpu.shape[0]
                                and int(current_chunks_cpu[end].item())
                                == int(current_chunks_cpu[row].item())
                            ):
                                end += 1
                            run_indices = top_chunk_indices_cpu[row:end]
                            run_valid = top_chunk_valid_cpu[row:end]
                            valid_indices = run_indices[run_valid]
                            union_size = int(torch.unique(valid_indices).numel())
                            run_rows = end - row
                            union_total += union_size
                            exact_reuse_tokens += int(run_valid.sum().item())
                            runs += 1
                            if run_rows > 1:
                                equal_adjacent = (
                                    run_indices[1:] == run_indices[:-1]
                                ).all(dim=-1)
                                valid_equal = (
                                    run_valid[1:] == run_valid[:-1]
                                ).all(dim=-1)
                                same_adjacent += int(
                                    (equal_adjacent & valid_equal).sum().item()
                                )
                                adjacent += run_rows - 1
                            row = end
                        avg_union = union_total / max(runs, 1)
                        avg_recall = exact_reuse_tokens / max(runs, 1)
                        reuse = avg_recall / max(avg_union, 1)
                        same_frac = same_adjacent / max(adjacent, 1)
                        _print_dsa_topk_stats(
                            f"layer={self.layer_idx} kv_group={group_idx} "
                            f"q_rows={chunk_len} key_len={key_len} "
                            f"runs={runs} top_k={chunk_top_k} "
                            f"avg_union={avg_union:.1f} "
                            f"avg_row_chunks_per_run={avg_recall:.1f} "
                            f"reuse_factor={reuse:.2f} "
                            f"same_adjacent_frac={same_frac:.4f}"
                        )
                    if (
                        share_chunk_topk
                        and not getattr(
                            self,
                            "q_indexer_use_shared_prefill_page_table_fa",
                            False,
                        )
                    ):
                        top_chunk_indices = torch.repeat_interleave(
                            top_chunk_indices,
                            shared_run_counts,
                            dim=0,
                        )
                        top_chunk_valid = torch.repeat_interleave(
                            top_chunk_valid,
                            shared_run_counts,
                            dim=0,
                        )
                else:
                    chunk_top_k = 0
                    top_chunk_indices = chunk_ids.new_empty(chunk_len, 0)
                    top_chunk_valid = torch.empty(
                        chunk_len,
                        0,
                        device=query_states.device,
                        dtype=torch.bool,
                    )
                score_topk_ms += _dsa_timing_elapsed(timing_start, timing_device)
                max_chunk_top_k = max(max_chunk_top_k, chunk_top_k)

                head_start = group_idx * group_size
                head_end = head_start + group_size
                group_query_states = query_states[
                    query_start:query_end,
                    head_start:head_end,
                ]
                if chunk_len > 1:
                    timing_start = _dsa_timing_start(timing_device)
                    if (
                        getattr(
                            self,
                            "q_indexer_use_union_superset_prefill_page_table_fa",
                            False,
                        )
                        and not share_chunk_topk
                    ):
                        page_table_output = (
                            self._forward_dsa_chunked_union_superset_page_table_fa_prefill(
                                query_states=group_query_states,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                block_table=block_table,
                                attn_metadata=attn_metadata,
                                top_chunk_indices=top_chunk_indices,
                                top_chunk_valid=top_chunk_valid,
                                current_chunks=current_chunks,
                                query_positions=query_positions,
                                key_len=key_len,
                                group_idx=group_idx,
                                softmax_scale=main_scale,
                            )
                        )
                        if page_table_output is None:
                            page_table_output = (
                                self._forward_dsa_chunked_page_table_fa_prefill(
                                    query_states=group_query_states,
                                    key_cache=key_cache,
                                    value_cache=value_cache,
                                    block_table=block_table,
                                    attn_metadata=attn_metadata,
                                    top_chunk_indices=top_chunk_indices,
                                    top_chunk_valid=top_chunk_valid,
                                    current_chunks=current_chunks,
                                    query_positions=query_positions,
                                    key_len=key_len,
                                    group_idx=group_idx,
                                    softmax_scale=main_scale,
                                )
                            )
                    elif (
                        getattr(
                            self,
                            "q_indexer_use_union_prefill_kernel",
                            False,
                        )
                        and not share_chunk_topk
                    ):
                        page_table_output = (
                            self._forward_dsa_chunked_union_kernel_prefill(
                                query_states=group_query_states,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                block_table=block_table,
                                attn_metadata=attn_metadata,
                                top_chunk_indices=top_chunk_indices,
                                top_chunk_valid=top_chunk_valid,
                                current_chunks=current_chunks,
                                query_positions=query_positions,
                                key_len=key_len,
                                group_idx=group_idx,
                                softmax_scale=main_scale,
                            )
                        )
                        if page_table_output is None:
                            page_table_output = (
                                self._forward_dsa_chunked_page_table_fa_prefill(
                                    query_states=group_query_states,
                                    key_cache=key_cache,
                                    value_cache=value_cache,
                                    block_table=block_table,
                                    attn_metadata=attn_metadata,
                                    top_chunk_indices=top_chunk_indices,
                                    top_chunk_valid=top_chunk_valid,
                                    current_chunks=current_chunks,
                                    query_positions=query_positions,
                                    key_len=key_len,
                                    group_idx=group_idx,
                                    softmax_scale=main_scale,
                                )
                            )
                    elif (
                        share_chunk_topk
                        and getattr(
                            self,
                            "q_indexer_use_shared_prefill_page_table_fa",
                            False,
                        )
                    ):
                        page_table_output = (
                            self._forward_dsa_chunked_shared_page_table_fa_prefill(
                                query_states=group_query_states,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                block_table=block_table,
                                attn_metadata=attn_metadata,
                                top_chunk_indices=top_chunk_indices,
                                top_chunk_valid=top_chunk_valid,
                                current_chunks=current_chunks,
                                query_positions=query_positions,
                                key_len=key_len,
                                group_idx=group_idx,
                                softmax_scale=main_scale,
                            )
                        )
                    else:
                        page_table_output = (
                            self._forward_dsa_chunked_page_table_fa_prefill(
                                query_states=group_query_states,
                                key_cache=key_cache,
                                value_cache=value_cache,
                                block_table=block_table,
                                attn_metadata=attn_metadata,
                                top_chunk_indices=top_chunk_indices,
                                top_chunk_valid=top_chunk_valid,
                                current_chunks=current_chunks,
                                query_positions=query_positions,
                                key_len=key_len,
                                group_idx=group_idx,
                                softmax_scale=main_scale,
                            )
                        )
                    prefill_page_table_ms += _dsa_timing_elapsed(
                        timing_start, timing_device
                    )
                    if page_table_output is not None:
                        prefill_page_table_calls += 1
                        output[query_start:query_end, head_start:head_end] = (
                            page_table_output
                        )
                        continue
                    if top_chunk_indices.shape[0] != chunk_len:
                        top_chunk_indices = torch.repeat_interleave(
                            top_chunk_indices,
                            shared_run_counts,
                            dim=0,
                        )
                        top_chunk_valid = torch.repeat_interleave(
                            top_chunk_valid,
                            shared_run_counts,
                            dim=0,
                        )
                else:
                    timing_start = _dsa_timing_start(timing_device)
                    page_table_output = self._forward_dsa_chunked_page_table_fa_decode(
                        query_states=group_query_states,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        block_table=block_table,
                        attn_metadata=attn_metadata,
                        top_chunk_indices=top_chunk_indices,
                        top_chunk_valid=top_chunk_valid,
                        current_chunks=current_chunks,
                        query_positions=query_positions,
                        key_len=key_len,
                        group_idx=group_idx,
                        softmax_scale=main_scale,
                    )
                    decode_page_table_ms += _dsa_timing_elapsed(
                        timing_start, timing_device
                    )
                    if page_table_output is not None:
                        decode_page_table_calls += 1
                        output[query_start:query_end, head_start:head_end] = (
                            page_table_output
                        )
                        continue

                _print_dsa_forward_debug(
                    "good news! vLLM Nemotron-H DSA chunked selective "
                    "attention ran "
                    f"top_chunks={chunk_top_k} "
                    f"requested_top_chunks={self.q_indexer_chunk_top_k} "
                    f"chunk_size={chunk_size} layer={self.layer_idx} "
                    f"q_len={q_len} key_len={key_len}"
                )

                timing_start = _dsa_timing_start(timing_device)
                if key_states is None:
                    key_states = self._gather_kv_sequence(
                        key_cache,
                        block_table,
                        key_len,
                    )
                if chunk_top_k > 0:
                    chunk_token_indices = (
                        top_chunk_indices[..., None] * chunk_size
                        + chunk_offsets[None, None, :]
                    )
                    chunk_token_valid = top_chunk_valid[..., None] & (
                        chunk_token_indices < key_len
                    )
                    chunk_token_indices = chunk_token_indices.reshape(
                        chunk_len,
                        chunk_top_k * chunk_size,
                    )
                    chunk_token_valid = chunk_token_valid.reshape(
                        chunk_len,
                        chunk_top_k * chunk_size,
                    )
                    recall_indices = torch.cat(
                        (chunk_token_indices, tail_indices),
                        dim=-1,
                    )
                    recall_valid = torch.cat((chunk_token_valid, tail_valid), dim=-1)
                else:
                    recall_indices = tail_indices
                    recall_valid = tail_valid

                safe_recall_indices = recall_indices.masked_fill(~recall_valid, 0)
                selected_k = key_states[:, group_idx].index_select(
                    0,
                    safe_recall_indices.reshape(-1),
                )
                selected_v = self._gather_kv_positions_for_head(
                    value_cache,
                    block_table,
                    safe_recall_indices,
                    group_idx,
                )
                recall_len = recall_indices.shape[-1]
                selected_k = selected_k.view(chunk_len, recall_len, self.head_dim)
                selected_v = selected_v.view(chunk_len, recall_len, self.head_dim)
                manual_materialize_ms += _dsa_timing_elapsed(
                    timing_start, timing_device
                )
                manual_calls += 1
                max_recall_len = max(max_recall_len, recall_len)

                timing_start = _dsa_timing_start(timing_device)
                main_logits = torch.einsum(
                    "qhd,qkd->hqk",
                    group_query_states.float(),
                    selected_k.float(),
                )
                main_logits.mul_(main_scale)
                main_logits = main_logits.masked_fill(
                    ~recall_valid[None, :, :],
                    torch.finfo(main_logits.dtype).min,
                )
                attn_weights = F.softmax(
                    main_logits,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query_states.dtype)
                output[query_start:query_end, head_start:head_end] = torch.einsum(
                    "hqk,qkd->qhd",
                    attn_weights,
                    selected_v,
                )
                manual_attention_ms += _dsa_timing_elapsed(
                    timing_start, timing_device
                )
        total_ms = _dsa_timing_elapsed(sequence_timing_start, timing_device)
        mode = "decode" if q_len == 1 else "prefill"
        _print_dsa_timing_debug(
            f"path=chunked_sequence mode={mode} layer={self.layer_idx} "
            f"q_len={q_len} key_len={key_len} num_chunks={num_chunks} "
            f"query_chunk_size={query_chunk_size} groups={self.num_kv_heads} "
            f"share_topk_group_size={self.q_indexer_share_topk_group_size} "
            f"share_topk_mode={self.q_indexer_share_topk_mode} "
            f"indexer_proj_ms={debug_indexer_proj_ms:.3f} "
            f"key_gather_ms={debug_key_gather_ms:.3f} "
            f"summary_ms={summary_ms:.3f} "
            f"score_topk_ms={score_topk_ms:.3f} "
            f"prefill_page_table_ms={prefill_page_table_ms:.3f} "
            f"prefill_page_table_calls={prefill_page_table_calls} "
            f"decode_page_table_ms={decode_page_table_ms:.3f} "
            f"decode_page_table_calls={decode_page_table_calls} "
            f"manual_materialize_ms={manual_materialize_ms:.3f} "
            f"manual_attention_ms={manual_attention_ms:.3f} "
            f"manual_calls={manual_calls} max_chunk_top_k={max_chunk_top_k} "
            f"max_recall_len={max_recall_len} total_ms={total_ms:.3f}"
        )
        return output

    def _forward_dsa_chunked_shared_page_table_fa_prefill(
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
        group_idx: int,
        softmax_scale: float,
    ) -> torch.Tensor | None:
        chunk_size = self.q_indexer_chunk_size
        device = query_states.device
        chunk_len = query_states.shape[0]

        run_starts, run_lens = self._dsa_shared_topk_run_starts(
            current_chunks=current_chunks,
            query_positions=query_positions,
            chunk_size=chunk_size,
        )
        run_ends = run_starts + run_lens
        run_top_rows = run_starts
        run_end_rows = run_ends - 1

        reason = self._dsa_page_table_fa_prefill_fallback_reason(
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn_metadata=attn_metadata,
            top_chunk_indices=top_chunk_indices,
            top_chunk_valid=top_chunk_valid,
            current_chunks=current_chunks,
            query_positions=query_positions,
            key_len=key_len,
            top_table_rows=run_starts.shape[0],
            allow_invalid_top_chunks=True,
        )
        if reason is not None:
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA shared prefill page-table FlashAttention "
                f"fallback: {reason} layer={self.layer_idx}"
            )
            return None

        assert flash_attn_varlen_func is not None
        if block_table.device != device:
            block_table = block_table.to(device=device)

        table_timing_start = _dsa_timing_start(device)
        requested_chunk_top_k = top_chunk_indices.shape[1]
        if top_chunk_valid.numel() > 0:
            valid_top_counts = top_chunk_valid.to(torch.int32).sum(dim=1)
            chunk_top_k = int(valid_top_counts.max().item())
        else:
            chunk_top_k = 0

        if requested_chunk_top_k > 0 and chunk_top_k > 0:
            if top_chunk_indices.shape[0] == run_starts.shape[0]:
                run_top_chunks = top_chunk_indices
                run_top_valid = top_chunk_valid
            else:
                run_top_chunks = top_chunk_indices.index_select(0, run_top_rows)
                run_top_valid = top_chunk_valid.index_select(0, run_top_rows)
            safe_run_top_chunks = run_top_chunks.masked_fill(~run_top_valid, 0)
            if not bool(run_top_valid.all().item()):
                compact_order = torch.argsort(
                    (~run_top_valid).to(torch.int64),
                    dim=1,
                    stable=True,
                )
                safe_run_top_chunks = safe_run_top_chunks.gather(
                    dim=1,
                    index=compact_order,
                )
                run_top_valid = run_top_valid.gather(
                    dim=1,
                    index=compact_order,
                )
            run_valid_counts = run_top_valid.to(torch.int32).sum(dim=1)
            compact_top_chunks = safe_run_top_chunks[:, :chunk_top_k]
            recalled_blocks = block_table.index_select(
                0, compact_top_chunks.reshape(-1).to(torch.long)
            ).view(compact_top_chunks.shape[0], chunk_top_k)
        else:
            recalled_blocks = block_table.new_empty((run_starts.shape[0], 0))
            run_valid_counts = torch.zeros(
                run_starts.shape[0],
                device=device,
                dtype=torch.int32,
            )

        run_current_chunks = current_chunks.index_select(0, run_end_rows)
        current_blocks = block_table.index_select(
            0, run_current_chunks.to(torch.long)
        ).view(run_starts.shape[0], 1)
        run_block_table = torch.cat((recalled_blocks, current_blocks), dim=1)
        if chunk_top_k > 0 and not bool(run_valid_counts.eq(chunk_top_k).all().item()):
            run_block_table.scatter_(
                dim=1,
                index=run_valid_counts.to(torch.long).view(-1, 1),
                src=current_blocks,
            )
        cu_seqlens_q = torch.cat(
            (
                torch.zeros(1, device=device, dtype=torch.int32),
                run_lens.to(torch.int32).cumsum(0, dtype=torch.int32),
            )
        )
        current_chunk_starts = run_current_chunks.to(torch.long) * chunk_size
        run_end_positions = query_positions.index_select(0, run_end_rows)
        tail_lens = run_end_positions.to(torch.long) - current_chunk_starts + 1
        seqused_k = (
            run_valid_counts.to(torch.long) * chunk_size + tail_lens
        ).to(torch.int32)
        max_seqlen_q = int(run_lens.max().item())
        max_seqlen_k = int(seqused_k.max().item())
        table_build_ms = _dsa_timing_elapsed(table_timing_start, device)

        group_key_cache = key_cache[:, :, group_idx : group_idx + 1, :]
        group_value_cache = value_cache[:, :, group_idx : group_idx + 1, :]
        output = torch.empty_like(query_states)
        impl = getattr(self.attn, "impl", None)
        fa_version = getattr(impl, "vllm_flash_attn_version", None)

        flash_attn_kwargs: dict[str, typing.Any] = {}
        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        fa_timing_start = _dsa_timing_start(device)
        flash_attn_varlen_func(
            q=query_states.contiguous(),
            k=group_key_cache,
            v=group_value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
            block_table=run_block_table,
            **flash_attn_kwargs,
        )
        fa_ms = _dsa_timing_elapsed(fa_timing_start, device)
        _print_dsa_timing_debug(
            "path=shared_prefill_page_table "
            f"layer={self.layer_idx} kv_group={group_idx} q_rows={chunk_len} "
            f"runs={int(run_starts.numel())} key_len={key_len} "
            f"top_pages={chunk_top_k} max_seqlen_q={max_seqlen_q} "
            f"max_seqlen_k={max_seqlen_k} "
            f"table_build_ms={table_build_ms:.3f} fa_ms={fa_ms:.3f}"
        )
        return output

    def _forward_dsa_chunked_union_superset_page_table_fa_prefill(
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
        group_idx: int,
        softmax_scale: float,
    ) -> torch.Tensor | None:
        if not getattr(
            self,
            "q_indexer_use_union_superset_prefill_page_table_fa",
            False,
        ):
            return None

        chunk_size = self.q_indexer_chunk_size
        device = query_states.device
        chunk_len = query_states.shape[0]
        union_rows = chunk_size
        if chunk_len % union_rows != 0:
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA union-superset prefill page-table "
                "FlashAttention fallback: "
                f"query rows must be divisible by chunk size, rows={chunk_len} "
                f"chunk_size={chunk_size} layer={self.layer_idx}"
            )
            return None

        reason = self._dsa_page_table_fa_prefill_fallback_reason(
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn_metadata=attn_metadata,
            top_chunk_indices=top_chunk_indices,
            top_chunk_valid=top_chunk_valid,
            current_chunks=current_chunks,
            query_positions=query_positions,
            key_len=key_len,
        )
        if reason is not None:
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA union-superset prefill page-table "
                f"FlashAttention fallback: {reason} layer={self.layer_idx}"
            )
            return None

        assert flash_attn_varlen_func is not None
        if block_table.device != device:
            block_table = block_table.to(device=device)

        table_timing_start = _dsa_timing_start(device)
        groups = chunk_len // union_rows
        top_k = top_chunk_indices.shape[1]
        flat_chunks = top_chunk_indices.to(torch.int32).view(
            groups,
            union_rows * top_k,
        )
        sorted_chunks, _ = flat_chunks.sort(dim=1)
        unique = torch.ones_like(sorted_chunks, dtype=torch.bool)
        unique[:, 1:] = sorted_chunks[:, 1:] != sorted_chunks[:, :-1]
        unique_rank_sorted = unique.cumsum(dim=1, dtype=torch.int32) - 1
        union_counts = unique.sum(dim=1, dtype=torch.int32)
        max_union = int(union_counts.max().item())

        union_chunks = torch.zeros(
            groups,
            max_union,
            device=device,
            dtype=torch.int32,
        )
        rank_clamped = unique_rank_sorted.clamp_max(max_union - 1).to(torch.long)
        union_chunks.scatter_(1, rank_clamped, sorted_chunks)

        recalled_blocks = block_table.index_select(
            0,
            union_chunks.reshape(-1).to(torch.long),
        ).view(groups, max_union)
        grouped_current = current_chunks.to(torch.int32).view(groups, union_rows)
        run_current_chunks = grouped_current[:, 0].contiguous()
        current_blocks = block_table.index_select(
            0,
            run_current_chunks.to(torch.long),
        ).view(groups, 1)
        run_block_table = torch.cat((recalled_blocks, current_blocks), dim=1)

        cu_seqlens_q = torch.arange(
            groups + 1,
            device=device,
            dtype=torch.int32,
        ) * union_rows
        grouped_positions = query_positions.to(torch.int32).view(groups, union_rows)
        run_end_positions = grouped_positions[:, -1]
        tail_lens = run_end_positions - run_current_chunks * chunk_size + 1
        seqused_k = (union_counts * chunk_size + tail_lens).to(torch.int32)
        max_seqlen_k = int(seqused_k.max().item())
        table_build_ms = _dsa_timing_elapsed(table_timing_start, device)

        group_key_cache = key_cache[:, :, group_idx : group_idx + 1, :]
        group_value_cache = value_cache[:, :, group_idx : group_idx + 1, :]
        output = torch.empty_like(query_states)
        impl = getattr(self.attn, "impl", None)
        fa_version = getattr(impl, "vllm_flash_attn_version", None)

        flash_attn_kwargs: dict[str, typing.Any] = {}
        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        fa_timing_start = _dsa_timing_start(device)
        flash_attn_varlen_func(
            q=query_states.contiguous(),
            k=group_key_cache,
            v=group_value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=union_rows,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=True,
            block_table=run_block_table,
            **flash_attn_kwargs,
        )
        fa_ms = _dsa_timing_elapsed(fa_timing_start, device)
        _print_dsa_timing_debug(
            "path=union_superset_prefill_page_table "
            f"layer={self.layer_idx} kv_group={group_idx} q_rows={chunk_len} "
            f"runs={groups} key_len={key_len} top_pages={top_k} "
            f"max_union_pages={max_union} max_seqlen_q={union_rows} "
            f"max_seqlen_k={max_seqlen_k} "
            f"table_build_ms={table_build_ms:.3f} fa_ms={fa_ms:.3f}"
        )
        return output

    def _forward_dsa_chunked_union_kernel_prefill(
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
        group_idx: int,
        softmax_scale: float,
    ) -> torch.Tensor | None:
        if not getattr(self, "q_indexer_use_union_prefill_kernel", False):
            return None
        if dsa_prefill_gqa_union_attention is None:
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA union prefill kernel fallback: "
                f"kernel import unavailable layer={self.layer_idx}"
            )
            return None

        chunk_size = self.q_indexer_chunk_size
        chunk_len = query_states.shape[0]
        union_rows = chunk_size
        if chunk_len % union_rows != 0:
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA union prefill kernel fallback: "
                f"query rows must be divisible by chunk size, rows={chunk_len} "
                f"chunk_size={chunk_size} layer={self.layer_idx}"
            )
            return None

        reason = self._dsa_page_table_fa_prefill_fallback_reason(
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn_metadata=attn_metadata,
            top_chunk_indices=top_chunk_indices,
            top_chunk_valid=top_chunk_valid,
            current_chunks=current_chunks,
            query_positions=query_positions,
            key_len=key_len,
        )
        if reason is not None:
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA union prefill kernel fallback: "
                f"{reason} layer={self.layer_idx}"
            )
            return None

        device = query_states.device
        if block_table.device != device:
            block_table = block_table.to(device=device)

        full_timing_start = _dsa_timing_start(device)
        table_timing_start = _dsa_timing_start(device)
        groups = chunk_len // union_rows
        top_k = top_chunk_indices.shape[1]
        flat_chunks = top_chunk_indices.to(torch.int32).view(
            groups,
            union_rows * top_k,
        )
        sorted_chunks, sorted_pos = flat_chunks.sort(dim=1)
        unique = torch.ones_like(sorted_chunks, dtype=torch.bool)
        unique[:, 1:] = sorted_chunks[:, 1:] != sorted_chunks[:, :-1]
        unique_rank_sorted = unique.cumsum(dim=1, dtype=torch.int32) - 1
        union_counts = unique.sum(dim=1, dtype=torch.int32)
        max_union = int(union_counts.max().item())

        union_chunks = torch.empty(
            groups,
            max_union,
            device=device,
            dtype=torch.int32,
        )
        rank_clamped = unique_rank_sorted.clamp_max(max_union - 1).to(torch.long)
        union_chunks.scatter_(1, rank_clamped, sorted_chunks)

        rank_for_flat = torch.empty_like(unique_rank_sorted)
        rank_for_flat.scatter_(1, sorted_pos, unique_rank_sorted)
        row_ids = torch.arange(
            union_rows,
            device=device,
            dtype=torch.int32,
        ).repeat_interleave(top_k)
        row_bits = (1 << row_ids).expand(groups, -1)
        union_masks = torch.zeros(
            groups,
            max_union,
            device=device,
            dtype=torch.int32,
        )
        union_masks.scatter_reduce_(
            1,
            rank_for_flat.to(torch.long),
            row_bits,
            reduce="sum",
            include_self=False,
        )

        row_starts = (
            torch.arange(groups, device=device, dtype=torch.int32) * union_rows
        )
        row_counts = torch.full(
            (groups,),
            union_rows,
            device=device,
            dtype=torch.int32,
        )
        grouped_current = current_chunks.to(torch.int32).view(groups, union_rows)
        run_current_chunks = grouped_current[:, 0].contiguous()
        grouped_positions = query_positions.to(torch.int32).view(groups, union_rows)
        tail_lens = (
            grouped_positions - run_current_chunks[:, None] * chunk_size + 1
        ).contiguous()
        table_build_ms = _dsa_timing_elapsed(table_timing_start, device)

        kernel_timing_start = _dsa_timing_start(device)
        output = dsa_prefill_gqa_union_attention(
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            union_chunks=union_chunks,
            union_masks=union_masks,
            union_counts=union_counts.contiguous(),
            row_starts=row_starts,
            row_counts=row_counts,
            current_chunks=run_current_chunks,
            tail_lens=tail_lens,
            group_idx=group_idx,
            softmax_scale=softmax_scale,
            chunks_per_iter=self.q_indexer_union_chunks_per_iter,
        )
        kernel_ms = _dsa_timing_elapsed(kernel_timing_start, device)
        total_ms = _dsa_timing_elapsed(full_timing_start, device)
        _print_dsa_timing_debug(
            "path=union_prefill_kernel "
            f"layer={self.layer_idx} kv_group={group_idx} q_rows={chunk_len} "
            f"runs={groups} key_len={key_len} top_pages={top_k} "
            f"max_union_pages={max_union} chunks_per_iter="
            f"{self.q_indexer_union_chunks_per_iter} "
            f"table_build_ms={table_build_ms:.3f} "
            f"kernel_ms={kernel_ms:.3f} total_ms={total_ms:.3f}"
        )
        return output

    def _forward_dsa_chunked_page_table_fa_prefill(
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
        group_idx: int,
        softmax_scale: float,
    ) -> torch.Tensor | None:
        if not getattr(self, "q_indexer_use_prefill_page_table_fa", False):
            return None

        reason = self._dsa_page_table_fa_prefill_fallback_reason(
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn_metadata=attn_metadata,
            top_chunk_indices=top_chunk_indices,
            top_chunk_valid=top_chunk_valid,
            current_chunks=current_chunks,
            query_positions=query_positions,
            key_len=key_len,
        )
        if reason is not None:
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA prefill page-table FlashAttention fallback: "
                f"{reason} layer={self.layer_idx}"
            )
            return None

        assert flash_attn_varlen_func is not None
        chunk_size = self.q_indexer_chunk_size
        device = query_states.device
        chunk_len = query_states.shape[0]
        if block_table.device != device:
            block_table = block_table.to(device=device)

        table_timing_start = _dsa_timing_start(device)
        chunk_top_k = top_chunk_indices.shape[1]
        if chunk_top_k > 0:
            recalled_blocks = block_table.index_select(
                0, top_chunk_indices.reshape(-1).to(torch.long)
            ).view(chunk_len, chunk_top_k)
        else:
            recalled_blocks = block_table.new_empty((chunk_len, 0))
        current_blocks = block_table.index_select(
            0, current_chunks.to(torch.long)
        ).view(chunk_len, 1)
        temp_block_table = torch.cat((recalled_blocks, current_blocks), dim=1)
        table_build_ms = _dsa_timing_elapsed(table_timing_start, device)

        current_chunk_starts = current_chunks.to(torch.long) * chunk_size
        tail_lens = query_positions.to(torch.long) - current_chunk_starts + 1
        seqused_k = (chunk_top_k * chunk_size + tail_lens).to(torch.int32)
        max_seqlen_k = int(seqused_k.max().item())

        cu_seqlens_q = torch.arange(
            chunk_len + 1,
            device=device,
            dtype=torch.int32,
        )
        group_key_cache = key_cache[:, :, group_idx : group_idx + 1, :]
        group_value_cache = value_cache[:, :, group_idx : group_idx + 1, :]
        output = torch.empty_like(query_states)
        impl = getattr(self.attn, "impl", None)
        fa_version = getattr(impl, "vllm_flash_attn_version", None)

        flash_attn_kwargs: dict[str, typing.Any] = {}
        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        fa_timing_start = _dsa_timing_start(device)
        flash_attn_varlen_func(
            q=query_states.contiguous(),
            k=group_key_cache,
            v=group_value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            block_table=temp_block_table,
            **flash_attn_kwargs,
        )
        fa_ms = _dsa_timing_elapsed(fa_timing_start, device)
        _print_dsa_timing_debug(
            "path=prefill_page_table "
            f"layer={self.layer_idx} kv_group={group_idx} q_rows={chunk_len} "
            f"key_len={key_len} top_pages={chunk_top_k} "
            f"min_tail_len={int(tail_lens.min().item())} "
            f"max_tail_len={int(tail_lens.max().item())} "
            f"max_seqlen_k={max_seqlen_k} "
            f"table_build_ms={table_build_ms:.3f} fa_ms={fa_ms:.3f}"
        )
        _print_dsa_page_table_fa_debug(
            "good news! vLLM Nemotron-H DSA prefill page-table "
            "FlashAttention ran "
            f"rows={chunk_len} top_pages={chunk_top_k} "
            f"requested_top_pages={self.q_indexer_chunk_top_k} "
            f"chunk_size={chunk_size} layer={self.layer_idx} "
            f"kv_group={group_idx} key_len={key_len} "
            f"max_seqlen_k={max_seqlen_k}"
        )
        return output

    def _forward_dsa_chunked_page_table_fa_decode(
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
        group_idx: int,
        softmax_scale: float,
    ) -> torch.Tensor | None:
        if not getattr(self, "q_indexer_use_page_table_fa", False):
            return None

        reason = self._dsa_page_table_fa_fallback_reason(
            query_states=query_states,
            key_cache=key_cache,
            value_cache=value_cache,
            block_table=block_table,
            attn_metadata=attn_metadata,
            query_positions=query_positions,
            key_len=key_len,
        )
        if reason is not None:
            if not reason.startswith("decode-only guard"):
                _print_dsa_page_table_fa_debug(
                    "Nemotron-H DSA page-table FlashAttention fallback: "
                    f"{reason} layer={self.layer_idx}"
                )
            return None

        assert flash_attn_varlen_func is not None
        chunk_size = self.q_indexer_chunk_size
        device = query_states.device
        decode_timing_start = _dsa_timing_start(device)
        if block_table.device != device:
            block_table = block_table.to(device=device)
        if top_chunk_indices.shape != top_chunk_valid.shape:
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA page-table FlashAttention fallback: top chunk "
                f"shape mismatch layer={self.layer_idx} "
                f"indices={tuple(top_chunk_indices.shape)} "
                f"valid={tuple(top_chunk_valid.shape)}"
            )
            return None
        if top_chunk_indices.shape[0] != 1 or current_chunks.numel() != 1:
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA page-table FlashAttention fallback: "
                f"decode-only chunk guard failed layer={self.layer_idx} "
                f"top_chunk_shape={tuple(top_chunk_indices.shape)} "
                f"current_chunks={int(current_chunks.numel())}"
            )
            return None

        table_timing_start = _dsa_timing_start(device)
        valid_top_chunks = top_chunk_indices[0].masked_select(top_chunk_valid[0])
        page_chunk_ids = torch.cat((valid_top_chunks, current_chunks[:1]))
        if int(page_chunk_ids.max().item()) >= int(block_table.shape[0]):
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA page-table FlashAttention fallback: recalled "
                f"page id exceeds block table length layer={self.layer_idx} "
                f"max_page_id={int(page_chunk_ids.max().item())} "
                f"block_table_len={int(block_table.shape[0])}"
            )
            return None
        temp_block_table = block_table.index_select(0, page_chunk_ids).view(1, -1)
        table_build_ms = _dsa_timing_elapsed(table_timing_start, device)

        current_chunk_start = int(current_chunks[0].item()) * chunk_size
        tail_len = int(query_positions[0].item()) - current_chunk_start + 1
        if tail_len <= 0 or tail_len > chunk_size:
            _print_dsa_page_table_fa_debug(
                "Nemotron-H DSA page-table FlashAttention fallback: invalid "
                f"decode tail length layer={self.layer_idx} tail_len={tail_len} "
                f"chunk_size={chunk_size}"
            )
            return None
        recalled_tokens = int(valid_top_chunks.numel()) * chunk_size + tail_len

        cu_seqlens_q = torch.tensor([0, 1], device=device, dtype=torch.int32)
        seqused_k = torch.tensor([recalled_tokens], device=device, dtype=torch.int32)
        group_key_cache = key_cache[:, :, group_idx : group_idx + 1, :]
        group_value_cache = value_cache[:, :, group_idx : group_idx + 1, :]
        output = torch.empty_like(query_states)
        impl = getattr(self.attn, "impl", None)
        fa_version = getattr(impl, "vllm_flash_attn_version", None)

        flash_attn_kwargs: dict[str, typing.Any] = {}
        if fa_version is not None:
            flash_attn_kwargs["fa_version"] = fa_version

        fa_timing_start = _dsa_timing_start(device)
        flash_attn_varlen_func(
            q=query_states.contiguous(),
            k=group_key_cache,
            v=group_value_cache,
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seqused_k,
            max_seqlen_k=recalled_tokens,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
            block_table=temp_block_table,
            **flash_attn_kwargs,
        )
        fa_ms = _dsa_timing_elapsed(fa_timing_start, device)
        total_ms = _dsa_timing_elapsed(decode_timing_start, device)
        _print_dsa_timing_debug(
            f"path=decode_page_table layer={self.layer_idx} kv_group={group_idx} "
            f"key_len={key_len} top_pages={int(valid_top_chunks.numel())} "
            f"tail_len={tail_len} recalled_tokens={recalled_tokens} "
            f"table_build_ms={table_build_ms:.3f} fa_ms={fa_ms:.3f} "
            f"total_ms={total_ms:.3f}"
        )
        _print_dsa_page_table_fa_debug(
            "good news! vLLM Nemotron-H DSA page-table FlashAttention ran "
            f"top_pages={int(valid_top_chunks.numel())} "
            f"requested_top_pages={self.q_indexer_chunk_top_k} "
            f"chunk_size={chunk_size} "
            f"kernel_block_size={key_cache.shape[1]} "
            "cache_config_block_size="
            f"{getattr(self, '_dsa_cache_config_block_size', None)} "
            f"layer={self.layer_idx} kv_group={group_idx} "
            f"key_len={key_len} recalled_tokens={recalled_tokens}"
        )
        return output

    def _dsa_page_table_fa_prefill_fallback_reason(
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
        top_table_rows: int | None = None,
        allow_invalid_top_chunks: bool = False,
    ) -> str | None:
        if flash_attn_varlen_func is None:
            return "flash_attn_varlen_func is unavailable"
        query_len = int(query_states.shape[0])
        if query_len <= 1:
            return "prefill guard requires more than one query token"
        if query_positions.numel() != query_len or current_chunks.numel() != query_len:
            return (
                "prefill row metadata must match query length, "
                f"query_len={query_len} positions={int(query_positions.numel())} "
                f"current_chunks={int(current_chunks.numel())}"
            )
        if int(query_positions.min().item()) < 0 or int(query_positions.max().item()) >= key_len:
            return (
                "prefill query positions must be within key sequence, "
                f"min={int(query_positions.min().item())} "
                f"max={int(query_positions.max().item())} key_len={key_len}"
            )
        if top_chunk_indices.shape != top_chunk_valid.shape:
            return (
                "top chunk shape mismatch, "
                f"indices={tuple(top_chunk_indices.shape)} "
                f"valid={tuple(top_chunk_valid.shape)}"
            )
        expected_top_rows = query_len if top_table_rows is None else top_table_rows
        if (
            top_chunk_indices.dim() != 2
            or top_chunk_indices.shape[0] != expected_top_rows
        ):
            return (
                "prefill page-table FA expects a 2D top chunk table, "
                f"got {tuple(top_chunk_indices.shape)} "
                f"expected_rows={expected_top_rows} query_len={query_len}"
            )
        if (
            not allow_invalid_top_chunks
            and not bool(top_chunk_valid.all().item())
        ):
            return "top chunk table contains invalid entries requiring compaction"
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
            self.num_kv_heads,
            self.head_dim,
        )
        if tuple(key_cache.shape[1:]) != expected_suffix:
            return (
                "paged FA prototype only supports NHD cache shape "
                "(blocks, block_size, kv_heads, head_dim), "
                f"got shape={tuple(key_cache.shape)} "
                f"expected_suffix={expected_suffix}"
            )
        kernel_block_size = int(key_cache.shape[1])
        if kernel_block_size != self.q_indexer_chunk_size:
            return (
                "kernel block size must equal DSA chunk size for parity, "
                f"kernel_block_size={kernel_block_size} "
                f"chunk_size={self.q_indexer_chunk_size}"
            )
        if block_table.dim() != 1:
            return (
                "expected a per-sequence 1D block table, "
                f"got {tuple(block_table.shape)}"
            )
        max_page_id = int(current_chunks.max().item())
        if top_chunk_indices.numel() > 0:
            max_page_id = max(max_page_id, int(top_chunk_indices.max().item()))
        if max_page_id >= int(block_table.shape[0]):
            return (
                "recalled page id exceeds block table length, "
                f"max_page_id={max_page_id} block_table_len={int(block_table.shape[0])}"
            )
        tail_lens = query_positions.to(torch.long) - (
            current_chunks.to(torch.long) * self.q_indexer_chunk_size
        ) + 1
        if (
            int(tail_lens.min().item()) <= 0
            or int(tail_lens.max().item()) > self.q_indexer_chunk_size
        ):
            return (
                "invalid prefill tail length, "
                f"min={int(tail_lens.min().item())} "
                f"max={int(tail_lens.max().item())} "
                f"chunk_size={self.q_indexer_chunk_size}"
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

    def _dsa_page_table_fa_fallback_reason(
        self,
        *,
        query_states: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_table: torch.Tensor,
        attn_metadata: typing.Any | None,
        query_positions: torch.Tensor,
        key_len: int,
    ) -> str | None:
        if flash_attn_varlen_func is None:
            return "flash_attn_varlen_func is unavailable"
        if query_states.shape[0] != 1 or query_positions.numel() != 1:
            return "decode-only guard requires exactly one query token"
        if int(query_positions[0].item()) != key_len - 1:
            return "decode-only guard requires the query position to be the final key"
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
            self.num_kv_heads,
            self.head_dim,
        )
        if tuple(key_cache.shape[1:]) != expected_suffix:
            return (
                "paged FA prototype only supports NHD cache shape "
                "(blocks, block_size, kv_heads, head_dim), "
                f"got shape={tuple(key_cache.shape)} "
                f"expected_suffix={expected_suffix}"
            )
        kernel_block_size = int(key_cache.shape[1])
        if kernel_block_size != self.q_indexer_chunk_size:
            return (
                "kernel block size must equal DSA chunk size for parity, "
                f"kernel_block_size={kernel_block_size} "
                f"chunk_size={self.q_indexer_chunk_size}"
            )
        if block_table.dim() != 1:
            return (
                "expected a per-sequence 1D block table, "
                f"got {tuple(block_table.shape)}"
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

    def _flash_attn_selected_topk(
        self,
        query_states: torch.Tensor,
        selected_k: torch.Tensor,
        selected_v: torch.Tensor,
        query_len: int,
        top_k: int,
        softmax_scale: float,
    ) -> torch.Tensor:
        assert flash_attn_varlen_func is not None
        q = query_states.contiguous()
        k = selected_k.reshape(query_len * top_k, 1, self.head_dim).contiguous()
        v = selected_v.reshape(query_len * top_k, 1, self.head_dim).contiguous()
        cu_seqlens_q = torch.arange(
            query_len + 1,
            device=q.device,
            dtype=torch.int32,
        )
        cu_seqlens_k = torch.arange(
            0,
            (query_len + 1) * top_k,
            top_k,
            device=q.device,
            dtype=torch.int32,
        )
        return flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_k=top_k,
            dropout_p=0.0,
            softmax_scale=softmax_scale,
            causal=False,
        )


class NemotronHAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        # Get per-layer config for heterogeneous models if exists
        get_layer_config = getattr(config, "get_nemotron_h_config_for_layer", None)
        layer_config = get_layer_config(layer_idx) if get_layer_config else config

        attention_cls: type[nn.Module] = (
            _get_dsa_attention_class()
            if getattr(layer_config, "q_indexer_dim", None) is not None
            else NemotronHAttention
        )

        self.mixer = attention_cls(
            layer_config,
            layer_idx,
            model_config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states=hidden_states, positions=positions)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "M": NemotronHMambaDecoderLayer,
    "-": NemotronHMLPDecoderLayer,
    "*": NemotronHAttentionDecoderLayer,
    "E": NemotronHMoEDecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class NemotronHModel(nn.Module, EagleModelMixin):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_types: Mapping[str, type[nn.Module]] | None = None,
    ):
        super().__init__()

        config: NemotronHConfig = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.has_moe = "E" in config.hybrid_override_pattern

        layer_types = decoder_layer_types or ALL_DECODER_LAYER_TYPES

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_type = config.hybrid_override_pattern[layer_idx]
            if layer_type not in layer_types:
                raise ValueError(
                    f"Unsupported layer type {layer_type!r} at layer {layer_idx}"
                )
            layer_class = layer_types[layer_type]
            return layer_class(
                config=config,
                layer_idx=layer_idx,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                parallel_config=parallel_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            len(config.hybrid_override_pattern), get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states

    def is_spec_layer(self, config: NemotronHConfig, weight_name: str) -> bool:
        return weight_name.startswith("mtp.")

    def _get_max_n_routed_experts(self) -> int:
        """Get max n_routed_experts from config or block_configs for puzzle models.

        For heterogeneous models with varying expert counts per layer,
        returns the MAX to ensure all expert weights can be loaded.
        """
        # First try top-level attribute
        n_routed_experts = getattr(self.config, "n_routed_experts", None)
        if n_routed_experts is not None:
            return n_routed_experts

        # For puzzle models, get MAX from all MoE blocks in block_configs
        # (different layers may have different expert counts)
        max_experts = 0
        block_configs = getattr(self.config, "block_configs", None)
        if block_configs:
            for block in block_configs:
                if isinstance(block, dict):
                    if block.get("block_type") == "moe":
                        max_experts = max(max_experts, block.get("n_routed_experts", 0))
                else:
                    # HF converts dicts to objects with attributes
                    if getattr(block, "block_type", "") == "moe":
                        max_experts = max(
                            max_experts, getattr(block, "n_routed_experts", 0)
                        )
        return max_experts

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Consumed by `get_moe_expert_mapping` (bitsandbytes / LoRA); the main
        # weight load is self-served by each `RoutedExperts` layer. Sized to the
        # MAX expert count so heterogeneous puzzle models load every expert.
        if self.has_moe:
            # (param_name, weight_name, expert_id, shard_id)
            return fused_moe_make_expert_params_mapping(
                # - FusedMoe.w1 (aka gate_proj) should be up_proj since that's
                #   what the activation is applied to
                # - FusedMoe.w3 (aka up_proj) should be ignored since we're
                #   using non-gated MoE
                self,
                ckpt_gate_proj_name="up_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="",
                num_experts=self._get_max_n_routed_experts(),
                num_redundant_experts=getattr(self, "num_redundant_experts", 0),
            )

        return []


class NemotronHForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    SupportsQuant,
    MixtureOfExperts,
    SupportsMambaPrefixCaching,
    SupportsReplaySSM,
):
    # Relevant only if self.has_moe is True
    is_non_gated_moe: bool = True

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"backbone": "model"},
        orig_to_new_substr={"A_log": "A", "embeddings": "embed_tokens"},
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
        },
    )

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    # Skip MTP (Multi-Token Prediction) layers during LoRA loading
    lora_skip_prefixes = ["mtp."]

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, ...]:
        cache_config = vllm_config.cache_config
        base_dtype = MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            cache_config.mamba_cache_dtype,
            cache_config.mamba_ssm_cache_dtype,
        )
        if cache_config.use_replayssm:
            return MambaStateDtypeCalculator.append_replayssm_ring(
                base_dtype, vllm_config.model_config.dtype
            )
        return base_dtype

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, ...], ...]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
            - x_cache/dt_cache/B_cache ring-buffer shapes (use_replayssm only)
        """
        parallel_config = vllm_config.parallel_config
        cache_config = vllm_config.cache_config
        hf_config = vllm_config.model_config.hf_config
        intermediate_size = hf_config.mamba_num_heads * hf_config.mamba_head_dim

        base_shape = MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=hf_config.n_groups,
            num_heads=hf_config.mamba_num_heads,
            head_dim=hf_config.mamba_head_dim,
            state_size=hf_config.ssm_state_size,
            conv_kernel=hf_config.conv_kernel,
            num_spec=vllm_config.num_speculative_tokens,
        )
        if cache_config.use_replayssm:
            return MambaStateShapeCalculator.append_replayssm_ring(
                base_shape,
                hf_config.n_groups,
                parallel_config.tensor_parallel_size,
                cache_config.replayssm_buffer_len,
            )
        return base_shape

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.mamba2_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        scheduler_config = vllm_config.scheduler_config

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = NemotronHModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        if self.model.has_moe:
            self.num_expert_groups = config.n_group

            self.moe_layers = []
            example_moe = None
            for layer in self.model.layers:
                if isinstance(layer, NemotronHMoEDecoderLayer):
                    # Pick last one layer since the first ones
                    # may be dense layers.
                    example_moe = layer.mixer
                    self.moe_layers.append(layer.mixer.experts)

            self.num_moe_layers = len(self.moe_layers)
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts  # noqa: E501
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, NemotronHMoEDecoderLayer):
                moe = layer.mixer
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self, skip_prefixes=["mtp"])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
