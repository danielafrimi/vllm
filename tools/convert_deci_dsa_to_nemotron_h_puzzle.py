# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Convert the Deci-style DSA checkpoint to Nemotron-H Puzzle layout.

This rewrites both the checkpoint config and safetensors tensor names from the
DeciLM layout used by the first DSA experiments to the newer
NemotronHPuzzleForCausalLM layout that vLLM routes through nemotron_h.py.

The generated Puzzle config accepts q_indexer_* fields so vLLM can instantiate
selective attention. The bundled HF modeling file is copied from a Puzzle
template checkpoint; q-index execution is implemented in vLLM, not in that HF
fallback model.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path(
    "/lustre/fsw/portfolios/coreai/users/mdabbah/deci/"
    "puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints/"
    "nemo3_repr_heads_128k_top2k_q128_sparse_topk2048"
)
DEFAULT_TEMPLATE = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/users/nnabwani/"
    "soups/task-arithmetic-s1.0"
)

Q_INDEXER_FIELDS = (
    "q_indexer_attn_mode",
    "q_indexer_chunk_size",
    "q_indexer_chunk_top_k",
    "q_indexer_chunked_query_chunk_size",
    "q_indexer_dim",
    "q_indexer_init_query_heads",
    "q_indexer_logit_scale",
    "q_indexer_query_chunk_size",
    "q_indexer_top_k",
)

OPTIONAL_METADATA_FILES = (
    "chat_template.jinja",
    "generation_config.json",
    "metadata.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

DSA_EXTRA_TENSOR_RE = re.compile(
    r"model\.layers\.\d+\.self_attn\.(?:indexer_.+|dsa_.+)"
)

PUZZLE_CONFIG_PY = '''from typing import Any

import dataclasses
from dataclasses import dataclass
from transformers.utils import logging

from .configuration_nemotron_h import NemotronHConfig


logger = logging.get_logger(__name__)


@dataclass
class AttentionConfig:
    sliding_window: int | None = None
    q_indexer_attn_mode: str | None = None
    q_indexer_chunk_size: int | None = None
    q_indexer_chunk_top_k: int | None = None
    q_indexer_chunked_query_chunk_size: int | None = None
    q_indexer_dim: int | None = None
    q_indexer_init_query_heads: list[int] | None = None
    q_indexer_logit_scale: float | None = None
    q_indexer_query_chunk_size: int | None = None
    q_indexer_top_k: int | None = None
    block_type: str = "attention"


@dataclass
class MoeConfig:
    n_routed_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    moe_shared_expert_intermediate_size: int
    moe_latent_size: int | None = None
    routed_scaling_factor: float | None = None
    n_group: int | None = None
    topk_group: int | None = None
    norm_topk_prob: bool | None = None
    block_type: str = "moe"


@dataclass
class MambaConfig:
    ssm_state_size: int | None = None
    block_type: str = "mamba"


@dataclass
class MlpConfig:
    intermediate_size: int
    block_type: str = "mlp"


BLOCK_TYPE_TO_CONFIG_CLASS = {
    config_class.block_type: config_class
    for config_class in (AttentionConfig, MoeConfig, MambaConfig, MlpConfig)
}
BLOCK_TYPE_TO_PATTERN = {"attention": "*", "moe": "E", "mamba": "M", "mlp": "-"}


class NemotronHPuzzleConfig(NemotronHConfig):
    model_type = "nemotron_h_puzzle"

    def __init__(self, **kwargs):
        self.block_configs = []
        if "block_configs" in kwargs:
            self.block_configs, kwargs["hybrid_override_pattern"] = build_block_configs(
                kwargs.pop("block_configs")
            )

        self.mtp_block_configs = []
        if "mtp_block_configs" in kwargs:
            (
                self.mtp_block_configs,
                kwargs["mtp_hybrid_override_pattern"],
            ) = build_block_configs(kwargs.pop("mtp_block_configs"))

        if self.is_sliding:
            logger.warning(
                "NemotronHPuzzleConfig: sliding window attention is enabled, "
                "setting attn_implementation to 'flash_attention_2'"
            )
            kwargs["attn_implementation"] = "flash_attention_2"

        super().__init__(**kwargs)

        self.blockwise_members = list(
            set(
                field.name
                for block_config in self.block_configs + self.mtp_block_configs
                for field in dataclasses.fields(block_config)
                if getattr(block_config, field.name) is not None
            )
        )
        for member in self.blockwise_members:
            if hasattr(self, member):
                delattr(self, member)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["block_configs"] = [
            {
                k: v
                for k, v in dataclasses.asdict(block_config).items()
                if v is not None
            }
            for block_config in self.block_configs
        ]
        output["mtp_block_configs"] = [
            {
                k: v
                for k, v in dataclasses.asdict(block_config).items()
                if v is not None
            }
            for block_config in self.mtp_block_configs
        ]
        return output

    def get_nemotron_h_config_for_layer(self, layer_idx: int) -> NemotronHConfig:
        block_config = dataclasses.asdict(self.block_configs[layer_idx])

        config_dict = self.to_dict()
        del config_dict["block_configs"]
        del config_dict["mtp_block_configs"]

        overlapping_fields = set(config_dict.keys()) & set(block_config.keys())
        additional_fields = set(block_config.keys()) - set(config_dict.keys())
        for field in overlapping_fields:
            if block_config[field] is not None:
                config_dict[field] = block_config[field]

        nemotron_h_config = NemotronHConfig.from_dict(config_dict)
        nemotron_h_config._attn_implementation = self._attn_implementation

        for field in additional_fields:
            if block_config[field] is not None:
                setattr(nemotron_h_config, field, block_config[field])

        return nemotron_h_config

    @property
    def is_sliding(self) -> bool:
        return any(
            block_config.block_type == "attention"
            and block_config.sliding_window is not None
            for block_config in self.block_configs
        )


def build_block_configs(block_config_dicts: list[dict]) -> tuple[list[dataclass], str]:
    block_configs = []
    for block_config in block_config_dicts:
        if isinstance(block_config, dict):
            config_class = BLOCK_TYPE_TO_CONFIG_CLASS[block_config["block_type"]]
            block_config = config_class(**block_config)
        block_configs.append(block_config)

    hybrid_override_pattern = "".join(
        BLOCK_TYPE_TO_PATTERN[block_config.block_type]
        for block_config in block_configs
    )
    return block_configs, hybrid_override_pattern
'''


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def first_present(values: list[Any], default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def get_first_mamba_config(block_configs: list[dict[str, Any]]) -> dict[str, Any]:
    for block in block_configs:
        attention = block.get("attention") or {}
        if not attention.get("no_op", False) and attention.get("mamba") is not None:
            return attention["mamba"]
    raise ValueError("Source config has no Mamba block")


def get_first_attention_config(block_configs: list[dict[str, Any]]) -> dict[str, Any]:
    for block in block_configs:
        attention = block.get("attention") or {}
        if not attention.get("no_op", False) and attention.get("mamba") is None:
            return attention
    raise ValueError("Source config has no attention block")


def get_first_moe_config(block_configs: list[dict[str, Any]]) -> dict[str, Any]:
    for block in block_configs:
        ffn = block.get("ffn") or {}
        moe = ffn.get("moe")
        if not ffn.get("no_op", False) and moe is not None:
            return moe
    raise ValueError("Source config has no MoE block")


def convert_block_configs(
    block_configs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    converted: list[dict[str, Any]] = []
    pattern = []

    for block in block_configs:
        attention = block.get("attention") or {}
        ffn = block.get("ffn") or {}

        if not attention.get("no_op", False):
            if attention.get("mamba") is not None:
                converted.append({"block_type": "mamba"})
                pattern.append("M")
            else:
                new_attention: dict[str, Any] = {"block_type": "attention"}
                if attention.get("window_length") is not None:
                    new_attention["sliding_window"] = attention["window_length"]
                for field in Q_INDEXER_FIELDS:
                    if attention.get(field) is not None:
                        new_attention[field] = attention[field]
                converted.append(new_attention)
                pattern.append("*")
        elif not ffn.get("no_op", False):
            moe = ffn.get("moe")
            if moe is None:
                converted.append(
                    {
                        "block_type": "mlp",
                        "intermediate_size": ffn["intermediate_size"],
                    }
                )
                pattern.append("-")
            else:
                converted.append(
                    {
                        "block_type": "moe",
                        "n_routed_experts": moe["num_local_experts"],
                        "num_experts_per_tok": moe["num_experts_per_tok"],
                        "moe_intermediate_size": moe["expert_intermediate_dim"],
                        "moe_shared_expert_intermediate_size": moe[
                            "shared_expert_intermediate_dim"
                        ],
                        "moe_latent_size": moe.get("moe_latent_size"),
                    }
                )
                pattern.append("E")
        else:
            raise ValueError(f"Cannot convert no-op-only block: {block}")

    return converted, "".join(pattern)


def build_converted_config(src: dict[str, Any], dtype: str | None) -> dict[str, Any]:
    src_blocks = src["block_configs"]
    block_configs, pattern = convert_block_configs(src_blocks)
    mamba = get_first_mamba_config(src_blocks)
    attention = get_first_attention_config(src_blocks)
    moe = get_first_moe_config(src_blocks)

    n_heads_in_group = attention.get("n_heads_in_group")
    if not n_heads_in_group:
        raise ValueError("Attention block is missing n_heads_in_group")

    converted: dict[str, Any] = {
        "architectures": ["NemotronHPuzzleForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_nemotron_h_puzzle.NemotronHPuzzleConfig",
            "AutoModelForCausalLM": (
                "modeling_nemotron_h_puzzle.NemotronHPuzzleForCausalLM"
            ),
        },
        "model_type": "nemotron_h_puzzle",
        "block_configs": block_configs,
        "mtp_block_configs": [],
        "hybrid_override_pattern": pattern,
        "mtp_hybrid_override_pattern": "",
        "num_nextn_predict_layers": 0,
        "vocab_size": src.get("vocab_size", 131072),
        "tie_word_embeddings": src.get("tie_word_embeddings", False),
        "hidden_size": src["hidden_size"],
        "intermediate_size": first_present(
            [
                (block.get("ffn") or {}).get("intermediate_size")
                for block in src_blocks
            ],
            0,
        ),
        "num_hidden_layers": len(block_configs),
        "num_attention_heads": src["num_attention_heads"],
        "num_key_value_heads": src["num_attention_heads"] // n_heads_in_group,
        "head_dim": src["head_dim"],
        "attention_bias": src.get("attention_bias", False),
        "attention_dropout": src.get("attention_dropout", 0.0),
        "hidden_dropout": 0.0,
        "mlp_bias": src.get("mlp_bias", False),
        "use_bias": bool(first_present([mamba.get("use_bias")], False)),
        "use_conv_bias": bool(first_present([mamba.get("use_conv_bias")], True)),
        "initializer_range": src.get("initializer_range", 0.02),
        "layer_norm_epsilon": src.get("rms_norm_eps", 1e-5),
        "norm_eps": src.get("rms_norm_eps", 1e-5),
        "residual_in_fp32": False,
        "use_cache": src.get("use_cache", True),
        "num_logits_to_keep": 1,
        "pad_token_id": src.get("pad_token_id", 0),
        "bos_token_id": src.get("bos_token_id", 1),
        "eos_token_id": src.get("eos_token_id", 2),
        "sliding_window": src.get("sliding_window"),
        "max_position_embeddings": src.get("max_position_embeddings", 4096),
        "use_mamba_kernels": True,
        "ssm_state_size": first_present([mamba.get("state_dim")], 128),
        "mamba_num_heads": first_present([mamba.get("num_heads")], 64),
        "mamba_head_dim": first_present([mamba.get("head_dim")], 64),
        "n_groups": first_present([mamba.get("num_groups")], 8),
        "conv_kernel": first_present([mamba.get("conv_kernel")], 4),
        "chunk_size": first_present([mamba.get("chunk_size")], 128),
        "expand": 2,
        "mamba_hidden_act": first_present([mamba.get("hidden_act")], "silu"),
        "mamba_proj_bias": bool(first_present([mamba.get("use_bias")], False)),
        "mamba_ssm_cache_dtype": "float32",
        "time_step_min": first_present([mamba.get("time_step_min")], 0.001),
        "time_step_max": first_present([mamba.get("time_step_max")], 0.1),
        "time_step_limit": first_present(
            [mamba.get("time_step_limit")],
            [0.0, float("inf")],
        ),
        "time_step_floor": 1e-4,
        "mlp_hidden_act": "relu2",
        "n_routed_experts": moe["num_local_experts"],
        "n_shared_experts": 1,
        "moe_intermediate_size": moe["expert_intermediate_dim"],
        "moe_shared_expert_intermediate_size": moe[
            "shared_expert_intermediate_dim"
        ],
        "moe_latent_size": moe.get("moe_latent_size"),
        "moe_shared_expert_overlap": True,
        "num_experts_per_tok": moe["num_experts_per_tok"],
        "routed_scaling_factor": moe.get("routed_scaling_factor", 1.0),
        "n_group": moe.get("n_group", 1),
        "topk_group": moe.get("topk_group", 1),
        "norm_topk_prob": moe.get("norm_topk_prob", True),
        "transformers_version": src.get("transformers_version"),
    }

    if dtype is None:
        dtype = src.get("dtype", "float32")
    converted["dtype"] = dtype

    blockwise_members = set()
    for block in block_configs:
        for key, value in block.items():
            if value is not None:
                blockwise_members.add(key)
    converted["blockwise_members"] = sorted(blockwise_members)

    return converted


def rename_tensor_name(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "backbone.embeddings.weight"
    if name == "model.norm.weight":
        return "backbone.norm_f.weight"
    if name == "lm_head.weight":
        return name

    match = re.fullmatch(r"model\.layers\.(\d+)\.input_layernorm\.weight", name)
    if match:
        return f"backbone.layers.{match.group(1)}.norm.weight"

    match = re.fullmatch(
        r"model\.layers\.(\d+)\.post_attention_layernorm\.weight",
        name,
    )
    if match:
        return f"backbone.layers.{match.group(1)}.norm.weight"

    match = re.fullmatch(
        r"model\.layers\.(\d+)\.self_attn\.mamba_mixer\.(.+)",
        name,
    )
    if match:
        return f"backbone.layers.{match.group(1)}.mixer.{match.group(2)}"

    match = re.fullmatch(r"model\.layers\.(\d+)\.self_attn\.(.+)", name)
    if match:
        return f"backbone.layers.{match.group(1)}.mixer.{match.group(2)}"

    match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.router\.(.+)", name)
    if match:
        return f"backbone.layers.{match.group(1)}.mixer.gate.{match.group(2)}"

    match = re.fullmatch(
        r"model\.layers\.(\d+)\.mlp\.e_score_correction_bias",
        name,
    )
    if match:
        return f"backbone.layers.{match.group(1)}.mixer.gate.e_score_correction_bias"

    match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.shared_expert\.(.+)", name)
    if match:
        return f"backbone.layers.{match.group(1)}.mixer.shared_experts.{match.group(2)}"

    match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.(.+)", name)
    if match:
        return f"backbone.layers.{match.group(1)}.mixer.{match.group(2)}"

    raise ValueError(f"Do not know how to rename tensor {name!r}")


def target_weight_file(old_filename: str) -> str:
    return Path(old_filename).name


def is_dsa_extra_tensor(name: str) -> bool:
    return DSA_EXTRA_TENSOR_RE.fullmatch(name) is not None


def torch_dtype_from_config(dtype: str | None) -> Any:
    import torch

    if dtype is None:
        return None
    normalized = str(dtype).removeprefix("torch.").lower()
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    if normalized in {"float32", "fp32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype for tensor casting: {dtype!r}")


def validate_base_config(base_config: dict[str, Any], out_config: dict[str, Any]) -> None:
    checks = (
        "hidden_size",
        "head_dim",
        "num_attention_heads",
        "num_key_value_heads",
        "num_hidden_layers",
        "vocab_size",
        "hybrid_override_pattern",
    )
    mismatches = []
    for key in checks:
        base_value = base_config.get(key)
        out_value = out_config.get(key)
        if base_value is not None and out_value is not None and base_value != out_value:
            mismatches.append(f"{key}: base={base_value!r} converted={out_value!r}")

    if mismatches:
        raise ValueError(
            "Base Puzzle checkpoint is not architecture-compatible with the "
            "converted DSA config:\n  " + "\n  ".join(mismatches)
        )


def copy_metadata_files(source: Path, template: Path, output: Path) -> None:
    for filename in OPTIONAL_METADATA_FILES:
        src = source / filename
        if src.exists():
            shutil.copy2(src, output / filename)

    for filename in (
        "configuration_nemotron_h.py",
        "modeling_nemotron_h.py",
        "modeling_nemotron_h_puzzle.py",
    ):
        shutil.copy2(template / filename, output / filename)

    (output / "configuration_nemotron_h_puzzle.py").write_text(
        PUZZLE_CONFIG_PY,
        encoding="utf-8",
    )
    (output / "__init__.py").touch()


def rewrite_safetensors(
    source: Path,
    output: Path,
    groups: dict[str, list[tuple[str, str, str]]],
) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    for old_filename, mappings in sorted(groups.items()):
        src_file = source / old_filename
        dst_file = output / target_weight_file(old_filename)
        print(f"rewriting {src_file.name} -> {dst_file.name} ({len(mappings)} tensors)")

        tensors: dict[str, torch.Tensor] = {}
        with safe_open(src_file, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            for old_name, new_name, _new_filename in mappings:
                tensors[new_name] = f.get_tensor(old_name)

        tmp_file = dst_file.with_suffix(dst_file.suffix + ".tmp")
        save_file(tensors, tmp_file, metadata=metadata)
        os.replace(tmp_file, dst_file)


def rewrite_safetensors_with_base(
    source: Path,
    base_source: Path,
    output: Path,
    base_groups: dict[str, list[tuple[str, str, str]]],
    extra_groups: dict[str, list[tuple[str, str, str]]],
    cast_floating_dtype_name: str | None,
) -> None:
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    cast_floating_dtype = torch_dtype_from_config(cast_floating_dtype_name)
    base_groups_by_dst: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for base_filename, mappings in base_groups.items():
        for old_name, new_name, new_filename in mappings:
            base_groups_by_dst[new_filename].append(
                (base_filename, old_name, new_name)
            )

    missing_base_files = sorted(set(extra_groups) - set(base_groups_by_dst))
    if missing_base_files:
        raise ValueError(
            "DSA tensors map to files not present in the base checkpoint: "
            + ", ".join(missing_base_files)
        )

    for filename in sorted(base_groups_by_dst):
        dst_file = output / filename
        extras = extra_groups.get(filename, [])

        print(f"writing {dst_file.name} ({len(extras)} added DSA tensors)")
        tensors: dict[str, torch.Tensor] = {}
        metadata = None
        base_file_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for base_filename, old_name, new_name in base_groups_by_dst[filename]:
            base_file_groups[base_filename].append((old_name, new_name))

        for base_filename, mappings in sorted(base_file_groups.items()):
            src_file = base_source / base_filename
            with safe_open(src_file, framework="pt", device="cpu") as f:
                if metadata is None:
                    metadata = f.metadata()
                for old_name, new_name in mappings:
                    if new_name in tensors:
                        raise ValueError(
                            f"Refusing to overwrite base tensor {new_name!r}"
                        )
                    tensors[new_name] = f.get_tensor(old_name)

        for old_filename, old_name, new_name in extras:
            extra_src_file = source / old_filename
            with safe_open(extra_src_file, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(old_name)
            if tensor.is_floating_point() and cast_floating_dtype is not None:
                tensor = tensor.to(cast_floating_dtype)
            if new_name in tensors:
                raise ValueError(
                    f"Refusing to overwrite existing base tensor {new_name!r}"
                )
            tensors[new_name] = tensor

        tmp_file = dst_file.with_suffix(dst_file.suffix + ".tmp")
        save_file(tensors, tmp_file, metadata=metadata)
        os.replace(tmp_file, dst_file)


def convert_checkpoint_with_base(args: argparse.Namespace) -> None:
    source: Path = args.source
    base_source: Path = args.base_source
    output: Path = args.output
    template: Path = args.template_dir

    if not args.dry_run:
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"{output} already exists and is not empty; pass --overwrite"
            )
        output.mkdir(parents=True, exist_ok=True)

    src_config = load_json(source / "config.json")
    base_config = load_json(base_source / "config.json")
    out_dtype = args.dtype if args.dtype is not None else base_config.get("dtype")
    out_config = build_converted_config(src_config, out_dtype)

    old_index = load_json(source / "model.safetensors.index.json")
    old_weight_map = old_index["weight_map"]
    base_index = load_json(base_source / "model.safetensors.index.json")
    base_weight_map = base_index["weight_map"]
    base_is_puzzle = (
        base_config.get("model_type") == "nemotron_h_puzzle"
        or any(name.startswith("backbone.") for name in base_weight_map)
    )
    base_validation_config = (
        base_config
        if base_is_puzzle
        else build_converted_config(base_config, base_config.get("dtype"))
    )
    validate_base_config(base_validation_config, out_config)

    new_weight_map: dict[str, str] = {}
    base_groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for base_name, base_filename in base_weight_map.items():
        new_name = base_name if base_is_puzzle else rename_tensor_name(base_name)
        new_filename = target_weight_file(base_filename)
        if new_name in new_weight_map:
            raise ValueError(f"Duplicate converted base tensor name {new_name!r}")
        new_weight_map[new_name] = new_filename
        base_groups[base_filename].append((base_name, new_name, new_filename))

    extra_groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for old_name, old_filename in old_weight_map.items():
        if not is_dsa_extra_tensor(old_name):
            continue
        new_name = rename_tensor_name(old_name)
        new_filename = target_weight_file(old_filename)
        if new_name in new_weight_map:
            raise ValueError(
                f"Refusing to overwrite existing base tensor {new_name!r}"
            )
        new_weight_map[new_name] = new_filename
        extra_groups[new_filename].append((old_filename, old_name, new_name))

    if not extra_groups:
        raise ValueError("No DSA extra tensors found in source checkpoint")

    new_index = {
        "metadata": base_index.get("metadata", {}),
        "weight_map": new_weight_map,
    }

    print(f"source: {source}")
    print(f"base source: {base_source}")
    print(f"output: {output}")
    print(f"template: {template}")
    print(f"base format: {'puzzle' if base_is_puzzle else 'deci'}")
    print(f"layers: {out_config['num_hidden_layers']}")
    print(f"pattern: {out_config['hybrid_override_pattern']}")
    print(
        "dsa layers: "
        + str(
            [
                i
                for i, block in enumerate(out_config["block_configs"])
                if block.get("q_indexer_dim") is not None
            ]
        )
    )
    print(f"base tensors: {len(base_weight_map)}")
    print(f"added DSA tensors: {sum(len(v) for v in extra_groups.values())}")
    print(f"config dtype: {out_config['dtype']}")
    print(f"mamba_ssm_cache_dtype: {out_config['mamba_ssm_cache_dtype']}")

    if args.dry_run:
        return

    copy_metadata_files(base_source, template, output)
    dump_json(output / "config.json", out_config)
    dump_json(output / "model.safetensors.index.json", new_index)
    if args.skip_tensors:
        print("skipped safetensors rewrite")
        return
    rewrite_safetensors_with_base(
        source,
        base_source,
        output,
        base_groups,
        extra_groups,
        out_config["dtype"],
    )


def convert_checkpoint(args: argparse.Namespace) -> None:
    if args.base_source is not None:
        convert_checkpoint_with_base(args)
        return

    source: Path = args.source
    output: Path = args.output
    template: Path = args.template_dir

    if not args.dry_run:
        if output.exists() and any(output.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"{output} already exists and is not empty; pass --overwrite"
            )
        output.mkdir(parents=True, exist_ok=True)

    src_config = load_json(source / "config.json")
    out_config = build_converted_config(src_config, args.dtype)

    old_index = load_json(source / "model.safetensors.index.json")
    old_weight_map = old_index["weight_map"]
    new_weight_map: dict[str, str] = {}
    groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for old_name, old_filename in old_weight_map.items():
        new_name = rename_tensor_name(old_name)
        new_filename = target_weight_file(old_filename)
        new_weight_map[new_name] = new_filename
        groups[old_filename].append((old_name, new_name, new_filename))

    new_index = {
        "metadata": old_index.get("metadata", {}),
        "weight_map": new_weight_map,
    }

    print(f"source: {source}")
    print(f"output: {output}")
    print(f"template: {template}")
    print(f"layers: {out_config['num_hidden_layers']}")
    print(f"pattern: {out_config['hybrid_override_pattern']}")
    print(
        "dsa layers: "
        + str(
            [
                i
                for i, block in enumerate(out_config["block_configs"])
                if block.get("q_indexer_dim") is not None
            ]
        )
    )
    print(f"tensors: {len(new_weight_map)}")

    if args.dry_run:
        return

    copy_metadata_files(source, template, output)
    dump_json(output / "config.json", out_config)
    dump_json(output / "model.safetensors.index.json", new_index)
    if args.skip_tensors:
        print("skipped safetensors rewrite")
        return
    rewrite_safetensors(source, output, groups)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SOURCE.with_name(DEFAULT_SOURCE.name + "_nemotron_h_puzzle"),
    )
    parser.add_argument(
        "--base-source",
        type=Path,
        default=None,
        help=(
            "Optional Nemotron-H Puzzle checkpoint to use as the tensor base. "
            "Only DSA indexer tensors are injected from --source."
        ),
    )
    parser.add_argument("--template-dir", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument(
        "--dtype",
        default=None,
        help="dtype to write in config.json; defaults to the source config dtype",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-tensors",
        action="store_true",
        help="write config/code/index only; useful for validating metadata cheaply",
    )
    return parser.parse_args()


if __name__ == "__main__":
    convert_checkpoint(parse_args())
