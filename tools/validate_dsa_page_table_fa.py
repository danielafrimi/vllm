#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Focused checks for the experimental Nemotron-H DSA page-table FA path."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path

from vllm.v1.attention.backend import MultipleOf
from vllm.v1.worker.utils import select_common_block_size


class _MonkeyPatch:
    def setattr(self, obj, name: str, value):
        setattr(obj, name, value)


class _MockBackend:
    def __init__(self, sizes):
        self._sizes = sizes

    def get_supported_kernel_block_sizes(self):
        return self._sizes


def _load_dsa_test_module():
    repo_root = Path(__file__).resolve().parents[1]
    test_path = repo_root / "tests/model_executor/models/test_nemotron_h_dsa_chunked.py"
    spec = importlib.util.spec_from_file_location(
        "test_nemotron_h_dsa_chunked_for_page_table_validation",
        test_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_synthetic_checks() -> None:
    dsa_tests = _load_dsa_test_module()
    dsa_tests.test_dsa_chunked_recall_matches_causal_reference()
    dsa_tests.test_dsa_chunked_page_table_fa_decode_matches_gather_path(
        _MonkeyPatch()
    )


def _run_kernel_block_override_check() -> None:
    os.environ["VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE"] = "16"
    try:
        selected = select_common_block_size(
            640,
            [
                _MockBackend([16, 32, 64]),
                _MockBackend([MultipleOf(16)]),
            ],
        )
    finally:
        os.environ.pop("VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE", None)
    assert selected == 16, selected


def _inspect_model_config(model_path: Path, tp_size: int) -> dict[str, int | str]:
    config = json.loads((model_path / "config.json").read_text())
    dtype_bytes = 2
    conv_dtype_bytes = 2
    ssm_dtype_bytes = 4 if config.get("mamba_ssm_cache_dtype") == "float32" else 2
    kv_heads = max(1, config["num_key_value_heads"] // tp_size)
    attn_bytes_per_token = (
        kv_heads * (config["head_dim"] + config["head_dim"]) * dtype_bytes
    )
    intermediate_size = config["mamba_num_heads"] * config["mamba_head_dim"]
    conv_dim = intermediate_size + 2 * config["n_groups"] * config["ssm_state_size"]
    conv_elems = (conv_dim // tp_size) * (config["conv_kernel"] - 1)
    temporal_elems = (
        (config["mamba_num_heads"] // tp_size)
        * config["mamba_head_dim"]
        * config["ssm_state_size"]
    )
    mamba_page_bytes = (
        conv_elems * conv_dtype_bytes + temporal_elems * ssm_dtype_bytes
    )
    tokens_per_mamba_state = math.ceil(mamba_page_bytes / attn_bytes_per_token)
    mamba_chunk_size = int(config.get("chunk_size", 128))
    manager_block_size = (
        mamba_chunk_size
        * math.ceil(tokens_per_mamba_state / mamba_chunk_size)
    )
    dsa_block_config = next(
        block
        for block in config["block_configs"]
        if block.get("q_indexer_attn_mode") == "chunked_topk_sparse"
    )
    return {
        "model": str(model_path),
        "tp_size": tp_size,
        "dsa_chunk_size": dsa_block_config["q_indexer_chunk_size"],
        "configured_chunk_top_k": dsa_block_config["q_indexer_chunk_top_k"],
        "mamba_page_bytes": mamba_page_bytes,
        "attention_bytes_per_token": attn_bytes_per_token,
        "tokens_per_mamba_state": tokens_per_mamba_state,
        "manager_block_size": manager_block_size,
        "forced_kernel_block_size": 16,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=8)
    args = parser.parse_args()

    _run_synthetic_checks()
    _run_kernel_block_override_check()
    info = _inspect_model_config(args.model, args.tp_size)
    print(json.dumps(info, indent=2, sort_keys=True))
    print("dsa page-table FA focused checks passed")


if __name__ == "__main__":
    main()
