# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Print cheap header/config checks for a DSA Nemotron-H Puzzle checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from safetensors import safe_open


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint
    config = load_json(checkpoint / "config.json")
    dsa_layers = [
        i
        for i, block in enumerate(config["block_configs"])
        if block.get("q_indexer_dim") is not None
    ]
    index = load_json(checkpoint / "model.safetensors.index.json")
    dsa_names = sorted(
        name
        for name in index["weight_map"]
        if "indexer_q_proj" in name or "dsa_winner_query_heads" in name
    )

    print(f"checkpoint: {checkpoint}")
    print(f"config dtype: {config.get('dtype')}")
    print(f"mamba_ssm_cache_dtype: {config.get('mamba_ssm_cache_dtype')}")
    print(f"dsa layers: {dsa_layers}")
    print(f"dsa tensor count: {len(dsa_names)}")
    print(f"first dsa tensors: {dsa_names[:4]}")

    suffixes = (
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "o_proj.weight",
        "indexer_q_proj.weight",
        "dsa_winner_query_heads",
    )
    for layer in dsa_layers:
        path = checkpoint / f"block_{layer}_attention.safetensors"
        print(f"layer {layer}:")
        with safe_open(path, framework="pt", device="cpu") as f:
            for suffix in suffixes:
                name = f"backbone.layers.{layer}.mixer.{suffix}"
                tensor = f.get_slice(name)
                print(f"  {suffix}: {tensor.get_dtype()} {tensor.get_shape()}")


if __name__ == "__main__":
    main()
