#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Truncate generation JSONL outputs to a shorter token baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Truncate output_token_ids in a generation JSONL file."
    )
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=200,
        help="Maximum output tokens to retain per generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    return parser.parse_args()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def truncate_generation(record: dict[str, Any], max_output_tokens: int) -> dict[str, Any]:
    original_ids = [int(token_id) for token_id in record["output_token_ids"]]
    truncated_ids = original_ids[:max_output_tokens]
    truncated = dict(record)
    truncated["output_token_ids"] = truncated_ids
    truncated["output_token_len"] = len(truncated_ids)
    truncated["output_token_sha256"] = stable_hash(truncated_ids)
    truncated["truncated_from"] = {
        "output_token_len": len(original_ids),
        "output_token_sha256": record.get("output_token_sha256"),
        "finish_reason": record.get("finish_reason"),
        "stop_reason": record.get("stop_reason"),
    }
    truncated["truncation_max_output_tokens"] = max_output_tokens
    truncated.pop("output_text", None)
    if len(original_ids) > max_output_tokens:
        truncated["finish_reason"] = "length"
        truncated["stop_reason"] = None
    return truncated


def truncate_metadata(record: dict[str, Any], max_output_tokens: int) -> dict[str, Any]:
    truncated = dict(record)
    truncated["max_tokens"] = max_output_tokens
    truncated["truncated_from"] = {
        "max_tokens": record.get("max_tokens"),
        "prompt_file_sha256": record.get("prompt_file_sha256"),
    }
    truncated["truncation_created_unix_time"] = time.time()
    return truncated


def main() -> None:
    args = parse_args()
    if args.max_output_tokens <= 0:
        raise SystemExit("error: --max-output-tokens must be positive")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"error: {args.output} already exists; pass --overwrite")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    generation_count = 0
    with args.input.open(encoding="utf-8") as src, args.output.open(
        mode, encoding="utf-8"
    ) as dst:
        for line_no, line in enumerate(src, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            record_type = record.get("type", "generation")
            if record_type == "metadata":
                out_record = truncate_metadata(record, args.max_output_tokens)
            elif record_type == "generation":
                if "output_token_ids" not in record:
                    raise ValueError(f"{args.input}:{line_no} missing output_token_ids")
                out_record = truncate_generation(record, args.max_output_tokens)
                generation_count += 1
            else:
                out_record = record
            dst.write(json.dumps(out_record, ensure_ascii=False, sort_keys=True))
            dst.write("\n")

    print(
        f"Wrote {generation_count} truncated generations to {args.output} "
        f"with max_output_tokens={args.max_output_tokens}"
    )


if __name__ == "__main__":
    main()
