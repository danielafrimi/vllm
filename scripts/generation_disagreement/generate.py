#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generate deterministic outputs for a fast generation disagreement check.

This is a lightweight smoke test for refactors where bit-exact output is not
expected. It renders a fixed prompt database to a controlled token length,
runs greedy generation, and writes one JSONL record per prompt.

Example:
    .venv/bin/python scripts/generation_disagreement/generate.py \
        --model nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
        --output-file outputs/generation_disagreement/current.jsonl \
        --tensor-parallel-size 2 \
        --max-model-len 8192 \
        --max-num-seqs 4 \
        --enable-chunked-prefill \
        --max-num-batched-tokens 1024
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import time
from typing import Any

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


SCHEMA_VERSION = 1
DEFAULT_PROMPT_FILE = pathlib.Path(__file__).with_name("prompts.jsonl")
DEFAULT_TARGET_PROMPT_TOKENS = 4096
DEFAULT_MAX_TOKENS = 200


def parse_args() -> argparse.Namespace:
    parser = FlexibleArgumentParser(
        description=(
            "Generate deterministic long completions for token-prefix "
            "disagreement analysis."
        )
    )
    parser.add_argument(
        "--prompt-file",
        type=pathlib.Path,
        default=DEFAULT_PROMPT_FILE,
        help="JSONL prompt specification file.",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        required=True,
        help="Destination JSONL file for generated outputs.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N prompt specs.",
    )
    parser.add_argument(
        "--target-prompt-tokens",
        type=int,
        default=DEFAULT_TARGET_PROMPT_TOKENS,
        help="Exact prompt token length produced for each prompt.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of output tokens to generate per prompt.",
    )
    parser.add_argument(
        "--generation-seed",
        type=int,
        default=0,
        help="Sampling seed. Greedy decoding is used, but the seed is recorded.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable vLLM generation progress bars.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite --output-file if it already exists.",
    )
    parser.add_argument(
        "--include-output-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include generated text in JSONL records.",
    )
    parser.add_argument(
        "--include-prompt-text",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include decoded prompt text in JSONL records.",
    )

    parser = EngineArgs.add_cli_args(parser)
    return parser.parse_args()


def load_prompt_specs(path: pathlib.Path, limit: int | None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            spec = json.loads(stripped)
            for key in ("id", "category", "task", "context"):
                if key not in spec:
                    raise ValueError(f"{path}:{line_no} is missing required key {key}")
            prompt_id = str(spec["id"])
            if prompt_id in seen_ids:
                raise ValueError(f"{path}:{line_no} duplicates prompt id {prompt_id}")
            seen_ids.add(prompt_id)
            specs.append(spec)
            if limit is not None and len(specs) >= limit:
                break
    if not specs:
        raise ValueError(f"No prompt specs loaded from {path}")
    return specs


def encode_text(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    except TypeError:
        token_ids = tokenizer.encode(text)
    return [int(token_id) for token_id in token_ids]


def model_was_explicitly_passed(argv: list[str]) -> bool:
    return any(arg == "--model" or arg.startswith("--model=") for arg in argv[1:])


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_base(spec: dict[str, Any]) -> str:
    keywords = ", ".join(str(k) for k in spec.get("keywords", []))
    guidance = spec.get(
        "guidance",
        "Use the context carefully, state assumptions, and keep the response "
        "internally consistent.",
    )
    return (
        "Long-form generation regression prompt.\n"
        f"Prompt id: {spec['id']}\n"
        f"Category: {spec['category']}\n"
        f"Keywords: {keywords}\n\n"
        "Task:\n"
        f"{spec['task']}\n\n"
        "Context:\n"
        f"{spec['context']}\n\n"
        "Guidance:\n"
        f"{guidance}\n\n"
    )


def filler_piece(spec: dict[str, Any], index: int) -> str:
    keywords = [str(k) for k in spec.get("keywords", [])] or [str(spec["category"])]
    keyword = keywords[index % len(keywords)]
    digest = hashlib.sha256(f"{spec['id']}:{index}".encode("utf-8")).hexdigest()
    return (
        f"Evidence note {index + 1:03d} [{digest[:12]}]: "
        f"Keep tracking {keyword}, edge cases, tradeoffs, and causal links. "
        f"The local context remains: {spec['context']} "
        "Prefer concrete reasoning over slogans, and preserve distinctions "
        "between observations, inferences, and recommendations.\n"
    )


def render_prompt_token_ids(
    tokenizer: Any,
    spec: dict[str, Any],
    target_prompt_tokens: int,
) -> tuple[list[int], str]:
    suffix = (
        "\nNow write the answer. Use twelve numbered sections. "
        "Make the response detailed enough that small numerical changes in "
        "generation have room to appear, but do not mention this regression "
        "test.\nAnswer:\n"
    )
    base_ids = encode_text(
        tokenizer, prompt_base(spec), add_special_tokens=True
    )
    suffix_ids = encode_text(tokenizer, suffix, add_special_tokens=False)
    filler_budget = target_prompt_tokens - len(base_ids) - len(suffix_ids)
    if filler_budget < 0:
        raise ValueError(
            f"Prompt {spec['id']} is too long for target_prompt_tokens="
            f"{target_prompt_tokens}; needs at least "
            f"{len(base_ids) + len(suffix_ids)} tokens"
        )

    filler_ids: list[int] = []
    piece_index = 0
    while len(filler_ids) < filler_budget:
        piece = filler_piece(spec, piece_index)
        filler_ids.extend(
            encode_text(tokenizer, piece, add_special_tokens=False)
        )
        piece_index += 1

    prompt_token_ids = base_ids + filler_ids[:filler_budget] + suffix_ids
    prompt_text = tokenizer.decode(prompt_token_ids)
    if len(prompt_token_ids) != target_prompt_tokens:
        raise AssertionError(
            f"Expected {target_prompt_tokens} prompt tokens, got "
            f"{len(prompt_token_ids)}"
        )
    return prompt_token_ids, prompt_text


def make_record(
    spec: dict[str, Any],
    prompt_token_ids: list[int],
    prompt_text: str,
    output: Any,
    include_output_text: bool,
    include_prompt_text: bool,
) -> dict[str, Any]:
    completion = output.outputs[0]
    output_token_ids = [int(token_id) for token_id in completion.token_ids]
    record = {
        "type": "generation",
        "schema_version": SCHEMA_VERSION,
        "prompt_id": str(spec["id"]),
        "category": str(spec["category"]),
        "prompt_token_ids": prompt_token_ids,
        "prompt_token_len": len(prompt_token_ids),
        "prompt_token_sha256": stable_hash(prompt_token_ids),
        "output_token_ids": output_token_ids,
        "output_token_len": len(output_token_ids),
        "output_token_sha256": stable_hash(output_token_ids),
        "finish_reason": completion.finish_reason,
        "stop_reason": completion.stop_reason,
    }
    if completion.cumulative_logprob is not None:
        record["cumulative_logprob"] = completion.cumulative_logprob
    if include_prompt_text:
        record["prompt_text"] = prompt_text
    if include_output_text:
        record["output_text"] = completion.text
    return record


def write_jsonl_record(f: Any, record: dict[str, Any]) -> None:
    f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
    f.write("\n")
    f.flush()


def main() -> None:
    args = parse_args()
    if not model_was_explicitly_passed(sys.argv):
        raise SystemExit(
            "error: pass --model explicitly for the model under test"
        )
    if args.target_prompt_tokens <= 0:
        raise SystemExit("error: --target-prompt-tokens must be positive")
    if args.max_tokens <= 0:
        raise SystemExit("error: --max-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("error: --limit must be positive")
    if args.output_file.exists() and not args.overwrite:
        raise SystemExit(
            f"error: {args.output_file} already exists; pass --overwrite"
        )

    specs = load_prompt_specs(args.prompt_file, args.limit)
    engine_args = EngineArgs.from_cli_args(args)

    start_time = time.time()
    llm = LLM.from_engine_args(engine_args)
    tokenizer = llm.get_tokenizer()

    rendered = [
        render_prompt_token_ids(tokenizer, spec, args.target_prompt_tokens)
        for spec in specs
    ]
    prompt_inputs = [
        {"prompt_token_ids": prompt_token_ids, "prompt": prompt_text}
        for prompt_token_ids, prompt_text in rendered
    ]
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=args.max_tokens,
        seed=args.generation_seed,
    )

    outputs = llm.generate(
        prompt_inputs,
        sampling_params=sampling_params,
        use_tqdm=not args.disable_tqdm,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    with args.output_file.open(mode, encoding="utf-8") as f:
        write_jsonl_record(
            f,
            {
                "type": "metadata",
                "schema_version": SCHEMA_VERSION,
                "created_unix_time": time.time(),
                "elapsed_seconds": time.time() - start_time,
                "model": args.model,
                "prompt_file": str(args.prompt_file),
                "prompt_file_sha256": stable_hash(specs),
                "prompt_count": len(specs),
                "target_prompt_tokens": args.target_prompt_tokens,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "generation_seed": args.generation_seed,
            },
        )
        for spec, (prompt_token_ids, prompt_text), output in zip(
            specs, rendered, outputs, strict=True
        ):
            write_jsonl_record(
                f,
                make_record(
                    spec,
                    prompt_token_ids,
                    prompt_text,
                    output,
                    args.include_output_text,
                    args.include_prompt_text,
                ),
            )

    output_lens = [len(output.outputs[0].token_ids) for output in outputs]
    print(f"Wrote {len(outputs)} generations to {args.output_file}")
    print(
        "Output tokens: "
        f"min={min(output_lens)} "
        f"mean={sum(output_lens) / len(output_lens):.1f} "
        f"max={max(output_lens)}"
    )


if __name__ == "__main__":
    main()
