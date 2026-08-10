# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare real-text prompt token IDs for fixed-IO serving benchmarks."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def iter_text_file(path: str) -> Iterator[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8")
    yield {"text": text, "source": path}


def iter_hf_dataset(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Preparing prompts from Hugging Face datasets requires the "
            "`datasets` package."
        ) from exc

    kwargs: dict[str, Any] = {
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if args.dataset_config:
        dataset = load_dataset(args.dataset_name, args.dataset_config, **kwargs)
    else:
        dataset = load_dataset(args.dataset_name, **kwargs)

    if args.shuffle_buffer > 0:
        if args.streaming:
            dataset = dataset.shuffle(
                seed=args.seed,
                buffer_size=args.shuffle_buffer,
            )
        else:
            dataset = dataset.shuffle(seed=args.seed)
    return dataset


def write_prompt_record(
    output,
    *,
    prompt_index: int,
    prompt_token_ids: list[int],
    source: dict[str, Any],
) -> None:
    output.write(
        json.dumps(
            {
                "request_id": prompt_index,
                "prompt_token_ids": prompt_token_ids,
                "source": source,
            },
            ensure_ascii=False,
        )
        + "\n"
    )


def create_prompt_token_ids_jsonl(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )
    rng = random.Random(args.seed)

    if args.source_text_file:
        records = iter_text_file(args.source_text_file)
        dataset_name = None
    else:
        records = iter_hf_dataset(args)
        dataset_name = args.dataset_name

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prompt_index = 0
    scanned_docs = 0
    skipped_short_docs = 0
    with output_path.open("w", encoding="utf-8") as output:
        for doc_index, record in enumerate(records):
            if args.max_docs is not None and scanned_docs >= args.max_docs:
                break
            scanned_docs += 1

            text = record.get(args.text_field)
            if not isinstance(text, str) or not text.strip():
                continue

            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
            ).input_ids
            if len(token_ids) < args.input_len:
                skipped_short_docs += 1
                continue

            start = rng.randint(0, len(token_ids) - args.input_len)
            source = {
                "dataset": dataset_name or record.get("source"),
                "split": None if args.source_text_file else args.split,
                "doc_index": doc_index,
                "start_token": start,
                "tokenized_doc_len": len(token_ids),
            }
            for field in args.metadata_field:
                value = record.get(field)
                if isinstance(value, (str, int, float, bool)) or value is None:
                    source[field] = value

            write_prompt_record(
                output,
                prompt_index=prompt_index,
                prompt_token_ids=token_ids[start : start + args.input_len],
                source=source,
            )
            prompt_index += 1

            if prompt_index == args.num_prompts:
                break

    if prompt_index < args.num_prompts:
        raise RuntimeError(
            f"Wrote {prompt_index} prompts to {output_path}, but expected "
            f"{args.num_prompts}. Scanned {scanned_docs} documents and skipped "
            f"{skipped_short_docs} shorter than {args.input_len} tokens."
        )

    print(
        f"Wrote {prompt_index} prompts of {args.input_len} tokens to "
        f"{output_path}"
    )


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--input-len", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--source-text-file", default=None)
    parser.add_argument("--dataset-name", default="deepmind/pg19")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-field", default="text")
    parser.add_argument(
        "--metadata-field",
        action="append",
        default=["short_book_title", "publication_date", "url"],
    )
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--shuffle-buffer", type=int, default=1000)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    create_prompt_token_ids_jsonl(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
