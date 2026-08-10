#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare two generation JSONL files by token-prefix agreement.

Example:
    .venv/bin/python scripts/generation_disagreement/compare.py \
        outputs/generation_disagreement/reference.jsonl \
        outputs/generation_disagreement/current.jsonl \
        --fail-under-median-agreement 64 \
        --early-threshold 8 \
        --fail-over-early-count 3
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PromptDiff:
    prompt_id: str
    category: str
    agreement: int
    reference_len: int
    candidate_len: int
    exact: bool
    reference_finish_reason: str | None
    candidate_finish_reason: str | None

    @property
    def agreement_fraction_reference(self) -> float:
        if self.reference_len == 0:
            return 1.0 if self.candidate_len == 0 else 0.0
        return self.agreement / self.reference_len

    @property
    def length_delta(self) -> int:
        return self.candidate_len - self.reference_len


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare generation outputs by common output-token prefix."
    )
    parser.add_argument("reference", type=pathlib.Path)
    parser.add_argument("candidate", type=pathlib.Path)
    parser.add_argument(
        "--allow-prompt-token-mismatch",
        action="store_true",
        help="Compare outputs even when prompt token IDs differ.",
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=10,
        help="Number of lowest-agreement prompts to print.",
    )
    parser.add_argument(
        "--early-threshold",
        type=int,
        default=10,
        help="Agreement below this token count is considered early divergence.",
    )
    parser.add_argument(
        "--fail-under-min-agreement",
        type=int,
        default=None,
        help="Exit nonzero if any prompt has fewer agreement tokens.",
    )
    parser.add_argument(
        "--fail-under-p10-agreement",
        type=float,
        default=None,
        help="Exit nonzero if p10 agreement is below this value.",
    )
    parser.add_argument(
        "--fail-under-median-agreement",
        type=float,
        default=None,
        help="Exit nonzero if median agreement is below this value.",
    )
    parser.add_argument(
        "--fail-over-early-count",
        type=int,
        default=None,
        help=(
            "Exit nonzero if more than this many prompts have agreement below "
            "--early-threshold."
        ),
    )
    parser.add_argument(
        "--output-json",
        type=pathlib.Path,
        default=None,
        help="Optional destination for machine-readable summary JSON.",
    )
    return parser.parse_args()


def load_generation_records(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("type", "generation") != "generation":
                continue
            if "prompt_id" not in record:
                raise ValueError(f"{path}:{line_no} is missing prompt_id")
            if "output_token_ids" not in record:
                raise ValueError(f"{path}:{line_no} is missing output_token_ids")
            prompt_id = str(record["prompt_id"])
            if prompt_id in records:
                raise ValueError(f"{path}:{line_no} duplicates prompt id {prompt_id}")
            records[prompt_id] = record
    if not records:
        raise ValueError(f"No generation records found in {path}")
    return records


def common_prefix_len(left: list[int], right: list[int]) -> int:
    count = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        count += 1
    return count


def percentile(values: list[int], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * pct / 100.0
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def compare_records(
    reference: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
    allow_prompt_token_mismatch: bool,
) -> list[PromptDiff]:
    reference_ids = set(reference)
    candidate_ids = set(candidate)
    missing = sorted(reference_ids - candidate_ids)
    extra = sorted(candidate_ids - reference_ids)
    if missing or extra:
        raise ValueError(
            "Prompt id sets differ. "
            f"Missing in candidate: {missing[:10]}; extra in candidate: {extra[:10]}"
        )

    diffs: list[PromptDiff] = []
    prompt_mismatches: list[str] = []
    for prompt_id in sorted(reference):
        ref = reference[prompt_id]
        cand = candidate[prompt_id]
        if ref.get("prompt_token_ids") != cand.get("prompt_token_ids"):
            prompt_mismatches.append(prompt_id)
        ref_ids = list(ref["output_token_ids"])
        cand_ids = list(cand["output_token_ids"])
        agreement = common_prefix_len(ref_ids, cand_ids)
        diffs.append(
            PromptDiff(
                prompt_id=prompt_id,
                category=str(ref.get("category", "")),
                agreement=agreement,
                reference_len=len(ref_ids),
                candidate_len=len(cand_ids),
                exact=ref_ids == cand_ids,
                reference_finish_reason=ref.get("finish_reason"),
                candidate_finish_reason=cand.get("finish_reason"),
            )
        )

    if prompt_mismatches and not allow_prompt_token_mismatch:
        preview = ", ".join(prompt_mismatches[:10])
        raise ValueError(
            "Prompt token IDs differ for "
            f"{len(prompt_mismatches)} prompt(s): {preview}. "
            "Pass --allow-prompt-token-mismatch to compare anyway."
        )
    return diffs


def make_summary(diffs: list[PromptDiff], early_threshold: int) -> dict[str, Any]:
    agreements = [diff.agreement for diff in diffs]
    ref_lens = [diff.reference_len for diff in diffs]
    cand_lens = [diff.candidate_len for diff in diffs]
    agreement_fracs = [diff.agreement_fraction_reference for diff in diffs]
    length_deltas = [diff.length_delta for diff in diffs]
    early_count = sum(diff.agreement < early_threshold for diff in diffs)
    finish_reason_mismatch_count = sum(
        diff.reference_finish_reason != diff.candidate_finish_reason
        for diff in diffs
    )
    thresholds = [1, 2, 5, 10, 25, 50, 100, 250, 500, 1000]

    return {
        "count": len(diffs),
        "exact_match_count": sum(diff.exact for diff in diffs),
        "diverged_count": sum(not diff.exact for diff in diffs),
        "agreement": {
            "min": min(agreements),
            "p10": percentile(agreements, 10),
            "p25": percentile(agreements, 25),
            "median": statistics.median(agreements),
            "mean": statistics.mean(agreements),
            "p75": percentile(agreements, 75),
            "p90": percentile(agreements, 90),
            "max": max(agreements),
        },
        "agreement_fraction_reference": {
            "mean": statistics.mean(agreement_fracs),
            "min": min(agreement_fracs),
        },
        "reference_output_len": {
            "min": min(ref_lens),
            "mean": statistics.mean(ref_lens),
            "max": max(ref_lens),
        },
        "candidate_output_len": {
            "min": min(cand_lens),
            "mean": statistics.mean(cand_lens),
            "max": max(cand_lens),
        },
        "length_delta": {
            "min": min(length_deltas),
            "mean": statistics.mean(length_deltas),
            "max": max(length_deltas),
        },
        "early_threshold": early_threshold,
        "early_count": early_count,
        "finish_reason_mismatch_count": finish_reason_mismatch_count,
        "counts_below_agreement": {
            str(threshold): sum(diff.agreement < threshold for diff in diffs)
            for threshold in thresholds
        },
    }


def print_summary(summary: dict[str, Any], diffs: list[PromptDiff], worst: int) -> None:
    count = summary["count"]
    exact = summary["exact_match_count"]
    agreement = summary["agreement"]
    ref_len = summary["reference_output_len"]
    cand_len = summary["candidate_output_len"]
    print(f"Compared {count} prompt generations")
    print(f"Exact token matches: {exact}/{count} ({exact / count:.1%})")
    print(
        "Agreement tokens: "
        f"min={agreement['min']} "
        f"p10={agreement['p10']:.1f} "
        f"p25={agreement['p25']:.1f} "
        f"median={agreement['median']:.1f} "
        f"mean={agreement['mean']:.1f} "
        f"p75={agreement['p75']:.1f} "
        f"p90={agreement['p90']:.1f} "
        f"max={agreement['max']}"
    )
    print(
        "Output lengths: "
        f"reference mean={ref_len['mean']:.1f} "
        f"range=[{ref_len['min']}, {ref_len['max']}], "
        f"candidate mean={cand_len['mean']:.1f} "
        f"range=[{cand_len['min']}, {cand_len['max']}]"
    )
    print(
        f"Early divergences (<{summary['early_threshold']} tokens): "
        f"{summary['early_count']}/{count}"
    )
    print(
        "Finish reason mismatches: "
        f"{summary['finish_reason_mismatch_count']}/{count}"
    )
    print("Counts below agreement thresholds:")
    print(
        "  "
        + "  ".join(
            f"<{threshold}: {value}"
            for threshold, value in summary["counts_below_agreement"].items()
        )
    )

    if worst <= 0:
        return
    print(f"Worst {min(worst, len(diffs))} prompts:")
    print("  prompt_id category agreement ref_len cand_len ref_finish cand_finish")
    for diff in sorted(diffs, key=lambda d: (d.agreement, d.prompt_id))[:worst]:
        print(
            "  "
            f"{diff.prompt_id} "
            f"{diff.category} "
            f"{diff.agreement} "
            f"{diff.reference_len} "
            f"{diff.candidate_len} "
            f"{diff.reference_finish_reason} "
            f"{diff.candidate_finish_reason}"
        )


def check_failures(summary: dict[str, Any], args: argparse.Namespace) -> list[str]:
    failures: list[str] = []
    agreement = summary["agreement"]
    if (
        args.fail_under_min_agreement is not None
        and agreement["min"] < args.fail_under_min_agreement
    ):
        failures.append(
            f"min agreement {agreement['min']} < "
            f"{args.fail_under_min_agreement}"
        )
    if (
        args.fail_under_p10_agreement is not None
        and agreement["p10"] < args.fail_under_p10_agreement
    ):
        failures.append(
            f"p10 agreement {agreement['p10']:.1f} < "
            f"{args.fail_under_p10_agreement}"
        )
    if (
        args.fail_under_median_agreement is not None
        and agreement["median"] < args.fail_under_median_agreement
    ):
        failures.append(
            f"median agreement {agreement['median']:.1f} < "
            f"{args.fail_under_median_agreement}"
        )
    if (
        args.fail_over_early_count is not None
        and summary["early_count"] > args.fail_over_early_count
    ):
        failures.append(
            f"early count {summary['early_count']} > "
            f"{args.fail_over_early_count}"
        )
    return failures


def main() -> None:
    args = parse_args()
    if args.early_threshold < 0:
        raise SystemExit("error: --early-threshold must be nonnegative")
    if args.worst < 0:
        raise SystemExit("error: --worst must be nonnegative")

    reference = load_generation_records(args.reference)
    candidate = load_generation_records(args.candidate)
    diffs = compare_records(
        reference,
        candidate,
        allow_prompt_token_mismatch=args.allow_prompt_token_mismatch,
    )
    summary = make_summary(diffs, args.early_threshold)
    print_summary(summary, diffs, args.worst)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    failures = check_failures(summary, args)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
