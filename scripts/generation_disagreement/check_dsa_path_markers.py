#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check that a generation-disagreement run exercised required DSA paths."""

from __future__ import annotations

import argparse
import pathlib
import re
import sys


REQUIRED_MARKERS = (
    "config",
    "dense_prefill_page_table_bucket",
    "sparse_prefill_page_table_bucket",
    "sparse_decode",
)

MARKER_ALIASES = {
    "sparse_decode": (
        "sparse_decode",
        "sparse_decode_page_table_bucket",
        "sparse_decode_page_table_fa",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a generation log for DSA_PATH_MARKER lines and verify "
            "that dense prefill, sparse prefill, and sparse decode were seen."
        )
    )
    parser.add_argument("log_file", type=pathlib.Path)
    parser.add_argument(
        "--markers-output",
        type=pathlib.Path,
        default=None,
        help="Optional file for all DSA_PATH_MARKER log lines.",
    )
    parser.add_argument(
        "--check-output",
        type=pathlib.Path,
        default=None,
        help="Optional file for MARKER_FOUND/MARKER_MISSING status lines.",
    )
    return parser.parse_args()


def marker_pattern(marker: str) -> re.Pattern[str]:
    return re.compile(rf"(^|\s)marker={re.escape(marker)}(\s|$)")


def marker_seen(marker_lines: list[str], marker: str) -> bool:
    aliases = MARKER_ALIASES.get(marker, (marker,))
    patterns = [marker_pattern(alias) for alias in aliases]
    return any(
        pattern.search(line)
        for line in marker_lines
        for pattern in patterns
    )


def main() -> None:
    args = parse_args()
    marker_lines = [
        line.rstrip("\n")
        for line in args.log_file.read_text(errors="replace").splitlines()
        if "DSA_PATH_MARKER" in line
    ]

    if args.markers_output is not None:
        args.markers_output.parent.mkdir(parents=True, exist_ok=True)
        args.markers_output.write_text("\n".join(marker_lines) + "\n")

    status_lines: list[str] = []
    missing: list[str] = []
    for marker in REQUIRED_MARKERS:
        if marker_seen(marker_lines, marker):
            status_lines.append(f"MARKER_FOUND {marker}")
        else:
            status_lines.append(f"MARKER_MISSING {marker}")
            missing.append(marker)

    if args.check_output is not None:
        args.check_output.parent.mkdir(parents=True, exist_ok=True)
        args.check_output.write_text("\n".join(status_lines) + "\n")

    if missing:
        print("Missing required DSA path markers: " + ", ".join(missing))
        sys.exit(1)

    print("All required DSA path markers were seen.")


if __name__ == "__main__":
    main()
