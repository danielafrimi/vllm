#!/usr/bin/env python3
"""Codebase-agnostic Nsight Systems SQLite profile analyzer."""

from __future__ import annotations

import argparse
import contextlib
import io
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

NS = 1_000_000_000
MS = 1_000_000


@dataclass
class ProfileSummary:
    path: Path
    kernel_span_ms: float
    kernel_count: int
    union_busy_ms: float
    union_util_pct: float
    largest_gap_ms: float
    tiny_d2h_count: int
    tiny_d2h_bytes: int
    sync_count: int
    sync_runtime_ms: float


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (name,),
    ).fetchone()
    return row is not None


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"pragma table_info({table})")}


def string_map(conn: sqlite3.Connection) -> dict[int, str]:
    if not table_exists(conn, "StringIds"):
        return {}
    return {int(row[0]): row[1] for row in conn.execute("select id, value from StringIds")}


def enum_map(conn: sqlite3.Connection, table: str) -> dict[int, str]:
    if not table_exists(conn, table):
        return {}
    cols = table_columns(conn, table)
    label_col = "label" if "label" in cols else "name" if "name" in cols else None
    if label_col is None:
        return {}
    return {int(row[0]): row[1] for row in conn.execute(f"select id, {label_col} from {table}")}


def resolve_sqlite(path: Path) -> Path:
    if path.suffix == ".sqlite":
        return path
    if path.name.endswith(".nsys-rep"):
        sibling = path.with_suffix("").with_suffix(".sqlite")
        if sibling.exists():
            return sibling
        # Path.with_suffix strips only ".rep"; handle the common ".nsys-rep" suffix.
        sibling = Path(str(path).removesuffix(".nsys-rep") + ".sqlite")
        if sibling.exists():
            return sibling
        raise FileNotFoundError(
            f"No sibling SQLite export for {path}. Run: "
            f"nsys export --type sqlite --force-overwrite=true "
            f"--output {sibling} {path}"
        )
    return path


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        elif end > merged[-1][1]:
            merged[-1][1] = end
    return [(start, end) for start, end in merged]


def ms(value_ns: int | float | None) -> float:
    if value_ns is None:
        return 0.0
    return float(value_ns) / MS


def fmt_ms(value: float) -> str:
    return f"{value:.3f}"


def kernel_name(strings: dict[int, str], row: sqlite3.Row) -> str:
    for col in ("shortName", "demangledName", "mangledName"):
        if col in row.keys() and row[col] is not None:
            return strings.get(int(row[col]), str(row[col]))
    return "unknown"


def runtime_name(strings: dict[int, str], name_id: int | None) -> str:
    if name_id is None:
        return "unknown"
    return strings.get(int(name_id), str(name_id))


def nvtx_label(strings: dict[int, str], row: sqlite3.Row) -> str:
    if "text" in row.keys() and row["text"]:
        return str(row["text"])
    if "textId" in row.keys() and row["textId"] is not None:
        return strings.get(int(row["textId"]), str(row["textId"]))
    return "unknown"


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def summarize_profile(path: Path, args: argparse.Namespace) -> ProfileSummary:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    strings = string_map(conn)
    copy_kinds = enum_map(conn, "ENUM_CUDA_MEMCPY_OPER")
    sync_types = enum_map(conn, "ENUM_CUDA_SYNC_TYPE")

    print(f"\n## Profile: `{path}`\n")

    if not table_exists(conn, "CUPTI_ACTIVITY_KIND_KERNEL"):
        raise RuntimeError(f"{path} has no CUPTI_ACTIVITY_KIND_KERNEL table")

    kernel_bounds = conn.execute(
        "select min(start) start, max(end) end, count(*) count "
        "from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    kernel_start = int(kernel_bounds["start"] or 0)
    kernel_end = int(kernel_bounds["end"] or 0)
    kernel_count = int(kernel_bounds["count"] or 0)
    kernel_span_ms = ms(kernel_end - kernel_start)

    intervals = [
        (int(row["start"]), int(row["end"]))
        for row in conn.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL")
    ]
    merged = merge_intervals(intervals)
    union_busy_ms = ms(sum(end - start for start, end in merged))
    union_util_pct = 100.0 * union_busy_ms / kernel_span_ms if kernel_span_ms else 0.0

    device_rows = []
    for row in conn.execute(
        "select deviceId, min(start) start, max(end) end, count(*) count "
        "from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId order by deviceId"
    ):
        dev_intervals = [
            (int(item["start"]), int(item["end"]))
            for item in conn.execute(
                "select start, end from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=?",
                (row["deviceId"],),
            )
        ]
        dev_merged = merge_intervals(dev_intervals)
        span = ms(int(row["end"]) - int(row["start"]))
        busy = ms(sum(end - start for start, end in dev_merged))
        util = 100.0 * busy / span if span else 0.0
        device_rows.append(
            [
                str(row["deviceId"]),
                str(row["count"]),
                fmt_ms(span),
                fmt_ms(busy),
                f"{util:.1f}%",
            ]
        )

    print("### High-Level")
    print_table(
        ["kernel span ms", "kernel count", "all-GPU busy ms", "all-GPU util"],
        [[fmt_ms(kernel_span_ms), str(kernel_count), fmt_ms(union_busy_ms), f"{union_util_pct:.1f}%"]],
    )
    if device_rows:
        print("\n### Per-Device Kernel Utilization")
        print_table(["device", "kernels", "span ms", "busy ms", "util"], device_rows)

    gap_rows = []
    gaps: list[tuple[int, int, int]] = []
    for (_, prev_end), (next_start, _) in zip(merged, merged[1:]):
        if next_start > prev_end:
            gaps.append((next_start - prev_end, prev_end, next_start))
    gaps.sort(reverse=True)
    for gap_ns, gap_start, gap_end in gaps[: args.top_gaps]:
        prev_kernel = conn.execute(
            "select * from CUPTI_ACTIVITY_KIND_KERNEL "
            "where end<=? order by end desc limit 1",
            (gap_start,),
        ).fetchone()
        next_kernel = conn.execute(
            "select * from CUPTI_ACTIVITY_KIND_KERNEL "
            "where start>=? order by start asc limit 1",
            (gap_end,),
        ).fetchone()
        runtime_rows = []
        if table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
            runtime_rows = conn.execute(
                "select nameId, count(*) count, sum(end-start) total, max(end-start) max_time "
                "from CUPTI_ACTIVITY_KIND_RUNTIME "
                "where start>=? and end<=? "
                "group by nameId order by total desc limit ?",
                (gap_start, gap_end, args.top_runtime_in_gap),
            ).fetchall()
        runtime_summary = "; ".join(
            f"{runtime_name(strings, row['nameId'])} c={row['count']} "
            f"total_ms={fmt_ms(ms(row['total']))}"
            for row in runtime_rows
        ) or "none"
        gap_rows.append(
            [
                fmt_ms(ms(gap_ns)),
                fmt_ms(ms(gap_start - kernel_start)),
                kernel_name(strings, prev_kernel) if prev_kernel else "?",
                kernel_name(strings, next_kernel) if next_kernel else "?",
                runtime_summary,
            ]
        )
    largest_gap_ms = ms(gaps[0][0]) if gaps else 0.0
    print("\n### Largest All-GPU Idle Gaps")
    print_table(
        ["gap ms", "relative start ms", "previous kernel", "next kernel", "CUDA runtime inside gap"],
        gap_rows,
    )

    print("\n### Top Kernels By Total Time")
    kernel_rows = []
    for row in conn.execute(
        "select shortName, count(*) count, sum(end-start) total, "
        "avg(end-start) avg_time, max(end-start) max_time "
        "from CUPTI_ACTIVITY_KIND_KERNEL "
        "group by shortName order by total desc limit ?",
        (args.top,),
    ):
        kernel_rows.append(
            [
                strings.get(int(row["shortName"]), str(row["shortName"])),
                str(row["count"]),
                fmt_ms(ms(row["total"])),
                fmt_ms(ms(row["avg_time"])),
                fmt_ms(ms(row["max_time"])),
            ]
        )
    print_table(["kernel", "count", "total ms", "avg ms", "max ms"], kernel_rows)

    sync_count = 0
    sync_runtime_ms = 0.0
    if table_exists(conn, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        print("\n### Top CUDA Runtime APIs")
        runtime_rows = []
        for row in conn.execute(
            "select nameId, count(*) count, sum(end-start) total, "
            "avg(end-start) avg_time, max(end-start) max_time "
            "from CUPTI_ACTIVITY_KIND_RUNTIME "
            "group by nameId order by total desc limit ?",
            (args.top,),
        ):
            name = runtime_name(strings, row["nameId"])
            runtime_rows.append(
                [
                    name,
                    str(row["count"]),
                    fmt_ms(ms(row["total"])),
                    fmt_ms(ms(row["avg_time"])),
                    fmt_ms(ms(row["max_time"])),
                ]
            )
        print_table(["runtime API", "count", "total ms", "avg ms", "max ms"], runtime_rows)

        sync_like = conn.execute(
            "select nameId, count(*) count, sum(end-start) total "
            "from CUPTI_ACTIVITY_KIND_RUNTIME group by nameId"
        ).fetchall()
        for row in sync_like:
            name = runtime_name(strings, row["nameId"]).lower()
            if "synchronize" in name or "streamwaitevent" in name:
                sync_count += int(row["count"])
                sync_runtime_ms += ms(row["total"])

    tiny_d2h_count = 0
    tiny_d2h_bytes = 0
    if table_exists(conn, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        print("\n### Memcpy By Direction")
        memcpy_rows = []
        for row in conn.execute(
            "select copyKind, count(*) count, sum(bytes) bytes, "
            "sum(end-start) total, avg(bytes) avg_bytes, max(bytes) max_bytes "
            "from CUPTI_ACTIVITY_KIND_MEMCPY "
            "group by copyKind order by total desc"
        ):
            kind = copy_kinds.get(int(row["copyKind"]), str(row["copyKind"]))
            memcpy_rows.append(
                [
                    kind,
                    str(row["count"]),
                    str(row["bytes"]),
                    f"{float(row['avg_bytes']):.1f}",
                    str(row["max_bytes"]),
                    fmt_ms(ms(row["total"])),
                ]
            )
        print_table(["direction", "count", "bytes", "avg bytes", "max bytes", "total ms"], memcpy_rows)

        print("\n### Memcpy Size Distribution")
        dist_rows = []
        for row in conn.execute(
            "select copyKind, bytes, count(*) count, sum(end-start) total "
            "from CUPTI_ACTIVITY_KIND_MEMCPY "
            "group by copyKind, bytes "
            "order by copyKind, count desc, bytes limit ?",
            (args.top_memcpy_sizes,),
        ):
            kind = copy_kinds.get(int(row["copyKind"]), str(row["copyKind"]))
            dist_rows.append(
                [kind, str(row["bytes"]), str(row["count"]), fmt_ms(ms(row["total"]))]
            )
        print_table(["direction", "bytes per copy", "count", "total ms"], dist_rows)

        for row in conn.execute(
            "select copyKind, bytes, count(*) count, sum(bytes) total_bytes "
            "from CUPTI_ACTIVITY_KIND_MEMCPY "
            "where bytes<=? group by copyKind, bytes",
            (args.tiny_copy_bytes,),
        ):
            kind = copy_kinds.get(int(row["copyKind"]), str(row["copyKind"]))
            if kind == "Device-to-Host":
                tiny_d2h_count += int(row["count"])
                tiny_d2h_bytes += int(row["total_bytes"])

    if table_exists(conn, "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION"):
        print("\n### CUDA Synchronization Events")
        rows = conn.execute(
            "select syncType, count(*) count, sum(end-start) total, "
            "avg(end-start) avg_time, max(end-start) max_time "
            "from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION "
            "group by syncType order by total desc limit ?",
            (args.top,),
        ).fetchall()
        if rows:
            sync_rows = []
            for row in rows:
                kind = sync_types.get(int(row["syncType"]), str(row["syncType"]))
                sync_rows.append(
                    [
                        kind,
                        str(row["count"]),
                        fmt_ms(ms(row["total"])),
                        fmt_ms(ms(row["avg_time"])),
                        fmt_ms(ms(row["max_time"])),
                    ]
                )
            print_table(["sync type", "count", "total ms", "avg ms", "max ms"], sync_rows)
        else:
            print("No rows in `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION`.")

    if table_exists(conn, "NVTX_EVENTS"):
        rows = conn.execute(
            "select * from NVTX_EVENTS where end is not null limit 1"
        ).fetchall()
        if rows:
            print("\n### Top NVTX Ranges")
            nvtx_rows = []
            for row in conn.execute(
                "select text, textId, count(*) count, sum(end-start) total, max(end-start) max_time "
                "from NVTX_EVENTS where end is not null "
                "group by text, textId order by total desc limit ?",
                (args.top,),
            ):
                nvtx_rows.append(
                    [
                        nvtx_label(strings, row),
                        str(row["count"]),
                        fmt_ms(ms(row["total"])),
                        fmt_ms(ms(row["max_time"])),
                    ]
                )
            print_table(["NVTX label", "count", "total ms", "max ms"], nvtx_rows)

    return ProfileSummary(
        path=path,
        kernel_span_ms=kernel_span_ms,
        kernel_count=kernel_count,
        union_busy_ms=union_busy_ms,
        union_util_pct=union_util_pct,
        largest_gap_ms=largest_gap_ms,
        tiny_d2h_count=tiny_d2h_count,
        tiny_d2h_bytes=tiny_d2h_bytes,
        sync_count=sync_count,
        sync_runtime_ms=sync_runtime_ms,
    )


def print_profile_summary(summaries: list[ProfileSummary]) -> None:
    print("# Nsight Profile Analysis\n")
    if len(summaries) == 1:
        print("## Profile Summary")
    else:
        print("## Cross-Profile Summary")

    rows = []
    for item in summaries:
        rows.append(
            [
                item.path.name,
                fmt_ms(item.kernel_span_ms),
                str(item.kernel_count),
                fmt_ms(item.union_busy_ms),
                f"{item.union_util_pct:.1f}%",
                fmt_ms(item.largest_gap_ms),
                str(item.tiny_d2h_count),
                str(item.tiny_d2h_bytes),
                str(item.sync_count),
                fmt_ms(item.sync_runtime_ms),
            ]
        )
    print_table(
        [
            "profile",
            "kernel span ms",
            "kernels",
            "all-GPU busy ms",
            "util",
            "largest gap ms",
            "tiny D2H copies",
            "tiny D2H bytes",
            "sync-like runtime calls",
            "sync-like runtime ms",
        ],
        rows,
    )

    max_gap = max((item.largest_gap_ms for item in summaries), default=0.0)
    total_tiny_d2h = sum(item.tiny_d2h_count for item in summaries)
    total_sync = sum(item.sync_count for item in summaries)
    min_util = min((item.union_util_pct for item in summaries), default=0.0)
    print("\n## Generic TLDR Signals")
    scope = "in the profile" if len(summaries) == 1 else "across all profiles"
    signals = []
    if max_gap >= 10.0:
        signals.append(f"Large all-GPU idle gaps are present; largest observed gap is {max_gap:.1f} ms.")
    if min_util and min_util < 50.0:
        if len(summaries) == 1:
            signals.append(f"The profile has low all-GPU union utilization ({min_util:.1f}%).")
        else:
            signals.append(f"At least one profile has low all-GPU union utilization ({min_util:.1f}%).")
    if total_tiny_d2h:
        signals.append(
            f"Tiny device-to-host copies are present ({total_tiny_d2h} copies "
            f"of <= tiny-copy threshold {scope})."
        )
    if total_sync:
        signals.append(f"Synchronization-like CUDA runtime calls are present ({total_sync} total {scope}).")
    if not signals:
        signals.append("No generic high-severity signal crossed the default thresholds.")
    for index, signal in enumerate(signals, 1):
        print(f"{index}. {signal}")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path, help=".sqlite or .nsys-rep paths")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--top-gaps", type=int, default=8)
    parser.add_argument("--top-runtime-in-gap", type=int, default=5)
    parser.add_argument("--top-memcpy-sizes", type=int, default=24)
    parser.add_argument("--tiny-copy-bytes", type=int, default=8)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    sqlite_paths = [resolve_sqlite(path) for path in args.profiles]
    summaries = []
    detail_sections = []
    for path in sqlite_paths:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summaries.append(summarize_profile(path, args))
        detail_sections.append(buffer.getvalue())
    print_profile_summary(summaries)
    if len(summaries) == 1:
        print("\n## Profile Details")
    else:
        print("\n## Per-Profile Details")
    for section in detail_sections:
        print(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
