#!/usr/bin/env python
"""Extract CUDA kernel time/counts inside named NVTX ranges from an nsys SQLite DB."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


LABELS = ("repo_dense_core", "repo_summary_core", "batched_cuda_core")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite_db", type=Path)
    parser.add_argument("--labels", nargs="*", default=list(LABELS))
    return parser.parse_args()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type='table' and name=?",
        (name,),
    ).fetchone()
    return row is not None


def columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"pragma table_info({table})")}


def string_id_expr(conn: sqlite3.Connection, table_alias: str, column: str) -> str:
    if table_exists(conn, "StringIds"):
        return (
            f"coalesce((select value from StringIds where id = {table_alias}.{column}), "
            f"cast({table_alias}.{column} as text))"
        )
    if table_exists(conn, "StringIds2"):
        return (
            f"coalesce((select value from StringIds2 where id = {table_alias}.{column}), "
            f"cast({table_alias}.{column} as text))"
        )
    return f"cast({table_alias}.{column} as text)"


def get_nvtx_ranges(conn: sqlite3.Connection, labels: list[str]):
    if not table_exists(conn, "NVTX_EVENTS"):
        raise RuntimeError("NVTX_EVENTS table not found in nsys SQLite export")
    cols = columns(conn, "NVTX_EVENTS")
    text_col = "text" if "text" in cols else "textId"
    name_expr = (
        "n.text"
        if text_col == "text"
        else string_id_expr(conn, "n", text_col)
    )
    rows = conn.execute(
        f"""
        select {name_expr} as name, n.start, n.end
        from NVTX_EVENTS n
        where n.end is not null
        """
    ).fetchall()
    by_label = {label: [] for label in labels}
    for name, start, end in rows:
        if name in by_label:
            by_label[name].append((int(start), int(end)))
    return by_label


def get_kernel_rows(conn: sqlite3.Connection):
    candidates = [
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_KERNEL_NAMED",
        "CUDA_KERNEL",
    ]
    table = next((name for name in candidates if table_exists(conn, name)), None)
    if table is None:
        raise RuntimeError("No CUDA kernel activity table found in nsys SQLite export")
    cols = columns(conn, table)
    name_col = "demangledName" if "demangledName" in cols else "shortName"
    if name_col not in cols:
        name_col = "name" if "name" in cols else ""
    name_expr = (
        string_id_expr(conn, "k", name_col)
        if name_col and name_col != "name"
        else ("k.name" if name_col == "name" else "'<unknown>'")
    )
    return conn.execute(
        f"""
        select {name_expr} as name, k.start, k.end
        from {table} k
        where k.end is not null
        """
    ).fetchall()


def main() -> None:
    args = parse_args()
    conn = sqlite3.connect(str(args.sqlite_db))
    ranges = get_nvtx_ranges(conn, args.labels)
    kernels = get_kernel_rows(conn)

    for label in args.labels:
        selected = []
        for range_start, range_end in ranges[label]:
            for name, start, end in kernels:
                start = int(start)
                end = int(end)
                if start >= range_start and end <= range_end:
                    selected.append((name, start, end))
        total_ns = sum(end - start for _, start, end in selected)
        unique_names = sorted({name for name, _, _ in selected})
        print(
            f"NSYS label={label} range_count={len(ranges[label])} "
            f"kernel_time_us={total_ns / 1000.0:.3f} "
            f"kernel_count={len(selected)} "
            f"unique_kernel_count={len(unique_names)}"
        )
        for name in unique_names[:20]:
            print(f"  KERNEL {label}: {name}")
        if len(unique_names) > 20:
            print(f"  KERNEL {label}: ... {len(unique_names) - 20} more")


if __name__ == "__main__":
    main()
