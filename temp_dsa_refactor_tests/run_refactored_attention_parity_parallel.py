#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


TEST_FILE = "tests/model_executor/models/test_nemotron_h_dsa_refactored_attention.py"
COMPONENT_TEST_FILE = (
    "tests/model_executor/models/test_nemotron_h_chunked_dsa_components.py"
)
COUNT_RE = re.compile(
    r"(?P<count>\d+) (?P<kind>passed|failed|skipped|deselected|"
    r"xfailed|xpassed|error|errors)"
)


@dataclasses.dataclass
class ShardResult:
    label: str
    shard_index: int
    nodeids: list[str]
    log_path: str
    junit_path: str
    returncode: int
    elapsed_s: float
    counts: dict[str, int]
    failures: list[dict[str, str]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--repo-dir", default=os.getcwd())
    parser.add_argument(
        "--output-dir",
        default="temp_dsa_refactor_tests/parallel-refactor-parity",
    )
    parser.add_argument("--pytorch-workers", type=int, default=8)
    parser.add_argument("--efficient-workers", type=int, default=2)
    parser.add_argument("--pytorch-k", default="pytorch")
    parser.add_argument("--efficient-k", default="efficient")
    parser.add_argument("--limit-pytorch-nodeids", type=int, default=0)
    parser.add_argument("--limit-efficient-nodeids", type=int, default=0)
    parser.add_argument("--skip-components", action="store_true")
    parser.add_argument("--skip-pytorch", action="store_true")
    parser.add_argument("--skip-efficient", action="store_true")
    return parser.parse_args()


def _unique_output_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find unique output directory for {path}")


def _base_env(output_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0")
    env.setdefault("XDG_CACHE_HOME", str(output_dir / "cache"))
    return env


def _parse_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for match in COUNT_RE.finditer(output):
        kind = match.group("kind")
        if kind == "errors":
            kind = "error"
        counts[kind] = counts.get(kind, 0) + int(match.group("count"))
    return counts


def _xml_tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_junit_failures(junit_path: Path) -> list[dict[str, str]]:
    if not junit_path.exists():
        return []
    failures: list[dict[str, str]] = []
    try:
        root = ET.parse(junit_path).getroot()
    except ET.ParseError as exc:
        return [{
            "nodeid": str(junit_path),
            "kind": "junit-parse-error",
            "message": str(exc),
            "text": "",
        }]

    for testcase in root.iter():
        if _xml_tag_name(testcase.tag) != "testcase":
            continue
        classname = testcase.attrib.get("classname", "")
        name = testcase.attrib.get("name", "")
        file_name = testcase.attrib.get("file", "")
        line = testcase.attrib.get("line", "")
        nodeid = f"{classname}::{name}" if classname else name
        for child in testcase:
            kind = _xml_tag_name(child.tag)
            if kind not in {"failure", "error"}:
                continue
            failures.append({
                "nodeid": nodeid,
                "kind": kind,
                "message": child.attrib.get("message", ""),
                "type": child.attrib.get("type", ""),
                "file": file_name,
                "line": line,
                "text": child.text or "",
            })
    return failures


def _run_command(
    *,
    label: str,
    shard_index: int,
    command: list[str],
    repo_dir: Path,
    output_dir: Path,
    env: dict[str, str],
    nodeids: list[str],
) -> ShardResult:
    log_path = output_dir / f"{label}-shard-{shard_index:02d}.log"
    junit_path = output_dir / f"{label}-shard-{shard_index:02d}.junit.xml"
    start = time.monotonic()
    completed = subprocess.run(
        [*command, f"--junitxml={junit_path}"],
        cwd=repo_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed_s = time.monotonic() - start
    log_path.write_text(completed.stdout)
    return ShardResult(
        label=label,
        shard_index=shard_index,
        nodeids=nodeids,
        log_path=str(log_path),
        junit_path=str(junit_path),
        returncode=completed.returncode,
        elapsed_s=elapsed_s,
        counts=_parse_counts(completed.stdout),
        failures=_parse_junit_failures(junit_path),
    )


def _collect_nodeids(
    *,
    python: str,
    repo_dir: Path,
    test_file: str,
    selector: str,
    output_dir: Path,
    env: dict[str, str],
) -> list[str]:
    command = [
        python,
        "-m",
        "pytest",
        test_file,
        "--collect-only",
        "-q",
        "-k",
        selector,
    ]
    completed = subprocess.run(
        command,
        cwd=repo_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    collect_log = output_dir / f"collect-{selector.replace(' ', '_')}.log"
    collect_log.write_text(completed.stdout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"pytest collection failed for -k {selector!r}; see {collect_log}"
        )
    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if "::" in line and not line.startswith("=")
    ]


def _split_round_robin(nodeids: list[str], num_shards: int) -> list[list[str]]:
    shards = [[] for _ in range(num_shards)]
    for index, nodeid in enumerate(nodeids):
        shards[index % num_shards].append(nodeid)
    return [shard for shard in shards if shard]


def _run_shards(
    *,
    label: str,
    python: str,
    repo_dir: Path,
    output_dir: Path,
    env: dict[str, str],
    nodeids: list[str],
    workers: int,
) -> list[ShardResult]:
    shards = _split_round_robin(nodeids, max(1, workers))
    results: list[ShardResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = []
        for shard_index, shard in enumerate(shards):
            command = [python, "-m", "pytest", "-q", *shard]
            futures.append(
                pool.submit(
                    _run_command,
                    label=label,
                    shard_index=shard_index,
                    command=command,
                    repo_dir=repo_dir,
                    output_dir=output_dir,
                    env=env,
                    nodeids=shard,
                )
            )
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: result.shard_index)


def _aggregate(results: list[ShardResult]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for result in results:
        for kind, count in result.counts.items():
            counts[kind] = counts.get(kind, 0) + count
    return {
        "shards": len(results),
        "nodeids": sum(len(result.nodeids) for result in results),
        "elapsed_s_max_shard": max(
            (result.elapsed_s for result in results), default=0.0),
        "elapsed_s_sum_shards": sum(result.elapsed_s for result in results),
        "returncodes": [result.returncode for result in results],
        "counts": counts,
        "logs": [result.log_path for result in results],
        "junit_logs": [result.junit_path for result in results],
        "failed_shards": [
            {
                "label": result.label,
                "shard_index": result.shard_index,
                "returncode": result.returncode,
                "log_path": result.log_path,
                "junit_path": result.junit_path,
                "nodeids": result.nodeids,
                "failures": result.failures,
            }
            for result in results
            if result.returncode != 0 or result.failures
        ],
    }


def _write_failure_reports(
    *,
    output_dir: Path,
    results: list[ShardResult],
) -> dict[str, object]:
    failures_dir = output_dir / "failures"
    failures_dir.mkdir(exist_ok=True)
    failed_results = [
        result for result in results
        if result.returncode != 0 or result.failures
    ]
    failed_tests = [
        {
            "label": result.label,
            "shard_index": result.shard_index,
            "log_path": result.log_path,
            "junit_path": result.junit_path,
            **failure,
        }
        for result in failed_results
        for failure in result.failures
    ]
    failed_shards = []
    for result in failed_results:
        central_log = failures_dir / Path(result.log_path).name
        central_log.write_text(Path(result.log_path).read_text())
        failed_shards.append({
            "label": result.label,
            "shard_index": result.shard_index,
            "returncode": result.returncode,
            "log_path": result.log_path,
            "central_log_path": str(central_log),
            "junit_path": result.junit_path,
            "nodeids": result.nodeids,
            "failures": result.failures,
        })

    failures_json = failures_dir / "failures.json"
    failures_json.write_text(json.dumps({
        "failed_shards": failed_shards,
        "failed_tests": failed_tests,
    }, indent=2, sort_keys=True))

    failures_txt = failures_dir / "failures.txt"
    if not failed_results:
        failures_txt.write_text("No failed shards.\n")
    else:
        lines = [
            f"failed_shards={len(failed_results)}",
            f"failed_tests={len(failed_tests)}",
            "",
        ]
        for shard in failed_shards:
            lines.append(
                f"[{shard['label']} shard {shard['shard_index']}] "
                f"returncode={shard['returncode']}"
            )
            lines.append(f"log={shard['central_log_path']}")
            if not shard["failures"]:
                lines.append("failed_tests=<not reported by junit>")
            for failure in shard["failures"]:
                location = failure.get("file", "")
                if failure.get("line"):
                    location = f"{location}:{failure['line']}"
                lines.append(
                    f"- {failure['kind']} {failure['nodeid']} {location}"
                )
                message = failure.get("message", "")
                if message:
                    lines.append(f"  {message}")
            lines.append("")
        failures_txt.write_text("\n".join(lines))

    return {
        "dir": str(failures_dir),
        "json": str(failures_json),
        "text": str(failures_txt),
        "failed_shards": len(failed_results),
        "failed_tests": len(failed_tests),
    }


def _print_section(name: str, aggregate: dict[str, object]) -> None:
    counts = aggregate["counts"]
    counts_text = ", ".join(
        f"{kind}={count}" for kind, count in sorted(counts.items())
    )
    print(
        f"{name}: shards={aggregate['shards']} "
        f"nodeids={aggregate['nodeids']} "
        f"max_shard_s={aggregate['elapsed_s_max_shard']:.2f} "
        f"sum_shard_s={aggregate['elapsed_s_sum_shards']:.2f} "
        f"returncodes={aggregate['returncodes']} "
        f"{counts_text}"
    )


def main() -> int:
    args = _parse_args()
    repo_dir = Path(args.repo_dir).resolve()
    output_dir = _unique_output_dir((repo_dir / args.output_dir).resolve())
    output_dir.mkdir(parents=True)
    env = _base_env(output_dir)
    start = time.monotonic()
    summary: dict[str, object] = {
        "repo_dir": str(repo_dir),
        "output_dir": str(output_dir),
        "pytorch_workers": args.pytorch_workers,
        "efficient_workers": args.efficient_workers,
        "sections": {},
    }
    all_results: list[ShardResult] = []

    if not args.skip_components:
        component_result = _run_command(
            label="components",
            shard_index=0,
            command=[args.python, "-m", "pytest", "-q", COMPONENT_TEST_FILE],
            repo_dir=repo_dir,
            output_dir=output_dir,
            env=env,
            nodeids=[COMPONENT_TEST_FILE],
        )
        all_results.append(component_result)
        summary["sections"]["components"] = _aggregate([component_result])
        _print_section("components", summary["sections"]["components"])

    if not args.skip_pytorch:
        pytorch_nodeids = _collect_nodeids(
            python=args.python,
            repo_dir=repo_dir,
            test_file=TEST_FILE,
            selector=args.pytorch_k,
            output_dir=output_dir,
            env=env,
        )
        if args.limit_pytorch_nodeids > 0:
            pytorch_nodeids = pytorch_nodeids[:args.limit_pytorch_nodeids]
        pytorch_results = _run_shards(
            label="pytorch",
            python=args.python,
            repo_dir=repo_dir,
            output_dir=output_dir,
            env=env,
            nodeids=pytorch_nodeids,
            workers=args.pytorch_workers,
        )
        all_results.extend(pytorch_results)
        summary["sections"]["pytorch"] = _aggregate(pytorch_results)
        _print_section("pytorch", summary["sections"]["pytorch"])

    if not args.skip_efficient:
        efficient_nodeids = _collect_nodeids(
            python=args.python,
            repo_dir=repo_dir,
            test_file=TEST_FILE,
            selector=args.efficient_k,
            output_dir=output_dir,
            env=env,
        )
        if args.limit_efficient_nodeids > 0:
            efficient_nodeids = efficient_nodeids[:args.limit_efficient_nodeids]
        efficient_results = _run_shards(
            label="efficient",
            python=args.python,
            repo_dir=repo_dir,
            output_dir=output_dir,
            env=env,
            nodeids=efficient_nodeids,
            workers=args.efficient_workers,
        )
        all_results.extend(efficient_results)
        summary["sections"]["efficient"] = _aggregate(efficient_results)
        _print_section("efficient", summary["sections"]["efficient"])

    summary["elapsed_s_total"] = time.monotonic() - start
    summary["returncode"] = 0 if all(
        result.returncode == 0 for result in all_results) else 1
    summary["failure_report"] = _write_failure_reports(
        output_dir=output_dir,
        results=all_results,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary_json={summary_path}")
    print(f"failure_report={summary['failure_report']['text']}")
    print(f"total_elapsed_s={summary['elapsed_s_total']:.2f}")
    return int(summary["returncode"])


if __name__ == "__main__":
    sys.exit(main())
