# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Toggle vLLM profiler endpoints for short wall-clock capture windows."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProfileEvent:
    event: str
    window: int
    monotonic_s: float
    status: int | None
    ok: bool
    error: str | None = None


def post_empty(url: str, timeout_s: float) -> tuple[int, str]:
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")


def record_event(
    events: list[ProfileEvent],
    *,
    event: str,
    window: int,
    status: int | None,
    ok: bool,
    error: str | None = None,
) -> None:
    item = ProfileEvent(
        event=event,
        window=window,
        monotonic_s=time.monotonic(),
        status=status,
        ok=ok,
        error=error,
    )
    events.append(item)
    status_text = "none" if status is None else str(status)
    print(
        f"[profile-window] {event} window={window} "
        f"status={status_text} ok={ok}",
        flush=True,
    )
    if error:
        print(f"[profile-window] error: {error}", file=sys.stderr, flush=True)


def toggle(
    *,
    base_url: str,
    action: str,
    window: int,
    timeout_s: float,
    events: list[ProfileEvent],
) -> bool:
    url = f"{base_url.rstrip('/')}/{action}_profile"
    try:
        status, text = post_empty(url, timeout_s)
        ok = 200 <= status < 300
        record_event(
            events,
            event=action,
            window=window,
            status=status,
            ok=ok,
            error=None if ok else text[:1000],
        )
        return ok
    except Exception as exc:
        record_event(
            events,
            event=action,
            window=window,
            status=None,
            ok=False,
            error=repr(exc),
        )
        return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    events: list[ProfileEvent] = []
    base_url = args.base_url.rstrip("/")
    start_time = time.monotonic()
    print(
        "[profile-window] "
        f"base_url={base_url} windows={args.windows} "
        f"window_seconds={args.window_seconds} gap_seconds={args.gap_seconds} "
        f"initial_delay_seconds={args.initial_delay_seconds}",
        flush=True,
    )

    if args.initial_delay_seconds > 0:
        time.sleep(args.initial_delay_seconds)

    for window in range(args.windows):
        started = toggle(
            base_url=base_url,
            action="start",
            window=window,
            timeout_s=args.timeout_s,
            events=events,
        )
        try:
            time.sleep(args.window_seconds)
        finally:
            stopped = toggle(
                base_url=base_url,
                action="stop",
                window=window,
                timeout_s=args.timeout_s,
                events=events,
            )
        if args.strict and (not started or not stopped):
            break
        if window != args.windows - 1 and args.gap_seconds > 0:
            time.sleep(args.gap_seconds)

    summary = {
        "base_url": base_url,
        "windows": args.windows,
        "window_seconds": args.window_seconds,
        "gap_seconds": args.gap_seconds,
        "initial_delay_seconds": args.initial_delay_seconds,
        "duration_s": time.monotonic() - start_time,
        "ok": all(event.ok for event in events)
        and len(events) == args.windows * 2,
        "events": [asdict(event) for event in events],
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"[profile-window] wrote {output_path}", flush=True)
    return summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    parser.add_argument("--gap-seconds", type=float, default=30.0)
    parser.add_argument("--initial-delay-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Continue even if one profile endpoint request fails.",
    )
    parser.set_defaults(strict=True)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if args.windows < 1:
        raise ValueError("--windows must be >= 1")
    if args.window_seconds <= 0:
        raise ValueError("--window-seconds must be > 0")
    if args.gap_seconds < 0:
        raise ValueError("--gap-seconds must be >= 0")
    if args.initial_delay_seconds < 0:
        raise ValueError("--initial-delay-seconds must be >= 0")
    summary = run(args)
    return 0 if summary["ok"] or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
