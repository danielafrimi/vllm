# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-token HTTP serving benchmark.

This benchmark sends OpenAI completions requests with token-id prompts so the
input length is exactly controlled by the request payload. It forces the output
length with ``max_tokens == min_tokens`` and ``ignore_eos`` and validates the
server-reported usage for every completed request.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

Payload = dict[str, Any] | bytes


@dataclass
class RequestResult:
    request_id: int
    success: bool
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
    http_status: int | None = None
    error: str | None = None


def parse_headers(items: list[str] | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Invalid header {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        headers[key.strip()] = value.strip()
    return headers


def parse_json_object(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--extra-body must parse to a JSON object")
    return parsed


def load_prompt_token_ids_jsonl(
    path: str,
    *,
    num_prompts: int,
    input_len: int,
) -> list[list[int]]:
    prompts: list[list[int]] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                token_ids = record.get("prompt_token_ids")
                if token_ids is None and isinstance(record.get("prompt"), list):
                    token_ids = record["prompt"]
            elif isinstance(record, list):
                token_ids = record
            else:
                token_ids = None

            if not isinstance(token_ids, list):
                raise ValueError(
                    f"{path}:{line_number} must contain a prompt_token_ids list"
                )
            if len(token_ids) != input_len:
                raise ValueError(
                    f"{path}:{line_number} has {len(token_ids)} prompt tokens; "
                    f"expected {input_len}"
                )
            invalid_token_id = any(
                not isinstance(token_id, int) or token_id < 0
                for token_id in token_ids
            )
            if invalid_token_id:
                raise ValueError(
                    f"{path}:{line_number} prompt_token_ids must be non-negative ints"
                )
            prompts.append(token_ids)
            if len(prompts) == num_prompts:
                break

    if len(prompts) < num_prompts:
        raise ValueError(
            f"{path} contains {len(prompts)} usable prompts; expected {num_prompts}"
        )
    return prompts


def make_prompt_token_ids(
    request_id: int,
    input_len: int,
    token_id_start: int,
    token_id_range: int,
) -> list[int]:
    return [
        token_id_start + ((request_id * input_len + offset) % token_id_range)
        for offset in range(input_len)
    ]


def build_constant_prompt_payload_bytes(
    *,
    model: str,
    token_id: int,
    input_len: int,
    output_len: int,
    temperature: float,
    top_p: float | None,
    seed: int | None,
    return_token_ids: bool,
    add_special_tokens: bool,
    extra_body: dict[str, Any],
) -> bytes:
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": output_len,
        "min_tokens": output_len,
        "ignore_eos": True,
        "temperature": temperature,
        "stream": False,
        "add_special_tokens": add_special_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if seed is not None:
        payload["seed"] = seed
    if return_token_ids:
        payload["return_token_ids"] = True
    payload.update(extra_body)

    token = str(token_id)
    prompt = token if input_len == 1 else (token + ",") * (input_len - 1) + token
    items = [f"{json.dumps(key)}:{json.dumps(value)}" for key, value in payload.items()]
    items.append(f'"prompt":[{prompt}]')
    return ("{" + ",".join(items) + "}").encode("utf-8")


def build_payload(
    *,
    model: str,
    request_id: int,
    input_len: int,
    output_len: int,
    prompt_token_ids: list[int] | None,
    token_id_start: int,
    token_id_range: int,
    temperature: float,
    top_p: float | None,
    seed: int | None,
    return_token_ids: bool,
    add_special_tokens: bool,
    extra_body: dict[str, Any],
) -> dict[str, Any]:
    if prompt_token_ids is None:
        prompt_token_ids = make_prompt_token_ids(
            request_id=request_id,
            input_len=input_len,
            token_id_start=token_id_start,
            token_id_range=token_id_range,
        )
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": output_len,
        "min_tokens": output_len,
        "ignore_eos": True,
        "temperature": temperature,
        "stream": False,
        "add_special_tokens": add_special_tokens,
    }
    if top_p is not None:
        payload["top_p"] = top_p
    if seed is not None:
        payload["seed"] = seed + request_id
    if return_token_ids:
        payload["return_token_ids"] = True
    payload.update(extra_body)
    return payload


def post_json(
    url: str,
    base_url: str,
    headers: dict[str, str],
    payload: Payload | None,
    timeout_s: float,
    insecure: bool,
) -> tuple[int, str]:
    if payload is None:
        data = None
    elif isinstance(payload, bytes):
        data = payload
    else:
        data = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", **headers}
    request = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method="GET" if payload is None else "POST",
    )
    context = None
    if insecure and url.startswith("https://"):
        context = ssl._create_unverified_context()
    try:
        with urllib.request.urlopen(  # noqa: S310
            request, timeout=timeout_s, context=context
        ) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach {base_url}: {exc}") from exc


def fetch_first_model(
    base_url: str,
    headers: dict[str, str],
    timeout_s: float,
    insecure: bool,
) -> str:
    status, text = post_json(
        f"{base_url.rstrip('/')}/v1/models",
        base_url=base_url,
        headers=headers,
        payload=None,
        timeout_s=timeout_s,
        insecure=insecure,
    )
    if status != 200:
        raise RuntimeError(f"Failed to fetch model list: HTTP {status}: {text}")
    data = json.loads(text)
    models = data.get("data") or []
    if not models:
        raise RuntimeError(f"No models found at {base_url}/v1/models")
    return str(models[0]["id"])


def post_completion(
    *,
    api_url: str,
    base_url: str,
    headers: dict[str, str],
    payload: Payload,
    request_id: int,
    timeout_s: float,
    insecure: bool,
) -> RequestResult:
    start = time.perf_counter()
    try:
        status, text = post_json(
            api_url,
            base_url=base_url,
            headers=headers,
            payload=payload,
            timeout_s=timeout_s,
            insecure=insecure,
        )
        latency_s = time.perf_counter() - start
        if status != 200:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_s=latency_s,
                http_status=status,
                error=text[:2000],
            )

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            return RequestResult(
                request_id=request_id,
                success=False,
                latency_s=latency_s,
                http_status=status,
                error=f"Invalid JSON response: {exc}",
            )

        usage = data.get("usage") or {}
        choices = data.get("choices") or []
        finish_reason = None
        if choices:
            finish_reason = choices[0].get("finish_reason")
        return RequestResult(
            request_id=request_id,
            success=True,
            latency_s=latency_s,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            finish_reason=finish_reason,
            http_status=status,
        )
    except Exception as exc:
        return RequestResult(
            request_id=request_id,
            success=False,
            latency_s=time.perf_counter() - start,
            error=repr(exc),
        )


def run_requests(
    *,
    api_url: str,
    base_url: str,
    headers: dict[str, str],
    payloads: list[Payload],
    request_rate: float,
    max_concurrency: int | None,
    timeout_s: float,
    insecure: bool,
) -> tuple[float, list[RequestResult]]:
    max_workers = max_concurrency or len(payloads)
    max_workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        start = time.perf_counter()
        futures = []
        for request_id, payload in enumerate(payloads):
            futures.append(
                executor.submit(
                    post_completion,
                    api_url=api_url,
                    base_url=base_url,
                    headers=headers,
                    payload=payload,
                    request_id=request_id,
                    timeout_s=timeout_s,
                    insecure=insecure,
                )
            )
            if not math.isinf(request_rate) and request_id != len(payloads) - 1:
                time.sleep(1.0 / request_rate)
        results = [future.result() for future in futures]
        duration_s = time.perf_counter() - start
    return duration_s, results


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_results(
    *,
    args: argparse.Namespace,
    model: str,
    duration_s: float,
    results: list[RequestResult],
) -> dict[str, Any]:
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    latencies = [r.latency_s for r in successes]
    prompt_lens = [r.prompt_tokens for r in successes]
    output_lens = [r.completion_tokens for r in successes]

    prompt_lengths_exact = all(length == args.input_len for length in prompt_lens)
    output_lengths_exact = all(length == args.output_len for length in output_lens)
    verified = (
        len(successes) == args.num_prompts
        and not failures
        and prompt_lengths_exact
        and output_lengths_exact
    )

    total_prompt_tokens = sum(length or 0 for length in prompt_lens)
    total_completion_tokens = sum(length or 0 for length in output_lens)
    total_tokens = total_prompt_tokens + total_completion_tokens
    summary: dict[str, Any] = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "base_url": args.base_url,
        "endpoint": args.endpoint,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "request_rate": "inf" if math.isinf(args.request_rate) else args.request_rate,
        "max_concurrency": args.max_concurrency,
        "prompt_token_ids_jsonl": args.prompt_token_ids_jsonl,
        "constant_prompt_token_id": args.constant_prompt_token_id,
        "duration_s": duration_s,
        "completed": len(successes),
        "failed": len(failures),
        "verified": verified,
        "prompt_lengths_exact": prompt_lengths_exact,
        "output_lengths_exact": output_lengths_exact,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "request_throughput": len(successes) / duration_s if duration_s else 0.0,
        "prompt_token_throughput": (
            total_prompt_tokens / duration_s if duration_s else 0.0
        ),
        "completion_token_throughput": (
            total_completion_tokens / duration_s if duration_s else 0.0
        ),
        "total_token_throughput": total_tokens / duration_s if duration_s else 0.0,
        "latency_s": {
            "mean": sum(latencies) / len(latencies) if latencies else 0.0,
            "p50": percentile(latencies, 50),
            "p90": percentile(latencies, 90),
            "p99": percentile(latencies, 99),
            "max": max(latencies) if latencies else 0.0,
        },
        "input_lens": prompt_lens,
        "output_lens": output_lens,
        "errors": [asdict(r) for r in failures[:20]],
    }
    if args.save_detailed:
        summary["requests"] = [asdict(r) for r in results]
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print("=" * 50)
    print("Fixed IO Serving Benchmark Result")
    print("=" * 50)
    print(f"Verified exact lengths:      {summary['verified']}")
    print(f"Successful requests:         {summary['completed']}")
    print(f"Failed requests:             {summary['failed']}")
    print(f"Benchmark duration (s):      {summary['duration_s']:.2f}")
    print(f"Total prompt tokens:         {summary['total_prompt_tokens']}")
    print(f"Total completion tokens:     {summary['total_completion_tokens']}")
    print(f"Request throughput (req/s):  {summary['request_throughput']:.2f}")
    print(f"Prompt throughput (tok/s):   {summary['prompt_token_throughput']:.2f}")
    print(
        "Completion throughput (tok/s): "
        f"{summary['completion_token_throughput']:.2f}"
    )
    print(f"Total throughput (tok/s):    {summary['total_token_throughput']:.2f}")
    print(f"P50 latency (s):             {summary['latency_s']['p50']:.2f}")
    print(f"P99 latency (s):             {summary['latency_s']['p99']:.2f}")
    print(f"Max latency (s):             {summary['latency_s']['max']:.2f}")
    if summary["failed"]:
        print("First errors:")
        for error in summary["errors"][:3]:
            print(f"  request {error['request_id']}: {error['error']}")
    print("=" * 50)


def main(args: argparse.Namespace) -> int:
    if args.input_len < 1:
        raise ValueError("--input-len must be >= 1")
    if args.output_len < 1:
        raise ValueError("--output-len must be >= 1")
    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be >= 1")
    if args.request_rate <= 0:
        raise ValueError("--request-rate must be positive or inf")
    if args.token_id_start < 0:
        raise ValueError("--token-id-start must be >= 0")
    if args.token_id_range < 1:
        raise ValueError("--token-id-range must be >= 1")

    headers = parse_headers(args.header)
    extra_body = parse_json_object(args.extra_body)
    base_url = args.base_url.rstrip("/")
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    api_url = f"{base_url}{endpoint}"

    model = args.model or fetch_first_model(
        base_url=base_url,
        headers=headers,
        timeout_s=args.timeout_s,
        insecure=args.insecure,
    )

    if args.model is None:
        print(f"Fetched first served model: {model}")

    prompt_token_ids_by_request: list[list[int]] | None = None
    if args.prompt_token_ids_jsonl is not None:
        if args.constant_prompt_token_id is not None:
            raise ValueError(
                "--prompt-token-ids-jsonl cannot be combined with "
                "--constant-prompt-token-id"
            )
        prompt_token_ids_by_request = load_prompt_token_ids_jsonl(
            args.prompt_token_ids_jsonl,
            num_prompts=args.num_prompts,
            input_len=args.input_len,
        )
        print(
            "Loaded "
            f"{len(prompt_token_ids_by_request)} prompt-token sequences from "
            f"{args.prompt_token_ids_jsonl}"
        )

    constant_payload_cache: dict[tuple[int, int], Payload] = {}
    if args.constant_prompt_token_id is not None:
        constant_payload_cache[(args.input_len, args.output_len)] = (
            build_constant_prompt_payload_bytes(
                model=model,
                token_id=args.constant_prompt_token_id,
                input_len=args.input_len,
                output_len=args.output_len,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=args.seed,
                return_token_ids=args.return_token_ids,
                add_special_tokens=args.add_special_tokens,
                extra_body=extra_body,
            )
        )

    def make_payload(
        request_id: int, *, input_len: int, output_len: int
    ) -> Payload:
        prompt_token_ids = None
        if (
            prompt_token_ids_by_request is not None
            and input_len == args.input_len
            and output_len == args.output_len
            and request_id < len(prompt_token_ids_by_request)
        ):
            prompt_token_ids = prompt_token_ids_by_request[request_id]
        if args.constant_prompt_token_id is not None:
            key = (input_len, output_len)
            if key not in constant_payload_cache:
                constant_payload_cache[key] = build_constant_prompt_payload_bytes(
                    model=model,
                    token_id=args.constant_prompt_token_id,
                    input_len=input_len,
                    output_len=output_len,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    seed=args.seed,
                    return_token_ids=args.return_token_ids,
                    add_special_tokens=args.add_special_tokens,
                    extra_body=extra_body,
                )
            return constant_payload_cache[key]
        return build_payload(
            model=model,
            request_id=request_id,
            input_len=input_len,
            output_len=output_len,
            prompt_token_ids=prompt_token_ids,
            token_id_start=args.token_id_start,
            token_id_range=args.token_id_range,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            return_token_ids=args.return_token_ids,
            add_special_tokens=args.add_special_tokens,
            extra_body=extra_body,
        )

    if args.num_warmups:
        if args.warmup_input_len is not None and args.warmup_input_len < 1:
            raise ValueError("--warmup-input-len must be >= 1")
        if args.warmup_output_len is not None and args.warmup_output_len < 1:
            raise ValueError("--warmup-output-len must be >= 1")
        print(f"Warming up with {args.num_warmups} request(s)...")
        warmup_input_len = args.warmup_input_len or args.input_len
        warmup_output_len = args.warmup_output_len or args.output_len
        warmup_payloads = [
            make_payload(
                args.num_prompts + i,
                input_len=warmup_input_len,
                output_len=warmup_output_len,
            )
            for i in range(args.num_warmups)
        ]
        _, warmup_results = run_requests(
            api_url=api_url,
            base_url=base_url,
            headers=headers,
            payloads=warmup_payloads,
            request_rate=float("inf"),
            max_concurrency=1,
            timeout_s=args.timeout_s,
            insecure=args.insecure,
        )
        failed_warmups = [r for r in warmup_results if not r.success]
        if failed_warmups:
            print(f"Warmup failed: {failed_warmups[0].error}", file=sys.stderr)
            return 1

    print("Preparing fixed-token payloads...")
    payloads = [
        make_payload(i, input_len=args.input_len, output_len=args.output_len)
        for i in range(args.num_prompts)
    ]

    print("Starting main benchmark run...")
    duration_s, results = run_requests(
        api_url=api_url,
        base_url=base_url,
        headers=headers,
        payloads=payloads,
        request_rate=args.request_rate,
        max_concurrency=args.max_concurrency,
        timeout_s=args.timeout_s,
        insecure=args.insecure,
    )

    summary = summarize_results(
        args=args,
        model=model,
        duration_s=duration_s,
        results=results,
    )
    print_summary(summary)

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote result JSON: {output_path}")

    if args.strict and not summary["verified"]:
        return 1
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/v1/completions")
    parser.add_argument("--model", default=None)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--num-warmups", type=int, default=1)
    parser.add_argument(
        "--warmup-input-len",
        type=int,
        default=None,
        help="Override warmup input length without changing the measured run.",
    )
    parser.add_argument(
        "--warmup-output-len",
        type=int,
        default=None,
        help="Override warmup output length without changing the measured run.",
    )
    parser.add_argument("--timeout-s", type=float, default=6 * 60 * 60)
    parser.add_argument("--token-id-start", type=int, default=1000)
    parser.add_argument("--token-id-range", type=int, default=10000)
    parser.add_argument(
        "--constant-prompt-token-id",
        type=int,
        default=None,
        help=(
            "Use one repeated token id for every prompt and reuse a raw JSON "
            "request body. This avoids materializing huge Python token lists."
        ),
    )
    parser.add_argument(
        "--prompt-token-ids-jsonl",
        default=None,
        help=(
            "JSONL file containing one prompt_token_ids list per measured "
            "request. Each prompt must have exactly --input-len token IDs."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--header", action="append", default=None)
    parser.add_argument("--extra-body", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--save-detailed", action="store_true")
    parser.add_argument("--return-token-ids", action="store_true")
    parser.add_argument("--add-special-tokens", action="store_true")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for HTTPS servers.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Exit successfully even if exact-length validation fails.",
    )
    parser.set_defaults(strict=True)
    return parser


def cli_main() -> int:
    parser = create_parser()
    args = parser.parse_args()
    return main(args)


if __name__ == "__main__":
    raise SystemExit(cli_main())
