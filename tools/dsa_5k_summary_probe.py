#!/usr/bin/env python
"""Send a ~5K-token summary probe to a running vLLM server."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import time
import urllib.error
import urllib.request


DEFAULT_MODEL_DIR = (
    "/lustre/fsw/portfolios/coreai/users/mdabbah/deci/"
    "puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints/"
    "nemo3_repr_heads_128k_top2k_q128_sparse_topk2048_nemotron_h_puzzle"
)


def _section(index: int) -> str:
    teams = [
        "retrieval",
        "serving",
        "evaluation",
        "data governance",
        "operations",
        "model quality",
    ]
    team = teams[index % len(teams)]
    return f"""
Section {index}: {team.title()} Workstream
The program is moving a production question-answering assistant from a
prototype stack into a controlled inference service. The main objective is to
serve long technical documents, preserve citations, and provide concise
summaries that a review team can compare against source material. The current
checkpoint already handles ordinary short prompts, but the next milestone is to
exercise longer prompts with realistic structure, repeated references, and a
mix of operational details.

The {team} team reported three observations. First, the prompt shape matters:
requests with headings, tables described in prose, and numbered decision logs
make it easier for reviewers to inspect whether the model is tracking the
document. Second, latency is acceptable for offline evaluation, but interactive
use will require batching controls and careful limits on generated tokens.
Third, any sparse or selective attention path must log enough metadata for
debugging without flooding the normal request log.

Risks remain around silent degradation. A model can appear fluent while missing
constraints, inventing owners, or confusing draft recommendations with approved
decisions. The review process therefore asks for summaries that identify the
primary objective, open risks, accepted decisions, and concrete next steps. A
summary that rambles, repeats unrelated nouns, changes topic abruptly, or
ignores these requested fields should be treated as low quality even when the
syntax looks grammatical.

The near-term plan is modest. Keep the server configuration stable, run one
long prompt at a time, collect timing and raw response logs, and compare the
answer to the document's obvious themes. If the result is coherent, the team
will move on to longer prompts and a small set of real evaluation documents. If
the result is unstable, the team will inspect attention routing, prompt length
accounting, and tokenizer behavior before changing model weights.
""".strip()


def build_prompt(target_words: int) -> tuple[str, int]:
    instruction = """
You are a technical reviewer. Read the following internal planning memo and
write a concise summary with four parts: primary objective, important risks,
accepted decisions, and concrete next steps. Do not quote the memo verbatim.
If the memo repeats itself, merge repeated points instead of listing them
again.

<document>
""".strip()
    suffix = """
</document>

Summary:
""".strip()

    sections: list[str] = []
    prompt = f"{instruction}\n\n{suffix}"
    while len(prompt.split()) < target_words:
        sections.append(_section(len(sections) + 1))
        prompt = f"{instruction}\n\n{'\n\n'.join(sections)}\n\n{suffix}"
    return prompt, len(prompt.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint",
                        default="http://pool0-01594:8000/v1/completions")
    parser.add_argument("--model", default="dsa-nemotron-h-puzzle")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--target-prompt-words", type=int, default=4300)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"dsa_5k_summary_probe_{stamp}.log"

    prompt, prompt_words = build_prompt(args.target_prompt_words)
    payload = {
        "model": args.model,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0.2,
        "top_p": 0.95,
    }

    with log_path.open("w", encoding="utf-8") as log:
        log.write("== DSA 5K Summary Probe ==\n")
        log.write(f"started_at: {dt.datetime.now().isoformat()}\n")
        log.write(f"endpoint: {args.endpoint}\n")
        log.write(f"model: {args.model}\n")
        log.write(f"target_prompt_words: {args.target_prompt_words}\n")
        log.write(f"actual_prompt_words: {prompt_words}\n")
        log.write(f"max_tokens: {args.max_tokens}\n\n")
        log.write("== Prompt ==\n")
        log.write(prompt)
        log.write("\n\n== Raw Response ==\n")
        log.flush()

        request = urllib.request.Request(
            args.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        status = "unknown"
        raw = ""
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as resp:
                status = str(resp.status)
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            status = f"http_error_{exc.code}"
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001 - log probe failures verbosely.
            status = f"exception_{type(exc).__name__}"
            raw = str(exc)
        elapsed = time.monotonic() - start

        log.write(raw)
        log.write("\n\n== Timing ==\n")
        log.write(f"status: {status}\n")
        log.write(f"elapsed_seconds: {elapsed:.2f}\n")
        log.write(f"finished_at: {dt.datetime.now().isoformat()}\n")

        answer = ""
        reported_prompt_tokens = None
        try:
            body = json.loads(raw)
            answer = body["choices"][0]["text"]
            reported_prompt_tokens = body.get("usage", {}).get(
                "prompt_tokens")
        except Exception:
            answer = ""
        log.write(f"reported_prompt_tokens: {reported_prompt_tokens}\n")
        log.write("\n== Extracted Answer ==\n")
        log.write(answer)
        log.write("\n")

    print(json.dumps({
        "log_path": str(log_path),
        "status": status,
        "elapsed_seconds": round(elapsed, 2),
        "prompt_words": prompt_words,
        "reported_prompt_tokens": reported_prompt_tokens,
        "answer": answer,
    }))


if __name__ == "__main__":
    main()
