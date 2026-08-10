#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE_URL=${BASE_URL:-"http://127.0.0.1:8000"}
MODEL=${MODEL:-""}
INPUT_LEN=${INPUT_LEN:-8192}
OUTPUT_LEN=${OUTPUT_LEN:-1}
NUM_PROMPTS=${NUM_PROMPTS:-100}
REQUEST_RATE=${REQUEST_RATE:-inf}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-""}
NUM_WARMUPS=${NUM_WARMUPS:-1}
TOKEN_ID_START=${TOKEN_ID_START:-1000}
TOKEN_ID_RANGE=${TOKEN_ID_RANGE:-10000}
CONSTANT_PROMPT_TOKEN_ID=${CONSTANT_PROMPT_TOKEN_ID:-""}
PROMPT_TOKEN_IDS_JSONL=${PROMPT_TOKEN_IDS_JSONL:-""}
TEMPERATURE=${TEMPERATURE:-0}
TOP_P=${TOP_P:-""}
SEED=${SEED:-""}
EXTRA_BODY=${EXTRA_BODY:-""}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/fixed_io_results"}
SAVE_DETAILED=${SAVE_DETAILED:-0}
RETURN_TOKEN_IDS=${RETURN_TOKEN_IDS:-0}
ADD_SPECIAL_TOKENS=${ADD_SPECIAL_TOKENS:-0}
INSECURE=${INSECURE:-0}

usage() {
  printf "Usage: %s [options]\n" "$0"
  printf "Options:\n"
  printf "  --base-url URL              Server base URL (default: %s)\n" "$BASE_URL"
  printf "  --model MODEL               Served model name"
  printf " (default: fetch first model)\n"
  printf "  --input-len N               Exact input token length"
  printf " (default: %s)\n" "$INPUT_LEN"
  printf "  --output-len N              Exact output token length"
  printf " (default: %s)\n" "$OUTPUT_LEN"
  printf "  --num-prompts N             Number of requests"
  printf " (default: %s)\n" "$NUM_PROMPTS"
  printf "  --request-rate R            Requests/sec, or inf for one burst"
  printf " (default: %s)\n" "$REQUEST_RATE"
  printf "  --max-concurrency N         Optional client-side concurrency cap\n"
  printf "  --num-warmups N             Warmup requests excluded from timing"
  printf " (default: %s)\n" "$NUM_WARMUPS"
  printf "  --token-id-start N          First prompt token ID to use"
  printf " (default: %s)\n" "$TOKEN_ID_START"
  printf "  --token-id-range N          Prompt token ID range size"
  printf " (default: %s)\n" "$TOKEN_ID_RANGE"
  printf "  --constant-prompt-token-id N"
  printf "  Reuse one raw JSON prompt body with token ID N\n"
  printf "  --prompt-token-ids-jsonl PATH"
  printf "  JSONL with one prompt_token_ids list per prompt\n"
  printf "  --temperature F             Sampling temperature"
  printf " (default: %s)\n" "$TEMPERATURE"
  printf "  --top-p F                   Optional top-p\n"
  printf "  --seed N                    Optional per-request seed base\n"
  printf "  --extra-body JSON           Extra JSON body merged into each request\n"
  printf "  --output-dir DIR            Result directory (default: %s)\n" "$OUTPUT_DIR"
  printf "  --save-detailed             Include per-request records in JSON\n"
  printf "  --return-token-ids          Ask server to return token IDs\n"
  printf "  --add-special-tokens        Ask server to add special tokens"
  printf " to token-id prompt\n"
  printf "  --insecure                  Disable TLS certificate verification\n"
  printf "  -h, --help                  Show this help message\n"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --input-len)
      INPUT_LEN="$2"
      shift 2
      ;;
    --output-len)
      OUTPUT_LEN="$2"
      shift 2
      ;;
    --num-prompts)
      NUM_PROMPTS="$2"
      shift 2
      ;;
    --request-rate)
      REQUEST_RATE="$2"
      shift 2
      ;;
    --max-concurrency)
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    --num-warmups)
      NUM_WARMUPS="$2"
      shift 2
      ;;
    --token-id-start)
      TOKEN_ID_START="$2"
      shift 2
      ;;
    --token-id-range)
      TOKEN_ID_RANGE="$2"
      shift 2
      ;;
    --constant-prompt-token-id)
      CONSTANT_PROMPT_TOKEN_ID="$2"
      shift 2
      ;;
    --prompt-token-ids-jsonl)
      PROMPT_TOKEN_IDS_JSONL="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --top-p)
      TOP_P="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --extra-body)
      EXTRA_BODY="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --save-detailed)
      SAVE_DETAILED=1
      shift
      ;;
    --return-token-ids)
      RETURN_TOKEN_IDS=1
      shift
      ;;
    --add-special-tokens)
      ADD_SPECIAL_TOKENS=1
      shift
      ;;
    --insecure)
      INSECURE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf "Unknown argument: %s\n" "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$OUTPUT_DIR"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_CMD=("$PYTHON")
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_CMD=("$REPO_ROOT/.venv/bin/python")
elif command -v uv >/dev/null 2>&1; then
  PYTHON_CMD=(uv run python)
else
  printf "Could not find .venv/bin/python or uv. Set PYTHON explicitly.\n" >&2
  exit 1
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SAFE_MODEL="${MODEL:-first-model}"
SAFE_MODEL="${SAFE_MODEL##*/}"
RESULT_JSON="$OUTPUT_DIR/fixed_io_${SAFE_MODEL}_in${INPUT_LEN}"
RESULT_JSON+="_out${OUTPUT_LEN}_n${NUM_PROMPTS}_${TIMESTAMP}.json"

ARGS=(
  "$SCRIPT_DIR/benchmark_fixed_io_serving.py"
  --base-url "$BASE_URL"
  --input-len "$INPUT_LEN"
  --output-len "$OUTPUT_LEN"
  --num-prompts "$NUM_PROMPTS"
  --request-rate "$REQUEST_RATE"
  --num-warmups "$NUM_WARMUPS"
  --token-id-start "$TOKEN_ID_START"
  --token-id-range "$TOKEN_ID_RANGE"
  --temperature "$TEMPERATURE"
  --output-json "$RESULT_JSON"
)

if [[ -n "$MODEL" ]]; then
  ARGS+=(--model "$MODEL")
fi
if [[ -n "$MAX_CONCURRENCY" ]]; then
  ARGS+=(--max-concurrency "$MAX_CONCURRENCY")
fi
if [[ -n "$TOP_P" ]]; then
  ARGS+=(--top-p "$TOP_P")
fi
if [[ -n "$SEED" ]]; then
  ARGS+=(--seed "$SEED")
fi
if [[ -n "$CONSTANT_PROMPT_TOKEN_ID" ]]; then
  ARGS+=(--constant-prompt-token-id "$CONSTANT_PROMPT_TOKEN_ID")
fi
if [[ -n "$PROMPT_TOKEN_IDS_JSONL" ]]; then
  ARGS+=(--prompt-token-ids-jsonl "$PROMPT_TOKEN_IDS_JSONL")
fi
if [[ -n "$EXTRA_BODY" ]]; then
  ARGS+=(--extra-body "$EXTRA_BODY")
fi
if [[ "$SAVE_DETAILED" == "1" ]]; then
  ARGS+=(--save-detailed)
fi
if [[ "$RETURN_TOKEN_IDS" == "1" ]]; then
  ARGS+=(--return-token-ids)
fi
if [[ "$ADD_SPECIAL_TOKENS" == "1" ]]; then
  ARGS+=(--add-special-tokens)
fi
if [[ "$INSECURE" == "1" ]]; then
  ARGS+=(--insecure)
fi

printf "Running fixed IO serving benchmark\n"
printf "  base url:       %s\n" "$BASE_URL"
printf "  model:          %s\n" "${MODEL:-first served model}"
printf "  input len:      %s\n" "$INPUT_LEN"
printf "  output len:     %s\n" "$OUTPUT_LEN"
printf "  num prompts:    %s\n" "$NUM_PROMPTS"
printf "  request rate:   %s\n" "$REQUEST_RATE"
printf "  constant token: %s\n" "${CONSTANT_PROMPT_TOKEN_ID:-off}"
printf "  prompt ids:     %s\n" "${PROMPT_TOKEN_IDS_JSONL:-off}"
printf "  result JSON:    %s\n" "$RESULT_JSON"

"${PYTHON_CMD[@]}" "${ARGS[@]}"
