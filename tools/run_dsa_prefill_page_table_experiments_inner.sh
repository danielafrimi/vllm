#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SRC_DIR="${SRC_DIR:-/workspace/vllm-src}"
LOG_MOUNT="${LOG_MOUNT:-/logs}"
MODEL_NAME="${MODEL_NAME:-nano-dsa-16chunk_size-1024chunks}"
REQUEST_FILE="${REQUEST_FILE:-aalcr_request_row6_chunked_16x1024_131k_think1024.json}"
MODEL_DIR="${MODEL_DIR:-/models/${MODEL_NAME}}"
REQUEST_PATH="${REQUEST_PATH:-/requests/${REQUEST_FILE}}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-8021}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
TIMING_PRINT_LIMIT="${TIMING_PRINT_LIMIT:-20000}"
TIMING="${TIMING:-1}"
RUN_TESTS="${RUN_TESTS:-1}"
USE_FULL_ATTN_SHORT_SEQ="${USE_FULL_ATTN_SHORT_SEQ:-0}"
SHARE_CHUNK_TOPK="${SHARE_CHUNK_TOPK:-0}"
USE_SHARED_PREFILL_PAGE_TABLE_FA="${USE_SHARED_PREFILL_PAGE_TABLE_FA:-0}"
TOPK_STATS="${TOPK_STATS:-0}"
TOPK_STATS_LIMIT="${TOPK_STATS_LIMIT:-200}"
USE_UNION_PREFILL_KERNEL="${USE_UNION_PREFILL_KERNEL:-0}"
USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA="${USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA:-0}"
UNION_CHUNKS_PER_ITER="${UNION_CHUNKS_PER_ITER:-8}"
CASES="${CASES:-manual_q16:16:0:0 manual_q64:64:0:0 prefill_pt_q64:64:1:0 prefill_pt_q128:128:1:0}"

SITE=$(python3 - <<'PY'
import site
print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))
PY
)
cp -a "${SRC_DIR}/vllm" "${SITE}/"

if [[ "${RUN_TESTS}" == "1" ]]; then
  if python3 -c "import pytest" >/dev/null 2>&1; then
    python3 -m pytest \
      "${SRC_DIR}/tests/model_executor/models/test_nemotron_h_dsa_chunked.py" \
      -q 2>&1 | tee "${LOG_MOUNT}/prefill_page_table_unit_${STAMP}.log"
  else
    VLLM_DSA_PARITY_USE_SITE_PACKAGE=1 \
      python3 "${SRC_DIR}/tools/check_dsa_prefill_page_table_parity.py" \
      2>&1 | tee "${LOG_MOUNT}/prefill_page_table_unit_${STAMP}.log"
  fi
fi

wait_for_server() {
  local port="$1"
  for _ in $(seq 1 900); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 2
  done
  return 1
}

stop_server() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}" >/dev/null 2>&1 || true
    wait "${pid}" >/dev/null 2>&1 || true
  fi
  sleep 5
}

extract_answer() {
  local response_json="$1"
  local answer_txt="$2"
  python3 - "$response_json" "$answer_txt" <<'PY'
import json
import sys

response_path, answer_path = sys.argv[1:3]
with open(response_path) as f:
    data = json.load(f)
content = data["choices"][0]["message"]["content"]
with open(answer_path, "w") as f:
    f.write(content)
print(content[:500].replace("\n", "\\n"))
PY
}

summarize_timings() {
  local server_log="$1"
  local summary_json="$2"
  python3 - "$server_log" "$summary_json" <<'PY'
import json
import re
import statistics as st
import sys

server_log, summary_json = sys.argv[1:3]
rows = []
pat = re.compile(r"DSA_TIMING (.*)")
with open(server_log, errors="replace") as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        row = {}
        for token in m.group(1).split():
            if "=" in token:
                k, v = token.split("=", 1)
                row[k] = v
        rows.append(row)

metrics = [
    "indexer_proj_ms",
    "key_gather_ms",
    "summary_ms",
    "score_topk_ms",
    "prefill_page_table_ms",
    "decode_page_table_ms",
    "manual_materialize_ms",
    "manual_attention_ms",
    "table_build_ms",
    "fa_ms",
    "total_ms",
]
summary = {"num_timing_rows": len(rows), "groups": {}}
for row in rows:
    key = row.get("path", "unknown")
    if "mode" in row:
        key += ":" + row["mode"]
    summary["groups"].setdefault(key, []).append(row)

for key, group_rows in list(summary["groups"].items()):
    stats = {"count": len(group_rows)}
    layers = sorted(
        {r.get("layer") for r in group_rows if r.get("layer") is not None},
        key=int,
    )
    if layers:
        stats["layers"] = layers
    for metric in metrics:
        vals = [float(r[metric]) for r in group_rows if metric in r]
        if vals:
            stats[metric] = {
                "avg": st.mean(vals),
                "p50": st.median(vals),
                "min": min(vals),
                "max": max(vals),
            }
    summary["groups"][key] = stats

with open(summary_json, "w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(json.dumps(summary, indent=2, sort_keys=True)[:4000])
PY
}

run_case() {
  local name="$1"
  local query_chunk_size="$2"
  local use_prefill_page_table="$3"
  local use_full_attention_short_seq="${4:-${USE_FULL_ATTN_SHORT_SEQ}}"
  local server_log="${LOG_MOUNT}/${name}_server_${STAMP}.log"
  local response_json="${LOG_MOUNT}/${name}_response_${STAMP}.json"
  local curl_meta="${LOG_MOUNT}/${name}_curl_${STAMP}.meta"
  local answer_txt="${LOG_MOUNT}/${name}_answer_${STAMP}.txt"
  local timing_summary="${LOG_MOUNT}/${name}_timing_summary_${STAMP}.json"

  export VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=1
  export VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16
  export VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA="${use_prefill_page_table}"
  export VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ="${use_full_attention_short_seq}"
  export VLLM_NEMOTRON_H_DSA_SHARE_CHUNK_TOPK="${SHARE_CHUNK_TOPK}"
  export VLLM_NEMOTRON_H_DSA_USE_SHARED_PREFILL_PAGE_TABLE_FA="${USE_SHARED_PREFILL_PAGE_TABLE_FA}"
  export VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE="${query_chunk_size}"
  export VLLM_NEMOTRON_H_DSA_TIMING="${TIMING}"
  export VLLM_NEMOTRON_H_DSA_TIMING_PRINT_LIMIT="${TIMING_PRINT_LIMIT}"
  export VLLM_NEMOTRON_H_DSA_TOPK_STATS="${TOPK_STATS}"
  export VLLM_NEMOTRON_H_DSA_TOPK_STATS_LIMIT="${TOPK_STATS_LIMIT}"
  export VLLM_NEMOTRON_H_DSA_USE_UNION_PREFILL_KERNEL="${USE_UNION_PREFILL_KERNEL}"
  export VLLM_NEMOTRON_H_DSA_USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA="${USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA}"
  export VLLM_NEMOTRON_H_DSA_UNION_CHUNKS_PER_ITER="${UNION_CHUNKS_PER_ITER}"
  export HF_MODULES_CACHE="/tmp/hf_modules_${name}_${STAMP}"
  export VLLM_USE_DEEP_GEMM=0
  export VLLM_MOE_USE_DEEP_GEMM=0
  export VLLM_DEEP_GEMM_WARMUP=skip
  export VLLM_WORKER_MULTIPROC_METHOD=fork

  echo "running ${name}: query_chunk_size=${query_chunk_size} prefill_page_table=${use_prefill_page_table} full_attn_short_seq=${use_full_attention_short_seq}"
  python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_DIR}" \
    --trust-remote-code \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --served-model-name "${MODEL_NAME}" \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --mamba-ssm-cache-dtype float32 \
    --moe-backend triton \
    --no-enable-log-requests \
    --max-num-seqs 1 \
    --enforce-eager \
    > "${server_log}" 2>&1 &
  local server_pid="$!"
  echo "${server_pid}" > "${LOG_MOUNT}/${name}_server_${STAMP}.pid"

  if ! wait_for_server "${PORT}"; then
    echo "server did not become ready for ${name}" | tee "${curl_meta}"
    tail -200 "${server_log}" || true
    stop_server "${server_pid}"
    return 1
  fi

  curl -sS \
    -H "Content-Type: application/json" \
    --data-binary "@${REQUEST_PATH}" \
    -o "${response_json}" \
    -w "http_code=%{http_code}\ntime_total=%{time_total}\n" \
    "http://127.0.0.1:${PORT}/v1/chat/completions" \
    | tee "${curl_meta}"

  extract_answer "${response_json}" "${answer_txt}" | tee -a "${curl_meta}"
  summarize_timings "${server_log}" "${timing_summary}" \
    > "${LOG_MOUNT}/${name}_timing_summary_${STAMP}.txt"

  stop_server "${server_pid}"
  tail -120 "${server_log}" > "${LOG_MOUNT}/${name}_server_tail_${STAMP}.log" || true
}

baseline_answer=""
for case_spec in ${CASES}; do
  IFS=: read -r name query_chunk_size use_prefill_page_table \
    use_full_attention_short_seq <<<"${case_spec}"
  use_full_attention_short_seq="${use_full_attention_short_seq:-${USE_FULL_ATTN_SHORT_SEQ}}"
  run_case \
    "${name}" \
    "${query_chunk_size}" \
    "${use_prefill_page_table}" \
    "${use_full_attention_short_seq}"
  answer_path="${LOG_MOUNT}/${name}_answer_${STAMP}.txt"
  if [[ -z "${baseline_answer}" ]]; then
    baseline_answer="$(cat "${answer_path}")"
    echo "baseline_case=${name}" > "${LOG_MOUNT}/correctness_${STAMP}.txt"
    echo "baseline_answer=${baseline_answer}" >> "${LOG_MOUNT}/correctness_${STAMP}.txt"
  else
    answer="$(cat "${answer_path}")"
    if [[ "${answer}" == "${baseline_answer}" ]]; then
      echo "${name}: exact_match" >> "${LOG_MOUNT}/correctness_${STAMP}.txt"
    else
      echo "${name}: mismatch" >> "${LOG_MOUNT}/correctness_${STAMP}.txt"
      echo "answer=${answer}" >> "${LOG_MOUNT}/correctness_${STAMP}.txt"
    fi
  fi
done

cat "${LOG_MOUNT}/correctness_${STAMP}.txt"
