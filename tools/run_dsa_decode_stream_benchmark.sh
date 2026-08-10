#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
MODEL_PARENT="${MODEL_PARENT:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints}"
MODEL_NAME="${MODEL_NAME:-nano-dsa-16chunk_size-1024chunks}"
MODEL_DIR="${MODEL_DIR:-/models/${MODEL_NAME}}"
REQUEST_DIR="${REQUEST_DIR:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/vllm_repos/vllm_v0.20.1/logs}"
REQUEST_FILE="${REQUEST_FILE:-aalcr_request_row6_chunked_16x1024_131k_think1024.json}"
BASE_IMAGE="${BASE_IMAGE:-${REPO_DIR}/outputs/containers/vllm-openai_v0.20.1_nemotron-h-dsa-moonshot-shared-page-table_20260531_115102.sqsh}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/decode_stream_benchmark_$(date +%Y%m%d_%H%M%S)}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-8023}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-1000}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
TIMING="${TIMING:-0}"
TIMING_PRINT_LIMIT="${TIMING_PRINT_LIMIT:-200000}"
CASES="${CASES:-current_pt:chunked_topk_sparse:1:1:1:1:1 dense_full:disabled:0:0:0:0:0}"

mkdir -p "${LOG_DIR}"

echo "stamp: ${STAMP}"
echo "repo: ${REPO_DIR}"
echo "image: ${BASE_IMAGE}"
echo "model: ${MODEL_PARENT}/${MODEL_NAME}"
echo "request: ${REQUEST_DIR}/${REQUEST_FILE}"
echo "logs: ${LOG_DIR}"
echo "max_completion_tokens: ${MAX_COMPLETION_TOKENS}"
echo "timing: ${TIMING}"
echo "cases: ${CASES}"

srun \
  --account="${SLURM_ACCOUNT:-nemotron_compress_dev}" \
  --partition="${SLURM_PARTITION:-interactive}" \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node="${GPUS_PER_NODE:-8}" \
  --time="${SLURM_TIME:-04:00:00}" \
  --container-image="${BASE_IMAGE}" \
  --container-mounts="${REPO_DIR}:/workspace/vllm-src,${MODEL_PARENT}:/models,${REQUEST_DIR}:/requests,${LOG_DIR}:/logs" \
  bash -lc '
    set -euo pipefail

    MODEL_NAME="'"${MODEL_NAME}"'"
    MODEL_DIR="'"${MODEL_DIR}"'"
    REQUEST_FILE="'"${REQUEST_FILE}"'"
    PORT="'"${PORT}"'"
    STAMP="'"${STAMP}"'"
    MAX_COMPLETION_TOKENS="'"${MAX_COMPLETION_TOKENS}"'"
    MAX_NUM_BATCHED_TOKENS="'"${MAX_NUM_BATCHED_TOKENS}"'"
    TIMING="'"${TIMING}"'"
    TIMING_PRINT_LIMIT="'"${TIMING_PRINT_LIMIT}"'"
    CASES="'"${CASES}"'"

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

    make_stream_request() {
      local output_path="$1"
      python3 - "/requests/${REQUEST_FILE}" "${output_path}" "${MAX_COMPLETION_TOKENS}" <<'"'"'PY'"'"'
import json
import sys

src, dst, max_tokens = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(src, encoding="utf-8") as f:
    data = json.load(f)
data["max_completion_tokens"] = max_tokens
data["stream"] = True
data["stream_options"] = {"include_usage": True}
data["ignore_eos"] = True
data["temperature"] = 0
with open(dst, "w", encoding="utf-8") as f:
    json.dump(data, f)
PY
    }

    stream_request() {
      local request_json="$1"
      local stream_path="$2"
      local meta_path="$3"
      python3 - "${request_json}" "${stream_path}" "${meta_path}" "${PORT}" <<'"'"'PY'"'"'
import json
import sys
import time
import urllib.error
import urllib.request

request_json, stream_path, meta_path, port = sys.argv[1:5]
with open(request_json, "rb") as f:
    payload = f.read()

req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

start = time.perf_counter()
headers_at = None
first_content_at = None
status = None
error = None

try:
    with urllib.request.urlopen(req, timeout=None) as resp, open(
        stream_path, "wb"
    ) as out:
        status = resp.status
        headers_at = time.perf_counter()
        for raw in resp:
            now = time.perf_counter()
            out.write(raw)
            out.flush()
            line = raw.strip()
            if first_content_at is not None or not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:"):].strip()
            if not payload or payload == b"[DONE]":
                continue
            try:
                row = json.loads(payload)
            except Exception:
                continue
            for choice in row.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    first_content_at = now
                    break
except urllib.error.HTTPError as exc:
    status = exc.code
    error = str(exc)
    with open(stream_path, "wb") as out:
        out.write(exc.read())
except Exception as exc:  # pragma: no cover - benchmark helper
    error = repr(exc)

total_at = time.perf_counter()
metrics = {
    "http_code": status,
    "error": error,
    "time_headers": None if headers_at is None else headers_at - start,
    "time_first_content": (
        None if first_content_at is None else first_content_at - start
    ),
    "time_total": total_at - start,
}
if first_content_at is not None:
    metrics["decode_window"] = total_at - first_content_at
with open(meta_path, "w", encoding="utf-8") as f:
    for key, value in metrics.items():
        f.write(f"{key}={value}\n")
print(json.dumps(metrics, indent=2, sort_keys=True))
PY
    }

    parse_stream() {
      local stream_path="$1"
      local summary_path="$2"
      python3 - "${stream_path}" "${summary_path}" <<'"'"'PY'"'"'
import json
import sys

stream_path, summary_path = sys.argv[1:3]
content = []
usage = None
done = False
chunks = 0
with open(stream_path, encoding="utf-8", errors="replace") as f:
    for raw in f:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            done = True
            continue
        if not payload:
            continue
        row = json.loads(payload)
        chunks += 1
        if row.get("usage"):
            usage = row["usage"]
        for choice in row.get("choices", []):
            delta = choice.get("delta") or {}
            piece = delta.get("content")
            if piece:
                content.append(piece)

text = "".join(content)
summary = {
    "done": done,
    "chunks": chunks,
    "content_chars": len(text),
    "content_preview": text[:500],
    "content_tail": text[-500:],
    "usage": usage,
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(json.dumps(summary, indent=2, sort_keys=True))
PY
    }

    summarize_dsa_timings() {
      local server_log="$1"
      local summary_json="$2"
      python3 - "${server_log}" "${summary_json}" <<'"'"'PY'"'"'
import json
import re
import statistics as st
import sys
from collections import defaultdict

server_log, summary_json = sys.argv[1:3]
pat = re.compile(r"DSA_TIMING (.*)")
rows = []
with open(server_log, encoding="utf-8", errors="replace") as f:
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
    "decode_page_table_ms",
    "manual_materialize_ms",
    "manual_attention_ms",
    "table_build_ms",
    "fa_ms",
    "total_ms",
]
groups = defaultdict(list)
for row in rows:
    key = row.get("path", "unknown")
    if "mode" in row:
        key += ":" + row["mode"]
    groups[key].append(row)

summary = {"num_timing_rows": len(rows), "groups": {}}
for key, group_rows in groups.items():
    stats = {"count": len(group_rows)}
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

with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, sort_keys=True)
print(json.dumps(summary, indent=2, sort_keys=True)[:4000])
PY
    }

    run_case() {
      local name="$1"
      local attn_mode="$2"
      local use_page_table="$3"
      local use_prefill_page_table="$4"
      local use_full_attention_short_seq="$5"
      local share_chunk_topk="$6"
      local use_shared_prefill_page_table="$7"

      local request_json="/logs/${name}_request_${STAMP}.json"
      local stream_out="/logs/${name}_stream_${STAMP}.sse"
      local curl_meta="/logs/${name}_curl_${STAMP}.meta"
      local stream_summary="/logs/${name}_stream_summary_${STAMP}.json"
      local server_log="/logs/${name}_server_${STAMP}.log"
      local timing_summary="/logs/${name}_timing_summary_${STAMP}.json"

      make_stream_request "${request_json}"

      export VLLM_NEMOTRON_H_DSA_ATTN_MODE="${attn_mode}"
      export VLLM_NEMOTRON_H_DSA_CHUNK_SIZE=16
      export VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K=1024
      export VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16
      export VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA="${use_page_table}"
      export VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA="${use_prefill_page_table}"
      export VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ="${use_full_attention_short_seq}"
      export VLLM_NEMOTRON_H_DSA_SHARE_CHUNK_TOPK="${share_chunk_topk}"
      export VLLM_NEMOTRON_H_DSA_USE_SHARED_PREFILL_PAGE_TABLE_FA="${use_shared_prefill_page_table}"
      export VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE=256
      export VLLM_NEMOTRON_H_DSA_USE_UNION_PREFILL_KERNEL=0
      export VLLM_NEMOTRON_H_DSA_USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA=0
      export VLLM_NEMOTRON_H_DSA_TIMING="${TIMING}"
      export VLLM_NEMOTRON_H_DSA_TIMING_PRINT_LIMIT="${TIMING_PRINT_LIMIT}"
      export HF_MODULES_CACHE="/tmp/hf_modules_${name}_${STAMP}"
      export VLLM_USE_DEEP_GEMM=0
      export VLLM_MOE_USE_DEEP_GEMM=0
      export VLLM_DEEP_GEMM_WARMUP=skip
      export VLLM_WORKER_MULTIPROC_METHOD=fork

      echo "running ${name}: attn_mode=${attn_mode} page_table=${use_page_table} prefill_page_table=${use_prefill_page_table} shared_prefill=${use_shared_prefill_page_table} timing=${TIMING}"
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
      echo "${server_pid}" > "/logs/${name}_server_${STAMP}.pid"

      if ! wait_for_server "${PORT}"; then
        echo "server did not become ready for ${name}" | tee "${curl_meta}"
        tail -200 "${server_log}" || true
        stop_server "${server_pid}"
        return 1
      fi

      stream_request "${request_json}" "${stream_out}" "${curl_meta}" \
        | tee "/logs/${name}_stream_client_${STAMP}.txt"

      parse_stream "${stream_out}" "${stream_summary}" \
        > "/logs/${name}_stream_summary_${STAMP}.txt"
      summarize_dsa_timings "${server_log}" "${timing_summary}" \
        > "/logs/${name}_timing_summary_${STAMP}.txt"

      stop_server "${server_pid}"
      tail -160 "${server_log}" > "/logs/${name}_server_tail_${STAMP}.log" || true
    }

    for case_spec in ${CASES}; do
      IFS=: read -r name attn_mode use_page_table use_prefill_page_table \
        use_full_attention_short_seq share_chunk_topk \
        use_shared_prefill_page_table <<<"${case_spec}"
      run_case \
        "${name}" \
        "${attn_mode}" \
        "${use_page_table}" \
        "${use_prefill_page_table}" \
        "${use_full_attention_short_seq}" \
        "${share_chunk_topk}" \
        "${use_shared_prefill_page_table}"
    done

    python3 - "/logs" "${STAMP}" <<'"'"'PY'"'"'
import glob
import json
import os
import sys

log_dir, stamp = sys.argv[1:3]
rows = []
for meta_path in sorted(glob.glob(os.path.join(log_dir, f"*_curl_{stamp}.meta"))):
    name = os.path.basename(meta_path).split(f"_curl_{stamp}.meta")[0]
    vals = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.strip().split("=", 1)
                if v == "None":
                    vals[k] = None
                else:
                    vals[k] = v
    summary_path = os.path.join(log_dir, f"{name}_stream_summary_{stamp}.json")
    usage = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            usage = (json.load(f).get("usage") or {})
    ttft_raw = vals.get("time_first_content")
    total_raw = vals.get("time_total")
    decode_raw = vals.get("decode_window")
    ttft = None if ttft_raw is None else float(ttft_raw)
    total = None if total_raw is None else float(total_raw)
    decode_s = None if decode_raw is None else float(decode_raw)
    completion_tokens = usage.get("completion_tokens")
    toks_per_s = None
    if completion_tokens and decode_s and decode_s > 0:
        toks_per_s = completion_tokens / decode_s
    rows.append({
        "case": name,
        "http_code": vals.get("http_code"),
        "error": vals.get("error"),
        "ttft_s": ttft,
        "total_s": total,
        "decode_window_s": decode_s,
        "completion_tokens": completion_tokens,
        "decode_tokens_per_s": toks_per_s,
        "usage": usage,
    })

out = {"stamp": stamp, "rows": rows}
with open(os.path.join(log_dir, f"decode_stream_summary_{stamp}.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print(json.dumps(out, indent=2, sort_keys=True))
PY
  '
