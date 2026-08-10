#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
MODEL_PARENT="${MODEL_PARENT:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints}"
MODEL_NAME="${MODEL_NAME:-nano-dsa-16chunk_size-1024chunks}"
REQUEST_DIR="${REQUEST_DIR:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/vllm_repos/vllm_v0.20.1/logs}"
REQUEST_FILE="${REQUEST_FILE:-aalcr_request_row6_chunked_16x1024_131k_think1024.json}"
BASE_IMAGE="${BASE_IMAGE:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/vllm_repos/vllm_v0.20.1/outputs/containers/vllm-openai_v0.20.1_nemotron-h-dsa-chunked-16x1024_20260525.sqsh}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-8011}"

mkdir -p "${LOG_DIR}"

echo "stamp: ${STAMP}"
echo "repo: ${REPO_DIR}"
echo "model: ${MODEL_PARENT}/${MODEL_NAME}"
echo "request: ${REQUEST_DIR}/${REQUEST_FILE}"
echo "logs: ${LOG_DIR}"

srun \
  --account="${SLURM_ACCOUNT:-nemotron_compress_dev}" \
  --partition="${SLURM_PARTITION:-interactive}" \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node="${GPUS_PER_NODE:-8}" \
  --time="${SLURM_TIME:-02:00:00}" \
  --container-image="${BASE_IMAGE}" \
  --container-mounts="${REPO_DIR}:/workspace/vllm-src,${MODEL_PARENT}:/models,${REQUEST_DIR}:/requests,${LOG_DIR}:/logs" \
  bash -lc '
    set -euo pipefail
    STAMP="'"${STAMP}"'"
    MODEL_NAME="'"${MODEL_NAME}"'"
    REQUEST_FILE="'"${REQUEST_FILE}"'"
    PORT="'"${PORT}"'"

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

    run_case() {
      local name="$1"
      local use_overlay="$2"
      local server_log="/logs/${name}_server_${STAMP}.log"
      local response_json="/logs/${name}_response_${STAMP}.json"
      local curl_meta="/logs/${name}_curl_${STAMP}.meta"

      if [[ "${use_overlay}" == "1" ]]; then
        SITE=$(python3 - <<PY
import site
print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))
PY
)
        cp -a /workspace/vllm-src/vllm "$SITE"/
        export VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=1
        export VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16
      else
        unset VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA
        unset VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE
      fi

      export HF_MODULES_CACHE="/tmp/hf_modules_${name}_${STAMP}"
      export VLLM_USE_DEEP_GEMM=0
      export VLLM_MOE_USE_DEEP_GEMM=0
      export VLLM_DEEP_GEMM_WARMUP=skip
      export VLLM_WORKER_MULTIPROC_METHOD=fork

      python3 -m vllm.entrypoints.openai.api_server \
        --model "/models/${MODEL_NAME}" \
        --trust-remote-code \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --served-model-name "${MODEL_NAME}" \
        --tensor-parallel-size 8 \
        --max-model-len 131072 \
        --max-num-batched-tokens 131072 \
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

      curl -sS \
        -H "Content-Type: application/json" \
        --data-binary "@/requests/${REQUEST_FILE}" \
        -o "${response_json}" \
        -w "http_code=%{http_code}\ntime_total=%{time_total}\n" \
        "http://127.0.0.1:${PORT}/v1/chat/completions" \
        | tee "${curl_meta}"

      stop_server "${server_pid}"
      tail -80 "${server_log}" > "/logs/${name}_server_tail_${STAMP}.log" || true
    }

    run_case old 0
    run_case page_table 1
  ' 2>&1 | tee "${LOG_DIR}/aalcr_compare_${STAMP}.log"
