#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Default generation-disagreement launcher for Nemotron-H DSA path coverage.
# Submit with:
#   sbatch --export=ALL,IMAGE=/path/to/current-code.sqsh \
#     scripts/generation_disagreement/run_sparse_agreement.sbatch

#SBATCH --job-name=gd-sparse-agree
#SBATCH --account=nemotron_n4_compress
#SBATCH --partition=batch_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --mem=900G
#SBATCH --time=04:00:00
#SBATCH --output=generation_disagreement_sparse_agreement_%j.log
#SBATCH --error=generation_disagreement_sparse_agreement_%j.log

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" && -d "${SLURM_SUBMIT_DIR}/vllm" ]]; then
    REPO_DIR="$(git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel)"
  else
    REPO_DIR="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"
  fi
fi

: "${IMAGE:?set IMAGE to the current-code SQSH image before submitting}"

TP_SIZE="${TP_SIZE:-2}"
MODEL_DIR="${MODEL_DIR:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints/nano-dsa-16chunk_size-2048chunks}"
BASELINE="${BASELINE:-${REPO_DIR}/tools/codex/skills/vllm-generation-disagreement/baselines/baseline_sparse_4096_200.jsonl}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/generation_disagreement}"
RUN_NAME="${RUN_NAME:-sparse_agreement_current}"
RUN_DIR="${RUN_DIR:-${OUT_ROOT}/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"

TARGET_PROMPT_TOKENS="${TARGET_PROMPT_TOKENS:-4096}"
MAX_TOKENS="${MAX_TOKENS:-200}"
GENERATION_SEED="${GENERATION_SEED:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-1024}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DSA_DENSE_PREFILL_THRESHOLD="${DSA_DENSE_PREFILL_THRESHOLD:-2048}"
DSA_PATH_DEBUG_PRINT_LIMIT="${DSA_PATH_DEBUG_PRINT_LIMIT:-32}"
RUN_COMPARE="${RUN_COMPARE:-1}"
PATCH_DSA_MARKERS="${PATCH_DSA_MARKERS:-1}"
COPY_REPO_SOURCE="${COPY_REPO_SOURCE:-1}"
NANO_DSA_ENFORCE_EAGER="${NANO_DSA_ENFORCE_EAGER:-1}"
NANO_DSA_CUDAGRAPH_MODE="${NANO_DSA_CUDAGRAPH_MODE:-}"

mkdir -p "${RUN_DIR}" "${REPO_DIR}/logs"
export BASELINE COPY_REPO_SOURCE DSA_DENSE_PREFILL_THRESHOLD
export DSA_PATH_DEBUG_PRINT_LIMIT GENERATION_SEED GPU_MEMORY_UTILIZATION
export IMAGE MAX_MODEL_LEN MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS MAX_TOKENS
export MODEL_DIR PATCH_DSA_MARKERS REPO_DIR RUN_COMPARE RUN_DIR
export NANO_DSA_CUDAGRAPH_MODE NANO_DSA_ENFORCE_EAGER
export TARGET_PROMPT_TOKENS TP_SIZE

{
  echo "job_id=${SLURM_JOB_ID:-manual}"
  echo "role=current"
  echo "host=$(hostname)"
  echo "tp_size=${TP_SIZE}"
  echo "repo=${REPO_DIR}"
  echo "image=${IMAGE}"
  echo "model=${MODEL_DIR}"
  echo "baseline=${BASELINE}"
  echo "run_dir=${RUN_DIR}"
  echo "target_prompt_tokens=${TARGET_PROMPT_TOKENS}"
  echo "max_tokens=${MAX_TOKENS}"
  echo "max_model_len=${MAX_MODEL_LEN}"
  echo "max_num_seqs=${MAX_NUM_SEQS}"
  echo "max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
  echo "dsa_dense_prefill_threshold=${DSA_DENSE_PREFILL_THRESHOLD}"
  echo "nano_dsa_enforce_eager=${NANO_DSA_ENFORCE_EAGER}"
  echo "nano_dsa_cudagraph_mode=${NANO_DSA_CUDAGRAPH_MODE}"
  git -C "${REPO_DIR}" rev-parse HEAD
} | tee "${RUN_DIR}/run_metadata.txt"

srun \
  --container-image="${IMAGE}" \
  --container-mounts="${REPO_DIR}:/workspace/vllm-src,${RUN_DIR}:/workspace/out,/lustre:/lustre,/scratch:/scratch" \
  bash -lc '
    set -euo pipefail

    cd /tmp
    unset PYTHONPATH
    export HOME="/tmp/${USER}/gd-sparse-agree-tp${TP_SIZE}-home"
    export XDG_CACHE_HOME="/tmp/${USER}/gd-sparse-agree-tp${TP_SIZE}-cache"
    export UV_CACHE_DIR="/tmp/${USER}/gd-sparse-agree-tp${TP_SIZE}-uv-cache"
    export FLASHINFER_WORKSPACE_DIR="/tmp/${USER}/gd-sparse-agree-tp${TP_SIZE}-cache/flashinfer"
    export VLLM_USAGE_SOURCE="generation-disagreement-sparse-agreement-tp${TP_SIZE}"
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export VLLM_WORKER_MULTIPROC_METHOD=fork
    export VLLM_USE_DEEP_GEMM=0
    export VLLM_MOE_USE_DEEP_GEMM=0
    export VLLM_DEEP_GEMM_WARMUP=skip
    export VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16
    export VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K=128
    export VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=1
    export VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA=1
    export VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ=1
    export VLLM_NEMOTRON_H_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA=1
    export VLLM_NEMOTRON_H_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA=1
    export VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE=4096
    export VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING=1
    export VLLM_NEMOTRON_H_DSA_USE_TRITON_BATCHED_SUMMARIES=1
    export VLLM_NEMOTRON_H_DSA_DENSE_PREFILL_KV_THRESHOLD_TOKENS="${DSA_DENSE_PREFILL_THRESHOLD}"
    export VLLM_NEMOTRON_H_DSA_PATH_DEBUG_PRINT_LIMIT="${DSA_PATH_DEBUG_PRINT_LIMIT}"

    mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${UV_CACHE_DIR}" \
      "${FLASHINFER_WORKSPACE_DIR}" /workspace/out

    /usr/local/bin/uv venv \
      --system-site-packages \
      --python /usr/bin/python3 \
      "gd-sparse-agree-tp${TP_SIZE}-venv"
    PY="gd-sparse-agree-tp${TP_SIZE}-venv/bin/python"

    SITE="$("${PY}" - <<'"'"'PY_SITE'"'"'
import importlib.util
import pathlib

spec = importlib.util.find_spec("vllm")
if spec is None or spec.origin is None:
    raise SystemExit("vllm package not found")
print(pathlib.Path(spec.origin).parent.parent)
PY_SITE
)"

    if [[ "${COPY_REPO_SOURCE}" == "1" ]]; then
      cp -a /workspace/vllm-src/vllm "${SITE}/"
    fi

    if [[ "${PATCH_DSA_MARKERS}" == "1" ]]; then
      "${PY}" \
        /workspace/vllm-src/scripts/generation_disagreement/patch_dsa_attention_markers.py \
        "${SITE}/vllm/model_executor/models/nemotron_h_dsa_attention_legacy.py" \
        | tee /workspace/out/dsa_marker_patch.txt
    fi

    sha256sum \
      /workspace/vllm-src/vllm/model_executor/models/nemotron_h.py \
      /workspace/vllm-src/vllm/model_executor/models/nemotron_h_dsa_attention_legacy.py \
      "${SITE}/vllm/model_executor/models/nemotron_h.py" \
      "${SITE}/vllm/model_executor/models/nemotron_h_dsa_attention_legacy.py" \
      | tee /workspace/out/source_hashes.txt

    "${PY}" - <<'"'"'PY_CHECK'"'"' | tee /workspace/out/import_check.txt
import importlib
import inspect
import os
import vllm
import vllm._C

modules = [
    "vllm.model_executor.models.nemotron_h",
    "vllm.model_executor.models.nemotron_h_dsa_attention_legacy",
    "vllm.model_executor.models.nemotron_h_dsa_triton_scoring",
    "vllm.model_executor.models.nemotron_h_dsa_triton_summaries",
]
print("vllm_file", vllm.__file__)
print("vllm_C_file", vllm._C.__file__)
print("dsa_chunk_top_k_env", os.environ["VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K"])
print(
    "dsa_dense_prefill_threshold_env",
    os.environ["VLLM_NEMOTRON_H_DSA_DENSE_PREFILL_KV_THRESHOLD_TOKENS"],
)
print(
    "dsa_path_debug_print_limit_env",
    os.environ["VLLM_NEMOTRON_H_DSA_PATH_DEBUG_PRINT_LIMIT"],
)
for module_name in modules:
    module = importlib.import_module(module_name)
    print(module_name, inspect.getfile(module))
PY_CHECK

    limit_args=()
    if [[ -n "${GENERATION_LIMIT:-}" ]]; then
      limit_args=(--limit "${GENERATION_LIMIT}")
    fi
    graph_args=()
    if [[ "${NANO_DSA_ENFORCE_EAGER}" == "1" ]]; then
      graph_args=(--enforce-eager)
    elif [[ -n "${NANO_DSA_CUDAGRAPH_MODE}" ]]; then
      graph_args=(
        --compilation-config
        "{\"mode\":3,\"cudagraph_mode\":\"${NANO_DSA_CUDAGRAPH_MODE}\"}"
      )
    fi

    "${PY}" /workspace/vllm-src/scripts/generation_disagreement/generate.py \
      --model "${MODEL_DIR}" \
      --output-file /workspace/out/current.jsonl \
      "${limit_args[@]}" \
      --target-prompt-tokens "${TARGET_PROMPT_TOKENS}" \
      --max-tokens "${MAX_TOKENS}" \
      --generation-seed "${GENERATION_SEED}" \
      --disable-tqdm \
      --overwrite \
      --trust-remote-code \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
      --attention-backend FLASH_ATTN \
      --enable-expert-parallel \
      --model-loader-extra-config "{\"enable_multithread_load\":true,\"num_threads\":96}" \
      --max-num-seqs "${MAX_NUM_SEQS}" \
      --mamba-ssm-cache-dtype float32 \
      --no-enable-prefix-caching \
      --enable-chunked-prefill \
      "${graph_args[@]}" \
      --block-size 16 \
      --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --seed 0 \
      2>&1 | tee /workspace/out/current.log

    "${PY}" /workspace/vllm-src/scripts/generation_disagreement/check_dsa_path_markers.py \
      /workspace/out/current.log \
      --markers-output /workspace/out/path_markers.txt \
      --check-output /workspace/out/path_marker_check.txt

    if [[ "${RUN_COMPARE}" == "1" ]]; then
      if [[ ! -f "${BASELINE}" ]]; then
        echo "Comparison baseline not found: ${BASELINE}" >&2
        exit 1
      fi
      "${PY}" /workspace/vllm-src/scripts/generation_disagreement/compare.py \
        "${BASELINE}" \
        /workspace/out/current.jsonl \
        --worst 20 \
        --early-threshold 10 \
        --output-json /workspace/out/compare_summary.json \
        2>&1 | tee /workspace/out/compare.txt
    fi
  '
