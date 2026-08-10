#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
MODEL_DIR="${MODEL_DIR:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints/nemo3_repr_heads_128k_top2k_q128_sparse_topk2048_nemotron_h_puzzle}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
TP_SIZE="${TP_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-dsa-nemotron-h-puzzle}"
MOE_BACKEND="${MOE_BACKEND:-triton}"
DTYPE="${DTYPE:-}"
MAMBA_SSM_CACHE_DTYPE="${MAMBA_SSM_CACHE_DTYPE:-float32}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"

cd "${REPO_DIR}"
mkdir -p logs

LOG_FILE="${LOG_FILE:-logs/dsa_nemotron_h_puzzle_server_${SLURM_JOB_ID:-manual}.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "log: ${REPO_DIR}/${LOG_FILE}"
date
hostname
nvidia-smi

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/${USER}/uv-cache}"
mkdir -p "${UV_CACHE_DIR}"

export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export VLLM_MOE_USE_DEEP_GEMM="${VLLM_MOE_USE_DEEP_GEMM:-0}"
export VLLM_DEEP_GEMM_WARMUP="${VLLM_DEEP_GEMM_WARMUP:-skip}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    uv venv --python 3.12
    VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
fi

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then
        PYTHON_BIN=".venv/bin/python"
    elif [[ -x "/usr/local/bin/python3.12" ]]; then
        PYTHON_BIN="/usr/local/bin/python3.12"
    else
        PYTHON_BIN="python3"
    fi
fi

args=(
    --model "${MODEL_DIR}"
    --trust-remote-code
    --host "${HOST}"
    --port "${PORT}"
    --served-model-name "${SERVED_MODEL_NAME}"
    --tensor-parallel-size "${TP_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --mamba-ssm-cache-dtype "${MAMBA_SSM_CACHE_DTYPE}"
    --moe-backend "${MOE_BACKEND}"
    --no-enable-log-requests
    --max-num-seqs 1
)

if [[ -n "${DTYPE}" ]]; then
    args+=(--dtype "${DTYPE}")
fi

if [[ "${ENFORCE_EAGER}" == "1" ]]; then
    args+=(--enforce-eager)
fi

"${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server "${args[@]}"
