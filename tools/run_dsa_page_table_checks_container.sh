#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
MODEL_PARENT="${MODEL_PARENT:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints}"
MODEL_NAME="${MODEL_NAME:-nano-dsa-16chunk_size-1024chunks}"
BASE_IMAGE="${BASE_IMAGE:-/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/containers/vllm/vllm-openai_v0.20.1.sqsh}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/dsa_page_table_checks_${STAMP}.log}"

mkdir -p "${LOG_DIR}"

echo "log: ${LOG_FILE}"
echo "repo: ${REPO_DIR}"
echo "model: ${MODEL_PARENT}/${MODEL_NAME}"

srun \
  --account="${SLURM_ACCOUNT:-nemotron_compress_dev}" \
  --partition="${SLURM_PARTITION:-interactive}" \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node="${GPUS_PER_NODE:-1}" \
  --time="${SLURM_TIME:-00:30:00}" \
  --container-image="${BASE_IMAGE}" \
  --container-mounts="${REPO_DIR}:/workspace/vllm-src,${MODEL_PARENT}:/models" \
  bash -lc '
    set -euo pipefail
    SITE=$(python3 - <<PY
import site
print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))
PY
)
    cp -a /workspace/vllm-src/vllm "$SITE"/
    python3 /workspace/vllm-src/tools/validate_dsa_page_table_fa.py \
      --model "/models/'"${MODEL_NAME}"'" \
      --tp-size "${TP_SIZE:-8}"
  ' 2>&1 | tee "${LOG_FILE}"
