#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
BASE_IMAGE="${BASE_IMAGE:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/vllm_repos/vllm_v0.20.1/outputs/containers/vllm-openai_v0.20.1_nemotron-h-dsa-routeb-prefill-page-table-fa_20260530_033300.sqsh}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/moonshot_microbench}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SPLIT_TOP_CHUNKS="${SPLIT_TOP_CHUNKS:-16 32 64 128 256}"
COMMON_ARGS="${COMMON_ARGS:---bench --kernels dense,splitk --key-len 131072 --q-len 8192 --top-chunks 1024 --query-chunk-size 1024 --warmup 2 --iters 3}"

mkdir -p "${LOG_DIR}"

echo "stamp: ${STAMP}"
echo "repo: ${REPO_DIR}"
echo "image: ${BASE_IMAGE}"
echo "logs: ${LOG_DIR}"
echo "split_top_chunks: ${SPLIT_TOP_CHUNKS}"
echo "common_args: ${COMMON_ARGS}"

srun \
  --account="${SLURM_ACCOUNT:-nemotron_compress_dev}" \
  --partition="${SLURM_PARTITION:-interactive}" \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node="${GPUS_PER_NODE:-1}" \
  --time="${SLURM_TIME:-01:30:00}" \
  --container-image="${BASE_IMAGE}" \
  --container-mounts="${REPO_DIR}:/workspace/vllm-src,${LOG_DIR}:/logs" \
  bash -lc '
    set -euo pipefail
    SITE=$(python3 - <<PY
import site
print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))
PY
)
    cp -a /workspace/vllm-src/vllm "$SITE"/
    cd /tmp
    export VLLM_DSA_BENCH_USE_SITE_PACKAGE=1
    export VLLM_NEMOTRON_H_DSA_TIMING=0
    for split in '"${SPLIT_TOP_CHUNKS}"'; do
      echo "=== split_top_chunks=${split} ==="
      python3 /workspace/vllm-src/tools/bench_dsa_moonshot_prefill.py \
        '"${COMMON_ARGS}"' \
        --split-top-chunks "${split}" \
        --json-out /logs/moonshot_splitk_${split}_'"${STAMP}"'.json \
        2>&1 | tee /logs/moonshot_splitk_${split}_'"${STAMP}"'.log
    done
  '
