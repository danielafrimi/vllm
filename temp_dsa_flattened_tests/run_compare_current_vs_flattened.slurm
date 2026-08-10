#!/usr/bin/env bash
#SBATCH --account=nemotron_compress_dev
#SBATCH --partition=batch
#SBATCH --job-name=dsa-flat-compare
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-dsa-flat-compare-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/fsw/portfolios/coreai/users/otropp/mycode/vllm}"
cd "${REPO_DIR}"
export VLLM_SOURCE_REPO="${REPO_DIR}"

PY="${PY:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x /scratch/fsw/portfolios/coreai/users/otropp/mycode/SparseAttention/.venv/bin/python ]]; then
    PY=/scratch/fsw/portfolios/coreai/users/otropp/mycode/SparseAttention/.venv/bin/python
  elif [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
  else
    echo "Set PY to a vLLM Python interpreter, for example: PY=.venv/bin/python sbatch $0" >&2
    exit 1
  fi
fi

"${PY}" temp_dsa_flattened_tests/compare_current_vs_flattened_page_table_fa.py \
  --dtype bf16 \
  --block-size 16 \
  --kv-heads 2 \
  --heads 32 \
  --head-dim 64 \
  --top-k 8 \
  --fa-version 2 \
  --warmup-iters 3 \
  --bench-iters 10
