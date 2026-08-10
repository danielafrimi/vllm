#!/usr/bin/env bash
#SBATCH --account=nemotron_n4_compress
#SBATCH --partition=batch
#SBATCH --job-name=dsa-refactor-parallel
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=temp_dsa_refactor_tests/slurm-dsa-refactor-parallel-%j.out
#SBATCH --error=temp_dsa_refactor_tests/slurm-dsa-refactor-parallel-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/lustre/fsw/portfolios/coreai/users/otropp/mycode/vllm/outputs/worktrees/nemotron-h-dsa-simplify}"
cd "${REPO_DIR}"
export VLLM_SOURCE_REPO="${REPO_DIR}"

PY="${PY:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
  else
    echo "Set PY to a vLLM Python interpreter, for example: PY=.venv/bin/python sbatch $0" >&2
    exit 1
  fi
fi

CPU_WORKERS="${CPU_WORKERS:-8}"
CUDA_WORKERS="${CUDA_WORKERS:-2}"
OUT_DIR="${OUT_DIR:-temp_dsa_refactor_tests/parallel-refactor-parity-${SLURM_JOB_ID:-manual}}"

"${PY}" temp_dsa_refactor_tests/run_refactored_attention_parity_parallel.py \
  --python "${PY}" \
  --repo-dir "${REPO_DIR}" \
  --output-dir "${OUT_DIR}" \
  --pytorch-workers "${CPU_WORKERS}" \
  --efficient-workers "${CUDA_WORKERS}"
