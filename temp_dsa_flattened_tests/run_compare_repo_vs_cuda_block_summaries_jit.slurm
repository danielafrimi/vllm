#!/usr/bin/env bash
#SBATCH --account=nemotron_compress_dev
#SBATCH --partition=batch
#SBATCH --job-name=dsa-repo-summ
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-dsa-repo-summ-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/fsw/portfolios/coreai/users/otropp/mycode/vllm}"
cd "${REPO_DIR}"

PY="${PY:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x /scratch/fsw/portfolios/coreai/users/otropp/mycode/SparseAttention/.venv/bin/python ]]; then
    PY=/scratch/fsw/portfolios/coreai/users/otropp/mycode/SparseAttention/.venv/bin/python
  elif [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
  else
    echo "Set PY to a Python interpreter with torch, for example: PY=/path/to/python sbatch $0" >&2
    exit 1
  fi
fi

echo "host=$(hostname)"
echo "python=${PY}"
nvidia-smi || true

if [[ -z "${CUDA_HOME:-}" ]]; then
  PY_CUDA_HOME="$("${PY}" - <<'PY'
import site
from pathlib import Path

for site_dir in site.getsitepackages():
    cuda_home = Path(site_dir) / "nvidia" / "cu13"
    if (cuda_home / "bin" / "nvcc").exists():
        print(cuda_home)
        break
PY
)"
  if [[ -n "${PY_CUDA_HOME}" ]]; then
    export CUDA_HOME="${PY_CUDA_HOME}"
  fi
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

echo "CUDA_HOME=${CUDA_HOME:-unset}"
command -v nvcc || true
nvcc --version || true
"${PY}" -c 'import torch; print("torch", torch.__version__); print("cuda_available", torch.cuda.is_available()); print("cuda_device_count", torch.cuda.device_count())'

"${PY}" temp_dsa_flattened_tests/compare_repo_vs_cuda_block_summaries_jit.py \
  --dtype bf16 \
  --seed 123 \
  --sweep \
  --include-alias-diagnostic
