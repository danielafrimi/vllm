#!/usr/bin/env bash
#SBATCH --account=nemotron_compress_dev
#SBATCH --partition=batch
#SBATCH --job-name=dsa-summ-jit
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-dsa-summ-jit-%j.out

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

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/etc/profile.d/modules.sh
  source /etc/profile.d/modules.sh
fi

if command -v module >/dev/null 2>&1; then
  module load cuda/13.0 2>/dev/null || \
    module load cuda/13 2>/dev/null || \
    module load cuda 2>/dev/null || true
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  if command -v nvcc >/dev/null 2>&1; then
    CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    export CUDA_HOME
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  elif [[ -x /usr/local/cuda-13.0/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-13.0
  fi
fi

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

"${PY}" temp_dsa_flattened_tests/test_cuda_block_summaries_jit.py \
  --dtype bf16 \
  --batch 6 \
  --num-physical-blocks 64 \
  --block-size 16 \
  --kv-heads 4 \
  --head-dim 80 \
  --q-indexer-dim 64 \
  --seed 123 \
  --sweep
