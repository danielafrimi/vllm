#!/usr/bin/env bash
#SBATCH --account=nemotron_compress_dev
#SBATCH --partition=batch
#SBATCH --job-name=dsa-summ-nsys
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-dsa-summ-nsys-%j.out

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
NSYS_2026="${REPO_DIR}/temp_dsa_flattened_tests/nsight_2026_1/opt/nvidia/nsight-systems-cli/2026.1.1/bin"
export PATH="$(dirname "${PY}"):${PATH}"
if [[ -x "${NSYS_2026}/nsys" ]]; then
  export PATH="${NSYS_2026}:${PATH}"
fi
nvidia-smi || true

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck source=/etc/profile.d/modules.sh
  source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
  module load nsight-systems 2>/dev/null || \
    module load nsys 2>/dev/null || \
    module load cuda/13.0 2>/dev/null || \
    module load cuda 2>/dev/null || true
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

if ! command -v nsys >/dev/null 2>&1; then
  for candidate in \
    /usr/local/cuda/bin/nsys \
    /usr/local/cuda-13.0/bin/nsys \
    /opt/nvidia/nsight-systems/*/target-linux-x64/nsys \
    /opt/nvidia/nsight-systems/*/bin/nsys; do
    if [[ -x "${candidate}" ]]; then
      export PATH="$(dirname "${candidate}"):${PATH}"
      break
    fi
  done
fi

echo "CUDA_HOME=${CUDA_HOME:-unset}"
command -v nvcc || true
nvcc --version || true
command -v nsys || { echo "ERROR: nsys not found on compute node" >&2; exit 1; }
nsys --version
"${PY}" -c 'import torch; print("torch", torch.__version__); print("cuda_available", torch.cuda.is_available()); print("cuda_device_count", torch.cuda.device_count())'

REPORT_BASE="temp_dsa_flattened_tests/nsys-dsa-summ-${SLURM_JOB_ID}"
rm -f "${REPORT_BASE}.nsys-rep" "${REPORT_BASE}.sqlite"

nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --force-overwrite=true \
  --output="${REPORT_BASE}" \
  "${PY}" temp_dsa_flattened_tests/nsys_repo_vs_cuda_block_summaries_jit.py \
    --dtype bf16 \
    --batch 16 \
    --target-len 4096 \
    --ragged-window 768 \
    --block-size 16 \
    --kv-heads 8 \
    --head-dim 128 \
    --q-indexer-dim 96 \
    --num-physical-blocks 8192 \
    --warmup-iters 3 \
    --repeat 1 \
    --seed 123

nsys export --type sqlite --force-overwrite=true \
  --output="${REPORT_BASE}.sqlite" \
  "${REPORT_BASE}.nsys-rep"

"${PY}" temp_dsa_flattened_tests/extract_nsys_nvtx_kernel_stats.py \
  "${REPORT_BASE}.sqlite"
