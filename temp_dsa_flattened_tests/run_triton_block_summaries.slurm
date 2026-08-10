#!/usr/bin/env bash
#SBATCH --account=nemotron_n4_compress
#SBATCH --partition=batch_long
#SBATCH --job-name=dsa-summ-triton
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-dsa-summ-triton-%j.out

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/fsw/portfolios/coreai/users/otropp/mycode/vllm}"
cd "${REPO_DIR}"

VLLM_IMAGE="${VLLM_IMAGE:-}"
if [[ -z "${VLLM_IMAGE}" ]]; then
  VLLM_COMMIT="$(git rev-parse --short=10 HEAD 2>/dev/null || printf unknown)"
  VLLM_IMAGE="$(
    find "${REPO_DIR}/outputs/containers" -maxdepth 1 -type f \
      -name "vllm-v0.22.0-current-overlay-${VLLM_COMMIT}-*.sqsh" \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | sed -n '1s/^[^ ]* //p'
  )"
fi

if [[ -n "${VLLM_IMAGE}" && -f "${VLLM_IMAGE}" ]]; then
  echo "container_image=${VLLM_IMAGE}"
  srun \
    --container-image="${VLLM_IMAGE}" \
    --container-mounts="${REPO_DIR}:/workspace/vllm-src" \
    bash -lc '
      set -euo pipefail
      cd /tmp
      unset PYTHONPATH
      export HOME=/tmp/vllm-triton-home
      export TRITON_CACHE_DIR=/tmp/vllm-triton-cache
      mkdir -p "${HOME}" "${TRITON_CACHE_DIR}"
      echo "host=$(hostname)"
      nvidia-smi || true
      python3 - <<PY
import inspect
import torch
import triton
from vllm.model_executor.models import nemotron_h_dsa_triton_summaries
print("torch", torch.__version__)
print("triton", triton.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("summary_helper", inspect.getfile(nemotron_h_dsa_triton_summaries))
PY
      python3 /workspace/vllm-src/temp_dsa_flattened_tests/test_triton_block_summaries.py \
        --dtype bf16 \
        --batch 8 \
        --num-physical-blocks 256 \
        --block-size 16 \
        --kv-heads 4 \
        --head-dim 80 \
        --q-indexer-dim 64 \
        --seed 123 \
        --sweep
    '
  exit 0
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

PY="${PY:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x /scratch/fsw/portfolios/coreai/users/otropp/mycode/SparseAttention/.venv/bin/python ]]; then
    PY=/scratch/fsw/portfolios/coreai/users/otropp/mycode/SparseAttention/.venv/bin/python
  elif [[ -x .venv/bin/python ]]; then
    PY=.venv/bin/python
  else
    echo "Set PY to a Python interpreter with torch+triton, for example: PY=/path/to/python sbatch $0" >&2
    exit 1
  fi
fi

echo "host=$(hostname)"
echo "python=${PY}"
echo "PYTHONPATH=${PYTHONPATH}"
nvidia-smi || true
"${PY}" -c 'import torch, triton; print("torch", torch.__version__); print("triton", triton.__version__); print("cuda_available", torch.cuda.is_available()); print("cuda_device_count", torch.cuda.device_count())'

"${PY}" temp_dsa_flattened_tests/test_triton_block_summaries.py \
  --dtype bf16 \
  --batch 8 \
  --num-physical-blocks 256 \
  --block-size 16 \
  --kv-heads 4 \
  --head-dim 80 \
  --q-indexer-dim 64 \
  --seed 123 \
  --sweep
