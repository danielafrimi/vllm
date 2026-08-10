#!/usr/bin/env bash
#SBATCH --account=nemotron_compress_dev
#SBATCH --partition=batch
#SBATCH --job-name=sparse-vs-baseline
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=slurm-sparse-vs-baseline-%j.out

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"
cd "${SCRIPT_DIR}"

PY="${PY:-}"
if [[ -z "${PY}" ]]; then
  if [[ -x ../.venv/bin/python ]]; then
    PY=../.venv/bin/python
  elif [[ -x ../../SparseAttention/.venv/bin/python ]]; then
    PY=../../SparseAttention/.venv/bin/python
  else
    echo "Set PY to a vLLM Python interpreter, for example: PY=../.venv/bin/python sbatch $0" >&2
    exit 1
  fi
fi
NSYS="${NSYS:-../tools/nsight-systems-cli/opt/nvidia/nsight-systems-cli/2026.1.1/bin/nsys}"
if [[ ! -x "${NSYS}" && -x ../../SparseAttention/tools/nsight-systems-cli/opt/nvidia/nsight-systems-cli/2026.1.1/bin/nsys ]]; then
  NSYS=../../SparseAttention/tools/nsight-systems-cli/opt/nvidia/nsight-systems-cli/2026.1.1/bin/nsys
fi
STAMP="${SLURM_JOB_ID:-manual}"

mkdir -p profiles

DECODE_REPORT="profiles/final_profile_decode_sparse_vs_baseline_b64_kv2_64k_ab512_${STAMP}"
PREFILL_REPORT="profiles/final_profile_chunk_prefill_sparse_vs_baseline_b1_q8192_128k_ab1024_${STAMP}"

"${NSYS}" profile \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  --export=sqlite \
  --output="${DECODE_REPORT}" \
  "${PY}" benchmark_vllm_decode_mask_sparse.py \
    --batch-size 64 \
    --context-len 65536 \
    --block-size 16 \
    --heads 32 \
    --kv-heads 2 \
    --head-dim 64 \
    --active-blocks 512 \
    --dtype bf16 \
    --fa-version 2 \
    --num-splits 0 \
    --warmup-iters 1 \
    --bench-iters 5 \
    --sparse-impl flattened \
    --separation-sleep 0.5

"${NSYS}" profile \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  --export=sqlite \
  --output="${PREFILL_REPORT}" \
  "${PY}" benchmark_vllm_chunk_prefill_sparse_flattened.py \
    --batch-size 1 \
    --context-len 131072 \
    --query-len 8192 \
    --block-size 16 \
    --heads 32 \
    --kv-heads 2 \
    --head-dim 64 \
    --active-blocks-per-token 1024 \
    --dtype bf16 \
    --fa-version 2 \
    --warmup-iters 3 \
    --bench-iters 1 \
    --skip-transposed \
    --skip-compare

echo "FINAL DECODE PROFILE: ${DECODE_REPORT}.nsys-rep"
echo "FINAL DECODE SQLITE:  ${DECODE_REPORT}.sqlite"
echo "FINAL PREFILL PROFILE: ${PREFILL_REPORT}.nsys-rep"
echo "FINAL PREFILL SQLITE:  ${PREFILL_REPORT}.sqlite"
