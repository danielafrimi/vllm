#!/usr/bin/env bash
#SBATCH --account=nemotron_compress_dev
#SBATCH --partition=batch
#SBATCH --job-name=sparse-core-correctness
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=slurm-sparse-core-correctness-%j.out

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

"${PY}" -m py_compile \
  sparse_attention_core.py \
  test_decode_causal_sparse_correctness.py \
  benchmark_vllm_decode_mask_sparse.py \
  benchmark_vllm_chunk_prefill_sparse_flattened.py

"${PY}" test_decode_causal_sparse_correctness.py \
  --past-lens 1024 1031 2049 4111 \
  --batch-size 2 \
  --block-size 16 \
  --heads 4 \
  --kv-heads 2 \
  --head-dim 64 \
  --active-blocks 8 \
  --dtype bf16 \
  --fa-version 2

"${PY}" test_decode_causal_sparse_correctness.py \
  --past-lens 65536 65543 65551 \
  --batch-size 1 \
  --ragged-past-lens 65536 65543 65551 \
  --block-size 16 \
  --heads 32 \
  --kv-heads 2 \
  --head-dim 64 \
  --active-blocks 512 \
  --dtype bf16 \
  --fa-version 2 \
  --skip-buggy-check

"${PY}" benchmark_vllm_decode_mask_sparse.py \
  --batch-size 2 \
  --context-len 1024 \
  --block-size 16 \
  --heads 4 \
  --kv-heads 2 \
  --head-dim 64 \
  --active-blocks 8 \
  --dtype bf16 \
  --fa-version 2 \
  --warmup-iters 1 \
  --bench-iters 1 \
  --sparse-impl flattened \
  --skip-dense \
  --check-flattened-sparse-correctness \
  --separation-sleep 0

"${PY}" benchmark_vllm_chunk_prefill_sparse_flattened.py \
  --batch-size 1 \
  --context-len 2048 \
  --query-len 256 \
  --block-size 16 \
  --heads 4 \
  --kv-heads 2 \
  --head-dim 64 \
  --active-blocks-per-token 16 \
  --dtype bf16 \
  --fa-version 2 \
  --warmup-iters 0 \
  --bench-iters 1 \
  --skip-dense \
  --check-legacy-naive
