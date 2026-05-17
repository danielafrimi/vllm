# FlashInfer Mamba Checkpointing SSM Cache Notes

This branch experiments with FlashInfer Mamba SSM cache variants for
Nemotron-H / Mamba2 layers, including Igor Shovkun's
`flashinfer.mamba.checkpointing_ssu` fused replay kernel.

## Environment

- vLLM branch: `flashinfer-checkpointing-ssm-cache`
- Model: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`
- Python env: `/my_home/venvs/vllm3`
- FlashInfer workspace: `/my_home/vllm-scratch/fi-cu130`
- vLLM scratch/cache root: `/my_home/vllm-scratch`
- Eval env: `/my_home/venvs/eval_venv`

## Kernel Paths

- Old FlashInfer Mamba path:
  `flashinfer.mamba.selective_state_update`
- New fused replay path:
  `flashinfer.mamba.checkpointing_ssu`

The new path is currently restricted to MTP/speculative decode calls in this
branch. Normal non-MTP decode falls back to the existing SSU path while fused
replay bookkeeping is validated.

## Important Integration Details

- `checkpointing_ssu` expects a 1D `state_batch_indices` vector, while vLLM MTP
  metadata uses a 2D state-index table. The adapter resolves the same initial
  slot as old FI MTP:
  `state_indices[seq, max(num_accepted_tokens[seq] - 1, 0)]`.
- `checkpointing_ssu` mutates replay payload buffers (`old_x`, `old_B`,
  `old_dt`, `old_cumAdt`) but treats `cache_buf_idx` and
  `prev_num_accepted_tokens` as inputs. vLLM updates those trackers after the
  kernel call.
- Tracker updates must be CUDA-graph-safe. The branch uses a small Triton kernel
  for these updates rather than Python boolean indexing, which is not allowed
  during graph capture.
- Varlen MTP passes packed tokens with `cu_seqlens`; `max_seqlen` must be the
  per-sequence maximum token count, not the total packed token count.

## Serve Commands

Old FI MTP baseline:

```bash
FLASHINFER_WORKSPACE_BASE=/my_home/vllm-scratch/fi-cu130 \
VLLM_CACHE_ROOT=/my_home/vllm-scratch/vllm-cu130-mtp-oldfi \
HF_HOME=/my_home/vllm-scratch/hf \
CUDA_VISIBLE_DEVICES=3 \
/my_home/venvs/vllm3/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --no-mamba-flashinfer-checkpointing-ssu \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --port 8054 \
  --served-model-name nemotron-fp16sr-fi-oldssu-mtp-cu130
```

New FI fused replay MTP:

```bash
FLASHINFER_WORKSPACE_BASE=/my_home/vllm-scratch/fi-cu130 \
VLLM_CACHE_ROOT=/my_home/vllm-scratch/vllm-cu130-mtp-newfi-graphsafe \
HF_HOME=/my_home/vllm-scratch/hf \
CUDA_VISIBLE_DEVICES=2 \
/my_home/venvs/vllm3/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --mamba-backend flashinfer \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --mamba-flashinfer-checkpointing-ssu \
  --mamba-checkpoint-interval 16 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --port 8055 \
  --served-model-name nemotron-fp16sr-fi-newssu-mtp-cu130
```

Triton Mamba MTP control:

```bash
VLLM_CACHE_ROOT=/my_home/vllm-scratch/vllm-cu130-mtp-triton \
HF_HOME=/my_home/vllm-scratch/hf \
CUDA_VISIBLE_DEVICES=1 \
/my_home/venvs/vllm3/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --mamba-backend triton \
  --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}' \
  --port 8056 \
  --served-model-name nemotron-fp16sr-triton-mtp-cu130
```

## GSM8K Results

| Variant | Port | Mamba path | strict-match | flexible-extract |
|---|---:|---|---:|---:|
| Non-MTP fp16+SR | 8053 | old FI `selective_state_update` | 0.9128 | 0.9242 |
| Non-MTP fp16+SR | 8052 | `checkpointing_ssu`, interval 1 | 0.0728 | 0.1175 |
| MTP fp16+SR | 8054 | old FI `selective_state_update` | 0.5572 | 0.6967 |
| MTP fp16+SR | 8055 | `checkpointing_ssu` fused replay, interval 16 | 0.0000 | 0.0000 |

The new fused replay MTP server on port `8055` reached readiness with CUDA
graphs enabled and passed a small arithmetic smoke test (`19 * 21 -> 399`), but
GSM8K collapsed to zero. This indicates the current vLLM adapter is still not
semantically equivalent to the old MTP SSU path despite serving successfully.

The Triton Mamba MTP control on port `8056` was launched to determine whether
the MTP quality drop is specific to the old FlashInfer MTP kernel or more
general to the MTP configuration. At the time of this note, it was still in
startup/warmup and had not reached readiness.
