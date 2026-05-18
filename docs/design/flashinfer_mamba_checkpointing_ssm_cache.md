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
| MTP fp16+SR | 8056 | Triton Mamba backend | 0.5572 | 0.6937 |

The new fused replay MTP server on port `8055` reached readiness with CUDA
graphs enabled and passed a small arithmetic smoke test (`19 * 21 -> 399`), but
GSM8K collapsed to zero. This indicates the current vLLM adapter is still not
semantically equivalent to the old MTP SSU path despite serving successfully.

The Triton Mamba MTP control on port `8056` was launched to determine whether
the MTP quality drop is specific to the old FlashInfer MTP kernel or more
general to the MTP configuration. Its GSM8K result matches the old FI MTP
baseline, suggesting the quality drop versus non-MTP is caused by the MTP
configuration/model behavior rather than the old FI MTP kernel.

Eager debug logging showed the first fused-replay adapter bug:
`num_accepted_tokens=[1]` while scheduled `seq_lens=[6]`, but the adapter updated
`prev_num_accepted_tokens` from 0 to 6. That replays all scheduled MTP tokens as
accepted tokens. The tracker update was changed to use scheduled sequence length
only for checkpoint/no-checkpoint branch selection, and to update the replay
count using `num_accepted_tokens`.

After that accepted-count fix, the fused-replay server still served with CUDA
graphs and no startup errors, but a quick GSM8K smoke (`--limit 50`) remained at
`strict=0.0000`, `flexible=0.0000`. So the accepted-count bug was real but not
the only semantic mismatch. The current adapter still does not reproduce old FI
MTP behavior.

## May 17 Status

What we established today:

- STP with the new checkpointing kernel gave bad quality with int8 SSM + SR.
- STP with fp16 SSM + SR was also bad when routed through
  `checkpointing_ssu`, while fp16 SSM + SR with the old SSU path was healthy.
  That points to the new checkpoint/replay integration path, not only 8-bit
  quantization.
- For MTP, old FlashInfer SSU and Triton Mamba are aligned:
  old FI MTP was `strict=0.5572`, `flexible=0.6967`; Triton MTP was
  `strict=0.5572`, `flexible=0.6937`.
- The fused-replay MTP server reaches readiness with CUDA graphs and passes a
  short arithmetic smoke test, but GSM8K still collapses to zero.

Implementation state:

- vLLM now allocates replay state in the Mamba2 cache tuple:
  `old_x`, `old_B`, `old_dt`, `old_cumAdt`, `cache_buf_idx`, and
  `prev_num_accepted_tokens`.
- The accepted-token tracker is maintained per cache slot and updated with a
  CUDA-graph-safe Triton kernel.
- A regression test covers the confirmed tracker bug: scheduled sequence length
  is used for checkpoint rotation, while `num_accepted_tokens` is used for the
  replay count.
- The fused-replay adapter now maps vLLM's 2D MTP state table to a stable
  per-request replay slot (`state_indices[:, 0]`). The previous adapter used
  `num_accepted_tokens - 1`, which matches old FI's materialized speculative
  state-slot selection but can move replay counters/buffers between token slots.
  The stable slot vector is materialized as contiguous because
  `checkpointing_ssu` validates `state_batch_indices.is_contiguous()`.

Remaining suspected bug:

- `checkpointing_ssu` appends the current scheduled MTP token window into the
  replay buffers, but vLLM only learns how many of those tokens were accepted on
  the next speculative step. The adapter still needs to make the replay buffer
  contents and `prev_num_accepted_tokens` advance at the same semantic point as
  old FI MTP state-slot selection.
- The next debugging step should be a small repeated-step parity test:
  old FI MTP state progression versus fused-replay MTP over an accepted-token
  pattern such as `[1, 3, 2]`, checking `prev_num_accepted_tokens`,
  `cache_buf_idx`, and `old_*` after each step before rerunning full GSM8K.

Stable-slot retry:

- After changing the fused-replay state index to the stable request slot and
  making it contiguous, the new MTP server on port `8055` started successfully
  with CUDA graphs enabled.
- Smoke test still passed (`19 * 21 -> 399`).
- Quick GSM8K `--limit 50` stayed at `strict=0.0000`,
  `flexible=0.0000`.
- Samples no longer show a startup/kernel failure; they generate short invalid
  fragments such as `We need to parse...`, so the replay state is still
  semantically wrong under long few-shot prompts.

## May 18 Kernel-level Diagnosis

To stop using full GSM8K as the only signal, a focused parity pytest was
added in `tests/kernels/mamba/test_checkpointing_ssu_parity.py`. It runs
`flashinfer.mamba.checkpointing_ssu` directly with very small fp16 SSM
inputs, drives the same scheduling patterns vLLM does at MTP step time,
and compares the state HBM update against (a) an fp64 PyTorch SSM
recurrence reference and (b) a single-call variant of the same logical
tokens.

Headline results (run as
`FLASHINFER_WORKSPACE_BASE=... CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest tests/kernels/mamba/test_checkpointing_ssu_parity.py -v -s`):

- `test_single_call_vs_two_no_ckpt_appends` **FAILS**: feeding the same
  four tokens as a single `T=4` call vs as two `T=2` no-checkpoint
  appends produces different state HBM after the same forced flush
  (max abs ≈ `0.1` on a state whose values are in `[-0.04, 0.04]`).
  This is purely kernel-vs-kernel — no reference is involved — so it
  rules out the reference recurrence as the source of mismatch.
- `test_old_cumAdt_is_locally_reset_on_append` **PASSES** and pins down
  the actual bug. After two `T=2` no-checkpoint append steps the head-0
  active buffer of `old_cumAdt` contains:

  ```
  observed                = [-0.176, -0.315, -0.412, -0.818]
  expected global cumsum  = [-0.176, -0.315, -0.727, -1.133]
  expected local-reset    = [-0.176, -0.315, -0.412, -0.818]   (matches)
  diff_vs_global    = 0.3153
  diff_vs_local_reset = 3.2e-08
  ```

  Positions `[0..1]` come from step 1; positions `[2..3]` come from
  step 2 and equal `c3` and `c3+c4` (step-2 local cumsum) instead of
  `c1+c2+c3` and `c1+c2+c3+c4` (the globally consistent inclusive
  cumsum the replay code needs).

### Kernel code location

In `kernel_checkpointing_ssu.cuh` (FlashInfer) the buffer-side write is:

```cpp
// One warp, lanes 0..seq_len-1.
if (d_tile == 0 && warp == 1 && lane < seq_len) {
    int64_t const ca_w_base = cache_slot * params.old_cumAdt_stride_seq +
                              buf_write * params.old_cumAdt_stride_dbuf +
                              head * params.old_cumAdt_stride_head;
    old_cumAdt_w[ca_w_base + write_offset + lane] = smem.cumAdt[lane];
}
```

`smem.cumAdt` is computed in `compute_cumAdt` as an inclusive prefix sum
of `A * dt_proc` over the new tokens **starting from 0**, with no
addition of any prior `total_old_cumAdt`. So on the no-checkpoint
append path (`write_offset = prev_k > 0`) the buffer ends up with a
piecewise-local cumsum.

The replay code, however, reads it as if it were a global cumsum:

```cpp
float total_cumAdt = (prev_k > 0) ? smem.old_cumAdt[prev_k - 1] : 0.f;
float total_decay  = (prev_k > 0) ? __expf(total_cumAdt) : 1.f;
// ...
coeff[k] = (k < prev_k)
    ? __expf(total_cumAdt - smem.old_cumAdt[k]) * smem.old_dt[k]
    : 0.f;
```

When the buffer was filled across multiple no-ckpt appends, this gives
wrong `total_decay` and wrong `coeff[k]` for the prefix tokens of every
sub-step except the most recent one. That is the per-step compounding
error that turns into garbage state at MTP step ~3 onwards under
`--mamba-checkpoint-interval 16`.

### Fix paths

1. **Kernel-side (preferred long-term, upstream)**: bias the buffer write
   by `total_old_cumAdt = smem.old_cumAdt[prev_k - 1]` (loaded earlier in
   `load_data` from the active buffer) when `write_offset > 0`, so the
   stored value becomes a globally consistent cumsum within the buffer.
2. **vLLM-side workaround (landed in this branch)**: a small Triton
   kernel `_fixup_old_cumAdt_append_kernel` (in
   `vllm/model_executor/layers/mamba/ops/ssu_dispatch.py`) runs
   immediately after every `checkpointing_ssu` call and before the
   tracker update. For each request that resolved to the
   no-checkpoint append path
   (`prev_k_old > 0 AND prev_k_old + seq_len <= max_window`) it reads
   `total_old = old_cumAdt[slot, buf, head, prev_k_old - 1]` (which the
   FlashInfer call did not modify) and adds it in place to
   `old_cumAdt[slot, buf, head, prev_k_old : prev_k_old + seq_len]`.
   This restores the global-cumsum invariant inductively without
   needing a snapshot tensor. No-op on the checkpoint path
   (`buf_write != buf_read`, fresh buffer) and on the first step of
   a fresh buffer (`prev_k_old == 0`). CUDA-graph safe: same launch
   shape every step, no Python-side allocation.

### Test status (`tests/kernels/mamba/test_checkpointing_ssu_parity.py`)

| Test | Status | What it shows |
|---|---|---|
| `test_old_cumAdt_is_locally_reset_on_append` | PASS | Pins the kernel bug (per-step local cumsum) directly from buffer contents. |
| `test_vllm_fixup_recovers_global_cumAdt_buffer` | PASS | After applying the vLLM fixup, the buffer matches the global cumsum to fp32 precision (`diff ≈ 1e-7`). |
| `test_vllm_fixup_makes_single_vs_split_agree` | PASS | With the fixup, single-call vs split-call patterns produce bit-identical state HBM. |
| `test_single_call_vs_two_no_ckpt_appends` | XFAIL (strict) | Raw FlashInfer without fixup: ~`0.11` max diff. Becomes PASS when the upstream kernel fix lands; the strict xfail then fires so the marker is removed. |
| `test_two_no_ckpt_appends_then_flush_matches_reference` | XFAIL (strict) | Same as above, with the fp64 reference. |
| `test_single_step_all_accepted_matches_reference` | XFAIL (strict) | Single-step path; fp16 + `__expf` drift currently ~`0.24`. Should tighten significantly with the kernel fix. |

### End-to-end GSM8K validation after fixup

Two fp16+SR+MTP servers were brought up today on the same branch
(`flashinfer-checkpointing-ssm-cache`) and benchmarked back-to-back so
the comparison is apples-to-apples (same code, same caches, same eval
seed):

| Variant | Port | GPU | Adapter flag |
|---|---:|---|---|
| Old FI MTP path | 8054 | 1 | `--no-mamba-flashinfer-checkpointing-ssu` |
| New FI fused-replay + cumAdt fixup | 8055 | 0 | `--mamba-flashinfer-checkpointing-ssu --mamba-checkpoint-interval 16` |

Serve config (only the adapter flag differs between the two):

```bash
FLASHINFER_WORKSPACE_BASE=/my_home/vllm-scratch/fi-cu130 \
HF_HOME=/my_home/vllm-scratch/hf \
/my_home/venvs/vllm3/bin/python -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code --tensor-parallel-size 1 \
  --max-model-len 8192 --gpu-memory-utilization 0.85 \
  --mamba-backend flashinfer --mamba-ssm-cache-dtype float16 \
  --enable-mamba-cache-stochastic-rounding \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
```

Smoke test (both servers): prompt `Q: What is 19 * 21?\nA:` → output
`399\nQ: What is 19 * 22?\nA: 418`.

GSM8K 5-shot, full 1319 items,
`temperature=0.0, top_p=0.95, seed=1, num_concurrent=50`:

| Variant | Port | Mamba path | strict-match | flexible-extract |
|---|---:|---|---:|---:|
| Non-MTP fp16+SR | 8053 | old FI `selective_state_update` | 0.9128 | 0.9242 |
| Non-MTP fp16+SR | 8052 | `checkpointing_ssu`, interval 1 | 0.0728 | 0.1175 |
| MTP fp16+SR yesterday (stale) | 8054 | old FI `selective_state_update` | 0.5572 | 0.6967 |
| MTP fp16+SR yesterday (broken) | 8055 | `checkpointing_ssu`, no fixup | 0.0000 | 0.0000 |
| MTP fp16+SR | 8056 | Triton Mamba backend | 0.5572 | 0.6937 |
| **MTP fp16+SR today** | **8054** | **old FI `selective_state_update`** | **0.9272 ± 0.0072** | **0.9325 ± 0.0069** |
| **MTP fp16+SR today** | **8055** | **`checkpointing_ssu` + cumAdt fixup** | **0.8582 ± 0.0096** | **0.8673 ± 0.0093** |

Results JSON:

- `/my_home/vllm/results/gsm8k-fp16sr-fi-newssu-mtp-cumadt-fixup/`
- `/my_home/vllm/results/gsm8k-fp16sr-fi-oldssu-mtp-today/`

Headline:

- The cumAdt fixup converts the previously broken fused-replay MTP path
  (`0.0000` strict yesterday) into a healthy one (`0.8582` strict
  today) — a `~85 pp` swing on the same eval. This validates the
  kernel-level diagnosis end-to-end: the per-step local-cumsum write
  was the dominant bug in our integration.
- Today's old-FI MTP run on the same code/branch came in at
  `0.9272 / 0.9325` — substantially higher than yesterday's
  `0.5572 / 0.6967` recorded for the same nominal config. That earlier
  baseline appears to have been polluted by a different transient
  issue (perhaps an in-flight adapter change that landed and was
  reverted later in the day; we no longer have the exact intermediate
  commit). Treat the `0.9272 / 0.9325` numbers as the real old-FI
  baseline.
- The fused-replay path with the fixup still lags the old FI MTP
  baseline by `~6.9 pp` strict-match today. The cumAdt fix unblocks
  the path but does not yet make it preferable on quality — a residual
  semantic mismatch is suspected, most likely in the
  stable-per-request-slot vs. old FI's
  `state_indices[seq, num_accepted - 1]` per-accepted-token slot
  selection (see the §2 adapter notes). Investigation of this is the
  next step.
- The kernel-side cumAdt fix should still land upstream in FlashInfer
  so the vLLM workaround can be removed and so other downstream
  consumers do not need to re-discover and re-patch the same bug.
