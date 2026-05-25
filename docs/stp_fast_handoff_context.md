# STP Fast FlashInfer Mamba Handoff

## Current Goal

We are working on vLLM Mamba support for the new FlashInfer
`checkpointing_ssu` kernel.

Scope is **STP only**:

1. Clean STP branch.
2. Direct FlashInfer `checkpointing_ssu` path.
3. No slow compact-copy / grouped fallback in the normal STP path.
4. Accuracy matches old FlashInfer `selective_state_update`.
5. Speed matches or beats old-kernel STP.
6. MTP starts only after STP is correct and fast.

Ignore MTP and int8/fp8 for now. The current comparison is fp16/bf16 STP.

## Branch / Worktree

Codex worktree:

```text
/lustre/fsw/coreai_nvfm_llm/dafrimi/vllm-stp-fast-clean
branch: flashinfer-checkpointing-ssu-fast-stp-clean
```

If another agent wants to work manually, create a sibling worktree and do not
edit `vllm-stp-fast-clean` directly:

```bash
cd /lustre/fsw/coreai_nvfm_llm/dafrimi
git -C vllm-stp-fast-clean worktree add \
  -b dafrimi/stp-manual-$(date +%H%M) \
  /lustre/fsw/coreai_nvfm_llm/dafrimi/vllm-stp-manual \
  flashinfer-checkpointing-ssu-fast-stp-clean
```

Links:

```text
vLLM PR:        https://github.com/vllm-project/vllm/pull/43439
FlashInfer PR: https://github.com/flashinfer-ai/flashinfer/pull/3324
Friend PR:     https://github.com/vllm-project/vllm/pull/43518
```

The useful clue from the friend PR is that `old_cumAdt` must stay cumulative.
Our wrapper already has `_fixup_old_cumAdt_after_append` for the no-checkpoint
append case.

## Container Setup

Assume the agent is already inside a cluster container where
`/lustre/fsw/coreai_nvfm_llm/dafrimi` is mounted as `/my_home`.

Use:

```bash
cd /my_home/vllm-stp-fast-clean
source /my_home/venvs/vllm3/bin/activate
export PYTHONPATH=/my_home/vllm-stp-fast-clean:/my_home/flashinfer:${PYTHONPATH:-}
export HF_HOME=/my_home/vllm-scratch/hf
export FLASHINFER_WORKSPACE_BASE=/my_home/vllm-scratch/fi-cu130
```

Eval venv:

```bash
source /my_home/venvs/eval_venv/bin/activate
```

## Mental Model

Old kernel:

```text
Every decode token updates the full materialized SSM state in state[slot].
```

New checkpointing kernel:

```text
Most decode tokens do not write the full state.
They append token data into replay buffers:
  old_x
  old_B
  old_dt
  old_cumAdt

When replay would overflow, the kernel replays cached tokens, writes the
materialized state, flips the double buffer, and starts again.
```

STP prefill:

```text
Prefill already writes the final prompt state into state[slot].
After prefill, reset:
  prev_num_accepted_tokens[slot] = 0
  cache_buf_idx[slot] = 0

Do not seed old_x/old_B/old_dt/old_cumAdt from the prompt. Replay buffers are
only for decode tokens after prefill.
```

## Important Shapes

Nemotron STP production shape:

```text
state:      [cache_size, 128, 64, 128]
old_x:      [cache_size, window, 128, 64]
old_B:      [cache_size, 2, window, 8, 128]
old_dt:     [cache_size, 2, 128, window]
old_cumAdt: [cache_size, 2, 128, window]

cache_buf_idx:            [cache_size]
prev_num_accepted_tokens: [cache_size]
```

Grouped B/C:

```text
nheads = 128
ngroups = 8
heads_per_group = 16
group = head // 16
```

Fast target:

```text
checkpointing_ssu(state, old_x, old_B, old_dt, old_cumAdt, ...)
```

`old_B` stays grouped as `[cache, 2, window, 8, 128]`. Do not loop over groups
and do not compact/gather active slots for the normal STP path.

## Code Changes

Main file:

```text
vllm/model_executor/layers/mamba/ops/ssu_dispatch.py
```

Current branch changes:

1. Added direct grouped `checkpointing_ssu` path for true one-token decode.
2. Removed/bypassed the slow compact-copy/group-loop fallback for STP.
3. Kept slot-copy support only for read-slot -> write-slot movement.
4. Added `_fixup_old_cumAdt_after_append` so appended `old_cumAdt` remains
   cumulative when no checkpoint happens.
5. Reset replay trackers after prefill / non-checkpointing paths.
6. Kept prefill on the existing scan/final-state path.
7. Tried a synthetic `cu_seqlens=[0,1,2,...,batch]` varlen route for simple
   STP decode; it did **not** fix the mismatch, so the current target is the
   replay/checkpoint math after one cached token.

## Current Bug / Current State

Slow compatibility path:

```text
accurate but too slow
job 1890274: strict=0.90, flexible=0.90, about 6.8s/item
```

Old kernel baseline:

```text
job 1889998: strict=0.90, flexible=0.90 on GSM8K limit50
```

Fast direct STP eval:

```text
job 1891072
interval=6, limit=50, concurrency=50
speed improved: about 50 requests in 41s
accuracy failed: strict=0.00, flexible=0.02
```

Conclusion from that old run: the first direct path was fast but wrong.

Current conclusion after later fixes:

```text
The new grouped checkpointing kernel is healthy in eager mode after the
rand_seed fix, and the focused CUDA graph op replay test passes. The remaining
blocker is full vLLM CUDA-graph serving accuracy, which is still around
0.78-0.82 on GSM8K limit50 instead of the old-kernel 0.94 baseline.
```

## What Is Suspicious Now

The grouped tensor shape is no longer the main concern.

The fast direct path now passes `old_B` to FlashInfer in the shape the new
kernel expects:

```text
old_B: [cache_size, 2, window, ngroups, dstate]
       [cache_size, 2, 6,      8,       128] for Nemotron STP interval 6
```

So the clean target is still:

```text
one direct grouped checkpointing_ssu launch
no per-group loop
no compact active-row fallback
no copy into a fake [active_batch, ...] cache for normal STP
```

The suspicious part moved from "can the kernel accept grouped B?" to:

```text
Does vLLM pass exactly the right runtime slot/index/tracker lifecycle to the
direct kernel?
```

Things to verify:

1. Real serving actually runs with `checkpoint_window = 6`, not an effective
   window 1 path.
2. Real serving `state_batch_indices` / `dst_state_batch_indices` shapes are
   flattened correctly before calling `checkpointing_ssu`.
3. Source-slot to destination-slot movement copies all replay metadata.
4. `cache_buf_idx` and `prev_num_accepted_tokens` are updated on the destination
   slot after scheduler slot movement.
5. `old_cumAdt` stays cumulative after appending decode tokens before the
   checkpoint boundary.
6. Prefill resets replay trackers instead of leaving stale replay-buffer
   content behind.

These are wrapper/runtime contract questions. They are why the next tests are
server-shaped op tests and small debug serve runs, not another compact-copy
fallback.

## Interval 1 Is Not The Old Kernel

It is tempting to say:

```text
checkpoint interval 1 == old kernel, because state is updated every token
```

That is not how the implementation works.

Old kernel path:

```text
decode token t
  read state[slot]
  update state[slot] directly
  return output
```

New checkpointing path with interval/window 1:

```text
decode token t
  use checkpointing_ssu
  write/read replay tensors old_x/old_B/old_dt/old_cumAdt
  checkpoint every token
  flip the double buffer every token
  update cache_buf_idx and prev_num_accepted_tokens
```

So interval 1 should be mathematically close to the old kernel, but it is not
the same code path. Raw FlashInfer tests show interval 1 accumulates larger
checkpointing-kernel parity error than interval 6. That makes interval 1 useful
as a diagnostic, but it should not block the production STP interval-6 fix.

Production target:

```text
STP interval/window 6 fast direct path
```

Interval 1 target:

```text
diagnostic/follow-up after interval 6 serving accuracy is fixed
```

## Reproducer 1: vLLM Wrapper

Script:

```text
debug_stp_2d_indices.py
```

Run inside the container:

```bash
python debug_stp_2d_indices.py
```

What it compares:

```text
old2: old FlashInfer selective_state_update with indices shaped [batch,1]
old1: old FlashInfer selective_state_update with indices shaped [batch]
new:  vLLM wrapper using new FlashInfer checkpointing_ssu
```

Why it exists:

```text
It reproduces the accuracy bug in seconds at the Mamba op boundary. This is
better than starting with GSM8K because it immediately shows which decode step
diverges.
```

Important result before synthetic-varlen attempt:

```text
step 0 old2-old1 0.0 old2-new 0.00390625 old1-new 0.00390625 tracker 1
step 1 old2-old1 0.0 old2-new 21.658203125 old1-new 21.658203125 tracker 1
step 2 old2-old1 0.0 old2-new 3072.06982421875 old1-new 3072.06982421875 tracker 1
```

Result after synthetic-varlen attempt:

```text
step 0 old2-old1 0.0 old2-new 0.00390625 old1-new 0.00390625 tracker 1
step 1 old2-old1 0.0 old2-new 688.040771484375 old1-new 688.040771484375 tracker 1
step 2 old2-old1 0.0 old2-new 327.927734375 old1-new 327.927734375 tracker 1
```

Meaning:

```text
slot shape is not the issue: old [batch,1] == old [batch]
new path is close on decode step 0
new path diverges starting decode step 1
synthetic varlen does not solve it
```

## Reproducer 2: FlashInfer Kernel vs Reference

Script:

```text
debug_fi_checkpoint_t1.py
```

Run inside the container:

```bash
python debug_fi_checkpoint_t1.py
```

What it compares:

```text
FlashInfer checkpointing_ssu CUDA kernel
vs
FlashInfer Triton reference replay_selective_state_update
```

Why it exists:

```text
This bypasses vLLM's wrapper and tests the raw kernel/contract for the exact
production-like STP shape:
  T=1
  max_window=1
  nheads=128
  ngroups=8
  heads_per_group=16
  fp16 state
  bf16 x/B/C
```

Interpretation:

```text
If this fails, investigate FlashInfer kernel behavior or the kernel contract
for T=1/window=1.

If this passes, the kernel is likely okay and the vLLM wrapper is passing or
updating something incorrectly.
```

## Current Hypothesis

The original slow-accurate path copied active cache rows into compact scratch,
called the kernel on a fake compact batch, then scattered back. That avoided a
lot of runtime-shape problems, but it destroyed speed.

The current fast path sends the full cache directly to FlashInfer. Accuracy now
depends on every slot/index/tracker detail being exactly right.

The likely bug is one of these:

```text
1. real serving index shape is not fully covered by the op tests
2. source-slot -> destination-slot copy is not preserving all replay metadata
3. tracker updates happen against the wrong slot after slot movement
4. real serving is entering/resetting the checkpoint path differently from tests
5. old_cumAdt append fixup still misses a lifecycle case
```

The older window-1 wrapper repro showed divergence in the replay/checkpoint
step after one cached token.

Flow:

```text
decode step 0:
  prev_num_accepted_tokens = 0
  kernel appends token 0 to old_x/old_B/old_dt/old_cumAdt
  state is not written yet
  output is close to old kernel

decode step 1:
  prev_num_accepted_tokens = 1
  kernel must replay token 0 from old_* and process token 1
  output diverges badly
```

The next decision depends on `debug_fi_checkpoint_t1.py`:

```text
kernel-vs-reference fails -> inspect FlashInfer T=1/window=1 replay path
kernel-vs-reference passes -> inspect vLLM wrapper tensors and tracker updates
```

Note: that repro used window 1. It remains useful, but it is not the production
interval-6 decision point.

## Latest Findings

Raw FlashInfer sequential checkpointing test:

```text
debug_seq_checkpoint_vs_old.py
job 1891656
```

Findings:

```text
window=1, scale=0.2:
  step 1 checkpoint output diff about 0.094
  later steps drift more

window=6, scale=0.2:
  steps before checkpoint are small
  first checkpoint around step 6 output diff about 0.019
  next step output diff about 0.027

window=6, small scale:
  output/state diffs are tiny
```

Meaning:

```text
Raw direct checkpointing is much healthier at window 6 than at window 1.
This supports keeping the production target on interval 6 and treating
interval 1 separately.
```

Focused vLLM window-6 tests:

```text
job 1891662
```

Passed:

```text
test_checkpointing_ssu_stp_window6_strided_long_decode_matches_old_flashinfer
test_checkpointing_ssu_stp_window6_padded_graph_batch_matches_old_flashinfer
```

Failed:

```text
test_checkpointing_ssu_stp_server_like_2d_indices_match_old_flashinfer
```

Important correction:

```text
That failing server-like test was still using max_window = 1.
It was not testing the production interval-6 path.
```

Current experiment:

```text
job 1891690
```

Change:

```text
The server-like 2D-index test now uses max_window = 6 and checks tracker state.
```

Purpose:

```text
Verify the real server index shape with the real production checkpoint window.
If this passes, synthetic op tests cover more of the serving lifecycle and the
remaining mismatch is likely in real serving metadata/state reset, not grouped B.
If this fails, fix the 2D index/window-6 wrapper path before running GSM8K again.
```

Result:

```text
6 passed
```

Broad STP op suite:

```text
job 1891697
34 passed
```

Eager old-kernel baseline:

```text
job 1891693
--enforce-eager
GSM8K limit50 strict=0.94 flexible=0.94
```

First eager new-kernel run:

```text
job 1891692
--enforce-eager
mamba_checkpoint_interval=6
```

Result:

```text
smoke completion worked
server then OOMed in FlashInfer FP4 MoE during eval
not a Mamba/checkpointing stack trace
```

Useful debug from that run:

```text
Mamba checkpointing SSU direct-entry:
  window=6
  x=(1, 128, 64)
  dt=(1, 128, 64)
  B=(1, 8, 128)
  C=(1, 8, 128)
  state_indices_shape=(1, 1)
  dst_indices_shape=(1, 1)
  kernel_slots=1
  cu_seqlens=False
  prev_range=0..0
  buf_range=0..0

Mamba checkpointing SSU direct-exit:
  x=(1, 1, 128, 64)
  dt=(1, 1, 128, 64)
  B=(1, 1, 8, 128)
  C=(1, 1, 8, 128)
  prev_range=1..1
  buf_range=0..0
```

Meaning:

```text
Real serving does enter the direct path with the expected window 6 and [batch,1]
state-index shape. The first debug budget only covered the first decode token
through many layers, so it proved 0 -> 1 tracker movement but not the full
1 -> 2 -> ... -> 6 replay cycle.
```

Eager new-kernel rerun:

```text
job 1891714
--enforce-eager
--gpu-memory-utilization 0.75
```

Result:

```text
GSM8K limit50 strict=0.02 flexible=0.06
```

Sample behavior:

```text
The first few tokens/steps can be coherent, then the answer collapses into
repeated or junk text mid-solution.
```

Meaning:

```text
--enforce-eager on both old and new kernels proves CUDA graphs are not the
reason. The old eager baseline is good; the new direct checkpointing path is
still corrupting decode state/replay after serving has started.
```

## Current Speed Bug Found

A separate performance/memory bug was found while checking larger windows:

```text
job 1891725
new eager, interval=32, small eval
failed before eval with OOM in mamba_mixer2.py:_get_contiguous_checkpointing_buffer
while allocating a contiguous old_x mirror
```

Why this matters:

```text
The previous wrapper tried to make old_x/old_B/old_dt/old_cumAdt contiguous
before calling FlashInfer. For Nemotron these tensors are huge. Copying them
for every layer/request path can dominate latency and can OOM as soon as the
checkpoint window grows.
```

FlashInfer does not need that for the large replay tensors:

```text
The wrapper passes outer strides for old_x/old_B/old_dt/old_cumAdt into the
kernel, and the C++ checks only require the inner dimensions to be contiguous.
So the clean fast path should pass the strided cache pages directly.
```

But FlashInfer does require the two tracker vectors to be contiguous:

```text
cache_buf_idx
prev_num_accepted_tokens
```

Current patch direction:

```text
1. Keep old_x/old_B/old_dt/old_cumAdt as direct strided cache tensors.
2. Mirror only cache_buf_idx and prev_num_accepted_tokens into tiny contiguous
   vectors, then copy those trackers back after decode.
3. Skip checkpoint-slot copy when source and destination state-index tensors
   are the same object. In normal no-spec STP decode, read slot == write slot,
   so allocating scratch/copying all replay tensors is wasted work.
```

This fixes the obvious slow/OOM path. It does not by itself prove accuracy; run
the focused op tests and then a small eager GSM8K comparison after applying it.

Latest result after applying this patch:

```text
job 1891960
new kernel, --enforce-eager, interval=6, max_num_seqs=8, concurrency=1
GSM8K limit20 strict=0.90 flexible=0.90
```

Larger validation:

```text
job 1892100
new kernel, --enforce-eager, interval=6, max_num_seqs=8, concurrency=1
GSM8K limit50 strict=0.92 flexible=0.92
```

This is the most important evidence so far. Before this patch, the same new
kernel family failed badly:

```text
job 1891714
new kernel, --enforce-eager, interval=6
GSM8K limit50 strict=0.02 flexible=0.06
```

Old eager baseline:

```text
job 1891693
old kernel, --enforce-eager
GSM8K limit50 strict=0.94 flexible=0.94
```

So CUDA graphs are not the issue, and the main bad behavior was very likely in
the vLLM wrapper/cache lifecycle rather than the grouped FlashInfer kernel
itself.

Clean STP op suite after removing temporary diagnostics:

```text
job 1892134
tests/kernels/mamba/test_checkpointing_ssu_stp.py
34 passed, 16 warnings in 14.69s
```

## What Is Actually Suspicious

The grouped B tensor is probably not the blocker anymore:

```text
old_B shape is [cache_size, 2, window, 8, 128]
FlashInfer gets heads_per_group = nheads // ngroups = 128 // 8 = 16
The kernel maps group = head // heads_per_group internally.
```

The stronger suspicion is now runtime lifecycle:

```text
prefill writes only final state and resets replay trackers
decode appends token replay data into old_x/old_B/old_dt/old_cumAdt
when prev reaches the checkpoint window, kernel replays, writes state, flips
cache_buf_idx, and starts the next replay buffer
```

If any one of these is wrong in real serving, output may look fine for a few
tokens and then degrade badly:

```text
1. prev_num_accepted_tokens updated on the wrong slot
2. cache_buf_idx flipped on the wrong slot
3. source-slot -> destination-slot movement misses replay metadata
4. old_cumAdt appended values are not cumulative when replayed later
5. replay trackers not reset after prefill / old-kernel fallback path
```

Prefill note:

```text
For STP prefill, do not fill replay buffers with the last prompt window.
Only the final materialized SSM state matters. Replay buffers are for decode
tokens after prefill, and should start empty/reset.
```

## Focused vLLM Tests

Current focused command:

```bash
python -m pytest -q \
  tests/kernels/mamba/test_checkpointing_ssu_stp.py::test_checkpointing_ssu_stp_window6_strided_long_decode_matches_old_flashinfer \
  tests/kernels/mamba/test_checkpointing_ssu_stp.py::test_checkpointing_ssu_stp_window6_padded_graph_batch_matches_old_flashinfer \
  tests/kernels/mamba/test_checkpointing_ssu_stp.py::test_checkpointing_ssu_stp_server_like_2d_indices_match_old_flashinfer \
  --tb=short -s
```

Runtime debug logging is behind:

```bash
export VLLM_MAMBA_CKPT_DEBUG_CALLS=20
```

It prints for the first N direct checkpointing calls:

```text
checkpoint window
x/dt/B/C shapes
original state_batch_indices shape
dst_state_batch_indices shape
flattened kernel slot count
prev_num_accepted_tokens min/max
cache_buf_idx min/max
```

Use this on a small serve/eval run to verify real serving is actually doing:

```text
window=6
state indices shaped as expected
prev tracker progresses 0 -> 1 -> ... -> 6 -> 1 ...
buffer flips only on checkpoint
```

## Old-vs-New Real Serving Compare

Temporary instrumentation in `ssu_dispatch.py` can run the old kernel and the
new checkpointing kernel for the same selected runtime state pointer and log
the output/cache ranges.

Enable it with:

```bash
export VLLM_MAMBA_CKPT_COMPARE_CALLS=6
export VLLM_MAMBA_CKPT_COMPARE_MIN_PREV=1
```

Meaning:

```text
COMPARE_CALLS: number of replay-active calls to log
COMPARE_MIN_PREV: skip warmup calls until prev_num_accepted_tokens >= this
```

The `min_prev=1` setting is important. The first serving compare job only
logged warmup calls:

```text
job 1892059
step=0 slots=[1] prev=0..0 buf=0..0 ckpt=False out_abs_max=1 old_abs_max=428 new_abs_max=428
step=1 slots=[2] prev=0..0 buf=0..0 ckpt=False out_abs_max=0.5 old_abs_max=154 new_abs_max=154
```

Those calls prove the first-token path is not wildly broken, but they do not
test replay. The useful comparison must show calls with:

```text
prev=1..5: append/replay without checkpoint
prev=6: checkpoint boundary, state materialized, buffer flip
```

Current replay-active compare job:

```text
job 1892092
new kernel, --enforce-eager, interval=6, limit=1
VLLM_MAMBA_CKPT_COMPARE_CALLS=6
VLLM_MAMBA_CKPT_COMPARE_MIN_PREV=1
```

Result:

```text
prev=1 and prev=2 real-serving replay calls match old kernel closely.
max output diff: 0.125 to 0.5
mean output diff: about 1e-4 to 4e-4
old/new output magnitudes: up to 524
limit1 eval: strict=1 flexible=1
```

Checkpoint-boundary compare:

```text
job 1892117
new kernel, --enforce-eager, interval=6, limit=1
VLLM_MAMBA_CKPT_COMPARE_CALLS=8
VLLM_MAMBA_CKPT_COMPARE_MIN_PREV=5
```

Result:

```text
prev=5, ckpt=False:
  max output diff: 0.25 to 1.0
  mean output diff: about 1e-4 to 3e-4
  old/new output magnitudes: up to 756

prev=6, ckpt=True:
  max output diff: 0.25 to 1.0
  mean output diff: about 1.7e-4 to 6.7e-4
  old/new output magnitudes: up to 564
```

Meaning:

```text
The direct grouped checkpointing_ssu path matches the old kernel through real
serving replay and the checkpoint/materialize/buffer-flip boundary. The old
"grouped tensor may be wrong" suspicion is now very unlikely.
```

Read:

```bash
tail -f /my_home/vllm-stp-no-fallback-artifacts/server-logs/stp-new-eager-ci6-real-compare-prev1-v2-limit1-20260524_082711.log
```

Interpretation:

```text
small output diff for prev>=1 and checkpoint boundary:
  direct new kernel is probably correct; clean diagnostics and run larger eval

large output diff starting at prev=1:
  replay buffer contents or old_cumAdt lifecycle is still wrong

large output diff only at checkpoint boundary:
  checkpoint materialization / buffer flip / state writeback is suspect

no prev>=1 logs:
  serving is not reaching replay-active decode before the compare budget, or
  the environment patch was not loaded
```

Cleanup note:

```text
The shadow old-kernel compare and VLLM_MAMBA_CKPT_* env logging were temporary
diagnostics. They were removed from the clean branch after jobs 1892092 and
1892117 captured the replay/checkpoint evidence.
```

## Manual Serve/Eval Commands

Use these only after op-level tests pass.

New-kernel eager serve:

```bash
cd /my_home/vllm-stp-fast-clean
source /my_home/venvs/vllm3/bin/activate
export PYTHONPATH=/my_home/vllm-stp-fast-clean:${PYTHONPATH:-}

python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port 8421 \
  --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --tokenizer nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --trust-remote-code \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --served-model-name nemotron-stp-new-eager \
  --mamba-backend flashinfer \
  --mamba-checkpoint-interval 6 \
  --enable-mamba-cache-stochastic-rounding \
  --gpu-memory-utilization 0.85 \
  --mamba-ssm-cache-dtype float16 \
  --enforce-eager
```

Eval:

```bash
source /my_home/venvs/eval_venv/bin/activate
python -m lm_eval \
  --model local-completions \
  --model_args "model=nemotron-stp-new-eager,tokenizer=nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4,tokenizer_backend=huggingface,trust_remote_code=True,base_url=http://127.0.0.1:8421/v1/completions,api_key=EMPTY,num_concurrent=50,timeout=45000" \
  --tasks gsm8k \
  --num_fewshot 5 \
  --limit 50 \
  --gen_kwargs temperature=0.0,top_p=0.95,do_sample=true,seed=1 \
  --output_path /my_home/vllm/results/stp-new-eager-limit50 \
  --log_samples
```

Old-kernel comparison should use:

```text
/my_home/vllm-main-mtp-old-kernel
```

with the same serve/eval settings, different port, and different
`--served-model-name`.

## What Not To Do Yet

- Do not resume MTP.
- Do not bring back the compact group-loop fallback as the final STP path.
- Do not use GSM8K as the first debugger; pass the op-level repro first.
- Do not seed prompt replay buffers from prefill; STP prefill only needs final
  state plus replay reset.
- Do not treat interval 1 as the old kernel. It is a separate checkpointing
  path and currently a lower-priority diagnostic.

## 2026-05-24 Current Status

STP is not PR-ready yet. The clean/new kernel path is fast enough in the
direct op sense, but CUDA-graph serving accuracy is still the blocker.

Current branch:

```text
/lustre/fsw/coreai_nvfm_llm/dafrimi/vllm-stp-fast-clean
branch: flashinfer-checkpointing-ssu-fast-stp-clean
```

Current uncommitted files:

```text
vllm/model_executor/layers/mamba/ops/ssu_dispatch.py
vllm/model_executor/layers/mamba/mamba_mixer2.py
vllm/v1/attention/backends/mamba_attn.py
vllm/v1/worker/gpu_model_runner.py
tests/kernels/mamba/test_checkpointing_ssu_stp.py
docs/stp_fast_handoff_context.md
```

The active production-code changes are intentionally small:

1. `ssu_dispatch.py`: do not pass a stochastic-rounding `rand_seed` into the
   checkpointing kernel path. The old kernel stochastic-rounds the full state
   every token; checkpointing only materializes at interval boundaries, so
   seeding the checkpointing path changed the stochastic process and hurt
   accuracy.
2. `gpu_model_runner.py`: after CUDA graph capture, reset Mamba checkpointing
   cache tensors. The latest local patch resets `kv_cache[1:]`, including
   `ssm_state` plus replay buffers and trackers. The earlier attempt reset
   only `kv_cache[2:]` and did not recover accuracy.
3. `test_checkpointing_ssu_stp.py`: added a focused CUDA graph replay test
   with persistent block-table gathers, window 6, grouped `old_B`, and slot
   movement.
4. `mamba_mixer2.py`: temporary debug logging only. It must be removed before
   PR. It skips CPU tensor copies during CUDA graph capture because those crash
   graph memory profiling.
5. `mamba_attn.py`: temporary metadata logging only. It logs CUDA-graph Mamba
   metadata outside capture so we can compare state rows, block indices, and
   gathered input/output slots.

Important results:

```text
Old STP CUDA graph baseline job 1893009:
  gsm8k limit50: strict=0.94 flexible=0.94
  generation throughput around 464 tok/s

New STP CUDA graph before graph fixes job 1893029:
  strict=0.32 flexible=0.34

New STP eager with SR enabled before rand_seed fix job 1893138:
  strict=0.70 flexible=0.70

New STP eager with SR disabled job 1893201:
  strict=0.94 flexible=0.94

New STP eager with SR enabled after rand_seed fix job 1893252:
  strict=0.92 flexible=0.92
  main remaining diff was doc_id=43

New STP CUDA graph after rand_seed fix job 1893292:
  strict=0.80 flexible=0.78

New STP CUDA graph after lazy rand_seed allocation cleanup job 1893340:
  strict=0.82 flexible=0.80

New STP CUDA graph after replay-buffer-only capture reset job 1893550:
  strict=0.82 flexible=0.80

Focused CUDA graph op replay test job 1893584:
  PASSED
  This means raw checkpointing_ssu can replay inside CUDA graph with
  server-style block-table gathers. The bug is likely runner/model state
  lifecycle around capture or real request metadata, not the direct op itself.

New STP CUDA graph after stronger capture reset job 1894000:
  strict=0.80 flexible=0.82
  graph capture finished, eval finished, still not old-kernel accurate

Debug CUDA graph run job 1894275:
  crashed during CUDA graph memory profiling because the temporary Mamba
  debug logging copied CUDA tensors to CPU while graph capture was active.
  The debug logging was patched to skip logging while
  torch.cuda.is_current_stream_capturing() is true.
  Before the crash, non-capture debug showed dummy/capture warmup forwards
  touching slot 0 with max_seqlen=1 and zero trackers.

Fresh metadata diagnostic job 1894296:
  BaseMambaAttentionMetadataBuilder.supports_update_block_table was
  temporarily disabled to force a fresh Mamba metadata build instead of reusing
  cached metadata and only replacing block-table-derived state indices.
  Result: strict=0.78 flexible=0.80.
  Throughput after warmup was about 324 generation tok/s.
  Conclusion: fresh metadata did not recover accuracy, so the broad
  supports_update_block_table=False diagnostic should not be kept. The bug is
  lower/different than the simple metadata-reuse fast path.

Metadata debug jobs 1894484, 1894522, 1894535:
  The first metadata logger was placed in
  _update_metadata_for_cudagraph_capture and spent its budget during graph
  capture/profiling dummy forwards. Those rows were all state [[0]].
  After moving logging into update_block_table, real request metadata looked
  sane for default mamba_cache_mode=none:
    prefill rows used state_p values like [2], [3], [4], [5]
    decode rows used state_d pairs like [[2], [8]], [[3], [9]],
    [[4], [10]], [[5], [11]]
  The repeated [[1], [7]] rows from _update_metadata_for_cudagraph_capture are
  probably not stale metadata; they correspond to the first Mamba layer/group
  metadata built before later layer groups are refreshed. Do not patch that
  blindly.

Current SSU debug direction:
  Metadata/block-table refresh is no longer the leading suspect. The next
  useful logs are around the actual checkpointing kernel call in
  ssu_dispatch.py, comparing slots plus cache_buf_idx/prev_num_accepted_tokens
  before the kernel, after the kernel, and after vLLM's tracker update.
  Job 1894552 proved the first SSU logger was consumed by graph-capture dummy
  rows (all slot 0, trackers 0). The follow-up job 1894562 filters those dummy
  rows and should show real-request tracker movement.

SSU debug update 2026-05-24:
  Job 1894562 was cancelled because the filter still allowed CUDA graph
  capture/profiling rows with x_ckpt batch size exactly 16. The dummy rows
  again burned the log budget before real eval requests.
  The filter was tightened from x_ckpt.size(0) > 16 to >= 16.
  Replacement job 1894575 still burned the debug budget during CUDA graph
  decode capture. The capture-time rows had slots all zero, cache all zero,
  prev all zero, and batch sizes 8/4. torch.cuda.is_current_stream_capturing()
  did not suppress these rows, so the logger now skips all rows where every
  sampled slot is zero. Real request metadata seen earlier uses nonzero state
  slots like [2], [8], [3], [9].
  Replacement job 1894583 is running on branch
  flashinfer-checkpointing-ssu-fast-stp-clean with:
    served_model=nemotron-stp-fastclean-cg-ci6-ssu-real3
    port=8479
    checkpoint_interval=6
    enforce_eager=False
    cudagraph_mode=FULL_AND_PIECEWISE
    MAMBA_SSU_DEBUG_CALLS=180
  Server log:
    pending in /lustre/fsw/coreai_nvfm_llm/dafrimi/vllm-stp-no-fallback-artifacts/server-logs/*ssu-real3*
  Eval log:
    pending in /lustre/fsw/coreai_nvfm_llm/dafrimi/vllm-stp-no-fallback-artifacts/eval-logs/*ssu-real3*

SSU debug result 1894583:
  The all-zero skip worked. Capture completed without burning the SSU budget,
  and real request rows appeared. Limit-2 GSM8K scored 1.0/1.0, which is only
  a smoke result. Real SSU rows showed per-call tracker update works:
    before slots=[1] cache=[0] prev=[0]
    after_kernel slots=[1] cache=[0] prev=[0]
    after_tracker slots=[1] cache=[0] prev=[1]
  The pattern repeated for slots [1], [2], [3], [4], [5]. This does not yet
  prove a bug or correctness, because each Mamba layer owns its own KV cache
  and the SSU log does not include layer name. Repeated slot ids across
  different layers can look like "prev reset to 0" even when each layer is
  correct.

Layer-aware debug update:
  mamba_mixer2.py already had STP_DEBUG around selective_state_update with
  layer names, input/output slots, prev/cache before and after. It was
  consumed by all-zero capture rows. It was patched to skip all-zero
  input/output slot rows. Replacement job 1894599 is running with:
    served_model=nemotron-stp-fastclean-cg-ci6-stp-real4
    port=8480
    MAMBA_SSU_DEBUG_CALLS=0
    VLLM_MAMBA_STP_DEBUG_CALLS=160
  Goal: verify whether the same layer's prev/cache advances over real decode
  steps, or whether a reset/copyback path clears trackers between steps.

Layer-aware debug result 1894599:
  The run completed, but the Python-side layer hooks only saw the visible
  non-replay path:
    graph=NONE
    prefill=1 decode=1 tokens=1 max_seqlen=1
    per-layer prev/cache moved from [0]/[0] to [1]/[0]
  This is useful but not enough. It proves the normal Python decode call can
  advance the checkpoint trackers for real slots. It does not observe full
  CUDA graph replay, because replay bypasses Python layer code and calls
  cudagraph.replay() directly.

Important correction:
  NULL_BLOCK_ID is 0 in this tree. Mamba's padded full-CUDA-graph decode rows
  are filled with slot 0, and the new tracker kernels mask slot 0. So padded
  rows corrupting the final cache slot is not the current leading hypothesis.

Current leading hypothesis:
  The raw checkpointing_ssu CUDA graph replay test passes, eager serving is
  close/good, and visible Python decode tracker movement works. The remaining
  failure likely sits in the full-vLLM CUDA graph input lifecycle:
    1. CUDA graph replay bypasses model/layer Python code.
    2. CUDAGraphWrapper does not copy runtime inputs itself.
    3. Full-graph replay depends on persistent/static runner and attention
       metadata buffers being refreshed before replay.
  Therefore the next diagnostic is runner-level logging outside the captured
  graph, not more logging inside ssu_dispatch.py or mamba_mixer2.py.

Runner debug added:
  gpu_model_runner.py now has env-controlled MAMBA_RUNNER_DEBUG_CALLS logging
  immediately before _model_forward. It logs:
    cg mode and batch descriptor
    unpadded/padded token and request counts
    max_query_len
    first Mamba metadata layer/group
    state_indices_tensor_d rows used by full graph metadata
    query_start_loc_d
    first checkpointing kv_cache tracker sample for those slots
  This is intentionally temporary and must be removed before the final PR.

Runner debug result 1894662:
  Real eval requests definitely use full CUDA graph replay:
    cg=FULL
    desc=BatchDescriptor(num_tokens=2, num_reqs=2, uniform=True)
    later desc=BatchDescriptor(num_tokens=1, num_reqs=1, uniform=True)
    slots initially [[1], [7]], then [[7]] as one request finished
  Limit-2 GSM8K was 1.0/1.0, which is a smoke-only result. The important
  finding is not the score; it is that the bad path is full graph replay, not
  accidental eager fallback.
  The first runner tracker sampler returned empty cache/prev lists because it
  looked only in static_forward_context. MambaMixer2 owns kv_cache on the
  actual model modules (`mamba_mixer2.py` has `self.kv_cache = tuple(...)`).
  The runner sampler was updated to also scan `self.model.named_modules()` for
  kv_cache tuples.

Runner debug result 1894708:
  The eval again used full graph replay, but tracker samples were still empty.
  Reason: `self.model` is a CUDAGraphWrapper in this path. The raw model is
  available through `get_model()`, which unwraps CUDAGraphWrapper/UBatchWrapper.
  The runner sampler was corrected to scan `self.get_model().named_modules()`.

Runner debug result 1894742:
  The eval again showed full graph replay, but tracker samples stayed empty.
  Reading the cache binding code explained why: `bind_kv_cache` assigns
  Mamba caches as a Python list of 8 tensors (`state_tensors`), not a tuple.
  The diagnostic sampler was checking only tuple. It was fixed to accept
  both tuple and list of length 8.
  This also exposed a real bug in the post-capture cleanup: the
  `_reset_mamba_checkpointing_cache_after_cudagraph_capture()` helper used
  the same tuple-only check, so it did not reset Mamba caches after CUDA graph
  capture. Earlier "reset after capture did not fix CUDA graph accuracy"
  results are therefore not valid evidence against the reset idea.
  The helper was fixed to accept both tuple and list of length 8.

Current status / PR readiness:
  STP is not PR-ready yet. Eager new-kernel behavior is close to old after the
  stochastic-rounding seed fix, but CUDA graph remains around 0.78-0.82 on
  GSM8K limit50 versus old-kernel 0.94. Do not start the MTP branch until STP
  CUDA graph reaches old-kernel accuracy and comparable speed.
```

Useful polling commands:

```bash
squeue -j <jobid> -o "%.18i %.9T %.12M %.40j"
tail -f /lustre/fsw/coreai_nvfm_llm/dafrimi/vllm-scratch/logs/stp-cg-debug2-<jobid>.log
ls -t /lustre/fsw/coreai_nvfm_llm/dafrimi/vllm-stp-no-fallback-artifacts/server-logs/* | head
ls -t /lustre/fsw/coreai_nvfm_llm/dafrimi/vllm-stp-no-fallback-artifacts/eval-logs/* | head
```

Current interpretation:

```text
The new STP op is not broadly broken:
  eager no-SR matches old baseline
  eager SR is close after rand_seed fix
  synthetic CUDA graph op replay passes

The still-suspicious layer is full vLLM CUDA graph serving:
  capture dummy forward mutates Mamba state/cache
  graph wrapper replays with static tensor addresses
  resetting replay buffers and SSM state after capture did not fix accuracy
  forcing fresh Mamba metadata did not fix accuracy
  remaining suspicion is exact per-step state-index/block-index/slot movement
  in the full serving path, or another persistent graph input/cache lifecycle
  issue not covered by the focused op replay test
```

Next steps:

```text
1. Rerun the small CUDA graph eval with MAMBA_RUNNER_DEBUG_CALLS enabled after
   the model-module kv_cache sampler fix.
2. For FULL rows, compare state_indices_tensor_d and tracker samples across
   steps:
     slots should be nonzero real slots
     prev should advance 0 -> 1 -> ... -> 6 and then reset/flip buffer
     cache_buf_idx should flip only at checkpoint boundary
3. If runner FULL rows feed the right slots and trackers advance, move to a
   focused full-runner test or compare hidden/logit drift old vs new.
4. If runner FULL rows show stale slots or tracker reset, fix the static
   Mamba metadata / replay input lifecycle.
5. Remove temporary runner, ssu_dispatch, mamba_mixer2, and mamba_attn debug
   logging before PR.
6. Keep the rand_seed checkpointing fix and any narrow CUDA graph lifecycle fix.
7. Do not start MTP until STP CUDA graph matches old-kernel accuracy and speed.
```

## 2026-05-24 Late STP CUDA Graph Update

Branch still under test:

```text
/lustre/fsw/coreai_nvfm_llm/dafrimi/vllm-stp-fast-clean
branch: flashinfer-checkpointing-ssu-fast-stp-clean
```

Important latest results:

```text
old STP CUDA graph baseline:
  job 1893009 / run stp-oldkernel-cg-limit50-limit50-20260524_121942
  GSM8K limit50 strict=0.94 flexible=0.94
  generation throughput observed around 464 tok/s earlier, final log burst ~same order

new STP CUDA graph after rand_seed fix and list-aware capture reset:
  job 1894785 / run nemotron-stp-fastclean-cg-ci6-reset-list-20260524_155550
  checkpoint interval=6, stochastic rounding enabled, CUDA graph enabled
  GSM8K limit50 strict=0.74 flexible=0.80
  final generation throughput burst ~616 tok/s
  conclusion: speed is no longer the main issue; CUDA graph accuracy is still wrong

runner debug after list-aware cache inspection:
  job 1894809 / run nemotron-stp-fastclean-runner-debug-list-20260524_160301
  GSM8K limit2 strict=1.0 flexible=1.0
  confirmed real eval uses cg=FULL rows
  confirmed tracker sampling now sees real Mamba kv_cache list
  example tracker lifecycle for slots [1, 7]:
    prev advances 1,2,3,4,5,6 then cache_buf_idx flips and prev resets to 1
  conclusion: the checkpoint tracker kernel is running during graph replay
```

Sample-level CUDA graph regression set from old vs new limit50:

```text
old correct / new wrong doc_ids: [1, 2, 7, 15, 27, 39, 43]
both wrong doc_ids: [8, 12, 20]
old strict correct count: 47/50
new strict correct count: 40/50
```

Current live hypothesis under test:

```text
The direct checkpointing path is probably OK in eager and the raw op can replay
under a synthetic CUDA graph. The remaining bug appears in full vLLM serving
where batches mix prefill/decode/chunk rows before/around graph replay.

The non-checkpointing fallback path materializes state directly. If that path
runs for a slot, any replay-buffer tracker state for that slot is stale. The
old reset condition only ran when num_accepted_tokens is None. A narrow patch
now resets checkpoint replay trackers after every non-checkpointing fallback
when checkpoint tensors exist.
```

Patch currently being evaluated:

```text
vllm/model_executor/layers/mamba/ops/ssu_dispatch.py
  fallback/non-checkpointing path now does:
    if cache_buf_idx is not None and prev_num_accepted_tokens is not None:
      reset checkpointing trackers for dst/state indices
  instead of requiring num_accepted_tokens is None.
```

Active validation:

```text
job 1894822
run name should be nemotron-stp-fastclean-cg-ci6-reset-fallback-<timestamp>
mode: STP new kernel, CUDA graph, checkpoint interval 6, GSM8K limit50
served model: nemotron-stp-fastclean-cg-ci6-reset-fallback
pass condition: match old baseline near strict/flexible 0.94 and comparable speed
```

If job 1894822 passes:

```text
1. Remove temporary debug logging from gpu_model_runner.py, mamba_mixer2.py,
   mamba_attn.py, and ssu_dispatch.py.
2. Keep the rand_seed checkpointing fix and fallback tracker reset.
3. Re-run clean CUDA graph limit50 and focused kernel tests.
4. Then prepare STP PR branch.
```

If job 1894822 fails:

```text
Do not start MTP. Continue with doc-id targeted repro using old-correct/new-wrong
ids [1, 2, 7, 15, 27, 39, 43], and compare per-request slot movement / logits
rather than running more blind aggregate evals.
```

### Job 1894822 Result

```text
run: nemotron-stp-fastclean-cg-ci6-reset-fallback-20260524_161040
patch: reset checkpoint replay trackers after every fallback/non-checkpointing path
result: GSM8K limit50 strict=0.82 flexible=0.84
old baseline: strict=0.94 flexible=0.94
```

Interpretation:

```text
This improved over 1894785 (0.74/0.80) and fixed doc_ids [1, 2, 7, 27],
but it introduced regressions on doc_ids [13, 17]. Therefore tracker reset is
part of the issue, but resetting after every fallback is too broad and can wipe
valid replay state for active simple decode rows.
```

Follow-up patch now under test:

```text
Only reset checkpoint replay trackers after fallback when not simple_decode.
This targets prefill/chunk/non-simple materialization paths while preserving
active simple decode replay windows.
```

Active validation:

```text
job 1894840
served model: nemotron-stp-fastclean-cg-ci6-reset-nonsimple
mode: STP new kernel, CUDA graph, checkpoint interval 6, GSM8K limit50
```

### Job 1894840 Result

```text
run: nemotron-stp-fastclean-cg-ci6-reset-nonsimple-20260524_161729
patch: reset checkpoint replay trackers only after fallback when not simple_decode
result: GSM8K limit50 strict=0.86 flexible=0.88
old baseline: strict=0.94 flexible=0.94
```

Sample movement:

```text
1894785 old-correct/new-wrong: [1, 2, 7, 15, 27, 39, 43]
1894822 broad reset old-correct/new-wrong: [13, 15, 17, 39, 43]
1894840 narrowed reset old-correct/new-wrong: [15, 27, 43]
fixed by narrowed reset vs 1894785: [1, 2, 7, 39]
regressions vs 1894785: []
```

Interpretation:

```text
The narrowed reset is a real improvement and should likely stay, but it is not
the full fix. Remaining failures [15, 27, 43] should be debugged with targeted
requests/logits/slot tracking. Do not run more blind limit50 loops before
instrumenting those ids.
```

### Job 1894862 No-SR Discriminator

```text
run: nemotron-stp-fastclean-cg-ci6-nosr-check-20260524_162504
mode: STP new kernel, CUDA graph, checkpoint interval 6, stochastic rounding disabled
result: GSM8K limit50 strict=0.38 flexible=0.44
```

Interpretation:

```text
Do not use no-SR CUDA graph as the primary serving validation for this quantized
model path. It is much worse than SR-enabled narrowed reset. The remaining
0.86/0.88 -> 0.94 gap is not explained by simply disabling stochastic rounding.
Continue with targeted debug on [15, 27, 43].
```

## 2026-05-24 Late STP Status Update

Current branch: `flashinfer-checkpointing-ssu-fast-stp-clean` in
`/lustre/fsw/coreai_nvfm_llm/dafrimi/vllm-stp-fast-clean`.

What is confirmed:

- Old FlashInfer STP CUDA-graph baseline remains `0.94/0.94` on GSM8K limit50
  (`stp-oldkernel-cg-limit50-limit50-20260524_121942`).
- New checkpointing STP eager with stochastic rounding, after removing the
  checkpointing rand seed, reached `0.92/0.92`.
- New checkpointing STP CUDA graph is still not PR-ready. Best useful run so
  far is the narrow non-simple reset patch: `0.86/0.88`
  (`nemotron-stp-fastclean-cg-ci6-reset-nonsimple-20260524_161729`).
- Speed is not the current blocker. CUDA-graph generation throughput is in the
  same broad regime as old kernel in the successful server runs; accuracy is the
  blocker.
- The raw grouped FlashInfer checkpointing kernel is not the obvious failure:
  eager parity tests and synthetic CUDA graph replay tests pass with grouped
  `old_B` and interval/window 6.
- Full vLLM CUDA graph serving is the failing surface. Python layer hooks do not
  see FULL graph replay; runner-level debug confirms FULL decode rows and slot
  tracker cycles.

Current code experiments/findings:

- Real fix retained: checkpointing path no longer passes stochastic rounding
  `rand_seed` into `checkpointing_ssu`; old kernel still gets SR seed.
- Real fix retained: reset checkpointing trackers after CUDA graph capture;
  important because capture mutates replay buffers. This had to handle Mamba
  `kv_cache` as a list, not only tuple.
- Real fix retained for now: reset checkpointing trackers after non-simple
  fallback paths (`not simple_decode`). Broad fallback reset helped but caused
  regressions; narrow non-simple reset improved without those regressions.
- Temporary diagnostic: tracker boundary changed from `prev + seq_len > window`
  to `prev + seq_len >= window` for job `1895017`
  (`nemotron-stp-fastclean-cg-ci6-boundaryge`). Focused tests showed this changes
  tracker expectation at boundary (`prev` resets at step 6), so only keep it if
  the lm_eval result improves materially; otherwise revert.
- Target-doc direct completions probe job `1894965` ran with CUDA graph and
  produced all-wrong responses, but this probe is not lm_eval-equivalent enough
  to use as a primary accuracy signal: responses were truncated/malformed around
  56s. Treat it only as stress evidence, not as the metric.

Current blocker:

The remaining gap is the full vLLM CUDA-graph STP lifecycle, likely around
checkpoint replay tracker/cache state across mixed prefill, fallback/non-simple
paths, and full decode. Do not start MTP until STP reaches old-kernel accuracy
with CUDA graph or there is a clearly documented unresolved kernel issue.

## 2026-05-24 17:05 boundary experiment result

- CUDA-graph STP eval job 1895017 (nemotron-stp-fastclean-cg-ci6-boundaryge-20260524_165951) ran production-style CUDA graph with checkpoint interval 6, stochastic rounding enabled, and both PIECEWISE/FULL graph capture completed.
- Result was a clear regression: GSM8K limit50 strict=0.16, flexible=0.18. Throughput was graph-shaped after warmup, about 4934 prompt tok/s and 226 gen tok/s, so this was not an eager/slow-path artifact.
- The temporary tracker boundary change from greater-than to greater-or-equal was reverted. This experiment should not be repeated unless kernel semantics change; it also broke focused tracker assertions.
- Current best STP CUDA-graph accuracy remains the narrow fallback reset run nemotron-stp-fastclean-cg-ci6-reset-nonsimple-20260524_161729: strict=0.86, flexible=0.88 versus old-kernel CUDA-graph baseline strict=0.94, flexible=0.94. STP is still not PR-ready.

## 2026-05-24 17:15 concurrency isolation

- Same STP CUDA-graph branch, same checkpoint interval 6 and SR enabled, rerun with different lm_eval concurrency.
- Concurrency 1 job 1895053 (nemotron-stp-fastclean-cg-ci6-conc1-20260524_170826): strict=0.90, flexible=0.94. This is close to old-kernel CUDA-graph baseline, so the checkpointing kernel itself can be accurate under CUDA graph.
- Concurrency 8 job 1895054 (nemotron-stp-fastclean-cg-ci6-conc8-20260524_170908): strict=0.86, flexible=0.86.
- Prior concurrency 50 best remains strict=0.86, flexible=0.88. Wrong samples grow with concurrency and include invalid/malformed responses, which points to concurrent request/slot lifecycle rather than raw kernel math.
- Tried source-tracker-clear experiment after copying checkpoint state from source to destination slot. Focused tests caught a contract regression: test_checkpointing_ssu_copies_replay_state_to_destination_slot expected source tracker state to remain unchanged. Experiment was reverted and patched evals 1895695/1895696 were cancelled.
- Next target should be the full CUDA-graph concurrent slot lifecycle around state_indices_tensor_d input/output gather, block_idx_last_computed/scheduled metadata, and tracker reset on true request allocation/reuse, without violating the source-preservation copy contract.

## 2026-05-24 17:50 debug and interval controls

- Debug conc8 CUDA-graph job 1895729 (nemotron-stp-fastclean-cg-ci6-conc8-debug-20260524_173301) completed with strict=0.80, flexible=0.84. This is worse than the earlier conc8 run, but the logs were useful.
- Runner debug showed steady FULL graph rows with coherent cache/prev cycles. Example around calls 55-70: active slots stayed stable and `prev_num_accepted_tokens` advanced through the checkpoint window as expected.
- The suspicious transition is when long `cg=NONE` mixed prefill/extend rows interrupt active decode rows, then FULL graph resumes. Example calls 71/73 had large token counts (~1200/~1165) while decode slots continued and new request slots entered.
- Wrong samples are not only scorer artifacts. Some are clipped before `####`, while others are real content/arithmetic drift (for example doc 15 predicts 29 instead of 125). Treat this as model-state perturbation under serving concurrency.
- CUDA-graph control job 1895871 used the new kernel with default checkpoint interval 1, conc8, SR enabled. It was a hard regression: strict=0.00, flexible=0.04. Do not use interval 1 for the new checkpointing STP path.
- Previous CI=6 no-SR discriminator job 1894862 already showed strict=0.38, flexible=0.44. Disabling stochastic rounding is not the fix; SR-on CI=6 remains the best serving path.
- Current best STP CUDA-graph path remains checkpoint interval 6, SR enabled, narrow non-simple fallback reset: strict=0.86, flexible=0.88 at high concurrency, and strict=0.90, flexible=0.94 at concurrency 1.

## 2026-05-24 19:35 STP current best facts

- STP-only work. MTP has not been started in this clean branch.
- Old FlashInfer STP CUDA-graph baseline remains GSM8K limit50 strict=0.94, flexible=0.94 (`stp-oldkernel-cg-limit50-limit50-20260524_121942`).
- New checkpointing STP eager/no-SR now matches old baseline exactly: strict=0.94, flexible=0.94 (`nemotron-stp-fastclean-eager-ci6-nosr-latest-20260524_184323`). This confirms the new eager kernel can be accurate and fast enough for the limit50 run.
- New checkpointing STP eager/SR is strict=0.92, flexible=0.92 (`nemotron-stp-fastclean-eager-ci6-latest-20260524_183511`), with one extra corrupted sample (doc 19 repeating `2`). SR interaction is suspicious but not the main no-SR graph issue.
- Resetting all Mamba kv_cache tensors after CUDA graph capture improved the graph path substantially. It fixed the old catastrophic no-SR graph run (previously 0.38/0.44) and moved SR graph to strict=0.90, flexible=0.90 (`nemotron-stp-fastclean-cg-ci6-resetallcache-20260524_182309`).
- CUDA graph no-SR at concurrency 1 is flexible-baseline clean: strict=0.90, flexible=0.94 (`nemotron-stp-fastclean-cg-ci6-nosr-c1-afterreset-20260524_184854`). Extra strict failures are formatting-only docs 1/3.
- CUDA graph no-SR at concurrency 8 remains bad: strict=0.88, flexible=0.90 (`nemotron-stp-fastclean-cg-ci6-nosr-c8-afterreset-20260524_190741`). Extra real flexible failures are docs 15 and 38; doc 38 truncates at `He runs 3/2=`. This matches the c50 no-SR shape (`nemotron-stp-fastclean-cg-ci6-nosr-afterreset-20260524_183538`, strict=0.88, flexible=0.90).
- A new focused unit test was added: `test_checkpointing_ssu_stp_cudagraph_replay_varying_active_rows_matches_old_flashinfer`. It replays a captured SSU graph with fixed padded tensors while active row counts and request slots churn. Job 1896863 passed: 1 passed, 16 warnings in 9.24s. This means the raw checkpointing SSU op tolerates server-like active-row churn in isolation.
- Therefore the remaining graph bug is above the raw SSU op, likely in the full runner/model graph path: padded hidden-state flow, conv update, or Mamba cache lifecycle around full graph replay under multiple active requests.
- Diagnostic env flag added in `gpu_model_runner.py`: `MAMBA_DISABLE_BATCHED_FULL_CUDAGRAPH=1` disables FULL graph only for runtime uniform decode batches with `num_reqs > 1` and does not affect forced capture/dummy paths. First attempt 1896879 failed before serving because the flag affected dummy capture; fixed by guarding with `force_uniform_decode is None`. Resubmitted as job 1896906 (`nemotron-stp-fastclean-cg-ci6-nosr-nobatchfull-c8v2-20260524_193029`) and still running at this update.
- Do not treat the env flag as final. It is a discriminator: if c8/no-SR recovers to 0.94 flexible, the corruption is specifically batched FULL replay around Mamba, not the new STP kernel.

## 2026-05-24 20:05 STP null-row hypothesis

- Diagnostic job 1896906 completed and disproved the batched-FULL-disable idea. Run `nemotron-stp-fastclean-cg-ci6-nosr-nobatchfull-c8v2-20260524_193029` scored strict=0.86, flexible=0.86, worse than normal c8 no-SR. Extra corruptions included doc 5 repeating `#### 160` and docs 25/38 truncating. The temporary `MAMBA_DISABLE_BATCHED_FULL_CUDAGRAPH` code was removed from `gpu_model_runner.py`.
- Added focused conv+SSU graph replay coverage. After fixing a test dimension typo, job 1896942 passed: 2 passed, 16 warnings in 7.02s. Together with job 1896863, raw SSU and conv+SSU graph replay tolerate active-row churn.
- New hypothesis: FULL graph padded/null decode rows may leave stale static `preallocated_ssm_out_d` contents after SSU skips `NULL_BLOCK_ID` rows, and those stale rows can leak through later full-model graph work at concurrency. A patch in `mamba_mixer2.py` now reuses a single `ssm_out_d` view and multiplies rows whose entire `state_indices_tensor_d_input` row equals `NULL_BLOCK_ID` by zero after SSU.
- Focused null-row patch test job 1896989 passed: 3 passed, 16 warnings in 7.20s.
- Production discriminator now running: job 1897015, run label `nemotron-stp-fastclean-cg-ci6-nosr-nullrow-c8`, STP CUDA graph, checkpoint interval 6, no stochastic rounding, max_num_seqs/eval concurrency 8, GSM8K limit50. Earlier attempts 1897002 and 1897007 were launcher mistakes (`SOURCE_ROOT` missing, then CI=1) and should be ignored. Compare 1897015 directly to pre-patch c8 no-SR `0.88/0.90` and c1 no-SR `0.90/0.94`.
- Job 1897015 completed successfully: strict=0.92, flexible=0.92. Throughput after warmup was graph-shaped (`~2406/2367 prompt tok/s`, `~233/275 gen tok/s`) with 8 running requests.
- The null-row patch fixed the obvious graph corruption class: pre-patch c8 no-SR had extra real flexible failures docs 15 and 38, with doc 38 truncated at `He runs 3/2=`. Post-patch wrong docs are 8/12/20/28; 8/12/20 are old/eager baseline misses, and only doc 28 is new graph-specific drift (`20` instead of `25`). No short/truncated samples remain.
- Next useful run is high-concurrency no-SR (c50/max_num_seqs 50 or 128) with the null-row patch. If c50 also moves to `0.92/0.92` or better, this should be retained as a real fix, but STP is still short of the old-kernel `0.94/0.94` target.


## 2026-05-24 20:25 STP null-row high-concurrency result

- High-concurrency null-row discriminator job 1897127 completed: run `nemotron-stp-fastclean-cg-ci6-nosr-nullrow-c50-limit50-20260524_200443`, STP CUDA graph, checkpoint interval 6, stochastic rounding disabled, `max_num_seqs=128`, lm_eval concurrency 50. Result regressed to strict=0.78, flexible=0.78.
- This is not the same tiny/empty-output class as earlier. Samples have normal-ish lengths and no <=10 word outputs, but many new misses terminate mid-answer or mid-number before the final `####`: docs 2, 7, 15, 26, 39, 43, 47. Doc 47 also has stray filler near the end (`s`, `8`, `s`). Wrong docs for c50 null-row are [2, 7, 8, 12, 15, 20, 26, 31, 39, 43, 47]. Old/eager baseline misses remain [8, 12, 20]; c8 null-row misses were [8, 12, 20, 28].
- Interpretation: zeroing all-null SSU rows is a real partial fix for c8 because it removed the doc15/doc38 truncation path, but it does not solve larger FULL-graph/high-concurrency state contamination. At c50 the failure looks like normal generation is interrupted or corrupted late, not raw SSU graph replay math.
- Two threshold jobs were submitted to locate the concurrency/graph-shape break point using the same null-row patch and no SR: job 1897503 (`c16`, `PORT=8430`, `MAX_NUM_SEQS=16`, `EVAL_CONCURRENCY=16`) and job 1897504 (`c24`, `PORT=8431`, `MAX_NUM_SEQS=32`, `EVAL_CONCURRENCY=24`). Compare against c8 null-row 0.92/0.92 and c50 null-row 0.78/0.78.
- STP remains not PR-ready. Keep MTP paused. Next suspect area is full-model CUDA graph replay/static buffer state at larger captured decode shapes, above raw conv+SSU focused replay coverage.


## 2026-05-24 20:35 STP threshold and rejected padded-prefill experiment

- Null-row-only threshold jobs completed and show the graph/concurrency break begins above c8:
  - job 1897503, `nemotron-stp-fastclean-cg-ci6-nosr-nullrow-c16-limit50-20260524_201437`, `max_num_seqs=16`, eval concurrency 16: strict=0.84, flexible=0.84.
  - job 1897504, `nemotron-stp-fastclean-cg-ci6-nosr-nullrow-c24-limit50-20260524_201506`, `max_num_seqs=32`, eval concurrency 24: strict=0.82, flexible=0.84.
  - Earlier c8 null-row was strict=0.92, flexible=0.92; c50 null-row was strict=0.78, flexible=0.78.
- Tried a narrow diagnostic in `gpu_model_runner.py` forcing `is_prefilling[num_reqs:num_reqs_padded] = False` for FULL-CG padded virtual rows. Job 1897663 (`nemotron-stp-fastclean-cg-ci6-nosr-nullrow-padprefill-c50-limit50-20260524_201942`) regressed badly to strict=0.40, flexible=0.46. The diagnostic was reverted locally and copied back to the remote branch. Do not keep or repeat this patch.
- Interpretation: padded request-row prefill classification was not the fix. The c16/c24/c50 shape suggests the remaining issue scales with larger captured FULL decode graph sizes / more concurrent active rows. The null-row SSU output zeroing is still a useful c8 partial fix, but high-concurrency corruption remains above raw conv+SSU graph replay.
- Current remote branch has the padded-prefill diagnostic reverted. Keep MTP paused. Next useful directions: compare c16/c24/c50 wrong sample tails to c8, add layer/output debug for larger FULL graph shapes, or test a targeted zeroing of padded token rows at a later full-model boundary without altering request prefill classification.
