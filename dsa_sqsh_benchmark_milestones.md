# DSA SQSH Benchmark Milestones

This file tracks benchmarkable SQSH images for the Nemotron-H chunked DSA work.
Each image is built with the copy trick: start from the stock vLLM v0.22.0
SQSH, copy the patched local `vllm/` Python package over the installed package
inside the image, and preserve the stock compiled extensions.

## Common Runtime Flags

For the nano DSA checkpoint trained with 16-token chunks, keep the DSA chunk
size, vLLM cache block size, and attention kernel block size aligned:

```bash
export VLLM_NEMOTRON_H_DSA_ATTN_MODE=chunked_topk_sparse
export VLLM_NEMOTRON_H_DSA_CHUNK_SIZE=16
export VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K=4096
export VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=1
export VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA=1
export VLLM_NEMOTRON_H_DSA_USE_SUMMARY_CACHE=1
export VLLM_NEMOTRON_H_DSA_SUMMARY_CACHE_MAX_BLOCKS=65536
export VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16
```

For milestone 3 and later, enable the fused Triton chunk scorer:

```bash
export VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING=1
```

Launch vLLM with `--block-size 16` when using the page-table FA path.

Optional shared recall modes remain available:

```bash
export VLLM_NEMOTRON_H_DSA_SHARE_CHUNK_TOPK=1
export VLLM_NEMOTRON_H_DSA_SHARE_TOPK_GROUP_SIZE=16
export VLLM_NEMOTRON_H_DSA_SHARE_TOPK_MODE=mean
```

Supported `VLLM_NEMOTRON_H_DSA_SHARE_TOPK_MODE` values are `representative`,
`mean`, `histogram`, `union`, and `strict_union_sum`. `causal_mean` is removed
and should not be used.

## Milestone 1: Row-Wise Page-Table FA

SQSH:

```text
/scratch/fsw/portfolios/coreai/projects/coreai_nvfm_llm/users/mdabbah/deci/vllm/outputs/containers/vllm-v0.22.0-nemotron-h-dsa-rowwise-page-table-fa-no-causal-mean-9c294b2c04-20260607_071909.sqsh
```

Base image:

```text
/scratch/fsw/portfolios/coreai/projects/coreai_nvfm_llm/containers/vllm/vllm_v0.22.0.sqsh
```

Commit:

```text
9c294b2c04 Optimize Nemotron-H DSA page-table FA rows
```

SHA256:

```text
0c3dcc4ed38b17e94a38636887a62bef2c1ddc29271c33148c3e8d3b72bff512
```

Changes:

- Removes `causal_mean` share mode and the old causal-mean query aggregation.
- Routes both decode and prefill page-table FlashAttention through a row-wise
  helper.
- Compacts invalid recalled chunks per row so rows with fewer than
  `chunk_top_k` valid prior chunks still use page-table FA instead of the slow
  materialized fallback.
- Uses per-row `seqused_k = valid_recalled_chunks * chunk_size + tail_len`.

Validation:

- `git diff --check` passed before commit.
- Manual DSA test harness passed inside the stock v0.22.0 container.
- Saved SQSH smoke test confirmed `prefill_page_table_rows` is present and
  `causal_mean` is absent from `nemotron_h.py`.

Notes:

- This milestone still has Python-side metadata decisions and is not expected
  to be the final no-`--enforce-eager` implementation.
- Next milestone should move page-table / row metadata into Izik-style CPU
  plumbing and reusable workspaces so mixed piggyback batches can be split and
  launched without per-sequence hot-path loops.

## Milestone 2: CPU Metadata Plumbing

SQSH:

```text
/scratch/fsw/portfolios/coreai/projects/coreai_nvfm_llm/users/mdabbah/deci/vllm/outputs/containers/vllm-v0.22.0-nemotron-h-dsa-cpu-metadata-plumbing-34ce55b25a-20260607_074217.sqsh
```

Base image:

```text
/scratch/fsw/portfolios/coreai/projects/coreai_nvfm_llm/containers/vllm/vllm_v0.22.0.sqsh
```

Commit:

```text
34ce55b25a Plumb CPU attention metadata into Nemotron-H DSA
```

SHA256:

```text
66b5876551a783e6157c520c71095cc7ceee517576640ad1d84b5192cb22890b
```

Changes:

- Adds `query_start_loc_cpu` and `seq_lens_cpu` to v1 FlashAttention metadata,
  mirroring Izik's CPU metadata plumbing.
- Passes CPU query starts and sequence lengths from common attention metadata
  into `FlashAttentionMetadata`.
- Updates Nemotron-H DSA active sequence discovery to prefer CPU metadata and
  avoid device `.item()` synchronization when identifying prefill/decode rows.

Validation:

- `git diff --check` passed before commit.
- Manual DSA test harness passed inside the stock v0.22.0 container.
- Saved SQSH smoke test confirmed row-wise FA code, CPU metadata fields, and
  removal of `causal_mean` are present in the image.

Notes:

- The login-node saved-image smoke test does not import CUDA extensions because
  `libcuda.so.1` is not mounted there. Run benchmark and numerical validation on
  a GPU node with the normal driver mount.
- This milestone still uses Python to build selected block tables. The next
  milestone should fuse chunk scoring / top-k selection to reduce Python and
  PyTorch launch overhead before moving more work into captured execution.

## Milestone 3: Triton Scoring Selector

SQSH:

```text
/scratch/fsw/portfolios/coreai/projects/coreai_nvfm_llm/users/mdabbah/deci/vllm/outputs/containers/vllm-v0.22.0-nemotron-h-dsa-triton-scoring-selector-834834b499-20260607_080330.sqsh
```

Base image:

```text
/scratch/fsw/portfolios/coreai/projects/coreai_nvfm_llm/containers/vllm/vllm_v0.22.0.sqsh
```

Commit:

```text
834834b499 Add Triton DSA chunk scoring selector
```

SHA256:

```text
fb327fde5c172176fe35ad9c7350c0dd25c25031da3709b1ab830ff08e92ef89
```

Additional flag:

```bash
export VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING=1
```

Optional scoring tile override:

```bash
export VLLM_NEMOTRON_H_DSA_TRITON_SCORING_BLOCK_CHUNKS=64
```

Changes:

- Adds a Triton chunk-score kernel that computes query-summary dot products
  into a float32 score workspace.
- Uses vLLM's CUDA `top_k_per_row_prefill` op for chunk top-k selection, so
  the hot score path can avoid `torch.matmul` plus `torch.topk` when enabled.
- Keeps existing chunked DSA semantics and share modes. `representative`,
  `mean`, `histogram`, `union`, `strict_union_sum`, and no-share all flow
  through the same page-table FA logic after selection.
- Falls back to the existing PyTorch scoring path if Triton, CUDA, or the
  vLLM CUDA top-k op is unavailable.

Validation:

- `git diff --check` passed before commit.
- Manual DSA test harness passed inside the stock v0.22.0 container with local
  source mounted.
- Saved SQSH smoke test confirmed the Triton scoring module, top-k op call,
  CPU metadata fields, and removal of `causal_mean` are present in the image.

Notes:

- The login-node smoke test cannot execute the Triton/CUDA scorer because
  `libcuda.so.1` is not mounted there. Benchmark this image on a GPU node with
  `VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING=1`.
- Timing output includes `triton_score_topk_calls`; when this remains zero on a
  GPU run, inspect whether Triton or the vLLM CUDA top-k op was unavailable.
