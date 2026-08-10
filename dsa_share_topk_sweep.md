# DSA Shared Top-K Sweep

## Container

Use this sqsh for the shared top-k sweep:

```bash
/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/users/mdabbah/deci/vllm_repos/vllm_v0.20.1.worktrees/nano-chunked-dsa-moonshot/outputs/containers/vllm-openai_v0.20.1_nemotron-h-dsa-moonshot-share-topk-strict-union-sum_20260603_105104.sqsh
```

The code includes the prefill boundary full-FA fallback, configurable query share group size, and these shared top-k modes:

```text
representative
mean
histogram
union
strict_union_sum
```

All share modes apply only in prefill. Decode remains per-token.

## Base Flags

Set these for every sweep run:

```bash
export VLLM_NEMOTRON_H_DSA_ATTN_MODE=chunked_topk_sparse
export VLLM_NEMOTRON_H_DSA_CHUNK_SIZE=16
export VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K=1024
export VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16

export VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=1
export VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA=1
export VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ=1

export VLLM_NEMOTRON_H_DSA_SHARE_CHUNK_TOPK=1
export VLLM_NEMOTRON_H_DSA_USE_SHARED_PREFILL_PAGE_TABLE_FA=1
export VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE=4096
```

Keep these runtime flags unless the eval specifically wants eager/deep-gemm variants:

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
```

## Sweep Grid

Sweep this 2 x 5 grid:

```text
VLLM_NEMOTRON_H_DSA_SHARE_TOPK_GROUP_SIZE: 4, 8
VLLM_NEMOTRON_H_DSA_SHARE_TOPK_MODE: representative, mean, histogram, union, strict_union_sum
```

Example run environment:

```bash
export VLLM_NEMOTRON_H_DSA_SHARE_TOPK_GROUP_SIZE=4
export VLLM_NEMOTRON_H_DSA_SHARE_TOPK_MODE=strict_union_sum
```

## Mode Meanings

`representative`: one representative q-indexer row scores history for the share group. This is the current fast shared mode baseline.

`mean`: averages all q-indexer rows in the share group, then scores history once. This is a prefill-only routing approximation that uses future prompt rows inside the group.

`histogram`: computes per-query top-k, unions those candidates, then picks the chunks with the most votes. Final recall width is capped at `CHUNK_TOP_K`.

`union`: computes per-query top-k and recalls the union for the share group. This can recall more than `CHUNK_TOP_K` chunks. The shared page table is padded and uses per-row `seqused_k`.

`strict_union_sum`: computes per-query top-k, takes the union candidates, sums the original chunk scores over the share group for those candidates, then runs one final top-k. Final recall width is capped at `CHUNK_TOP_K`.

## Optional Union Cap

Only `union` uses this cap:

```bash
export VLLM_NEMOTRON_H_DSA_SHARE_TOPK_UNION_MAX_CHUNKS=2048
```

Default is `max(CHUNK_TOP_K, CHUNK_TOP_K * 2)`, so with `CHUNK_TOP_K=1024` the default is `2048`.

If `union` is accurate but slow, try:

```text
1280, 1536, 2048
```

## Conservative Baseline

To disable sharing in this same container:

```bash
export VLLM_NEMOTRON_H_DSA_SHARE_CHUNK_TOPK=0
export VLLM_NEMOTRON_H_DSA_USE_SHARED_PREFILL_PAGE_TABLE_FA=0
```

This keeps prefill/decode page-table FA but returns to per-query top-k in prefill.

## Useful Logging

For timing:

```bash
export VLLM_NEMOTRON_H_DSA_TIMING=1
```

The timing line includes:

```text
share_topk_group_size=<N>
share_topk_mode=<mode>
```

For top-k reuse stats:

```bash
export VLLM_NEMOTRON_H_DSA_TOPK_STATS=1
export VLLM_NEMOTRON_H_DSA_TOPK_STATS_LIMIT=200
```
