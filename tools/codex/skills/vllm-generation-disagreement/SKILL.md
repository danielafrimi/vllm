---
name: vllm-generation-disagreement
description: Run and analyze the vLLM fast correctness test / generation disagreement smoke test for performance refactors, especially Nano/Nemotron-H DSA work. Use when asked to run the fast correctness test on the current code, check current-code generation correctness, create a fast non-bit-exact correctness baseline, run same-model repeatability generations, compare current generations against a JSONL reference, truncate existing generation artifacts to the 200-output-token baseline, or inspect disagreement statistics such as common-prefix length, early divergence, output length drift, and finish-reason mismatches.
---

# vLLM Generation Disagreement

## Workflow

Use the repo-local tooling under `scripts/generation_disagreement/`.

When the user asks to "run the fast correctness test on the current code" or
similar, generate a fresh current-code JSONL artifact and compare it against
the default sparse-coverage 200-token baseline below.

1. Generate deterministic outputs with `generate.py`.
   - Default workload: 50 fixed prompt specs, 4096 prompt tokens each, 200 output tokens.
   - Pass `--model` explicitly.
   - Use greedy decoding: the script sets `temperature=0`.
   - For Nano/Nemotron-H DSA Slurm/Pyxis runs, prefer the tracked launcher
     `scripts/generation_disagreement/run_sparse_agreement.sbatch` and pass the
     current-code SQSH image with `IMAGE=/path/to/image.sqsh`.
   - The default DSA coverage shape is:
     `--target-prompt-tokens 4096`, `--max-tokens 200`,
     `--max-model-len 8192`, `--max-num-seqs 4`,
     `--enable-chunked-prefill`, `--max-num-batched-tokens 1024`,
     `--block-size 16`, and `--enforce-eager`.
   - Use these DSA environment variables for the default path-coverage run:
     `VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16`,
     `VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K=128`,
     `VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=1`,
     `VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA=1`,
     `VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ=1`,
     `VLLM_NEMOTRON_H_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA=1`,
     `VLLM_NEMOTRON_H_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA=1`,
     `VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE=4096`,
     `VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING=1`,
     `VLLM_NEMOTRON_H_DSA_USE_TRITON_BATCHED_SUMMARIES=1`,
     `VLLM_NEMOTRON_H_DSA_DENSE_PREFILL_KV_THRESHOLD_TOKENS=2048`, and
     `VLLM_NEMOTRON_H_DSA_PATH_DEBUG_PRINT_LIMIT` greater than zero.
   - After generation, run
     `scripts/generation_disagreement/check_dsa_path_markers.py` on the log.
     Treat missing markers as an execution failure. Required coverage is:
     `config`, `dense_prefill_page_table_bucket`,
     `sparse_prefill_page_table_bucket`, and `sparse_decode`. In the user
     report, it is enough to say whether all required DSA path markers were
     seen; do not paste the marker lines unless asked.

2. Compare outputs with `compare.py`.
   - Compare JSONL files by `output_token_ids`, not text.
   - Use `tools/codex/skills/vllm-generation-disagreement/baselines/baseline_sparse_4096_200.jsonl` as the default reference baseline when it exists.
   - Do not compare a 4096-token sparse-coverage candidate against the legacy
     1000-token `baseline_200.jsonl` unless the user explicitly asks for a
     legacy comparison and accepts `--allow-prompt-token-mismatch`.
   - Do not pass `--fail-under-*` or `--fail-over-*` options; compute pass/fail from the reported metrics after the comparison.
   - Use `--early-threshold 10` only to report the early-divergence bucket.
   - Report exact match count, min/p10/p25/median/mean/p75/p90/max agreement tokens, output length ranges, early-divergence count, finish-reason mismatches, and worst prompts.

3. Truncate old artifacts with `truncate.py` when converting an older long-generation run to the 200-token baseline.
   - Example:
     ```bash
     .venv/bin/python scripts/generation_disagreement/truncate.py \
       outputs/generation_disagreement/repeatability_83683/run1.jsonl \
       tools/codex/skills/vllm-generation-disagreement/baselines/baseline_200.jsonl \
       --max-output-tokens 200 --overwrite
   ```
   - The script removes stale `output_text`, updates token lengths and token hashes, and records `truncated_from` metadata.

## Correctness Status

For the default 200-token disagreement comparison, classify the run by mean
agreement tokens:

- `PASS`: mean agreement is at least 40 tokens.
- `FAIL`: mean agreement is below 40 tokens, required artifacts are missing, the
  generation job fails, or finish-reason mismatches indicate an execution error.

When reporting status, state both the mean agreement in tokens and the normalized
mean agreement (`mean / 200`) when using the 200-token baseline. Do not invert
this into an error threshold unless the user explicitly asks for error.

## Baseline

Default reference baseline:
`tools/codex/skills/vllm-generation-disagreement/baselines/baseline_sparse_4096_200.jsonl`

The baseline JSONL is intentionally gitignored because it is generated data.
Keep it co-located with the skill locally, but do not add it to commits.
If it is missing, recreate it by running a fresh same-model branch-start DSA
baseline with the default sparse-coverage workload.

The default sparse-coverage baseline came from Slurm job `93605`, using:

- Model: `/lustre/fsw/portfolios/coreai/users/mdabbah/deci/puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints/nano-dsa-16chunk_size-2048chunks`
- Image: `outputs/containers/vllm-v0.22.0-fixedio-current-bae8a130f-dirty-stripped-20260611_030209.sqsh`
- Prompt count: 50
- Prompt length: 4096 tokens
- Output length: 200 tokens
- Engine shape: TP2, `--max-model-len 8192`, `--max-num-seqs 4`,
  `--enable-chunked-prefill`, `--max-num-batched-tokens 1024`, block size 16,
  dense prefill threshold 2048.

This baseline is a branch-start DSA implementation (`bae8a130f`) with the DSA
checkpoint, not vanilla Nano. The older `baseline_200.jsonl` and Slurm job
`83683` are legacy 1000-token prompt artifacts and should not be the default
for sparse-path coverage.

## Notes

- Do not modify repo runtime source for this task unless explicitly asked. The
  sparse launcher may transiently instrument the installed container copy of
  `nemotron_h_dsa_attention_legacy.py` so marker coverage can be checked.
- Keep generated artifacts under `outputs/generation_disagreement/`.
- Use `.venv/bin/python` for local Python commands in this repo. Create it with `UV_CACHE_DIR=/tmp/uv-cache uv venv --python 3.12` if missing.
- Slurm commands may require unsandboxed execution because they contact the cluster controller.
