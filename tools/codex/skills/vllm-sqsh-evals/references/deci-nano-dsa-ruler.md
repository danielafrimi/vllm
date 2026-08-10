# Deci Nano DSA RULER Eval

Use this reference when launching Deci/Nemo evaluator runs for Nano DSA RULER tasks.

## Current Baseline Shape

Defaults used in the known-good launch:

- `TASKS`: `ruler-128k-completions` or `ruler-1m-completions`
- `TOPK=1024` for RULER-128K
- `TOPK=2048` for RULER-1M
- `QUERY_CHUNK_SIZE=4096`
- `NUM_NODES=4`
- `NUM_INSTANCES=4`
- `TP_SIZE=8`
- `DP_SIZE=1`
- `MAX_NUM_SEQS=8`
- `MAX_MODEL_LEN=131200` for RULER-128K completions
- `MAX_MODEL_LEN=1048704` for RULER-1M completions

The DSA checkpoint config supplies `q_indexer_attn_mode`, `q_indexer_chunk_size`, and `q_indexer_chunk_top_k`, so the launcher should not force those unless intentionally testing an override.

## vLLM Flags

Use:

- `--enforce-eager`
- `--enable-chunked-prefill`
- `--max-model-len <benchmark length>`
- `--max-num-seqs 8`
- `--no-enable-prefix-caching`
- `--block-size 16`
- `--attention-backend FLASH_ATTN`
- `--enable-expert-parallel`

Use environment overrides:

- `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`
- `VLLM_WORKER_MULTIPROC_METHOD=fork`
- `VLLM_USE_DEEP_GEMM=0`
- `VLLM_MOE_USE_DEEP_GEMM=0`
- `VLLM_DEEP_GEMM_WARMUP=skip`
- `VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16`
- `VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=1`
- `VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA=1`
- `VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ=1`
- `VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE=4096`
- `VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING=1`

## Launch Pattern

Prefer a repo-local script, if present:

```bash
./scripts/launch_nano_dsa_eval.sh ruler-128k-completions --config-only
./scripts/launch_nano_dsa_eval.sh ruler-128k-completions
./scripts/launch_nano_dsa_eval.sh ruler-1m-completions --config-only
./scripts/launch_nano_dsa_eval.sh ruler-1m-completions
```

The launcher should:

1. Run or verify `uv sync` in the `deci-evals` checkout.
2. Source secrets from a file without printing them.
3. Fail early if required secrets are missing.
4. Generate a CPU sbatch wrapper.
5. Use `.venv/bin/evaluate ... --config-only` for dry runs.
6. Use `sbatch` for actual launches.

## Monitoring

Check:

```bash
squeue -j <gpu_job>,<dependency_job>
sacct -j <gpu_job> --format=JobID,JobName%50,State,Elapsed,ExitCode,NNodes,NodeList%40 --parsable2
```

Results are usually under `<run_dir>/artifacts/results.yml`. Parse that file for overall and per-task accuracy. Treat server/proxy steps cancelled after export as normal if the main job is `COMPLETED` with `0:0` and results are present.

Never print raw logs that may include exported secrets. Use redacted `rg` patterns for `Traceback|ERROR|Exception|OOM|Killed|FAILED` and task markers such as `Run data preparation for task` and `Run predictions generation for task`.
