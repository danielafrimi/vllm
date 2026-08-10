---
name: performance-benchmark
description: Run the vLLM performance benchmark on Slurm/Pyxis. Use when asked to run, launch, rerun, compare, tune, or debug "the performance benchmark" for vLLM/Nano DSA with specific serve options, environment variables, TP size, SQSH images, prompt counts, warmups, request rates, or benchmark result artifacts.
---

# Performance Benchmark

## Core Workflow

Use `benchmarks/launch_fixed_io_slurm_benchmark.sh` from the vLLM checkout when the user wants the performance benchmark without Nsight Systems profiling. It starts `vllm serve` inside a Pyxis/SQSH container, runs `benchmarks/benchmark_fixed_io_serving.py`, and writes a JSON result.

1. Work from the vLLM repo and read local instructions such as `AGENTS.md`.
2. Verify the intended SQSH image. If current local code must be inside the image, use `$vllm-sqsh-evals` first.
3. Prefer explicit images, model paths, TP size, GPU count, request shape, and job name. Avoid relying on auto-detected images for comparisons.
4. Launch with `--wait` only when the user wants this turn to monitor completion. Otherwise capture the printed `job_id` and `run_dir`, then monitor with `squeue`/`sacct`.
5. Summarize `results/fixed_io_result.json`, plus the relevant server and Slurm log paths. If the result is missing, inspect the Slurm log and `logs/server.log`.

## Common Launches

Smoke test a container:

```bash
benchmarks/launch_fixed_io_slurm_benchmark.sh \
  --smoke \
  --image "$IMAGE" \
  --job-name fixedio-smoke \
  --wait
```

Short Nano DSA long-context run, TP2:

```bash
benchmarks/launch_fixed_io_slurm_benchmark.sh \
  --nano-dsa \
  --image "$IMAGE" \
  --tp-size 2 \
  --gpus 2 \
  --input-len 1000000 \
  --output-len 10 \
  --num-prompts 2 \
  --num-warmups 0 \
  --partition batch \
  --time 03:00:00 \
  --job-name fixedio-nano-dsa-tp2-short \
  --wait
```

TP4 uses the same command with `--tp-size 4 --gpus 4`. Increase `--num-prompts`, `--num-warmups`, and walltime only after the short run succeeds.

Decode-oriented Nano DSA run, TP2:

Use this when the user wants decoding throughput rather than a mostly-prefill
shape. Keep ISL high enough to exercise long-context decode, but keep OSL short
enough that the run finishes quickly.

```bash
benchmarks/launch_fixed_io_slurm_benchmark.sh \
  --nano-dsa \
  --image "$IMAGE" \
  --tp-size 2 \
  --gpus 2 \
  --input-len 100000 \
  --output-len 2048 \
  --num-prompts 64 \
  --max-concurrency 64 \
  --max-num-seqs 64 \
  --num-warmups 0 \
  --max-model-len 131072 \
  --partition batch \
  --time 01:00:00 \
  --job-name fixedio-nano-dsa-decode-tp2 \
  --wait
```

For this scenario, report both total throughput and completion throughput. Treat
completion throughput as the first-pass decode signal, and inspect server logs
to confirm prompt throughput has dropped to `0.0` while all requests are running.

Long realistic Nano DSA run, TP2:

Use this only when the user asks for the longer or long benchmark scenario. It
uses 64 real 256K-token prompts and a 4K-token generation so the run reaches a
steady state where all 64 requests are in generation. It forces DSA to select
1024 chunks (`VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K=1024`). Use TP2 unless the user
explicitly changes it.

Prepare the prompt-token file once, using the same tokenizer/model path as the
served Nano DSA model. The default source is PG-19, because it provides long
Project Gutenberg book text that can produce contiguous 256K-token slices. If
the user gives a better local corpus, pass it with `--source-text-file` instead.

```bash
MODEL=/lustre/fsw/portfolios/coreai/users/mdabbah/deci/puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints/nano-dsa-16chunk_size-2048chunks
PROMPT_IDS=outputs/fixed_io_prompts/pg19_nano_dsa_64x262144_seed0.jsonl

.venv/bin/python benchmarks/prepare_fixed_io_prompt_token_ids.py \
  --dataset-name deepmind/pg19 \
  --split train \
  --tokenizer "$MODEL" \
  --trust-remote-code \
  --num-prompts 64 \
  --input-len 262144 \
  --seed 0 \
  --output "$PROMPT_IDS"
```

Then launch:

```bash
benchmarks/launch_fixed_io_slurm_benchmark.sh \
  --nano-dsa \
  --image "$IMAGE" \
  --tp-size 2 \
  --gpus 2 \
  --gpu-mem 0.95 \
  --mem 256G \
  --input-len 262144 \
  --output-len 4096 \
  --num-prompts 64 \
  --max-concurrency 64 \
  --max-num-seqs 64 \
  --prompt-token-ids-jsonl "$PROMPT_IDS" \
  --server-env VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K=1024 \
  --num-warmups 0 \
  --max-model-len 270336 \
  --partition batch_long \
  --time 12:00:00 \
  --benchmark-extra-args "--timeout-s 43200" \
  --job-name fixedio-nano-dsa-long-real-tp2 \
  --wait
```

For this scenario, report completion throughput as the main decode signal and
also report total throughput. Inspect `logs/server.log` for a sustained interval
where prompt throughput has dropped to `0.0` while all requests are still
running; if the all-generation window is absent, rerun with `--output-len 8192`
and keep the other dimensions fixed.

## Nano DSA Knobs

`--nano-dsa` sets the long-context model defaults and appends the DSA server
flags. Each query token must compute its own chunk scores and top-k selection;
do not add or restore top-k sharing environment variables.

Use `--server-env KEY=VALUE` for additional server-side environment variables, and `--serve-extra-args` or `--benchmark-extra-args` for one-off CLI flags.

## Outputs

Default output root:

```text
outputs/fixed_io_benchmark/<job-name>_<timestamp>/
```

Important files:

- `results/fixed_io_result.json`
- `logs/server.log`
- `logs/<job-name>_<jobid>.log`
- `launchers/<job-name>_<timestamp>.sbatch`

Useful summary:

```bash
jq '{completed,failed,verified,duration_s,request_throughput,prompt_token_throughput,completion_token_throughput,total_token_throughput,input_len,output_len,num_prompts}' \
  "$RUN_DIR/results/fixed_io_result.json"
```

## Guardrails

- Use `$performance-benchmark-profile` instead when the user asks for Nsight Systems, CUDA/NVTX traces, `.nsys-rep`, or windowed profiling.
- Do not compare runs with different images, model paths, request shapes, TP size, or DSA env unless that difference is the point.
- The benchmark launcher has no `--nodelist` option at the time of writing. If a reserved node is required, add a small launcher option or generate with `--print-only` and submit an explicitly inspected sbatch.
