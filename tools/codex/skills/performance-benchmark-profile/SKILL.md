---
name: performance-benchmark-profile
description: Run the vLLM performance benchmark with windowed Nsight Systems profiling on Slurm/Pyxis for the current code state. Use when asked to profile the current vLLM/Nano DSA checkout, build or use the current-code SQSH image, capture `.nsys-rep`/SQLite CUDA/NVTX artifacts, adjust TP size, use short profile windows, reuse reserved nodes, or inspect profiling artifacts. Compare to another profile only when the user explicitly asks for comparison.
---

# Performance Benchmark Profile

## Core Workflow

Use this skill to run the fixed-IO performance benchmark under short Nsight Systems capture windows for exactly one code state: the code that exists in the current checkout/image. Do not introduce a comparison framing unless the user explicitly asks to compare profiles.

1. Work from the vLLM repo and read local instructions such as `AGENTS.md`.
2. Ensure the SQSH image contains the current checkout. If local code must be copied into an image, use `$vllm-sqsh-evals` first.
3. Run a single current-code profiled benchmark. Prefer a single-image/current-only launcher when the checkout provides one.
4. If the checkout only has an older two-case profile launcher, use it as a template or `--print-only` source and submit a one-case Slurm script for the current image only. Do not run a duplicate companion case just to satisfy a legacy launcher shape.
5. Use short runs first: few prompts, no warmups, one or more 1-second profile windows.
6. Prefer explicit current image, TP size, GPU count, model, profile window schedule, job name, output root, and nodelist when reusing a reserved node.
7. After completion, report the run directory, benchmark result JSON, `profile_windows.json`, and exact `.nsys-rep`/`.sqlite` paths for the current-code run.

## Common Launches

Short current-code profile, TP2:

```bash
CURRENT_IMAGE=/path/to/current-code.sqsh
TP_SIZE=2 \
EXPORT_SQLITE=1 \
OUT_ROOT="$PWD/outputs/fixed_io_nsys_profile_current" \
benchmarks/launch_fixed_io_nsys_profile.sh \
  --image "$CURRENT_IMAGE" \
  --gpus 2 \
  --num-prompts 1 \
  --num-warmups 0 \
  --profile-windows 2 \
  --profile-window-seconds 1 \
  --profile-gap-seconds 5 \
  --profile-initial-delay 20 \
  --partition batch \
  --time 03:00:00 \
  --job-name fixedio-nsys-current-tp2-short \
  --wait
```

If the checkout does not have `benchmarks/launch_fixed_io_nsys_profile.sh`, derive a one-case sbatch from the available profiling launcher:

1. Run the existing launcher with `--print-only`.
2. Edit the generated sbatch so it launches only the current image once.
3. Keep the same `vllm serve` Nsight wrapper, `profile_window_controller.py`, benchmark client, and SQLite export steps.
4. Submit that one-case sbatch and report only the current-code artifacts.

Reuse a reserved node:

```bash
TP_SIZE=2 \
EXPORT_SQLITE=1 \
benchmarks/launch_fixed_io_nsys_profile.sh ... \
  --gpus 2 \
  --nodelist pool0-0019 \
  --partition batch \
  --wait
```

TP4 uses `TP_SIZE=4` plus `--gpus 4`. `TP_SIZE` is an environment variable for this launcher, not a CLI flag.

Decode-oriented current-code profile, TP2:

Use this when the user wants a decoding profile rather than a chunked-prefill
profile. The request shape is ISL 100K / OSL 2K with 64 concurrent sequences.
The matrix launcher takes ISL/OSL and concurrency from environment variables.
Use a late enough initial delay that the captured windows land after prefill;
after the run, confirm in `logs/current/server.log` that the selected windows
correspond to `Avg prompt throughput: 0.0` and `Running: 64 reqs`.

```bash
CURRENT_IMAGE=/path/to/current-code.sqsh
TP_SIZE=2 \
INPUT_LEN=100000 \
OUTPUT_LEN=2048 \
NUM_PROMPTS=64 \
MAX_CONCURRENCY=64 \
MAX_NUM_SEQS=64 \
MAX_MODEL_LEN=131072 \
EXPORT_SQLITE=1 \
CURRENT_ONLY=1 \
OUT_ROOT="$PWD/outputs/fixed_io_nsys_profile_decode" \
benchmarks/launch_fixed_io_nsys_profile_matrix.sh \
  --current-image "$CURRENT_IMAGE" \
  --gpus 2 \
  --num-prompts 64 \
  --num-warmups 0 \
  --profile-windows 6 \
  --profile-window-seconds 1 \
  --profile-gap-seconds 15 \
  --profile-initial-delay 180 \
  --partition batch \
  --time 01:00:00 \
  --job-name fixedio-nsys-decode100k-2k-tp2 \
  --current-only \
  --wait
```

If the first profile window still overlaps prefill, rerun with a larger
`--profile-initial-delay`; do not increase OSL just to get a decode window unless
the user asks for a longer benchmark.

## Profile Windows

The launcher wraps `vllm serve` in:

```text
nsys profile --trace=cuda,nvtx --cuda-graph-trace=graph --capture-range=cudaProfilerApi --capture-range-end=repeat
```

`benchmarks/profile_window_controller.py` calls vLLM `/start_profile` and `/stop_profile`, so capture is off initially and only enabled during the requested windows.

The fixed-IO profile launcher uses graph-level CUDA graph tracing by default.
Keep this enabled so graph replay intervals are visible without collecting
node-level details. Use `--cuda-graph-trace=node` only for deliberate graph-node
debugging.

Key knobs:

- `--profile-windows N`
- `--profile-window-seconds S`
- `--profile-gap-seconds S`
- `--profile-initial-delay S`
- `NSYS_EXTRA_ARGS=...` for extra Nsight flags
- `EXPORT_SQLITE=1` to export `.sqlite` next to `.nsys-rep`

## Configuration Caution

At the time of writing, this profile launcher hard-codes several Nano DSA environment variables inside the generated container script, including batched summaries off:

```text
VLLM_NEMOTRON_H_DSA_USE_TRITON_BATCHED_SUMMARIES=0
VLLM_NEMOTRON_H_DSA_USE_SUMMARY_CACHE=0
```

Do not assume outer environment variables override these. For a summary-cache or batched-summary profile, first inspect with `--print-only` and either edit the generated sbatch deliberately or update the launcher to expose server env overrides using the benchmark launcher's `--server-env` pattern. Report that change clearly.

## Outputs

Default output root:

```text
outputs/fixed_io_nsys_profile_current/<job-name>_<timestamp>/
```

Important files:

- `results/fixed_io_result.json`
- `results/profile_windows.json`
- `profiles/*.nsys-rep`
- `profiles/*.sqlite`
- `logs/server.log`
- `logs/profile_windows.log`
- `logs/nsys_export.log` when SQLite export is enabled

Useful checks:

```bash
jq '{ok,windows,window_seconds,gap_seconds,initial_delay_seconds,events:[.events[] | {event,window,status,ok}]}' \
  "$RUN_DIR/results/profile_windows.json"

find "$RUN_DIR/profiles" -maxdepth 1 -type f \( -name '*.nsys-rep' -o -name '*.sqlite' \) | sort

jq '{completed,failed,verified,duration_s,request_throughput,prompt_token_throughput,total_token_throughput}' \
  "$RUN_DIR/results/fixed_io_result.json"
```

## Guardrails

- Use `$performance-benchmark` instead when the user wants numbers only and not Nsight profile files.
- Keep windows short. Profiling the full long-context run can produce files too large to inspect.
- If `.nsys-rep` exists but SQLite is missing, run `nsys export --type sqlite --force-overwrite=true --output <out.sqlite> <report.nsys-rep>` using the same Nsight CLI path the launcher printed.
- When explaining profile artifacts, distinguish profile-window timing from benchmark timing. A successful benchmark can still have failed profile-window toggles; inspect `profile_windows.json`.
- Do not report speedups, regressions, or cross-profile comparisons unless the user explicitly supplied another profile or asked for a comparison.
