---
name: quick-regression-test
description: Orchestrate a quick report-only vLLM current-code regression/correctness run. Use when the user asks to "Launch the Quick Correctness Test", "Launch the Quick Regression Test", "run a quick regression", "run quick correctness", or similar free-language requests that should build a current-code SQSH image, run the generation disagreement correctness smoke test, run a short Nsight profiled benchmark, analyze profile statistics such as elapsed time, kernels, CUDA gaps, synchronization, memcpy behavior, and summarize artifacts without pass/fail gating.
---

# Quick Regression Test

## Purpose

Run the fastest useful report-only check for the current vLLM checkout:

1. Build or verify one SQSH image that contains the current code.
2. Run the fast generation disagreement correctness smoke test from that image.
3. Run a short current-code Nsight profiled benchmark from that same image.
4. Analyze and report correctness statistics, benchmark timing, kernel timing, GPU gaps, synchronization, memcpy, and artifact paths.

This skill is not a pass/fail gate. Do not report "passed" or "failed" based on disagreement or profile metrics unless an underlying command/job itself failed.

## Required Skills

Use these existing skills as subroutines:

- `$vllm-sqsh-evals` for current-code SQSH build and smoke validation.
- `$vllm-generation-disagreement` for the fast correctness/disagreement run.
- `$performance-benchmark-profile` for the short profiled benchmark.
- `$profile-analysis` for Nsight SQLite analysis after the profile run completes.

Open each skill's `SKILL.md` when executing that phase, because the launchers, defaults, and guardrails may change.

## Default Run Shape

Treat "Quick Correctness Test" and "Quick Regression Test" as aliases for the full workflow in this skill: build/verify SQSH, run generation disagreement, run the profiled benchmark, analyze the profile, and summarize both branches. Run only one branch when the user explicitly asks for only correctness or only profiling.

Use user-provided settings first. If settings are missing, prefer the defaults and recent launch context from the underlying skills. If there is still no explicit choice, use this quick shape:

- Correctness: `$vllm-generation-disagreement` sparse-coverage default workload, 50 fixed prompt specs, 4096 prompt tokens, 200 output tokens, greedy decoding, default sparse 4096-token/200-output-token baseline, and required DSA path-marker checking.
- Profile: `$performance-benchmark-profile` short current-code profile, TP2/GPU2 when compatible with the model, 1 prompt, 0 warmups, 2 profile windows, 1 second per window, 5 second gaps, 20 second initial delay, SQLite export enabled.
- Slurm: use the user's requested partition/nodelist/time when supplied; otherwise use the launcher defaults from the underlying skill, commonly `batch` and a short walltime such as 3 hours.
- Model path: use the user-specified model path, a recent matching generation-disagreement/benchmark launcher model path, or the default Nano/Nemotron-H DSA model used by the underlying skills. Do not invent a different model path.

## Workflow

1. Read repo instructions such as `AGENTS.md`, then capture:
   - repo path
   - branch and commit
   - dirty state summary
   - touched areas that affect whether copy-overlay SQSH is valid
   - model path, TP size, GPU count, partition, nodelist, or other user-specified knobs

2. Build or verify exactly one current-code SQSH image.
   - Use `$vllm-sqsh-evals`.
   - Prefer an existing verified image only if it demonstrably contains the current commit and dirty source state.
   - If the worktree is dirty, build fresh unless a matching dirty-source marker/hash is already verified.
   - Smoke-test the image before launching the two branches.
   - Record image path, base image, commit, dirty marker/hash, build start/end time, elapsed time, and smoke-test evidence.

3. After the SQSH image is ready, launch the correctness and profiling branches in parallel when possible.
   - If sub-agent tools are available and the current user request explicitly authorizes sub-agents, delegation, or parallel agent work, spawn two worker agents after the image is built.
   - If sub-agent use is not authorized or unavailable, launch both Slurm workflows from the main agent without waiting for one to finish before submitting the other, unless resources or user constraints require serialization.
   - Never let either branch rebuild the SQSH image. Pass the verified image path to both branches.

4. Correctness branch:
   - Use `$vllm-generation-disagreement`.
   - Generate a fresh current-code JSONL artifact from the verified SQSH image
     with `scripts/generation_disagreement/run_sparse_agreement.sbatch`.
   - Keep the sparse-coverage defaults unless the user supplied different
     settings: 4096 prompt tokens, 200 output tokens, max model len 8192,
     max num seqs 4, chunked prefill enabled, max num batched tokens 1024,
     dense prefill threshold 2048, and DSA path debug enabled.
   - Compare against the default sparse 4096-token/200-output-token baseline
     unless the user supplied a different reference.
   - Check `path_marker_check.txt` or rerun
     `scripts/generation_disagreement/check_dsa_path_markers.py` on
     `current.log`. Treat missing required markers as a correctness-branch
     execution failure. Required markers are `config`,
     `dense_prefill_page_table_bucket`, `sparse_prefill_page_table_bucket`, and
     `sparse_decode`.
   - Do not use fail-under or fail-over thresholds.
   - Report exact matches, agreement-token distribution, early-divergence count, finish-reason mismatches, output length ranges, worst prompts, artifact paths, Slurm job ID, elapsed time, and whether all required DSA path markers were seen.

5. Profiling branch:
   - Use `$performance-benchmark-profile`.
   - Run a short current-code Nsight profile first: few prompts, no warmups, short capture windows, and SQLite export when possible.
   - After the profile run completes, use `$profile-analysis` on the produced SQLite files.
   - Report benchmark duration, request/token throughput, profile window status, `.nsys-rep` and `.sqlite` paths, GPU busy/utilization, largest all-GPU idle gaps, top kernels, CUDA runtime overhead, synchronization, memcpy distribution, and any notable NVTX ranges.

6. Merge the two branch results in the main conversation.
   - Lead with a concise TLDR.
   - Include a timeline with SQSH build time, correctness elapsed time, profile benchmark elapsed time, and total wall-clock time.
   - Include the verified SQSH image path and source identity.
   - Include separate "Correctness Statistics" and "Profile Statistics" sections.
   - Include all important artifact paths and job IDs.
   - State "report-only; no pass/fail threshold was applied."
   - If a branch fails, report the failure and available artifacts from the other branch instead of hiding partial results.

## Sub-Agent Prompts

Use prompts like these after the SQSH image is built. Fill in concrete paths and knobs.

Correctness worker:

```text
Use $vllm-generation-disagreement in <repo>. Run the sparse-coverage current-code correctness/disagreement smoke test using this already-verified SQSH image: <image>. Use scripts/generation_disagreement/run_sparse_agreement.sbatch, do not rebuild the image, and do not modify repo runtime source. Compare against the default sparse 4096-token/200-output-token baseline unless a different reference is supplied. This is report-only: do not apply fail-under/fail-over thresholds. Check the DSA path markers and return whether all required markers were seen, plus the Slurm job ID, run directory, JSONL artifacts, compare output, exact-match count, agreement-token distribution, early-divergence count, finish-reason mismatches, output length ranges, worst prompts, and elapsed time.
```

Profiling worker:

```text
Use $performance-benchmark-profile and then $profile-analysis in <repo>. Run a short current-code Nsight profiled benchmark using this already-verified SQSH image: <image>. Do not rebuild the image and do not modify runtime code. Prefer SQLite export. Analyze the produced SQLite profile(s). Return the Slurm job ID, run directory, benchmark result JSON, profile window status, .nsys-rep/.sqlite paths, benchmark duration and throughput, GPU busy/utilization, largest all-GPU idle gaps, top kernels, CUDA runtime overhead, synchronization, memcpy distribution, notable NVTX ranges, and elapsed time.
```

Tell workers they are not alone in the codebase, must not revert others' changes, and must keep generated artifacts under the existing `outputs/` locations used by the underlying skills.

## Reporting Template

```text
TLDR
<one or two sentences with the current-code image, both branch statuses, and the biggest observation>

Timeline
- SQSH build/verify: <elapsed>
- Correctness branch: <elapsed>
- Profiling branch: <elapsed>
- Total wall-clock: <elapsed>

Source Image
- Repo: <path>
- Commit/dirty state: <identity>
- SQSH: <image>
- Smoke evidence: <brief>

Correctness Statistics
- Exact matches: <n>/<total>
- Agreement tokens: min/p10/p25/median/mean/p75/p90/max
- Early divergence: <n> below threshold
- Finish reason mismatches: <n>
- Output length range: <range>
- DSA path markers: all required markers seen / missing <markers>
- Worst prompts: <brief>
- Artifacts: <paths>

Profile Statistics
- Benchmark result: completed/failed/verified, duration, throughput
- Profile windows: <status>
- GPU busy/utilization and largest gaps: <numbers>
- Top kernels: <brief table or list>
- CUDA runtime/sync/memcpy: <brief>
- Artifacts: <paths>

Report-only; no pass/fail threshold was applied.
```

## Guardrails

- Do not open or propose upstream PRs from this skill.
- Do not modify repo runtime source unless the user explicitly asks for fixes.
  The correctness launcher may transiently instrument the installed container
  copy of the DSA attention module so marker coverage can be checked.
- Do not compare against another code state unless the user explicitly asks for a comparison.
- Do not print secrets or raw logs that may contain secrets. Redact tokens and credentials before quoting logs.
- Follow repo Python rules: use `uv` and `.venv/bin/python`, not system `python3` or bare `pip`.
- Slurm/Pyxis commands may require escalation because they contact cluster services; request approval when sandboxing blocks required cluster or network access.
