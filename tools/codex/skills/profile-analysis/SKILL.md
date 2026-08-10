---
name: profile-analysis
description: Analyze Nsight Systems profile artifacts for the current code state in a structured, codebase-agnostic way. Use when asked to "analyze the profile", "analyze the current profile", inspect `.nsys-rep` or `.sqlite` profiles, summarize GPU idle gaps, CUDA runtime overhead, synchronization, memcpy sizes/directions, NVTX ranges, kernel timing, or generally identify inefficient patterns in one performance profile. Compare multiple profiles only when the user explicitly asks for comparison.
---

# Profile Analysis

## Core Workflow

Use this skill to analyze Nsight Systems CUDA/NVTX profiles without assuming anything about the profiled application or source code. The default unit of analysis is one current-code profile, or several capture windows from the same current-code run. Do not introduce a comparison framing unless the user explicitly asks to compare separate profiles.

1. Accept a current profile path, a run directory, or a set of capture-window paths from the same current-code run.
2. Prefer SQLite exports. If the user gives `.nsys-rep`, use the sibling `.sqlite` when present. If it is missing, export it with the same `nsys` version used for capture when available:

```bash
nsys export --type sqlite --force-overwrite=true --output profile.sqlite profile.nsys-rep
```

3. Run the bundled analyzer on the current-code SQLite profile(s). For one profile:

```bash
python tools/codex/skills/profile-analysis/scripts/analyze_nsys_sqlite.py \
  /path/to/current-profile.sqlite
```

For multiple windows from the same current-code run, pass each populated window:

```bash
python tools/codex/skills/profile-analysis/scripts/analyze_nsys_sqlite.py \
  /path/to/window1.sqlite /path/to/window2.sqlite /path/to/window3.sqlite
```

The script uses only Python stdlib. If a repo instruction requires `uv` or `.venv/bin/python`, follow that repo instruction when invoking the script.

## Response Shape

Always answer with a concise TLDR first, then supporting detail.

Use this structure:

```text
TLDR
I analyzed the current-code profile: <short description from filename or path>.

Main results:
1. <largest generic issue, e.g. long all-GPU idle gaps between kernels>
2. <sync issue, e.g. many tiny device-to-host copies and synchronize calls>
3. <memcpy issue, e.g. large device-to-device movement or many small transfers>
4. <kernel/runtime concentration or other obvious inefficiency>

What Was Profiled
<paths, SQLite/export status, visible NVTX labels if useful>

High-Level Numbers
<table from analyzer: span, GPU busy/util, largest gap, tiny copies, sync time>

Issue 1: ...
<evidence and interpretation>

Issue 2: ...
<evidence and interpretation>

Detailed Tables
<top kernels, runtime APIs, memcpy distribution, top NVTX ranges>
```

If the user explicitly asks to compare profiles, label them by user-provided name, timestamp, image, commit, or path. Do not invent comparison labels that the user did not provide.

## What To Look For

- **GPU idle gaps:** Large intervals between kernels after merging all GPU kernel intervals. Explain what kernels/ranges border the gap, and what CUDA runtime calls are visible inside the gap.
- **CPU-side overhead:** Long CUDA runtime calls, frequent launch calls, module loads, allocations, profiler stop overhead, or long gaps with little visible CUDA work.
- **Synchronization:** `cudaStreamSynchronize`, `cudaDeviceSynchronize`, `cudaEventSynchronize`, synchronization table entries, or runtime calls following tiny device-to-host copies.
- **Memcpy behavior:** Direction, total bytes, count, duration, and suspicious size distributions. Tiny device-to-host copies are especially important because they often imply scalar reads and CPU-GPU synchronization.
- **Kernel concentration:** Kernels that dominate total GPU time or grow across multiple windows.
- **NVTX structure:** Use NVTX labels only as observed profile labels. Do not infer code architecture from names unless the profile/logs directly support it.

## Guardrails

- Do not hard-code project-specific kernel names, model names, phases, or source paths into the analysis.
- Do not assume a specific application phase or workflow unless the profile labels, filenames, logs, or user state that explicitly.
- It is fine to quote kernel or NVTX names that appear in the profile data, but frame them as observed labels.
- Distinguish CUDA runtime time from invisible CPU work. A large kernel gap with only tiny CUDA API calls inside is host-side time not explained by GPU work in the trace.
- Distinguish copy direction: host-to-device, device-to-host, and device-to-device have different meanings.
- Keep the default report focused on the current profile. Avoid speedup/regression language unless the user supplied another profile and asked for comparison.
