---
name: vllm-sqsh-evals
description: Build and validate current-code vLLM SquashFS/Enroot/Pyxis `.sqsh` images and launch Deci/Nemo evaluator runs from them. Use when the user asks to put local vLLM changes into a squash image, use repo code inside a container, build a current-code `.sqsh`, run DSA/Nemotron-H/RULER/Deci evals, launch or monitor NEL/Slurm eval jobs, or debug whether a job is actually using the intended vLLM checkout.
---

# vLLM SQSH Evals

## Core Rule

Do not assume a Slurm/Pyxis job is using the local checkout. Verify or build a `.sqsh` that contains the current code, then smoke-test that image before launching evals.

## Workflow

1. Identify the vLLM repo, branch, commit, dirty state, base image, and target benchmark. Read local repo instructions such as `AGENTS.md` first.
2. Decide whether the copy-overlay SQSH build is valid:
   - Use it for Python/package changes under `vllm/` when the base image already has compatible compiled extensions.
   - Do not use it for C/C++/CUDA/ABI changes; build or install vLLM inside the image instead, or ask for the intended build path.
3. Build or locate a current-code `.sqsh`.
   - Prefer an existing current-commit image under `outputs/containers/`.
   - Otherwise run `scripts/build_vllm_sqsh_copy_overlay.sh` from this skill. Read the script before changing it.
   - Use shared Lustre/Scratch for source snapshots. Do not stage the snapshot only in login-node `/tmp`; compute nodes may not see it.
4. Smoke-test the saved image:
   - Import `vllm`, the touched module, and `vllm._C`.
   - Compare a hash or explicit marker from the local source to the file inside the image.
   - Record the base image, source commit, and output image path.
5. Launch evals from a small CPU wrapper that calls `deci-evals/.venv/bin/evaluate`; the wrapper should submit the actual GPU job.
6. Monitor with `squeue`, `sacct`, result artifacts, and redacted log scans. Avoid dumping raw `run.sub` or Slurm logs when they may contain secrets from `set -x`.

## Common Commands

Build a Python-overlay SQSH:

```bash
SKILL=tools/codex/skills/vllm-sqsh-evals
BASE_IMAGE=/path/to/base-vllm.sqsh \
  "$SKILL/scripts/build_vllm_sqsh_copy_overlay.sh" \
  --repo /path/to/vllm
```

Launch the Nano DSA RULER eval when the repo already has `scripts/launch_nano_dsa_eval.sh`:

```bash
./scripts/launch_nano_dsa_eval.sh ruler-128k-completions --config-only
./scripts/launch_nano_dsa_eval.sh ruler-128k-completions

./scripts/launch_nano_dsa_eval.sh ruler-1m-completions --config-only
./scripts/launch_nano_dsa_eval.sh ruler-1m-completions
```

If the repo lacks that launcher, read `references/deci-nano-dsa-ruler.md` and create a local wrapper using the same pattern.

## Secrets

Use a secrets file when available, but never print secret values. Expected Deci eval variables usually include:

- `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`
- `JUDGE_API_KEY`
- `DECI_INFERENCE_HUB_KEY`
- `DECI_BUILD_NVDEV_KEY`

Before printing logs, redact `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `JUDGE_API_KEY`, `DECI_`, `API_KEY`, `Authorization`, `Bearer`, `hf_`, `nvapi-`, and `sk-`.

## References

- Read `references/sqsh-copy-overlay.md` before building or debugging a SQSH image.
- Read `references/deci-nano-dsa-ruler.md` before launching Deci/NEL RULER evals.
