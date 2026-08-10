# SQSH Copy-Overlay Build

Use this reference when the user needs a `.sqsh` image that contains the local vLLM Python code.

## Preconditions

- The base `.sqsh` already has compatible vLLM compiled extensions for the target code.
- The local changes are Python/package-level changes, usually under `vllm/`.
- The source snapshot and output image live on a filesystem visible to the Slurm compute node.

If the change touches compiled code, CUDA kernels that must be rebuilt, ABI contracts, or dependency versions, stop and use a real build/install flow instead of the copy-overlay trick.

## Build Pattern

The bundled script `scripts/build_vllm_sqsh_copy_overlay.sh`:

1. Creates a source snapshot under `<repo>/outputs/sqsh_build_src_<stamp>`.
2. Excludes `.git`, `outputs`, `logs`, `.venv`, caches, and bytecode.
3. Starts the base image with `srun --container-image`.
4. Copies `/workspace/vllm-src/vllm` over the installed `site-packages/vllm`.
5. Saves the modified image with `--container-save`.
6. Writes `/opt/vllm-current-overlay-build-info/README.txt` inside the image.
7. Smoke-tests imports for `vllm`, `vllm._C`, and an optional module.

## Validation Checklist

- `vllm.__file__` points inside the container site-packages.
- The touched module path is inside site-packages and contains the expected marker or hash.
- `import vllm._C` succeeds.
- The output `.sqsh` path includes the source commit and timestamp.
- A short record of base image, commit, and smoke-test output is saved in the conversation or local run notes.

## Pitfalls

- A source snapshot in `/tmp` may not be visible inside a Slurm allocation.
- `uv pip install -e .` inside the base image can accidentally resolve a different wheel or fail version inference without `.git`.
- A generated `.sqsh` proves nothing until smoke-tested.
- If eval logs say vLLM env vars are unknown, verify whether that is just vLLM env validation or whether the code path actually reads them elsewhere.
