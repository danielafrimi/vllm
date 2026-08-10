# Nemotron-H DSA backend selection

Use this guide for any Nemotron-H DSA server, benchmark, evaluation, or
correctness run. The backend choice is independent of the workload driver.

## The two selection layers

Nemotron-H DSA has two separate choices:

1. `VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS` chooses the complete attention
   implementation.
2. `VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS` chooses a component bundle **only
   when refactored attention is active**.

The overall default is Moonshot/vanilla attention. If the attention variable is
unset, the code behaves as if it were set to `moonshot`.

Refactored attention has a second default: when no provider is specified, it
uses the efficient bundle on CUDA and the PyTorch bundle when CUDA is not
available. Do not rely on this conditional default in reproducible runs; use a
combined attention alias.

| Desired path | Canonical setting | Intended use |
| --- | --- | --- |
| Moonshot/vanilla (overall default) | `VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS=moonshot` | Original implementation from the Moonshot branch |
| Refactored + efficient | `VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS=refactored-efficient` | Performance runs on CUDA |
| Refactored + PyTorch | `VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS=refactored-pytorch` | Reference, debugging, and parity work |

`vanilla` is an alias for `moonshot`. `pytorch` and `torch` are provider
aliases; `efficient`, `cuda`, and `triton` are efficient-provider aliases.
`legacy` selects a separate legacy attention class and is not another name for
the Moonshot default.

## Clean backend activation

Start by removing inherited selections. This matters when several runs share a
shell or a generated Slurm environment:

```bash
unset VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS
unset VLLM_NEMOTRON_H_DSA_ATTENTION_MODULE
unset VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS
unset VLLM_NEMOTRON_H_DSA_PROVIDER_MODULE
```

Then select exactly one path.

### Moonshot/vanilla attention (default)

Explicit, reproducible selection:

```bash
export VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS=moonshot
```

Leaving the variable unset has the same effect. The provider variable has no
effect on this implementation. Do not describe a run as using the efficient
bundle merely because `VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS=efficient` is set;
the attention class must be refactored too.

The Moonshot forward is compiler-disabled. Existing repository launchers
therefore normally pair Moonshot/vanilla with eager execution. Use
`--enforce-eager` unless a specific experiment has established another safe
configuration.

### Refactored attention with the efficient bundle

```bash
export VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS=refactored-efficient
```

The combined alias also sets the provider to `efficient`. Setting both
variables is allowed but redundant:

```bash
export VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS=refactored
export VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS=efficient
```

For the optimized no-Q-sharing path, enable the GPU implementations used by
the efficient bundle:

```bash
export VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16
export VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=1
export VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA=1
export VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ=1
export VLLM_NEMOTRON_H_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA=1
export VLLM_NEMOTRON_H_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA=1
export VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE=4096
export VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING=1
export VLLM_NEMOTRON_H_DSA_USE_TRITON_BATCHED_SUMMARIES=1
```

Do not enable the Moonshot shared-top-k/Q-share variables for this path. The
checkpoint normally supplies the selection budget. If overriding
`VLLM_NEMOTRON_H_DSA_CHUNK_TOP_K`, remember that it is a number of chunks, not
a number of tokens.

Refactored attention supports PIECEWISE CUDA graphs. Do not pass
`--enforce-eager`; pass this server argument instead:

```bash
--compilation-config '{"mode":3,"cudagraph_mode":"PIECEWISE"}'
```

PIECEWISE is intentional. `vllm::nemotron_h_dsa_attention_with_output` is an
opaque graph boundary around dynamic sparse work, while compatible surrounding
operations are compiled and captured. Do not substitute `FULL` without a new
validation.

### Refactored attention with the PyTorch bundle

```bash
export VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS=refactored-pytorch
```

Equivalent explicit form:

```bash
export VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS=refactored
export VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS=pytorch
```

This is the readable reference implementation. Use it for parity and
diagnostics, not as the performance baseline. The refactored graph boundary is
still available, so PIECEWISE may compile/capture the surrounding model, but
the PyTorch DSA body itself remains outside the captured regions. Use eager mode
when isolating provider behavior; use PIECEWISE when comparing end-to-end
behavior under the same outer graph configuration as the efficient bundle.

## Settings that are orthogonal to backend selection

- `--attention-backend FLASH_ATTN` does not select Moonshot versus refactored
  DSA. It selects the dense/page-table attention kernel used within a path.
- Model checkpoint, tensor parallelism, batching limits, and selection budget
  do not select the DSA backend.
- Q-share/shared-top-k environment variables configure Moonshot features; they
  do not turn refactored attention into the efficient bundle.
- Launcher variables such as `DSA_ATTENTION_CLASS` or `DSA_PROVIDER_CLASS` are
  wrapper-specific. Confirm that a launcher exports the canonical `VLLM_*`
  variables above. When in doubt, inspect its generated Slurm script.

## Slurm and sandbox execution

Run Slurm operations on the host, outside an agent command sandbox. This
includes `sbatch`, `srun`, `squeue`, and `sacct`, plus Pyxis/Enroot container
launches and cluster-visible filesystem checks. In an agent environment, use
the supported escalation/approval mechanism rather than treating a sandbox
failure as a Slurm or code failure.

Before submission, make the generated job environment self-contained:

- export the backend choice into the job, not only the submitting shell;
- record the Git commit and dirty state;
- record the image path and verify the image contains that source state;
- record the final server arguments, especially `--enforce-eager` versus the
  compilation configuration;
- keep secrets out of printed commands and logs.

Use repository launchers when they cover the workload, but inspect the emitted
script or metadata. A launcher name is not evidence that the requested backend
reached the worker process.

## Verification

### Record the resolved configuration

At minimum, capture these values in run metadata:

```bash
env | sort | rg '^VLLM_NEMOTRON_H_DSA_(ATTENTION|PROVIDER|USE_TRITON|USE_.*PAGE_TABLE|PATH_DEBUG)'
```

Also verify the loaded modules come from the intended checkout or image, not a
stale site-packages copy:

```bash
.venv/bin/python - <<'PY'
import inspect
from vllm.model_executor.models import (
    nemotron_h,
    nemotron_h_dsa_attention_refactored,
    nemotron_h_chunked_dsa_components_efficient,
    nemotron_h_chunked_dsa_components_pytorch,
)

for module in (
    nemotron_h,
    nemotron_h_dsa_attention_refactored,
    nemotron_h_chunked_dsa_components_efficient,
    nemotron_h_chunked_dsa_components_pytorch,
):
    print(module.__name__, inspect.getfile(module))
PY
```

All Python commands must follow the repository rule: use `uv` and
`.venv/bin/python`, never system `python3` or bare `pip`.

### Efficient-bundle markers

Efficient path markers are opt-in:

```bash
export VLLM_NEMOTRON_H_DSA_PATH_DEBUG_PRINT_LIMIT=10
```

Search the worker/server log:

```bash
rg 'DSA_PATH_MARKER|Capturing CUDA graphs|cudagraph_mode|splitting_ops' server.log
```

Expected efficient markers include:

- `marker=config` with `triton_scoring_provider=gpu_tile_plan` and
  `use_triton_scoring=True`;
- `marker=triton_batched_score_tile_plan`;
- `marker=triton_batched_scoring`;
- `marker=triton_batched_unified_page_table`;
- `marker=dense_prefill_page_table_bucket` for an exercised dense bucket;
- `marker=sparse_prefill_page_table_bucket` for an exercised sparse prefill;
- `marker=sparse_decode` for exercised sparse decode.

Markers prove only paths exercised by the workload. A missing dense marker is
expected when no request enters a dense bucket. For mixed-path coverage, run
the repository marker checker against a log from a workload that deliberately
contains dense prefill, sparse prefill, and sparse decode.

The PyTorch bundle does not emit the efficient `DSA_PATH_MARKER` set. Verify it
through the explicit combined alias, recorded job environment, loaded source,
and parity behavior. Absence of efficient markers alone is not proof that the
PyTorch bundle ran.

Moonshot/vanilla logs emit bounded messages beginning with `good news! vLLM
Nemotron-H DSA ... ran` when their selective or chunked paths execute. Treat
those as execution hints, and pair them with the recorded attention-class
selection.

### CUDA graph evidence

For a refactored PIECEWISE run, require all of the following in startup logs:

- `enforce_eager=False`;
- `cudagraph_mode=<CUDAGraphMode.PIECEWISE: 1>`;
- `vllm::nemotron_h_dsa_attention_with_output` in `splitting_ops`;
- `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)`;
- a final CUDA graph pool memory line.

These lines indicate a misconfigured graph run:

- `Enforce eager set, disabling torch.compile and CUDAGraphs`;
- `Cudagraph is disabled under eager mode`;
- `Skipping CUDA graph capture`;
- `cudagraph_mode=<CUDAGraphMode.NONE: 0>`.

## Minimal decision rule

- No backend variable: Moonshot/vanilla, the overall default.
- Performance on CUDA: `refactored-efficient` plus PIECEWISE graphs and the
  optimized GPU-path variables.
- Reference/debugging: `refactored-pytorch`, with eager or PIECEWISE chosen to
  match the purpose of the comparison.
- Never infer the backend from `--attention-backend`, the checkpoint name, or
  the launcher name; verify the canonical environment and worker logs.
