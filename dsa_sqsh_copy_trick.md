# Building a DSA vLLM sqsh with the copy trick

This note captures how we built a saved `.sqsh` that starts from the stock vLLM
v0.20.1 container but contains the local DSA/q_indexer Python changes.

The verified output image from this run was:

```bash
/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/users/mdabbah/deci/vllm_repos/vllm_v0.20.1/outputs/containers/vllm-openai_v0.20.1-dsa-qindexer-copy-20260520_075604.sqsh
```

## Why the copy trick

An editable install inside the container can fail or do the wrong thing for this
workflow:

1. A source snapshot without `.git` makes `setuptools-scm` unable to infer the
   vLLM version.
2. `VLLM_USE_PRECOMPILED=1 uv pip install -e .` may try to fetch a nightly
   precompiled wheel instead of staying pinned to the stock v0.20.1 container.
3. The stock container already has the correct compiled CUDA extensions, so we
   only need to replace the Python package files that contain our changes.

The copy trick simply copies the local `vllm/` source package over the installed
`/usr/local/lib/python3.12/site-packages/vllm/` package inside the stock
container, then saves that modified container. This preserves the stock compiled
extensions such as `vllm/_C.abi3.so`.

## Build steps

Run this from the repo root:

```bash
cd /lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/users/mdabbah/deci/vllm_repos/vllm_v0.20.1
```

Create a clean source snapshot on shared Lustre. Do not use `/tmp` for the
snapshot because the login-node `/tmp` is not visible from the allocated compute
node.

```bash
STAMP=20260520_075604
SRC="$PWD/outputs/sqsh_build_src_${STAMP}"
OUT="$PWD/outputs/containers/vllm-openai_v0.20.1-dsa-qindexer-copy-${STAMP}.sqsh"

mkdir -p "$SRC" "$PWD/outputs/containers"
rsync -a \
  --exclude .git \
  --exclude logs \
  --exclude outputs \
  --exclude .venv \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  ./ "$SRC"/
```

Build and save the container. This uses the `interactive` GPU partition and the
`nemotron_compress_dev` account.

```bash
srun \
  --account=nemotron_compress_dev \
  --partition=interactive \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --time=01:00:00 \
  --container-image=/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/containers/vllm/vllm-openai_v0.20.1.sqsh \
  --container-save="$OUT" \
  --container-mounts="$SRC:/workspace/vllm-src" \
  bash -lc '
    set -euo pipefail
    echo "BUILD_NODE=$(hostname)"

    python3 - <<PY
import site
import vllm
print("before_vllm_file", vllm.__file__)
print("site_packages", site.getsitepackages())
PY

    SITE=$(python3 - <<PY
import site
print(next(p for p in site.getsitepackages() if p.endswith("site-packages")))
PY
)

    cp -a /workspace/vllm-src/vllm "$SITE"/

    mkdir -p /opt/vllm-dsa-build-info
    cp /workspace/vllm-src/vllm/model_executor/models/nemotron_h.py \
      /opt/vllm-dsa-build-info/nemotron_h.py
    printf "%s\n" \
      "base=/lustre/fs1/portfolios/coreai/projects/coreai_nvfm_llm/containers/vllm/vllm-openai_v0.20.1.sqsh" \
      "source_snapshot=/workspace/vllm-src" \
      "method=copy patched vllm package over stock site-packages, preserving stock compiled extensions" \
      > /opt/vllm-dsa-build-info/README.txt

    python3 - <<PY
import inspect
import vllm
from vllm.model_executor.models import nemotron_h
path = inspect.getfile(nemotron_h)
text = open(path, encoding="utf-8").read()
print("after_vllm_file", vllm.__file__)
print("nemotron_h", path)
print("has_dsa", "NemotronHDSASelectiveAttention" in text)
print("has_flash_guard", "VLLM_NEMOTRON_H_DSA_USE_FLASH_TOPK" in text)
print("has_vllm_C", end=" ")
try:
    import vllm._C
    print(True, vllm._C.__file__)
except Exception as exc:
    print(False, repr(exc))
PY
  '
```

Clean up the temporary source snapshot after the saved image is verified:

```bash
rm -rf "$SRC"
```

## Smoke test the saved image

Launch the saved image and verify that the patched Python file and stock CUDA
extension are both importable:

```bash
srun \
  --account=nemotron_compress_dev \
  --partition=interactive \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --time=00:15:00 \
  --container-image="$OUT" \
  bash -lc '
    set -euo pipefail
    python3 - <<PY
import inspect
import torch
import vllm
from vllm.model_executor.models import nemotron_h
path = inspect.getfile(nemotron_h)
text = open(path, encoding="utf-8").read()
print("node_import_ok")
print("vllm", vllm.__file__)
print("nemotron_h", path)
print("has_dsa", "NemotronHDSASelectiveAttention" in text)
print("has_flash_guard", "VLLM_NEMOTRON_H_DSA_USE_FLASH_TOPK" in text)
print("cuda_available", torch.cuda.is_available())
import vllm._C
print("vllm_C", vllm._C.__file__)
PY
  '
```

To confirm the image contains the exact current file, compare hashes:

```bash
sha256sum vllm/model_executor/models/nemotron_h.py

srun \
  --account=nemotron_compress_dev \
  --partition=interactive \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node=1 \
  --time=00:15:00 \
  --container-image="$OUT" \
  bash -lc '
    sha256sum /usr/local/lib/python3.12/site-packages/vllm/model_executor/models/nemotron_h.py
  '
```

For the verified image, both hashes were:

```text
47393b5287313b864051012cd21830395d7b891b65e0bdeeb497d2fbe68ff624
```

## Caveats

This is intentionally a Python-package overlay. It is appropriate for changes
under `vllm/` that do not require rebuilding C++/CUDA extensions. If the change
touches `csrc/`, generated kernels, compiled extensions, or packaging metadata
that affects native code, rebuild properly instead of using this trick.

The saved image includes whatever `vllm/` Python files were present in the source
snapshot. Check `git status` before building so the image does not accidentally
capture unrelated local edits.
