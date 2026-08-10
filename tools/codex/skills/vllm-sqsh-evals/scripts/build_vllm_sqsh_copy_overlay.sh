#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Build a vLLM .sqsh by copying the local vllm/ Python package over a base image.

Usage:
  build_vllm_sqsh_copy_overlay.sh --repo /path/to/vllm --base-image /path/base.sqsh [options]

Options:
  --repo PATH          vLLM repo to snapshot. Default: current directory.
  --base-image PATH    Base .sqsh image. Or set BASE_IMAGE.
  --out PATH           Output .sqsh. Default: <repo>/outputs/containers/vllm-current-overlay-<commit>-<stamp>.sqsh
  --account NAME       Slurm account. Default: nemotron_compress_dev
  --partition NAME     Slurm partition. Default: interactive
  --time HH:MM:SS      Slurm walltime. Default: 01:00:00
  --gpus N             GPUs for the build allocation. Default: 1
  --check-module MOD   Module to import in the smoke test. Default: vllm.model_executor.models.nemotron_h
  --keep-src           Keep the source snapshot after successful build.
  -h, --help           Show this help.

This is for Python/package overlays only. Do not use it for compiled-code or ABI changes.
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

REPO_DIR="$PWD"
BASE_IMAGE="${BASE_IMAGE:-}"
OUT="${OUT:-}"
ACCOUNT="${ACCOUNT:-nemotron_compress_dev}"
PARTITION="${PARTITION:-interactive}"
WALLTIME="${WALLTIME:-01:00:00}"
GPUS="${GPUS:-1}"
CHECK_MODULE="${CHECK_MODULE:-vllm.model_executor.models.nemotron_h}"
KEEP_SRC=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      [[ $# -ge 2 ]] || die "--repo requires a value"
      REPO_DIR="$2"
      shift 2
      ;;
    --base-image)
      [[ $# -ge 2 ]] || die "--base-image requires a value"
      BASE_IMAGE="$2"
      shift 2
      ;;
    --out)
      [[ $# -ge 2 ]] || die "--out requires a value"
      OUT="$2"
      shift 2
      ;;
    --account)
      [[ $# -ge 2 ]] || die "--account requires a value"
      ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      [[ $# -ge 2 ]] || die "--partition requires a value"
      PARTITION="$2"
      shift 2
      ;;
    --time)
      [[ $# -ge 2 ]] || die "--time requires a value"
      WALLTIME="$2"
      shift 2
      ;;
    --gpus)
      [[ $# -ge 2 ]] || die "--gpus requires a value"
      GPUS="$2"
      shift 2
      ;;
    --check-module)
      [[ $# -ge 2 ]] || die "--check-module requires a value"
      CHECK_MODULE="$2"
      shift 2
      ;;
    --keep-src)
      KEEP_SRC=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

REPO_DIR="$(cd -- "$REPO_DIR" && pwd)"
[[ -d "$REPO_DIR/vllm" ]] || die "repo does not contain vllm/: $REPO_DIR"
[[ -n "$BASE_IMAGE" ]] || die "pass --base-image PATH or set BASE_IMAGE"
[[ -f "$BASE_IMAGE" ]] || die "base image not found: $BASE_IMAGE"

cd "$REPO_DIR"
COMMIT="$(git rev-parse --short=10 HEAD 2>/dev/null || printf unknown)"
STAMP="$(date +%Y%m%d_%H%M%S)"
SRC="$REPO_DIR/outputs/sqsh_build_src_${STAMP}"
if [[ -z "$OUT" ]]; then
  OUT="$REPO_DIR/outputs/containers/vllm-current-overlay-${COMMIT}-${STAMP}.sqsh"
fi
mkdir -p "$SRC" "$(dirname "$OUT")"

printf 'Creating source snapshot: %s\n' "$SRC"
rsync -a \
  --exclude .git \
  --exclude .agents \
  --exclude .codex \
  --exclude .worker_worktrees \
  --exclude .venv \
  --exclude logs \
  --exclude outputs \
  --exclude __pycache__ \
  --exclude '*.pyc' \
  "$REPO_DIR"/ "$SRC"/

cleanup() {
  if [[ "$KEEP_SRC" != "1" ]]; then
    rm -rf "$SRC"
  fi
}
trap cleanup EXIT

printf 'Building image:\n'
printf '  base: %s\n' "$BASE_IMAGE"
printf '  out:  %s\n' "$OUT"
printf '  commit: %s\n' "$COMMIT"

srun \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-node="$GPUS" \
  --time="$WALLTIME" \
  --container-image="$BASE_IMAGE" \
  --container-save="$OUT" \
  --container-mounts="$SRC:/workspace/vllm-src" \
  bash -lc '
    set -euo pipefail
    CHECK_MODULE="$1"
    COMMIT="$2"
    BASE_IMAGE="$3"

    export HOME=/tmp/vllm-sqsh-build-home
    export XDG_CACHE_HOME=/tmp/vllm-sqsh-build-cache
    export FLASHINFER_WORKSPACE_DIR=/tmp/vllm-sqsh-build-cache/flashinfer
    mkdir -p "$HOME" "$XDG_CACHE_HOME" "$FLASHINFER_WORKSPACE_DIR"

    SITE=$(python3 - <<PY
from pathlib import Path
import site
import sysconfig

candidates = []
candidates.extend(site.getsitepackages())
purelib = sysconfig.get_paths().get("purelib")
if purelib:
    candidates.append(purelib)

seen = []
for path in candidates:
    if path and path not in seen:
        seen.append(path)

for path in seen:
    if (Path(path) / "vllm").exists():
        print(path)
        break
else:
    for path in seen:
        if path.endswith(("site-packages", "dist-packages")):
            print(path)
            break
    else:
        raise SystemExit(f"could not find Python package directory from: {seen}")
PY
)

    python3 - <<PY
import vllm
print("before_vllm_file", vllm.__file__)
PY

    cp -a /workspace/vllm-src/vllm "$SITE"/

    mkdir -p /opt/vllm-current-overlay-build-info
    {
      printf "base=%s\n" "$BASE_IMAGE"
      printf "source_commit=%s\n" "$COMMIT"
      printf "source_snapshot=/workspace/vllm-src\n"
      printf "method=copy local vllm package over stock site-packages, preserving compiled extensions\n"
    } > /opt/vllm-current-overlay-build-info/README.txt

    python3 - <<PY
import importlib
import inspect
import vllm
print("after_vllm_file", vllm.__file__)
mod = importlib.import_module(${CHECK_MODULE@Q})
print("check_module", inspect.getfile(mod))
try:
    import vllm._C
    print("has_vllm_C", True, vllm._C.__file__)
except Exception as exc:
    print("has_vllm_C", False, repr(exc))
    raise
PY
  ' _ "$CHECK_MODULE" "$COMMIT" "$BASE_IMAGE"

printf 'Saved image: %s\n' "$OUT"
