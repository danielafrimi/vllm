#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"

ACCOUNT="${ACCOUNT:-${SLURM_ACCOUNT:-nemotron_n4_compress}}"
PARTITION="${PARTITION:-${SLURM_PARTITION:-batch_long}}"
DEFAULT_WALLTIME="00:30:00"
WALLTIME="${WALLTIME:-${DEFAULT_WALLTIME}}"
GPUS="${GPUS:-1}"
MEM="${MEM:-64G}"
DEFAULT_JOB_NAME="fixed-io-bench"
JOB_NAME="${JOB_NAME:-${DEFAULT_JOB_NAME}}"

DEFAULT_MODEL="facebook/opt-125m"
MODEL="${MODEL:-${DEFAULT_MODEL}}"
VLLM_IMAGE="${VLLM_IMAGE:-}"
PORT="${PORT:-}"
DEFAULT_TP_SIZE="1"
TP_SIZE="${TP_SIZE:-${DEFAULT_TP_SIZE}}"
GPU_MEM="${GPU_MEM:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS:-}"
SERVER_ENV_VARS="${SERVER_ENV_VARS:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"

INPUT_LEN="${INPUT_LEN:-128}"
OUTPUT_LEN="${OUTPUT_LEN:-16}"
NUM_PROMPTS="${NUM_PROMPTS:-8}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"
NUM_WARMUPS="${NUM_WARMUPS:-1}"
WARMUP_INPUT_LEN="${WARMUP_INPUT_LEN:-}"
WARMUP_OUTPUT_LEN="${WARMUP_OUTPUT_LEN:-}"
TOKEN_ID_START="${TOKEN_ID_START:-1000}"
TOKEN_ID_RANGE="${TOKEN_ID_RANGE:-10000}"
CONSTANT_PROMPT_TOKEN_ID="${CONSTANT_PROMPT_TOKEN_ID:-}"
PROMPT_TOKEN_IDS_JSONL="${PROMPT_TOKEN_IDS_JSONL:-}"
BENCHMARK_EXTRA_ARGS="${BENCHMARK_EXTRA_ARGS:-}"

OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/fixed_io_benchmark}"
PRINT_ONLY=0
WAIT=0
SMOKE=0
NANO_DSA=0
DSA_ATTENTION="${DSA_ATTENTION:-}"
DSA_PROVIDER="${DSA_PROVIDER:-}"
NANO_DSA_ENFORCE_EAGER="${NANO_DSA_ENFORCE_EAGER:-1}"
NANO_DSA_CUDAGRAPH_MODE="${NANO_DSA_CUDAGRAPH_MODE:-}"

usage() {
  cat <<USAGE
Launch a Slurm job that starts vLLM serve and runs fixed-token IO benchmark.

Usage:
  $0 [options]

Options:
  --model MODEL                 Model/path for vllm serve. Default: ${MODEL}
  --image PATH                  Pyxis/SquashFS image. Default: auto-detect.
  --port PORT                   vLLM API port. Default: derived from Slurm job id.
  --job-name NAME               Slurm job name and run directory prefix.
  --account NAME                Slurm account. Default: ${ACCOUNT}
  --partition NAME              Slurm partition. Default: ${PARTITION}
  --time HH:MM:SS               Walltime. Default: ${WALLTIME}
  --gpus N                      GPUs for the job. Default: ${GPUS}
  --mem MEM                     Slurm memory. Default: ${MEM}
  --input-len N                 Exact input tokens. Default: ${INPUT_LEN}
  --output-len N                Exact output tokens. Default: ${OUTPUT_LEN}
  --num-prompts N               Number of requests. Default: ${NUM_PROMPTS}
  --request-rate R              Requests/sec or inf. Default: ${REQUEST_RATE}
  --max-concurrency N           Optional client concurrency cap.
  --num-warmups N               Warmup requests. Default: ${NUM_WARMUPS}
  --warmup-input-len N          Optional warmup input length override.
  --warmup-output-len N         Optional warmup output length override.
  --max-model-len N             vLLM max model len. Default: ${MAX_MODEL_LEN}
  --tp-size N                   Tensor parallel size. Default: ${TP_SIZE}
  --gpu-mem F                   GPU memory utilization. Default: ${GPU_MEM}
  --max-num-seqs N              Optional vLLM max concurrent sequences.
  --port PORT                   Server/client localhost port. Default: job-id-derived.
  --constant-prompt-token-id N  Reuse one raw JSON prompt body with token ID N.
  --prompt-token-ids-jsonl PATH JSONL with one prompt_token_ids list per prompt.
  --serve-extra-args ARGS       Extra args appended to vllm serve.
  --server-env KEY=VALUE        Export an env var inside the server container.
  --benchmark-extra-args ARGS   Extra args appended to benchmark client.
  --out-root DIR                Output root. Default: ${OUT_ROOT}
  --nano-dsa                    Use local Nano DSA TP2 long-context defaults.
  --dsa-attention NAME          DSA attention: refactored, legacy, or class path.
  --dsa-provider NAME           Refactored provider: efficient, pytorch, or class path.
  --nano-dsa-no-enforce-eager   Do not append --enforce-eager in --nano-dsa mode.
  --nano-dsa-cudagraph-mode M   In --nano-dsa mode, append compilation config
                                with cudagraph_mode M and disable enforce eager.
  --smoke                       Tiny dummy-load smoke benchmark.
  --print-only                  Write sbatch file but do not submit.
  --wait                        Submit and wait for terminal Slurm state.
  -h, --help                    Show this help.
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

q() {
  printf '%q' "$1"
}

resolve_dsa_attention_class() {
  case "$1" in
    refactored)
      printf '%s' \
        "vllm.model_executor.models.nemotron_h_dsa_attention_refactored.NemotronHDSARefactoredAttention"
      ;;
    legacy)
      printf '%s' \
        "vllm.model_executor.models.nemotron_h_dsa_attention_legacy.NemotronHDSALegacyAttention"
      ;;
    *)
      printf '%s' "$1"
      ;;
  esac
}

resolve_dsa_provider_class() {
  case "$1" in
    efficient|cuda|triton)
      printf '%s' \
        "vllm.model_executor.models.nemotron_h_chunked_dsa_components_efficient.EfficientChunkedDSAProviderBundle"
      ;;
    pytorch|torch)
      printf '%s' \
        "vllm.model_executor.models.nemotron_h_chunked_dsa_components_pytorch.TorchChunkedDSAProviderBundle"
      ;;
    nonchunked|nonchunked-pytorch|token|token-pytorch)
      printf '%s' \
        "vllm.model_executor.models.nemotron_h_nonchunked_dsa_components_pytorch.TorchNonChunkedDSAProviderBundle"
      ;;
    *)
      printf '%s' "$1"
      ;;
  esac
}

is_legacy_dsa_attention() {
  case "$1" in
    legacy|\
vllm.model_executor.models.nemotron_h_dsa_attention_legacy.NemotronHDSALegacyAttention)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      [[ $# -ge 2 ]] || die "--model requires a value"
      MODEL="$2"
      shift 2
      ;;
    --image)
      [[ $# -ge 2 ]] || die "--image requires a value"
      VLLM_IMAGE="$2"
      shift 2
      ;;
    --port)
      [[ $# -ge 2 ]] || die "--port requires a value"
      PORT="$2"
      shift 2
      ;;
    --job-name)
      [[ $# -ge 2 ]] || die "--job-name requires a value"
      JOB_NAME="$2"
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
    --mem)
      [[ $# -ge 2 ]] || die "--mem requires a value"
      MEM="$2"
      shift 2
      ;;
    --input-len)
      [[ $# -ge 2 ]] || die "--input-len requires a value"
      INPUT_LEN="$2"
      shift 2
      ;;
    --output-len)
      [[ $# -ge 2 ]] || die "--output-len requires a value"
      OUTPUT_LEN="$2"
      shift 2
      ;;
    --num-prompts)
      [[ $# -ge 2 ]] || die "--num-prompts requires a value"
      NUM_PROMPTS="$2"
      shift 2
      ;;
    --request-rate)
      [[ $# -ge 2 ]] || die "--request-rate requires a value"
      REQUEST_RATE="$2"
      shift 2
      ;;
    --max-concurrency)
      [[ $# -ge 2 ]] || die "--max-concurrency requires a value"
      MAX_CONCURRENCY="$2"
      shift 2
      ;;
    --num-warmups)
      [[ $# -ge 2 ]] || die "--num-warmups requires a value"
      NUM_WARMUPS="$2"
      shift 2
      ;;
    --warmup-input-len)
      [[ $# -ge 2 ]] || die "--warmup-input-len requires a value"
      WARMUP_INPUT_LEN="$2"
      shift 2
      ;;
    --warmup-output-len)
      [[ $# -ge 2 ]] || die "--warmup-output-len requires a value"
      WARMUP_OUTPUT_LEN="$2"
      shift 2
      ;;
    --max-model-len)
      [[ $# -ge 2 ]] || die "--max-model-len requires a value"
      MAX_MODEL_LEN="$2"
      shift 2
      ;;
    --tp-size)
      [[ $# -ge 2 ]] || die "--tp-size requires a value"
      TP_SIZE="$2"
      shift 2
      ;;
    --gpu-mem)
      [[ $# -ge 2 ]] || die "--gpu-mem requires a value"
      GPU_MEM="$2"
      shift 2
      ;;
    --max-num-seqs)
      [[ $# -ge 2 ]] || die "--max-num-seqs requires a value"
      MAX_NUM_SEQS="$2"
      shift 2
      ;;
    --constant-prompt-token-id)
      [[ $# -ge 2 ]] || die "--constant-prompt-token-id requires a value"
      CONSTANT_PROMPT_TOKEN_ID="$2"
      shift 2
      ;;
    --prompt-token-ids-jsonl)
      [[ $# -ge 2 ]] || die "--prompt-token-ids-jsonl requires a value"
      PROMPT_TOKEN_IDS_JSONL="$2"
      shift 2
      ;;
    --serve-extra-args)
      [[ $# -ge 2 ]] || die "--serve-extra-args requires a value"
      SERVE_EXTRA_ARGS="$2"
      shift 2
      ;;
    --server-env)
      [[ $# -ge 2 ]] || die "--server-env requires a value"
      SERVER_ENV_VARS="${SERVER_ENV_VARS} $2"
      shift 2
      ;;
    --benchmark-extra-args)
      [[ $# -ge 2 ]] || die "--benchmark-extra-args requires a value"
      BENCHMARK_EXTRA_ARGS="$2"
      shift 2
      ;;
    --out-root)
      [[ $# -ge 2 ]] || die "--out-root requires a value"
      OUT_ROOT="$2"
      shift 2
      ;;
    --nano-dsa)
      NANO_DSA=1
      shift
      ;;
    --dsa-attention)
      [[ $# -ge 2 ]] || die "--dsa-attention requires a value"
      DSA_ATTENTION="$2"
      shift 2
      ;;
    --dsa-provider)
      [[ $# -ge 2 ]] || die "--dsa-provider requires a value"
      DSA_PROVIDER="$2"
      shift 2
      ;;
    --nano-dsa-no-enforce-eager)
      NANO_DSA_ENFORCE_EAGER=0
      shift
      ;;
    --nano-dsa-cudagraph-mode)
      [[ $# -ge 2 ]] || die "--nano-dsa-cudagraph-mode requires a value"
      NANO_DSA_ENFORCE_EAGER=0
      NANO_DSA_CUDAGRAPH_MODE="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    --wait)
      WAIT=1
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

if [[ "${NANO_DSA}" == "1" ]]; then
  if [[ "${MODEL}" == "${DEFAULT_MODEL}" ]]; then
    MODEL="/lustre/fsw/portfolios/coreai/users/mdabbah/deci/puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints/nano-dsa-16chunk_size-2048chunks"
  fi
  if [[ "${TP_SIZE}" == "${DEFAULT_TP_SIZE}" ]]; then
    TP_SIZE=2
  fi
  if [[ "${GPUS}" == "1" ]]; then
    GPUS=2
  fi
  if [[ "${WALLTIME}" == "${DEFAULT_WALLTIME}" ]]; then
    WALLTIME="12:00:00"
  fi
  if [[ -z "${CONSTANT_PROMPT_TOKEN_ID}" && -z "${PROMPT_TOKEN_IDS_JSONL}" ]]; then
    CONSTANT_PROMPT_TOKEN_ID="${TOKEN_ID_START}"
  fi
  if [[ -z "${MAX_NUM_SEQS}" ]]; then
    MAX_NUM_SEQS=8
  fi
  if [[ -z "${DSA_ATTENTION}" ]]; then
    DSA_ATTENTION=refactored
  fi
  if [[ -z "${DSA_PROVIDER}" ]] && ! is_legacy_dsa_attention "${DSA_ATTENTION}"; then
    DSA_PROVIDER=efficient
  fi

  NANO_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA="${NANO_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA:-1}"
  NANO_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA="${NANO_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA:-1}"

  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16"
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=1"
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA=1"
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ=1"
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA=${NANO_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA}"
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA=${NANO_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA}"
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE=4096"
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING=1"
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_USE_TRITON_BATCHED_SUMMARIES=1"

  NANO_DSA_GRAPH_ARGS=""
  if [[ "${NANO_DSA_ENFORCE_EAGER}" == "1" ]]; then
    NANO_DSA_GRAPH_ARGS="--enforce-eager"
  elif [[ -n "${NANO_DSA_CUDAGRAPH_MODE}" ]]; then
    NANO_DSA_GRAPH_ARGS="--compilation-config {\"mode\":3,\"cudagraph_mode\":\"${NANO_DSA_CUDAGRAPH_MODE}\"}"
  fi

  SERVE_EXTRA_ARGS="--trust-remote-code --attention-backend FLASH_ATTN --enable-expert-parallel --model-loader-extra-config {\"enable_multithread_load\":true,\"num_threads\":96} --max-num-seqs ${MAX_NUM_SEQS} --mamba-ssm-cache-dtype float32 --no-enable-prefix-caching --enable-chunked-prefill ${NANO_DSA_GRAPH_ARGS} --block-size 16 --max-num-batched-tokens 8192 --skip-tokenizer-init ${SERVE_EXTRA_ARGS}"
  BENCHMARK_EXTRA_ARGS="--return-token-ids ${BENCHMARK_EXTRA_ARGS}"
fi

if [[ -n "${DSA_ATTENTION}" ]]; then
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS=$(resolve_dsa_attention_class "${DSA_ATTENTION}")"
fi

if [[ -n "${DSA_PROVIDER}" ]]; then
  SERVER_ENV_VARS="${SERVER_ENV_VARS} VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS=$(resolve_dsa_provider_class "${DSA_PROVIDER}")"
fi

if [[ -n "${MAX_NUM_SEQS}" && "${NANO_DSA}" != "1" ]]; then
  SERVE_EXTRA_ARGS="${SERVE_EXTRA_ARGS} --max-num-seqs ${MAX_NUM_SEQS}"
fi

if [[ -n "${CONSTANT_PROMPT_TOKEN_ID}" && -n "${PROMPT_TOKEN_IDS_JSONL}" ]]; then
  die "--constant-prompt-token-id cannot be combined with --prompt-token-ids-jsonl"
fi

if [[ -n "${PROMPT_TOKEN_IDS_JSONL}" && ! -f "${PROMPT_TOKEN_IDS_JSONL}" ]]; then
  die "--prompt-token-ids-jsonl does not exist: ${PROMPT_TOKEN_IDS_JSONL}"
fi

if [[ "${SMOKE}" == "1" ]]; then
  if [[ "${JOB_NAME}" == "${DEFAULT_JOB_NAME}" ]]; then
    JOB_NAME="fixed-io-smoke"
  fi
  INPUT_LEN=16
  OUTPUT_LEN=2
  NUM_PROMPTS=2
  NUM_WARMUPS=1
  MAX_MODEL_LEN=128
  if [[ "${WALLTIME}" == "${DEFAULT_WALLTIME}" ]]; then
    WALLTIME="00:15:00"
  fi
  SERVE_EXTRA_ARGS="--load-format dummy --enforce-eager --skip-tokenizer-init ${SERVE_EXTRA_ARGS}"
  BENCHMARK_EXTRA_ARGS="--return-token-ids ${BENCHMARK_EXTRA_ARGS}"
fi

if [[ -z "${VLLM_IMAGE}" ]]; then
  VLLM_COMMIT="$(git -C "${REPO_DIR}" rev-parse --short=10 HEAD 2>/dev/null \
    || printf unknown)"
  VLLM_IMAGE="$(
    find "${REPO_DIR}/outputs/containers" -maxdepth 1 -type f \
      -name "vllm-v0.22.0-current-overlay-${VLLM_COMMIT}-*.sqsh" \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | sed -n '1s/^[^ ]* //p'
  )"
  if [[ -z "${VLLM_IMAGE}" ]]; then
    VLLM_IMAGE="$(
      find "${REPO_DIR}/outputs/containers" -maxdepth 1 -type f \
        -name 'vllm-*.sqsh' \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr \
        | sed -n '1s/^[^ ]* //p'
    )"
  fi
fi

[[ -n "${VLLM_IMAGE}" && -f "${VLLM_IMAGE}" ]] \
  || die "no .sqsh image found; pass --image PATH or set VLLM_IMAGE"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${JOB_NAME}_${STAMP}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"
LAUNCHER_DIR="${RUN_DIR}/launchers"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${LAUNCHER_DIR}"

SBATCH_FILE="${LAUNCHER_DIR}/${JOB_NAME}_${STAMP}.sbatch"
SLURM_LOG="${LOG_DIR}/${JOB_NAME}_%j.log"
SERVER_LOG="${LOG_DIR}/server.log"
RESULT_JSON="${RESULT_DIR}/fixed_io_result.json"

{
  printf '#!/usr/bin/env bash\n'
  printf '#SBATCH --job-name=%s\n' "${JOB_NAME}"
  printf '#SBATCH --account=%s\n' "${ACCOUNT}"
  printf '#SBATCH --partition=%s\n' "${PARTITION}"
  printf '#SBATCH --nodes=1\n'
  printf '#SBATCH --ntasks=1\n'
  printf '#SBATCH --gres=gpu:%s\n' "${GPUS}"
  printf '#SBATCH --mem=%s\n' "${MEM}"
  printf '#SBATCH --time=%s\n' "${WALLTIME}"
  printf '#SBATCH --export=ALL\n'
  printf '#SBATCH --output=%s\n' "${SLURM_LOG}"
  printf '#SBATCH --error=%s\n' "${SLURM_LOG}"
  printf '\nset -euo pipefail\n\n'
  printf 'REPO_DIR=%s\n' "$(q "${REPO_DIR}")"
  printf 'VLLM_IMAGE=%s\n' "$(q "${VLLM_IMAGE}")"
  printf 'MODEL=%s\n' "$(q "${MODEL}")"
  printf 'PORT=%s\n' "$(q "${PORT}")"
  printf 'TP_SIZE=%s\n' "$(q "${TP_SIZE}")"
  printf 'GPU_MEM=%s\n' "$(q "${GPU_MEM}")"
  printf 'MAX_MODEL_LEN=%s\n' "$(q "${MAX_MODEL_LEN}")"
  printf 'SERVE_EXTRA_ARGS=%s\n' "$(q "${SERVE_EXTRA_ARGS}")"
  printf 'SERVER_ENV_VARS=%s\n' "$(q "${SERVER_ENV_VARS}")"
  printf 'INPUT_LEN=%s\n' "$(q "${INPUT_LEN}")"
  printf 'OUTPUT_LEN=%s\n' "$(q "${OUTPUT_LEN}")"
  printf 'NUM_PROMPTS=%s\n' "$(q "${NUM_PROMPTS}")"
  printf 'REQUEST_RATE=%s\n' "$(q "${REQUEST_RATE}")"
  printf 'MAX_CONCURRENCY=%s\n' "$(q "${MAX_CONCURRENCY}")"
  printf 'NUM_WARMUPS=%s\n' "$(q "${NUM_WARMUPS}")"
  printf 'WARMUP_INPUT_LEN=%s\n' "$(q "${WARMUP_INPUT_LEN}")"
  printf 'WARMUP_OUTPUT_LEN=%s\n' "$(q "${WARMUP_OUTPUT_LEN}")"
  printf 'TOKEN_ID_START=%s\n' "$(q "${TOKEN_ID_START}")"
  printf 'TOKEN_ID_RANGE=%s\n' "$(q "${TOKEN_ID_RANGE}")"
  printf 'CONSTANT_PROMPT_TOKEN_ID=%s\n' "$(q "${CONSTANT_PROMPT_TOKEN_ID}")"
  printf 'PROMPT_TOKEN_IDS_JSONL=%s\n' "$(q "${PROMPT_TOKEN_IDS_JSONL}")"
  printf 'BENCHMARK_EXTRA_ARGS=%s\n' "$(q "${BENCHMARK_EXTRA_ARGS}")"
  printf 'LOG_DIR=%s\n' "$(q "${LOG_DIR}")"
  printf 'RESULT_DIR=%s\n' "$(q "${RESULT_DIR}")"
  printf 'SERVER_LOG=%s\n' "$(q "${SERVER_LOG}")"
  printf 'RESULT_JSON=%s\n' "$(q "${RESULT_JSON}")"
  cat <<'SBATCH_BODY'

export REPO_DIR VLLM_IMAGE MODEL PORT TP_SIZE GPU_MEM MAX_MODEL_LEN
export SERVE_EXTRA_ARGS SERVER_ENV_VARS INPUT_LEN OUTPUT_LEN NUM_PROMPTS REQUEST_RATE
export MAX_CONCURRENCY NUM_WARMUPS WARMUP_INPUT_LEN WARMUP_OUTPUT_LEN
export TOKEN_ID_START TOKEN_ID_RANGE
export CONSTANT_PROMPT_TOKEN_ID PROMPT_TOKEN_IDS_JSONL BENCHMARK_EXTRA_ARGS
export LOG_DIR RESULT_DIR SERVER_LOG RESULT_JSON

if [[ -z "${PORT}" ]]; then
  PORT="$((20000 + (${SLURM_JOB_ID:-$$} % 40000)))"
  export PORT
fi

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "image=${VLLM_IMAGE}"
echo "model=${MODEL}"
echo "port=${PORT}"
echo "repo=${REPO_DIR}"
echo "result_json=${RESULT_JSON}"

srun \
  --container-image="${VLLM_IMAGE}" \
  --container-mounts="${REPO_DIR}:/workspace/vllm-src,${LOG_DIR}:/logs,${RESULT_DIR}:/results,/lustre:/lustre,/scratch:/scratch" \
  bash -lc '
    set -euo pipefail

    cd /tmp
    unset PYTHONPATH
    export HOME="/tmp/${USER}/fixed-io-home"
    export XDG_CACHE_HOME="/tmp/${USER}/fixed-io-cache"
    export VLLM_USAGE_SOURCE="fixed-io-benchmark"
    mkdir -p "${HOME}" "${XDG_CACHE_HOME}" /logs /results
    for env_assignment in ${SERVER_ENV_VARS}; do
      export "${env_assignment}"
    done

    PYTHON_BIN="${PYTHON_BIN:-python3}"

    echo "container_host=$(hostname)"
    echo "python_bin=${PYTHON_BIN}"
    nvidia-smi || true
    "${PYTHON_BIN}" - <<PY
import importlib
import vllm

print("vllm_file", vllm.__file__)
try:
    mod = importlib.import_module("vllm._C")
    print("vllm_C_file", getattr(mod, "__file__", "built-in"))
except Exception as exc:
    print("vllm_C_import_error", repr(exc))
    raise
PY

    read -r -a SERVE_ARGS <<< "${SERVE_EXTRA_ARGS}"
    read -r -a BENCH_ARGS <<< "${BENCHMARK_EXTRA_ARGS}"

    server_ready() {
      "${PYTHON_BIN}" - "${PORT}" <<PY
import sys
import urllib.request

url = f"http://127.0.0.1:{sys.argv[1]}/v1/models"
try:
    with urllib.request.urlopen(url, timeout=2) as response:
        raise SystemExit(0 if 200 <= response.status < 300 else 1)
except Exception:
    raise SystemExit(1)
PY
    }

    stop_server() {
      if [[ -n "${SERVER_PID:-}" ]] && kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        kill "${SERVER_PID}" >/dev/null 2>&1 || true
        wait "${SERVER_PID}" >/dev/null 2>&1 || true
      fi
    }
    trap stop_server EXIT

    "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "${MODEL}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --max-model-len "${MAX_MODEL_LEN}" \
      --gpu-memory-utilization "${GPU_MEM}" \
      --no-enable-log-requests \
      "${SERVE_ARGS[@]}" \
      >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!

    for _ in $(seq 1 900); do
      if server_ready; then
        break
      fi
      if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
        echo "vllm server exited before readiness" >&2
        tail -200 "${SERVER_LOG}" >&2 || true
        exit 1
      fi
      sleep 2
    done

    server_ready || {
      echo "vllm server did not become ready" >&2
      tail -200 "${SERVER_LOG}" >&2 || true
      exit 1
    }

    CLIENT_ARGS=(
      /workspace/vllm-src/benchmarks/benchmark_fixed_io_serving.py
      --base-url "http://127.0.0.1:${PORT}"
      --input-len "${INPUT_LEN}"
      --output-len "${OUTPUT_LEN}"
      --num-prompts "${NUM_PROMPTS}"
      --request-rate "${REQUEST_RATE}"
      --num-warmups "${NUM_WARMUPS}"
      --token-id-start "${TOKEN_ID_START}"
      --token-id-range "${TOKEN_ID_RANGE}"
      --output-json "${RESULT_JSON}"
      --save-detailed
    )
    if [[ -n "${MAX_CONCURRENCY}" ]]; then
      CLIENT_ARGS+=(--max-concurrency "${MAX_CONCURRENCY}")
    fi
    if [[ -n "${WARMUP_INPUT_LEN}" ]]; then
      CLIENT_ARGS+=(--warmup-input-len "${WARMUP_INPUT_LEN}")
    fi
    if [[ -n "${WARMUP_OUTPUT_LEN}" ]]; then
      CLIENT_ARGS+=(--warmup-output-len "${WARMUP_OUTPUT_LEN}")
    fi
    if [[ -n "${CONSTANT_PROMPT_TOKEN_ID}" ]]; then
      CLIENT_ARGS+=(--constant-prompt-token-id "${CONSTANT_PROMPT_TOKEN_ID}")
    fi
    if [[ -n "${PROMPT_TOKEN_IDS_JSONL}" ]]; then
      CLIENT_ARGS+=(--prompt-token-ids-jsonl "${PROMPT_TOKEN_IDS_JSONL}")
    fi
    CLIENT_ARGS+=("${BENCH_ARGS[@]}")

    "${PYTHON_BIN}" "${CLIENT_ARGS[@]}"
  '
SBATCH_BODY
} >"${SBATCH_FILE}"

chmod 755 "${SBATCH_FILE}"

printf 'Wrote sbatch launcher: %s\n' "${SBATCH_FILE}"
printf 'Slurm log pattern:     %s\n' "${SLURM_LOG}"
printf 'Server log:            %s\n' "${SERVER_LOG}"
printf 'Result JSON:           %s\n' "${RESULT_JSON}"

if [[ "${PRINT_ONLY}" == "1" ]]; then
  exit 0
fi

SUBMIT_OUTPUT="$(sbatch "${SBATCH_FILE}")"
printf '%s\n' "${SUBMIT_OUTPUT}"
JOB_ID="$(sed -n 's/^Submitted batch job //p' <<<"${SUBMIT_OUTPUT}")"
[[ -n "${JOB_ID}" ]] || die "could not parse job id from sbatch output"

printf 'job_id=%s\n' "${JOB_ID}"
printf 'run_dir=%s\n' "${RUN_DIR}"

if [[ "${WAIT}" != "1" ]]; then
  exit 0
fi

while squeue -h -j "${JOB_ID}" >/dev/null 2>&1 \
  && [[ -n "$(squeue -h -j "${JOB_ID}" 2>/dev/null)" ]]
do
  printf '[%s] job %s still running or pending\n' "$(date +%H:%M:%S)" "${JOB_ID}"
  sleep 20
done

if command -v sacct >/dev/null 2>&1; then
  sacct -j "${JOB_ID}" --format=JobID,JobName%24,State,ExitCode,Elapsed
fi

if [[ -f "${RESULT_JSON}" ]]; then
  printf 'Result JSON exists: %s\n' "${RESULT_JSON}"
else
  printf 'Result JSON missing: %s\n' "${RESULT_JSON}" >&2
  exit 1
fi
