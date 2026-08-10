#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_DIR="${REPO_DIR:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"

ACCOUNT="${ACCOUNT:-${SLURM_ACCOUNT:-nemotron_n4_compress}}"
PARTITION="${PARTITION:-${SLURM_PARTITION:-batch}}"
WALLTIME="${WALLTIME:-03:00:00}"
GPUS="${GPUS:-2}"
MEM="${MEM:-64G}"
NODELIST="${NODELIST:-}"
JOB_NAME="${JOB_NAME:-fixedio-nsys-prev}"

MODEL="${MODEL:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints/nano-dsa-16chunk_size-2048chunks}"
BASELINE_IMAGE="${BASELINE_IMAGE:-}"
CURRENT_IMAGE="${CURRENT_IMAGE:-}"
PORT="${PORT:-}"
TP_SIZE="${TP_SIZE:-2}"
GPU_MEM="${GPU_MEM:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1049600}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"

INPUT_LEN="${INPUT_LEN:-1000000}"
OUTPUT_LEN="${OUTPUT_LEN:-10}"
NUM_PROMPTS="${NUM_PROMPTS:-2}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"
NUM_WARMUPS="${NUM_WARMUPS:-0}"
TOKEN_ID_START="${TOKEN_ID_START:-1000}"
TOKEN_ID_RANGE="${TOKEN_ID_RANGE:-10000}"
CONSTANT_PROMPT_TOKEN_ID="${CONSTANT_PROMPT_TOKEN_ID:-1000}"
BENCHMARK_EXTRA_ARGS="${BENCHMARK_EXTRA_ARGS:---return-token-ids --timeout-s 12600}"

PROFILE_WINDOWS="${PROFILE_WINDOWS:-3}"
PROFILE_WINDOW_SECONDS="${PROFILE_WINDOW_SECONDS:-1}"
PROFILE_GAP_SECONDS="${PROFILE_GAP_SECONDS:-30}"
PROFILE_INITIAL_DELAY_SECONDS="${PROFILE_INITIAL_DELAY_SECONDS:-20}"
PROFILE_TIMEOUT_SECONDS="${PROFILE_TIMEOUT_SECONDS:-30}"
NSYS_CMD="${NSYS_CMD:-}"
NSYS_HOST_PATH="${NSYS_HOST_PATH:-}"
NSYS_HOST_ROOT="${NSYS_HOST_ROOT:-}"
DEFAULT_NSYS_HOST_PATH="${DEFAULT_NSYS_HOST_PATH:-${REPO_DIR}/outputs/tools/nsight-systems-cli/unpacked/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/nsys}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"
NSYS_CUDA_GRAPH_TRACE="${NSYS_CUDA_GRAPH_TRACE:-graph}"
EXPORT_SQLITE="${EXPORT_SQLITE:-1}"
CURRENT_ONLY="${CURRENT_ONLY:-0}"
USE_TRITON_BATCHED_SUMMARIES="${USE_TRITON_BATCHED_SUMMARIES:-1}"
USE_FLATTENED_PREFILL_PAGE_TABLE_FA="${USE_FLATTENED_PREFILL_PAGE_TABLE_FA:-1}"
USE_FLATTENED_DECODE_PAGE_TABLE_FA="${USE_FLATTENED_DECODE_PAGE_TABLE_FA:-1}"
BATCHED_SUMMARY_PRINT_LIMIT="${BATCHED_SUMMARY_PRINT_LIMIT:-0}"
PATH_DEBUG_PRINT_LIMIT="${PATH_DEBUG_PRINT_LIMIT:-0}"
NANO_DSA_ENFORCE_EAGER="${NANO_DSA_ENFORCE_EAGER:-1}"
NANO_DSA_CUDAGRAPH_MODE="${NANO_DSA_CUDAGRAPH_MODE:-}"

OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/outputs/fixed_io_nsys_profile_matrix}"
PRINT_ONLY=0
WAIT=0

usage() {
  cat <<USAGE
Launch one Slurm allocation that profiles the previous fixed-IO chunked-prefill
scenario under Nsight Systems for baseline first, then current.

Usage:
  $0 [options]

Options:
  --baseline-image PATH         Baseline Pyxis/SQSH image.
  --current-image PATH          Current Pyxis/SQSH image.
  --model PATH                  Model/checkpoint path. Default: ${MODEL}
  --job-name NAME               Slurm job/run prefix. Default: ${JOB_NAME}
  --account NAME                Slurm account. Default: ${ACCOUNT}
  --partition NAME              Slurm partition. Default: ${PARTITION}
  --time HH:MM:SS               Walltime. Default: ${WALLTIME}
  --gpus N                      GPUs for the job. Default: ${GPUS}
  --mem MEM                     Slurm memory. Default: ${MEM}
  --nodelist NODELIST           Optional Slurm node list.
  --num-prompts N               Measured prompts per case. Default: ${NUM_PROMPTS}
  --num-warmups N               Warmup prompts per case. Default: ${NUM_WARMUPS}
  --profile-windows N           Number of capture windows. Default: ${PROFILE_WINDOWS}
  --profile-window-seconds S    Seconds per capture window. Default: ${PROFILE_WINDOW_SECONDS}
  --profile-gap-seconds S       Gap between windows. Default: ${PROFILE_GAP_SECONDS}
  --profile-initial-delay S     Delay before first window. Default: ${PROFILE_INITIAL_DELAY_SECONDS}
  --nsys-cmd PATH               Nsight Systems command. Default: auto-detect.
  --nsys-host-path PATH         Host nsys path to mount into the container.
                                Default: local downloaded CLI when present.
  --nsys-host-root DIR          Host Nsight root to mount. Default: parent of nsys bin dir.
  --out-root DIR                Output root. Default: ${OUT_ROOT}
  --current-only                Profile only the current image.
  --nano-dsa-no-enforce-eager   Do not append --enforce-eager.
  --nano-dsa-cudagraph-mode M   Append compilation config with cudagraph_mode M
                                and disable enforce eager.
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

latest_image() {
  local pattern=$1
  find "${REPO_DIR}/outputs/containers" -maxdepth 1 -type f \
    -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | sed -n '1s/^[^ ]* //p'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --baseline-image)
      [[ $# -ge 2 ]] || die "--baseline-image requires a value"
      BASELINE_IMAGE="$2"
      shift 2
      ;;
    --current-image)
      [[ $# -ge 2 ]] || die "--current-image requires a value"
      CURRENT_IMAGE="$2"
      shift 2
      ;;
    --model)
      [[ $# -ge 2 ]] || die "--model requires a value"
      MODEL="$2"
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
    --nodelist)
      [[ $# -ge 2 ]] || die "--nodelist requires a value"
      NODELIST="$2"
      shift 2
      ;;
    --num-prompts)
      [[ $# -ge 2 ]] || die "--num-prompts requires a value"
      NUM_PROMPTS="$2"
      shift 2
      ;;
    --num-warmups)
      [[ $# -ge 2 ]] || die "--num-warmups requires a value"
      NUM_WARMUPS="$2"
      shift 2
      ;;
    --profile-windows)
      [[ $# -ge 2 ]] || die "--profile-windows requires a value"
      PROFILE_WINDOWS="$2"
      shift 2
      ;;
    --profile-window-seconds)
      [[ $# -ge 2 ]] || die "--profile-window-seconds requires a value"
      PROFILE_WINDOW_SECONDS="$2"
      shift 2
      ;;
    --profile-gap-seconds)
      [[ $# -ge 2 ]] || die "--profile-gap-seconds requires a value"
      PROFILE_GAP_SECONDS="$2"
      shift 2
      ;;
    --profile-initial-delay)
      [[ $# -ge 2 ]] || die "--profile-initial-delay requires a value"
      PROFILE_INITIAL_DELAY_SECONDS="$2"
      shift 2
      ;;
    --nsys-cmd)
      [[ $# -ge 2 ]] || die "--nsys-cmd requires a value"
      NSYS_CMD="$2"
      shift 2
      ;;
    --nsys-host-path)
      [[ $# -ge 2 ]] || die "--nsys-host-path requires a value"
      NSYS_HOST_PATH="$2"
      shift 2
      ;;
    --nsys-host-root)
      [[ $# -ge 2 ]] || die "--nsys-host-root requires a value"
      NSYS_HOST_ROOT="$2"
      shift 2
      ;;
    --out-root)
      [[ $# -ge 2 ]] || die "--out-root requires a value"
      OUT_ROOT="$2"
      shift 2
      ;;
    --current-only)
      CURRENT_ONLY=1
      shift
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

if [[ -z "${BASELINE_IMAGE}" ]]; then
  BASELINE_IMAGE="$(latest_image 'vllm-v0.22.0-fixedio-baseline-*.sqsh')"
fi
if [[ -z "${CURRENT_IMAGE}" ]]; then
  CURRENT_IMAGE="$(latest_image 'vllm-v0.22.0-fixedio-current-*.sqsh')"
fi

[[ "${CURRENT_ONLY}" == "1" || ( -n "${BASELINE_IMAGE}" && -f "${BASELINE_IMAGE}" ) ]] \
  || die "baseline image not found; pass --baseline-image"
[[ -n "${CURRENT_IMAGE}" && -f "${CURRENT_IMAGE}" ]] \
  || die "current image not found; pass --current-image"

NSYS_CONTAINER_MOUNT=""
if [[ -z "${NSYS_CMD}" && -z "${NSYS_HOST_PATH}" \
    && -x "${DEFAULT_NSYS_HOST_PATH}" ]]; then
  NSYS_HOST_PATH="${DEFAULT_NSYS_HOST_PATH}"
fi
if [[ -n "${NSYS_HOST_PATH}" ]]; then
  if [[ "${NSYS_HOST_PATH}" != /* ]]; then
    NSYS_HOST_PATH="${REPO_DIR}/${NSYS_HOST_PATH}"
  fi
  if [[ -z "${NSYS_HOST_ROOT}" ]]; then
    NSYS_HOST_DIR="${NSYS_HOST_PATH%/*}"
    NSYS_HOST_ROOT="${NSYS_HOST_DIR%/*}"
  fi
  NSYS_REL_PATH="${NSYS_HOST_PATH#${NSYS_HOST_ROOT}/}"
  NSYS_CONTAINER_MOUNT=",${NSYS_HOST_ROOT}:/opt/nsys-host"
  NSYS_CMD="/opt/nsys-host/${NSYS_REL_PATH}"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUT_ROOT}/${JOB_NAME}_${STAMP}"
LOG_DIR="${RUN_DIR}/logs"
RESULT_DIR="${RUN_DIR}/results"
PROFILE_DIR="${RUN_DIR}/profiles"
LAUNCHER_DIR="${RUN_DIR}/launchers"
mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${PROFILE_DIR}" "${LAUNCHER_DIR}"

SBATCH_FILE="${LAUNCHER_DIR}/${JOB_NAME}_${STAMP}.sbatch"
SLURM_LOG="${LOG_DIR}/${JOB_NAME}_%j.log"

{
  printf '#!/usr/bin/env bash\n'
  printf '#SBATCH --job-name=%s\n' "${JOB_NAME}"
  printf '#SBATCH --account=%s\n' "${ACCOUNT}"
  printf '#SBATCH --partition=%s\n' "${PARTITION}"
  printf '#SBATCH --nodes=1\n'
  printf '#SBATCH --ntasks=1\n'
  printf '#SBATCH --gres=gpu:%s\n' "${GPUS}"
  printf '#SBATCH --mem=%s\n' "${MEM}"
  if [[ -n "${NODELIST}" ]]; then
    printf '#SBATCH --nodelist=%s\n' "${NODELIST}"
  fi
  printf '#SBATCH --time=%s\n' "${WALLTIME}"
  printf '#SBATCH --export=ALL\n'
  printf '#SBATCH --output=%s\n' "${SLURM_LOG}"
  printf '#SBATCH --error=%s\n' "${SLURM_LOG}"
  printf '\nset -euo pipefail\n\n'
  printf 'REPO_DIR=%s\n' "$(q "${REPO_DIR}")"
  printf 'BASELINE_IMAGE=%s\n' "$(q "${BASELINE_IMAGE}")"
  printf 'CURRENT_IMAGE=%s\n' "$(q "${CURRENT_IMAGE}")"
  printf 'MODEL=%s\n' "$(q "${MODEL}")"
  printf 'PORT=%s\n' "$(q "${PORT}")"
  printf 'TP_SIZE=%s\n' "$(q "${TP_SIZE}")"
  printf 'GPU_MEM=%s\n' "$(q "${GPU_MEM}")"
  printf 'MAX_MODEL_LEN=%s\n' "$(q "${MAX_MODEL_LEN}")"
  printf 'MAX_NUM_SEQS=%s\n' "$(q "${MAX_NUM_SEQS}")"
  printf 'INPUT_LEN=%s\n' "$(q "${INPUT_LEN}")"
  printf 'OUTPUT_LEN=%s\n' "$(q "${OUTPUT_LEN}")"
  printf 'NUM_PROMPTS=%s\n' "$(q "${NUM_PROMPTS}")"
  printf 'REQUEST_RATE=%s\n' "$(q "${REQUEST_RATE}")"
  printf 'MAX_CONCURRENCY=%s\n' "$(q "${MAX_CONCURRENCY}")"
  printf 'NUM_WARMUPS=%s\n' "$(q "${NUM_WARMUPS}")"
  printf 'TOKEN_ID_START=%s\n' "$(q "${TOKEN_ID_START}")"
  printf 'TOKEN_ID_RANGE=%s\n' "$(q "${TOKEN_ID_RANGE}")"
  printf 'CONSTANT_PROMPT_TOKEN_ID=%s\n' "$(q "${CONSTANT_PROMPT_TOKEN_ID}")"
  printf 'BENCHMARK_EXTRA_ARGS=%s\n' "$(q "${BENCHMARK_EXTRA_ARGS}")"
  printf 'PROFILE_WINDOWS=%s\n' "$(q "${PROFILE_WINDOWS}")"
  printf 'PROFILE_WINDOW_SECONDS=%s\n' "$(q "${PROFILE_WINDOW_SECONDS}")"
  printf 'PROFILE_GAP_SECONDS=%s\n' "$(q "${PROFILE_GAP_SECONDS}")"
  printf 'PROFILE_INITIAL_DELAY_SECONDS=%s\n' "$(q "${PROFILE_INITIAL_DELAY_SECONDS}")"
  printf 'PROFILE_TIMEOUT_SECONDS=%s\n' "$(q "${PROFILE_TIMEOUT_SECONDS}")"
  printf 'NSYS_CMD=%s\n' "$(q "${NSYS_CMD}")"
  printf 'NSYS_CONTAINER_MOUNT=%s\n' "$(q "${NSYS_CONTAINER_MOUNT}")"
  printf 'NSYS_EXTRA_ARGS=%s\n' "$(q "${NSYS_EXTRA_ARGS}")"
  printf 'NSYS_CUDA_GRAPH_TRACE=%s\n' "$(q "${NSYS_CUDA_GRAPH_TRACE}")"
  printf 'EXPORT_SQLITE=%s\n' "$(q "${EXPORT_SQLITE}")"
  printf 'CURRENT_ONLY=%s\n' "$(q "${CURRENT_ONLY}")"
  printf 'USE_TRITON_BATCHED_SUMMARIES=%s\n' "$(q "${USE_TRITON_BATCHED_SUMMARIES}")"
  printf 'USE_FLATTENED_PREFILL_PAGE_TABLE_FA=%s\n' "$(q "${USE_FLATTENED_PREFILL_PAGE_TABLE_FA}")"
  printf 'USE_FLATTENED_DECODE_PAGE_TABLE_FA=%s\n' "$(q "${USE_FLATTENED_DECODE_PAGE_TABLE_FA}")"
  printf 'BATCHED_SUMMARY_PRINT_LIMIT=%s\n' "$(q "${BATCHED_SUMMARY_PRINT_LIMIT}")"
  printf 'PATH_DEBUG_PRINT_LIMIT=%s\n' "$(q "${PATH_DEBUG_PRINT_LIMIT}")"
  printf 'NANO_DSA_ENFORCE_EAGER=%s\n' "$(q "${NANO_DSA_ENFORCE_EAGER}")"
  printf 'NANO_DSA_CUDAGRAPH_MODE=%s\n' "$(q "${NANO_DSA_CUDAGRAPH_MODE}")"
  printf 'LOG_DIR=%s\n' "$(q "${LOG_DIR}")"
  printf 'RESULT_DIR=%s\n' "$(q "${RESULT_DIR}")"
  printf 'PROFILE_DIR=%s\n' "$(q "${PROFILE_DIR}")"
  cat <<'SBATCH_BODY'

export REPO_DIR MODEL PORT TP_SIZE GPU_MEM MAX_MODEL_LEN MAX_NUM_SEQS
export INPUT_LEN OUTPUT_LEN NUM_PROMPTS REQUEST_RATE MAX_CONCURRENCY NUM_WARMUPS
export TOKEN_ID_START TOKEN_ID_RANGE CONSTANT_PROMPT_TOKEN_ID BENCHMARK_EXTRA_ARGS
export PROFILE_WINDOWS PROFILE_WINDOW_SECONDS PROFILE_GAP_SECONDS
export PROFILE_INITIAL_DELAY_SECONDS PROFILE_TIMEOUT_SECONDS
export NSYS_CMD NSYS_CONTAINER_MOUNT NSYS_EXTRA_ARGS NSYS_CUDA_GRAPH_TRACE
export EXPORT_SQLITE CURRENT_ONLY USE_TRITON_BATCHED_SUMMARIES
export USE_FLATTENED_PREFILL_PAGE_TABLE_FA USE_FLATTENED_DECODE_PAGE_TABLE_FA
export BATCHED_SUMMARY_PRINT_LIMIT PATH_DEBUG_PRINT_LIMIT
export NANO_DSA_ENFORCE_EAGER NANO_DSA_CUDAGRAPH_MODE
export LOG_DIR RESULT_DIR PROFILE_DIR

if [[ -z "${PORT}" ]]; then
  PORT="$((20000 + (${SLURM_JOB_ID:-$$} % 30000)))"
  export PORT
fi

echo "job_id=${SLURM_JOB_ID:-unknown}"
echo "host=$(hostname)"
echo "repo=${REPO_DIR}"
echo "model=${MODEL}"
echo "base_port=${PORT}"
echo "baseline_image=${BASELINE_IMAGE}"
echo "current_image=${CURRENT_IMAGE}"
echo "profile_dir=${PROFILE_DIR}"

run_case() {
  local case_name=$1
  local image=$2
  local case_port=$3
  local case_log_dir="${LOG_DIR}/${case_name}"
  local case_result_dir="${RESULT_DIR}/${case_name}"
  local case_profile_dir="${PROFILE_DIR}/${case_name}"
  local server_log="${case_log_dir}/server.log"
  local controller_log="${case_log_dir}/profile_windows.log"
  local benchmark_log="${case_log_dir}/benchmark.log"
  local result_json="${case_result_dir}/fixed_io_result.json"
  local windows_json="${case_result_dir}/profile_windows.json"
  local report_base="${case_profile_dir}/${case_name}_nsys_windows"

  mkdir -p "${case_log_dir}" "${case_result_dir}" "${case_profile_dir}"
  echo "===== case=${case_name} image=${image} port=${case_port} ====="

  srun \
    --container-image="${image}" \
    --container-mounts="${REPO_DIR}:/workspace/vllm-src,${case_log_dir}:/logs,${case_result_dir}:/results,${case_profile_dir}:/profiles,/lustre:/lustre,/scratch:/scratch${NSYS_CONTAINER_MOUNT}" \
    bash -lc '
      set -euo pipefail

      cd /tmp
      unset PYTHONPATH
      export HOME="/tmp/${USER}/fixed-io-nsys-home-'${case_name}'"
      export XDG_CACHE_HOME="/tmp/${USER}/fixed-io-nsys-cache-'${case_name}'"
      export VLLM_USAGE_SOURCE="fixed-io-nsys-profile"
      export VLLM_WORKER_MULTIPROC_METHOD=spawn
      export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
      export VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=16
      export VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=1
      export VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA=1
      export VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ=1
      export VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE=4096
      export VLLM_NEMOTRON_H_DSA_USE_TRITON_BATCHED_SUMMARIES="${USE_TRITON_BATCHED_SUMMARIES}"
      export VLLM_NEMOTRON_H_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA="${USE_FLATTENED_PREFILL_PAGE_TABLE_FA}"
      export VLLM_NEMOTRON_H_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA="${USE_FLATTENED_DECODE_PAGE_TABLE_FA}"
      mkdir -p "${HOME}" "${XDG_CACHE_HOME}" /logs /results /profiles

      PYTHON_BIN="${PYTHON_BIN:-python3}"
      CASE_PROFILE_DIR="'${case_profile_dir}'"

      if [[ -f /etc/profile.d/modules.sh ]]; then
        # shellcheck source=/etc/profile.d/modules.sh
        source /etc/profile.d/modules.sh
      fi
      if command -v module >/dev/null 2>&1; then
        module load nsight-systems 2>/dev/null || \
          module load nsys 2>/dev/null || \
          module load cuda/13.0 2>/dev/null || \
          module load cuda 2>/dev/null || true
      fi

      if [[ -n "${NSYS_CMD}" ]]; then
        NSYS_BIN="${NSYS_CMD}"
      else
        NSYS_BIN="$(command -v nsys || true)"
      fi
      if [[ -z "${NSYS_BIN}" ]]; then
        for candidate in \
          /usr/local/cuda/bin/nsys \
          /usr/local/cuda-13.0/bin/nsys \
          /opt/nvidia/nsight-systems/*/target-linux-x64/nsys \
          /opt/nvidia/nsight-systems/*/bin/nsys; do
          if [[ -x "${candidate}" ]]; then
            NSYS_BIN="${candidate}"
            break
          fi
        done
      fi
      [[ -n "${NSYS_BIN}" && -x "${NSYS_BIN}" ]] || {
        echo "ERROR: nsys not found inside container on $(hostname)" >&2
        exit 1
      }

      echo "container_host=$(hostname)"
      echo "python_bin=${PYTHON_BIN}"
      echo "nsys_bin=${NSYS_BIN}"
      "${NSYS_BIN}" --version || true
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

      read -r -a BENCH_ARGS <<< "${BENCHMARK_EXTRA_ARGS}"
      read -r -a EXTRA_NSYS_ARGS <<< "${NSYS_EXTRA_ARGS}"
      GRAPH_ARGS=()
      if [[ "${NANO_DSA_ENFORCE_EAGER}" == "1" ]]; then
        GRAPH_ARGS+=(--enforce-eager)
      elif [[ -n "${NANO_DSA_CUDAGRAPH_MODE}" ]]; then
        GRAPH_ARGS+=(
          --compilation-config
          "{\"mode\":3,\"cudagraph_mode\":\"${NANO_DSA_CUDAGRAPH_MODE}\"}"
        )
      fi

      server_ready() {
        "${PYTHON_BIN}" - "'${case_port}'" <<PY
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

      "${NSYS_BIN}" profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cpuctxsw=none \
        --trace-fork-before-exec=true \
        --cuda-graph-trace="${NSYS_CUDA_GRAPH_TRACE}" \
        --capture-range=cudaProfilerApi \
        --capture-range-end=repeat \
        --force-overwrite=true \
        --output="/profiles/'${case_name}'_nsys_windows" \
        "${EXTRA_NSYS_ARGS[@]}" \
        "${PYTHON_BIN}" -m vllm.entrypoints.cli.main serve "'${MODEL}'" \
          --host 127.0.0.1 \
          --port "'${case_port}'" \
          --tensor-parallel-size "'${TP_SIZE}'" \
          --max-model-len "'${MAX_MODEL_LEN}'" \
          --gpu-memory-utilization "'${GPU_MEM}'" \
          --no-enable-log-requests \
          --trust-remote-code \
          --attention-backend FLASH_ATTN \
          --enable-expert-parallel \
          --model-loader-extra-config "{\"enable_multithread_load\":true,\"num_threads\":96}" \
          --max-num-seqs "'${MAX_NUM_SEQS}'" \
          --mamba-ssm-cache-dtype float32 \
          --no-enable-prefix-caching \
          --enable-chunked-prefill \
          "${GRAPH_ARGS[@]}" \
          --block-size 16 \
          --max-num-batched-tokens 8192 \
          --skip-tokenizer-init \
          --profiler-config.profiler cuda \
          > /logs/server.log 2>&1 &
      SERVER_PID=$!

      for _ in $(seq 1 900); do
        if server_ready; then
          break
        fi
        if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
          echo "vllm server exited before readiness" >&2
          tail -200 /logs/server.log >&2 || true
          exit 1
        fi
        sleep 2
      done

      server_ready || {
        echo "vllm server did not become ready" >&2
        tail -200 /logs/server.log >&2 || true
        exit 1
      }

      "${PYTHON_BIN}" /workspace/vllm-src/benchmarks/profile_window_controller.py \
        --base-url "http://127.0.0.1:'${case_port}'" \
        --windows "${PROFILE_WINDOWS}" \
        --window-seconds "${PROFILE_WINDOW_SECONDS}" \
        --gap-seconds "${PROFILE_GAP_SECONDS}" \
        --initial-delay-seconds "${PROFILE_INITIAL_DELAY_SECONDS}" \
        --timeout-s "${PROFILE_TIMEOUT_SECONDS}" \
        --output-json /results/profile_windows.json \
        > /logs/profile_windows.log 2>&1 &
      CONTROLLER_PID=$!

      CLIENT_ARGS=(
        /workspace/vllm-src/benchmarks/benchmark_fixed_io_serving.py
        --base-url "http://127.0.0.1:'${case_port}'"
        --input-len "'${INPUT_LEN}'"
        --output-len "'${OUTPUT_LEN}'"
        --num-prompts "'${NUM_PROMPTS}'"
        --request-rate "'${REQUEST_RATE}'"
        --num-warmups "'${NUM_WARMUPS}'"
        --token-id-start "'${TOKEN_ID_START}'"
        --token-id-range "'${TOKEN_ID_RANGE}'"
        --constant-prompt-token-id "'${CONSTANT_PROMPT_TOKEN_ID}'"
        --output-json /results/fixed_io_result.json
        --save-detailed
      )
      if [[ -n "'${MAX_CONCURRENCY}'" ]]; then
        CLIENT_ARGS+=(--max-concurrency "'${MAX_CONCURRENCY}'")
      fi
      CLIENT_ARGS+=("${BENCH_ARGS[@]}")

      "${PYTHON_BIN}" "${CLIENT_ARGS[@]}" > /logs/benchmark.log 2>&1
      wait "${CONTROLLER_PID}"
      stop_server
      trap - EXIT

      shopt -s nullglob
      REPORT_FILES=(/profiles/'${case_name}'_nsys_windows*.nsys-rep)
      if [[ "${EXPORT_SQLITE}" == "1" && "${#REPORT_FILES[@]}" -gt 0 ]]; then
        for report_file in "${REPORT_FILES[@]}"; do
          sqlite_file="${report_file%.nsys-rep}.sqlite"
          "${NSYS_BIN}" export \
            --type sqlite \
            --force-overwrite=true \
            --output="${sqlite_file}" \
            "${report_file}" \
            >> /logs/nsys_export.log 2>&1 || {
              echo "WARNING: nsys sqlite export failed for ${report_file}" >&2
              tail -100 /logs/nsys_export.log >&2 || true
            }
        done
      fi

      echo "case='${case_name}' result_json='${result_json}'"
      echo "case='${case_name}' windows_json='${windows_json}'"
      if [[ "${#REPORT_FILES[@]}" -eq 0 ]]; then
        echo "case='${case_name}' nsys_report_glob='${report_base}'*.nsys-rep missing"
      else
        for report_file in "${REPORT_FILES[@]}"; do
          echo "case='${case_name}' nsys_report=${report_file/#\/profiles/${CASE_PROFILE_DIR}}"
        done
      fi
      SQLITE_FILES=(/profiles/'${case_name}'_nsys_windows*.sqlite)
      if [[ "${#SQLITE_FILES[@]}" -gt 0 ]]; then
        for sqlite_file in "${SQLITE_FILES[@]}"; do
          echo "case='${case_name}' nsys_sqlite=${sqlite_file/#\/profiles/${CASE_PROFILE_DIR}}"
        done
      fi
    '
}

if [[ "${CURRENT_ONLY}" == "1" ]]; then
  run_case current "${CURRENT_IMAGE}" "${PORT}"
else
  run_case baseline "${BASELINE_IMAGE}" "${PORT}"
  run_case current "${CURRENT_IMAGE}" "$((PORT + 1))"
fi

echo "all cases complete"
SBATCH_BODY
} >"${SBATCH_FILE}"

chmod 755 "${SBATCH_FILE}"

printf 'Wrote sbatch launcher: %s\n' "${SBATCH_FILE}"
printf 'Run directory:          %s\n' "${RUN_DIR}"
printf 'Slurm log pattern:     %s\n' "${SLURM_LOG}"
printf 'Baseline image:        %s\n' "${BASELINE_IMAGE}"
printf 'Current image:         %s\n' "${CURRENT_IMAGE}"
if [[ -n "${NSYS_HOST_PATH}" ]]; then
  printf 'Host nsys path:        %s\n' "${NSYS_HOST_PATH}"
  printf 'Host nsys root:        %s\n' "${NSYS_HOST_ROOT}"
  printf 'Container nsys path:   %s\n' "${NSYS_CMD}"
fi
printf 'Profiles:              %s\n' "${PROFILE_DIR}"
printf 'Results:               %s\n' "${RESULT_DIR}"

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
  sleep 30
done

if command -v sacct >/dev/null 2>&1; then
  sacct -j "${JOB_ID}" --format=JobID,JobName%24,State,ExitCode,Elapsed
fi

if [[ "${CURRENT_ONLY}" != "1" && ! -f "${RESULT_DIR}/baseline/fixed_io_result.json" ]]; then
  printf 'Baseline result missing\n' >&2
  exit 1
fi
if [[ ! -f "${RESULT_DIR}/current/fixed_io_result.json" ]]; then
  printf 'Current result missing\n' >&2
  exit 1
fi
