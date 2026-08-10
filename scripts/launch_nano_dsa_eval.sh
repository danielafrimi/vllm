#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Launch Nemotron-H DSA evals from the current vLLM checkout.

Usage:
  scripts/launch_nano_dsa_eval.sh <benchmark> [options]

Benchmarks:
  ruler-128k-completions, ruler-128k, 128k
  ruler-1m-completions, ruler-1m, 1m
  mmlu-pro, ns_mmlu_pro
  aalcr, aa-lcr, ns_aa_lcr

Common options:
  --config-only        Run the deci-evals config dry run locally; do not submit.
  --no-sync           Skip "uv sync" in the deci-evals checkout.
  --image PATH        Use this vLLM .sqsh instead of auto-detecting one.
  --model-family NAME Model/checkpoint family: nano, nano-vanilla, super, or ultra.
                      Default: MODEL_FAMILY env var, otherwise nano.
  --deci-repo PATH    deci-evals checkout. Default: ~/mycode/deci-evals on Lustre.
  --secrets-file PATH Env file with HF_TOKEN, JUDGE_API_KEY,
                      DECI_INFERENCE_HUB_KEY, and DECI_BUILD_NVDEV_KEY.
  --out-root PATH     Result root. Default: this repo's outputs/evaluation_results.
  --print-only        Generate the sbatch launcher and print its path; do not run.
  -h, --help          Show this help.

Useful environment overrides:
  MODEL_FAMILY=nano TOPK=1024 QUERY_CHUNK_SIZE=4096
  MODEL_FAMILY=nano-vanilla
  MODEL_FAMILY=super TOPK=4096 SUPER_DSA_TUNING=qindexer-kl-init-fixed-131k-tuning
  USE_TRITON_BATCHED_SUMMARIES=1
  DSA_ATTENTION_CLASS=moonshot|refactored-efficient|refactored-pytorch
  DSA_ATTN_MODE=chunked_topk_sparse CHUNK_SIZE=16
  SHARE_CHUNK_TOPK=1 SHARE_TOPK_GROUP_SIZE=16 SHARE_TOPK_MODE=representative
  SHARE_TOPK_UNION_MAX_CHUNKS=2048 USE_SHARED_PREFILL_PAGE_TABLE_FA=1
  USE_PAGE_TABLE_FA=0 USE_PREFILL_PAGE_TABLE_FA=0 USE_FULL_ATTN_SHORT_SEQ=0
  DENSE_PREFILL_KV_THRESHOLD_TOKENS=<tokens>
  USE_FLATTENED_PREFILL_PAGE_TABLE_FA=1 USE_FLATTENED_DECODE_PAGE_TABLE_FA=1
  PACKED_TP2=1 PACKED_TP2_INSTANCES_PER_NODE=4 PACKED_TP2_PORT_BASE=8000
  PACKED_TP2_CUDA_GROUPS='0,1;2,3;4,5;6,7'
  EVAL_CONFIG=deci-super EVAL_TOKENIZER=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
  EVAL_TASK_INDEX=9 EVAL_PARALLELISM=64 EVAL_JUDGE_PARALLELISM=1
  EVAL_NUM_REPEATS=16 EVAL_MAX_NEW_TOKENS=1024 EVAL_RULER_MAX_SEQ_LENGTH=131072
  EVAL_RULER_TASK_INDEX=14
  EVAL_LIMIT_SAMPLES=4 EVAL_SUBTASKS=niah_single_1
  MAX_NUM_BATCHED_TOKENS=8192 PACKED_TP2_DRY_RUN=1
  ACCOUNT=nemotron_n4_compress EVAL_PARTITION=batch_long NUM_NODES=4 NUM_INSTANCES=16
  TP_SIZE=2 DP_SIZE=1 MAX_NUM_SEQS=8 GPU_MEM=0.90 WALLTIME=12:00:00
  ENFORCE_EAGER=1 USE_COMPILATION_CONFIG=0 for legacy attention
  USE_DSA_ENV=0 for vanilla Nano baselines

Examples:
  scripts/launch_nano_dsa_eval.sh ruler-128k-completions --config-only
  scripts/launch_nano_dsa_eval.sh ruler-128k-completions
  scripts/launch_nano_dsa_eval.sh ruler-1m-completions --secrets-file ~/secrets/deci-evals.env
  scripts/launch_nano_dsa_eval.sh aalcr --config-only

The script submits a small CPU launcher job. By default it launches packed TP2
Nano evals: four vLLM servers per 8-GPU node, using four nodes total. The
super/ultra families default to one TP8 vLLM server per node.
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

q() {
  printf '%q' "$1"
}

verify_vllm_image_contains_batched_summaries() {
  local image="$1"
  local package_root=""
  local root
  local summary_text
  local nemotron_text
  local dsa_attention_text
  local image_nemotron_hash
  local image_dsa_attention_hash
  local image_summary_hash
  local local_nemotron_hash
  local local_dsa_attention_hash
  local local_summary_hash

  command -v unsquashfs >/dev/null 2>&1 || {
    die "unsquashfs not found; cannot verify vLLM image contents"
  }

  for root in \
    usr/local/lib/python3.12/dist-packages \
    usr/local/lib/python3.12/site-packages
  do
    if unsquashfs -cat "${image}" \
      "${root}/vllm/model_executor/models/nemotron_h_dsa_triton_summaries.py" \
      >/dev/null 2>&1
    then
      package_root="${root}"
      break
    fi
  done

  [[ -n "${package_root}" ]] || {
    die "image does not contain vllm/model_executor/models/nemotron_h_dsa_triton_summaries.py; rebuild the .sqsh from this worktree"
  }

  summary_text="$(
    unsquashfs -cat "${image}" \
      "${package_root}/vllm/model_executor/models/nemotron_h_dsa_triton_summaries.py"
  )"
  grep -q "def dsa_block_summaries_triton" <<<"${summary_text}" \
    || die "image contains the summary module but not dsa_block_summaries_triton"

  nemotron_text="$(
    unsquashfs -cat "${image}" \
      "${package_root}/vllm/model_executor/models/nemotron_h.py"
  )"
  dsa_attention_text="$(
    unsquashfs -cat "${image}" \
      "${package_root}/vllm/model_executor/models/nemotron_h_dsa_attention_legacy.py"
  )"
  grep -q "NemotronHDSALegacyAttention" <<<"${dsa_attention_text}" \
    || die "image nemotron_h_dsa_attention_legacy.py is missing the legacy DSA class"
  grep -q "VLLM_NEMOTRON_H_DSA_USE_TRITON_BATCHED_SUMMARIES" \
    <<<"${dsa_attention_text}" \
    || die "image nemotron_h_dsa_attention_legacy.py is missing the batched-summary env guard"
  grep -q "VLLM_NEMOTRON_H_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA" \
    <<<"${dsa_attention_text}" \
    || die "image nemotron_h_dsa_attention_legacy.py is missing the flattened decode env guard"
  grep -q "def _forward_dsa_chunked_flattened_decode_page_table_fa_sequence" \
    <<<"${dsa_attention_text}" \
    || die "image nemotron_h_dsa_attention_legacy.py is missing the flattened decode implementation"
  grep -q "flattened decode page-table FA" <<<"${dsa_attention_text}" \
    || die "image nemotron_h_dsa_attention_legacy.py is missing the flattened decode runtime marker"
  ! grep -q "summary_cache\\|_dsa_summary\\|cached_batched" \
    <<<"${dsa_attention_text}" \
    || die "image nemotron_h_dsa_attention_legacy.py still contains summary-cache code"

  image_nemotron_hash="$(
    unsquashfs -cat "${image}" \
      "${package_root}/vllm/model_executor/models/nemotron_h.py" \
      | sha256sum \
      | awk '{print $1}'
  )"
  local_nemotron_hash="$(
    sha256sum "${VLLM_REPO}/vllm/model_executor/models/nemotron_h.py" \
      | awk '{print $1}'
  )"
  [[ "${image_nemotron_hash}" == "${local_nemotron_hash}" ]] || {
    die "image nemotron_h.py hash does not match the current worktree"
  }

  image_dsa_attention_hash="$(
    unsquashfs -cat "${image}" \
      "${package_root}/vllm/model_executor/models/nemotron_h_dsa_attention_legacy.py" \
      | sha256sum \
      | awk '{print $1}'
  )"
  local_dsa_attention_hash="$(
    sha256sum "${VLLM_REPO}/vllm/model_executor/models/nemotron_h_dsa_attention_legacy.py" \
      | awk '{print $1}'
  )"
  [[ "${image_dsa_attention_hash}" == "${local_dsa_attention_hash}" ]] || {
    die "image nemotron_h_dsa_attention_legacy.py hash does not match the current worktree"
  }

  image_summary_hash="$(
    unsquashfs -cat "${image}" \
      "${package_root}/vllm/model_executor/models/nemotron_h_dsa_triton_summaries.py" \
      | sha256sum \
      | awk '{print $1}'
  )"
  local_summary_hash="$(
    sha256sum \
      "${VLLM_REPO}/vllm/model_executor/models/nemotron_h_dsa_triton_summaries.py" \
      | awk '{print $1}'
  )"
  [[ "${image_summary_hash}" == "${local_summary_hash}" ]] || {
    die "image nemotron_h_dsa_triton_summaries.py hash does not match the current worktree"
  }

  printf 'Verified image contains current DSA code: %s (%s)\n' \
    "${image}" "${package_root}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

benchmark=""
CONFIG_ONLY=0
SYNC_DECI=1
PRINT_ONLY=0
AUTO_DETECTED_IMAGE=0
TOPK_WAS_SET=0
EVAL_TOKENIZER_WAS_SET=0
[[ -n "${TOPK+x}" ]] && TOPK_WAS_SET=1
[[ -n "${EVAL_TOKENIZER+x}" ]] && EVAL_TOKENIZER_WAS_SET=1

DECI_REPO="${DECI_REPO:-/lustre/fsw/portfolios/coreai/users/${USER}/mycode/deci-evals}"
SECRETS_FILE="${SECRETS_FILE:-/lustre/fsw/portfolios/coreai/users/${USER}/secrets/deci-evals.env}"
VLLM_IMAGE="${VLLM_IMAGE:-}"
PROXY_IMAGE="${PROXY_IMAGE:-}"
OUT_ROOT="${OUT_ROOT:-${VLLM_REPO}/outputs/evaluation_results/prefill_decode_models/no-disagg-vllm_current}"
MODEL_FAMILY="${MODEL_FAMILY:-nano}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
      [[ $# -ge 2 ]] || die "--benchmark requires a value"
      benchmark="$2"
      shift 2
      ;;
    --config-only|--dry-run)
      CONFIG_ONLY=1
      shift
      ;;
    --no-sync|--skip-sync)
      SYNC_DECI=0
      shift
      ;;
    --image)
      [[ $# -ge 2 ]] || die "--image requires a value"
      VLLM_IMAGE="$2"
      shift 2
      ;;
    --model-family)
      [[ $# -ge 2 ]] || die "--model-family requires a value"
      MODEL_FAMILY="$2"
      shift 2
      ;;
    --deci-repo)
      [[ $# -ge 2 ]] || die "--deci-repo requires a value"
      DECI_REPO="$2"
      shift 2
      ;;
    --secrets-file)
      [[ $# -ge 2 ]] || die "--secrets-file requires a value"
      SECRETS_FILE="$2"
      shift 2
      ;;
    --out-root)
      [[ $# -ge 2 ]] || die "--out-root requires a value"
      OUT_ROOT="$2"
      shift 2
      ;;
    --print-only)
      PRINT_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -z "${benchmark}" ]]; then
        benchmark="$1"
      else
        die "unexpected argument: $1"
      fi
      shift
      ;;
  esac
done

[[ -n "${benchmark}" ]] || {
  usage >&2
  exit 2
}

DEFAULT_EVAL_RULER_MAX_SEQ_LENGTH=""
DEFAULT_EVAL_RULER_TASK_INDEX=""
case "${benchmark,,}" in
  ruler-128k-completions|ruler128k-completions|ruler-128k-completion|ruler128k-completion|ruler-128k|ruler128k|128k)
    TASKS="${TASKS:-ruler-128k-completions}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-131200}"
    EVAL_CONFIG="${EVAL_CONFIG:-deci-super}"
    TOPK="${TOPK:-1024}"
    DEFAULT_EVAL_RULER_MAX_SEQ_LENGTH="131072"
    DEFAULT_EVAL_RULER_TASK_INDEX="14"
    BENCH_SLUG="ruler128k-completions"
    JOB_BENCH="r128kc"
    ;;
  ruler-1m-completions|ruler1m-completions|ruler-1m-completion|ruler1m-completion|ruler-1m|ruler1m|1m|ruler-1million|ruler-1-million)
    TASKS="${TASKS:-ruler-1m-completions}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-1048704}"
    EVAL_CONFIG="${EVAL_CONFIG:-deci-super}"
    TOPK="${TOPK:-2048}"
    DEFAULT_EVAL_RULER_MAX_SEQ_LENGTH="1048576"
    DEFAULT_EVAL_RULER_TASK_INDEX="17"
    BENCH_SLUG="ruler1m-completions"
    JOB_BENCH="r1mc"
    ;;
  mmlu-pro|mmlu_pro|mmlupro|ns-mmlu-pro|ns_mmlu_pro)
    TASKS="${TASKS:-ns_mmlu_pro}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
    EVAL_CONFIG="${EVAL_CONFIG:-deci-super}"
    TOPK="${TOPK:-1024}"
    BENCH_SLUG="mmlu_pro"
    JOB_BENCH="mmlu"
    ;;
  aalcr|aa-lcr|aa_lcr|ns-aa-lcr|ns_aa_lcr)
    TASKS="${TASKS:-ns_aa_lcr}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
    MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-131072}"
    EVAL_CONFIG="${EVAL_CONFIG:-deci-super}"
    TOPK="${TOPK:-2048}"
    EVAL_TOKENIZER="${EVAL_TOKENIZER:-nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
    EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-0.99999}"
    EVAL_TOP_P="${EVAL_TOP_P:-0.99999}"
    EVAL_TASK_INDEX="${EVAL_TASK_INDEX:-9}"
    BENCH_SLUG="aalcr"
    JOB_BENCH="aalcr"
    ;;
  *)
    die "unsupported benchmark '${benchmark}'. Use ruler-128k-completions, ruler-1m-completions, mmlu-pro, or aalcr."
    ;;
esac

MODEL_FAMILY="${MODEL_FAMILY,,}"
MODEL_CHECKPOINT_ROOT="${MODEL_CHECKPOINT_ROOT:-/lustre/fsw/portfolios/coreai/users/mdabbah/deci/puzzletron.worktrees/attention_hash/outputs/dsa_indexer_checkpoints}"
VANILLA_NANO_MODEL_DIR="${VANILLA_NANO_MODEL_DIR:-/scratch/fsw/portfolios/coreai/projects/coreai_nvfm_llm/models/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16}"
SUPER_DSA_TUNING="${SUPER_DSA_TUNING:-qindexer-kl-init-fixed-131k-tuning}"
if [[ "${MODEL_FAMILY}" == "super" && "${TOPK_WAS_SET}" != "1" ]]; then
  TOPK="${SUPER_TOPK:-4096}"
fi

case "${MODEL_FAMILY}" in
  nano)
    MODEL_LABEL="nano"
    DEFAULT_MODEL_DIR="${MODEL_CHECKPOINT_ROOT}/nano-dsa-16chunk_size-${TOPK}chunks"
    DEFAULT_PACKED_TP2=1
    DEFAULT_NONPACKED_NUM_INSTANCES=4
    DEFAULT_NONPACKED_TP_SIZE=8
    DEFAULT_NONPACKED_DP_SIZE=1
    DEFAULT_MAX_NUM_SEQS=8
    DEFAULT_NONPACKED_MAX_NUM_BATCHED_TOKENS=""
    DEFAULT_TRUST_REMOTE_CODE=0
    DEFAULT_REASONING_PARSER=""
    DEFAULT_REASONING_PARSER_PLUGIN=""
    DEFAULT_USE_DSA_ENV=1
    ;;
  nano-vanilla|vanilla-nano|vanilla)
    MODEL_FAMILY="nano-vanilla"
    MODEL_LABEL="nano-vanilla"
    DEFAULT_MODEL_DIR="${VANILLA_NANO_MODEL_DIR}"
    DEFAULT_PACKED_TP2=0
    DEFAULT_NONPACKED_NUM_INSTANCES=1
    DEFAULT_NONPACKED_TP_SIZE=1
    DEFAULT_NONPACKED_DP_SIZE=8
    DEFAULT_MAX_NUM_SEQS=16
    DEFAULT_NONPACKED_MAX_NUM_BATCHED_TOKENS=131072
    DEFAULT_TRUST_REMOTE_CODE=1
    DEFAULT_REASONING_PARSER="nano_v3"
    DEFAULT_REASONING_PARSER_PLUGIN="/checkpoint/nano_v3_reasoning_parser.py"
    DEFAULT_USE_DSA_ENV=0
    ;;
  super)
    MODEL_LABEL="super"
    DEFAULT_MODEL_DIR="${MODEL_CHECKPOINT_ROOT}/${SUPER_DSA_TUNING}/ultra-chunked-dsa-16x${TOPK}-vllm"
    DEFAULT_PACKED_TP2=0
    DEFAULT_NONPACKED_NUM_INSTANCES=1
    DEFAULT_NONPACKED_TP_SIZE=8
    DEFAULT_NONPACKED_DP_SIZE=1
    DEFAULT_MAX_NUM_SEQS=1
    DEFAULT_NONPACKED_MAX_NUM_BATCHED_TOKENS=131072
    DEFAULT_TRUST_REMOTE_CODE=1
    DEFAULT_REASONING_PARSER="ultra_v3"
    DEFAULT_REASONING_PARSER_PLUGIN="/checkpoint/ultra_v3_reasoning_parser.py"
    DEFAULT_USE_DSA_ENV=1
    ;;
  ultra)
    MODEL_LABEL="ultra"
    DEFAULT_MODEL_DIR="${MODEL_CHECKPOINT_ROOT}/ultra-chunked-dsa-16x${TOPK}-vllm"
    DEFAULT_PACKED_TP2=0
    DEFAULT_NONPACKED_NUM_INSTANCES=1
    DEFAULT_NONPACKED_TP_SIZE=8
    DEFAULT_NONPACKED_DP_SIZE=1
    DEFAULT_MAX_NUM_SEQS=1
    DEFAULT_NONPACKED_MAX_NUM_BATCHED_TOKENS=131072
    DEFAULT_TRUST_REMOTE_CODE=1
    DEFAULT_REASONING_PARSER="ultra_v3"
    DEFAULT_REASONING_PARSER_PLUGIN="/checkpoint/ultra_v3_reasoning_parser.py"
    DEFAULT_USE_DSA_ENV=1
    ;;
  *)
    die "unsupported MODEL_FAMILY='${MODEL_FAMILY}'. Use nano, nano-vanilla, super, or ultra."
    ;;
esac

if [[ "${MODEL_FAMILY}" != "nano" && "${EVAL_TOKENIZER_WAS_SET}" != "1" ]]; then
  EVAL_TOKENIZER=""
fi

cd "${VLLM_REPO}"
VLLM_COMMIT="$(git rev-parse --short=10 HEAD 2>/dev/null || printf unknown)"

if [[ -z "${VLLM_IMAGE}" ]]; then
  AUTO_DETECTED_IMAGE=1
  VLLM_IMAGE="$(
    find "${VLLM_REPO}/outputs/containers" -maxdepth 1 -type f \
      -name "vllm-v0.22.0-current-overlay-${VLLM_COMMIT}-*.sqsh" \
      -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr \
      | sed -n '1s/^[^ ]* //p'
  )"
fi

[[ -d "${DECI_REPO}" ]] || die "deci-evals checkout not found: ${DECI_REPO}"
[[ -n "${VLLM_IMAGE}" ]] || die "no current-code .sqsh found for commit ${VLLM_COMMIT}; pass --image PATH"
[[ -f "${VLLM_IMAGE}" ]] || die "vLLM image not found: ${VLLM_IMAGE}"
VLLM_IMAGE="$(realpath "${VLLM_IMAGE}")"
PROXY_IMAGE="${PROXY_IMAGE:-${VLLM_IMAGE}}"
verify_vllm_image_contains_batched_summaries "${VLLM_IMAGE}"

if [[ "${CONFIG_ONLY}" != "1" ]]; then
  if [[ -f "${SECRETS_FILE}" ]]; then
    # shellcheck source=/dev/null
    set -a
    source "${SECRETS_FILE}"
    set +a
  fi
  if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
  fi

  missing=()
  for name in HF_TOKEN JUDGE_API_KEY DECI_INFERENCE_HUB_KEY DECI_BUILD_NVDEV_KEY; do
    if [[ -z "${!name:-}" ]]; then
      missing+=("${name}")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    die "missing required secret env vars: ${missing[*]}; pass --secrets-file PATH or export them"
  fi
fi

UV_BIN="${UV_BIN:-$(command -v uv || true)}"
if [[ "${SYNC_DECI}" == "1" ]]; then
  [[ -n "${UV_BIN}" ]] || die "uv not found; pass --no-sync if ${DECI_REPO} is already synced"
  export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/${USER}/uv-cache}"
  export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${DECI_REPO}/.uv-python}"
  mkdir -p "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"
  printf 'Syncing deci-evals: %s\n' "${DECI_REPO}"
  (cd "${DECI_REPO}" && "${UV_BIN}" sync)
fi
[[ -x "${DECI_REPO}/.venv/bin/evaluate" ]] || die "missing ${DECI_REPO}/.venv/bin/evaluate; run uv sync"

QUERY_CHUNK_SIZE="${QUERY_CHUNK_SIZE:-4096}"
USE_TRITON_SCORING="${USE_TRITON_SCORING:-1}"
USE_TRITON_BATCHED_SUMMARIES="${USE_TRITON_BATCHED_SUMMARIES:-1}"
DSA_PROVIDER_CLASS="${DSA_PROVIDER_CLASS:-efficient}"
DSA_FORCE_KERNEL_BLOCK_SIZE="${DSA_FORCE_KERNEL_BLOCK_SIZE:-16}"
USE_PAGE_TABLE_FA="${USE_PAGE_TABLE_FA:-1}"
USE_PREFILL_PAGE_TABLE_FA="${USE_PREFILL_PAGE_TABLE_FA:-1}"
USE_FULL_ATTN_SHORT_SEQ="${USE_FULL_ATTN_SHORT_SEQ:-1}"
USE_FLATTENED_PREFILL_PAGE_TABLE_FA="${USE_FLATTENED_PREFILL_PAGE_TABLE_FA:-1}"
USE_FLATTENED_DECODE_PAGE_TABLE_FA="${USE_FLATTENED_DECODE_PAGE_TABLE_FA:-1}"
DENSE_PREFILL_KV_THRESHOLD_TOKENS="${DENSE_PREFILL_KV_THRESHOLD_TOKENS:-}"
DSA_ATTENTION_CLASS="${DSA_ATTENTION_CLASS:-}"
DSA_ATTENTION_MODULE="${DSA_ATTENTION_MODULE:-}"
DSA_ATTN_MODE="${DSA_ATTN_MODE:-}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
USE_FLASH_TOPK="${USE_FLASH_TOPK:-}"
SHARE_CHUNK_TOPK="${SHARE_CHUNK_TOPK:-}"
SHARE_TOPK_GROUP_SIZE="${SHARE_TOPK_GROUP_SIZE:-}"
SHARE_TOPK_MODE="${SHARE_TOPK_MODE:-}"
SHARE_TOPK_UNION_MAX_CHUNKS="${SHARE_TOPK_UNION_MAX_CHUNKS:-}"
USE_SHARED_PREFILL_PAGE_TABLE_FA="${USE_SHARED_PREFILL_PAGE_TABLE_FA:-}"
USE_UNION_PREFILL_KERNEL="${USE_UNION_PREFILL_KERNEL:-}"
USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA="${USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA:-}"
UNION_CHUNKS_PER_ITER="${UNION_CHUNKS_PER_ITER:-}"
USE_SUMMARY_CACHE="${USE_SUMMARY_CACHE:-}"
SUMMARY_CACHE_MAX_BLOCKS="${SUMMARY_CACHE_MAX_BLOCKS:-}"
USE_DSA_ENV="${USE_DSA_ENV:-${DEFAULT_USE_DSA_ENV}}"
PACKED_TP2="${PACKED_TP2:-${DEFAULT_PACKED_TP2}}"
PACKED_TP2_INSTANCES_PER_NODE="${PACKED_TP2_INSTANCES_PER_NODE:-4}"
PACKED_TP2_PORT_BASE="${PACKED_TP2_PORT_BASE:-8000}"
PACKED_TP2_CUDA_GROUPS="${PACKED_TP2_CUDA_GROUPS:-0,1;2,3;4,5;6,7}"
PACKED_TP2_DRY_RUN="${PACKED_TP2_DRY_RUN:-0}"
ACCOUNT="${ACCOUNT:-nemotron_n4_compress}"
EVAL_PARTITION="${EVAL_PARTITION:-batch_long}"
if [[ "${PACKED_TP2}" == "1" ]]; then
  NUM_NODES="${NUM_NODES:-4}"
  NUM_INSTANCES="${NUM_INSTANCES:-$((NUM_NODES * PACKED_TP2_INSTANCES_PER_NODE))}"
else
  NUM_INSTANCES="${NUM_INSTANCES:-${DEFAULT_NONPACKED_NUM_INSTANCES}}"
  NUM_NODES="${NUM_NODES:-${NUM_INSTANCES}}"
fi
WALLTIME="${WALLTIME:-12:00:00}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${DEFAULT_MAX_NUM_SEQS}}"
if [[ "${PACKED_TP2}" == "1" ]]; then
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
else
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${DEFAULT_NONPACKED_MAX_NUM_BATCHED_TOKENS}}"
fi
EVAL_TOKENIZER="${EVAL_TOKENIZER:-}"
EVAL_TOKENIZER_BACKEND="${EVAL_TOKENIZER_BACKEND:-hf}"
EVAL_TEMPERATURE="${EVAL_TEMPERATURE:-}"
EVAL_TOP_P="${EVAL_TOP_P:-}"
EVAL_PARALLELISM="${EVAL_PARALLELISM:-}"
EVAL_JUDGE_PARALLELISM="${EVAL_JUDGE_PARALLELISM:-}"
EVAL_NUM_REPEATS="${EVAL_NUM_REPEATS:-}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-}"
EVAL_RULER_MAX_SEQ_LENGTH="${EVAL_RULER_MAX_SEQ_LENGTH:-${DEFAULT_EVAL_RULER_MAX_SEQ_LENGTH}}"
EVAL_TASK_INDEX="${EVAL_TASK_INDEX:-}"
EVAL_RULER_TASK_INDEX="${EVAL_RULER_TASK_INDEX:-${EVAL_TASK_INDEX:-${DEFAULT_EVAL_RULER_TASK_INDEX}}}"
EVAL_LIMIT_SAMPLES="${EVAL_LIMIT_SAMPLES:-}"
EVAL_SUBTASKS="${EVAL_SUBTASKS:-}"
if [[ "${PACKED_TP2}" == "1" ]]; then
  TP_SIZE="${TP_SIZE:-2}"
  DP_SIZE="${DP_SIZE:-1}"
else
  TP_SIZE="${TP_SIZE:-${DEFAULT_NONPACKED_TP_SIZE}}"
  DP_SIZE="${DP_SIZE:-${DEFAULT_NONPACKED_DP_SIZE}}"
fi
if [[ "${PACKED_TP2}" == "1" ]]; then
  GPU_MEM="${GPU_MEM:-0.90}"
else
  GPU_MEM="${GPU_MEM:-0.85}"
fi
case "${DSA_ATTENTION_CLASS}" in
  ""|moonshot|vanilla|legacy|*NemotronHDSALegacyAttention)
    DEFAULT_ENFORCE_EAGER=1
    DEFAULT_USE_COMPILATION_CONFIG=0
    ;;
  *)
    DEFAULT_ENFORCE_EAGER=0
    DEFAULT_USE_COMPILATION_CONFIG=1
    ;;
esac
ENFORCE_EAGER="${ENFORCE_EAGER:-${DEFAULT_ENFORCE_EAGER}}"
USE_COMPILATION_CONFIG="${USE_COMPILATION_CONFIG:-${DEFAULT_USE_COMPILATION_CONFIG}}"
if [[ "${ENFORCE_EAGER}" != "0" && "${ENFORCE_EAGER}" != "1" ]]; then
  die "ENFORCE_EAGER must be 0 or 1"
fi
if [[ "${USE_COMPILATION_CONFIG}" != "0" && "${USE_COMPILATION_CONFIG}" != "1" ]]; then
  die "USE_COMPILATION_CONFIG must be 0 or 1"
fi
for bool_name in \
  USE_PAGE_TABLE_FA \
  USE_PREFILL_PAGE_TABLE_FA \
  USE_FULL_ATTN_SHORT_SEQ \
  USE_FLATTENED_PREFILL_PAGE_TABLE_FA \
  USE_FLATTENED_DECODE_PAGE_TABLE_FA \
  USE_TRITON_SCORING \
  USE_TRITON_BATCHED_SUMMARIES
do
  bool_value="${!bool_name}"
  if [[ "${bool_value}" != "0" && "${bool_value}" != "1" ]]; then
    die "${bool_name} must be 0 or 1"
  fi
done
MODEL_DIR="${MODEL_DIR:-${DEFAULT_MODEL_DIR}}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-${DEFAULT_TRUST_REMOTE_CODE}}"
REASONING_PARSER="${REASONING_PARSER:-${DEFAULT_REASONING_PARSER}}"
REASONING_PARSER_PLUGIN="${REASONING_PARSER_PLUGIN:-${DEFAULT_REASONING_PARSER_PLUGIN}}"
[[ -d "${MODEL_DIR}" ]] || die "model checkpoint not found: ${MODEL_DIR}"
if [[ -n "${REASONING_PARSER_PLUGIN}" && "${REASONING_PARSER_PLUGIN}" == /checkpoint/* ]]; then
  parser_file="${REASONING_PARSER_PLUGIN#/checkpoint/}"
  [[ -f "${MODEL_DIR}/${parser_file}" ]] \
    || die "checkpoint is missing reasoning parser ${parser_file}: ${MODEL_DIR}"
fi
if [[ "${PACKED_TP2}" == "1" ]]; then
  if [[ "${USE_DSA_ENV}" == "1" ]]; then
    MODEL_DESC="${MODEL_LABEL}-dsa-16x${TOPK}"
  else
    MODEL_DESC="${MODEL_LABEL}"
  fi
  OUT_SUFFIX="${OUT_SUFFIX:-${MODEL_DESC}-current-${VLLM_COMMIT}-${BENCH_SLUG}-q${QUERY_CHUNK_SIZE}-packed-tp${TP_SIZE}-n${NUM_NODES}i${NUM_INSTANCES}}"
  DIRTY_TAG="${DIRTY_TAG:-${OUT_SUFFIX}}"
else
  if [[ "${USE_DSA_ENV}" == "1" ]]; then
    MODEL_DESC="${MODEL_LABEL}-dsa-16x${TOPK}"
  else
    MODEL_DESC="${MODEL_LABEL}"
  fi
  OUT_SUFFIX="${OUT_SUFFIX:-${MODEL_DESC}-current-${VLLM_COMMIT}-${BENCH_SLUG}-q${QUERY_CHUNK_SIZE}-tp${TP_SIZE}dp${DP_SIZE}}"
  DIRTY_TAG="${DIRTY_TAG:-${OUT_SUFFIX}-n${NUM_INSTANCES}}"
fi

if [[ "${USE_DSA_ENV}" != "0" && "${USE_DSA_ENV}" != "1" ]]; then
  die "USE_DSA_ENV must be 0 or 1"
fi

if [[ "${PACKED_TP2}" == "1" ]]; then
  [[ "${USE_DSA_ENV}" == "1" ]] || die "PACKED_TP2=1 is only supported for DSA evals"
  [[ "${TP_SIZE}" == "2" ]] || die "PACKED_TP2=1 requires TP_SIZE=2"
  [[ "${DP_SIZE}" == "1" ]] || die "PACKED_TP2=1 requires DP_SIZE=1"
  [[ "${NUM_INSTANCES}" -eq $((NUM_NODES * PACKED_TP2_INSTANCES_PER_NODE)) ]] \
    || die "PACKED_TP2=1 requires NUM_INSTANCES=NUM_NODES*PACKED_TP2_INSTANCES_PER_NODE"
fi

LAUNCHER_DIR="${VLLM_REPO}/outputs/eval_launchers"
LOG_DIR="${VLLM_REPO}/logs"
mkdir -p "${LAUNCHER_DIR}" "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ "${USE_DSA_ENV}" == "1" ]]; then
  RUN_KIND="dsa"
else
  RUN_KIND="vanilla"
fi
LAUNCHER="${LAUNCHER_DIR}/launch_${MODEL_LABEL}_${RUN_KIND}_${BENCH_SLUG}_${VLLM_COMMIT}_${STAMP}.sbatch"

{
  printf '#!/usr/bin/env bash\n'
  printf '#SBATCH --job-name=%s-%s-%s\n' "${MODEL_LABEL}" "${RUN_KIND}" "${JOB_BENCH}"
  printf '#SBATCH --account=%s\n' "${ACCOUNT}"
  printf '#SBATCH --partition=cpu\n'
  printf '#SBATCH --time=01:00:00\n'
  printf '#SBATCH --nodes=1\n'
  printf '#SBATCH --ntasks=1\n'
  printf '#SBATCH --output=%s/%s_%s_%s_%s_%%j.log\n' "${LOG_DIR}" "${MODEL_LABEL}" "${RUN_KIND}" "${BENCH_SLUG}" "${VLLM_COMMIT}"
  printf '#SBATCH --error=%s/%s_%s_%s_%s_%%j.log\n' "${LOG_DIR}" "${MODEL_LABEL}" "${RUN_KIND}" "${BENCH_SLUG}" "${VLLM_COMMIT}"
  printf '\nset -euo pipefail\n\n'
  printf 'VLLM_REPO=%s\n' "$(q "${VLLM_REPO}")"
  printf 'DECI_REPO=%s\n' "$(q "${DECI_REPO}")"
  printf 'SECRETS_FILE=%s\n' "$(q "${SECRETS_FILE}")"
  printf 'VLLM_IMAGE=%s\n' "$(q "${VLLM_IMAGE}")"
  printf 'PROXY_IMAGE=%s\n' "$(q "${PROXY_IMAGE}")"
  printf 'VLLM_COMMIT=%s\n' "$(q "${VLLM_COMMIT}")"
  printf 'MODEL_FAMILY=%s\n' "$(q "${MODEL_FAMILY}")"
  printf 'MODEL_LABEL=%s\n' "$(q "${MODEL_LABEL}")"
  printf 'TASKS=%s\n' "$(q "${TASKS}")"
  printf 'TOPK=%s\n' "$(q "${TOPK}")"
  printf 'QUERY_CHUNK_SIZE=%s\n' "$(q "${QUERY_CHUNK_SIZE}")"
  printf 'USE_TRITON_SCORING=%s\n' "$(q "${USE_TRITON_SCORING}")"
  printf 'USE_TRITON_BATCHED_SUMMARIES=%s\n' "$(q "${USE_TRITON_BATCHED_SUMMARIES}")"
  printf 'DSA_PROVIDER_CLASS=%s\n' "$(q "${DSA_PROVIDER_CLASS}")"
  printf 'DSA_FORCE_KERNEL_BLOCK_SIZE=%s\n' "$(q "${DSA_FORCE_KERNEL_BLOCK_SIZE}")"
  printf 'USE_PAGE_TABLE_FA=%s\n' "$(q "${USE_PAGE_TABLE_FA}")"
  printf 'USE_PREFILL_PAGE_TABLE_FA=%s\n' "$(q "${USE_PREFILL_PAGE_TABLE_FA}")"
  printf 'USE_FULL_ATTN_SHORT_SEQ=%s\n' "$(q "${USE_FULL_ATTN_SHORT_SEQ}")"
  printf 'USE_FLATTENED_PREFILL_PAGE_TABLE_FA=%s\n' "$(q "${USE_FLATTENED_PREFILL_PAGE_TABLE_FA}")"
  printf 'USE_FLATTENED_DECODE_PAGE_TABLE_FA=%s\n' "$(q "${USE_FLATTENED_DECODE_PAGE_TABLE_FA}")"
  printf 'DENSE_PREFILL_KV_THRESHOLD_TOKENS=%s\n' "$(q "${DENSE_PREFILL_KV_THRESHOLD_TOKENS}")"
  printf 'DSA_ATTENTION_CLASS=%s\n' "$(q "${DSA_ATTENTION_CLASS}")"
  printf 'DSA_ATTENTION_MODULE=%s\n' "$(q "${DSA_ATTENTION_MODULE}")"
  printf 'DSA_ATTN_MODE=%s\n' "$(q "${DSA_ATTN_MODE}")"
  printf 'CHUNK_SIZE=%s\n' "$(q "${CHUNK_SIZE}")"
  printf 'USE_FLASH_TOPK=%s\n' "$(q "${USE_FLASH_TOPK}")"
  printf 'SHARE_CHUNK_TOPK=%s\n' "$(q "${SHARE_CHUNK_TOPK}")"
  printf 'SHARE_TOPK_GROUP_SIZE=%s\n' "$(q "${SHARE_TOPK_GROUP_SIZE}")"
  printf 'SHARE_TOPK_MODE=%s\n' "$(q "${SHARE_TOPK_MODE}")"
  printf 'SHARE_TOPK_UNION_MAX_CHUNKS=%s\n' "$(q "${SHARE_TOPK_UNION_MAX_CHUNKS}")"
  printf 'USE_SHARED_PREFILL_PAGE_TABLE_FA=%s\n' "$(q "${USE_SHARED_PREFILL_PAGE_TABLE_FA}")"
  printf 'USE_UNION_PREFILL_KERNEL=%s\n' "$(q "${USE_UNION_PREFILL_KERNEL}")"
  printf 'USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA=%s\n' "$(q "${USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA}")"
  printf 'UNION_CHUNKS_PER_ITER=%s\n' "$(q "${UNION_CHUNKS_PER_ITER}")"
  printf 'USE_SUMMARY_CACHE=%s\n' "$(q "${USE_SUMMARY_CACHE}")"
  printf 'SUMMARY_CACHE_MAX_BLOCKS=%s\n' "$(q "${SUMMARY_CACHE_MAX_BLOCKS}")"
  printf 'USE_DSA_ENV=%s\n' "$(q "${USE_DSA_ENV}")"
  printf 'PACKED_TP2=%s\n' "$(q "${PACKED_TP2}")"
  printf 'PACKED_TP2_INSTANCES_PER_NODE=%s\n' "$(q "${PACKED_TP2_INSTANCES_PER_NODE}")"
  printf 'PACKED_TP2_PORT_BASE=%s\n' "$(q "${PACKED_TP2_PORT_BASE}")"
  printf 'PACKED_TP2_CUDA_GROUPS=%s\n' "$(q "${PACKED_TP2_CUDA_GROUPS}")"
  printf 'PACKED_TP2_DRY_RUN=%s\n' "$(q "${PACKED_TP2_DRY_RUN}")"
  printf 'ACCOUNT=%s\n' "$(q "${ACCOUNT}")"
  printf 'EVAL_PARTITION=%s\n' "$(q "${EVAL_PARTITION}")"
  printf 'NUM_INSTANCES=%s\n' "$(q "${NUM_INSTANCES}")"
  printf 'NUM_NODES=%s\n' "$(q "${NUM_NODES}")"
  printf 'WALLTIME=%s\n' "$(q "${WALLTIME}")"
  printf 'EVAL_CONFIG=%s\n' "$(q "${EVAL_CONFIG}")"
  printf 'EVAL_TOKENIZER=%s\n' "$(q "${EVAL_TOKENIZER}")"
  printf 'EVAL_TOKENIZER_BACKEND=%s\n' "$(q "${EVAL_TOKENIZER_BACKEND}")"
  printf 'EVAL_TEMPERATURE=%s\n' "$(q "${EVAL_TEMPERATURE}")"
  printf 'EVAL_TOP_P=%s\n' "$(q "${EVAL_TOP_P}")"
  printf 'EVAL_PARALLELISM=%s\n' "$(q "${EVAL_PARALLELISM}")"
  printf 'EVAL_JUDGE_PARALLELISM=%s\n' "$(q "${EVAL_JUDGE_PARALLELISM}")"
  printf 'EVAL_NUM_REPEATS=%s\n' "$(q "${EVAL_NUM_REPEATS}")"
  printf 'EVAL_MAX_NEW_TOKENS=%s\n' "$(q "${EVAL_MAX_NEW_TOKENS}")"
  printf 'EVAL_RULER_MAX_SEQ_LENGTH=%s\n' "$(q "${EVAL_RULER_MAX_SEQ_LENGTH}")"
  printf 'EVAL_RULER_TASK_INDEX=%s\n' "$(q "${EVAL_RULER_TASK_INDEX}")"
  printf 'EVAL_TASK_INDEX=%s\n' "$(q "${EVAL_TASK_INDEX}")"
  printf 'EVAL_LIMIT_SAMPLES=%s\n' "$(q "${EVAL_LIMIT_SAMPLES}")"
  printf 'EVAL_SUBTASKS=%s\n' "$(q "${EVAL_SUBTASKS}")"
  printf 'MAX_MODEL_LEN=%s\n' "$(q "${MAX_MODEL_LEN}")"
  printf 'MAX_NUM_SEQS=%s\n' "$(q "${MAX_NUM_SEQS}")"
  printf 'MAX_NUM_BATCHED_TOKENS=%s\n' "$(q "${MAX_NUM_BATCHED_TOKENS}")"
  printf 'TP_SIZE=%s\n' "$(q "${TP_SIZE}")"
  printf 'DP_SIZE=%s\n' "$(q "${DP_SIZE}")"
  printf 'GPU_MEM=%s\n' "$(q "${GPU_MEM}")"
  printf 'ENFORCE_EAGER=%s\n' "$(q "${ENFORCE_EAGER}")"
  printf 'USE_COMPILATION_CONFIG=%s\n' "$(q "${USE_COMPILATION_CONFIG}")"
  printf 'MODEL_DIR=%s\n' "$(q "${MODEL_DIR}")"
  printf 'TRUST_REMOTE_CODE=%s\n' "$(q "${TRUST_REMOTE_CODE}")"
  printf 'REASONING_PARSER=%s\n' "$(q "${REASONING_PARSER}")"
  printf 'REASONING_PARSER_PLUGIN=%s\n' "$(q "${REASONING_PARSER_PLUGIN}")"
  printf 'OUT_ROOT=%s\n' "$(q "${OUT_ROOT}")"
  printf 'OUT_SUFFIX=%s\n' "$(q "${OUT_SUFFIX}")"
  printf 'DIRTY_TAG=%s\n' "$(q "${DIRTY_TAG}")"
  cat <<'LAUNCHER_BODY'

cd "${DECI_REPO}"

export HOME="/tmp/${USER}/home"
export XDG_CACHE_HOME="/tmp/${USER}/deci-evals-cache"
export HF_HOME="/tmp/${USER}/hf-home"
export HF_DATASETS_CACHE="/tmp/${USER}/hf-datasets-cache"
export UV_CACHE_DIR="/tmp/${USER}/uv-cache"
export UV_PYTHON_INSTALL_DIR="${DECI_REPO}/.uv-python"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${HF_DATASETS_CACHE}" "${UV_CACHE_DIR}" "${UV_PYTHON_INSTALL_DIR}"

if [[ -n "${SECRETS_FILE}" && -f "${SECRETS_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${SECRETS_FILE}"
  set +a
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi

if [[ "${CONFIG_ONLY:-0}" != "1" ]]; then
  missing=()
  for name in HF_TOKEN JUDGE_API_KEY DECI_INFERENCE_HUB_KEY DECI_BUILD_NVDEV_KEY; do
    if [[ -z "${!name:-}" ]]; then
      missing+=("${name}")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    printf 'Missing required eval secret env vars: %s\n' "${missing[*]}" >&2
    printf 'Set them in the environment or pass SECRETS_FILE=/path/to/secrets.\n' >&2
    exit 1
  fi
fi

OUT="${OUT_ROOT}/${OUT_SUFFIX}"
EXTRA_ARGS=""
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  ENFORCE_EAGER_BOOL=true
else
  ENFORCE_EAGER_BOOL=false
fi
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  EXTRA_ARGS="--trust-remote-code"
fi
if [[ -n "${REASONING_PARSER_PLUGIN}" ]]; then
  [[ -n "${REASONING_PARSER}" ]] || {
    printf 'REASONING_PARSER_PLUGIN requires REASONING_PARSER.\n' >&2
    exit 1
  }
  EXTRA_ARGS="${EXTRA_ARGS:+${EXTRA_ARGS} }--reasoning-parser-plugin ${REASONING_PARSER_PLUGIN} --reasoning-parser ${REASONING_PARSER}"
fi
EXTRA_ARGS="${EXTRA_ARGS:+${EXTRA_ARGS} }--no-enable-log-requests --enable-auto-tool-choice --tool-call-parser qwen3_coder --attention-backend FLASH_ATTN --enable-expert-parallel --model-loader-extra-config \\{\\\"enable_multithread_load\\\":true,\\\"num_threads\\\":96\\} --max-num-seqs ${MAX_NUM_SEQS} --max-model-len ${MAX_MODEL_LEN} --mamba-ssm-cache-dtype float32 --no-enable-prefix-caching --enable-chunked-prefill"
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --enforce-eager"
fi
if [[ "${USE_COMPILATION_CONFIG}" == "1" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --compilation-config \\{\\\"mode\\\":3,\\\"cudagraph_mode\\\":\\\"PIECEWISE\\\"\\}"
fi
EXTRA_ARGS="${EXTRA_ARGS} --block-size 16"
if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  EXTRA_ARGS="${EXTRA_ARGS} --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS}"
fi

export PACKED_TP2
export PACKED_TP2_INSTANCES_PER_NODE
export PACKED_TP2_PORT_BASE
export PACKED_TP2_CUDA_GROUPS
export PACKED_TP2_DRY_RUN
export PACKED_TP2_NUM_NODES="${NUM_NODES}"

if [[ "${PACKED_TP2}" == "1" ]]; then
  cmd=(
    "${DECI_REPO}/.venv/bin/python" "${VLLM_REPO}/scripts/submit_nano_dsa_packed_tp2.py"
    "${MODEL_DIR}" "${EVAL_CONFIG}" "${OUT}"
  )
  if [[ "${PACKED_TP2_DRY_RUN}" == "1" ]]; then
    cmd+=(--packed-script-dry-run)
  fi
else
  cmd=(.venv/bin/evaluate "${MODEL_DIR}" "${EVAL_CONFIG}" "${OUT}")
fi

cmd+=(
  --tasks "${TASKS}"
  --cluster aws-pdx
  --account "${ACCOUNT}"
  --dirty-tag "${DIRTY_TAG}"
  --skip-ssh
  --overrides "execution.num_nodes=${NUM_NODES}"
  --overrides "execution.num_instances=${NUM_INSTANCES}"
  --overrides "execution.walltime=${WALLTIME}"
  --overrides "execution.partition=${EVAL_PARTITION}"
  --overrides "execution.gres=gpu:8"
  --overrides "execution.mounts.mount_home=false"
  --overrides "++execution.env_vars.deployment.HF_HUB_READ_TIMEOUT=lit:60"
  --overrides "++execution.env_vars.evaluation.HF_HUB_READ_TIMEOUT=lit:60"
  --overrides "deployment.image=${VLLM_IMAGE}"
  --overrides "execution.proxy.image=${PROXY_IMAGE}"
  --overrides "deployment.tensor_parallel_size=${TP_SIZE}"
  --overrides "deployment.data_parallel_size=${DP_SIZE}"
  --overrides "deployment.gpu_memory_utilization=${GPU_MEM}"
  --overrides "deployment.enforce_eager=${ENFORCE_EAGER_BOOL}"
  --overrides "++deployment.max_model_len=${MAX_MODEL_LEN}"
  --overrides "++deployment.max_num_seqs=${MAX_NUM_SEQS}"
  --overrides "++deployment.enable_chunked_prefill=true"
  --overrides "++deployment.env_vars.HF_MODULES_CACHE=lit:/tmp/hf_modules_${DIRTY_TAG}"
  --overrides "++deployment.env_vars.VLLM_ALLOW_LONG_MAX_MODEL_LEN=lit:1"
  --overrides "++deployment.env_vars.VLLM_WORKER_MULTIPROC_METHOD=lit:fork"
  --overrides "++deployment.env_vars.VLLM_USE_DEEP_GEMM=lit:0"
  --overrides "++deployment.env_vars.VLLM_MOE_USE_DEEP_GEMM=lit:0"
  --overrides "++deployment.env_vars.VLLM_DEEP_GEMM_WARMUP=lit:skip"
  --overrides "deployment.extra_args='${EXTRA_ARGS}'"
)
if [[ "${USE_DSA_ENV}" == "1" ]]; then
  cmd+=(
  --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_FORCE_KERNEL_BLOCK_SIZE=lit:${DSA_FORCE_KERNEL_BLOCK_SIZE}"
  --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_USE_PAGE_TABLE_FA=lit:${USE_PAGE_TABLE_FA}"
  --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_USE_PREFILL_PAGE_TABLE_FA=lit:${USE_PREFILL_PAGE_TABLE_FA}"
  --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_USE_FULL_ATTN_SHORT_SEQ=lit:${USE_FULL_ATTN_SHORT_SEQ}"
  --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_USE_FLATTENED_PREFILL_PAGE_TABLE_FA=lit:${USE_FLATTENED_PREFILL_PAGE_TABLE_FA}"
  --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_USE_FLATTENED_DECODE_PAGE_TABLE_FA=lit:${USE_FLATTENED_DECODE_PAGE_TABLE_FA}"
  --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_CHUNKED_QUERY_CHUNK_SIZE=lit:${QUERY_CHUNK_SIZE}"
  --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_USE_TRITON_SCORING=lit:${USE_TRITON_SCORING}"
  --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_USE_TRITON_BATCHED_SUMMARIES=lit:${USE_TRITON_BATCHED_SUMMARIES}"
  --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_PROVIDER_CLASS=lit:${DSA_PROVIDER_CLASS}"
  )
  if [[ -n "${DENSE_PREFILL_KV_THRESHOLD_TOKENS}" ]]; then
    cmd+=(
      --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_DENSE_PREFILL_KV_THRESHOLD_TOKENS=lit:${DENSE_PREFILL_KV_THRESHOLD_TOKENS}"
    )
  fi
  if [[ -n "${DSA_ATTENTION_CLASS}" ]]; then
    cmd+=(
      --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_ATTENTION_CLASS=lit:${DSA_ATTENTION_CLASS}"
    )
  fi
  if [[ -n "${DSA_ATTENTION_MODULE}" ]]; then
    cmd+=(
      --overrides "++deployment.env_vars.VLLM_NEMOTRON_H_DSA_ATTENTION_MODULE=lit:${DSA_ATTENTION_MODULE}"
    )
  fi
  moonshot_env_names=(
    DSA_ATTN_MODE
    CHUNK_SIZE
    USE_FLASH_TOPK
    SHARE_CHUNK_TOPK
    SHARE_TOPK_GROUP_SIZE
    SHARE_TOPK_MODE
    SHARE_TOPK_UNION_MAX_CHUNKS
    USE_SHARED_PREFILL_PAGE_TABLE_FA
    USE_UNION_PREFILL_KERNEL
    USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA
    UNION_CHUNKS_PER_ITER
    USE_SUMMARY_CACHE
    SUMMARY_CACHE_MAX_BLOCKS
  )
  moonshot_vllm_env_names=(
    VLLM_NEMOTRON_H_DSA_ATTN_MODE
    VLLM_NEMOTRON_H_DSA_CHUNK_SIZE
    VLLM_NEMOTRON_H_DSA_USE_FLASH_TOPK
    VLLM_NEMOTRON_H_DSA_SHARE_CHUNK_TOPK
    VLLM_NEMOTRON_H_DSA_SHARE_TOPK_GROUP_SIZE
    VLLM_NEMOTRON_H_DSA_SHARE_TOPK_MODE
    VLLM_NEMOTRON_H_DSA_SHARE_TOPK_UNION_MAX_CHUNKS
    VLLM_NEMOTRON_H_DSA_USE_SHARED_PREFILL_PAGE_TABLE_FA
    VLLM_NEMOTRON_H_DSA_USE_UNION_PREFILL_KERNEL
    VLLM_NEMOTRON_H_DSA_USE_UNION_SUPERSET_PREFILL_PAGE_TABLE_FA
    VLLM_NEMOTRON_H_DSA_UNION_CHUNKS_PER_ITER
    VLLM_NEMOTRON_H_DSA_USE_SUMMARY_CACHE
    VLLM_NEMOTRON_H_DSA_SUMMARY_CACHE_MAX_BLOCKS
  )
  for i in "${!moonshot_env_names[@]}"; do
    name="${moonshot_env_names[$i]}"
    value="${!name}"
    if [[ -n "${value}" ]]; then
      cmd+=(
        --overrides "++deployment.env_vars.${moonshot_vllm_env_names[$i]}=lit:${value}"
      )
    fi
  done
fi

if [[ -n "${EVAL_TOKENIZER}" ]]; then
  cmd+=(
    --overrides "evaluation.nemo_evaluator_config.config.params.extra.tokenizer=${EVAL_TOKENIZER}"
    --overrides "evaluation.nemo_evaluator_config.config.params.extra.tokenizer_backend=${EVAL_TOKENIZER_BACKEND}"
  )
fi
if [[ -n "${EVAL_TEMPERATURE}" ]]; then
  cmd+=(--overrides "evaluation.nemo_evaluator_config.config.params.temperature=${EVAL_TEMPERATURE}")
fi
if [[ -n "${EVAL_TOP_P}" ]]; then
  cmd+=(--overrides "evaluation.nemo_evaluator_config.config.params.top_p=${EVAL_TOP_P}")
fi
if [[ -n "${EVAL_PARALLELISM}" ]]; then
  if [[ -n "${EVAL_TASK_INDEX}" ]]; then
    cmd+=(
      --overrides "evaluation.tasks.${EVAL_TASK_INDEX}.nemo_evaluator_config.config.params.parallelism=${EVAL_PARALLELISM}"
    )
  else
    cmd+=(--overrides "evaluation.nemo_evaluator_config.config.params.parallelism=${EVAL_PARALLELISM}")
  fi
fi
if [[ -n "${EVAL_JUDGE_PARALLELISM}" ]]; then
  [[ -n "${EVAL_TASK_INDEX}" ]] || {
    printf 'EVAL_JUDGE_PARALLELISM requires EVAL_TASK_INDEX for task-specific configs.\n' >&2
    exit 1
  }
  cmd+=(
    --overrides "evaluation.tasks.${EVAL_TASK_INDEX}.nemo_evaluator_config.config.params.extra.judge.parallelism=${EVAL_JUDGE_PARALLELISM}"
  )
fi
if [[ -n "${EVAL_NUM_REPEATS}" ]]; then
  if [[ -n "${EVAL_TASK_INDEX}" ]]; then
    cmd+=(
      --overrides "evaluation.tasks.${EVAL_TASK_INDEX}.nemo_evaluator_config.config.params.extra.num_repeats=${EVAL_NUM_REPEATS}"
    )
  else
    cmd+=(--overrides "evaluation.nemo_evaluator_config.config.params.extra.num_repeats=${EVAL_NUM_REPEATS}")
  fi
fi
if [[ -n "${EVAL_MAX_NEW_TOKENS}" ]]; then
  if [[ -n "${EVAL_TASK_INDEX}" ]]; then
    cmd+=(
      --overrides "evaluation.tasks.${EVAL_TASK_INDEX}.nemo_evaluator_config.config.params.max_new_tokens=${EVAL_MAX_NEW_TOKENS}"
    )
  else
    cmd+=(--overrides "evaluation.nemo_evaluator_config.config.params.max_new_tokens=${EVAL_MAX_NEW_TOKENS}")
  fi
fi
if [[ -n "${EVAL_RULER_MAX_SEQ_LENGTH}" ]]; then
  cmd+=(--overrides "++evaluation.nemo_evaluator_config.config.params.extra.ruler.max_seq_length=${EVAL_RULER_MAX_SEQ_LENGTH}")
  if [[ -n "${EVAL_RULER_TASK_INDEX}" ]]; then
    cmd+=(
      --overrides "evaluation.tasks.${EVAL_RULER_TASK_INDEX}.nemo_evaluator_config.config.params.extra.ruler.max_seq_length=${EVAL_RULER_MAX_SEQ_LENGTH}"
    )
  elif [[ -n "${EVAL_TASK_INDEX}" ]]; then
    cmd+=(
      --overrides "evaluation.tasks.${EVAL_TASK_INDEX}.nemo_evaluator_config.config.params.extra.ruler.max_seq_length=${EVAL_RULER_MAX_SEQ_LENGTH}"
    )
  else
    cmd+=(
      --overrides "++evaluation.tasks.0.nemo_evaluator_config.config.params.extra.ruler.max_seq_length=${EVAL_RULER_MAX_SEQ_LENGTH}"
    )
  fi
fi
if [[ -n "${EVAL_LIMIT_SAMPLES}" ]]; then
  cmd+=(
    --overrides "++evaluation.nemo_evaluator_config.config.params.limit_samples=${EVAL_LIMIT_SAMPLES}"
    --overrides "++evaluation.tasks.0.nemo_evaluator_config.config.params.limit_samples=${EVAL_LIMIT_SAMPLES}"
  )
fi
if [[ -n "${EVAL_SUBTASKS}" ]]; then
  cmd+=(
    --overrides "++evaluation.nemo_evaluator_config.config.params.extra.subtasks=${EVAL_SUBTASKS}"
    --overrides "++evaluation.tasks.0.nemo_evaluator_config.config.params.extra.subtasks=${EVAL_SUBTASKS}"
  )
fi

if [[ "${CONFIG_ONLY:-0}" == "1" ]]; then
  cmd+=(--config-only)
fi

printf 'Running deci-evals command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
LAUNCHER_BODY
} > "${LAUNCHER}"
chmod +x "${LAUNCHER}"

printf 'Generated launcher: %s\n' "${LAUNCHER}"
printf 'Benchmark: %s\n' "${TASKS}"
printf 'Image: %s\n' "${VLLM_IMAGE}"
printf 'Output suffix: %s\n' "${OUT_SUFFIX}"
if [[ "${PACKED_TP2}" == "1" ]]; then
  printf 'Packed TP2: nodes=%s instances=%s instances_per_node=%s ports=%s..%s cuda_groups=%s\n' \
    "${NUM_NODES}" "${NUM_INSTANCES}" "${PACKED_TP2_INSTANCES_PER_NODE}" \
    "${PACKED_TP2_PORT_BASE}" \
    "$((PACKED_TP2_PORT_BASE + PACKED_TP2_INSTANCES_PER_NODE - 1))" \
    "${PACKED_TP2_CUDA_GROUPS}"
fi

if [[ "${PRINT_ONLY}" == "1" ]]; then
  exit 0
fi

if [[ "${CONFIG_ONLY}" == "1" ]]; then
  printf 'Running config-only dry run...\n'
  CONFIG_ONLY=1 bash "${LAUNCHER}"
else
  printf 'Submitting launcher with sbatch...\n'
  sbatch "${LAUNCHER}"
fi
