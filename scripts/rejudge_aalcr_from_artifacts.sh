#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Re-run only the AALCR judge/summarizer against an existing artifact folder.

Usage:
  scripts/rejudge_aalcr_from_artifacts.sh ARTIFACTS_DIR [options]

Options:
  --secrets-file PATH       Env file with DECI_BUILD_NVDEV_KEY and HF/JUDGE keys.
  --judge-parallelism N     Per-repeat judge concurrency. Default: 1.
  --num-repeats N           AALCR repeat count to judge. Default: 16.
  --account NAME            Slurm account. Default: nemotron_n4_compress.
  --partition NAME          Slurm partition. Default: cpu.
  --time HH:MM:SS           Slurm walltime. Default: 12:00:00.
  --container IMAGE         Nemo Skills container for AALCR judging.
  --print-only              Write the sbatch script but do not submit it.
  -h, --help                Show this help.

The artifact folder should be the failed ns_aa_lcr/artifacts directory that
already contains tmp-eval-results/aalcr/output-rs*.jsonl(.done).
USAGE
}

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

q() {
  printf '%q' "$1"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VLLM_REPO="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

ARTIFACTS_DIR=""
SECRETS_FILE="${SECRETS_FILE:-/lustre/fsw/portfolios/coreai/users/${USER}/secrets/deci-evals.env}"
JUDGE_PARALLELISM="${JUDGE_PARALLELISM:-1}"
NUM_REPEATS="${NUM_REPEATS:-16}"
ACCOUNT="${ACCOUNT:-nemotron_n4_compress}"
PARTITION="${PARTITION:-cpu}"
WALLTIME="${WALLTIME:-12:00:00}"
CONTAINER="${CONTAINER:-gitlab-master.nvidia.com#dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/nemo-skills:dev-2026-02-20T12-06-022e9ec9}"
PRINT_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --secrets-file)
      [[ $# -ge 2 ]] || die "--secrets-file requires a value"
      SECRETS_FILE="$2"
      shift 2
      ;;
    --judge-parallelism)
      [[ $# -ge 2 ]] || die "--judge-parallelism requires a value"
      JUDGE_PARALLELISM="$2"
      shift 2
      ;;
    --num-repeats)
      [[ $# -ge 2 ]] || die "--num-repeats requires a value"
      NUM_REPEATS="$2"
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
    --container)
      [[ $# -ge 2 ]] || die "--container requires a value"
      CONTAINER="$2"
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
    -*)
      die "unknown option: $1"
      ;;
    *)
      if [[ -z "${ARTIFACTS_DIR}" ]]; then
        ARTIFACTS_DIR="$1"
      else
        die "unexpected argument: $1"
      fi
      shift
      ;;
  esac
done

[[ -n "${ARTIFACTS_DIR}" ]] || {
  usage >&2
  exit 2
}

[[ "${JUDGE_PARALLELISM}" =~ ^[0-9]+$ ]] || die "--judge-parallelism must be an integer"
[[ "${JUDGE_PARALLELISM}" -ge 1 ]] || die "--judge-parallelism must be >= 1"
[[ "${NUM_REPEATS}" =~ ^[0-9]+$ ]] || die "--num-repeats must be an integer"
[[ "${NUM_REPEATS}" -ge 1 ]] || die "--num-repeats must be >= 1"

[[ -d "${ARTIFACTS_DIR}" ]] || die "artifact folder not found: ${ARTIFACTS_DIR}"
ARTIFACTS_DIR="$(realpath "${ARTIFACTS_DIR}")"
[[ -f "${ARTIFACTS_DIR}/run_config.yml" ]] || die "missing run_config.yml in ${ARTIFACTS_DIR}"
[[ -d "${ARTIFACTS_DIR}/tmp-eval-results/aalcr" ]] || die "missing tmp-eval-results/aalcr in ${ARTIFACTS_DIR}"
[[ -f "${SECRETS_FILE}" ]] || die "secrets file not found: ${SECRETS_FILE}"

done_count="$(
  find "${ARTIFACTS_DIR}/tmp-eval-results/aalcr" -maxdepth 1 \
    -name 'output-rs*.jsonl.done' -type f | wc -l
)"
[[ "${done_count}" -gt 0 ]] || die "no completed AALCR generation shards found"

LAUNCHER_DIR="${VLLM_REPO}/outputs/eval_launchers"
mkdir -p "${LAUNCHER_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCHER="${LAUNCHER_DIR}/rejudge_aalcr_${STAMP}.sbatch"
LOG_DIR="$(dirname "${ARTIFACTS_DIR}")/logs"
mkdir -p "${LOG_DIR}"

{
  printf '#!/usr/bin/env bash\n'
  printf '#SBATCH --job-name=aalcr-rejudge\n'
  printf '#SBATCH --account=%s\n' "${ACCOUNT}"
  printf '#SBATCH --partition=%s\n' "${PARTITION}"
  printf '#SBATCH --time=%s\n' "${WALLTIME}"
  printf '#SBATCH --nodes=1\n'
  printf '#SBATCH --ntasks=1\n'
  printf '#SBATCH --output=%s/aalcr_rejudge_%%j.log\n' "${LOG_DIR}"
  printf '#SBATCH --error=%s/aalcr_rejudge_%%j.log\n' "${LOG_DIR}"
  printf '\nset -euo pipefail\n\n'
  printf 'ARTIFACTS_DIR=%s\n' "$(q "${ARTIFACTS_DIR}")"
  printf 'SECRETS_FILE=%s\n' "$(q "${SECRETS_FILE}")"
  printf 'JUDGE_PARALLELISM=%s\n' "$(q "${JUDGE_PARALLELISM}")"
  printf 'NUM_REPEATS=%s\n' "$(q "${NUM_REPEATS}")"
  printf 'CONTAINER=%s\n' "$(q "${CONTAINER}")"
  cat <<'SBATCH_BODY'

if [[ -f "${SECRETS_FILE}" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${SECRETS_FILE}"
  set +a
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi
if [[ -z "${DECI_BUILD_NVDEV_KEY:-}" ]]; then
  printf 'Missing DECI_BUILD_NVDEV_KEY in secrets or environment.\n' >&2
  exit 1
fi
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}"
export INFERENCE_API_KEY="${DECI_BUILD_NVDEV_KEY}"
export API_KEY="${API_KEY:-DUMMY_VALUE}"
export JUDGE_PARALLELISM
export NUM_REPEATS
export NEMO_EVALUATOR_TELEMETRY_LEVEL="${NEMO_EVALUATOR_TELEMETRY_LEVEL:-2}"
export NEMO_EVALUATOR_TELEMETRY_SESSION_ID="${NEMO_EVALUATOR_TELEMETRY_SESSION_ID:-aalcr-rejudge-${SLURM_JOB_ID}}"

WORK_DIR="/tmp/${USER}/aalcr-rejudge-${SLURM_JOB_ID}"
mkdir -p "${WORK_DIR}"

srun --mpi pmix --nodes 1 --ntasks 1 \
  --container-image "${CONTAINER}" \
  --container-mounts "${ARTIFACTS_DIR}:/results,/lustre:/lustre,/scratch:/scratch,/tmp:/tmp,${WORK_DIR}:/work" \
  --no-container-mount-home \
  --container-env API_KEY,HF_TOKEN,HUGGINGFACE_HUB_TOKEN,INFERENCE_API_KEY,JUDGE_API_KEY,NEMO_EVALUATOR_TELEMETRY_LEVEL,NEMO_EVALUATOR_TELEMETRY_SESSION_ID,JUDGE_PARALLELISM,NUM_REPEATS \
  bash -lc '
set -euo pipefail
mkdir -p /tmp/nel-results /work
cp -r /results/. /tmp/nel-results/
rm -rf /tmp/nel-results/eval-results/aalcr

awk -v jp="${JUDGE_PARALLELISM}" '"'"'
  /^      judge:$/ {
    in_judge = 1
  }
  in_judge && /^      [[:alnum:]_]+:/ && !/^      judge:$/ {
    in_judge = 0
  }
  in_judge && /^        parallelism:/ {
    print "        parallelism: " jp
    next
  }
  /^      num_repeats:/ {
    print "      num_repeats: " ENVIRON["NUM_REPEATS"]
    next
  }
  { print }
'"'"' /results/run_config.yml > /work/run_config.yml

sync_pid=""
(while true; do
  sleep 300
  cp -au /tmp/nel-results/. /results/
done) &
sync_pid=$!

cleanup() {
  code=$?
  if [[ -n "${sync_pid}" ]]; then
    kill "${sync_pid}" 2>/dev/null || true
  fi
  cp -r /tmp/nel-results/. /results/
  exit "${code}"
}
trap cleanup EXIT

cmd=$(command -v nemo-evaluator >/dev/null 2>&1 && echo nemo-evaluator || echo eval-factory)
"${cmd}" run_eval --run_config /work/run_config.yml
'
SBATCH_BODY
} > "${LAUNCHER}"
chmod +x "${LAUNCHER}"

printf 'Generated AALCR rejudge launcher: %s\n' "${LAUNCHER}"
printf 'Artifacts: %s\n' "${ARTIFACTS_DIR}"
printf 'Completed generation shards found: %s\n' "${done_count}"
printf 'Judge parallelism: %s\n' "${JUDGE_PARALLELISM}"
printf 'Num repeats: %s\n' "${NUM_REPEATS}"

if [[ "${PRINT_ONLY}" == "1" ]]; then
  exit 0
fi

sbatch "${LAUNCHER}"
