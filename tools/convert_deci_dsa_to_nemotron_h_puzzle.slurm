#!/usr/bin/env bash
#SBATCH --job-name=dsa2puzzle
#SBATCH --account=nemotron_compress_dev
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/dsa2puzzle_%j.out
#SBATCH --error=logs/dsa2puzzle_%j.err

set -euo pipefail

REPO_DIR="/lustre/fsw/portfolios/coreai/users/mdabbah/deci/vllm_repos/vllm_v0.20.1"
cd "${REPO_DIR}"

date
hostname

.venv/bin/python -u tools/convert_deci_dsa_to_nemotron_h_puzzle.py "$@"

date
