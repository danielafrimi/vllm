#!/usr/bin/env bash
#SBATCH --account=nemotron_compress_dev
#SBATCH --partition=batch
#SBATCH --job-name=find-nsys
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm-find-nsys-%j.out

set -euo pipefail

echo "host=$(hostname)"
if [[ -f /etc/profile.d/modules.sh ]]; then
  source /etc/profile.d/modules.sh
fi
echo "PATH=${PATH}"
command -v nsys || true
module --version 2>&1 || true
module avail 2>&1 | grep -Ei 'nsight|nsys|cuda|nvidia' || true
find /opt /usr/local /cm /apps -maxdepth 6 \( -name nsys -o -path '*nsight*' \) 2>/dev/null | sed -n '1,200p' || true
