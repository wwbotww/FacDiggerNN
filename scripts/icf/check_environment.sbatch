#!/usr/bin/env bash
#SBATCH --job-name=facdigger-env
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --qos=teaching
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
set -euo pipefail
: "${FD_CODE_ROOT:?Set a fixed Lustre clone}" "${FD_PYTHON:?Set its locked Python}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export POLARS_MAX_THREADS="$OMP_NUM_THREADS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
cd "$FD_CODE_ROOT"
srun "$FD_PYTHON" "$FD_CODE_ROOT/scripts/icf/job.py" --check-environment
