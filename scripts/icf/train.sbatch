#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit with scripts/icf/submit.sh}" "${FD_CODE_ROOT:?}" "${FD_PYTHON:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export POLARS_MAX_THREADS="$OMP_NUM_THREADS"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
cd "$FD_CODE_ROOT"
"$FD_PYTHON" "$FD_CODE_ROOT/scripts/icf/job.py"
