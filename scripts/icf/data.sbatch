#!/usr/bin/env bash
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Submit with scripts/icf/submit_data.sh}" "${FD_CODE_ROOT:?}" "${FD_PYTHON:?}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export POLARS_MAX_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
cd "$FD_CODE_ROOT"
"$FD_PYTHON" "$FD_CODE_ROOT/scripts/icf/data_job.py"
