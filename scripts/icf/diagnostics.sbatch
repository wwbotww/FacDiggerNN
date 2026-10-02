#!/usr/bin/env bash
# School wrapper only. submit with the preregistered phase-specific allocation caps.
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --qos=teaching
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${FD_DIAG_CODE:?clean isolated checkout}"
: "${FD_DIAG_COMMIT:?pinned source commit}"
: "${FD_DIAG_ROOT:?isolated evidence directory}"
: "${FD_DIAG_PYTHON:?existing locked Python environment}"
: "${FD_DIAG_DATASET:?corrected wf3 snapshot}"
: "${FD_DIAG_CHECKSUMS:?immutable snapshot checksum inventory}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONPATH="$FD_DIAG_CODE/src"
cd "$FD_DIAG_CODE"
test "$(git rev-parse HEAD)" = "$FD_DIAG_COMMIT"
test -z "$(git status --porcelain)"
case "${1:?phase: R, cache, statistics_linear, statistics_mlp, finance}" in
  R)
    : "${FD_DIAG_OLD_MATRIX:?historical matrix directory}"
    : "${FD_DIAG_OLD_SNAPSHOTS:?historical snapshot parent}"
    srun --ntasks=1 --cpu-bind=cores /usr/bin/time -v "$FD_DIAG_PYTHON" \
      scripts/research/finance_history_diagnostics.py \
      --matrix "$FD_DIAG_OLD_MATRIX" --snapshots "$FD_DIAG_OLD_SNAPSHOTS" \
      --output "$FD_DIAG_ROOT/R" --budget-seconds 14280
    ;;
  cache)
    srun --ntasks=1 --cpu-bind=cores /usr/bin/time -v "$FD_DIAG_PYTHON" \
      scripts/research/finance_prefix_diagnostics.py prepare \
      --dataset "$FD_DIAG_DATASET" --checksums "$FD_DIAG_CHECKSUMS" \
      --config configs/experiments/finance_patch_transformer_scratch.yaml \
      --output "$FD_DIAG_ROOT/cache" --budget-seconds 7080
    ;;
  statistics_linear|statistics_mlp|finance)
    if [[ "$1" == finance ]]; then FD_DIAG_SECONDS=23280; else FD_DIAG_SECONDS=2580; fi
    srun --ntasks=1 --cpu-bind=cores /usr/bin/time -v "$FD_DIAG_PYTHON" \
      scripts/research/finance_prefix_diagnostics.py run \
      --dataset "$FD_DIAG_DATASET" --checksums "$FD_DIAG_CHECKSUMS" \
      --config configs/experiments/finance_patch_transformer_scratch.yaml \
      --cache "$FD_DIAG_ROOT/cache" --candidate "$1" --output "$FD_DIAG_ROOT/C/$1" \
      --budget-seconds "$FD_DIAG_SECONDS"
    ;;
  *) exit 2 ;;
esac
