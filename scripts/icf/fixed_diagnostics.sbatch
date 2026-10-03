#!/usr/bin/env bash
# Site configuration only. Finance 2h, Linear continuation 45m, MLP 20m.
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --qos=teaching
#SBATCH --cpus-per-task=4
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${FD_DIAG_CODE:?clean isolated checkout}"
: "${FD_DIAG_COMMIT:?pinned source commit}"
: "${FD_DIAG_ROOT:?new isolated output root}"
: "${FD_DIAG_PYTHON:?existing locked Python environment}"
: "${FD_DIAG_DATASET:?corrected snapshot}"
: "${FD_DIAG_CHECKSUMS:?immutable snapshot checksum inventory}"
: "${FD_DIAG_SOURCE:?completed C candidate parent directory}"
: "${FD_DIAG_CACHE:?original verified statistics cache}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONPATH="$FD_DIAG_CODE/src"
cd "$FD_DIAG_CODE"
test "$(git rev-parse HEAD)" = "$FD_DIAG_COMMIT"
test -z "$(git status --porcelain)"
case "${1:?candidate: finance, statistics_linear, statistics_mlp}" in
  finance) FD_DIAG_SECONDS=7080; FD_DIAG_CUMULATIVE_SECONDS=7200; FD_DIAG_MARGIN_SECONDS=120 ;;
  statistics_linear) FD_DIAG_SECONDS=2100; FD_DIAG_CUMULATIVE_SECONDS=3600; FD_DIAG_MARGIN_SECONDS=300 ;;
  statistics_mlp) FD_DIAG_SECONDS=1080; FD_DIAG_CUMULATIVE_SECONDS=1200; FD_DIAG_MARGIN_SECONDS=120 ;;
  *) exit 2 ;;
esac
FD_DIAG_REUSE_ARGS=()
if [[ -n "${FD_DIAG_REUSE_LINEAR_CHUNKS:-}" ]]; then
  test "$1" = statistics_linear
  FD_DIAG_REUSE_ARGS=(--reuse-linear-chunks "$FD_DIAG_REUSE_LINEAR_CHUNKS")
fi
srun --ntasks=1 --cpu-bind=cores /usr/bin/time -v "$FD_DIAG_PYTHON" \
  scripts/research/finance_fixed_diagnostics.py \
  --dataset "$FD_DIAG_DATASET" --checksums "$FD_DIAG_CHECKSUMS" \
  --config configs/experiments/finance_patch_transformer_scratch.yaml \
  --cache "$FD_DIAG_CACHE" --source "$FD_DIAG_SOURCE/$1" \
  --output "$FD_DIAG_ROOT/$1" --budget-seconds "$FD_DIAG_SECONDS" \
  --cumulative-budget-seconds "$FD_DIAG_CUMULATIVE_SECONDS" \
  --shutdown-margin-seconds "$FD_DIAG_MARGIN_SECONDS" "${FD_DIAG_REUSE_ARGS[@]}"
