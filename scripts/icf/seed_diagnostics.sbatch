#!/usr/bin/env bash
# Approved seed repeats only. Site resources do not change the two-epoch scientific budget.
#SBATCH --partition=Teaching
#SBATCH --account=teaching
#SBATCH --qos=teaching
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --time=00:45:00
#SBATCH --signal=USR1@300
#SBATCH --no-requeue
set -euo pipefail
umask 077
: "${FD_DIAG_CODE:?clean isolated checkout}"
: "${FD_DIAG_COMMIT:?pinned source commit}"
: "${FD_DIAG_ROOT:?new seed-repeat output root}"
: "${FD_DIAG_PYTHON:?existing locked Python environment}"
: "${FD_DIAG_DATASET:?corrected snapshot}"
: "${FD_DIAG_CHECKSUMS:?immutable checksum inventory}"
: "${FD_DIAG_CACHE:?original verified statistics cache}"
case "${1:?statistics_linear or statistics_mlp}" in
  statistics_linear|statistics_mlp) ;;
  *) exit 2 ;;
esac
case "${2:?seed 17 or 73}" in
  17|73) ;;
  *) exit 2 ;;
esac
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 POLARS_MAX_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONPATH="$FD_DIAG_CODE/src"
cd "$FD_DIAG_CODE"
test "$(git rev-parse HEAD)" = "$FD_DIAG_COMMIT"
test -z "$(git status --porcelain)"
srun --ntasks=1 --cpu-bind=cores /usr/bin/time -v "$FD_DIAG_PYTHON" \
  scripts/research/finance_prefix_diagnostics.py run \
  --dataset "$FD_DIAG_DATASET" --checksums "$FD_DIAG_CHECKSUMS" \
  --config configs/experiments/finance_patch_transformer_scratch.yaml \
  --cache "$FD_DIAG_CACHE" --candidate "$1" --seed "$2" \
  --output "$FD_DIAG_ROOT/seed-$2/$1" \
  --observation-scope fit-selection --observation-precision fp32 --full-fit \
  --budget-seconds 2100 --cumulative-budget-seconds 2100 --shutdown-margin-seconds 300
