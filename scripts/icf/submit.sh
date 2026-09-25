#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/icf/submit.sh /absolute/resources.env [--test-only]' >&2
  exit 2
fi
FD_ENV_FILE=$(realpath "$1")
set -a
# The file is an explicit, trusted deployment configuration.
source "$FD_ENV_FILE"
set +a
: "${FD_CODE_ROOT:?}" "${FD_LOG_ROOT:?}" "${FD_RUN_DIR:?}" "${FD_PYTHON:?}"
[[ -x "$FD_PYTHON" ]]
[[ -f "$FD_CONFIG" && -f "$FD_RUNTIME" ]]
[[ "$FD_RUN_DIR" != *REPLACE_WITH* ]]
mkdir -p "$FD_LOG_ROOT"
FD_SUBMIT_OPTIONS=()
if [[ $# == 2 ]]; then
  [[ "$2" == --test-only ]]
  FD_SUBMIT_OPTIONS+=(--test-only)
fi
sbatch "${FD_SUBMIT_OPTIONS[@]}" \
  --partition="$FD_PARTITION" --account="$FD_ACCOUNT" --qos="$FD_QOS" \
  --gres="$FD_GRES" --ntasks=1 --cpus-per-task="$FD_CPUS" \
  --mem="$FD_MEMORY" --time="$FD_TIME" --requeue --signal=USR1@300 \
  --chdir="$FD_CODE_ROOT" --job-name=facdigger-train \
  --output="$FD_LOG_ROOT/%x-%j.out" --export=ALL \
  "$FD_CODE_ROOT/scripts/icf/train.sbatch"
