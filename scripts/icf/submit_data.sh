#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo 'Usage: bash scripts/icf/submit_data.sh /absolute/data.env [--test-only]' >&2
  exit 2
fi
FD_ENV_FILE=$(realpath "$1")
set -a
source "$FD_ENV_FILE"
set +a
: "${FD_ROOT:?}" "${FD_CODE_ROOT:?}" "${FD_PYTHON:?}" "${FD_LOG_ROOT:?}"
[[ -x "$FD_PYTHON" && -f "$FD_DATA_CONFIG" && -f "$FD_RESEARCH_CONFIG" ]]
case "$FD_DATA_MODE" in plan|ingest|prepare) ;; *) exit 2 ;; esac
mkdir -p "$FD_LOG_ROOT"
FD_SUBMIT_OPTIONS=()
if [[ $# == 2 ]]; then
  [[ "$2" == --test-only ]]
  FD_SUBMIT_OPTIONS+=(--test-only)
fi
# Send deployment values only. The API token is loaded on the compute node,
# never embedded in Slurm's saved job environment or submission command.
FD_EXPORTS=()
for FD_KEY in FD_ROOT FD_CODE_ROOT FD_PYTHON FD_LOG_ROOT FD_DATA_MODE FD_DATA_CONFIG \
  FD_RESEARCH_CONFIG FD_RUNTIME FD_PREPARE_OUTPUT FD_CREDENTIAL_FILE \
  FD_RESERVE_API_CALLS FD_REQUESTS_PER_MINUTE; do
  [[ -n "${!FD_KEY}" && "${!FD_KEY}" != *,* && "${!FD_KEY}" != *$'\n'* ]]
  FD_EXPORTS+=("${FD_KEY}=${!FD_KEY}")
done
env -i "PATH=$PATH" "HOME=$HOME" "USER=$USER" LANG=C.UTF-8 \
  "${FD_EXPORTS[@]}" "$(command -v sbatch)" "${FD_SUBMIT_OPTIONS[@]}" \
  --partition="$FD_PARTITION" --account="$FD_ACCOUNT" --qos="$FD_QOS" \
  --ntasks=1 --cpus-per-task="$FD_CPUS" --mem="$FD_MEMORY" --time="$FD_TIME" \
  --no-requeue --chdir="$FD_CODE_ROOT" --job-name="facdigger-$FD_DATA_MODE" \
  --output="$FD_LOG_ROOT/%x-%j.out" --export=ALL \
  "$FD_CODE_ROOT/scripts/icf/data.sbatch"
