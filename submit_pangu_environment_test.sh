#!/usr/bin/env bash
# Single entrypoint for Pangu test and setup jobs. Keep the repository root tidy.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mode="${1:-setup}"
case "$mode" in
  setup|environment) job_script=slurm/prepare_pangu_lscratch.slurm; prefix=pangu-lscratch ;;
  one-step)          job_script=slurm/test_pangu_one_step.slurm; prefix=pangu-one-step ;;
  gpu)               job_script=slurm/verify_pangu_gpu.slurm; prefix=pangu-gpu ;;
  42day)             job_script=slurm/test_pangu_42day.slurm; prefix=pangu-42day ;;
  convert)
    echo "CPU conversion still uses the old /work environment; do not submit before node-local setup is fixed." >&2
    exit 2 ;;
  *)
    echo "Usage: bash $(basename "$0") [setup|environment|one-step|gpu|42day]" >&2
    exit 2 ;;
esac
RUN_ID="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
raw="$(sbatch --parsable \
  --output="$LOG_DIR/$prefix-%j.log" \
  --error="$LOG_DIR/$prefix-%j.log" \
  "$job_script")"
job="${raw%%;*}"
printf 'Mode: %s\nJob: %s\nLog: %s/%s-%s.log\n' "$mode" "$job" "$LOG_DIR" "$prefix" "$job"
