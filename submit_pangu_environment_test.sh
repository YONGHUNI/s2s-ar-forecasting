#!/usr/bin/env bash
# Submit the Pangu environment smoke test using the project log layout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

RUN_ID="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

job_raw="$(sbatch --parsable \
  --output="$LOG_DIR/pangu-env-%j.log" \
  --error="$LOG_DIR/pangu-env-%j.log" \
  slurm/test_pangu_environment.slurm)"
job_id="${job_raw%%;*}"

printf 'Pangu environment job: %s\n' "$job_id"
printf 'Log directory:         %s\n' "$LOG_DIR"
printf 'Log file:              %s/pangu-env-%s.log\n' "$LOG_DIR" "$job_id"
