#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN_ID="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
raw="$(sbatch --parsable \
  --output="$LOG_DIR/pangu-one-step-%j.log" \
  --error="$LOG_DIR/pangu-one-step-%j.log" \
  slurm/test_pangu_one_step.slurm)"
job="${raw%%;*}"
echo "Job: $job"
echo "Log: $LOG_DIR/pangu-one-step-$job.log"
