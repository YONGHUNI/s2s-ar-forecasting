#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN_ID="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
raw="$(sbatch --parsable \
  --output="$LOG_DIR/pangu-gpu-%j.log" \
  --error="$LOG_DIR/pangu-gpu-%j.log" \
  slurm/verify_pangu_gpu.slurm)"
job="${raw%%;*}"
printf 'Job: %s\nLog: %s/pangu-gpu-%s.log\n' "$job" "$LOG_DIR" "$job"
