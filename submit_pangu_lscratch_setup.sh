#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
RUN_ID="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
raw="$(sbatch --parsable --output="$LOG_DIR/pangu-lscratch-%j.log" --error="$LOG_DIR/pangu-lscratch-%j.log" slurm/prepare_pangu_lscratch.slurm)"
job="${raw%%;*}"
printf 'Job: %s\nLog: %s/pangu-lscratch-%s.log\n' "$job" "$LOG_DIR" "$job"
