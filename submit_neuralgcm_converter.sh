#!/usr/bin/env bash
# Attach a CPU converter to an already-submitted NeuralGCM GPU array.
# Do NOT resubmit the producer. Intended for login-node execution.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PRODUCER_JOB_ID="${1:-}"
[[ "$PRODUCER_JOB_ID" =~ ^[0-9]+$ ]] || {
  echo "Usage: bash submit_neuralgcm_converter.sh <existing_producer_job_id>" >&2
  exit 2
}
[[ -f "$ROOT/scripts/convert_neuralgcm_netcdf.py" ]] || exit 1
[[ -f "$ROOT/slurm/run_neuralgcm_convert.slurm" ]] || exit 1

RUN_ID="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
raw="$(sbatch --parsable \
    --output="$LOG_DIR/neuralgcm-convert-%j.log" \
    --error="$LOG_DIR/neuralgcm-convert-%j.log" \
    --export="ALL,PRODUCER_JOB_ID=$PRODUCER_JOB_ID" \
    slurm/run_neuralgcm_convert.slurm)"
job="${raw%%;*}"
printf 'Existing producer:     %s\n' "$PRODUCER_JOB_ID"
printf 'CPU NetCDF converter:  %s\n' "$job"
printf 'Log directory:         %s\n' "$LOG_DIR"
printf 'Monitor: squeue -j %s,%s\n' "$PRODUCER_JOB_ID" "$job"
printf 'NetCDF destination: /scratch/%s/weather-ai/neuralgcm-operational/netcdf/\n' "$USER"
