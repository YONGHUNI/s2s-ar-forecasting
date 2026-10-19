#!/usr/bin/env bash
# SFNO GPU producer + concurrent CPU NetCDF converter (single entrypoint).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-config/sfno_pilot.yaml}"
[[ -f "$CONFIG" ]] || { echo "Missing SFNO config: $CONFIG" >&2; exit 1; }
[[ -f "$ROOT/pixi.lock" ]] || { echo "Missing root Pixi lockfile" >&2; exit 1; }
if [[ "$CONFIG" != "config/sfno_pilot.yaml" && "${SFNO_PRODUCTION_CONFIRMED:-0}" != "1" ]]; then
  echo "SFNO production is blocked until the pilot is validated; set SFNO_PRODUCTION_CONFIRMED=1 after validation." >&2
  exit 1
fi

# The pilot config must not launch 20 GPU jobs for a single initialization.
if [[ -n "${PANGU_ARRAY:-}" ]]; then
  ARRAY="$PANGU_ARRAY"
elif [[ "$CONFIG" == "config/sfno_pilot.yaml" ]]; then
  ARRAY="0"
else
  ARRAY="0-19%2"
fi

RUN_ID="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

producer_raw="$(sbatch --parsable \
  --array="$ARRAY" \
  --output="$LOG_DIR/sfno-forecast-%A_%a.log" \
  --error="$LOG_DIR/sfno-forecast-%A_%a.log" \
  --export="ALL,CONFIG=$CONFIG" \
  slurm/run_sfno_forecast.slurm)"
producer_job="${producer_raw%%;*}"

converter_raw="$(sbatch --parsable \
  --output="$LOG_DIR/sfno-convert-%j.log" \
  --error="$LOG_DIR/sfno-convert-%j.log" \
  --export="ALL,CONFIG=$CONFIG,PRODUCER_JOB_ID=$producer_job" \
  slurm/run_sfno_convert.slurm)"
converter_job="${converter_raw%%;*}"

printf 'SFNO config:         %s\n' "$CONFIG"
printf 'Producer array:       %s (%s)\n' "$producer_job" "$ARRAY"
printf 'CPU converter:        %s\n' "$converter_job"
printf 'Log directory:        %s\n' "$LOG_DIR"
printf 'Monitor: squeue -j %s,%s\n' "$producer_job" "$converter_job"
