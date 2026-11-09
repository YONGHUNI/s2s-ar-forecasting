#!/usr/bin/env bash
# Submit NeuralGCM GPU forecast array plus CPU NetCDF converter from login node.
# The converter automatically reclaims a Zarr only after validating its NetCDF.
# No IVT calculation is performed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -f "$ROOT/scripts/run_neuralgcm_batch.py" ]] || {
  echo "Missing NeuralGCM batch driver" >&2
  exit 1
}
[[ -f "$ROOT/slurm/run_neuralgcm_forecast.slurm" ]] || {
  echo "Missing NeuralGCM Slurm script" >&2
  exit 1
}
[[ -f "$ROOT/pixi.toml" ]] || {
  echo "Missing Pixi manifest" >&2
  exit 1
}

MODE="${NGCM_MODE:-pilot}"
OUTPUT_DIR="${NGCM_OUTPUT_DIR:-neuralgcm-operational-lead0}"
[[ "$OUTPUT_DIR" == "neuralgcm-operational-lead0" ]] || {
  echo "169-lead pipeline requires NGCM_OUTPUT_DIR=neuralgcm-operational-lead0" >&2
  exit 2
}
case "$MODE" in
  pilot)
    ARRAY="0"
    LIMIT="1"
    ;;
  production)
    [[ "${NGCM_PRODUCTION_CONFIRMED:-0}" == "1" ]] || {
      echo "Production needs NGCM_PRODUCTION_CONFIRMED=1 (607 initialization dates)." >&2
      exit 2
    }
    ARRAY="0-19%4"
    LIMIT=""
    ;;
  *)
    echo "NGCM_MODE must be pilot or production (got: $MODE)" >&2
    exit 2
    ;;
esac

RUN_ID="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"

producer_raw="$(
  sbatch --parsable \
    --array="$ARRAY" \
    --output="$LOG_DIR/neuralgcm-forecast-%A_%a.log" \
    --error="$LOG_DIR/neuralgcm-forecast-%A_%a.log" \
    --export="ALL,NGCM_LIMIT=$LIMIT,NGCM_OUTPUT_DIR=$OUTPUT_DIR" \
    slurm/run_neuralgcm_forecast.slurm
)"
producer_job="${producer_raw%%;*}"

printf 'NeuralGCM mode:       %s\n' "$MODE"
printf 'Output directory:      %s\n' "$OUTPUT_DIR"
printf 'Forecast array:       %s (%s)\n' "$producer_job" "$ARRAY"
printf 'Per-worker limit:     %s\n' "${LIMIT:-none}"
printf 'Log directory:        %s\n' "$LOG_DIR"
printf 'Monitor: squeue -j %s\n' "$producer_job"
printf 'Per-case logs: /scratch/%s/weather-ai/%s/logs/\n' "$USER" "$OUTPUT_DIR"

converter_raw="$(
  sbatch --parsable \
    --output="$LOG_DIR/neuralgcm-convert-%j.log" \
    --error="$LOG_DIR/neuralgcm-convert-%j.log" \
    --export="ALL,PRODUCER_JOB_ID=$producer_job,NGCM_CONVERT_LIMIT=$LIMIT,NGCM_OUTPUT_DIR=$OUTPUT_DIR" \
    slurm/run_neuralgcm_convert.slurm
)"
converter_job="${converter_raw%%;*}"
printf 'CPU converter:        %s\n' "$converter_job"
printf 'Monitor both: squeue -j %s,%s\n' "$producer_job" "$converter_job"
printf 'NetCDF: /scratch/%s/weather-ai/%s/netcdf/\n' "$USER" "$OUTPUT_DIR"
