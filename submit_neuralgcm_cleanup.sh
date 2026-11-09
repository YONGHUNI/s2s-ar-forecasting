#!/usr/bin/env bash
# Login-node wrapper to audit / reclaim NeuralGCM Zarr after NetCDF validation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="${NGCM_CLEANUP_MODE:-audit}"
case "$MODE" in
  audit) ;;
  delete)
    [[ "${NGCM_CLEANUP_CONFIRMED:-0}" == "1" ]] || {
      echo "Deletion requires NGCM_CLEANUP_CONFIRMED=1" >&2
      exit 2
    }
    ;;
  *) echo "NGCM_CLEANUP_MODE must be audit or delete" >&2; exit 2 ;;
esac
RUN_ID="$(date +'%Y%m%d_%H%M%S')"
LOG_DIR="$ROOT/logs/$RUN_ID"
mkdir -p "$LOG_DIR"
raw="$(sbatch --parsable \
    --output="$LOG_DIR/neuralgcm-cleanup-%j.log" \
    --error="$LOG_DIR/neuralgcm-cleanup-%j.log" \
    --export="ALL,NGCM_CLEANUP_MODE=$MODE" \
    slurm/run_neuralgcm_cleanup.slurm)"
job="${raw%%;*}"
printf 'Mode:             %s\n' "$MODE"
printf 'Cleanup Job ID:   %s\n' "$job"
printf 'Log directory:    %s\n' "$LOG_DIR"
printf 'Monitor: squeue -j %s\n' "$job"
