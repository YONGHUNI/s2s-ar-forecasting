#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-config/graphcast_operational.yaml}"

[[ -f "$CONFIG" ]] || {
    echo "GraphCast config not found: $CONFIG" >&2
    exit 1
}

producer_raw="$(
    sbatch         --parsable         --export="ALL,CONFIG=$CONFIG"         slurm/run_graphcast_forecast.slurm
)"
producer_job="${producer_raw%%;*}"

converter_raw="$(
    sbatch         --parsable         --dependency="after:${producer_job}"         --export="ALL,CONFIG=$CONFIG"         slurm/run_graphcast_convert.slurm
)"
converter_job="${converter_raw%%;*}"

printf 'GraphCast config:       %s\n' "$CONFIG"
printf 'GraphCast producer job: %s\n' "$producer_job"
printf 'NetCDF converter job:   %s\n' "$converter_job"
printf 'Monitor: squeue -j %s,%s\n' "$producer_job" "$converter_job"
