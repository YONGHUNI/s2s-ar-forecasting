#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

producer_raw="$(sbatch --parsable slurm/run_graphcast_forecast.slurm)"
producer_job="${producer_raw%%;*}"

converter_raw="$(
    sbatch         --parsable         --dependency="after:${producer_job}"         slurm/run_graphcast_convert.slurm
)"
converter_job="${converter_raw%%;*}"

printf 'GraphCast producer job: %s\n' "$producer_job"
printf 'NetCDF converter job:   %s\n' "$converter_job"
printf 'Monitor: squeue -j %s,%s\n' "$producer_job" "$converter_job"
