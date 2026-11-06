#!/usr/bin/env bash
# Submit staged reference fetch and per-initialization GIF generation.
set -euo pipefail
mode="${1:-}"
if [[ "$mode" != "all" && "$mode" != "prepare" && "$mode" != "plots" ]]; then
  echo "Usage: bash fig/submit.sh {prepare|plots|all}" >&2
  exit 2
fi
if [[ ! -f fig/config.yaml || ! -f pixi.toml ]]; then
  echo "Run this from the weather-ai repository root" >&2
  exit 2
fi

mkdir -p fig/logs
PREP_CONCURRENCY="${PREP_CONCURRENCY:-4}"
PLOT_CONCURRENCY="${PLOT_CONCURRENCY:-8}"
COUNT=$("$HOME/.local/bin/nix" develop --command pixi run -e postprocess \
  python fig/prepare_era5.py --config fig/config.yaml --count-months)
if [[ ! "$COUNT" =~ ^[0-9]+$ || "$COUNT" -lt 1 ]]; then
  echo "Invalid ERA5 month count: $COUNT" >&2
  exit 1
fi
end=$((COUNT - 1))
echo "ERA5 months: $COUNT; 607 forecast cases"

if [[ "$mode" == "all" || "$mode" == "prepare" ]]; then
  prep=$(sbatch --parsable --array="0-$end%$PREP_CONCURRENCY" fig/run_prepare.slurm)
  prep="${prep%%;*}"
  echo "Submitted ERA5 preparation array: $prep"
fi
if [[ "$mode" == "all" || "$mode" == "plots" ]]; then
  dependency=()
  if [[ "$mode" == "all" ]]; then
    dependency=("--dependency=afterok:$prep")
  fi
  job=$(sbatch --parsable --array="0-606%$PLOT_CONCURRENCY" \
    "${dependency[@]}" fig/run_plots.slurm)
  echo "Submitted 607 GIF jobs: $job"
fi
