#!/usr/bin/env bash
#SBATCH --job-name=era5-fig-cache
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=36:00:00
#SBATCH --output=fig/logs/era5-%A_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$HOME/.local/bin/nix" develop --command \
  pixi run -e graphcast python -u fig/prepare_era5.py \
    --config fig/config.yaml --month-index "$SLURM_ARRAY_TASK_ID"
