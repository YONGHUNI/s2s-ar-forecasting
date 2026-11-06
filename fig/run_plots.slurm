#!/usr/bin/env bash
#SBATCH --job-name=graphcast-10panel
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=fig/logs/plot-%A_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MPLBACKEND=Agg

"$HOME/.local/bin/nix" develop --command \
  pixi run -e postprocess python -u -m fig.plot_10panel \
    --config fig/config.yaml --index "$SLURM_ARRAY_TASK_ID"
