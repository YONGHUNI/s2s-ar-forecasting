#!/usr/bin/env bash

#SBATCH --job-name=graphcast-op
#SBATCH --partition=hu_p
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=128G
#SBATCH --time=20:00:00
#SBATCH --output=graphcast-%j.out
#SBATCH --error=graphcast-%j.err

set -euo pipefail

PROJECT="/work/whlab/$USER/project/weather-ai"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Start:  $(date)"
echo "========================================"

nvidia-smi -L

# Rootless Nix bootstrap once on the allocated node
cd "$BOOTSTRAP"
./bootstrap.sh --store-root "/lscratch/$USER/.nix"

cd "$PROJECT"
mkdir -p logs

# Prepare Pixi environment once
"$NIX" develop --command bash -c '
    set -e
    pixi --version
    pixi install -e graphcast
'

# Two Slurm tasks:
#   task 0 -> worker 0 -> GPU 0
#   task 1 -> worker 1 -> GPU 1
srun \
    --ntasks=2 \
    --cpus-per-task=8 \
    --gpus-per-task=1 \
    --gpu-bind=single:1 \
    "$NIX" develop --command bash -c '
        set -euo pipefail

        export OMP_NUM_THREADS=8
        export OPENBLAS_NUM_THREADS=8
        export MKL_NUM_THREADS=8
        export NUMEXPR_NUM_THREADS=8
        export NC_WORKERS=8

        WORKER="$SLURM_PROCID"

        echo "========================================"
        echo "Worker: $WORKER"
        echo "Node: $(hostname)"
        echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        echo "========================================"

        pixi run -e graphcast \
            python scripts/run_graphcast_operational_season.py \
            --worker "$WORKER" \
            --num-workers 2
    '

echo
echo "========================================"
echo "Completed: $(date)"
echo "========================================"
