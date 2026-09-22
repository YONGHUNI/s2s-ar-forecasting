#!/usr/bin/env bash

#SBATCH --job-name=graphcast-fcst
#SBATCH --partition=hu_p
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:2
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=graphcast-forecast-%j.out
#SBATCH --error=graphcast-forecast-%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"

[[ -f "$PROJECT/pixi.toml" ]] || {
    echo "pixi.toml not found in PROJECT=$PROJECT" >&2
    exit 1
}

cd "$BOOTSTRAP"
./bootstrap.sh --store-root "/lscratch/$USER/.nix"

cd "$PROJECT"
"$NIX" develop --command bash -c '
    set -euo pipefail
    pixi install -e graphcast
'

srun     --ntasks="$SLURM_NTASKS"     --cpus-per-task="$SLURM_CPUS_PER_TASK"     --gpus-per-task=1     --gpu-bind=single:1     "$NIX" develop --command bash -c '
        set -euo pipefail
        export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"

        echo "Producer=$SLURM_PROCID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

        pixi run -e graphcast             python scripts/run_graphcast_forecast.py             --config config/graphcast_operational.yaml             --worker "$SLURM_PROCID"             --num-workers "$SLURM_NTASKS"
    '
