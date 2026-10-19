#!/usr/bin/env bash

#SBATCH --job-name=sfno-fcst
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-19%2
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
CONFIG="${CONFIG:-config/sfno_operational.yaml}"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"

[[ -f "$PROJECT/pixi.toml" ]] || {
    echo "pixi.toml not found in PROJECT=$PROJECT" >&2
    exit 1
}

[[ -f "$PROJECT/$CONFIG" ]] || {
    echo "SFNO config not found: $PROJECT/$CONFIG" >&2
    exit 1
}

mkdir -p "$HOME/.cache"
(
    flock -x 9
    cd "$BOOTSTRAP"
    ./bootstrap.sh --store-root "/lscratch/$USER/.nix"
) 9>"$HOME/.cache/rootless-nix-bootstrap.lock"

cd "$PROJECT"
(
    flock -x 9
    "$NIX" develop --command bash -c '
        set -euo pipefail
        pixi install -e sfno --locked
    '
) 9>"/lscratch/$USER/pixi-sfno-install.lock"

export CONFIG

NUM_SHARDS=20
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This producer must be submitted as a Slurm array task." >&2
    exit 1
fi

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= NUM_SHARDS )); then
    echo "Array task ID $SLURM_ARRAY_TASK_ID is outside logical shard range 0-$((NUM_SHARDS - 1))." >&2
    exit 1
fi
export NUM_SHARDS

srun --ntasks="$SLURM_NTASKS" \
     --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     --gpus-per-task=1 --gpu-bind=single:1 \
     "$NIX" develop --command bash -c '
  set -euo pipefail
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
  export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
  export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
  export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
  echo "SFNO producer=$SLURM_ARRAY_TASK_ID shards=$NUM_SHARDS config=$CONFIG gpu=$CUDA_VISIBLE_DEVICES"
  pixi run -e sfno --locked -- bash scripts/pixi_exec.sh \
    python -u scripts/run_forecast.py \
    --config "$CONFIG" --worker "$SLURM_ARRAY_TASK_ID" --num-workers "$NUM_SHARDS"
'
