#!/usr/bin/env bash

#SBATCH --job-name=graphcast-fcst
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --array=0-19%8
#SBATCH --output=graphcast-forecast-%A_%a.out
#SBATCH --error=graphcast-forecast-%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
CONFIG="${CONFIG:-config/graphcast_operational.yaml}"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"

[[ -f "$PROJECT/pixi.toml" ]] || {
    echo "pixi.toml not found in PROJECT=$PROJECT" >&2
    exit 1
}

[[ -f "$PROJECT/$CONFIG" ]] || {
    echo "GraphCast config not found: $PROJECT/$CONFIG" >&2
    exit 1
}

cd "$BOOTSTRAP"
./bootstrap.sh --store-root "/lscratch/$USER/.nix"

cd "$PROJECT"
"$NIX" develop --command bash -c '
    set -euo pipefail
    pixi install -e graphcast
'

export CONFIG

NUM_SHARDS=20
if [[ "${SLURM_ARRAY_TASK_COUNT:-0}" -ne "$NUM_SHARDS" ]]; then
    echo "Expected $NUM_SHARDS array tasks, got ${SLURM_ARRAY_TASK_COUNT:-unset}" >&2
    exit 1
fi
export NUM_SHARDS

srun     --ntasks="$SLURM_NTASKS"     --cpus-per-task="$SLURM_CPUS_PER_TASK"     --gpus-per-task=1     --gpu-bind=single:1     "$NIX" develop --command bash -c '
        set -euo pipefail
        export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops

        GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
        JAX_CACHE_TAG="$(
            printf '%s' "$GPU_NAME" \
            | sed -E 's/^NVIDIA[[:space:]]+//' \
            | tr '[:upper:]' '[:lower:]' \
            | sed -E 's/[^a-z0-9]+/-/g; s/^-+//; s/-+$//'
        )"
        JAX_CACHE_TAG="${JAX_CACHE_TAG:-generic}"
        export JAX_COMPILATION_CACHE_DIR="/scratch/$USER/weather-ai/jax-cache/$JAX_CACHE_TAG"
        mkdir -p "$JAX_COMPILATION_CACHE_DIR"

        echo "Array producer=$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CONFIG=$CONFIG XLA_FLAGS=$XLA_FLAGS"
        echo "GPU_NAME=$GPU_NAME JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR"

        pixi run -e graphcast \
            python scripts/run_graphcast_forecast.py \
            --config "$CONFIG" \
            --worker "$SLURM_ARRAY_TASK_ID" \
            --num-workers "$NUM_SHARDS"
    '
