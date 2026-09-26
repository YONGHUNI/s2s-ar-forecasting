#!/usr/bin/env bash

#SBATCH --job-name=gc-h100-samearch
#SBATCH --partition=gpu_p
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:H100:2
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=graphcast-forecast-%j.out
#SBATCH --error=graphcast-forecast-%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
CONFIG_A="${CONFIG_A:-config/graphcast_benchmark_42d_h100_samearch_a.yaml}"
CONFIG_B="${CONFIG_B:-config/graphcast_benchmark_42d_h100_samearch_b.yaml}"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"

[[ -f "$PROJECT/pixi.toml" ]] || {
    echo "pixi.toml not found in PROJECT=$PROJECT" >&2
    exit 1
}

for CONFIG in "$CONFIG_A" "$CONFIG_B"; do
    [[ -f "$PROJECT/$CONFIG" ]] || {
        echo "GraphCast config not found: $PROJECT/$CONFIG" >&2
        exit 1
    }
done

cd "$BOOTSTRAP"
./bootstrap.sh --store-root "/lscratch/$USER/.nix"

cd "$PROJECT"
"$NIX" develop --command bash -c '
    set -euo pipefail
    pixi install -e graphcast
'

export CONFIG_A CONFIG_B

srun     --label     --ntasks="$SLURM_NTASKS"     --cpus-per-task="$SLURM_CPUS_PER_TASK"     --gpus-per-task=1     --gpu-bind=single:1     "$NIX" develop --command bash -c '
        set -euo pipefail
        export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
        export XLA_FLAGS=--xla_gpu_exclude_nondeterministic_ops

        if [[ "$SLURM_PROCID" -eq 0 ]]; then
            CONFIG="$CONFIG_A"
        else
            CONFIG="$CONFIG_B"
        fi

        echo "Replica=$SLURM_PROCID CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CONFIG=$CONFIG XLA_FLAGS=$XLA_FLAGS"

        pixi run -e graphcast \
            python scripts/run_graphcast_forecast.py \
            --config "$CONFIG" \
            --worker 0 \
            --num-workers 1
    '
