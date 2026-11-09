#!/usr/bin/env bash
#SBATCH --job-name=neuralgcm-fc
#SBATCH --partition=hu_p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-19%2
#SBATCH --requeue
#SBATCH --output=neuralgcm-forecast-%A_%a.out
#SBATCH --error=neuralgcm-forecast-%A_%a.err

set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"
[[ -f "$PROJECT/pixi.toml" ]] || { echo "pixi.toml missing" >&2; exit 1; }
[[ -f "$PROJECT/scripts/run_neuralgcm_batch.py" ]] || { echo "batch driver missing" >&2; exit 1; }

mkdir -p "$HOME/.cache" "/lscratch/$USER"
(
    flock -x 9
    cd "$BOOTSTRAP"
    ./bootstrap.sh --store-root "/lscratch/$USER/.nix"
) 9>"$HOME/.cache/rootless-nix-bootstrap.lock"

cd "$PROJECT"
(
    flock -x 9
    "$NIX" develop --command bash -c 'set -euo pipefail; pixi install -e neuralgcm'
) 9>"/lscratch/$USER/pixi-neuralgcm-install.lock"

NUM_SHARDS=20
[[ -n "${SLURM_ARRAY_TASK_ID:-}" ]] || { echo "Submit as Slurm array" >&2; exit 1; }
(( SLURM_ARRAY_TASK_ID >= 0 && SLURM_ARRAY_TASK_ID < NUM_SHARDS )) || exit 1
export NUM_SHARDS

srun --cpu-bind=none --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --gpus-per-task=1 --gpu-bind=single:1 \
  "$NIX" develop --command bash -c '
    set -euo pipefail
    export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
    echo "NeuralGCM worker=$SLURM_ARRAY_TASK_ID / $NUM_SHARDS gpu=$CUDA_VISIBLE_DEVICES limit=${NGCM_LIMIT:-all}"
    LIMIT_ARGS=()
    if [[ -n "${NGCM_LIMIT:-}" ]]; then
      [[ "$NGCM_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "NGCM_LIMIT must be a positive integer" >&2; exit 2; }
      LIMIT_ARGS=(--limit "$NGCM_LIMIT")
    fi
    pixi run -e neuralgcm -- python -u scripts/run_neuralgcm_batch.py \
      --worker "$SLURM_ARRAY_TASK_ID" --num-workers "$NUM_SHARDS" "${LIMIT_ARGS[@]}"
  '
