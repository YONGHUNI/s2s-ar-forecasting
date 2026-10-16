#!/bin/bash
#SBATCH --job-name=graphcast-qc
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/graphcast-qc-%j.log

set -euo pipefail

REPO="$HOME/work/project/weather-ai"
ROOT="/scratch/$USER/weather-ai/graphcast-operational"
OUTDIR="$REPO/qc"

QC_WORKERS="${QC_WORKERS:-${SLURM_CPUS_PER_TASK:-8}}"
QC_MODE="${QC_MODE:-sampled}"
QC_SKIP_LIGHTWEIGHT="${QC_SKIP_LIGHTWEIGHT:-1}"
QC_SPATIAL_STRIDE="${QC_SPATIAL_STRIDE:-4}"

cd "$REPO"

mkdir -p "$OUTDIR"
mkdir -p "/lscratch/$USER/tmp"

export TMPDIR="/lscratch/$USER/tmp"
export PYTHONUNBUFFERED=1

# Prevent each process from spawning additional BLAS/OpenMP threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

if ! "$HOME/.local/bin/nix" --version >/dev/null 2>&1; then
    cd "$HOME/rootless-nix-bootstrap"
    ./bootstrap.sh --store-root "/lscratch/$USER/.nix"
    cd "$REPO"
fi

echo "============================================================"
echo "GraphCast QC"
echo "Job ID:            $SLURM_JOB_ID"
echo "Node:              $(hostname)"
echo "Root:              $ROOT"
echo "CPUs:              $SLURM_CPUS_PER_TASK"
echo "Workers:           $QC_WORKERS"
echo "Mode:              $QC_MODE"
echo "Skip lightweight:  $QC_SKIP_LIGHTWEIGHT"
echo "Spatial stride:    $QC_SPATIAL_STRIDE"
echo "Started:           $(date)"
echo "============================================================"

ARGS=(
    --root "$ROOT"
    --outdir "$OUTDIR"
    --expected-files 607
    --workers "$QC_WORKERS"
    --seed 42
    --samples-per-season 3
    --deep-mode "$QC_MODE"
    --spatial-stride "$QC_SPATIAL_STRIDE"
)

if [[ "$QC_SKIP_LIGHTWEIGHT" == "1" ]]; then
    ARGS+=(--skip-lightweight)
fi

"$HOME/.local/bin/nix" develop --command \
    pixi run -e graphcast \
    python -u scripts/qc_graphcast.py "${ARGS[@]}"

echo
echo "Finished: $(date)"
