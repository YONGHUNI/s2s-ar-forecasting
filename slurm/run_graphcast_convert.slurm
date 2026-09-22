#!/usr/bin/env bash

#SBATCH --job-name=graphcast-nc
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=graphcast-convert-%j.out
#SBATCH --error=graphcast-convert-%j.err

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

srun     --ntasks="$SLURM_NTASKS"     --cpus-per-task="$SLURM_CPUS_PER_TASK"     "$NIX" develop --command bash -c '
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        export NC_WORKERS="$SLURM_CPUS_PER_TASK"

        pixi run -e graphcast             python scripts/convert_graphcast_netcdf.py             --config config/graphcast_operational.yaml             --worker "$SLURM_PROCID"             --num-workers "$SLURM_NTASKS"
    '
