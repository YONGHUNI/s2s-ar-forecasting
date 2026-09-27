#!/usr/bin/env bash

#SBATCH --job-name=graphcast-nc
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=graphcast-convert-%j.out
#SBATCH --error=graphcast-convert-%j.err

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
CONFIG="${CONFIG:-config/graphcast_operational_deterministic.yaml}"
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
        pixi install -e graphcast
    '
) 9>"/lscratch/$USER/pixi-graphcast-install.lock"

export CONFIG

srun     --ntasks=1     --cpus-per-task="$SLURM_CPUS_PER_TASK"     "$NIX" develop --command bash -c '
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1

        echo "Converter manager CONFIG=$CONFIG"

        pixi run -e graphcast             python scripts/convert_graphcast_netcdf.py             --config "$CONFIG"
    '
