#!/usr/bin/env bash
#SBATCH --job-name=pangu-nc-test
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?Submit from repository root}"
MANIFEST="$PROJECT/environments/pangu/pixi.toml"
NIX="$HOME/.local/bin/nix"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
mkdir -p "$HOME/.cache"
(
 flock -x 9
 "$BOOTSTRAP/bootstrap.sh" --store-root "/lscratch/$USER/.nix"
) 9>"$HOME/.cache/rootless-nix-bootstrap.lock"
cd "$PROJECT"
echo "job=$SLURM_JOB_ID node=$(hostname) git=$(git rev-parse HEAD)"
"$NIX" develop --command bash -c '
 set -euo pipefail
 manifest="$1"
 project="$2"
 lib="$project/environments/pangu/.pixi/envs/default/lib"
 pixi run --manifest-path "$manifest" --locked \
   env LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}" \
   python -u scripts/convert_pangu_42day_test.py
' bash "$MANIFEST" "$PROJECT"
