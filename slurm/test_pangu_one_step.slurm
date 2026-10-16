#!/usr/bin/env bash
#SBATCH --job-name=pangu-one-step
#SBATCH --partition=hu_p
#SBATCH --account=whlab
#SBATCH --qos=hu_p_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
MANIFEST="$PROJECT/environments/pangu/pixi.toml"
NIX="$HOME/.local/bin/nix"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
mkdir -p "$HOME/.cache"
(
  flock -x 9
  "$BOOTSTRAP/bootstrap.sh" --store-root "/lscratch/$USER/.nix"
) 9>"$HOME/.cache/rootless-nix-bootstrap.lock"
cd "$PROJECT"
echo "commit=$(git rev-parse HEAD) node=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi
"$NIX" develop --command bash -c '
  set -euo pipefail
  manifest="$1"
  project="$2"
  lib="$project/environments/pangu/.pixi/envs/default/lib"
  pixi run --manifest-path "$manifest" --locked \
    env LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}" \
    python -u scripts/test_pangu_one_step.py
' bash "$MANIFEST" "$PROJECT"
