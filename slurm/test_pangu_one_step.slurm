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
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse HEAD)"
"$NIX" develop --command bash -c '
 set -euo pipefail
 project="$1"
 bash "$project/scripts/pangu_pixi_run.sh" python -u scripts/test_pangu_one_step.py
' bash "$PROJECT"
