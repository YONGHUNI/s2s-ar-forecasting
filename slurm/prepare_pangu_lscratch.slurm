#!/usr/bin/env bash
#SBATCH --job-name=pangu-lscratch
#SBATCH --partition=hu_p
#SBATCH --account=whlab
#SBATCH --qos=hu_p_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?Submit from repository root}"
cd "$PROJECT"
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(git rev-parse HEAD)"
BOOTSTRAP="$HOME/rootless-nix-bootstrap/bootstrap.sh"
mkdir -p "$HOME/.cache"
(
  flock -x 9
  "$BOOTSTRAP" --store-root "/lscratch/$USER/.nix"
) 9>"$HOME/.cache/rootless-nix-bootstrap.lock"
"$HOME/.local/bin/nix" develop --command bash -c '
  set -euo pipefail
  project="$1"
  bash "$project/scripts/pangu_pixi_run.sh" python -u "$project/scripts/test_pangu_environment.py"
' bash "$PROJECT"
