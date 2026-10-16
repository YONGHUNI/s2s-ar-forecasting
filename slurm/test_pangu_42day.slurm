#!/usr/bin/env bash
#SBATCH --job-name=pangu-42day
#SBATCH --partition=hu_p
#SBATCH --account=whlab
#SBATCH --qos=hu_p_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:L40S:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
PROJECT="${SLURM_SUBMIT_DIR:?Submit from repository root}"
MANIFEST="$PROJECT/environments/pangu/pixi.toml"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"
cd "$PROJECT"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse HEAD)"
"$NIX" develop --command bash -c '
 set -euo pipefail
 project="$1"
 bash "$project/scripts/pangu_pixi_run.sh" python -u scripts/run_pangu_42day_test.py --config config/pangu_42day_test.yaml
' bash "$PROJECT"
