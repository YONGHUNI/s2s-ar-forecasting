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
mkdir -p "$HOME/.cache"
(
 flock -x 9
 "$BOOTSTRAP/bootstrap.sh" --store-root "/lscratch/$USER/.nix"
) 9>"$HOME/.cache/rootless-nix-bootstrap.lock"
echo "job=$SLURM_JOB_ID node=$(hostname) git=$(git rev-parse HEAD)"
nvidia-smi
"$NIX" develop --command bash -c '
 set -euo pipefail
 manifest="$1"
 project="$2"
 lib="$project/environments/pangu/.pixi/envs/default/lib"
 pixi run --manifest-path "$manifest" --locked \
   env LD_LIBRARY_PATH="$lib:${LD_LIBRARY_PATH:-}" \
   python -u scripts/run_pangu_42day_test.py --config config/pangu_42day_test.yaml
' bash "$MANIFEST" "$PROJECT"
