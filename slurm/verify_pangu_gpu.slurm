#!/usr/bin/env bash
#SBATCH --job-name=pangu-gpu-check
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
PROJECT="${SLURM_SUBMIT_DIR:?Submit from project root}"
MANIFEST="$PROJECT/environments/pangu/pixi.toml"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"
cd "$PROJECT"
echo "job=$SLURM_JOB_ID node=$(hostname) commit=$(git rev-parse HEAD)"
# Monitor the allocated GPU throughout the test.
(
 while true; do
  echo "=== GPU sample $(date --iso-8601=seconds) ==="
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
  sleep 2
 done
) &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
"$NIX" develop --command bash -c '
 set -euo pipefail
 project="$1"
 bash "$project/scripts/pangu_pixi_run.sh" python -u scripts/verify_pangu_gpu.py
' bash "$PROJECT"
