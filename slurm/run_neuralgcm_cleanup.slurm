#!/usr/bin/env bash
#SBATCH --job-name=neuralgcm-clean
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=neuralgcm-cleanup-%j.out
#SBATCH --error=neuralgcm-cleanup-%j.err

set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"
cd "$PROJECT"
mkdir -p "$HOME/.cache" "/lscratch/$USER"
(
  flock -x 9
  cd "$BOOTSTRAP"
  ./bootstrap.sh --store-root "/lscratch/$USER/.nix"
) 9>"$HOME/.cache/rootless-nix-bootstrap.lock"
(
  flock -x 9
  "$NIX" develop --command bash -c 'set -euo pipefail; pixi install -e postprocess --locked'
) 9>"/lscratch/$USER/pixi-neuralgcm-postprocess-install.lock"

case "${NGCM_CLEANUP_MODE:-audit}" in
  audit) ARGS=(--cleanup-only --dry-run --delete-verified-zarr) ;;
  delete) ARGS=(--cleanup-only --delete-verified-zarr) ;;
  *) echo "NGCM_CLEANUP_MODE must be audit or delete" >&2; exit 2 ;;
esac
echo "NeuralGCM Zarr cleanup mode=${NGCM_CLEANUP_MODE:-audit}"
srun --cpu-bind=none --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$NIX" develop --command bash -c '
    set -euo pipefail
    pixi run -e postprocess --locked -- bash scripts/pixi_exec.sh \
      python -u scripts/convert_neuralgcm_netcdf.py "$@"
  ' bash "${ARGS[@]}"
