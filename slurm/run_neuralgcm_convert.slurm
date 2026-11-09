#!/usr/bin/env bash
#SBATCH --job-name=neuralgcm-nc
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail
PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit from repository root}}"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"
[[ -f "$PROJECT/pixi.toml" ]] || { echo "Missing pixi.toml" >&2; exit 1; }
[[ -f "$PROJECT/scripts/convert_neuralgcm_netcdf.py" ]] || {
  echo "Missing NeuralGCM NetCDF converter" >&2; exit 1;
}

mkdir -p "$HOME/.cache" "/lscratch/$USER"
(
  flock -x 9
  cd "$BOOTSTRAP"
  ./bootstrap.sh --store-root "/lscratch/$USER/.nix"
) 9>"$HOME/.cache/rootless-nix-bootstrap.lock"

cd "$PROJECT"
(
  flock -x 9
  "$NIX" develop --command bash -c 'set -euo pipefail; pixi install -e postprocess --locked'
) 9>"/lscratch/$USER/pixi-neuralgcm-postprocess-install.lock"

export CONVERTER_REQUEUE_AFTER_SECONDS="${CONVERTER_REQUEUE_AFTER_SECONDS:-165600}"
echo "NeuralGCM converter producer=${PRODUCER_JOB_ID:-untracked} restart=${SLURM_RESTART_COUNT:-0}"

set +e
srun --cpu-bind=none --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$NIX" develop --command bash -c '
  set -euo pipefail
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  LIMIT_ARGS=()
  if [[ -n "${NGCM_CONVERT_LIMIT:-}" ]]; then
    [[ "$NGCM_CONVERT_LIMIT" =~ ^[1-9][0-9]*$ ]] || { echo "NGCM_CONVERT_LIMIT must be positive integer" >&2; exit 2; }
    LIMIT_ARGS=(--limit "$NGCM_CONVERT_LIMIT")
  fi
  pixi run -e postprocess --locked -- bash scripts/pixi_exec.sh \
    python -u scripts/convert_neuralgcm_netcdf.py \
      --producer-job-id "${PRODUCER_JOB_ID:-}" \
      --workers 12 --compression-level 2 "${LIMIT_ARGS[@]}"
'
status=$?
set -e

if [[ "$status" -eq 75 ]]; then
  echo "Converter requests Slurm requeue: job=$SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" || exit 75
  exit 0
fi
exit "$status"
