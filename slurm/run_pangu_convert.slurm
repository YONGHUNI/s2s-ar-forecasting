#!/usr/bin/env bash

#SBATCH --job-name=pangu-nc
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

PROJECT="${PROJECT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
CONFIG="${CONFIG:-config/pangu_operational.yaml}"
BOOTSTRAP="$HOME/rootless-nix-bootstrap"
NIX="$HOME/.local/bin/nix"

[[ -f "$PROJECT/pixi.toml" ]] || {
    echo "pixi.toml not found in PROJECT=$PROJECT" >&2
    exit 1
}

[[ -f "$PROJECT/$CONFIG" ]] || {
    echo "Pangu6 config not found: $PROJECT/$CONFIG" >&2
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
        pixi install -e graphcast --locked
    '
) 9>"/lscratch/$USER/pixi-pangu-install.lock"

export CONFIG
# Hand control back to Slurm before the 48-hour allocation expires. The
# Python manager stops accepting new work at this threshold, drains active
# conversions, and exits with REQUEUE_EXIT_CODE=75 when more lifecycle work
# may remain. Forty-six hours leaves two hours for slow in-flight encodes to
# finish cleanly before the allocation hard limit.
export CONVERTER_REQUEUE_AFTER_SECONDS="${CONVERTER_REQUEUE_AFTER_SECONDS:-165600}"

echo "Pangu6 converter uses shared GraphCast Pixi I/O feature (netcdf4) until dedicated I/O feature is locked." 
echo "Converter lifecycle restart_count=${SLURM_RESTART_COUNT:-0} requeue_after_seconds=$CONVERTER_REQUEUE_AFTER_SECONDS producer_job=${PRODUCER_JOB_ID:-untracked}"

set +e
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
     "$NIX" develop --command bash -c '
  set -euo pipefail
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  echo "Pangu converter CONFIG=$CONFIG"
  pixi run -e graphcast --locked -- bash scripts/pixi_exec.sh \
    python -u scripts/convert_netcdf.py --config "$CONFIG"
'
status=$?
set -e

if [[ "$status" -eq 75 ]]; then
    echo "Converter requested lifecycle requeue for Slurm job $SLURM_JOB_ID."
    if ! scontrol requeue "$SLURM_JOB_ID"; then
        echo "Failed to requeue converter job $SLURM_JOB_ID." >&2
        exit 75
    fi
    # scontrol normally tears down this allocation immediately. If execution
    # continues briefly, exit successfully rather than marking the old run as
    # an application failure.
    exit 0
fi

exit "$status"
