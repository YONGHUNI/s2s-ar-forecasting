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
ENV_DIR="$PROJECT/environments/pangu/.pixi/envs/default"
PYTHON="$ENV_DIR/bin/python"
cd "$PROJECT"

if [[ ! -x "$PYTHON" ]]; then
  echo "Pangu environment Python missing: $PYTHON" >&2
  echo "Install the locked Pangu environment on a network-enabled compute node first." >&2
  exit 1
fi

export LD_LIBRARY_PATH="$ENV_DIR/lib:${LD_LIBRARY_PATH:-}"
export PATH="$ENV_DIR/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "job=$SLURM_JOB_ID node=$(hostname) git=$(git rev-parse HEAD)"
echo "python=$PYTHON"
"$PYTHON" -u scripts/convert_pangu_42day_test.py
