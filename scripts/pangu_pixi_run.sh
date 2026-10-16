#!/usr/bin/env bash
# Run a command in the Pangu Pixi workspace with node-local detached environments.
# Invoke from within the project's Nix develop shell (pixi must be on PATH).
set -euo pipefail
PROJECT="${PROJECT:-$(pwd)}"
MANIFEST="$PROJECT/environments/pangu/pixi.toml"
CONFIG_DIR="$PROJECT/environments/pangu/.pixi"
test -f "$MANIFEST"
test -f "$PROJECT/environments/pangu/pixi.lock" || {
  echo "Missing Pangu pixi.lock; refusing non-reproducible installation" >&2
  exit 1
}
mkdir -p "$CONFIG_DIR" "/lscratch/$USER/pixi-envs" "/lscratch/$USER/pixi-cache"
printf 'detached-environments = "/lscratch/%s/pixi-envs"\n' "$USER" > "$CONFIG_DIR/config.toml"
export PIXI_CACHE_DIR="/lscratch/$USER/pixi-cache"
echo "pangu_pixi_config=$(cat "$CONFIG_DIR/config.toml")"
# pixi run may materialize environment at first use; --locked prohibits lock changes.
pixi run --manifest-path "$MANIFEST" --locked -- bash -c '
  set -euo pipefail
  py="$(command -v python)"
  env_dir="$(dirname "$(dirname "$py")")"
  case "$env_dir/" in
    /lscratch/"$USER"/pixi-envs/*) ;;
    *) echo "ERROR: Pangu Pixi environment is not on /lscratch: $env_dir" >&2; exit 1 ;;
  esac
  export LD_LIBRARY_PATH="$env_dir/lib:${LD_LIBRARY_PATH:-}"
  echo "pangu_python=$py"
  echo "pangu_environment=$env_dir"
  exec "$@"
' bash "$@"
