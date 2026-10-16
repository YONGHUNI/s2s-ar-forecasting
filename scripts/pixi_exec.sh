#!/usr/bin/env bash
# Run a command from an activated Pixi environment with its own C++ runtime.
set -euo pipefail
py="$(command -v python)"
env_dir="$(dirname "$(dirname "$py")")"
case "$env_dir/" in
  /lscratch/"$USER"/pixi-envs/*) ;;
  *) echo "Pixi environment is not on node-local /lscratch: $env_dir" >&2; exit 1 ;;
esac
export LD_LIBRARY_PATH="$env_dir/lib:${LD_LIBRARY_PATH:-}"
echo "pixi_environment=$env_dir" >&2
exec "$@"
