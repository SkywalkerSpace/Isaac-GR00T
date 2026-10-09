#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GR00T_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

CONFIG_PATH="${1:-$SCRIPT_DIR/finetune_config.yaml}"
PROFILES_PATH="${2:-$SCRIPT_DIR/profiles.yaml}"
CONFIG_PATH="$(cd -- "$(dirname -- "$CONFIG_PATH")" && pwd)/$(basename -- "$CONFIG_PATH")"
PROFILES_PATH="$(cd -- "$(dirname -- "$PROFILES_PATH")" && pwd)/$(basename -- "$PROFILES_PATH")"

cd "$GR00T_ROOT"
python "$SCRIPT_DIR/finetune.py" \
  --config "$CONFIG_PATH" \
  --profiles "$PROFILES_PATH"
