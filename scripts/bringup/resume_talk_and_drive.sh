#!/usr/bin/env bash
# After a power cycle, restore the live talk-and-drive loop we left on.
# Does not rebuild Cosmos engines. Wheels may move once the ready phrase plays.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONUNBUFFERED=1
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$REPO/third_party/tensorrt-edge-llm/build/libNvInfer_edgellm_plugin.so}"

exec "$REPO/.venv/bin/python3" "$REPO/scripts/bringup/talk_and_drive.py" \
  --auto-listen \
  --mic-seconds 8 \
  --leave-loaded \
  "$@"
