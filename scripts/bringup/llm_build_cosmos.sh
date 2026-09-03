#!/usr/bin/env bash
# Guarded Cosmos SM87 engine build. Delegates to JETSON_BUILD.sh with locked flags.
# Never drive motors. Do not install PyTorch.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export TENSORRT_EDGELLM_ROOT="${EDGELLM_ROOT:-$REPO_ROOT/third_party/tensorrt-edge-llm}"
export COSMOS_ONNX_DIR="${ONNX_DIR:-$REPO_ROOT/data/edgellm/cosmos/onnx}"
export COSMOS_ENGINE_DIR="${WORKSPACE_DIR:-$REPO_ROOT/data/edgellm/cosmos}/engines"
# Prefer thin-stack compatibility names when those symlinks exist.
if [[ -d "$HOME/jetbot-thin-stack/cosmos-onnx" ]]; then
  export COSMOS_ONNX_DIR="$HOME/jetbot-thin-stack/cosmos-onnx"
fi
if [[ -d "$HOME/jetbot-thin-stack/cosmos-engines" ]]; then
  export COSMOS_ENGINE_DIR="$HOME/jetbot-thin-stack/cosmos-engines"
fi
exec "$REPO_ROOT/scripts/bringup/JETSON_BUILD.sh" "$@"
