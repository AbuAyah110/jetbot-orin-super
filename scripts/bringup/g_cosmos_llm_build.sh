#!/usr/bin/env bash
# Compatibility name retained for issue links. The canonical guarded builder is
# llm_build_cosmos.sh. It exits 2 before invoking TensorRT when ONNX is absent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/llm_build_cosmos.sh" "$@"
