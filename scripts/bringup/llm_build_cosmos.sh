#!/usr/bin/env bash
# Build SM87 TensorRT LLM engines for Cosmos-Reason2-2B (Edge-LLM v0.10.0).
# Do not run until workstation INT4 ONNX is rsynced. Never drive motors.
#
# Workstation export (already baked into ONNX; llm_build rejects these flags):
#   --externalize-weights int4_ffn --int4-gemm-plugin-version 1
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EDGELLM_ROOT="${EDGELLM_ROOT:-$REPO_ROOT/third_party/tensorrt-edge-llm}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$REPO_ROOT/data/edgellm/cosmos}"
ONNX_DIR="${ONNX_DIR:-$WORKSPACE_DIR/onnx}"
ONNX_LLM="$ONNX_DIR/llm/model.onnx"
LLM_BUILD="${LLM_BUILD:-$EDGELLM_ROOT/build/examples/llm/llm_build}"

export PATH="/usr/local/cuda-12.6/bin:${PATH}"
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so}"

if [[ ! -f "$ONNX_LLM" ]]; then
  echo "Cosmos ONNX absent: missing $ONNX_LLM" >&2
  echo "On the x86 workstation, after INT4 AWQ export:" >&2
  echo "  rsync -avP --checksum ~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4/onnx/ \\" >&2
  echo "    impulse110@192.168.50.65:~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/" >&2
  echo "Do not llm_build until that tree has onnx/llm/model.onnx." >&2
  exit 2
fi

if [[ ! -x "$LLM_BUILD" ]]; then
  echo "llm_build missing: $LLM_BUILD" >&2
  exit 3
fi

tag="$(git -C "$EDGELLM_ROOT" describe --tags --always)"
if [[ "$tag" != v0.10.0* ]]; then
  echo "Edge-LLM pin mismatch: expected v0.10.0, got $tag" >&2
  exit 3
fi

echo "Edge-LLM $tag  llm_build=$LLM_BUILD"
echo "ONNX dir $ONNX_DIR"
echo "Export flags already in ONNX: --externalize-weights int4_ffn --int4-gemm-plugin-version 1"
echo "llm_build (v0.10.0) accepts: --maxBatchSize 1 --maxInputLen 3072 --maxKVCacheCapacity 4096"

mkdir -p "$WORKSPACE_DIR/engines/llm" "$WORKSPACE_DIR/logs"
"$LLM_BUILD" \
  --onnxDir "$ONNX_DIR/llm" \
  --engineDir "$WORKSPACE_DIR/engines/llm" \
  --maxBatchSize 1 \
  --maxInputLen 3072 \
  --maxKVCacheCapacity 4096
