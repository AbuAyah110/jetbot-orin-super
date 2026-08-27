#!/usr/bin/env bash
# Build SM87 TensorRT engines for Cosmos-Reason2-2B INT4 ONNX (Edge-LLM v0.10.0).
# Does not drive motors. Does not use Qwen2.5-VL artifacts.
# Export flags (--externalize-weights int4_ffn --int4-gemm-plugin-version 1) are
# workstation-only; llm_build does not accept them.
set -euo pipefail

EDGELLM_ROOT="${EDGELLM_ROOT:-$HOME/Documents/_edgellm_ref/repo}"
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4}"
ONNX_LLM="$WORKSPACE_DIR/onnx/llm/model.onnx"
ONNX_VIS="$WORKSPACE_DIR/onnx/visual/model.onnx"
LLM_BUILD="$EDGELLM_ROOT/build/examples/llm/llm_build"
VIS_BUILD="$EDGELLM_ROOT/build/examples/multimodal/visual_build"

export PATH="/usr/local/cuda-12.6/bin:${PATH}"
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so}"

if [[ ! -f "$ONNX_LLM" ]]; then
  echo "Cosmos ONNX absent: missing $ONNX_LLM" >&2
  echo "Rsync workstation INT4 export to: $WORKSPACE_DIR/onnx/" >&2
  echo "Do not use ~/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-ModelOpt-INT4" >&2
  exit 2
fi

tag="$(git -C "$EDGELLM_ROOT" describe --tags --always)"
if [[ "$tag" != v0.10.0* ]]; then
  echo "Edge-LLM pin mismatch: expected v0.10.0, got $tag" >&2
  exit 3
fi

echo "Building LLM engine from $ONNX_LLM"
mkdir -p "$WORKSPACE_DIR/engines/llm" "$WORKSPACE_DIR/logs"
"$LLM_BUILD" \
  --onnxDir "$WORKSPACE_DIR/onnx/llm" \
  --engineDir "$WORKSPACE_DIR/engines/llm" \
  --maxBatchSize 1 \
  --maxInputLen 3072 \
  --maxKVCacheCapacity 4096

if [[ -f "$ONNX_VIS" ]]; then
  echo "Building visual engine from $ONNX_VIS"
  "$VIS_BUILD" \
    --onnxDir "$WORKSPACE_DIR/onnx/visual" \
    --engineDir "$WORKSPACE_DIR/engines" \
    --minImageTokens 8 \
    --maxImageTokens 2048 \
    --maxImageTokensPerImage 2048
else
  echo "Visual ONNX not present yet ($ONNX_VIS); LLM engine done." >&2
fi
