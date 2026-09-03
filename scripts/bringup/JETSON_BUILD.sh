#!/usr/bin/env bash
# Build Cosmos-Reason2-2B Edge-LLM engines ON THE JETSON (SM87). Do not run on x86.
#
# INT4 AWQ is already baked into the x86 ONNX export. On the Jetson we only build
# engines; generate length is a runtime concern for the orchestrator.
#
# llm_build (v0.10.0) accepts only:
#   --onnxDir --engineDir --maxBatchSize 1 --maxInputLen 3072 --maxKVCacheCapacity 4096
# Never pass --externalize-weights, --int4-gemm-plugin-version, FP8, NVFP4, or
# --memPoolSize on Cosmos. Never re-quantize INT4 on the Nano. Never drive motors.
set -euo pipefail

ROOT="${TENSORRT_EDGELLM_ROOT:-$HOME/TensorRT-Edge-LLM}"
ONNX="${COSMOS_ONNX_DIR:-$HOME/jetbot-thin-stack/cosmos-onnx}"
ENG="${COSMOS_ENGINE_DIR:-$HOME/jetbot-thin-stack/cosmos-engines}"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "Refusing to build engines on $(uname -m). Run this script on the Jetson." >&2
  exit 1
fi

export PATH="/usr/local/cuda-12.6/bin:${PATH}"
export EDGELLM_PLUGIN_PATH="${EDGELLM_PLUGIN_PATH:-$ROOT/build/libNvInfer_edgellm_plugin.so}"

if [[ ! -f "$ONNX/llm/model.onnx" ]]; then
  echo "Cosmos ONNX absent: $ONNX/llm/model.onnx" >&2
  exit 2
fi

# INT4 FFN layout must already be in the ONNX; it is an export-time property.
if ! grep -q 'int4_ffn_weights' "$ONNX/llm/config.json"; then
  echo "ONNX is not INT4-FFN externalized (no int4_ffn_weights in $ONNX/llm/config.json)." >&2
  exit 4
fi
if grep -qiE '"kv_cache_dtype": *"(fp8|nvfp4)"' "$ONNX/llm/config.json"; then
  echo "Refusing FP8/NVFP4 KV on SM87." >&2
  exit 4
fi

echo "TENSORRT_EDGELLM_ROOT=$ROOT"
echo "COSMOS_ONNX_DIR=$ONNX -> $(readlink -f "$ONNX" 2>/dev/null || echo "$ONNX")"
echo "COSMOS_ENGINE_DIR=$ENG -> $(readlink -f "$ENG" 2>/dev/null || echo "$ENG")"
git -C "$ROOT" describe --tags --always 2>/dev/null | sed 's/^/Edge-LLM /' || true

mkdir -p "$ENG/llm" "$ENG"
cd "$ROOT"

if [[ "${SKIP_IF_ENGINES:-}" == "1" && -f "$ENG/llm/llm.engine" && -f "$ENG/visual/visual.engine" ]]; then
  echo "SKIP_IF_ENGINES=1 and both engines exist; not rebuilding."
else
  set -x
  ./build/examples/llm/llm_build \
    --onnxDir "$ONNX/llm" \
    --engineDir "$ENG/llm" \
    --maxBatchSize 1 \
    --maxInputLen 3072 \
    --maxKVCacheCapacity 4096

  ./build/examples/multimodal/visual_build \
    --onnxDir "$ONNX/visual" \
    --engineDir "$ENG" \
    --minImageTokens 64 \
    --maxImageTokens 280 \
    --maxImageTokensPerImage 280
  set +x
fi

echo "Engines under $ENG"
echo "Next: tegrastats while loading the LLM+visual engines. Expect ~4.3-4.7 GiB for Cosmos, not 2.1."
