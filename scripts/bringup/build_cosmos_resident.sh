#!/usr/bin/env bash
# Compile tracked cosmos_resident.cpp using the existing llm_inference device-link
# object so CUDA kernel registration from libedgellmCore.a resolves.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EDGELLM="${TENSORRT_EDGELLM_ROOT:-$HOME/TensorRT-Edge-LLM}"
BUILD="$EDGELLM/build/examples/llm"
SRC="$ROOT/scripts/bringup/cosmos_resident.cpp"
OBJ="$ROOT/scripts/bringup/cosmos_resident.o"
OUT="$ROOT/scripts/bringup/cosmos_resident"
CUDA_INC=/usr/local/cuda/targets/aarch64-linux/include
CUDA_LIB=/usr/local/cuda/targets/aarch64-linux/lib

g++ -std=gnu++17 -O3 -DNDEBUG \
  -Wno-deprecated-declarations -Wall -Wno-error=unused-parameter \
  -DTRT_EDGELLM_CUDA_LIBRARY_T_COMPAT \
  -I"$EDGELLM/cpp" \
  -I"$EDGELLM/examples/multimodal" \
  -I"$EDGELLM/3rdParty/nlohmannJson/include" \
  -I"$EDGELLM/3rdParty/stb" \
  -I"$EDGELLM/3rdParty/miniaudio" \
  -I"$EDGELLM/examples/utils" \
  -isystem "$CUDA_INC" \
  -c "$SRC" -o "$OBJ"

g++ -Wno-deprecated-declarations -Wall -Wno-error=unused-parameter -O3 -DNDEBUG \
  -L"$CUDA_LIB" -L"$CUDA_LIB/stubs" \
  -Wl,--unresolved-symbols=ignore-in-shared-libs \
  "$OBJ" \
  "$BUILD/CMakeFiles/llm_inference.dir/cmake_device_link.o" \
  -o "$OUT" \
  -Wl,-rpath,"$CUDA_LIB" \
  "$EDGELLM/build/cpp/libedgellmCore.a" \
  "$EDGELLM/build/examples/utils/libexampleUtils.a" \
  "$EDGELLM/build/cpp/libedgellmCore.a" \
  -ldl \
  "$EDGELLM/cpp/kernels/cuteDSLArtifact/aarch64/sm_87/libcutedsl_aarch64.a" \
  "$EDGELLM/build/cpp/libtrt_edgellm_cutedsl_cudart_shim.a" \
  -lcuda \
  /usr/lib/aarch64-linux-gnu/libnvonnxparser.so \
  /usr/lib/aarch64-linux-gnu/libnvinfer.so \
  "$CUDA_LIB/libcudart.so" \
  -lcudadevrt -lcudart_static -lrt -lpthread -ldl

echo "built $OUT"
