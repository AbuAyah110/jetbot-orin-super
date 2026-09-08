#!/usr/bin/env bash
# Fetch the compact CPU-only Zipformer ASR + Piper VITS TTS model pair.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ASR_ROOT="${ROOT}/data/models/zipformer"
TTS_ROOT="${ROOT}/data/models/piper"
ASR_NAME="sherpa-onnx-zipformer-small-en-2023-06-26"
TTS_NAME="vits-piper-en_US-lessac-low-int8"
ASR_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${ASR_NAME}.tar.bz2"
TTS_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/${TTS_NAME}.tar.bz2"
ASR_SHA256="c8bff1091c26c49731cddbcd60ef18061142ea11523df1b73bf1b14451b9c15e"
TTS_SHA256="af63fbe60d8bdcfccdee61ba057304a11dfc077145da383d4d351ec3c594d5e2"

fetch_extract() {
  local root="$1" name="$2" url="$3" sha="$4" marker="$5"
  if [[ -f "${root}/${name}/${marker}" ]]; then
    echo "have ${name}"
    return
  fi
  mkdir -p "$root"
  local archive="${root}/${name}.tar.bz2"
  curl -fL --retry 3 -o "$archive" "$url"
  echo "${sha}  ${archive}" | sha256sum --check -
  tar -xjf "$archive" -C "$root" --no-same-owner --no-same-permissions
  rm -f "$archive"
}

fetch_extract "$ASR_ROOT" "$ASR_NAME" "$ASR_URL" "$ASR_SHA256" \
  "encoder-epoch-99-avg-1.int8.onnx"
fetch_extract "$TTS_ROOT" "$TTS_NAME" "$TTS_URL" "$TTS_SHA256" \
  "en_US-lessac-low.onnx"

# The ASR release contains fp32 duplicates. The runtime always selects int8.
rm -f \
  "${ASR_ROOT}/${ASR_NAME}/encoder-epoch-99-avg-1.onnx" \
  "${ASR_ROOT}/${ASR_NAME}/decoder-epoch-99-avg-1.onnx" \
  "${ASR_ROOT}/${ASR_NAME}/joiner-epoch-99-avg-1.onnx"

echo "Zipformer and Piper models are ready under ${ROOT}/data/models"
