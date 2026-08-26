#!/usr/bin/env bash
# F5: fetch the ONNX acoustic model + HiFi-GAN vocoder pair for on-device TTS.
#
# NVIDIA's own FastPitch and HiFi-GAN checkpoints ARE reachable from the
# bring-up sandbox (see the F5 section of docs/bringup/06-voice.md for the exact
# URLs), but NGC publishes them only as `.nemo` archives -- torch checkpoints
# that cannot be loaded or exported without nemo_toolkit, which resolves to
# torch 2.13 plus a CUDA 13 wheel stack that does not match JetPack 6's CUDA
# 12.6 Tegra iGPU. So the mel generator is substituted while the vocoder stage
# stays HiFi-GAN: Matcha-TTS (text -> mel) + HiFi-GAN (mel -> waveform), both
# ONNX, executed by the sherpa-onnx already installed for F4.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${ROOT}/data/models/f5"
TTS_BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models"
VOC_BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models"

ACOUSTIC=matcha-icefall-en_US-ljspeech
VOCODERS=(hifigan_v1.onnx hifigan_v2.onnx)

mkdir -p "$DEST"
cd "$DEST"

if [[ -f "${ACOUSTIC}/model-steps-3.onnx" ]]; then
  echo "have ${ACOUSTIC}"
else
  echo "fetching ${ACOUSTIC}"
  curl -fSL -o "${ACOUSTIC}.tar.bz2" "${TTS_BASE}/${ACOUSTIC}.tar.bz2"
  tar xjf "${ACOUSTIC}.tar.bz2" --no-same-owner --no-same-permissions
fi

for v in "${VOCODERS[@]}"; do
  if [[ -f "$v" ]]; then
    echo "have ${v}"
    continue
  fi
  echo "fetching ${v}"
  curl -fSL -o "$v" "${VOC_BASE}/${v}"
done

echo "models in ${DEST}"
