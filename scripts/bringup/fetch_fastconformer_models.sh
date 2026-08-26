#!/usr/bin/env bash
# F4: fetch NVIDIA NeMo FastConformer CTC checkpoints exported to ONNX.
#
# huggingface.co is unreachable from the bring-up sandbox and the NGC model API
# requires an authenticated key, so these come from the k2-fsa/sherpa-onnx
# release mirror, which redistributes the NVIDIA NeMo exports plus tokens and
# real LibriSpeech test utterances with reference transcripts.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${ROOT}/data/models/f4"
BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"

MODELS=(
  sherpa-onnx-nemo-fast-conformer-ctc-en-24500-int8
  sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms-int8
)

mkdir -p "$DEST"
cd "$DEST"
for m in "${MODELS[@]}"; do
  if [[ -f "${m}/model.int8.onnx" ]]; then
    echo "have ${m}"
    continue
  fi
  echo "fetching ${m}"
  curl -fSL -o "${m}.tar.bz2" "${BASE}/${m}.tar.bz2"
  tar xjf "${m}.tar.bz2" --no-same-owner --no-same-permissions
done
echo "models in ${DEST}"
