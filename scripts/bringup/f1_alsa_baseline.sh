#!/usr/bin/env bash
# F1: resolve SSS1629 by ALSA name, apply safe mixer, sequential capture then playback.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -x "${ROOT}/.venv/bin/python3" ]]; then
  PYTHON="${ROOT}/.venv/bin/python3"
else
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" "${ROOT}/scripts/bringup/f1_alsa_baseline.py" "$@"
