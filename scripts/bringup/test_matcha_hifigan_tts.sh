#!/usr/bin/env bash
# F5 gate. Set JETBOT_F5_PLAYBACK=1 to also play the WAVs once through the
# name-resolved SSS1629 endpoint at a capped volume with sidetone forced off.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/.venv/bin/python3"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"${ROOT}/scripts/bringup/fetch_tts_models.sh"
exec "$PYTHON" "${ROOT}/scripts/bringup/test_matcha_hifigan_tts.py" "$@"
