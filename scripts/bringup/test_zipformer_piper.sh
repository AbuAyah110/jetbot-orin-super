#!/usr/bin/env bash
# Offline by default. Add --live-capture for a short name-resolved mic capture.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${JETBOT_PYTHON:-${ROOT}/.venv/bin/python3}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-1}"

"${ROOT}/scripts/bringup/fetch_voice_models.sh"
exec "$PYTHON" "${ROOT}/scripts/bringup/test_zipformer_piper.py" "$@"
