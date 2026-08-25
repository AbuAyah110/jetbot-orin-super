#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/.venv/bin/python3"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" "${ROOT}/scripts/bringup/test_webrtc_apm.py" "$@"
