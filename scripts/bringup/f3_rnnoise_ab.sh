#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/.venv/bin/python3"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"${ROOT}/scripts/bringup/install_rnnoise.sh"
exec "$PYTHON" "${ROOT}/scripts/bringup/f3_rnnoise_ab.py" "$@"
