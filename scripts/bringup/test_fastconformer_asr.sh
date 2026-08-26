#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/.venv/bin/python3"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"${ROOT}/scripts/bringup/fetch_fastconformer_models.sh"
exec "$PYTHON" "${ROOT}/scripts/bringup/test_fastconformer_asr.py" "$@"
