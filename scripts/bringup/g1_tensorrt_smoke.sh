#!/usr/bin/env bash
# G1: TensorRT runtime inventory + tiny-engine smoke test.
#
# The TensorRT Python bindings ship as an apt package (python3-libnvinfer) into
# /usr/lib/python3.10/dist-packages, which the repo .venv cannot see because it
# was created without --system-site-packages. Rather than rebuild the shared
# .venv (Stage F agents depend on it), put the system dist-packages on
# PYTHONPATH. The venv interpreter is the same CPython 3.10.12 as /usr/bin/python3,
# so the cp310 bindings load unchanged.
#
# onnx is installed into a repo-local --target dir with --no-deps so pip cannot
# resolve numpy away from the version the Stage F voice gates are pinned against.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${ROOT}/.venv/bin/python3"
PYLIBS="${ROOT}/data/bringup/g1/pylibs"
SYSTEM_DIST="/usr/lib/python3.10/dist-packages"

mkdir -p "$PYLIBS"
if ! PYTHONPATH="$PYLIBS" "$PYTHON" -c "import onnx" >/dev/null 2>&1; then
  "$PYTHON" -m pip install --no-deps --disable-pip-version-check \
    --target "$PYLIBS" onnx protobuf ml_dtypes
fi

export PYTHONPATH="${ROOT}:${SYSTEM_DIST}:${PYLIBS}"
export PATH="/usr/src/tensorrt/bin:/usr/local/cuda/bin:${PATH}"
exec "$PYTHON" "${ROOT}/scripts/bringup/g1_tensorrt_smoke.py" "$@"
