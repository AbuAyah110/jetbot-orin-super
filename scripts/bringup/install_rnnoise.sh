#!/usr/bin/env bash
# F3: install RNNoise + the resampler it forces on a 16 kHz pipeline.
#
# pyrnnoise ships a prebuilt aarch64 librnnoise.so inside a pure-Python wheel, so
# no autotools build and no media.xiph.org model download is needed. It is
# installed with --no-deps on purpose: its declared dependencies (audiolab/PyAV,
# matplotlib) exist only for its file/CLI helpers, and F3 drives the bundled
# library through the ctypes shim in pyrnnoise/rnnoise.py, which needs numpy only.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PIP="${ROOT}/.venv/bin/pip"
PYTHON="${ROOT}/.venv/bin/python3"
CACHE="${ROOT}/data/models/f3"
mkdir -p "$CACHE"

if ! "$PYTHON" - <<'EOF'
import importlib.util, pathlib, sys
spec = importlib.util.find_spec("pyrnnoise")
if spec is None or not spec.submodule_search_locations:
    sys.exit(1)
lib = pathlib.Path(list(spec.submodule_search_locations)[0]) / "librnnoise.so"
sys.exit(0 if lib.exists() else 1)
EOF
then
  "$PIP" download pyrnnoise==0.4.3 --no-deps -d "$CACHE"
  "$PIP" install --no-deps "$CACHE"/pyrnnoise-0.4.3-*.whl
fi

"$PYTHON" -c "import soxr" 2>/dev/null || "$PIP" install "soxr==1.1.0"

"$PYTHON" - <<'EOF'
import ctypes, importlib.util, pathlib
spec = importlib.util.find_spec("pyrnnoise")
mod_path = pathlib.Path(list(spec.submodule_search_locations)[0]) / "rnnoise.py"
sub = importlib.util.spec_from_file_location("_rn", mod_path)
m = importlib.util.module_from_spec(sub)
sub.loader.exec_module(m)
import soxr
print(f"rnnoise ok: frame_size={m.FRAME_SIZE} @ {m.SAMPLE_RATE} Hz, soxr {soxr.__version__}")
EOF
