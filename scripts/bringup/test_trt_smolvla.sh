#!/usr/bin/env bash
# G3 dummy I/O gate — currently an honest miss, not a silent skip.
#
# Issue #18 asked for a dummy SmolVLA forward with no PWM. That forward is
# blocked on Jetson PyTorch (#30) and there is no published ONNX/engine.
# This script runs the import-safe unit tests and the export-script refusal,
# then exits 1 so the ticket is not marked pass.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON="${ROOT}/.venv/bin/python"
elif [[ -x /home/impulse110/Documents/jetbot-orin-super/.venv/bin/python ]]; then
  PYTHON=/home/impulse110/Documents/jetbot-orin-super/.venv/bin/python
else
  echo "no .venv python" >&2
  exit 1
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m pytest tests/unit/test_trt_vla_motor.py -q
echo
echo "export script (must refuse until torch+lerobot exist):"
set +e
"$PYTHON" "${ROOT}/scripts/bringup/export_smolvla_onnx.py"
rc=$?
set -e
echo "export exit code: ${rc} (expected nonzero)"
echo
echo "G3 dummy forward did not run: no engine, no torch. See docs/bringup/07-smolvla-trt.md"
echo "Safety: this script does not open I2C, PCA9685, or /dev/snd."
exit 1
