#!/usr/bin/env bash
# G5: check this board against NVIDIA TensorRT Edge-LLM's Official Support Matrix.
#
# Read-only. Clones the upstream repo at a pinned tag into an ignored data dir and
# reads its docs and CMake; installs nothing, builds nothing, and never touches the
# GPU. Safe to run sandboxed, and safe to run while another agent is compiling.
#
# Unlike g1_tensorrt_smoke.sh this needs no TensorRT bindings and no PYTHONPATH
# bridge into /usr/lib/python3.10/dist-packages — it is pure stdlib plus git — so
# there is nothing here that can disturb the shared .venv's pinned numpy.
#
# Corrects the Stage G1 conclusion that "there is no NVIDIA product called
# 'TensorRT Edge-LLM'". Evidence record: docs/bringup/07b-tensorrt-edge-llm.md
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${ROOT}/.venv/bin/python3"
[[ -x "$PYTHON" ]] || PYTHON=/usr/bin/python3

exec "$PYTHON" "${ROOT}/scripts/bringup/g5_edgellm_probe.py" "$@"
