#!/usr/bin/env bash
# Stage C: IMX219 CSI capture via existing gst_csi demo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p data/images
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
python3 scripts/demo_camera.py --backend gst_csi --frames 5 --out data/images/csi_bringup.jpg
ls -l data/images/csi_bringup.jpg
