#!/usr/bin/env python3
"""Probe the front VL53L0X. Distance only — no motors, no PWM."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jetbot_agent.hardware.vl53l0x import (  # noqa: E402
    CLEAR_MM,
    DEFAULT_ADDRESS,
    DEFAULT_BUS,
    STOP_MM,
    VL53L0X,
    interpret_range_mm,
)


def main() -> int:
    samples = 5
    with VL53L0X() as tof:
        print(
            json.dumps(
                {
                    'sensor': 'VL53L0X',
                    'bus': tof.bus_id,
                    'address': hex(tof.address),
                    'revision': hex(tof.revision),
                    'stop_mm': STOP_MM,
                    'clear_mm': CLEAR_MM,
                }
            ),
            flush=True,
        )
        for index in range(samples):
            millimetres = tof.range_mm()
            policy = interpret_range_mm(millimetres)
            print(json.dumps({'n': index, 'range_mm': millimetres, **policy}), flush=True)
            time.sleep(0.15)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
