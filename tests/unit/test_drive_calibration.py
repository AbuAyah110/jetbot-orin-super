from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jetbot_agent.robot_loop.drive_calibration import (  # noqa: E402
    DURATION_HARD_MAX,
    SPEED_HARD_MAX,
    clamp_calibration,
    load_calibration,
)


def test_hard_caps_block_runaway_config():
    cal = clamp_calibration(speed=9.0, duration_s=99.0)
    assert cal.speed == SPEED_HARD_MAX == 0.7
    assert cal.duration_s == DURATION_HARD_MAX == 2.0


def test_yaml_is_the_measured_live_duty():
    cal = load_calibration()
    assert cal.speed == 0.65
    assert cal.duration_s == 1.2
    assert cal.measured_on == '2026-08-27'
