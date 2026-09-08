"""Measured drive duty for the deterministic intent path.

The numbers come from a duty ramp on this chassis, not from the VLM: 0.15 only
hummed (stiction / pack sag held the wheels still). The last confirmed travel
pulse was **0.65** ("that worked"). Cosmos is not allowed to supply them because
it has asked for vx=0.03 and reversed a backward request. Every motion intent
(forward, back, left, right) uses this same ``speed`` as |wheel duty|.

``config/robot.yaml`` is the single source of truth. Everything read from it is
clamped by the hard caps below, so a typo (or a future "just make it faster"
edit) cannot produce a runaway pulse. This module never opens I2C.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

CONFIG_PATH = Path(__file__).resolve().parents[2] / 'config' / 'robot.yaml'
CONFIG_SECTION = 'drive_calibration'

# A stalled motor overheats and drains the pack; a long pulse drives into
# furniture. These caps bound both regardless of what the config says.
SPEED_HARD_MAX = 0.7
DURATION_HARD_MAX = 2.0
# Below this the wheels only hum on this chassis.
SPEED_HARD_MIN = 0.2

DEFAULT_SPEED = 0.65
DEFAULT_DURATION_S = 1.2


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class DriveCalibration:
    """Bounded duty/duration pair for one spoken motion nudge."""

    speed: float = DEFAULT_SPEED
    duration_s: float = DEFAULT_DURATION_S
    source: str = 'defaults'
    measured_on: str = ''

    def summary(self) -> str:
        return (
            'drive_calibration speed={0:.2f} duration_s={1:.2f} '
            'caps=({2:.2f},{3:.2f}s) source={4} measured_on={5}'
        ).format(
            self.speed,
            self.duration_s,
            SPEED_HARD_MAX,
            DURATION_HARD_MAX,
            self.source,
            self.measured_on or 'unrecorded',
        )


def _as_float(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number != number or number in (float('inf'), float('-inf')):
        return fallback
    return number


def clamp_calibration(
    *,
    speed: Any = DEFAULT_SPEED,
    duration_s: Any = DEFAULT_DURATION_S,
    source: str = 'defaults',
    measured_on: Any = '',
) -> DriveCalibration:
    """Build a calibration with every field forced inside the hard caps."""
    return DriveCalibration(
        speed=_clamp(
            abs(_as_float(speed, DEFAULT_SPEED)), SPEED_HARD_MIN, SPEED_HARD_MAX
        ),
        duration_s=_clamp(
            abs(_as_float(duration_s, DEFAULT_DURATION_S)), 0.0, DURATION_HARD_MAX
        ),
        source=source,
        measured_on=str(measured_on or ''),
    )


def load_calibration(path: Optional[Path] = None) -> DriveCalibration:
    """Read the measured duty from config, falling back to the defaults."""
    config_path = Path(path) if path is not None else CONFIG_PATH
    try:
        import yaml

        raw = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    except Exception:
        return clamp_calibration()
    section = raw.get(CONFIG_SECTION) if isinstance(raw, dict) else None
    if not isinstance(section, dict):
        return clamp_calibration()
    return clamp_calibration(
        speed=section.get('speed', DEFAULT_SPEED),
        duration_s=section.get('duration_s', DEFAULT_DURATION_S),
        source=str(config_path),
        measured_on=section.get('measured_on', ''),
    )
