from __future__ import annotations

from .base import MotorDriver, MotorHealth


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


class MockMotorDriver(MotorDriver):
    """In-memory motor backend for unit tests and safe bring-up."""

    def __init__(self) -> None:
        self._left = 0.0
        self._right = 0.0
        self._estop = False
        self._command_count = 0

    def set_velocity(self, left: float, right: float) -> None:
        if self._estop:
            self._left = 0.0
            self._right = 0.0
            return
        self._left = _clamp(left)
        self._right = _clamp(right)
        self._command_count += 1

    def stop(self) -> None:
        self._left = 0.0
        self._right = 0.0

    def trigger_estop(self) -> None:
        self._estop = True
        self.stop()

    def clear_estop(self) -> None:
        self._estop = False

    def health(self) -> MotorHealth:
        return MotorHealth(
            ok=True,
            backend='mock',
            message='mock backend',
            left_velocity=self._left,
            right_velocity=self._right,
            estop_active=self._estop,
            details={'command_count': self._command_count},
        )
