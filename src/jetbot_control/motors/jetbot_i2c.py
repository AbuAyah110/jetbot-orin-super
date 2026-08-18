from __future__ import annotations

"""Optional real backend wrapping the classic ``jetbot`` package.

Disabled by default. Enable only after ``docs/safety.md`` checklist and
wheels-off-ground testing. See ``docs/hardware_motors.md``.
"""

from .base import MotorDriver, MotorHealth


def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


class JetbotI2CMotorDriver(MotorDriver):
    def __init__(self, i2c_bus: int = 1, i2c_address: int = 112) -> None:
        try:
            from jetbot import Robot  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                'jetbot package not installed; cannot use jetbot_i2c backend'
            ) from exc

        self._robot = Robot(i2c_bus=i2c_bus, i2c_address=i2c_address)
        self._left = 0.0
        self._right = 0.0
        self._estop = False
        self._i2c_bus = i2c_bus
        self._i2c_address = i2c_address

    def set_velocity(self, left: float, right: float) -> None:
        if self._estop:
            self.stop()
            return
        self._left = _clamp(left)
        self._right = _clamp(right)
        self._robot.set_motors(self._left, self._right)

    def stop(self) -> None:
        self._left = 0.0
        self._right = 0.0
        self._robot.stop()

    def trigger_estop(self) -> None:
        self._estop = True
        self.stop()

    def clear_estop(self) -> None:
        self._estop = False

    def health(self) -> MotorHealth:
        return MotorHealth(
            ok=not self._estop,
            backend='jetbot_i2c',
            message='Adafruit/SparkFun via jetbot.Robot',
            left_velocity=self._left,
            right_velocity=self._right,
            estop_active=self._estop,
            details={'i2c_bus': self._i2c_bus, 'i2c_address': self._i2c_address},
        )

    def close(self) -> None:
        self.stop()
