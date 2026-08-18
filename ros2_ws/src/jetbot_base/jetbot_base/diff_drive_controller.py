from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

# Allow importing jetbot_control whether installed or run from repo.
import sys
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parents[4] / 'src'
if _REPO_SRC.exists() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from jetbot_control.kinematics import limit_twist, twist_to_wheel_speeds
from jetbot_control.motors.base import MotorDriver, MotorHealth


@dataclass
class ControllerConfig:
    max_linear_velocity: float = 0.25
    max_angular_velocity: float = 1.0
    max_wheel_velocity: float = 1.0
    wheel_separation_m: float = 0.12
    cmd_vel_timeout_sec: float = 0.5
    estop_latch: bool = True


class DiffDriveController:
    """ROS-independent controller core (limits, watchdog, e-stop)."""

    def __init__(self, driver: MotorDriver, config: Optional[ControllerConfig] = None) -> None:
        self._driver = driver
        self._config = config or ControllerConfig()
        self._last_cmd_time = 0.0
        self._have_cmd = False
        self._estop = False
        self._last_linear = 0.0
        self._last_angular = 0.0

    @property
    def estop_active(self) -> bool:
        return self._estop

    def trigger_estop(self) -> None:
        self._estop = True
        self._driver.stop()
        if hasattr(self._driver, 'trigger_estop'):
            self._driver.trigger_estop()

    def clear_estop(self) -> None:
        if hasattr(self._driver, 'clear_estop'):
            self._driver.clear_estop()
        self._estop = False

    def command_twist(self, linear: float, angular: float, now: Optional[float] = None) -> Tuple[float, float]:
        stamp = time.monotonic() if now is None else now
        self._last_cmd_time = stamp
        self._have_cmd = True
        self._last_linear = linear
        self._last_angular = angular

        if self._estop:
            self._driver.stop()
            return 0.0, 0.0

        linear, angular = limit_twist(
            linear,
            angular,
            self._config.max_linear_velocity,
            self._config.max_angular_velocity,
        )
        left, right = twist_to_wheel_speeds(
            linear,
            angular,
            self._config.wheel_separation_m,
            self._config.max_wheel_velocity,
        )
        self._driver.set_velocity(left, right)
        return left, right

    def tick(self, now: Optional[float] = None) -> None:
        stamp = time.monotonic() if now is None else now
        if self._estop:
            self._driver.stop()
            return
        if not self._have_cmd:
            return
        if (stamp - self._last_cmd_time) > self._config.cmd_vel_timeout_sec:
            self._driver.stop()
            self._have_cmd = False

    def stop(self) -> None:
        self._driver.stop()
        self._have_cmd = False

    def health(self) -> MotorHealth:
        h = self._driver.health()
        h.estop_active = self._estop or h.estop_active
        return h

    def status_dict(self) -> dict:
        h = self.health()
        return {
            'backend': h.backend,
            'ok': h.ok and not self._estop,
            'estop_active': self._estop,
            'left_velocity': h.left_velocity,
            'right_velocity': h.right_velocity,
            'last_linear': self._last_linear,
            'last_angular': self._last_angular,
            'watchdog_armed': self._have_cmd,
            'message': h.message,
        }
