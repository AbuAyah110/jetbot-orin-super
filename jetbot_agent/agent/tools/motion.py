"""The only motion surface the tool layer is allowed to see — Stage H / I2.

This module defines a deliberately narrow, high-level motion contract. It has
**no** imports from ``jetbot_control``, ``jetbot_base``, ``jetbot_agent.hardware``,
``smbus``, or any GPIO/PWM library, and it must stay that way: the structural
test in ``tests/unit/test_tool_safety.py`` walks the AST of every module in this
package and fails on such an import.

Where the real implementation lives
-----------------------------------
The concrete adapter that fulfils :class:`MotionInterface` by publishing
``/cmd_vel`` (or by driving ``jetbot_base.DiffDriveController``, which owns the
velocity clamps, the command watchdog, and the e-stop) lives **outside** this
package and is wired in Stage H / I5 + I8. The tool layer only ever receives an
object satisfying this protocol, never a ``MotorDriver``, controller, or I2C
handle — see ``ToolContext`` in :mod:`jetbot_agent.agent.tools.base`.

Consequences of that split:

* A tool cannot express "set left PWM to 200"; the vocabulary is
  ``drive``/``rotate``/``stop``/``status``/``limits``.
* Limits are readable (:class:`MotionLimits`) but not writable: there is no
  setter anywhere in the contract.
* ``duration_sec`` is a *request*. The authoritative stop is the deterministic
  ``cmd_vel`` watchdog below this boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Tuple, runtime_checkable


class MotionDenied(RuntimeError):
    """Motion refused below the tool boundary (e-stop, limits, or no backend)."""


@dataclass(frozen=True)
class MotionLimits:
    """Read-only view of the deterministic limits enforced below the boundary."""

    max_linear_velocity: float = 0.25
    max_angular_velocity: float = 1.0
    cmd_vel_timeout_sec: float = 0.5
    max_duration_sec: float = 5.0


@dataclass(frozen=True)
class MotionStatus:
    """What a tool is allowed to know about the base."""

    moving: bool = False
    estop_active: bool = False
    last_linear: float = 0.0
    last_angular: float = 0.0
    watchdog_armed: bool = False
    backend: str = 'unknown'
    detail: str = ''


@runtime_checkable
class MotionInterface(Protocol):
    """High-level motion vocabulary. No PWM, no GPIO, no I2C, no wheel speeds."""

    def drive(self, linear: float, angular: float, duration_sec: float) -> MotionStatus:
        """Request a bounded body twist. Clamped and watchdogged below here."""

    def rotate(self, angular: float, duration_sec: float) -> MotionStatus:
        """Request a bounded in-place rotation."""

    def stop(self) -> MotionStatus:
        """Request an immediate stop. Always permitted."""

    def status(self) -> MotionStatus:
        """Current base status."""

    def limits(self) -> MotionLimits:
        """Read-only limits view."""


@dataclass
class MockMotionInterface:
    """In-memory motion backend for tests. Touches no hardware at all.

    Clamps like the real path so tool tests see realistic refusals, and records
    every call so tests can assert what crossed the boundary.
    """

    motion_limits: MotionLimits = field(default_factory=MotionLimits)
    estop_active: bool = False
    calls: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    _linear: float = 0.0
    _angular: float = 0.0

    @staticmethod
    def _clamp(value: float, magnitude: float) -> float:
        return max(-magnitude, min(magnitude, float(value)))

    def _record(self, name: str, **fields: Any) -> None:
        self.calls.append((name, dict(fields)))

    @property
    def commands(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self.calls)

    def drive(self, linear: float, angular: float, duration_sec: float) -> MotionStatus:
        self._record('drive', linear=linear, angular=angular, duration_sec=duration_sec)
        if self.estop_active:
            raise MotionDenied('e-stop active; motion refused below the tool boundary')
        self._linear = self._clamp(linear, self.motion_limits.max_linear_velocity)
        self._angular = self._clamp(angular, self.motion_limits.max_angular_velocity)
        return self.status()

    def rotate(self, angular: float, duration_sec: float) -> MotionStatus:
        self._record('rotate', angular=angular, duration_sec=duration_sec)
        if self.estop_active:
            raise MotionDenied('e-stop active; motion refused below the tool boundary')
        self._linear = 0.0
        self._angular = self._clamp(angular, self.motion_limits.max_angular_velocity)
        return self.status()

    def stop(self) -> MotionStatus:
        self._record('stop')
        self._linear = 0.0
        self._angular = 0.0
        return self.status()

    def status(self) -> MotionStatus:
        return MotionStatus(
            moving=bool(self._linear or self._angular),
            estop_active=self.estop_active,
            last_linear=self._linear,
            last_angular=self._angular,
            watchdog_armed=bool(self._linear or self._angular),
            backend='mock',
            detail='MockMotionInterface',
        )

    def limits(self) -> MotionLimits:
        return self.motion_limits


__all__ = [
    'MockMotionInterface',
    'MotionDenied',
    'MotionInterface',
    'MotionLimits',
    'MotionStatus',
]
