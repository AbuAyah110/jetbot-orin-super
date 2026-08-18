from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class MotorHealth:
    ok: bool
    backend: str
    message: str = ''
    left_velocity: float = 0.0
    right_velocity: float = 0.0
    estop_active: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class MotorDriver(ABC):
    """Hardware-independent differential-drive interface.

    Velocities are normalized wheel commands in [-1, 1].
    Positive values drive the robot forward for that wheel.
    """

    @abstractmethod
    def set_velocity(self, left: float, right: float) -> None:
        """Set left/right wheel velocities in [-1, 1]."""

    @abstractmethod
    def stop(self) -> None:
        """Immediately stop both wheels."""

    @abstractmethod
    def health(self) -> MotorHealth:
        """Return backend health and last commanded velocities."""

    def close(self) -> None:
        """Release resources; default stops motors."""
        self.stop()
