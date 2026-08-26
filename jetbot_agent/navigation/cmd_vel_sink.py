"""Velocity sinks — where a bounded twist leaves the agent process (Stage H / I5).

A *sink* is transport only. Everything policy-shaped — velocity clamping,
duration deadlines, the agent-facing e-stop latch — belongs to
:class:`~jetbot_agent.navigation.motion_adapter.BoundedMotionAdapter` above it.
A sink receives an already bounded ``(linear, angular)`` body twist and can
always be told to halt.

Three sinks ship here:

* :class:`MockCmdVelSink` — in memory, and it models the ``cmd_vel`` watchdog so
  tests observe the same staleness behaviour as the real base. The default.
* :class:`ControllerCmdVelSink` — forwards to an injected ``jetbot_base``
  controller. The controller is duck typed, so this module imports nothing from
  ``jetbot_base`` at all; :func:`mock_controller_sink` builds one over the mock
  motor backend for integration tests.
* :class:`RosCmdVelSink` — publishes ``geometry_msgs/Twist`` on ``/cmd_vel``.
  The message type is imported lazily inside the constructor, so importing this
  module never requires ROS 2 on the path, and the sink never calls
  ``rclpy.init()`` or creates a node of its own.

No sink opens a bus, a device node, or a GPIO line. The real one publishes a
topic; ``jetbot_base`` (velocity limits + command watchdog + e-stop) stays the
authoritative last line of defence in front of the motor backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable

LOGGER = logging.getLogger('jetbot_agent.navigation.cmd_vel_sink')

DEFAULT_CMD_VEL_TOPIC = '/cmd_vel'


class MotionBackendUnavailable(RuntimeError):
    """The requested velocity backend cannot be reached from this process."""


@dataclass(frozen=True)
class TwistCommand:
    """One bounded body twist as it crossed the sink."""

    linear: float
    angular: float
    ts: float


@runtime_checkable
class CmdVelSink(Protocol):
    """Transport for bounded twists. Deliberately tiny."""

    name: str

    def publish(self, linear: float, angular: float, now: float) -> None:
        """Forward an already clamped twist."""

    def halt(self, reason: str = '') -> None:
        """Command zero velocity. Must never raise for policy reasons."""

    def tick(self, now: float) -> None:
        """Advance whatever deterministic watchdog lives below this sink."""

    def describe(self) -> str:
        """Short human-readable backend description."""


@dataclass
class MockCmdVelSink:
    """In-memory sink that models the deterministic ``cmd_vel`` watchdog.

    Recording every command and halt is what lets tests assert on the twist
    that actually crossed the boundary rather than on log text. Modelling
    staleness matters just as much: on the real robot a command that is not
    refreshed within ``cmd_vel_timeout_sec`` stops the wheels, and a mock that
    kept "moving" forever would hide exactly the bug this layer exists to
    prevent.
    """

    cmd_vel_timeout_sec: float = 0.5
    name: str = 'mock'
    commands: List[TwistCommand] = field(default_factory=list)
    halts: List[str] = field(default_factory=list)
    ticks: List[float] = field(default_factory=list)
    _linear: float = 0.0
    _angular: float = 0.0
    _last_ts: float = 0.0

    def publish(self, linear: float, angular: float, now: float) -> None:
        self._linear = float(linear)
        self._angular = float(angular)
        self._last_ts = float(now)
        self.commands.append(TwistCommand(self._linear, self._angular, self._last_ts))

    def halt(self, reason: str = '') -> None:
        self._linear = 0.0
        self._angular = 0.0
        self.halts.append(reason)
        self.commands.append(TwistCommand(0.0, 0.0, self._last_ts))

    def tick(self, now: float) -> None:
        self.ticks.append(float(now))

    def describe(self) -> str:
        return 'MockCmdVelSink(no hardware)'

    # ------------------------------------------------------------- assertions

    @property
    def twists(self) -> Tuple[Tuple[float, float], ...]:
        return tuple((cmd.linear, cmd.angular) for cmd in self.commands)

    @property
    def last_twist(self) -> Tuple[float, float]:
        return (self._linear, self._angular)

    def is_stale(self, now: float) -> bool:
        return (float(now) - self._last_ts) > self.cmd_vel_timeout_sec

    def moving_at(self, now: float) -> bool:
        """What the wheels would be doing: a stale command means stopped."""
        if self.is_stale(now):
            return False
        return bool(self._linear or self._angular)


class ControllerCmdVelSink:
    """Forward twists into a ``jetbot_base`` controller object.

    The controller is duck typed on ``command_twist`` / ``stop`` (plus optional
    ``tick`` and ``status_dict``) so this module never imports ``jetbot_base``
    or ``jetbot_control``. That keeps the sink usable in a plain virtualenv and
    keeps the ROS packages an integration detail of the caller.

    The controller owns the velocity clamps, the command watchdog, and its own
    e-stop. This sink never reaches past it: it does not touch the motor
    driver, and it deliberately does not proxy the controller's e-stop clear.
    """

    def __init__(self, controller: Any, *, name: str = 'controller',
                 logger: Optional[logging.Logger] = None) -> None:
        for method in ('command_twist', 'stop'):
            if not callable(getattr(controller, method, None)):
                raise MotionBackendUnavailable(
                    f'controller {type(controller).__name__} has no {method}(); '
                    'expected a jetbot_base differential-drive controller'
                )
        self._controller = controller
        self.name = name
        self._log = logger or LOGGER

    def publish(self, linear: float, angular: float, now: float) -> None:
        self._controller.command_twist(float(linear), float(angular), now=float(now))

    def halt(self, reason: str = '') -> None:
        try:
            self._controller.stop()
        except Exception as exc:  # noqa: BLE001 - a halt must never raise
            self._log.error('controller_halt_failed reason=%r error=%r', reason, exc)

    def tick(self, now: float) -> None:
        tick = getattr(self._controller, 'tick', None)
        if callable(tick):
            tick(now=float(now))

    def describe(self) -> str:
        return f'ControllerCmdVelSink({type(self._controller).__name__})'

    def status(self) -> dict:
        """Controller-reported status, for integration assertions."""
        status_dict = getattr(self._controller, 'status_dict', None)
        if callable(status_dict):
            return dict(status_dict())
        return {}


def mock_controller_sink(
    *,
    max_linear_velocity: float = 0.25,
    max_angular_velocity: float = 1.0,
    cmd_vel_timeout_sec: float = 0.5,
    wheel_separation_m: float = 0.12,
) -> ControllerCmdVelSink:
    """Build a controller sink over the **mock** motor backend.

    Imports are lazy so a plain virtualenv can still import this module. The
    mock driver is constructed directly rather than through the backend
    factory, so no configuration value can turn this into an I2C session.
    """
    try:
        from jetbot_base.diff_drive_controller import ControllerConfig, DiffDriveController
        from jetbot_control.motors.mock import MockMotorDriver
    except ImportError as exc:  # pragma: no cover - depends on sys.path wiring
        raise MotionBackendUnavailable(
            'jetbot_base / jetbot_control are not importable; add repo src and '
            'ros2_ws/src/jetbot_base to sys.path'
        ) from exc

    controller = DiffDriveController(
        MockMotorDriver(),
        ControllerConfig(
            max_linear_velocity=float(max_linear_velocity),
            max_angular_velocity=float(max_angular_velocity),
            wheel_separation_m=float(wheel_separation_m),
            cmd_vel_timeout_sec=float(cmd_vel_timeout_sec),
        ),
    )
    return ControllerCmdVelSink(controller, name='mock_controller')


def _import_twist() -> Any:
    try:
        from geometry_msgs.msg import Twist
    except ImportError as exc:
        raise MotionBackendUnavailable(
            'geometry_msgs is not importable; source a ROS 2 setup before using '
            'RosCmdVelSink (see docs/bringup/09b-agent-i5-navigation.md)'
        ) from exc
    return Twist


class RosCmdVelSink:
    """Publish bounded twists on ``/cmd_vel`` for ``jetbot_base`` to execute.

    Requires an already-initialised node from the caller. This class never
    calls ``rclpy.init()``, never spins, and never creates a node, so wiring it
    up is an explicit act by the process that already owns the ROS lifecycle.
    ``twist_type`` exists so the publish shape can be exercised without ROS
    installed.
    """

    def __init__(
        self,
        node: Any,
        *,
        topic: str = DEFAULT_CMD_VEL_TOPIC,
        queue_depth: int = 10,
        twist_type: Any = None,
        name: str = 'ros',
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if node is None:
            raise MotionBackendUnavailable(
                'RosCmdVelSink needs an initialised ROS 2 node; the agent does not '
                'own the ROS lifecycle'
            )
        if not callable(getattr(node, 'create_publisher', None)):
            raise MotionBackendUnavailable(
                f'{type(node).__name__} has no create_publisher(); expected a ROS 2 node'
            )
        self._twist_type = twist_type or _import_twist()
        self._publisher = node.create_publisher(self._twist_type, topic, queue_depth)
        self.topic = topic
        self.name = name
        self._log = logger or LOGGER

    def publish(self, linear: float, angular: float, now: float) -> None:
        message = self._twist_type()
        message.linear.x = float(linear)
        message.angular.z = float(angular)
        self._publisher.publish(message)

    def halt(self, reason: str = '') -> None:
        try:
            self.publish(0.0, 0.0, 0.0)
        except Exception as exc:  # noqa: BLE001 - a halt must never raise
            self._log.error('ros_halt_failed reason=%r error=%r', reason, exc)

    def tick(self, now: float) -> None:
        """No-op: the ``jetbot_base`` timer owns the deterministic watchdog."""

    def describe(self) -> str:
        return f'RosCmdVelSink(topic={self.topic})'


__all__ = [
    'CmdVelSink',
    'ControllerCmdVelSink',
    'DEFAULT_CMD_VEL_TOPIC',
    'MockCmdVelSink',
    'MotionBackendUnavailable',
    'RosCmdVelSink',
    'TwistCommand',
    'mock_controller_sink',
]
