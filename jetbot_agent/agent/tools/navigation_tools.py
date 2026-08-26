"""Navigation tools — Stage H / I5. High-level, bounded, cancellable motion.

Four tools, and between them they are the entire motion vocabulary a model gets:
``nav_drive``, ``nav_rotate``, ``nav_stop``, ``nav_status``. There is no verb
here for a wheel, a duty cycle, a bus, a register, or a watchdog, and by
:data:`~jetbot_agent.agent.tools.base.RESERVED_PARAM_NAMES` there cannot be — a
schema naming one of those keys is refused at class-definition time.

Two properties every actuation tool in this module holds to:

* **Bounded in magnitude.** Speeds and turn rates are capped in the closed
  schema, and clamped again by the adapter against ``config/robot.yaml``.
* **Bounded in duration.** A tool derives a duration from the requested
  distance or angle and caps it at :data:`MAX_NAV_DURATION_SEC`. Indefinite
  motion is not expressible: there is no "keep going" argument, and a call that
  is not refreshed inside the command watchdog window stops the base.

Distances and angles are open-loop dead reckoning, because this robot has no
odometry yet (``/odom`` is a stub). Tools therefore report both what was asked
for and what the bounded command can actually achieve, rather than implying a
closed-loop guarantee they cannot keep.

Everything below the ``drive`` / ``rotate`` / ``stop`` / ``status`` / ``limits``
calls in this file — clamping, the command watchdog, the latched e-stop, the
velocity backend — belongs to the adapter outside this package; see
``jetbot_agent/navigation/motion_adapter.py`` and
``docs/bringup/09b-agent-i5-navigation.md``.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar, Dict, Mapping, Tuple

from .base import ActuationTool, RiskClass, Tool, ToolContext
from .motion import MotionStatus
from .registry import ToolRegistry

#: Tool-level bounds. Tighter than the adapter's, on purpose: the tool surface
#: should be the most conservative layer, not the most permissive.
MAX_NAV_DISTANCE_M = 0.5
MIN_NAV_SPEED = 0.05
MAX_NAV_SPEED = 0.25
DEFAULT_NAV_SPEED = 0.1
MAX_NAV_ANGLE_DEG = 180.0
MIN_NAV_TURN_RATE = 0.1
MAX_NAV_TURN_RATE = 1.0
DEFAULT_NAV_TURN_RATE = 0.5

#: Hard ceiling on how long one tool call may leave the base moving.
MAX_NAV_DURATION_SEC = 3.0


def _status_payload(status: MotionStatus) -> Dict[str, Any]:
    """The only base state a model is told about."""
    return {
        'moving': bool(status.moving),
        'linear': float(status.last_linear),
        'angular': float(status.last_angular),
        'estop_active': bool(status.estop_active),
        'watchdog_armed': bool(status.watchdog_armed),
        'backend': status.backend,
    }


def _bounded_duration(magnitude: float, rate: float) -> Tuple[float, bool]:
    """Time to cover ``magnitude`` at ``rate``, capped. Returns (duration, capped)."""
    rate = abs(rate)
    if rate <= 0.0:
        # Unreachable through the schema; fail towards no motion, never towards more.
        return 0.0, True
    ideal = abs(magnitude) / rate
    duration = min(ideal, MAX_NAV_DURATION_SEC)
    return duration, duration < ideal


class NavDriveTool(ActuationTool):
    """Drive a short, bounded distance.

    ``distance_m`` is signed (negative reverses) and its magnitude only sets how
    long the bounded command runs — there is no odometry to close the loop on
    yet, so the result reports the achievable distance separately.
    """

    name: ClassVar[str] = 'nav_drive'
    description: ClassVar[str] = (
        'Drive a short bounded distance in metres at a capped speed, optionally '
        'while turning. Negative distance reverses. Motion is limited in both '
        'speed and duration and stops itself; velocity clamps, the command '
        'watchdog, and the e-stop all live below this call.'
    )
    timeout_sec: ClassVar[float] = 1.0
    parameters: ClassVar[Mapping[str, Any]] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'distance_m': {
                'type': 'number',
                'minimum': -MAX_NAV_DISTANCE_M,
                'maximum': MAX_NAV_DISTANCE_M,
                'description': 'Signed distance to travel in metres; negative reverses.',
            },
            'speed': {
                'type': 'number',
                'minimum': MIN_NAV_SPEED,
                'maximum': MAX_NAV_SPEED,
                'default': DEFAULT_NAV_SPEED,
                'description': 'Forward speed magnitude in m/s.',
            },
            'turn_rate': {
                'type': 'number',
                'minimum': -MAX_NAV_TURN_RATE,
                'maximum': MAX_NAV_TURN_RATE,
                'default': 0.0,
                'description': 'Optional yaw rate in rad/s while driving.',
            },
        },
        'required': ['distance_m'],
    }

    def _run(self, context: ToolContext, **kwargs: Any) -> Dict[str, Any]:
        motion = self.motion(context)
        distance = float(kwargs['distance_m'])
        speed = abs(float(kwargs.get('speed', DEFAULT_NAV_SPEED)))
        turn_rate = float(kwargs.get('turn_rate', 0.0))

        heading = -1.0 if distance < 0.0 else 1.0
        duration, capped = _bounded_duration(distance, speed)
        status = motion.drive(
            linear=heading * speed,
            angular=turn_rate,
            duration_sec=duration,
        )
        payload = _status_payload(status)
        payload.update({
            'requested_distance_m': distance,
            'reachable_distance_m': heading * abs(status.last_linear) * duration,
            'duration_sec': duration,
            'duration_capped': capped,
            'open_loop': True,
        })
        return payload


class NavRotateTool(ActuationTool):
    """Rotate in place by a bounded angle."""

    name: ClassVar[str] = 'nav_rotate'
    description: ClassVar[str] = (
        'Rotate in place by a signed angle in degrees (positive is left/CCW) at a '
        'capped turn rate. Bounded in rate and duration; stops itself.'
    )
    timeout_sec: ClassVar[float] = 1.0
    parameters: ClassVar[Mapping[str, Any]] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'angle_deg': {
                'type': 'number',
                'minimum': -MAX_NAV_ANGLE_DEG,
                'maximum': MAX_NAV_ANGLE_DEG,
                'description': 'Signed rotation in degrees; positive turns left.',
            },
            'turn_rate': {
                'type': 'number',
                'minimum': MIN_NAV_TURN_RATE,
                'maximum': MAX_NAV_TURN_RATE,
                'default': DEFAULT_NAV_TURN_RATE,
                'description': 'Yaw rate magnitude in rad/s.',
            },
        },
        'required': ['angle_deg'],
    }

    def _run(self, context: ToolContext, **kwargs: Any) -> Dict[str, Any]:
        motion = self.motion(context)
        angle_deg = float(kwargs['angle_deg'])
        turn_rate = abs(float(kwargs.get('turn_rate', DEFAULT_NAV_TURN_RATE)))

        radians = math.radians(angle_deg)
        heading = -1.0 if angle_deg < 0.0 else 1.0
        duration, capped = _bounded_duration(radians, turn_rate)
        status = motion.rotate(angular=heading * turn_rate, duration_sec=duration)
        payload = _status_payload(status)
        payload.update({
            'requested_angle_deg': angle_deg,
            'reachable_angle_deg': math.degrees(heading * abs(status.last_angular) * duration),
            'duration_sec': duration,
            'duration_capped': capped,
            'open_loop': True,
        })
        return payload


class NavStopTool(ActuationTool):
    """Stop the base. Actuation, but only ever in the fail-safe direction.

    Stop is never refused below the boundary: it works while the e-stop is
    latched, while a command is mid-flight, and while a sink is failing. It is
    also reachable *without* the tool layer at all — the registry issues it on a
    call timeout and on close, and an operator or the harness e-stop trips the
    latch directly — so a model losing its actuation grant can never leave the
    robot moving.
    """

    name: ClassVar[str] = 'nav_stop'
    description: ClassVar[str] = (
        'Stop the base immediately and cancel any in-flight motion. Always safe '
        'to call, takes no arguments.'
    )
    timeout_sec: ClassVar[float] = 0.5
    parameters: ClassVar[Mapping[str, Any]] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {},
        'required': [],
    }

    def _run(self, context: ToolContext, **kwargs: Any) -> Dict[str, Any]:
        return _status_payload(self.motion(context).stop())


class NavStatusTool(Tool):
    """Report base status and the limits in force.

    Read-only: it observes the base and commands nothing, so it is classified
    ``READ_ONLY`` and does not need the actuation grant. That matters — an agent
    without permission to move should still be able to see whether the robot is
    moving or e-stopped.
    """

    name: ClassVar[str] = 'nav_status'
    description: ClassVar[str] = (
        'Report whether the base is moving, whether the e-stop is latched, and '
        'the velocity limits in force. Read-only.'
    )
    risk: ClassVar[RiskClass] = RiskClass.READ_ONLY
    timeout_sec: ClassVar[float] = 0.5
    parameters: ClassVar[Mapping[str, Any]] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {},
        'required': [],
    }

    def _run(self, context: ToolContext, **kwargs: Any) -> Dict[str, Any]:
        motion = context.require_motion()
        limits = motion.limits()
        payload = _status_payload(motion.status())
        payload['limits'] = {
            'max_linear_velocity': limits.max_linear_velocity,
            'max_angular_velocity': limits.max_angular_velocity,
            'max_duration_sec': limits.max_duration_sec,
        }
        payload['max_distance_m'] = MAX_NAV_DISTANCE_M
        payload['max_angle_deg'] = MAX_NAV_ANGLE_DEG
        return payload


def navigation_tools() -> Tuple[Tool, ...]:
    """Fresh instances of the I5 tool set."""
    return (NavDriveTool(), NavRotateTool(), NavStopTool(), NavStatusTool())


def register_navigation_tools(
    registry: ToolRegistry,
    *,
    allow: bool = False,
) -> Tuple[str, ...]:
    """Catalogue the navigation tools on ``registry`` and return their names.

    Deny-by-default: with ``allow=False`` (the production default) nothing is
    invocable until wiring code names each tool in ``registry.allow()`` and the
    operator grants ``Capability.ACTUATE`` with ``operator_ack=True``.
    """
    names = []
    for tool in navigation_tools():
        registry.register(tool, allow=allow)
        names.append(tool.name)
    return tuple(names)


__all__ = [
    'DEFAULT_NAV_SPEED',
    'DEFAULT_NAV_TURN_RATE',
    'MAX_NAV_ANGLE_DEG',
    'MAX_NAV_DISTANCE_M',
    'MAX_NAV_DURATION_SEC',
    'MAX_NAV_SPEED',
    'MAX_NAV_TURN_RATE',
    'MIN_NAV_SPEED',
    'MIN_NAV_TURN_RATE',
    'NavDriveTool',
    'NavRotateTool',
    'NavStatusTool',
    'NavStopTool',
    'navigation_tools',
    'register_navigation_tools',
]
