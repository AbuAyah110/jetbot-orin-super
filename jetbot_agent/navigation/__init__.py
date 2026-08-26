"""Navigation below the tool boundary — Stage H / I5.

This package holds the concrete motion path the agent is allowed to reach:
the ``MotionInterface`` adapter, the velocity sinks it can target, and the seam
where a VLA policy's motion intents would enter.

It lives outside ``jetbot_agent/agent/tools/`` on purpose. The tool package may
not import a velocity backend, and
:func:`jetbot_agent.agent.tools.base.assert_narrow_motion` rejects anything that
looks like a driver or controller, so the only thing that can cross into the
tool layer is a :class:`~.motion_adapter.BoundedMotionAdapter` — high-level
verbs in, clamped and watchdogged velocity out.

Importing this package pulls in no ROS 2, no ``smbus``, no GPIO library, and no
model weights; the ROS sink imports its message type lazily.
"""

from .cmd_vel_sink import (
    CmdVelSink,
    ControllerCmdVelSink,
    DEFAULT_CMD_VEL_TOPIC,
    MockCmdVelSink,
    MotionBackendUnavailable,
    RosCmdVelSink,
    TwistCommand,
    mock_controller_sink,
)
from .motion_adapter import (
    BoundedMotionAdapter,
    ClampEvent,
    HARD_MAX_ANGULAR_VELOCITY,
    HARD_MAX_CMD_VEL_TIMEOUT_SEC,
    HARD_MAX_DURATION_SEC,
    HARD_MAX_LINEAR_VELOCITY,
    MIN_DURATION_SEC,
    MotionEstop,
    bounded_limits,
    limits_from_robot_config,
    load_robot_limits,
)
from .vla_seam import (
    INTENT_KINDS,
    MotionIntent,
    UnavailableVlaPolicy,
    VlaPolicy,
    apply_intent,
)

__all__ = [
    'BoundedMotionAdapter',
    'CmdVelSink',
    'ClampEvent',
    'ControllerCmdVelSink',
    'DEFAULT_CMD_VEL_TOPIC',
    'HARD_MAX_ANGULAR_VELOCITY',
    'HARD_MAX_CMD_VEL_TIMEOUT_SEC',
    'HARD_MAX_DURATION_SEC',
    'HARD_MAX_LINEAR_VELOCITY',
    'INTENT_KINDS',
    'MIN_DURATION_SEC',
    'MockCmdVelSink',
    'MotionBackendUnavailable',
    'MotionEstop',
    'MotionIntent',
    'RosCmdVelSink',
    'TwistCommand',
    'UnavailableVlaPolicy',
    'VlaPolicy',
    'apply_intent',
    'bounded_limits',
    'limits_from_robot_config',
    'load_robot_limits',
    'mock_controller_sink',
]
