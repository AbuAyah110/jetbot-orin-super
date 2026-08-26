"""Agent tools — Stage H I2–I7. LLM never PWM.

Public surface of the tool layer. The safety boundary is structural, not
advisory:

* Tools receive a :class:`~jetbot_agent.agent.tools.base.ToolContext`, whose
  ``motion`` field is validated by
  :func:`~jetbot_agent.agent.tools.base.assert_narrow_motion`. A ``MotorDriver``,
  a ``DiffDriveController``, or an I2C handle is rejected at construction.
* Motion is expressible only as the high-level
  :class:`~jetbot_agent.agent.tools.motion.MotionInterface` vocabulary
  (``drive`` / ``rotate`` / ``stop`` / ``status`` / ``limits``).
* No module in this package may import ``jetbot_control.motors``,
  ``jetbot_agent.hardware``, ``smbus``, ``Jetson.GPIO``, or friends; the
  structural test in ``tests/unit/test_tool_safety.py`` enforces it by AST scan.
* Registration is deny-by-default and ``ACTUATION`` needs an explicit operator
  acknowledgement.

The Stage-specific tool modules (``vision_tools``, ``search_tools``,
``navigation_tools``) stay stubs until their own tickets (I3–I5) and are not
imported here.
"""

from .base import (
    ActuationTool,
    Capability,
    DEFAULT_TOOL_TIMEOUT_SEC,
    FORBIDDEN_MOTION_ATTRS,
    MAX_TOOL_TIMEOUT_SEC,
    MIN_TOOL_TIMEOUT_SEC,
    RESERVED_PARAM_NAMES,
    RISK_CAPABILITY,
    RiskClass,
    Tool,
    ToolContext,
    ToolError,
    ToolExecutionError,
    ToolPermissionError,
    ToolRegistrationError,
    ToolSafetyViolation,
    ToolTimeoutError,
    ToolValidationError,
    assert_narrow_motion,
    effective_timeout,
    validate_arguments,
    validate_schema,
)
from .mocks import (
    FailingTool,
    MockActuationTool,
    MockNetworkTool,
    MockStopTool,
    MockTool,
    SlowActuationTool,
    SlowTool,
)
from .motion import (
    MockMotionInterface,
    MotionDenied,
    MotionInterface,
    MotionLimits,
    MotionStatus,
)
from .registry import ToolRegistry, ToolResult

__all__ = [
    'ActuationTool',
    'Capability',
    'DEFAULT_TOOL_TIMEOUT_SEC',
    'FORBIDDEN_MOTION_ATTRS',
    'FailingTool',
    'MAX_TOOL_TIMEOUT_SEC',
    'MIN_TOOL_TIMEOUT_SEC',
    'MockActuationTool',
    'MockMotionInterface',
    'MockNetworkTool',
    'MockStopTool',
    'MockTool',
    'MotionDenied',
    'MotionInterface',
    'MotionLimits',
    'MotionStatus',
    'RESERVED_PARAM_NAMES',
    'RISK_CAPABILITY',
    'RiskClass',
    'SlowActuationTool',
    'SlowTool',
    'Tool',
    'ToolContext',
    'ToolError',
    'ToolExecutionError',
    'ToolPermissionError',
    'ToolRegistrationError',
    'ToolRegistry',
    'ToolResult',
    'ToolSafetyViolation',
    'ToolTimeoutError',
    'ToolValidationError',
    'assert_narrow_motion',
    'effective_timeout',
    'validate_arguments',
    'validate_schema',
]
