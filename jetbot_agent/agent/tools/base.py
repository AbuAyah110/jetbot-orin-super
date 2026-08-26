"""Tool contract and safety boundary — Stage H / I2.

A tool is the *only* way an LLM can affect the world. The contract here is
built so that the dangerous options are not expressible rather than merely
discouraged (``docs/safety.md``, ``docs/architecture.md``):

1. **No low-level handle ever enters the tool layer.** :class:`ToolContext` is
   the sole capability carrier and validates what it is handed:
   :func:`assert_narrow_motion` rejects any object exposing wheel-level,
   PWM/GPIO/I2C, watchdog, or e-stop-clearing attributes. Passing a
   ``MotorDriver`` or a ``DiffDriveController`` raises
   :class:`ToolSafetyViolation` at construction time.
2. **No low-level parameter is nameable.** :data:`RESERVED_PARAM_NAMES` is
   refused at class-definition time, so a schema cannot advertise ``pwm``,
   ``duty_cycle``, ``i2c_bus``, ``gpio``, ``timeout``, or ``watchdog`` to a
   model.
3. **Schemas are closed.** ``additionalProperties`` must be ``False`` and
   arguments are validated before the tool body runs, so a model cannot smuggle
   extra keys through.
4. **The watchdog is not the agent's to change.** Timeouts are clamped by
   :func:`effective_timeout` to ``[MIN_TOOL_TIMEOUT_SEC, MAX_TOOL_TIMEOUT_SEC]``
   and the call path takes no caller-supplied timeout.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Mapping, Optional

from .motion import MotionInterface

LOGGER = logging.getLogger('jetbot_agent.agent.tools')

MIN_TOOL_TIMEOUT_SEC = 0.01
DEFAULT_TOOL_TIMEOUT_SEC = 2.0
# Hard ceiling. The agent cannot raise this, and no code path accepts a
# caller-supplied timeout, so a model cannot extend or disable the watchdog.
MAX_TOOL_TIMEOUT_SEC = 5.0

_NAME_RE = re.compile(r'^[a-z][a-z0-9_]{1,47}$')

#: Parameter names a tool schema may never expose to a model.
RESERVED_PARAM_NAMES = frozenset({
    'bus',
    'clear_estop',
    'deadline',
    'duty',
    'duty_cycle',
    'estop',
    'gpio',
    'i2c',
    'i2c_address',
    'i2c_bus',
    'left_pwm',
    'pin',
    'pwm',
    'raw',
    'register',
    'right_pwm',
    'set_velocity',
    'timeout',
    'timeout_sec',
    'watchdog',
    'watchdog_sec',
    'wheel_velocity',
})

#: Attributes whose presence proves an object is below the tool boundary.
FORBIDDEN_MOTION_ATTRS = frozenset({
    '_bus',
    '_driver',
    'bus',
    'clear_estop',
    'disable_watchdog',
    'duty_cycle',
    'gpio',
    'left_pwm',
    'motor',
    'motors',
    'pwm',
    'right_pwm',
    'set_duty_cycle',
    'set_pwm',
    'set_velocity',
    'set_watchdog',
    'write_byte',
    'write_byte_data',
    'write_i2c_block_data',
})

_MOTION_REQUIRED_METHODS = ('drive', 'rotate', 'stop', 'status', 'limits')

_SUPPORTED_TYPES: Dict[str, tuple] = {
    'string': (str,),
    'number': (int, float),
    'integer': (int,),
    'boolean': (bool,),
    'array': (list, tuple),
    'object': (dict,),
}


class ToolError(RuntimeError):
    """Base class for tool-layer faults."""


class ToolRegistrationError(ToolError):
    """A tool's declaration violates the contract."""


class ToolValidationError(ToolError):
    """Arguments failed schema validation."""


class ToolPermissionError(ToolError):
    """Tool is not registered, not allowed, or its capability is not granted."""


class ToolTimeoutError(ToolError):
    """The per-call watchdog fired."""


class ToolExecutionError(ToolError):
    """The tool body raised."""


class ToolSafetyViolation(ToolError):
    """A structural safety invariant was breached."""


class RiskClass(Enum):
    """How much damage a tool can do."""

    READ_ONLY = 'read_only'
    NETWORK = 'network'
    ACTUATION = 'actuation'


class Capability(Enum):
    """What an operator has opted the agent into."""

    READ = 'read'
    NETWORK = 'network'
    ACTUATE = 'actuate'


RISK_CAPABILITY: Dict[RiskClass, Capability] = {
    RiskClass.READ_ONLY: Capability.READ,
    RiskClass.NETWORK: Capability.NETWORK,
    RiskClass.ACTUATION: Capability.ACTUATE,
}


def effective_timeout(tool: 'Tool') -> float:
    """Clamp a tool's declared timeout into the allowed band.

    Mutating ``tool.timeout_sec`` cannot widen the window: the registry always
    routes through this function.
    """
    declared = getattr(tool, 'timeout_sec', DEFAULT_TOOL_TIMEOUT_SEC)
    try:
        value = float(declared)
    except (TypeError, ValueError):
        value = DEFAULT_TOOL_TIMEOUT_SEC
    if value != value:  # NaN
        value = DEFAULT_TOOL_TIMEOUT_SEC
    return max(MIN_TOOL_TIMEOUT_SEC, min(MAX_TOOL_TIMEOUT_SEC, value))


def assert_narrow_motion(obj: Any) -> None:
    """Refuse anything wider than :class:`MotionInterface`.

    This is the structural half of "LLMs never set PWM": the tool layer cannot
    be *given* a low-level handle in the first place.
    """
    if obj is None:
        return
    for name in sorted(FORBIDDEN_MOTION_ATTRS):
        if hasattr(obj, name):
            raise ToolSafetyViolation(
                f'{type(obj).__name__} exposes {name!r}; the tool layer may only '
                'receive a high-level MotionInterface (see docs/safety.md)'
            )
    for cls in type(obj).__mro__:
        if cls.__name__ in ('MotorDriver', 'DiffDriveController', 'MotorController'):
            raise ToolSafetyViolation(
                f'{type(obj).__name__} is a low-level motor object; hand the tool '
                'layer a MotionInterface adapter instead'
            )
        module = getattr(cls, '__module__', '') or ''
        if module.startswith(('jetbot_control.motors', 'jetbot_agent.hardware', 'smbus', 'Jetson')):
            raise ToolSafetyViolation(
                f'{type(obj).__name__} comes from {module!r}, which is below the tool boundary'
            )
    missing = [name for name in _MOTION_REQUIRED_METHODS if not callable(getattr(obj, name, None))]
    if missing:
        raise ToolSafetyViolation(
            f'{type(obj).__name__} does not satisfy MotionInterface (missing {missing})'
        )


def validate_schema(schema: Any, *, tool_name: str = '<tool>') -> Mapping[str, Any]:
    """Check a JSON-schema-ish parameter spec at declaration time."""
    if not isinstance(schema, Mapping):
        raise ToolRegistrationError(f'{tool_name}: parameters must be a mapping')
    if schema.get('type') != 'object':
        raise ToolRegistrationError(f'{tool_name}: parameters.type must be "object"')
    if schema.get('additionalProperties', None) is not False:
        raise ToolRegistrationError(
            f'{tool_name}: parameters.additionalProperties must be False (closed schema)'
        )
    properties = schema.get('properties', {})
    if not isinstance(properties, Mapping):
        raise ToolRegistrationError(f'{tool_name}: parameters.properties must be a mapping')
    required = schema.get('required', [])
    if not isinstance(required, (list, tuple)):
        raise ToolRegistrationError(f'{tool_name}: parameters.required must be a list')

    for key, spec in properties.items():
        if key in RESERVED_PARAM_NAMES:
            raise ToolRegistrationError(
                f'{tool_name}: parameter {key!r} is reserved; the tool surface may not '
                'expose low-level or watchdog controls'
            )
        if not isinstance(spec, Mapping):
            raise ToolRegistrationError(f'{tool_name}: spec for {key!r} must be a mapping')
        declared = spec.get('type')
        if declared not in _SUPPORTED_TYPES:
            raise ToolRegistrationError(
                f'{tool_name}: parameter {key!r} has unsupported type {declared!r}'
            )
    unknown_required = [name for name in required if name not in properties]
    if unknown_required:
        raise ToolRegistrationError(
            f'{tool_name}: required names not in properties: {unknown_required}'
        )
    return schema


def validate_arguments(
    schema: Mapping[str, Any],
    arguments: Optional[Mapping[str, Any]],
    *,
    tool_name: str = '<tool>',
) -> Dict[str, Any]:
    """Validate model-supplied arguments against a closed schema.

    Deny-by-default: unknown keys are rejected, not ignored.
    """
    args = dict(arguments or {})
    properties = schema.get('properties', {})
    required = list(schema.get('required', []))

    unknown = sorted(set(args) - set(properties))
    if unknown:
        raise ToolValidationError(f'{tool_name}: unknown parameter(s) {unknown}')
    missing = [name for name in required if name not in args]
    if missing:
        raise ToolValidationError(f'{tool_name}: missing required parameter(s) {missing}')

    validated: Dict[str, Any] = {}
    for key, value in args.items():
        spec = properties[key]
        validated[key] = _validate_value(key, value, spec, tool_name)
    for key, spec in properties.items():
        if key not in validated and 'default' in spec:
            validated[key] = spec['default']
    return validated


def _validate_value(key: str, value: Any, spec: Mapping[str, Any], tool_name: str) -> Any:
    declared = spec['type']
    allowed = _SUPPORTED_TYPES[declared]
    if declared in ('number', 'integer') and isinstance(value, bool):
        raise ToolValidationError(f'{tool_name}: {key!r} must be a {declared}, got bool')
    if not isinstance(value, allowed):
        raise ToolValidationError(
            f'{tool_name}: {key!r} must be a {declared}, got {type(value).__name__}'
        )
    if declared == 'number':
        value = float(value)

    if 'enum' in spec and value not in spec['enum']:
        raise ToolValidationError(f'{tool_name}: {key!r} must be one of {spec["enum"]}')
    if declared in ('number', 'integer'):
        minimum = spec.get('minimum')
        maximum = spec.get('maximum')
        if minimum is not None and value < minimum:
            raise ToolValidationError(f'{tool_name}: {key!r} must be >= {minimum}')
        if maximum is not None and value > maximum:
            raise ToolValidationError(f'{tool_name}: {key!r} must be <= {maximum}')
    if declared == 'string':
        max_length = spec.get('maxLength')
        min_length = spec.get('minLength')
        if min_length is not None and len(value) < min_length:
            raise ToolValidationError(f'{tool_name}: {key!r} is shorter than {min_length}')
        if max_length is not None and len(value) > max_length:
            raise ToolValidationError(f'{tool_name}: {key!r} is longer than {max_length}')
        pattern = spec.get('pattern')
        if pattern is not None and not re.match(pattern, value):
            raise ToolValidationError(f'{tool_name}: {key!r} does not match {pattern!r}')
    if declared == 'array':
        max_items = spec.get('maxItems')
        if max_items is not None and len(value) > max_items:
            raise ToolValidationError(f'{tool_name}: {key!r} has more than {max_items} items')
    return value


@dataclass(frozen=True)
class ToolContext:
    """The capability carrier handed to tools.

    Every field is a *narrow* interface. ``motion`` is validated by
    :func:`assert_narrow_motion`, so wiring code physically cannot pass a motor
    driver, a controller, or an I2C handle into the tool layer.
    """

    motion: Optional[MotionInterface] = None
    perception: Optional[Any] = None
    memory: Optional[Any] = None
    logger: logging.Logger = field(default=LOGGER, repr=False)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert_narrow_motion(self.motion)

    def require_motion(self) -> MotionInterface:
        if self.motion is None:
            raise ToolSafetyViolation('no motion interface wired; actuation refused')
        return self.motion


class Tool(ABC):
    """Base class for every tool. Subclass metadata is checked eagerly.

    Concrete subclasses declare ``name``, ``description``, ``parameters``, and
    ``risk``; ``timeout_sec`` is advisory and always clamped by
    :func:`effective_timeout`.
    """

    name: ClassVar[str] = ''
    description: ClassVar[str] = ''
    parameters: ClassVar[Mapping[str, Any]] = {}
    risk: ClassVar[RiskClass] = RiskClass.READ_ONLY
    timeout_sec: ClassVar[float] = DEFAULT_TOOL_TIMEOUT_SEC

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get('abstract', False) or not cls.__dict__.get('name', getattr(cls, 'name', '')):
            # Intermediate base (e.g. ActuationTool); validated on the concrete leaf.
            return
        if not _NAME_RE.match(cls.name):
            raise ToolRegistrationError(
                f'{cls.__name__}: tool name {cls.name!r} must be lower_snake_case'
            )
        if not cls.description or not cls.description.strip():
            raise ToolRegistrationError(f'{cls.name}: description is required')
        if not isinstance(cls.risk, RiskClass):
            raise ToolRegistrationError(f'{cls.name}: risk must be a RiskClass')
        validate_schema(cls.parameters, tool_name=cls.name)

    @property
    def capability(self) -> Capability:
        return RISK_CAPABILITY[self.risk]

    def describe(self) -> Dict[str, Any]:
        """Function-calling style description for a model prompt."""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': dict(self.parameters),
            'risk': self.risk.value,
            'timeout_sec': effective_timeout(self),
        }

    def validate(self, arguments: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        return validate_arguments(self.parameters, arguments, tool_name=self.name or '<tool>')

    def execute(self, context: ToolContext, arguments: Optional[Mapping[str, Any]] = None) -> Any:
        """Validate then run. Always validates, even on a direct call."""
        if not isinstance(context, ToolContext):
            raise ToolSafetyViolation('execute() requires a ToolContext')
        validated = self.validate(arguments)
        return self._run(context, **validated)

    @abstractmethod
    def _run(self, context: ToolContext, **kwargs: Any) -> Any:
        """Tool body. Only touches the narrow interfaces on ``context``."""


class ActuationTool(Tool):
    """Base for tools that move the robot.

    Motion goes out through :class:`~jetbot_agent.agent.tools.motion.MotionInterface`
    only, i.e. high-level tool → ``/cmd_vel`` → ``jetbot_base`` limits +
    watchdog → ``MotorDriver`` → wheels.
    """

    abstract = True
    risk: ClassVar[RiskClass] = RiskClass.ACTUATION

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.__dict__.get('name', getattr(cls, 'name', '')) and cls.risk is not RiskClass.ACTUATION:
            raise ToolRegistrationError(f'{cls.name}: ActuationTool must keep risk=ACTUATION')

    def motion(self, context: ToolContext) -> MotionInterface:
        return context.require_motion()


__all__ = [
    'ActuationTool',
    'Capability',
    'DEFAULT_TOOL_TIMEOUT_SEC',
    'FORBIDDEN_MOTION_ATTRS',
    'MAX_TOOL_TIMEOUT_SEC',
    'MIN_TOOL_TIMEOUT_SEC',
    'RESERVED_PARAM_NAMES',
    'RISK_CAPABILITY',
    'RiskClass',
    'Tool',
    'ToolContext',
    'ToolError',
    'ToolExecutionError',
    'ToolPermissionError',
    'ToolRegistrationError',
    'ToolSafetyViolation',
    'ToolTimeoutError',
    'ToolValidationError',
    'assert_narrow_motion',
    'effective_timeout',
    'validate_arguments',
    'validate_schema',
]
