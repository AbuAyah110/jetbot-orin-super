from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ros2_ws' / 'src' / 'jetbot_base'))

from jetbot_agent.agent.tools import (
    ActuationTool,
    Capability,
    FailingTool,
    MAX_TOOL_TIMEOUT_SEC,
    MIN_TOOL_TIMEOUT_SEC,
    MockActuationTool,
    MockMotionInterface,
    MockNetworkTool,
    MockStopTool,
    MockTool,
    MotionDenied,
    MotionInterface,
    RESERVED_PARAM_NAMES,
    RiskClass,
    SlowActuationTool,
    SlowTool,
    Tool,
    ToolContext,
    ToolExecutionError,
    ToolPermissionError,
    ToolRegistrationError,
    ToolRegistry,
    ToolSafetyViolation,
    ToolTimeoutError,
    ToolValidationError,
    effective_timeout,
)

TOOLS_DIR = ROOT / 'jetbot_agent' / 'agent' / 'tools'

FORBIDDEN_MODULE_PREFIXES = (
    'jetbot_control',
    'jetbot_base',
    'jetbot_agent.hardware',
    'smbus',
    'smbus2',
    'Adafruit_PCA9685',
    'adafruit_pca9685',
    'busio',
    'board',
    'Jetson',
    'RPi',
    'periphery',
)

FORBIDDEN_IDENTIFIERS = frozenset({
    'PCA9685',
    'SMBus',
    'DiffDriveController',
    'MockMotorDriver',
    'MotorController',
    'MotorDriver',
    'GPIO',
    'set_velocity',
    'set_pwm',
    'set_duty_cycle',
    'write_byte',
    'write_byte_data',
    'write_i2c_block_data',
    'twist_to_wheel_speeds',
})

FORBIDDEN_DEVICE_PATHS = ('/dev/i2c', '/dev/mem', '/sys/class/pwm', '/dev/gpiochip')


def _tool_modules():
    return sorted(p for p in TOOLS_DIR.glob('*.py'))


def _registry(*tools, capabilities=(Capability.READ,), motion=None, ack=False, allow=True):
    context = ToolContext(motion=motion)
    registry = ToolRegistry(context, capabilities=capabilities, operator_ack_actuation=ack)
    for tool in tools:
        registry.register(tool, allow=allow)
    return registry


# ------------------------------------------------- structural no-PWM boundary


def test_tool_package_has_modules_to_scan():
    names = {p.name for p in _tool_modules()}
    assert {'__init__.py', 'base.py', 'motion.py', 'registry.py', 'mocks.py'} <= names


@pytest.mark.parametrize('path', _tool_modules(), ids=lambda p: p.name)
def test_no_low_level_imports_in_tool_layer(path):
    """The tool layer cannot import its way down to PWM/GPIO/I2C."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.append(node.module)
    offenders = [
        name for name in imported
        if any(name == prefix or name.startswith(prefix + '.') for prefix in FORBIDDEN_MODULE_PREFIXES)
    ]
    assert offenders == [], f'{path.name} imports below the tool boundary: {offenders}'


@pytest.mark.parametrize('path', _tool_modules(), ids=lambda p: p.name)
def test_no_low_level_identifiers_in_tool_layer(path):
    """No module in the package names a wheel/PWM/I2C API in executable code."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    offenders = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_IDENTIFIERS:
            offenders.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_IDENTIFIERS:
            offenders.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            for device in FORBIDDEN_DEVICE_PATHS:
                if device in node.value:
                    offenders.add(device)
    assert offenders == set(), f'{path.name} references {sorted(offenders)}'


def test_importing_the_tool_layer_does_not_load_hardware_modules():
    program = (
        'import sys\n'
        'import jetbot_agent.agent.tools as t\n'
        'ctx = t.ToolContext(motion=t.MockMotionInterface())\n'
        'r = t.ToolRegistry(ctx, capabilities=[t.Capability.READ])\n'
        'r.register(t.MockTool(), allow=True)\n'
        'assert r.invoke("mock_echo", {"message": "hi"})["echo"] == "hi"\n'
        'bad = [m for m in sys.modules if m.startswith((\n'
        '    "jetbot_control", "jetbot_base", "jetbot_agent.hardware",\n'
        '    "smbus", "Jetson", "RPi", "busio", "board"))]\n'
        'assert not bad, bad\n'
        'r.close()\n'
        'print("OK")\n'
    )
    proc = subprocess.run([sys.executable, '-c', program], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert 'OK' in proc.stdout


def test_context_rejects_a_motor_driver():
    from jetbot_control.motors.mock import MockMotorDriver

    with pytest.raises(ToolSafetyViolation):
        ToolContext(motion=MockMotorDriver())


def test_context_rejects_the_limited_controller_itself():
    """Even the *correct* limited path object stays below the boundary."""
    from jetbot_base.diff_drive_controller import ControllerConfig, DiffDriveController
    from jetbot_control.motors.mock import MockMotorDriver

    controller = DiffDriveController(MockMotorDriver(), ControllerConfig())
    with pytest.raises(ToolSafetyViolation):
        ToolContext(motion=controller)


def test_context_rejects_the_hardware_stub_namespace():
    class SneakyMotion:
        __module__ = 'jetbot_agent.hardware.motor_controller'

        def drive(self, linear, angular, duration_sec):
            return None

        def rotate(self, angular, duration_sec):
            return None

        def stop(self):
            return None

        def status(self):
            return None

        def limits(self):
            return None

    with pytest.raises(ToolSafetyViolation):
        ToolContext(motion=SneakyMotion())


def test_context_rejects_objects_that_are_too_narrow():
    class NotMotion:
        def go(self):
            return None

    with pytest.raises(ToolSafetyViolation):
        ToolContext(motion=NotMotion())


def test_accepted_motion_object_exposes_no_wheel_level_api():
    motion = MockMotionInterface()
    ToolContext(motion=motion)  # accepted

    assert isinstance(motion, MotionInterface)
    for attr in ('set_velocity', 'set_pwm', 'duty_cycle', 'write_byte', 'clear_estop',
                 'disable_watchdog', '_driver', '_bus'):
        assert not hasattr(motion, attr), attr


def test_limits_are_readable_but_not_writable_through_the_boundary():
    motion = MockMotionInterface()
    limits = motion.limits()
    with pytest.raises(Exception):
        limits.max_linear_velocity = 10.0  # frozen dataclass
    assert motion.limits().max_linear_velocity == pytest.approx(0.25)


def test_motion_clamps_below_the_tool_boundary():
    motion = MockMotionInterface()
    registry = _registry(MockActuationTool(), capabilities=(Capability.ACTUATE,),
                         motion=motion, ack=True)
    # Schema already caps this; the boundary clamps again regardless.
    motion.drive(linear=99.0, angular=99.0, duration_sec=99.0)
    assert motion.status().last_linear == pytest.approx(0.25)
    assert motion.status().last_angular == pytest.approx(1.0)
    registry.close()


# ------------------------------------------------------- mocked LLM tool-call


def test_mocked_llm_cannot_call_a_pwm_tool():
    """Issue #21 gate: a model asking for PWM gets a permission refusal."""
    registry = _registry(MockTool())
    for name in ('set_pwm', 'set_motor_pwm', 'i2c_write', 'write_gpio', 'disable_watchdog'):
        with pytest.raises(ToolPermissionError):
            registry.invoke(name, {'value': 255})
    registry.close()


def test_mocked_llm_cannot_smuggle_pwm_arguments():
    motion = MockMotionInterface()
    registry = _registry(MockActuationTool(), capabilities=(Capability.ACTUATE,),
                         motion=motion, ack=True)
    for payload in (
        {'linear': 0.1, 'pwm': 255},
        {'linear': 0.1, 'duty_cycle': 0.9},
        {'linear': 0.1, 'i2c_bus': 7},
        {'linear': 0.1, 'timeout_sec': 999},
        {'linear': 0.1, 'left_wheel': 1.0},
    ):
        with pytest.raises(ToolValidationError):
            registry.invoke('mock_drive', payload)
    assert motion.calls == []
    registry.close()


def test_mocked_llm_cannot_exceed_declared_motion_bounds():
    motion = MockMotionInterface()
    registry = _registry(MockActuationTool(), capabilities=(Capability.ACTUATE,),
                         motion=motion, ack=True)
    with pytest.raises(ToolValidationError):
        registry.invoke('mock_drive', {'linear': 5.0})
    with pytest.raises(ToolValidationError):
        registry.invoke('mock_drive', {'linear': 0.1, 'duration_sec': 600.0})
    assert motion.calls == []
    registry.close()


def test_reserved_parameter_names_are_refused_at_declaration():
    for reserved in ('pwm', 'duty_cycle', 'i2c_bus', 'timeout_sec', 'watchdog', 'gpio'):
        with pytest.raises(ToolRegistrationError):
            class _Bad(Tool):
                name: ClassVar[str] = 'bad_tool'
                description: ClassVar[str] = 'exposes a low-level knob'
                risk: ClassVar[RiskClass] = RiskClass.READ_ONLY
                parameters: ClassVar[dict] = {
                    'type': 'object',
                    'additionalProperties': False,
                    'properties': {reserved: {'type': 'number'}},
                    'required': [],
                }

                def _run(self, context, **kwargs):
                    return None


def test_reserved_names_cover_the_dangerous_vocabulary():
    assert {'pwm', 'duty_cycle', 'i2c_bus', 'gpio', 'timeout', 'watchdog'} <= RESERVED_PARAM_NAMES


def test_open_schemas_are_refused():
    with pytest.raises(ToolRegistrationError):
        class _Open(Tool):
            name: ClassVar[str] = 'open_tool'
            description: ClassVar[str] = 'accepts anything'
            risk: ClassVar[RiskClass] = RiskClass.READ_ONLY
            parameters: ClassVar[dict] = {'type': 'object', 'properties': {}}

            def _run(self, context, **kwargs):
                return None


def test_bad_tool_metadata_is_refused():
    with pytest.raises(ToolRegistrationError):
        class _NoDescription(Tool):
            name: ClassVar[str] = 'quiet_tool'
            description: ClassVar[str] = ''
            parameters: ClassVar[dict] = {'type': 'object', 'additionalProperties': False,
                                          'properties': {}, 'required': []}

            def _run(self, context, **kwargs):
                return None

    with pytest.raises(ToolRegistrationError):
        class _BadName(Tool):
            name: ClassVar[str] = 'SetPWM'
            description: ClassVar[str] = 'bad name'
            parameters: ClassVar[dict] = {'type': 'object', 'additionalProperties': False,
                                          'properties': {}, 'required': []}

            def _run(self, context, **kwargs):
                return None


def test_actuation_tool_cannot_downgrade_its_risk():
    with pytest.raises(ToolRegistrationError):
        class _Sneaky(ActuationTool):
            name: ClassVar[str] = 'sneaky_drive'
            description: ClassVar[str] = 'pretends to be read-only'
            risk: ClassVar[RiskClass] = RiskClass.READ_ONLY
            parameters: ClassVar[dict] = {'type': 'object', 'additionalProperties': False,
                                          'properties': {}, 'required': []}

            def _run(self, context, **kwargs):
                return None


# ---------------------------------------------------------- input validation


def test_validation_rejects_unknown_missing_and_mistyped_arguments():
    registry = _registry(MockTool())
    with pytest.raises(ToolValidationError):
        registry.invoke('mock_echo', {'message': 'hi', 'extra': 1})
    with pytest.raises(ToolValidationError):
        registry.invoke('mock_echo', {})
    with pytest.raises(ToolValidationError):
        registry.invoke('mock_echo', {'message': 42})
    with pytest.raises(ToolValidationError):
        registry.invoke('mock_echo', {'message': 'hi', 'repeat': 9})
    with pytest.raises(ToolValidationError):
        registry.invoke('mock_echo', {'message': 'x' * 500})
    registry.close()


def test_validation_rejects_bool_for_number():
    registry = _registry(MockActuationTool(), capabilities=(Capability.ACTUATE,),
                         motion=MockMotionInterface(), ack=True)
    with pytest.raises(ToolValidationError):
        registry.invoke('mock_drive', {'linear': True})
    registry.close()


def test_defaults_are_applied_after_validation():
    tool = MockTool()
    registry = _registry(tool)
    result = registry.invoke('mock_echo', {'message': 'hi'})
    assert result == {'echo': 'hi', 'repeat': 1}
    assert tool.calls == [{'message': 'hi', 'repeat': 1}]
    registry.close()


def test_direct_execute_still_validates():
    tool = MockTool()
    with pytest.raises(ToolValidationError):
        tool.execute(ToolContext(), {'nope': 1})
    with pytest.raises(ToolSafetyViolation):
        tool.execute('not a context', {'message': 'hi'})  # type: ignore[arg-type]


# --------------------------------------------------------- deny-by-default


def test_registry_denies_unregistered_tools():
    registry = _registry()
    with pytest.raises(ToolPermissionError):
        registry.invoke('mock_echo', {'message': 'hi'})
    registry.close()


def test_registration_alone_does_not_permit_invocation():
    registry = _registry(MockTool(), allow=False)
    assert registry.names() == ('mock_echo',)
    assert registry.invocable() == ()
    with pytest.raises(ToolPermissionError):
        registry.invoke('mock_echo', {'message': 'hi'})

    registry.allow('mock_echo')
    assert registry.invocable() == ('mock_echo',)
    assert registry.invoke('mock_echo', {'message': 'hi'})['echo'] == 'hi'

    registry.deny('mock_echo')
    with pytest.raises(ToolPermissionError):
        registry.invoke('mock_echo', {'message': 'hi'})
    registry.close()


def test_capability_must_be_granted_even_when_allowed():
    registry = _registry(MockNetworkTool(), capabilities=())
    with pytest.raises(ToolPermissionError):
        registry.invoke('mock_search', {'query': 'jetbot'})
    registry.grant(Capability.NETWORK)
    assert registry.invoke('mock_search', {'query': 'jetbot'})['source'] == 'mock'
    registry.revoke(Capability.NETWORK)
    with pytest.raises(ToolPermissionError):
        registry.invoke('mock_search', {'query': 'jetbot'})
    registry.close()


def test_describe_hides_non_invocable_tools_from_the_model():
    registry = _registry(MockTool(), MockNetworkTool(), capabilities=(Capability.READ,))
    described = {entry['name'] for entry in registry.describe()}
    assert described == {'mock_echo'}
    assert {entry['name'] for entry in registry.describe(invocable_only=False)} == {
        'mock_echo', 'mock_search'}
    registry.close()


def test_duplicate_and_invalid_registration_are_refused():
    registry = _registry(MockTool())
    with pytest.raises(ToolRegistrationError):
        registry.register(MockTool(), allow=True)
    with pytest.raises(ToolRegistrationError):
        registry.register(object(), allow=True)  # type: ignore[arg-type]
    registry.close()


def test_registry_requires_a_tool_context():
    with pytest.raises(ToolSafetyViolation):
        ToolRegistry(object())  # type: ignore[arg-type]


# ------------------------------------------------------- actuation opt-in


def test_actuation_requires_explicit_operator_opt_in():
    motion = MockMotionInterface()
    context = ToolContext(motion=motion)

    with pytest.raises(ToolPermissionError):
        ToolRegistry(context, capabilities=(Capability.ACTUATE,))

    registry = ToolRegistry(context, capabilities=(Capability.READ,))
    registry.register(MockActuationTool(), allow=True)
    with pytest.raises(ToolPermissionError):
        registry.invoke('mock_drive', {'linear': 0.1})
    assert motion.calls == []

    with pytest.raises(ToolPermissionError):
        registry.grant(Capability.ACTUATE)

    registry.grant(Capability.ACTUATE, operator_ack=True)
    result = registry.invoke('mock_drive', {'linear': 0.1})
    assert result['last_linear'] == pytest.approx(0.1)
    assert motion.commands == ('drive',)
    registry.close()


def test_actuation_refused_without_a_motion_interface():
    registry = ToolRegistry(ToolContext(), capabilities=(Capability.ACTUATE,),
                            operator_ack_actuation=True)
    registry.register(MockActuationTool(), allow=True)
    with pytest.raises(ToolSafetyViolation):
        registry.invoke('mock_drive', {'linear': 0.1})
    registry.close()


def test_stop_tool_is_actuation_but_fail_safe():
    motion = MockMotionInterface()
    registry = _registry(MockStopTool(), capabilities=(Capability.ACTUATE,),
                         motion=motion, ack=True)
    assert registry.invoke('mock_stop') == {'moving': False, 'estop_active': False}
    assert motion.commands == ('stop',)
    registry.close()


def test_estop_below_the_boundary_refuses_motion():
    motion = MockMotionInterface(estop_active=True)
    registry = _registry(MockActuationTool(), capabilities=(Capability.ACTUATE,),
                         motion=motion, ack=True)
    with pytest.raises(ToolExecutionError):
        registry.invoke('mock_drive', {'linear': 0.1})
    assert motion.status().last_linear == 0.0
    assert MotionDenied is not None
    registry.close()


def test_registry_close_stops_motion():
    motion = MockMotionInterface()
    registry = _registry(MockActuationTool(), capabilities=(Capability.ACTUATE,),
                         motion=motion, ack=True)
    registry.invoke('mock_drive', {'linear': 0.2})
    registry.close()
    assert motion.commands[-1] == 'stop'
    with pytest.raises(Exception):
        registry.invoke('mock_drive', {'linear': 0.2})


# ------------------------------------------------------------------ watchdog


def test_every_call_runs_under_a_watchdog():
    registry = _registry(SlowTool())
    with pytest.raises(ToolTimeoutError):
        registry.invoke('mock_slow', {'sleep_sec': 0.4})
    assert registry.stats('mock_slow')['timeouts'] == 1
    registry.close()


def test_dispatch_reports_timeout_without_raising():
    registry = _registry(SlowTool())
    result = registry.dispatch('mock_slow', {'sleep_sec': 0.4})
    assert result.ok is False
    assert result.timed_out is True
    assert result.error_type == 'ToolTimeoutError'
    registry.close()


def test_actuation_timeout_stops_the_base():
    motion = MockMotionInterface()
    registry = _registry(SlowActuationTool(), capabilities=(Capability.ACTUATE,),
                         motion=motion, ack=True)
    with pytest.raises(ToolTimeoutError):
        registry.invoke('mock_slow_drive', {'sleep_sec': 0.4})
    assert 'stop' in motion.commands
    assert motion.commands[-1] == 'stop'
    registry.close()


def test_agent_cannot_extend_or_disable_the_watchdog():
    tool = SlowTool()
    tool.timeout_sec = 10_000.0  # a compromised tool or model raising its own window
    assert effective_timeout(tool) == MAX_TOOL_TIMEOUT_SEC

    tool.timeout_sec = 0.0
    assert effective_timeout(tool) == MIN_TOOL_TIMEOUT_SEC

    tool.timeout_sec = None  # type: ignore[assignment]
    assert MIN_TOOL_TIMEOUT_SEC <= effective_timeout(tool) <= MAX_TOOL_TIMEOUT_SEC

    tool.timeout_sec = 0.05
    registry = _registry(tool)
    described = registry.describe()[0]
    assert described['timeout_sec'] == pytest.approx(0.05)
    with pytest.raises(ToolTimeoutError):
        registry.invoke('mock_slow', {'sleep_sec': 0.4})
    registry.close()


def test_invoke_takes_no_caller_supplied_timeout():
    import inspect

    signature = inspect.signature(ToolRegistry.invoke)
    assert set(signature.parameters) == {'self', 'name', 'arguments'}
    dispatch_signature = inspect.signature(ToolRegistry.dispatch)
    assert set(dispatch_signature.parameters) == {'self', 'name', 'arguments'}


def test_tool_faults_are_wrapped_not_propagated_raw():
    registry = _registry(FailingTool())
    with pytest.raises(ToolExecutionError):
        registry.invoke('mock_fail')
    result = registry.dispatch('mock_fail')
    assert result.ok is False
    assert result.error_type == 'ToolExecutionError'
    registry.close()


def test_dispatch_success_carries_risk_and_duration():
    registry = _registry(MockTool())
    result = registry.dispatch('mock_echo', {'message': 'hi'})
    assert result.ok is True
    assert result.risk == RiskClass.READ_ONLY.value
    assert result.duration_sec >= 0.0
    registry.close()


def test_denials_are_counted_for_audit():
    registry = _registry(MockTool(), allow=False)
    registry.dispatch('mock_echo', {'message': 'hi'})
    assert registry.stats('mock_echo')['denials'] == 1
    registry.close()
