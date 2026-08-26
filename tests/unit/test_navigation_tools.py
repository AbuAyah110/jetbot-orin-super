from __future__ import annotations

import ast
import sys
import time
from pathlib import Path
from typing import Any, ClassVar, Mapping

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ros2_ws' / 'src' / 'jetbot_base'))

from jetbot_agent.agent.hermes_harness import HermesHarness, FakeBrain
from jetbot_agent.agent.tools import (
    ActuationTool,
    Capability,
    MockMotionInterface,
    RESERVED_PARAM_NAMES,
    RiskClass,
    ToolContext,
    ToolExecutionError,
    ToolPermissionError,
    ToolRegistry,
    ToolTimeoutError,
    ToolValidationError,
)
from jetbot_agent.agent.tools.navigation_tools import (
    MAX_NAV_ANGLE_DEG,
    MAX_NAV_DISTANCE_M,
    MAX_NAV_DURATION_SEC,
    MAX_NAV_SPEED,
    MAX_NAV_TURN_RATE,
    MIN_NAV_SPEED,
    NavDriveTool,
    NavRotateTool,
    NavStatusTool,
    NavStopTool,
    navigation_tools,
    register_navigation_tools,
)
from jetbot_agent.navigation import BoundedMotionAdapter, MockCmdVelSink, MotionEstop

TOOLS_DIR = ROOT / 'jetbot_agent' / 'agent' / 'tools'
NAV_TOOLS_PATH = TOOLS_DIR / 'navigation_tools.py'

ACTUATION_NAMES = ('nav_drive', 'nav_rotate', 'nav_stop')


class Clock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


def _registry(motion, *, capabilities=(Capability.READ, Capability.ACTUATE), ack=True,
              allow=True):
    registry = ToolRegistry(ToolContext(motion=motion), capabilities=capabilities,
                            operator_ack_actuation=ack)
    register_navigation_tools(registry, allow=allow)
    return registry


def _adapter(*, estop=None, clock=None, limits=None):
    clock = clock or Clock()
    sink = MockCmdVelSink()
    adapter = BoundedMotionAdapter(sink, limits, estop=estop, clock=clock)
    return adapter, sink, clock


# ------------------------------------------------------------ the tool set


def test_the_i5_tool_set_is_declared():
    tools = {tool.name: tool for tool in navigation_tools()}
    assert set(tools) == {'nav_drive', 'nav_rotate', 'nav_stop', 'nav_status'}
    for name in ACTUATION_NAMES:
        assert tools[name].risk is RiskClass.ACTUATION
        assert tools[name].capability is Capability.ACTUATE
    assert tools['nav_status'].risk is RiskClass.READ_ONLY
    assert tools['nav_status'].capability is Capability.READ


def test_registration_is_deny_by_default():
    registry = _registry(MockMotionInterface(), allow=False)
    assert registry.names() == ('nav_drive', 'nav_rotate', 'nav_status', 'nav_stop')
    assert registry.invocable() == ()
    with pytest.raises(ToolPermissionError):
        registry.invoke('nav_drive', {'distance_m': 0.1})
    registry.close()


def test_navigation_schemas_are_closed_and_bounded():
    for tool in navigation_tools():
        schema = tool.parameters
        assert schema['type'] == 'object'
        assert schema['additionalProperties'] is False
        assert set(schema['properties']).isdisjoint(RESERVED_PARAM_NAMES)
        for key, spec in schema['properties'].items():
            assert spec['type'] == 'number', key
            assert 'minimum' in spec and 'maximum' in spec, key


def test_no_reserved_parameter_is_reachable_through_a_navigation_tool():
    motion = MockMotionInterface()
    registry = _registry(motion)
    for payload in (
        {'distance_m': 0.1, 'pwm': 255},
        {'distance_m': 0.1, 'duty_cycle': 0.9},
        {'distance_m': 0.1, 'i2c_bus': 7},
        {'distance_m': 0.1, 'timeout_sec': 999},
        {'distance_m': 0.1, 'watchdog': 0},
        {'distance_m': 0.1, 'left_pwm': 1.0},
        {'distance_m': 0.1, 'wheel_velocity': 1.0},
    ):
        with pytest.raises(ToolValidationError):
            registry.invoke('nav_drive', payload)
    assert motion.calls == []
    registry.close()


# --------------------------------------------------------- magnitude bounds


def test_nav_drive_magnitudes_are_bounded_by_the_schema():
    motion = MockMotionInterface()
    registry = _registry(motion)
    for payload in (
        {'distance_m': 5.0},
        {'distance_m': -5.0},
        {'distance_m': 0.1, 'speed': 9.0},
        {'distance_m': 0.1, 'speed': 0.0},
        {'distance_m': 0.1, 'turn_rate': 9.0},
    ):
        with pytest.raises(ToolValidationError):
            registry.invoke('nav_drive', payload)
    assert motion.calls == []
    registry.close()


def test_nav_rotate_magnitudes_are_bounded_by_the_schema():
    motion = MockMotionInterface()
    registry = _registry(motion)
    for payload in (
        {'angle_deg': 720.0},
        {'angle_deg': -720.0},
        {'angle_deg': 90.0, 'turn_rate': 9.0},
        {'angle_deg': 90.0, 'turn_rate': 0.0},
    ):
        with pytest.raises(ToolValidationError):
            registry.invoke('nav_rotate', payload)
    assert motion.calls == []
    registry.close()


def test_the_tool_surface_is_more_conservative_than_the_base_limits():
    limits = MockMotionInterface().limits()
    assert MAX_NAV_SPEED <= limits.max_linear_velocity
    assert MAX_NAV_TURN_RATE <= limits.max_angular_velocity
    assert MAX_NAV_DURATION_SEC <= limits.max_duration_sec


def test_a_clamped_command_still_reaches_the_base_and_is_reported():
    """Clamping is a reduction, not a silent drop."""
    adapter, sink, _ = _adapter(limits=MockMotionInterface().limits())
    registry = _registry(adapter)
    result = registry.invoke('nav_drive', {'distance_m': 0.4, 'speed': MAX_NAV_SPEED})
    assert result['moving'] is True
    assert result['linear'] == pytest.approx(0.25)
    assert sink.last_twist == pytest.approx((0.25, 0.0))
    registry.close()


# ---------------------------------------------------------- duration bounds


def test_no_navigation_call_can_request_indefinite_motion():
    motion = MockMotionInterface()
    registry = _registry(motion)
    registry.invoke('nav_drive', {'distance_m': MAX_NAV_DISTANCE_M, 'speed': MIN_NAV_SPEED})
    registry.invoke('nav_rotate', {'angle_deg': MAX_NAV_ANGLE_DEG, 'turn_rate': 0.1})
    durations = [fields['duration_sec'] for _, fields in motion.calls]
    assert durations, 'both calls should have crossed the boundary'
    for duration in durations:
        assert 0.0 < duration <= MAX_NAV_DURATION_SEC
    registry.close()


def test_a_capped_duration_is_reported_not_hidden():
    registry = _registry(MockMotionInterface())
    result = registry.invoke('nav_drive',
                             {'distance_m': MAX_NAV_DISTANCE_M, 'speed': MIN_NAV_SPEED})
    assert result['duration_capped'] is True
    assert result['duration_sec'] == pytest.approx(MAX_NAV_DURATION_SEC)
    assert abs(result['reachable_distance_m']) < abs(result['requested_distance_m'])
    assert result['open_loop'] is True
    registry.close()


def test_a_short_move_is_not_capped():
    registry = _registry(MockMotionInterface())
    result = registry.invoke('nav_drive', {'distance_m': 0.1, 'speed': 0.1})
    assert result['duration_capped'] is False
    assert result['duration_sec'] == pytest.approx(1.0)
    registry.close()


def test_reverse_and_left_turns_keep_their_sign():
    motion = MockMotionInterface()
    registry = _registry(motion)
    registry.invoke('nav_drive', {'distance_m': -0.2, 'speed': 0.1})
    assert motion.calls[-1][1]['linear'] == pytest.approx(-0.1)
    registry.invoke('nav_rotate', {'angle_deg': -90.0, 'turn_rate': 0.5})
    assert motion.calls[-1][1]['angular'] == pytest.approx(-0.5)
    registry.invoke('nav_rotate', {'angle_deg': 90.0, 'turn_rate': 0.5})
    assert motion.calls[-1][1]['angular'] == pytest.approx(0.5)
    registry.close()


# ----------------------------------------------------------- actuation gate


def test_actuation_needs_the_capability_and_an_operator_ack():
    adapter, sink, _ = _adapter()
    context = ToolContext(motion=adapter)

    with pytest.raises(ToolPermissionError):
        ToolRegistry(context, capabilities=(Capability.ACTUATE,))

    registry = ToolRegistry(context, capabilities=(Capability.READ,))
    register_navigation_tools(registry, allow=True)
    for name, payload in (('nav_drive', {'distance_m': 0.1}),
                          ('nav_rotate', {'angle_deg': 45.0}),
                          ('nav_stop', {})):
        with pytest.raises(ToolPermissionError):
            registry.invoke(name, payload)
    assert sink.commands == []

    with pytest.raises(ToolPermissionError):
        registry.grant(Capability.ACTUATE)

    registry.grant(Capability.ACTUATE, operator_ack=True)
    assert registry.invoke('nav_drive', {'distance_m': 0.1})['moving'] is True
    registry.close()


def test_the_model_is_not_told_about_ungranted_actuation():
    registry = ToolRegistry(ToolContext(motion=MockMotionInterface()),
                            capabilities=(Capability.READ,))
    register_navigation_tools(registry, allow=True)
    described = {entry['name'] for entry in registry.describe()}
    assert described == {'nav_status'}
    registry.close()


def test_nav_status_is_read_only_and_needs_no_actuation_grant():
    adapter, sink, _ = _adapter()
    registry = ToolRegistry(ToolContext(motion=adapter), capabilities=(Capability.READ,))
    register_navigation_tools(registry, allow=True)
    status = registry.invoke('nav_status')
    assert status['moving'] is False
    assert status['estop_active'] is False
    assert status['limits']['max_linear_velocity'] == pytest.approx(0.25)
    assert sink.commands == [], 'a status read must not command the base'
    registry.close()


# ------------------------------------------------------------------- stop


def test_nav_stop_takes_no_arguments():
    registry = _registry(MockMotionInterface())
    with pytest.raises(ToolValidationError):
        registry.invoke('nav_stop', {'now': True})
    registry.close()


def test_stop_is_honoured_while_drive_is_refused():
    estop = MotionEstop()
    adapter, sink, _ = _adapter(estop=estop)
    registry = _registry(adapter)

    registry.invoke('nav_drive', {'distance_m': 0.3, 'speed': 0.1})
    estop.trigger('operator')

    with pytest.raises(ToolExecutionError):
        registry.invoke('nav_drive', {'distance_m': 0.1})
    assert registry.invoke('nav_stop')['moving'] is False
    assert sink.last_twist == pytest.approx((0.0, 0.0))
    registry.close()


def test_the_deterministic_stop_paths_do_not_go_through_the_tool_gate():
    """Losing the actuation grant must never leave the robot moving."""
    adapter, sink, _ = _adapter()
    registry = _registry(adapter)
    registry.invoke('nav_drive', {'distance_m': 0.3, 'speed': 0.1})
    assert adapter.status().moving is True

    registry.revoke(Capability.ACTUATE)
    with pytest.raises(ToolPermissionError):
        registry.invoke('nav_stop')
    registry.estop('operator')  # registry-level stop, no capability needed
    assert adapter.status().moving is False
    assert sink.last_twist == pytest.approx((0.0, 0.0))
    registry.close()


def test_registry_close_stops_the_base():
    adapter, sink, _ = _adapter()
    registry = _registry(adapter)
    registry.invoke('nav_drive', {'distance_m': 0.3, 'speed': 0.1})
    registry.close()
    assert adapter.status().moving is False
    assert sink.halts


# --------------------------------------------------------- watchdog / estop


def test_watchdog_expiry_stops_a_navigation_command():
    clock = Clock()
    adapter, sink, _ = _adapter(clock=clock)
    registry = _registry(adapter)
    registry.invoke('nav_drive', {'distance_m': 0.3, 'speed': 0.1})
    assert adapter.status().moving is True

    clock.advance(0.6)  # nobody refreshed the command inside cmd_vel_timeout_sec
    adapter.poll()
    assert adapter.status().moving is False
    assert 'watchdog_expired' in sink.halts

    status = registry.invoke('nav_status')
    assert status['moving'] is False
    assert status['watchdog_armed'] is False
    registry.close()


def test_a_tool_call_that_overruns_its_watchdog_stops_the_base():
    class OverrunningNavTool(ActuationTool):
        name: ClassVar[str] = 'nav_overrun'
        description: ClassVar[str] = 'Start motion then overrun. Test fixture only.'
        timeout_sec: ClassVar[float] = 0.05
        parameters: ClassVar[Mapping[str, Any]] = {
            'type': 'object', 'additionalProperties': False,
            'properties': {}, 'required': [],
        }

        def _run(self, context: ToolContext, **kwargs: Any) -> Any:
            context.require_motion().drive(0.1, 0.0, 2.0)
            time.sleep(0.4)
            return {'slept': True}

    adapter, sink, _ = _adapter()
    registry = ToolRegistry(ToolContext(motion=adapter),
                            capabilities=(Capability.ACTUATE,),
                            operator_ack_actuation=True)
    registry.register(OverrunningNavTool(), allow=True)
    with pytest.raises(ToolTimeoutError):
        registry.invoke('nav_overrun')
    assert adapter.status().moving is False
    assert sink.last_twist == pytest.approx((0.0, 0.0))
    registry.close()


def test_a_harness_estop_halts_navigation_until_it_is_cleared():
    estop = MotionEstop()
    adapter, sink, _ = _adapter(estop=estop)
    registry = _registry(adapter)
    harness = HermesHarness(FakeBrain([]))
    harness.add_estop_hook(estop.trigger)

    registry.invoke('nav_drive', {'distance_m': 0.3, 'speed': 0.1})
    assert adapter.status().moving is True

    harness.trigger_estop('operator button')
    assert adapter.status().moving is False
    assert adapter.status().estop_active is True
    assert sink.last_twist == pytest.approx((0.0, 0.0))

    with pytest.raises(ToolExecutionError):
        registry.invoke('nav_drive', {'distance_m': 0.1})
    with pytest.raises(ToolExecutionError):
        registry.invoke('nav_rotate', {'angle_deg': 45.0})
    assert registry.invoke('nav_stop')['moving'] is False

    harness.clear_estop()
    estop.clear()
    assert registry.invoke('nav_drive', {'distance_m': 0.1})['moving'] is True
    harness.shutdown()
    registry.close()


# ------------------------------------------------- structural guard coverage


def test_the_existing_ast_guard_covers_this_module():
    """The I2 parametrized scan globs the package, so I5's file is included."""
    scanned = sorted(path.name for path in TOOLS_DIR.glob('*.py'))
    assert 'navigation_tools.py' in scanned


def test_navigation_tools_module_stays_above_the_boundary():
    forbidden_modules = ('jetbot_control', 'jetbot_base', 'jetbot_agent.hardware',
                         'jetbot_agent.navigation', 'smbus', 'smbus2', 'busio',
                         'board', 'Jetson', 'RPi', 'periphery')
    forbidden_identifiers = {'PCA9685', 'SMBus', 'DiffDriveController', 'MotorDriver',
                             'MockMotorDriver', 'MotorController', 'GPIO', 'set_velocity',
                             'set_pwm', 'set_duty_cycle', 'write_byte', 'write_byte_data',
                             'write_i2c_block_data', 'twist_to_wheel_speeds'}
    forbidden_paths = ('/dev/i2c', '/dev/mem', '/sys/class/pwm', '/dev/gpiochip')

    tree = ast.parse(NAV_TOOLS_PATH.read_text(encoding='utf-8'), filename=str(NAV_TOOLS_PATH))
    imported = []
    offenders = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Name) and node.id in forbidden_identifiers:
            offenders.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in forbidden_identifiers:
            offenders.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            offenders.update(path for path in forbidden_paths if path in node.value)

    bad_imports = [name for name in imported
                   if any(name == prefix or name.startswith(prefix + '.')
                          for prefix in forbidden_modules)]
    assert bad_imports == [], bad_imports
    assert offenders == set(), sorted(offenders)


def test_a_navigation_tool_cannot_be_handed_the_controller_itself():
    from jetbot_agent.agent.tools import ToolSafetyViolation
    from jetbot_base.diff_drive_controller import ControllerConfig, DiffDriveController
    from jetbot_control.motors.mock import MockMotorDriver

    controller = DiffDriveController(MockMotorDriver(), ControllerConfig())
    with pytest.raises(ToolSafetyViolation):
        ToolContext(motion=controller)

    # The adapter over the same controller is accepted, because it is narrow.
    from jetbot_agent.navigation import ControllerCmdVelSink

    adapter = BoundedMotionAdapter(ControllerCmdVelSink(controller), clock=Clock())
    ToolContext(motion=adapter)
    registry = _registry(adapter)
    assert registry.invoke('nav_drive', {'distance_m': 0.1})['moving'] is True
    registry.close()


def test_tool_instances_are_stateless_between_registries():
    for _ in range(2):
        registry = _registry(MockMotionInterface())
        assert registry.invoke('nav_drive', {'distance_m': 0.1})['moving'] is True
        registry.close()


def test_nav_tool_classes_are_importable_individually():
    assert NavDriveTool.name == 'nav_drive'
    assert NavRotateTool.name == 'nav_rotate'
    assert NavStopTool.name == 'nav_stop'
    assert NavStatusTool.name == 'nav_status'
