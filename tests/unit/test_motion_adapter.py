from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ros2_ws' / 'src' / 'jetbot_base'))

from jetbot_agent._stage import StageNotReady
from jetbot_agent.agent.tools import (
    FORBIDDEN_MOTION_ATTRS,
    MotionDenied,
    MotionInterface,
    MotionLimits,
    ToolContext,
    assert_narrow_motion,
)
from jetbot_agent.navigation import (
    BoundedMotionAdapter,
    HARD_MAX_DURATION_SEC,
    HARD_MAX_LINEAR_VELOCITY,
    MIN_DURATION_SEC,
    MockCmdVelSink,
    MotionBackendUnavailable,
    MotionEstop,
    MotionIntent,
    RosCmdVelSink,
    UnavailableVlaPolicy,
    apply_intent,
    bounded_limits,
    limits_from_robot_config,
    load_robot_limits,
    mock_controller_sink,
)
from jetbot_agent.navigation.motion_adapter import LOGGER as ADAPTER_LOGGER

ADAPTER_LOGGER_NAME = ADAPTER_LOGGER.name


class Clock:
    """Manually advanced monotonic clock, matching the ``now=`` style used by
    ``tests/unit/test_motor_and_controller.py``."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


def _adapter(limits=None, *, estop=None, sink=None, clock=None):
    clock = clock or Clock()
    limits = limits or MotionLimits()
    sink = sink if sink is not None else MockCmdVelSink(
        cmd_vel_timeout_sec=limits.cmd_vel_timeout_sec)
    adapter = BoundedMotionAdapter(sink, limits, estop=estop, clock=clock)
    return adapter, sink, clock


# ------------------------------------------------- the boundary accepts it


def test_adapter_passes_assert_narrow_motion():
    adapter, _, _ = _adapter()
    assert_narrow_motion(adapter)  # would raise ToolSafetyViolation
    ToolContext(motion=adapter)  # and so would this
    assert isinstance(adapter, MotionInterface)


def test_adapter_exposes_no_attribute_below_the_boundary():
    adapter, _, _ = _adapter()
    for attr in sorted(FORBIDDEN_MOTION_ATTRS):
        assert not hasattr(adapter, attr), attr


def test_adapter_lives_outside_the_tools_package_and_is_not_a_controller():
    adapter, _, _ = _adapter()
    module = type(adapter).__module__
    assert module.startswith('jetbot_agent.navigation')
    assert 'agent.tools' not in module
    mro_names = {cls.__name__ for cls in type(adapter).__mro__}
    assert mro_names.isdisjoint({'MotorDriver', 'DiffDriveController', 'MotorController'})


def test_adapter_cannot_clear_the_estop_it_observes():
    """The latch is a separate object; the tool layer only ever sees the adapter."""
    estop = MotionEstop()
    adapter, _, _ = _adapter(estop=estop)
    assert not hasattr(adapter, 'clear_estop')
    assert callable(estop.clear)


# ------------------------------------------------------------------ clamping


def test_commands_at_the_limit_are_not_clamped():
    limits = MotionLimits()
    adapter, sink, _ = _adapter(limits)
    status = adapter.drive(limits.max_linear_velocity, limits.max_angular_velocity, 1.0)
    assert adapter.clamps == ()
    assert sink.last_twist == pytest.approx((0.25, 1.0))
    assert status.last_linear == pytest.approx(0.25)


def test_commands_beyond_the_limit_are_clamped_and_logged(caplog):
    adapter, sink, _ = _adapter()
    with caplog.at_level(logging.WARNING, logger=ADAPTER_LOGGER_NAME):
        status = adapter.drive(9.0, 9.0, 1.0)
    assert status.last_linear == pytest.approx(0.25)
    assert status.last_angular == pytest.approx(1.0)
    assert sink.last_twist == pytest.approx((0.25, 1.0))
    clamped = {event.field: event for event in adapter.clamps}
    assert set(clamped) == {'linear', 'angular'}
    assert clamped['linear'].requested == pytest.approx(9.0)
    assert clamped['linear'].applied == pytest.approx(0.25)
    assert 'motion_clamped' in caplog.text


def test_negative_commands_beyond_the_limit_are_clamped():
    adapter, sink, _ = _adapter()
    status = adapter.drive(-9.0, -9.0, 1.0)
    assert status.last_linear == pytest.approx(-0.25)
    assert sink.last_twist == pytest.approx((-0.25, -1.0))


def test_clamping_is_a_reduction_not_a_silent_rejection():
    adapter, sink, _ = _adapter()
    adapter.drive(9.0, 0.0, 1.0)
    assert adapter.status().moving is True
    assert sink.commands, 'a clamped command must still be issued'


def test_rotate_forces_linear_velocity_to_zero():
    adapter, sink, _ = _adapter()
    status = adapter.rotate(9.0, 1.0)
    assert status.last_linear == 0.0
    assert sink.last_twist == pytest.approx((0.0, 1.0))


def test_nan_magnitudes_fail_towards_stopped():
    adapter, sink, _ = _adapter()
    adapter.drive(float('nan'), float('nan'), 1.0)
    assert sink.last_twist == pytest.approx((0.0, 0.0))
    assert adapter.status().moving is False


def test_adapter_bounds_limits_it_is_handed():
    reckless = MotionLimits(max_linear_velocity=5.0, max_angular_velocity=50.0,
                            cmd_vel_timeout_sec=60.0, max_duration_sec=600.0)
    bounded = bounded_limits(reckless)
    assert bounded.max_linear_velocity == pytest.approx(HARD_MAX_LINEAR_VELOCITY)
    assert bounded.max_duration_sec == pytest.approx(HARD_MAX_DURATION_SEC)

    adapter, _, _ = _adapter(reckless)
    assert adapter.limits().max_linear_velocity == pytest.approx(HARD_MAX_LINEAR_VELOCITY)


# ----------------------------------------------------------- duration bounds


def test_duration_is_capped_by_the_limits():
    adapter, _, clock = _adapter()
    adapter.drive(0.1, 0.0, 1e9)
    capped = {event.field: event for event in adapter.clamps}['duration_sec']
    assert capped.applied == pytest.approx(MotionLimits().max_duration_sec)

    clock.advance(MotionLimits().max_duration_sec + 0.01)
    adapter.poll()
    assert adapter.status().moving is False


def test_zero_and_negative_durations_are_clamped_up_not_treated_as_forever():
    for requested in (0.0, -5.0, float('nan')):
        adapter, _, clock = _adapter()
        adapter.drive(0.1, 0.0, requested)
        applied = {event.field: event for event in adapter.clamps}['duration_sec'].applied
        assert applied == pytest.approx(MIN_DURATION_SEC)
        clock.advance(MIN_DURATION_SEC + 0.001)
        adapter.poll()
        assert adapter.status().moving is False


def test_no_command_can_outlive_the_hard_duration_ceiling():
    adapter, sink, clock = _adapter(MotionLimits(max_duration_sec=600.0))
    adapter.drive(0.1, 0.0, 600.0)
    # Refresh diligently for longer than the ceiling; the deadline still fires.
    for _ in range(200):
        clock.advance(0.2)
        adapter.poll()
        if not adapter.status().moving:
            break
    assert adapter.status().moving is False
    assert clock.now - 1000.0 <= HARD_MAX_DURATION_SEC + 0.2
    assert sink.halts, 'the base must have been halted'


# --------------------------------------------------------------- the watchdog


def test_watchdog_expiry_issues_a_stop():
    adapter, sink, clock = _adapter()
    adapter.drive(0.2, 0.0, 3.0)
    assert adapter.status().moving is True

    clock.advance(0.6)  # longer than cmd_vel_timeout_sec = 0.5
    status = adapter.poll()
    assert status.moving is False
    assert status.watchdog_armed is False
    assert 'watchdog_expired' in sink.halts
    assert sink.last_twist == pytest.approx((0.0, 0.0))


def test_a_stale_command_reads_as_stopped_even_without_a_poll():
    adapter, sink, clock = _adapter()
    adapter.drive(0.2, 0.0, 3.0)
    clock.advance(0.6)
    assert adapter.status().moving is False
    assert sink.moving_at(clock.now) is False


def test_polling_inside_the_window_sustains_motion():
    adapter, sink, clock = _adapter()
    adapter.drive(0.2, 0.0, 2.0)
    published = len(sink.commands)
    clock.advance(0.3)  # past the refresh interval, inside the watchdog window
    status = adapter.poll()
    assert status.moving is True
    assert len(sink.commands) == published + 1
    assert sink.last_twist == pytest.approx((0.2, 0.0))


def test_a_late_poll_stops_instead_of_resuming():
    adapter, sink, clock = _adapter()
    adapter.drive(0.2, 0.0, 3.0)
    clock.advance(1.5)
    adapter.poll()
    assert adapter.status().moving is False
    assert sink.last_twist == pytest.approx((0.0, 0.0))


def test_deadline_stops_before_the_watchdog_when_it_is_shorter():
    adapter, sink, clock = _adapter()
    adapter.drive(0.2, 0.0, 0.3)
    clock.advance(0.35)
    adapter.poll()
    assert 'duration_elapsed' in sink.halts
    assert adapter.status().moving is False


# ------------------------------------------------------------------- e-stop


def test_stop_is_always_permitted_even_while_estopped():
    estop = MotionEstop()
    adapter, sink, _ = _adapter(estop=estop)
    estop.trigger('operator')
    status = adapter.stop()  # must not raise
    assert status.moving is False
    assert sink.last_twist == pytest.approx((0.0, 0.0))


def test_stop_never_raises_even_when_the_sink_fails():
    class BrokenSink(MockCmdVelSink):
        def halt(self, reason: str = '') -> None:
            raise RuntimeError('transport down')

    adapter, sink, _ = _adapter(sink=BrokenSink())
    assert adapter.stop().moving is False


def test_estop_halts_motion_and_refuses_actuation_until_cleared():
    estop = MotionEstop()
    adapter, sink, _ = _adapter(estop=estop)
    adapter.drive(0.2, 0.0, 3.0)
    assert adapter.status().moving is True

    estop.trigger('harness')
    assert sink.last_twist == pytest.approx((0.0, 0.0))
    assert adapter.status().moving is False
    assert adapter.status().estop_active is True

    with pytest.raises(MotionDenied):
        adapter.drive(0.1, 0.0, 1.0)
    with pytest.raises(MotionDenied):
        adapter.rotate(0.5, 1.0)
    adapter.stop()  # still fine

    estop.clear()
    assert adapter.drive(0.1, 0.0, 1.0).moving is True


def test_estop_latches_until_an_explicit_clear():
    estop = MotionEstop()
    adapter, _, _ = _adapter(estop=estop)
    estop.trigger('operator')
    estop.trigger('again')
    assert estop.active is True
    assert estop.reason == 'operator'
    for _ in range(3):
        with pytest.raises(MotionDenied):
            adapter.drive(0.1, 0.0, 1.0)
    estop.clear()
    assert estop.active is False


def test_estop_hooks_are_fail_safe():
    estop = MotionEstop()
    estop.add_hook(lambda reason: (_ for _ in ()).throw(RuntimeError('bad hook')))
    adapter, sink, _ = _adapter(estop=estop)
    adapter.drive(0.2, 0.0, 2.0)
    estop.trigger('operator')  # a raising hook must not block the stop
    assert adapter.status().moving is False
    assert sink.last_twist == pytest.approx((0.0, 0.0))


def test_polling_while_estopped_keeps_the_base_stopped():
    estop = MotionEstop()
    adapter, sink, clock = _adapter(estop=estop)
    adapter.drive(0.2, 0.0, 3.0)
    estop.trigger('operator')
    clock.advance(0.1)
    assert adapter.poll().moving is False
    assert sink.last_twist == pytest.approx((0.0, 0.0))


# ------------------------------------------------------------------- limits


def test_limits_come_from_robot_yaml():
    limits = load_robot_limits()
    assert limits.max_linear_velocity == pytest.approx(0.25)
    assert limits.max_angular_velocity == pytest.approx(1.0)
    assert limits.cmd_vel_timeout_sec == pytest.approx(0.5)


def test_limits_from_a_config_mapping_fall_back_to_defaults():
    limits = limits_from_robot_config({})
    assert limits.max_linear_velocity == pytest.approx(MotionLimits().max_linear_velocity)


def test_limits_are_readable_but_not_writable():
    adapter, _, _ = _adapter()
    with pytest.raises(Exception):
        adapter.limits().max_linear_velocity = 10.0
    assert adapter.limits().max_linear_velocity == pytest.approx(0.25)


# --------------------------------------------------------------------- sinks


def test_mock_sink_models_the_command_watchdog():
    sink = MockCmdVelSink(cmd_vel_timeout_sec=0.5)
    sink.publish(0.2, 0.0, now=1000.0)
    assert sink.moving_at(1000.2) is True
    assert sink.moving_at(1000.8) is False
    sink.halt('test')
    assert sink.last_twist == (0.0, 0.0)


def test_adapter_drives_jetbot_base_through_the_controller_sink():
    """tool vocabulary -> adapter -> sink -> jetbot_base -> mock motor backend."""
    sink = mock_controller_sink()
    clock = Clock()
    adapter = BoundedMotionAdapter(sink, MotionLimits(), clock=clock)

    adapter.drive(0.2, 0.0, 2.0)
    assert sink.status()['left_velocity'] != 0.0
    assert sink.status()['backend'] == 'mock'

    clock.advance(0.6)
    adapter.poll()
    assert sink.status()['left_velocity'] == 0.0
    assert sink.status()['right_velocity'] == 0.0


def test_controller_sink_refuses_something_that_is_not_a_controller():
    from jetbot_agent.navigation import ControllerCmdVelSink

    with pytest.raises(MotionBackendUnavailable):
        ControllerCmdVelSink(object())


def test_ros_sink_needs_a_node_and_never_creates_one():
    with pytest.raises(MotionBackendUnavailable):
        RosCmdVelSink(None)
    with pytest.raises(MotionBackendUnavailable):
        RosCmdVelSink(object())


def test_ros_sink_publishes_a_twist_shaped_message():
    class Vector:
        def __init__(self) -> None:
            self.x = 0.0
            self.y = 0.0
            self.z = 0.0

    class FakeTwist:
        def __init__(self) -> None:
            self.linear = Vector()
            self.angular = Vector()

    class FakePublisher:
        def __init__(self) -> None:
            self.published = []

        def publish(self, message) -> None:
            self.published.append((message.linear.x, message.angular.z))

    class FakeNode:
        def __init__(self) -> None:
            self.publisher = FakePublisher()
            self.topics = []

        def create_publisher(self, message_type, topic, depth):
            self.topics.append((message_type, topic, depth))
            return self.publisher

    node = FakeNode()
    sink = RosCmdVelSink(node, twist_type=FakeTwist)
    assert node.topics[0][1] == '/cmd_vel'

    adapter = BoundedMotionAdapter(sink, MotionLimits(), clock=Clock())
    adapter.drive(9.0, 0.0, 1.0)
    assert node.publisher.published[-1] == pytest.approx((0.25, 0.0))
    adapter.stop()
    assert node.publisher.published[-1] == pytest.approx((0.0, 0.0))


def test_importing_the_navigation_package_pulls_in_no_ros_or_hardware():
    program = (
        'import sys\n'
        'import jetbot_agent.navigation as nav\n'
        'adapter = nav.BoundedMotionAdapter()\n'
        'adapter.drive(9.0, 9.0, 9.0)\n'
        'assert adapter.status().last_linear == 0.25\n'
        'bad = [m for m in sys.modules if m.startswith((\n'
        '    "rclpy", "geometry_msgs", "jetbot_base", "jetbot_control",\n'
        '    "jetbot_agent.hardware", "smbus", "Jetson", "RPi", "busio", "board"))]\n'
        'assert not bad, bad\n'
        'print("OK")\n'
    )
    proc = subprocess.run([sys.executable, '-c', program], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert 'OK' in proc.stdout


# ----------------------------------------------------------------- VLA seam


def test_smolvla_is_a_seam_not_an_implementation():
    policy = UnavailableVlaPolicy()
    assert policy.name == 'smolvla'
    with pytest.raises(StageNotReady):
        policy.propose({'image': None})


def test_vla_intents_are_clamped_by_the_same_path_as_a_tool_call():
    adapter, sink, _ = _adapter()
    status = apply_intent(adapter, MotionIntent('drive', linear=9.0, angular=9.0,
                                                duration_sec=9.0, source='smolvla',
                                                confidence=1.0))
    assert status.last_linear == pytest.approx(0.25)
    assert sink.last_twist == pytest.approx((0.25, 1.0))
    fields = {event.field for event in adapter.clamps}
    assert {'linear', 'angular', 'duration_sec'} <= fields


def test_vla_intents_are_refused_while_estopped():
    estop = MotionEstop()
    adapter, _, _ = _adapter(estop=estop)
    estop.trigger('operator')
    with pytest.raises(MotionDenied):
        apply_intent(adapter, MotionIntent('drive', linear=0.1, confidence=1.0))
    # a stop intent is still honoured
    assert apply_intent(adapter, MotionIntent('stop')).moving is False


def test_low_confidence_intents_stop_instead_of_moving():
    adapter, sink, _ = _adapter()
    with pytest.raises(MotionDenied):
        apply_intent(adapter, MotionIntent('drive', linear=0.2, confidence=0.1),
                     min_confidence=0.8)
    assert adapter.status().moving is False
    assert sink.last_twist == pytest.approx((0.0, 0.0))


def test_intent_kinds_are_closed():
    with pytest.raises(ValueError):
        MotionIntent('set_wheel_speeds', linear=1.0)
    with pytest.raises(TypeError):
        apply_intent(_adapter()[0], {'kind': 'drive'})  # type: ignore[arg-type]
