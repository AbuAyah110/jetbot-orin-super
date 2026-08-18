from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ros2_ws' / 'src' / 'jetbot_base'))

from jetbot_control.kinematics import limit_twist, twist_to_wheel_speeds
from jetbot_control.motors.factory import create_motor_driver
from jetbot_control.motors.mock import MockMotorDriver
from jetbot_base.diff_drive_controller import ControllerConfig, DiffDriveController


def test_mock_set_and_stop():
    m = MockMotorDriver()
    m.set_velocity(0.5, -0.25)
    h = m.health()
    assert h.left_velocity == pytest.approx(0.5)
    assert h.right_velocity == pytest.approx(-0.25)
    m.stop()
    h = m.health()
    assert h.left_velocity == 0.0
    assert h.right_velocity == 0.0


def test_mock_clamps():
    m = MockMotorDriver()
    m.set_velocity(5.0, -5.0)
    h = m.health()
    assert h.left_velocity == 1.0
    assert h.right_velocity == -1.0


def test_factory_mock():
    d = create_motor_driver('mock')
    assert d.health().backend == 'mock'


def test_limit_twist():
    lin, ang = limit_twist(1.0, 2.0, 0.25, 1.0)
    assert lin == 0.25
    assert ang == 1.0


def test_twist_to_wheels_forward():
    left, right = twist_to_wheel_speeds(0.2, 0.0, 0.12, 1.0)
    assert left == pytest.approx(right)
    assert left > 0


def test_watchdog_stops_on_timeout():
    driver = MockMotorDriver()
    ctrl = DiffDriveController(
        driver,
        ControllerConfig(cmd_vel_timeout_sec=0.1, max_linear_velocity=0.5),
    )
    ctrl.command_twist(0.2, 0.0, now=1000.0)
    assert driver.health().left_velocity != 0.0
    ctrl.tick(now=1000.05)
    assert driver.health().left_velocity != 0.0
    ctrl.tick(now=1000.25)
    assert driver.health().left_velocity == 0.0
    assert driver.health().right_velocity == 0.0


def test_estop_blocks_motion():
    driver = MockMotorDriver()
    ctrl = DiffDriveController(driver, ControllerConfig())
    ctrl.trigger_estop()
    left, right = ctrl.command_twist(0.2, 0.0)
    assert left == 0.0 and right == 0.0
    assert driver.health().left_velocity == 0.0
    ctrl.clear_estop()
    left, right = ctrl.command_twist(0.2, 0.0)
    assert left > 0.0
