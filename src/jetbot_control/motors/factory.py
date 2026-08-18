from __future__ import annotations

from typing import Any, Mapping

from .base import MotorDriver
from .mock import MockMotorDriver


def create_motor_driver(backend: str = 'mock', **kwargs: Any) -> MotorDriver:
    name = (backend or 'mock').strip().lower()
    if name == 'mock':
        return MockMotorDriver()
    if name in ('jetbot_i2c', 'jetbot', 'i2c'):
        from .jetbot_i2c import JetbotI2CMotorDriver

        return JetbotI2CMotorDriver(
            i2c_bus=int(kwargs.get('i2c_bus', 1)),
            i2c_address=int(kwargs.get('i2c_address', 112)),
        )
    raise ValueError('Unknown motor backend: {0}'.format(backend))


def create_motor_driver_from_config(config: Mapping[str, Any]) -> MotorDriver:
    control = config.get('control', {}) if isinstance(config, Mapping) else {}
    return create_motor_driver(
        backend=control.get('backend', 'mock'),
        i2c_bus=control.get('i2c_bus', 1),
        i2c_address=control.get('i2c_address', 112),
    )
