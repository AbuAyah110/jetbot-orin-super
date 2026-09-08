# Modified by SparkFun Electronics June 2021
# Adapted for Jetson Orin Nano / Orin Nano Super (I2C bus 1, motor addr 0x70)
#
#==================================================================================
# Copyright (c) 2021 SparkFun Electronics
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#==================================================================================

import traitlets
from traitlets.config.configurable import SingletonConfigurable
import qwiic
from Adafruit_MotorHAT import Adafruit_MotorHAT
from .motor import Motor

# Adafruit Motor HAT / PCA9685 addresses seen on JetBot boards:
#   96  (0x60) — original Jetson Nano JetBot default
#   112 (0x70) — common Orin / custom expansion board address
# SparkFun Serial Controlled Motor Driver:
#   93  (0x5D)
ADAFRUIT_MOTOR_ADDRS = (112, 96)
SPARKFUN_MOTOR_ADDR = 93


def _scan_addresses():
    try:
        return set(qwiic.scan() or [])
    except Exception:
        return set()


def detect_motor_backend(addresses=None):
    """Return ('adafruit', addr) or ('sparkfun', addr) or (None, None)."""
    addresses = set(addresses if addresses is not None else _scan_addresses())
    for addr in ADAFRUIT_MOTOR_ADDRS:
        if addr in addresses:
            return 'adafruit', addr
    if SPARKFUN_MOTOR_ADDR in addresses:
        return 'sparkfun', SPARKFUN_MOTOR_ADDR
    return None, None


class Robot(SingletonConfigurable):

    left_motor = traitlets.Instance(Motor)
    right_motor = traitlets.Instance(Motor)

    # JetPack 6.2 40-pin SDA/SCL (pins 3/5) is /dev/i2c-7 on this Orin Nano Super.
    # Onboard INA3221/FUSB301 stay on bus 1.
    i2c_bus = traitlets.Integer(default_value=7).tag(config=True)
    # Prefer Orin expansion address 112 (0x70); falls back to 96 (0x60) if present.
    i2c_address = traitlets.Integer(default_value=112).tag(config=True)
    left_motor_channel = traitlets.Integer(default_value=1).tag(config=True)
    left_motor_alpha = traitlets.Float(default_value=1.0).tag(config=True)
    right_motor_channel = traitlets.Integer(default_value=2).tag(config=True)
    # This chassis: +value on the right motor was driving backward. Invert.
    right_motor_alpha = traitlets.Float(default_value=-1.0).tag(config=True)

    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)

        addresses = _scan_addresses()
        backend, detected_addr = detect_motor_backend(addresses)

        if backend is None and self.i2c_address in ADAFRUIT_MOTOR_ADDRS:
            # Allow explicit config even if scan missed the device.
            backend, detected_addr = 'adafruit', self.i2c_address

        if backend == 'adafruit':
            if self.i2c_address in addresses:
                addr = self.i2c_address
            elif detected_addr is not None:
                addr = detected_addr
            else:
                addr = self.i2c_address
            self.motor_driver = Adafruit_MotorHAT(addr=addr, i2c_bus=self.i2c_bus)
            self.left_motor = Motor(
                self.motor_driver,
                channel=self.left_motor_channel,
                alpha=self.left_motor_alpha,
                backend='adafruit',
            )
            self.right_motor = Motor(
                self.motor_driver,
                channel=self.right_motor_channel,
                alpha=self.right_motor_alpha,
                backend='adafruit',
            )
            self._backend = 'adafruit'
        elif backend == 'sparkfun':
            self.motor_driver = qwiic.QwiicScmd()
            self.left_motor = Motor(
                self.motor_driver,
                channel=self.left_motor_channel,
                alpha=self.left_motor_alpha,
                backend='sparkfun',
            )
            self.right_motor = Motor(
                self.motor_driver,
                channel=self.right_motor_channel,
                alpha=self.right_motor_alpha,
                backend='sparkfun',
            )
            self.motor_driver.enable()
            self._backend = 'sparkfun'
        else:
            raise RuntimeError(
                'No motor driver found on I2C. Scanned addresses: {0}. '
                'Expected Adafruit Motor HAT at 112 (0x70) or 96 (0x60), '
                'or SparkFun SCMD at 93 (0x5D). '
                'On Orin Nano, confirm I2C bus 1 and wiring.'.format(sorted(addresses))
            )

    def set_motors(self, left_speed, right_speed):
        self.left_motor.value = left_speed
        self.right_motor.value = right_speed
        if self._backend == 'sparkfun':
            self.motor_driver.enable()

    def forward(self, speed=1.0, duration=None):
        if self._backend == 'sparkfun':
            speed_i = int(speed * 255)
            self.motor_driver.set_drive(0, 0, speed_i)
            self.motor_driver.set_drive(1, 0, speed_i)
            self.motor_driver.enable()
        else:
            self.left_motor.value = speed
            self.right_motor.value = speed

    def backward(self, speed=1.0):
        if self._backend == 'sparkfun':
            speed_i = int(speed * 255)
            self.motor_driver.set_drive(0, 1, speed_i)
            self.motor_driver.set_drive(1, 1, speed_i)
            self.motor_driver.enable()
        else:
            self.left_motor.value = -speed
            self.right_motor.value = -speed

    def left(self, speed=1.0):
        if self._backend == 'sparkfun':
            speed_i = int(speed * 255)
            self.motor_driver.set_drive(0, 1, speed_i)
            self.motor_driver.set_drive(1, 0, speed_i)
            self.motor_driver.enable()
        else:
            self.left_motor.value = -speed
            self.right_motor.value = speed

    def right(self, speed=1.0):
        if self._backend == 'sparkfun':
            speed_i = int(speed * 255)
            self.motor_driver.set_drive(0, 0, speed_i)
            self.motor_driver.set_drive(1, 1, speed_i)
            self.motor_driver.enable()
        else:
            self.left_motor.value = speed
            self.right_motor.value = -speed

    def stop(self):
        if self._backend == 'sparkfun':
            self.motor_driver.set_drive(0, 0, 0)
            self.motor_driver.set_drive(1, 1, 0)
            self.motor_driver.disable()
        else:
            self.left_motor.value = 0
            self.right_motor.value = 0
