# Modified by SparkFun Electronics June 2021
# Adapted for Jetson Orin Nano / Orin Nano Super
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
import atexit
from Adafruit_MotorHAT import Adafruit_MotorHAT
import traitlets
from traitlets.config.configurable import Configurable


class Motor(Configurable):

    value = traitlets.Float()
    alpha = traitlets.Float(default_value=1.0).tag(config=True)
    beta = traitlets.Float(default_value=0.0).tag(config=True)

    def __init__(self, driver, channel, backend='adafruit', *args, **kwargs):
        super(Motor, self).__init__(*args, **kwargs)
        self._driver = driver
        self._backend = backend
        self.channel = channel

        if backend == 'adafruit':
            self._motor = self._driver.getMotor(channel)
            if channel == 1:
                self._ina = 1
                self._inb = 0
            else:
                self._ina = 2
                self._inb = 3
        atexit.register(self._release)

    @traitlets.observe('value')
    def _observe_value(self, change):
        self._write_value(change['new'])

    def _write_value(self, value):
        """Sets motor value between [-1, 1]."""
        if self._backend == 'sparkfun':
            speed = int(255 * (self.alpha * value + self.beta))
            # Motor Number: A = 0, B = 1; Direction handled by signed speed in set_drive
            self._driver.set_drive(self.channel - 1, 0, speed)
            self._driver.enable()
            return

        mapped_value = int(255.0 * (self.alpha * value + self.beta))
        speed = min(max(abs(mapped_value), 0), 255)
        self._motor.setSpeed(speed)
        if mapped_value < 0:
            self._motor.run(Adafruit_MotorHAT.FORWARD)
            # Waveshare JetBot board PWM lines
            self._driver._pwm.setPWM(self._ina, 0, 0)
            self._driver._pwm.setPWM(self._inb, 0, speed * 16)
        else:
            self._motor.run(Adafruit_MotorHAT.BACKWARD)
            self._driver._pwm.setPWM(self._ina, 0, speed * 16)
            self._driver._pwm.setPWM(self._inb, 0, 0)

    def _release(self):
        """Stops motor by releasing control."""
        if self._backend == 'sparkfun':
            self._driver.disable()
            return
        self._motor.run(Adafruit_MotorHAT.RELEASE)
        self._driver._pwm.setPWM(self._ina, 0, 0)
        self._driver._pwm.setPWM(self._inb, 0, 0)
