"""PCA9685 I2C motor driver stub. LLM never PWM. Live path: scripts/bringup/test_motors.py."""

from jetbot_agent._stage import StageNotReady


class MotorController:
    def __init__(self, *args, **kwargs):
        raise StageNotReady(
            "hardware.motor_controller waits for Stage B wheels-up sign-off; LLM never PWM"
        )
