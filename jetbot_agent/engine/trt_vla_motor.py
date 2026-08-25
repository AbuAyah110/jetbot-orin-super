"""smolvla-jetbot TensorRT stub. Stage G. Dummy I/O only; no PWM."""

from jetbot_agent._stage import StageNotReady


class TrtVlaMotor:
    def __init__(self, *args, **kwargs):
        raise StageNotReady("engine.trt_vla_motor waits for Stage G")
