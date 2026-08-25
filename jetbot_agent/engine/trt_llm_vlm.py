"""Qwen2.5-VL TensorRT-Edge-LLM stub. Stage G / I7."""

from jetbot_agent._stage import StageNotReady


class TrtLlmVlm:
    def __init__(self, *args, **kwargs):
        raise StageNotReady("engine.trt_llm_vlm waits for Stage G")
