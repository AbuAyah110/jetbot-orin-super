"""Nemotron embedder stub. Stage G."""

from jetbot_agent._stage import StageNotReady


class TrtEmbedder:
    def __init__(self, *args, **kwargs):
        raise StageNotReady("engine.trt_embedder waits for Stage G")
