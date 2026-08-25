"""NVIDIA FastConformer ASR stub. Stage F4."""

from jetbot_agent._stage import StageNotReady


class FastConformerASR:
    def __init__(self, *args, **kwargs):
        raise StageNotReady("audio.fastconformer_asr waits for Stage F4")
