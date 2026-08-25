"""NVIDIA FastPitch + HiFi-GAN stub. Stage F5."""

from jetbot_agent._stage import StageNotReady


class FastPitchHiFiGANTTS:
    def __init__(self, *args, **kwargs):
        raise StageNotReady("audio.fastpitch_hifigan_tts waits for Stage F5")
