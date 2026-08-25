"""WebRTC APM front end (Stage F2). 16 kHz mono, 10 ms frames.

Import is safe without F2 packages. Constructing AudioPreprocessor requires
`numpy` and `pywebrtc-audio` in the venv.
"""

from __future__ import annotations

from dataclasses import dataclass

from jetbot_agent._stage import StageNotReady

FRAME_MS = 10
SAMPLE_RATE = 16000
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 160


@dataclass
class ApmConfig:
    sample_rate: int = SAMPLE_RATE
    echo_cancellation: bool = True
    noise_suppression: bool = True
    auto_gain_control: bool = True
    high_pass_filter: bool = True
    ns_level: int = 2
    stream_delay_ms: int = 0


class AudioPreprocessor:
    def __init__(self, config: ApmConfig | None = None):
        try:
            import numpy as np
            from pywebrtc_audio import AudioProcessor
        except ImportError as exc:
            raise StageNotReady(
                "audio.audio_preprocessor needs numpy + pywebrtc-audio (Stage F2)"
            ) from exc

        self._np = np
        self.config = config or ApmConfig()
        kwargs = dict(
            sample_rate=self.config.sample_rate,
            echo_cancellation=self.config.echo_cancellation,
            noise_suppression=self.config.noise_suppression,
            auto_gain_control=self.config.auto_gain_control,
            high_pass_filter=self.config.high_pass_filter,
        )
        try:
            self._ap = AudioProcessor(
                **kwargs,
                ns_level=self.config.ns_level,
                stream_delay_ms=self.config.stream_delay_ms,
            )
        except TypeError:
            self._ap = AudioProcessor(**kwargs)

    def process_int16(self, near, far=None):
        """Process contiguous int16 mono. Pads to whole 10 ms frames."""
        np = self._np
        near = np.asarray(near, dtype=np.int16).reshape(-1)
        if far is None:
            far = np.zeros_like(near)
        else:
            far = np.asarray(far, dtype=np.int16).reshape(-1)
            if far.shape[0] < near.shape[0]:
                far = np.pad(far, (0, near.shape[0] - far.shape[0]))
            elif far.shape[0] > near.shape[0]:
                far = far[: near.shape[0]]
        pad = (-near.shape[0]) % FRAME_SAMPLES
        if pad:
            near = np.pad(near, (0, pad))
            far = np.pad(far, (0, pad))
        out = np.empty_like(near)
        for i in range(0, near.shape[0], FRAME_SAMPLES):
            n = near[i : i + FRAME_SAMPLES]
            f = far[i : i + FRAME_SAMPLES]
            y = self._ap.process(n, f)
            out[i : i + FRAME_SAMPLES] = np.asarray(y, dtype=np.int16).reshape(-1)[:FRAME_SAMPLES]
        if pad:
            out = out[:-pad]
        return out

    @property
    def speech_probability(self):
        return getattr(self._ap, "speech_probability", None)
