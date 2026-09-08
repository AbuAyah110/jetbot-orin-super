"""CPU-only Piper VITS TTS adapter backed by sherpa-onnx."""

from __future__ import annotations

from pathlib import Path

from jetbot_agent._stage import StageNotReady


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "models"
    / "piper"
    / "vits-piper-en_US-lessac-low-int8"
)


class PiperTTS:
    """Synthesize speech with the int8 US English Lessac Piper voice."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, num_threads: int = 2):
        self.model_dir = Path(model_dir)
        model = self.model_dir / "en_US-lessac-low.onnx"
        tokens = self.model_dir / "tokens.txt"
        data_dir = self.model_dir / "espeak-ng-data"
        missing = [
            str(path)
            for path in (model, tokens, data_dir)
            if not path.exists()
        ]
        if missing:
            raise StageNotReady(
                "Piper TTS model is missing; run "
                "scripts/bringup/fetch_voice_models.sh. Missing: "
                + ", ".join(missing)
            )

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise StageNotReady(
                "sherpa-onnx==1.13.6 is required for Piper TTS"
            ) from exc

        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model),
                    tokens=str(tokens),
                    data_dir=str(data_dir),
                ),
                num_threads=num_threads,
                provider="cpu",
            ),
            max_num_sentences=1,
        )
        if not config.validate():
            raise StageNotReady(f"invalid Piper TTS model at {self.model_dir}")
        self._tts = sherpa_onnx.OfflineTts(config)

    @property
    def sample_rate(self) -> int:
        return self._tts.sample_rate

    def synthesize(self, text: str, speed: float = 1.0):
        """Return sherpa-onnx generated audio (samples and sample_rate)."""
        return self._tts.generate(text, sid=0, speed=speed)
