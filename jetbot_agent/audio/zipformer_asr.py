"""CPU-only Zipformer ASR adapter backed by sherpa-onnx."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from jetbot_agent._stage import StageNotReady


DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "models"
    / "zipformer"
    / "sherpa-onnx-zipformer-small-en-2023-06-26"
)


class ZipformerASR:
    """Offline int8 Zipformer transducer running on the CPU."""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, num_threads: int = 2):
        self.model_dir = Path(model_dir)
        files = {
            "encoder": self.model_dir / "encoder-epoch-99-avg-1.int8.onnx",
            "decoder": self.model_dir / "decoder-epoch-99-avg-1.int8.onnx",
            "joiner": self.model_dir / "joiner-epoch-99-avg-1.int8.onnx",
            "tokens": self.model_dir / "tokens.txt",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise StageNotReady(
                "Zipformer ASR model is missing; run "
                "scripts/bringup/fetch_voice_models.sh. Missing: "
                + ", ".join(missing)
            )

        try:
            import sherpa_onnx
        except ImportError as exc:
            raise StageNotReady(
                "sherpa-onnx==1.13.6 is required for Zipformer ASR"
            ) from exc

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            joiner=str(files["joiner"]),
            tokens=str(files["tokens"]),
            num_threads=num_threads,
            decoding_method="greedy_search",
            provider="cpu",
        )

    def transcribe(self, samples: Iterable[float], sample_rate: int = 16000) -> str:
        """Transcribe mono floating-point samples in the range [-1, 1]."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text
