from pathlib import Path

import pytest

from jetbot_agent._stage import StageNotReady
from jetbot_agent.audio.piper_tts import PiperTTS
from jetbot_agent.audio.zipformer_asr import ZipformerASR


def test_zipformer_import_safe_without_models(tmp_path: Path) -> None:
    with pytest.raises(StageNotReady, match="fetch_voice_models"):
        ZipformerASR(model_dir=tmp_path)


def test_piper_import_safe_without_models(tmp_path: Path) -> None:
    with pytest.raises(StageNotReady, match="fetch_voice_models"):
        PiperTTS(model_dir=tmp_path)
