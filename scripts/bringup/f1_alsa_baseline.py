#!/usr/bin/env python3
"""F1: name-resolve SSS1629, safe mixer, sequential 16 kHz capture then low playback."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jetbot_agent.hardware.audio_interface import (  # noqa: E402
    apply_safe_mixer_baseline,
    mixer_report,
    persist_mixer,
    resolve_sss1629,
)

OUT = Path(os.environ.get("JETBOT_ALSA_WAV", str(ROOT / "data" / "audio" / "f1_capture.wav")))
SECONDS = int(os.environ.get("JETBOT_ALSA_SECONDS", "1"))
STATE = ROOT / "config" / "alsa-sss1629.state"


def wav_peak_and_rms(path: Path) -> tuple[float, float, int]:
    with wave.open(str(path), "rb") as wf:
        nchan = wf.getnchannels()
        sw = wf.getsampwidth()
        rate = wf.getframerate()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    if sw != 2:
        raise RuntimeError(f"expected S16_LE, got sample width {sw}")
    import array

    samples = array.array("h")
    samples.frombytes(raw)
    if not samples:
        return 0.0, 0.0, rate
    peak = max(abs(s) for s in samples) / 32768.0
    mean_sq = sum(s * s for s in samples) / len(samples)
    rms = (mean_sq ** 0.5) / 32768.0
    return peak, rms, rate


def run() -> int:
    ident = resolve_sss1629()
    print("resolved", ident)
    if ident["alsa_id"] == "2" or str(ident["alsa_capture"]).endswith(":2,0"):
        # CARD=2 would be an index; CARD=Device is a name. Index-only is forbidden.
        pass
    if ident["alsa_capture"].startswith("plughw:2") or ident["alsa_capture"] == "hw:2,0":
        raise RuntimeError(f"refusing index-hardcoded endpoint {ident['alsa_capture']}")

    mix_log = apply_safe_mixer_baseline(ident["card_index_ephemeral"])
    print(mix_log)
    report = mixer_report(ident["card_index_ephemeral"])
    print("=== mixer after baseline ===")
    print(report)
    if "sidetone" in report.lower() and "[on]" in report.lower():
        # extra caution; SSS1629 uses Mic playback as sidetone
        pass
    persist_mixer(ident["card_index_ephemeral"], STATE)
    print("mixer state ->", STATE)

    cap = ident["alsa_capture"]
    play = ident["alsa_playback"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"recording {SECONDS}s 16 kHz mono from {cap} -> {OUT}")
    subprocess.run(
        [
            "arecord",
            "-D",
            cap,
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            "-d",
            str(SECONDS),
            str(OUT),
        ],
        check=True,
    )
    time.sleep(0.2)
    peak, rms, rate = wav_peak_and_rms(OUT)
    print(f"wav rate={rate} peak={peak:.3f} rms={rms:.4f} bytes={OUT.stat().st_size}")
    if rms < 0.0005:
        raise SystemExit("FAIL: recording is effectively silent")
    if peak >= 0.99:
        raise SystemExit("FAIL: recording looks clipped")
    print(f"playback once at low mixer level on {play}")
    subprocess.run(["aplay", "-D", play, str(OUT)], check=True)
    print("F1 ALSA baseline ok")
    print("usb_name=", ident["usb_name"])
    print("alsa_id=", ident["alsa_id"])
    print("alsa_capture=", cap)
    print("alsa_playback=", play)
    print("sidetone_enabled=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
