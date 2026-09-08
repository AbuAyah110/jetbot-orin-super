#!/usr/bin/env python3
"""One-process Zipformer ASR + Piper VITS TTS bring-up gate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jetbot_agent.audio.piper_tts import DEFAULT_MODEL_DIR as PIPER_DIR
from jetbot_agent.audio.piper_tts import PiperTTS
from jetbot_agent.audio.zipformer_asr import DEFAULT_MODEL_DIR as ZIPFORMER_DIR
from jetbot_agent.audio.zipformer_asr import ZipformerASR

FIXTURE = ZIPFORMER_DIR / "test_wavs" / "0.wav"
REFERENCE = (
    "AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE "
    "THE SQUALID QUARTER OF THE BROTHELS"
)
TTS_TEXT = "Testing one two three."
OUTPUT_WAV = ROOT / "data" / "audio" / "zipformer_piper" / "hello_world.wav"
REPORT = ROOT / "data" / "bringup" / "zipformer_piper.json"


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise SystemExit(f"{path}: expected mono 16-bit PCM")
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, rate


def write_wav(path: Path, samples, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes((pcm * 32767).astype(np.int16).tobytes())


def words(text: str) -> list[str]:
    return re.sub(r"[^A-Z' ]+", " ", text.upper()).split()


def error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = words(reference), words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, expected in enumerate(ref, 1):
        current = [i] + [0] * len(hyp)
        for j, actual in enumerate(hyp, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (expected != actual),
            )
        previous = current
    return previous[-1] / len(ref)


def peak_rss() -> tuple[int, float, float]:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            kib = int(line.split()[1])
            return kib, kib / 1024.0, kib * 1024 / 1_000_000
    raise RuntimeError("VmHWM is unavailable")


def model_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def live_capture(path: Path, seconds: int) -> Path:
    if not Path("/dev/snd").exists():
        raise SystemExit("--live-capture requested but /dev/snd is unavailable")
    from jetbot_agent.hardware.audio_interface import resolve_sss1629

    endpoint = resolve_sss1629()["alsa_capture"]
    if not endpoint.startswith("plughw:CARD="):
        raise SystemExit(f"refusing non-name-resolved capture endpoint: {endpoint}")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "arecord",
            "-D",
            endpoint,
            "-f",
            "S16_LE",
            "-r",
            "16000",
            "-c",
            "1",
            "-d",
            str(seconds),
            str(path),
        ],
        check=True,
        timeout=seconds + 10,
    )
    return path


def run(args: argparse.Namespace) -> int:
    live_tested = False

    load_start = time.perf_counter()
    asr = ZipformerASR(num_threads=args.threads)
    tts = PiperTTS(num_threads=args.threads)
    load_seconds = time.perf_counter() - load_start

    tts_start = time.perf_counter()
    generated = tts.synthesize(TTS_TEXT)
    tts_seconds = time.perf_counter() - tts_start
    write_wav(OUTPUT_WAV, generated.samples, generated.sample_rate)

    if args.live_capture:
        wav = live_capture(
            ROOT / "data" / "audio" / "zipformer_piper" / "live_capture.wav",
            args.capture_seconds,
        )
        samples, rate = read_wav(wav)
        reference = ""
        live_tested = True
    elif args.wav:
        wav = Path(args.wav)
        samples, rate = read_wav(wav)
        reference = REFERENCE if wav.resolve() == FIXTURE.resolve() else ""
    else:
        # The generated phrase is the smallest reproducible offline fixture and
        # exercises both resident C++ objects without a long-ASR activation peak.
        wav = OUTPUT_WAV
        samples = generated.samples
        rate = generated.sample_rate
        reference = TTS_TEXT

    asr_start = time.perf_counter()
    transcript = asr.transcribe(samples, rate)
    asr_seconds = time.perf_counter() - asr_start
    rss_kib, rss_mib, rss_mb = peak_rss()
    output_seconds = len(generated.samples) / generated.sample_rate

    report = {
        "runtime": "sherpa-onnx 1.13.6, one Python process wrapping one C++ runtime",
        "provider": "cpu",
        "gpu_vram_mib": 0,
        "threads": args.threads,
        "models": {
            "asr": ZIPFORMER_DIR.name,
            "asr_precision": "int8",
            "asr_extracted_bytes": model_bytes(ZIPFORMER_DIR),
            "tts": PIPER_DIR.name,
            "tts_precision": "int8",
            "tts_extracted_bytes": model_bytes(PIPER_DIR),
        },
        "load_seconds": load_seconds,
        "asr": {
            "wav": str(wav.relative_to(ROOT)),
            "audio_seconds": len(samples) / rate,
            "inference_seconds": asr_seconds,
            "rtf": asr_seconds / (len(samples) / rate),
            "reference": reference or None,
            "transcript": transcript,
            "wer": error_rate(reference, transcript) if reference else None,
        },
        "tts": {
            "text": TTS_TEXT,
            "wav": str(OUTPUT_WAV.relative_to(ROOT)),
            "sample_rate": generated.sample_rate,
            "audio_seconds": output_seconds,
            "inference_seconds": tts_seconds,
            "rtf": tts_seconds / output_seconds,
            "round_trip_transcript": transcript if wav == OUTPUT_WAV else None,
            "round_trip_wer": (
                error_rate(TTS_TEXT, transcript) if wav == OUTPUT_WAV else None
            ),
        },
        "peak_rss": {
            "kib": rss_kib,
            "mib": rss_mib,
            "decimal_mb": rss_mb,
            "under_200_mib": rss_mib < 200,
            "under_200_decimal_mb": rss_mb < 200,
        },
        "live_capture_tested": live_tested,
        "live_playback_tested": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if not transcript.strip():
        raise SystemExit("FAIL: ASR transcript is empty")
    if reference and report["asr"]["wer"] > 0.15:
        raise SystemExit(f"FAIL: fixture WER {report['asr']['wer']:.3f} exceeds 0.15")
    if not generated.samples:
        raise SystemExit("FAIL: Piper output is empty")
    print(
        f"Zipformer+Piper ok: ASR WER {report['asr']['wer']}, "
        f"transcript {transcript!r}, peak {rss_mib:.1f} MiB "
        f"({rss_mb:.1f} decimal MB)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--live-capture", action="store_true")
    parser.add_argument("--capture-seconds", type=int, default=5)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
