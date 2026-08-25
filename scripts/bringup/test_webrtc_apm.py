#!/usr/bin/env python3
"""F2: offline WebRTC APM on synthetic 16 kHz fixtures. No live duplex / no speaker."""

from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jetbot_agent.audio.audio_preprocessor import (  # noqa: E402
    FRAME_SAMPLES,
    SAMPLE_RATE,
    ApmConfig,
    AudioPreprocessor,
)

OUT = ROOT / "data" / "audio" / "f2"
LOG = ROOT / "data" / "bringup" / "f2_webrtc_apm.json"
OUT.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)


def write_wav(path: Path, samples: np.ndarray) -> None:
    samples = np.asarray(samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(samples.tobytes())


def rms(x: np.ndarray) -> float:
    x = x.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x.astype(np.int32))) / 32768.0)


def synth_speech(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    f0 = 140.0
    sig = (
        0.45 * np.sin(2 * np.pi * f0 * t)
        + 0.22 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.10 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 3.5 * t)
    bursts = (np.sin(2 * np.pi * 1.2 * t) > -0.2).astype(np.float64)
    sig = sig * env * bursts
    sig += 0.01 * rng.normal(size=n)
    sig = np.clip(sig, -0.9, 0.9)
    return (sig * 32767).astype(np.int16)


def run() -> int:
    seconds = 2.0
    n = int(seconds * SAMPLE_RATE)
    clean = synth_speech(seconds, seed=1)
    noise_only = (0.20 * np.random.default_rng(2).normal(size=n) * 32767).astype(np.int16)
    noisy_speech = np.clip(clean.astype(np.int32) + noise_only.astype(np.int32), -32768, 32767).astype(
        np.int16
    )
    t = np.arange(n) / SAMPLE_RATE
    far = (0.50 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    delay = int(0.04 * SAMPLE_RATE)
    echo_only = np.zeros(n, dtype=np.int16)
    echo_only[delay:] = (0.40 * far[:-delay]).astype(np.int16)

    write_wav(OUT / "clean.wav", clean)
    write_wav(OUT / "noise_only.wav", noise_only)
    write_wav(OUT / "noisy_speech.wav", noisy_speech)
    write_wav(OUT / "far_end.wav", far)
    write_wav(OUT / "echo_only.wav", echo_only)

    ns_cfg = ApmConfig(
        echo_cancellation=False,
        noise_suppression=True,
        auto_gain_control=False,
        high_pass_filter=True,
        ns_level=3,
    )
    aec_cfg = ApmConfig(
        echo_cancellation=True,
        noise_suppression=False,
        auto_gain_control=False,
        high_pass_filter=True,
    )
    full_cfg = ApmConfig()

    t0 = time.perf_counter()
    noise_out = AudioPreprocessor(ns_cfg).process_int16(noise_only)
    ns_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    echo_out = AudioPreprocessor(aec_cfg).process_int16(echo_only, far)
    aec_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    clean_out = AudioPreprocessor(full_cfg).process_int16(clean)
    clean_s = time.perf_counter() - t2

    noisy_out = AudioPreprocessor(full_cfg).process_int16(noisy_speech)

    write_wav(OUT / "noise_only_out.wav", noise_out)
    write_wav(OUT / "echo_only_out.wav", echo_out)
    write_wav(OUT / "clean_out.wav", clean_out)
    write_wav(OUT / "noisy_speech_out.wav", noisy_out)

    ns_ratio = rms(noise_only) / max(rms(noise_out), 1e-9)
    aec_ratio = rms(echo_only) / max(rms(echo_out), 1e-9)
    report = {
        "backend": "pywebrtc-audio AudioProcessor",
        "sample_rate": SAMPLE_RATE,
        "frame_samples": FRAME_SAMPLES,
        "duration_s": seconds,
        "ns_noise_only_in_rms": rms(noise_only),
        "ns_noise_only_out_rms": rms(noise_out),
        "noise_reduction_ratio": ns_ratio,
        "aec_echo_only_in_rms": rms(echo_only),
        "aec_echo_only_out_rms": rms(echo_out),
        "echo_reduction_ratio": aec_ratio,
        "clean_in_rms": rms(clean),
        "clean_out_rms": rms(clean_out),
        "clean_out_peak": peak(clean_out),
        "speech_probability_last": AudioPreprocessor(full_cfg).speech_probability,
        "latency_s": {"ns_noise_only": ns_s, "aec_echo_only": aec_s, "full_clean": clean_s},
        "rtf": {
            "ns_noise_only": ns_s / seconds,
            "aec_echo_only": aec_s / seconds,
            "full_clean": clean_s / seconds,
        },
        "live_duplex": False,
        "notes": "offline synthetic fixtures; AGC off for NS/AEC ratios; no ALSA playback",
    }
    ap_sp = AudioPreprocessor(full_cfg)
    ap_sp.process_int16(clean)
    report["speech_probability_last"] = ap_sp.speech_probability
    payload = json.dumps(report, indent=2)
    (OUT / "report.json").write_text(payload, encoding="utf-8")
    LOG.write_text(payload, encoding="utf-8")
    print(payload)

    if report["clean_out_rms"] < 0.02:
        raise SystemExit("FAIL: clean speech was destroyed")
    if report["clean_out_peak"] >= 0.99:
        raise SystemExit("FAIL: clean output clipped")
    if report["noise_reduction_ratio"] < 1.5:
        raise SystemExit("FAIL: noise not reduced")
    if report["echo_reduction_ratio"] < 2.0:
        raise SystemExit("FAIL: echo not reduced")
    print("F2 WebRTC APM ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
