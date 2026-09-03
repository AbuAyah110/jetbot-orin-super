#!/usr/bin/env python3
"""F3: optional RNNoise residual denoising, A/B against the F2 WebRTC APM alone.

Offline gate. No ALSA, no speaker, no live duplex.

The A/B reuses the exact F2 fixtures plus the real LibriSpeech utterances that F4
already measures, so the signal metrics and the downstream ASR numbers come from
the same audio as the two gates this compares.

RNNoise is fixed at 48 kHz / 480-sample frames, so every RNNoise config here is
really `16k -> soxr -> 48k -> RNNoise -> soxr -> 16k`. That resampler is part of
the cost being measured, not an implementation detail to hide.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
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

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 160
RNNOISE_RATE = 48000
RESAMPLE_QUALITY = "HQ"
ASR_THREADS = 2  # F4's recommended production configuration

F2_AUDIO = ROOT / "data" / "audio" / "f2"
OUT = ROOT / "data" / "audio" / "f3"
LOG = ROOT / "data" / "bringup" / "f3_rnnoise_ab.json"
ASR_MODEL = (
    ROOT / "data" / "models" / "f4" / "sherpa-onnx-nemo-fast-conformer-ctc-en-24500-int8"
)
LIBRI = (
    ROOT
    / "data"
    / "models"
    / "f4"
    / "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms-int8"
    / "test_wavs"
)
# F2's implementation lives here: the working-tree copy of
# jetbot_agent/audio/audio_preprocessor.py is currently a StageNotReady stub from
# an unrelated in-flight edit, and F4 preserved the functional HEAD module at
# this path. Prefer the package module whenever it works again.
APM_HEAD = ROOT / "data" / "models" / "f4" / "_apm_head" / "audio_preprocessor.py"


# --------------------------------------------------------------------------
# wav + metric helpers (RMS/peak match scripts/bringup/test_webrtc_apm.py)
# --------------------------------------------------------------------------


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise SystemExit(f"{path}: need mono 16-bit PCM")
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).copy(), rate


def write_wav(path: Path, samples: np.ndarray, rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


def rms(x: np.ndarray) -> float:
    x = x.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def peak(x: np.ndarray) -> float:
    return float(np.max(np.abs(x.astype(np.int32))) / 32768.0)


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), -32768, 32767).astype(np.int16)


def best_lag(ref: np.ndarray, test: np.ndarray, max_lag: int = 2048) -> int:
    """Integer sample delay of `test` behind `ref`, by FFT cross-correlation."""
    a = ref.astype(np.float64)
    b = test.astype(np.float64)
    n = 1 << int(np.ceil(np.log2(len(a) + len(b) + 1)))
    cc = np.fft.irfft(np.conj(np.fft.rfft(a, n)) * np.fft.rfft(b, n), n)
    return int(np.argmax(cc[: max_lag + 1]))


def align_gain(ref: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Undo processing delay and any AGC/denoiser gain before measuring distortion."""
    lag = best_lag(ref, test)
    t = test[lag:] if lag else test
    n = min(len(ref), len(t))
    r = ref[:n].astype(np.float64)
    t = t[:n].astype(np.float64)
    denom = float(np.dot(r, r))
    gain = float(np.dot(r, t) / denom) if denom > 0 else 0.0
    return r, t, lag, gain


def snr_db(ref: np.ndarray, test: np.ndarray) -> tuple[float, float, int, float]:
    """Global and segmental SNR of `test` as an estimate of `ref`, gain/delay normalised.

    Segmental SNR is averaged over active frames only (20 ms frames whose RMS is
    within 40 dB of the loudest frame) and clamped to [-10, 35] dB, which is the
    usual convention -- silent frames otherwise dominate the average.
    """
    r, t, lag, gain = align_gain(ref, test)
    if gain <= 0:
        return float("-inf"), float("-inf"), lag, gain
    t = t / gain
    err = r - t
    total = float(np.sum(r * r))
    global_snr = 10.0 * np.log10(total / max(float(np.sum(err * err)), 1e-12)) if total > 0 else float("nan")

    frame = 2 * FRAME_SAMPLES  # 20 ms
    nf = len(r) // frame
    if nf == 0:
        return global_snr, float("nan"), lag, gain
    rf = r[: nf * frame].reshape(nf, frame)
    ef = err[: nf * frame].reshape(nf, frame)
    re = np.sum(rf * rf, axis=1)
    active = re > (np.max(re) * 1e-4)
    if not np.any(active):
        return global_snr, float("nan"), lag, gain
    seg = 10.0 * np.log10(re[active] / np.maximum(np.sum(ef * ef, axis=1)[active], 1e-12))
    return global_snr, float(np.mean(np.clip(seg, -10.0, 35.0))), lag, gain


def norm_text(s: str) -> list[str]:
    return re.sub(r"[^A-Z' ]+", " ", s.upper()).split()


def wer(ref: str, hyp: str) -> float:
    r, h = norm_text(ref), norm_text(hyp)
    if not r:
        return float("nan")
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return prev[len(h)] / len(r)


def peak_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


# --------------------------------------------------------------------------
# RNNoise: the pyrnnoise wheel's bundled librnnoise.so, driven by ctypes
# --------------------------------------------------------------------------


def load_file_module(name: str, path: Path):
    """Import a single .py file under a private name.

    Registered in sys.modules before execution because `@dataclass` resolves
    annotations through `sys.modules[cls.__module__]`.
    """
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def load_rnnoise_shim():
    """Load pyrnnoise's low-level ctypes module without its heavy package __init__.

    `pyrnnoise/__init__.py` pulls in audiolab (PyAV) + matplotlib, which we
    deliberately did not install. `pyrnnoise/rnnoise.py` needs only numpy and
    ctypes, so load that file directly.
    """
    spec = importlib.util.find_spec("pyrnnoise")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("pyrnnoise is not installed in this interpreter")
    return load_file_module(
        "_f3_rnnoise", Path(list(spec.submodule_search_locations)[0]) / "rnnoise.py"
    )


class RNNoise:
    """Whole-buffer 16 kHz -> 48 kHz -> RNNoise -> 16 kHz denoiser."""

    def __init__(self, quality: str = RESAMPLE_QUALITY):
        import soxr

        self._soxr = soxr
        self._m = load_rnnoise_shim()
        self.frame_size = int(self._m.FRAME_SIZE)
        self.quality = quality
        self.speech_probs: list[float] = []

    def denoise_48k(self, up: np.ndarray) -> np.ndarray:
        m = self._m
        n = len(up)
        pad = (-n) % self.frame_size
        buf = np.ascontiguousarray(np.pad(up, (0, pad)) if pad else up, dtype=np.float32)
        state = m.create()
        probs = []
        try:
            for i in range(0, len(buf), self.frame_size):
                frame = np.ascontiguousarray(buf[i : i + self.frame_size], dtype=np.float32)
                ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                probs.append(float(m.lib.rnnoise_process_frame(state, ptr, ptr)))
                buf[i : i + self.frame_size] = frame
        finally:
            m.destroy(state)
        self.speech_probs = probs
        return buf[:n]

    def process_int16(self, x16: np.ndarray) -> np.ndarray:
        up = self._soxr.resample(
            np.asarray(x16, dtype=np.int16).astype(np.float32),
            SAMPLE_RATE,
            RNNOISE_RATE,
            quality=self.quality,
        )
        out = self.denoise_48k(np.asarray(up, dtype=np.float32).reshape(-1))
        down = self._soxr.resample(out, RNNOISE_RATE, SAMPLE_RATE, quality=self.quality)
        return to_int16(np.asarray(down).reshape(-1))


class RNNoiseStream:
    """Frame-at-a-time version, used only to measure real streaming delay/jitter."""

    def __init__(self, quality: str = RESAMPLE_QUALITY):
        import soxr

        self._m = load_rnnoise_shim()
        self.frame_size = int(self._m.FRAME_SIZE)
        self._up = soxr.ResampleStream(SAMPLE_RATE, RNNOISE_RATE, 1, dtype="float32", quality=quality)
        self._down = soxr.ResampleStream(RNNOISE_RATE, SAMPLE_RATE, 1, dtype="float32", quality=quality)
        self._state = self._m.create()
        self._buf = np.zeros(0, dtype=np.float32)

    def close(self) -> None:
        if self._state is not None:
            self._m.destroy(self._state)
            self._state = None

    def push(self, chunk16: np.ndarray, last: bool = False) -> np.ndarray:
        m = self._m
        up = self._up.resample_chunk(
            np.asarray(chunk16, dtype=np.int16).astype(np.float32).reshape(-1, 1), last=last
        )
        self._buf = np.concatenate([self._buf, np.asarray(up, dtype=np.float32).reshape(-1)])
        ready = []
        while len(self._buf) >= self.frame_size:
            frame = np.ascontiguousarray(self._buf[: self.frame_size], dtype=np.float32)
            self._buf = self._buf[self.frame_size :]
            ptr = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            m.lib.rnnoise_process_frame(self._state, ptr, ptr)
            ready.append(frame)
        if not ready:
            return np.zeros(0, dtype=np.int16)
        block = np.concatenate(ready).reshape(-1, 1)
        down = self._down.resample_chunk(block, last=last)
        return to_int16(np.asarray(down).reshape(-1))


# --------------------------------------------------------------------------
# F2 WebRTC APM
# --------------------------------------------------------------------------


def load_apm() -> tuple[object, object, str]:
    """Return (ApmConfig, AudioPreprocessor, provenance)."""
    try:
        from jetbot_agent.audio.audio_preprocessor import ApmConfig, AudioPreprocessor

        AudioPreprocessor(ApmConfig())
        return ApmConfig, AudioPreprocessor, "jetbot_agent.audio.audio_preprocessor (working tree)"
    except Exception as exc:
        pkg_err = repr(exc)
    if not APM_HEAD.exists():
        raise SystemExit(f"no usable F2 APM: package module failed ({pkg_err}) and {APM_HEAD} is missing")
    mod = load_file_module("_f3_apm_head", APM_HEAD)
    mod.AudioPreprocessor(mod.ApmConfig())
    return (
        mod.ApmConfig,
        mod.AudioPreprocessor,
        f"data/models/f4/_apm_head/audio_preprocessor.py (F2 HEAD copy; working-tree module unusable: {pkg_err})",
    )


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def synth_speech(seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """Verbatim copy of the F2 fixture generator, for byte-identical fixtures."""
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


def f2_fixtures() -> tuple[dict[str, np.ndarray], str]:
    """Prefer the WAVs F2 actually wrote; regenerate from F2's code if absent."""
    names = ("clean", "noise_only", "noisy_speech", "far_end", "echo_only")
    if all((F2_AUDIO / f"{n}.wav").exists() for n in names):
        out = {}
        for n in names:
            samples, rate = read_wav(F2_AUDIO / f"{n}.wav")
            if rate != SAMPLE_RATE:
                raise SystemExit(f"{n}.wav is {rate} Hz")
            out[n] = samples
        return out, "read from data/audio/f2/ (the fixtures F2 measured)"

    seconds, n = 2.0, int(2.0 * SAMPLE_RATE)
    clean = synth_speech(seconds, seed=1)
    noise_only = (0.20 * np.random.default_rng(2).normal(size=n) * 32767).astype(np.int16)
    noisy = np.clip(clean.astype(np.int32) + noise_only.astype(np.int32), -32768, 32767).astype(np.int16)
    t = np.arange(n) / SAMPLE_RATE
    far = (0.50 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)
    delay = int(0.04 * SAMPLE_RATE)
    echo = np.zeros(n, dtype=np.int16)
    echo[delay:] = (0.40 * far[:-delay]).astype(np.int16)
    return (
        {"clean": clean, "noise_only": noise_only, "noisy_speech": noisy, "far_end": far, "echo_only": echo},
        "regenerated from F2's seeded generator (data/audio/f2/ was empty)",
    )


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db_target: float) -> np.ndarray:
    s = speech.astype(np.float64)
    v = noise.astype(np.float64)
    ps, pv = float(np.mean(s * s)), float(np.mean(v * v))
    if pv <= 0:
        return speech.copy()
    scale = np.sqrt(ps / pv) * (10.0 ** (-snr_db_target / 20.0))
    return to_int16(s + v * scale)


def white_noise(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(scale=3000.0, size=n)


def robot_noise(n: int, seed: int) -> np.ndarray:
    """Fan hum + motor whine + broadband: the noise floor this robot actually has."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SAMPLE_RATE
    hum = sum((1.0 / k) * np.sin(2 * np.pi * 120.0 * k * t + rng.uniform(0, 6.28)) for k in (1, 2, 3, 4))
    whine = 0.5 * np.sin(2 * np.pi * (2800.0 + 40.0 * np.sin(2 * np.pi * 0.7 * t)) * t)
    whine += 0.25 * np.sin(2 * np.pi * 5600.0 * t)
    broadband = 0.6 * rng.normal(size=n)
    x = 1.2 * hum + whine + broadband
    return x / max(float(np.max(np.abs(x))), 1e-9) * 8000.0


# --------------------------------------------------------------------------
# configurations under test
# --------------------------------------------------------------------------

CONFIGS = [
    # label, apm kwargs or None, rnnoise?
    ("raw", None, False),
    ("apm_ns", dict(echo_cancellation=False, noise_suppression=True, auto_gain_control=False,
                    high_pass_filter=True, ns_level=3), False),
    ("apm", dict(echo_cancellation=True, noise_suppression=True, auto_gain_control=False,
                 high_pass_filter=True, ns_level=3), False),
    ("apm+rnnoise", dict(echo_cancellation=True, noise_suppression=True, auto_gain_control=False,
                         high_pass_filter=True, ns_level=3), True),
    ("apm_agc", dict(), False),
    ("apm_agc+rnnoise", dict(), True),
    ("rnnoise", None, True),
]


def apply_config(
    label: str,
    apm_kwargs: dict | None,
    use_rnnoise: bool,
    near: np.ndarray,
    far: np.ndarray | None,
    ApmConfig,
    AudioPreprocessor,
) -> dict:
    audio_s = len(near) / SAMPLE_RATE
    out = np.asarray(near, dtype=np.int16)
    cpu0, wall0 = time.process_time(), time.perf_counter()
    apm_s = rnn_s = 0.0
    speech_prob_apm = None
    rnn_probs: list[float] = []

    if apm_kwargs is not None:
        t0 = time.perf_counter()
        ap = AudioPreprocessor(ApmConfig(**apm_kwargs))
        out = ap.process_int16(out, far)
        apm_s = time.perf_counter() - t0
        speech_prob_apm = ap.speech_probability
    if use_rnnoise:
        t0 = time.perf_counter()
        rn = RNNoise()
        out = rn.process_int16(out)
        rnn_s = time.perf_counter() - t0
        rnn_probs = rn.speech_probs

    wall = time.perf_counter() - wall0
    cpu = time.process_time() - cpu0
    return {
        "config": label,
        "out": out,
        "wall_s": wall,
        "cpu_s": cpu,
        "apm_s": apm_s,
        "rnnoise_s": rnn_s,
        "rtf": wall / audio_s,
        "cpu_rtf": cpu / audio_s,
        "speech_probability_apm": speech_prob_apm,
        "speech_probability_rnnoise_mean": float(np.mean(rnn_probs)) if rnn_probs else None,
    }


# --------------------------------------------------------------------------
# latency / resampler cost
# --------------------------------------------------------------------------


def measure_latency(probe: np.ndarray) -> dict:
    """Measure the delay and CPU the RNNoise stage actually adds, stage by stage.

    The whole point of F3 is deciding whether the 16k->48k->16k detour is worth
    paying for, so the added delay is decomposed into the three things that could
    cause it: RNNoise's own algorithm, the resampler, and the buffer needed to
    turn soxr's bursty stream output into RNNoise's fixed 480-sample frames.
    """
    import soxr

    n_chunks = len(probe) // FRAME_SAMPLES
    probe = probe[: n_chunks * FRAME_SAMPLES]
    audio_s = len(probe) / SAMPLE_RATE

    # (1) full streaming chain
    stream = RNNoiseStream()
    frame_size = stream.frame_size
    pieces, per_chunk = [], []
    cpu0 = time.process_time()
    for i in range(n_chunks):
        chunk = probe[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
        t0 = time.perf_counter()
        pieces.append(stream.push(chunk, last=(i == n_chunks - 1)))
        per_chunk.append(time.perf_counter() - t0)
    cpu_total = time.process_time() - cpu0
    stream.close()
    streamed = np.concatenate(pieces) if pieces else np.zeros(0, np.int16)
    per_chunk_ms = np.array(per_chunk) * 1000.0

    # Exact buffering delay from the emission schedule: after feeding chunk i the
    # caller has supplied (i+1)*160 samples, so whatever has not come back out yet
    # is sitting in a buffer. Cross-correlating through a denoiser cannot measure
    # this reliably -- RNNoise changes the spectrum -- but counting samples can.
    emitted = np.cumsum([len(p) for p in pieces[:-1]])  # exclude the flush chunk
    supplied = (np.arange(len(emitted)) + 1) * FRAME_SAMPLES
    deficit = supplied - emitted

    # (2) RNNoise alone at its native 48 kHz: no resampling, no reframing
    m = load_rnnoise_shim()
    up_full = np.asarray(
        soxr.resample(probe.astype(np.float32), SAMPLE_RATE, RNNOISE_RATE, quality=RESAMPLE_QUALITY),
        dtype=np.float32,
    ).reshape(-1)
    n48 = len(up_full) // frame_size * frame_size
    src48 = np.ascontiguousarray(up_full[:n48])
    out48 = np.empty_like(src48)
    state = m.create()
    frames = [np.ascontiguousarray(src48[i : i + frame_size]) for i in range(0, n48, frame_size)]
    ptrs = [f.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for f in frames]
    t0 = time.perf_counter()
    for p in ptrs:
        m.lib.rnnoise_process_frame(state, p, p)
    rnnoise_c_s = time.perf_counter() - t0
    m.destroy(state)
    for i, f in enumerate(frames):
        out48[i * frame_size : (i + 1) * frame_size] = f
    rnnoise_lag48 = best_lag(src48, out48)

    # (3) streaming resampler round trip alone, chunk in / chunk straight out
    up = soxr.ResampleStream(SAMPLE_RATE, RNNOISE_RATE, 1, dtype="float32", quality=RESAMPLE_QUALITY)
    down = soxr.ResampleStream(RNNOISE_RATE, SAMPLE_RATE, 1, dtype="float32", quality=RESAMPLE_QUALITY)
    rt_pieces, burst_lens = [], []
    for i in range(n_chunks):
        last = i == n_chunks - 1
        a = up.resample_chunk(
            probe[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES].astype(np.float32).reshape(-1, 1),
            last=last,
        )
        burst_lens.append(int(len(a)))
        b = down.resample_chunk(np.asarray(a, dtype=np.float32).reshape(-1, 1), last=last)
        rt_pieces.append(np.asarray(b, dtype=np.float32).reshape(-1))
    rt_stream = to_int16(np.concatenate(rt_pieces))

    # (4) whole-buffer resampler round trip, for the offline CPU cost
    t0 = time.perf_counter()
    b_up = soxr.resample(probe.astype(np.float32), SAMPLE_RATE, RNNOISE_RATE, quality=RESAMPLE_QUALITY)
    b_down = soxr.resample(b_up, RNNOISE_RATE, SAMPLE_RATE, quality=RESAMPLE_QUALITY)
    resample_only_s = time.perf_counter() - t0

    chain_lag = best_lag(probe, streamed) if len(streamed) else -1
    rt_lag = best_lag(probe[: len(rt_stream)], rt_stream)
    return {
        "frame_ms": FRAME_MS,
        "input_frame_samples_16k": FRAME_SAMPLES,
        "rnnoise_frame_samples_48k": frame_size,
        "rnnoise_frames_per_input_frame": (FRAME_SAMPLES * RNNOISE_RATE / SAMPLE_RATE) / frame_size,
        "input_samples": int(len(probe)),
        "streamed_out_samples": int(len(streamed)),
        "sample_accurate": int(len(streamed)) == int(len(probe)),
        # added delay, decomposed
        "buffering_delay_samples_max": int(np.max(deficit)) if len(deficit) else 0,
        "buffering_delay_ms_max": float(np.max(deficit)) * 1000.0 / SAMPLE_RATE if len(deficit) else 0.0,
        "buffering_delay_ms_p50": float(np.percentile(deficit, 50)) * 1000.0 / SAMPLE_RATE
        if len(deficit)
        else 0.0,
        "chain_group_delay_samples_xcorr": chain_lag,
        "chain_group_delay_ms_xcorr": chain_lag * 1000.0 / SAMPLE_RATE if chain_lag >= 0 else None,
        "rnnoise_only_group_delay_samples_48k": rnnoise_lag48,
        "rnnoise_only_group_delay_ms": rnnoise_lag48 * 1000.0 / RNNOISE_RATE,
        "streaming_resampler_only_group_delay_samples": rt_lag,
        "streaming_resampler_only_group_delay_ms": rt_lag * 1000.0 / SAMPLE_RATE,
        "batch_resampler_group_delay_samples": best_lag(probe, to_int16(np.asarray(b_down).reshape(-1))),
        # cost
        "budget_ms": float(FRAME_MS),
        "per_chunk_ms_mean": float(np.mean(per_chunk_ms)),
        "per_chunk_ms_p50": float(np.percentile(per_chunk_ms, 50)),
        "per_chunk_ms_p95": float(np.percentile(per_chunk_ms, 95)),
        "per_chunk_ms_max": float(np.max(per_chunk_ms)),
        "chunks_over_budget": int(np.sum(per_chunk_ms > FRAME_MS)),
        "chunks_total": int(n_chunks),
        "streaming_cpu_rtf": cpu_total / audio_s,
        "rnnoise_c_only_rtf": rnnoise_c_s / (n48 / RNNOISE_RATE),
        "rnnoise_c_only_ms_per_frame": 1000.0 * rnnoise_c_s / max(len(frames), 1),
        "resample_roundtrip_only_rtf": resample_only_s / audio_s,
        "resampler_burst_len_p50": float(np.percentile(burst_lens, 50)),
        "resampler_burst_len_max": int(np.max(burst_lens)),
        "note": (
            "soxr's ResampleStream emits in bursts rather than one 480-sample block per "
            "10 ms input frame, so the chain needs a reframing buffer; that buffer, not "
            "RNNoise and not the resampler filters, is what the chain delay measures."
        ),
    }


# --------------------------------------------------------------------------
# ASR worker: one recognizer load, every fixture for one config
# --------------------------------------------------------------------------


def run_asr_worker(args: argparse.Namespace) -> int:
    import sherpa_onnx

    rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(ASR_MODEL / "model.int8.onnx"),
        tokens=str(ASR_MODEL / "tokens.txt"),
        num_threads=ASR_THREADS,
        provider="cpu",
    )
    for wav_path in json.loads(Path(args.manifest).read_text()):
        wav = Path(wav_path)
        samples, rate = read_wav(wav)
        if rate != SAMPLE_RATE:
            raise SystemExit(f"{wav}: expected {SAMPLE_RATE} Hz")
        floats = samples.astype(np.float32) / 32768.0
        t0 = time.perf_counter()
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, floats)
        rec.decode_stream(stream)
        infer_s = time.perf_counter() - t0
        print(
            "@@RESULT@@"
            + json.dumps(
                {
                    "wav": wav.name,
                    "audio_s": len(samples) / rate,
                    "infer_s": infer_s,
                    "rtf": infer_s * rate / len(samples),
                    "transcript": stream.result.text,
                }
            ),
            flush=True,
        )
    print("@@RSS@@" + json.dumps({"peak_rss_mib": peak_rss_mib()}), flush=True)
    return 0


def launch_asr(manifest: Path) -> tuple[list[dict], float]:
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--asr-worker", "--manifest", str(manifest)],
        capture_output=True,
        text=True,
    )
    rows, rss = [], float("nan")
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            rows.append(json.loads(line[len("@@RESULT@@") :]))
        elif line.startswith("@@RSS@@"):
            rss = json.loads(line[len("@@RSS@@") :])["peak_rss_mib"]
    if not rows:
        raise SystemExit(
            f"ASR worker produced no result\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return rows, rss


def tegrastats_sample(seconds: int = 3) -> str | None:
    try:
        proc = subprocess.run(
            ["tegrastats", "--interval", "1000"], capture_output=True, text=True, timeout=seconds
        )
        lines = (proc.stdout or "").strip().splitlines()
    except subprocess.TimeoutExpired as exc:
        raw = (exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        lines = raw.strip().splitlines()
    except Exception:
        return None
    return lines[-1] if lines else None


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    ApmConfig, AudioPreprocessor, apm_provenance = load_apm()
    rn_probe = RNNoise()
    f2, f2_provenance = f2_fixtures()

    # ---- fixture set -----------------------------------------------------
    # Every reference is real audio or an F2 fixture; nothing is invented.
    signal_fixtures: list[dict] = [
        {"name": "f2_noise_only", "near": f2["noise_only"], "far": None, "ref": None,
         "kind": "noise_only", "note": "F2 white-noise fixture, no speech"},
        {"name": "f2_clean", "near": f2["clean"], "far": None, "ref": f2["clean"],
         "kind": "speech", "note": "F2 'clean speech' fixture -- a synthetic tone complex, not real speech"},
        {"name": "f2_noisy_speech", "near": f2["noisy_speech"], "far": None, "ref": f2["clean"],
         "kind": "noisy_speech", "note": "F2 tone complex + white noise"},
        {"name": "f2_echo_only", "near": f2["echo_only"], "far": f2["far_end"], "ref": None,
         "kind": "echo_only", "note": "F2 echo fixture with the time-aligned far-end reference"},
    ]

    asr_specs: list[str] = []
    references: dict[str, str] = {}
    trans = LIBRI / "trans.txt"
    if trans.exists():
        for line in trans.read_text(encoding="utf-8").splitlines():
            name, _, text = line.partition(" ")
            references[name] = text.strip()

    libri_available = (LIBRI / "0.wav").exists() and bool(references)
    if libri_available:
        utterances = []
        for wav_name, tag in (("0.wav", "libri0"), ("1.wav", "libri1")):
            path = LIBRI / wav_name
            if not path.exists() or wav_name not in references:
                continue
            samples, rate = read_wav(path)
            if rate != SAMPLE_RATE:
                raise SystemExit(f"{path} is {rate} Hz")
            utterances.append((tag, samples, references[wav_name]))

        # seed offsets keep every mix reproducible and independent
        recipes = [
            ("clean", None, None, "unmodified real LibriSpeech utterance"),
            ("white_5db", white_noise, 5.0, "+ white noise at 5 dB SNR"),
            ("white_0db", white_noise, 0.0, "+ white noise at 0 dB SNR"),
            ("robot_10db", robot_noise, 10.0, "+ fan hum / motor whine at 10 dB SNR"),
            ("robot_5db", robot_noise, 5.0, "+ fan hum / motor whine at 5 dB SNR"),
            ("robot_0db", robot_noise, 0.0, "+ fan hum / motor whine at 0 dB SNR"),
        ]
        for u_idx, (tag, clean, ref_text) in enumerate(utterances):
            for r_idx, (suffix, noise_fn, snr, note) in enumerate(recipes):
                if suffix == "clean":
                    near = clean
                else:
                    near = mix_at_snr(clean, noise_fn(len(clean), 100 + 10 * u_idx + r_idx), snr)
                signal_fixtures.append(
                    {
                        "name": f"{tag}_{suffix}",
                        "near": near,
                        "far": None,
                        "ref": clean,
                        "kind": "speech" if suffix == "clean" else "noisy_speech",
                        "note": f"{tag} {note}",
                        "asr_reference": ref_text,
                    }
                )
                asr_specs.append(f"{tag}_{suffix}")

    for fx in signal_fixtures:
        write_wav(OUT / f"{fx['name']}__input.wav", fx["near"])

    # ---- one config sweep, reused for both the signal table and the ASR run ----
    signal_rows = []
    processed_paths: dict[tuple[str, str], Path] = {}
    for fx in signal_fixtures:
        ref = fx["ref"]
        in_rms = rms(fx["near"])
        for label, apm_kwargs, use_rn in CONFIGS:
            # Every config runs on every fixture on purpose, including rnnoise-only
            # on the echo fixture: "RNNoise cannot replace AEC" has to be measured.
            res = apply_config(
                label, apm_kwargs, use_rn, fx["near"], fx["far"], ApmConfig, AudioPreprocessor
            )
            out = res.pop("out")
            path = OUT / f"{fx['name']}__{label.replace('+', '_')}.wav"
            write_wav(path, out)
            processed_paths[(fx["name"], label)] = path
            row = {
                "fixture": fx["name"],
                "kind": fx["kind"],
                **res,
                "in_rms": in_rms,
                "out_rms": rms(out),
                "out_peak": peak(out),
                "rms_reduction_x": in_rms / max(rms(out), 1e-9),
            }
            if ref is not None:
                g, s, lag, gain = snr_db(ref, out)
                row.update(
                    {
                        "snr_vs_clean_db": g,
                        "seg_snr_vs_clean_db": s,
                        "delay_samples": lag,
                        "fitted_gain": gain,
                    }
                )
            signal_rows.append(row)

    # ---- latency: real speech probe, not the 2 s synthetic F2 fixture -----
    by_name = {fx["name"]: fx["near"] for fx in signal_fixtures}
    latency_probe = by_name.get("libri0_clean", by_name["f2_noisy_speech"])
    latency = measure_latency(np.asarray(latency_probe, dtype=np.int16))

    # ---- downstream ASR on the audio the sweep already produced ----------
    asr_rows = []
    asr_skip = None
    refs = {fx["name"]: fx.get("asr_reference") for fx in signal_fixtures}
    if not asr_specs:
        asr_skip = f"no LibriSpeech fixtures with reference text under {LIBRI}"
    elif not (ASR_MODEL / "model.int8.onnx").exists():
        asr_skip = (
            f"historical FastConformer model removed at {ASR_MODEL}; "
            "the committed F3 comparison remains the evidence"
        )
    else:
        for label, _kw, _rn in CONFIGS:
            paths = [str(processed_paths[(name, label)]) for name in asr_specs]
            manifest = OUT / f"_manifest_{label.replace('+', '_')}.json"
            manifest.write_text(json.dumps(paths), encoding="utf-8")
            rows, rss = launch_asr(manifest)
            for r, name in zip(rows, asr_specs):
                ref_text = refs[name]
                asr_rows.append(
                    {
                        "config": label,
                        "fixture": name,
                        "audio_s": r["audio_s"],
                        "infer_s": r["infer_s"],
                        "rtf": r["rtf"],
                        "peak_rss_mib": rss,
                        "reference": ref_text,
                        "transcript": r["transcript"],
                        "wer": wer(ref_text, r["transcript"]) if ref_text else None,
                    }
                )

    # Mean WER per config, split clean vs noisy: the headline decision number.
    wer_by_config = {}
    for label, _kw, _rn in CONFIGS:
        rows = [r for r in asr_rows if r["config"] == label and r["wer"] is not None]
        clean = [r["wer"] for r in rows if r["fixture"].endswith("_clean")]
        noisy = [r["wer"] for r in rows if not r["fixture"].endswith("_clean")]
        if rows:
            wer_by_config[label] = {
                "mean_wer_clean": float(np.mean(clean)) if clean else None,
                "mean_wer_noisy": float(np.mean(noisy)) if noisy else None,
                "max_wer_noisy": float(np.max(noisy)) if noisy else None,
                "n_noisy_fixtures": len(noisy),
            }

    report = {
        "gate": "F3 optional RNNoise residual denoising A/B",
        "verdict_owner": "see docs/bringup/06b-f3-rnnoise.md",
        "install": {
            "package": "pyrnnoise==0.4.3 (manylinux2014_aarch64 wheel, --no-deps)",
            "library": "bundled librnnoise.so (xiph RNNoise, built-in model)",
            "binding": "pyrnnoise/rnnoise.py ctypes shim, loaded directly to avoid audiolab/matplotlib",
            "resampler": "soxr==1.1.0 (libsoxr), quality=" + RESAMPLE_QUALITY,
            "rnnoise_rate": RNNOISE_RATE,
            "rnnoise_frame_samples": rn_probe.frame_size,
            "pipeline_rate": SAMPLE_RATE,
        },
        "apm_provenance": apm_provenance,
        "f2_fixture_provenance": f2_provenance,
        "asr": {
            "runtime": "sherpa-onnx OfflineRecognizer.from_nemo_ctc, int8 ONNX, cpu",
            "model": ASR_MODEL.name,
            "threads": ASR_THREADS,
            "audio": "real LibriSpeech utterances with shipped reference transcripts; "
                     "noise is synthetic and additively mixed at a measured SNR",
            "no_real_noisy_recording": "the robot's own microphone has not produced a noisy "
                                       "fixture with ground-truth text; nothing under data/audio/ "
                                       "is a real capture",
            "skipped": asr_skip,
            "wer_by_config": wer_by_config,
            "results": asr_rows,
        },
        "latency": latency,
        "signal_results": signal_rows,
        "peak_rss_mib_signal_process": peak_rss_mib(),
        "host_loadavg_1m": float(Path("/proc/loadavg").read_text().split()[0]),
        "tegrastats_sample": tegrastats_sample(),
        "live_duplex": False,
    }

    payload = json.dumps(report, indent=2)
    LOG.write_text(payload, encoding="utf-8")
    (OUT / "report.json").write_text(payload, encoding="utf-8")
    print(payload)
    summarize(report)
    return 0


def summarize(report: dict) -> None:
    print("\n=== F3 signal A/B ===")
    hdr = f"{'fixture':22} {'config':17} {'out_rms':>8} {'red_x':>7} {'peak':>6} {'segSNR':>7} {'RTF':>7}"
    print(hdr)
    for r in report["signal_results"]:
        seg = r.get("seg_snr_vs_clean_db")
        print(
            f"{r['fixture'][:22]:22} {r['config'][:17]:17} {r['out_rms']:8.4f} "
            f"{r['rms_reduction_x']:7.2f} {r['out_peak']:6.3f} "
            f"{(f'{seg:7.2f}' if isinstance(seg, float) and np.isfinite(seg) else '      -')} "
            f"{r['rtf']:7.4f}"
        )
    if report["asr"]["results"]:
        print("\n=== F3 downstream ASR (FastConformer CTC int8, 2 threads) ===")
        print(f"{'fixture':22} {'config':17} {'WER':>6}  transcript")
        for r in sorted(report["asr"]["results"], key=lambda r: (r["fixture"], r["config"])):
            w = r["wer"]
            print(
                f"{r['fixture'][:22]:22} {r['config'][:17]:17} "
                f"{(f'{w:6.3f}' if w is not None else '     -')}  {r['transcript'][:70]}"
            )
        print("\n=== F3 WER rollup ===")
        print(f"{'config':17} {'clean':>8} {'noisy':>8} {'worst':>8}")
        for label, agg in report["asr"]["wer_by_config"].items():
            print(
                f"{label:17} {agg['mean_wer_clean']:8.3f} "
                f"{agg['mean_wer_noisy']:8.3f} {agg['max_wer_noisy']:8.3f}"
            )

    lat = report["latency"]
    print(
        f"\nadded delay: RNNoise itself {lat['rnnoise_only_group_delay_ms']:.2f} ms, "
        f"soxr streaming filters {lat['streaming_resampler_only_group_delay_ms']:.2f} ms, "
        f"reframing buffer {lat['buffering_delay_ms_p50']:.2f} ms typical / "
        f"{lat['buffering_delay_ms_max']:.2f} ms worst case.\n"
        f"cost per 10 ms frame: mean {lat['per_chunk_ms_mean']:.3f} ms, "
        f"p95 {lat['per_chunk_ms_p95']:.3f} ms, max {lat['per_chunk_ms_max']:.3f} ms; "
        f"{lat['chunks_over_budget']}/{lat['chunks_total']} frames over the "
        f"{lat['budget_ms']:.0f} ms budget.\n"
        f"RNNoise C library alone: RTF {lat['rnnoise_c_only_rtf']:.4f} "
        f"({lat['rnnoise_c_only_ms_per_frame']:.3f} ms per 480-sample frame); "
        f"resampler round trip alone: RTF {lat['resample_roundtrip_only_rtf']:.5f}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asr-worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--manifest", default="")
    args = ap.parse_args()
    return run_asr_worker(args) if args.asr_worker else run()


if __name__ == "__main__":
    raise SystemExit(main())
