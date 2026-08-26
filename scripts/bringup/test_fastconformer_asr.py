#!/usr/bin/env python3
"""F4: NVIDIA NeMo FastConformer CTC transcription on-device, offline gate.

Each measured configuration runs in its own worker process so the reported peak
RSS belongs to that configuration alone and not to an accumulated sweep.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "data" / "models" / "f4"
OFFLINE_DIR = MODELS / "sherpa-onnx-nemo-fast-conformer-ctc-en-24500-int8"
STREAMING_DIR = MODELS / "sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms-int8"
AUDIO_OUT = ROOT / "data" / "audio" / "f4"
LOG = ROOT / "data" / "bringup" / "f4_fastconformer_asr.json"

SAMPLE_RATE = 16000
WARM_RUNS = 5


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise SystemExit(f"{path}: need mono 16-bit PCM")
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16), rate


def write_wav(path: Path, samples: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(np.asarray(samples, dtype=np.int16).tobytes())


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


def cpu_seconds() -> float:
    parts = Path("/proc/self/stat").read_text().split()
    ticks = int(parts[13]) + int(parts[14])
    return ticks / 100.0


# --------------------------------------------------------------------------
# worker: one recognizer configuration, one JSON line on stdout
# --------------------------------------------------------------------------


def run_worker(args: argparse.Namespace) -> int:
    import sherpa_onnx

    wav = Path(args.wav)
    samples, rate = read_wav(wav)
    if rate != SAMPLE_RATE:
        raise SystemExit(f"{wav}: expected {SAMPLE_RATE} Hz, got {rate}")
    audio_s = len(samples) / rate
    floats = samples.astype(np.float32) / 32768.0

    model_dir = Path(args.model_dir)
    model = str(model_dir / "model.int8.onnx")
    tokens = str(model_dir / "tokens.txt")

    load_t0 = time.perf_counter()
    if args.mode == "streaming":
        rec = sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
            tokens=tokens, model=model, num_threads=args.threads, provider="cpu"
        )
    else:
        rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=model, tokens=tokens, num_threads=args.threads, provider="cpu"
        )
    load_s = time.perf_counter() - load_t0

    def decode_once() -> str:
        if args.mode == "streaming":
            stream = rec.create_stream()
            chunk = int(0.1 * SAMPLE_RATE)
            for off in range(0, len(floats), chunk):
                stream.accept_waveform(SAMPLE_RATE, floats[off : off + chunk])
                while rec.is_ready(stream):
                    rec.decode_stream(stream)
            stream.input_finished()
            while rec.is_ready(stream):
                rec.decode_stream(stream)
            return rec.get_result(stream)
        stream = rec.create_stream()
        stream.accept_waveform(SAMPLE_RATE, floats)
        rec.decode_stream(stream)
        return stream.result.text

    cold_t0 = time.perf_counter()
    text = decode_once()
    cold_s = time.perf_counter() - cold_t0

    cpu0, wall0 = cpu_seconds(), time.perf_counter()
    warm = []
    for _ in range(WARM_RUNS):
        t0 = time.perf_counter()
        text = decode_once()
        warm.append(time.perf_counter() - t0)
    cpu_used = cpu_seconds() - cpu0
    wall_used = time.perf_counter() - wall0

    # The Jetson is a shared box during bring-up, so a warm run can be stolen by
    # unrelated load. Report the median as the headline number and keep min/max
    # so contention stays visible instead of quietly inflating the mean.
    warm_median = float(np.median(warm))
    out = {
        "label": args.label,
        "mode": args.mode,
        "model_dir": model_dir.name,
        "wav": wav.name,
        "threads": args.threads,
        "audio_s": audio_s,
        "load_s": load_s,
        "cold_infer_s": cold_s,
        "cold_total_s": load_s + cold_s,
        "warm_infer_s_median": warm_median,
        "warm_infer_s_mean": float(np.mean(warm)),
        "warm_infer_s_min": float(np.min(warm)),
        "warm_infer_s_max": float(np.max(warm)),
        "rtf_cold": cold_s / audio_s,
        "rtf_warm": warm_median / audio_s,
        "rtf_warm_best": float(np.min(warm)) / audio_s,
        "cpu_cores_busy": cpu_used / wall_used if wall_used else float("nan"),
        "host_loadavg_1m": float(Path("/proc/loadavg").read_text().split()[0]),
        "peak_rss_mib": peak_rss_mib(),
        "transcript": text,
    }
    print("@@RESULT@@" + json.dumps(out))
    return 0


# --------------------------------------------------------------------------
# parent: orchestrate configurations, sample system memory
# --------------------------------------------------------------------------


class MemSampler:
    """Track the lowest system MemAvailable seen while the gate runs."""

    def __init__(self) -> None:
        self.min_available_mib = float("inf")
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    @staticmethod
    def _available_mib() -> float:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
        return float("nan")

    def _loop(self) -> None:
        while not self._stop.wait(0.25):
            self.min_available_mib = min(self.min_available_mib, self._available_mib())

    def __enter__(self) -> MemSampler:
        self.min_available_mib = self._available_mib()
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)


def launch(label: str, mode: str, model_dir: Path, wav: Path, threads: int) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--label",
        label,
        "--mode",
        mode,
        "--model-dir",
        str(model_dir),
        "--wav",
        str(wav),
        "--threads",
        str(threads),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@") :])
    raise SystemExit(
        f"{label}: worker produced no result\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )


def apm_variant(src: Path, dest: Path) -> tuple[Path | None, str]:
    """Write the F2 WebRTC-APM-processed copy of src.

    Returns (path, reason). The APM cross-check is optional, but a skip must say
    why so a stubbed or broken F2 module is never mistaken for a clean run.
    """
    try:
        from jetbot_agent.audio.audio_preprocessor import ApmConfig, AudioPreprocessor
    except Exception as exc:
        return None, f"F2 preprocessor unavailable: {exc!r}"
    try:
        samples, rate = read_wav(src)
        if rate != SAMPLE_RATE:
            return None, f"{src.name} is {rate} Hz, expected {SAMPLE_RATE}"
        processed = AudioPreprocessor(ApmConfig()).process_int16(samples)
    except Exception as exc:
        return None, f"F2 preprocessor failed: {exc!r}"
    write_wav(dest, processed)
    return dest, "ok"


def tegrastats_sample(seconds: int = 3) -> str | None:
    try:
        proc = subprocess.run(
            ["tegrastats", "--interval", "1000"],
            capture_output=True,
            text=True,
            timeout=seconds,
        )
        lines = (proc.stdout or "").strip().splitlines()
    except subprocess.TimeoutExpired as exc:
        lines = ((exc.stdout or b"").decode() if isinstance(exc.stdout, bytes) else exc.stdout or "")
        lines = lines.strip().splitlines()
    except Exception:
        return None
    return lines[-1] if lines else None


def run() -> int:
    if not (OFFLINE_DIR / "model.int8.onnx").exists():
        raise SystemExit(
            f"missing {OFFLINE_DIR}; run scripts/bringup/fetch_fastconformer_models.sh first"
        )
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    # Only the streaming archive ships reference transcripts, and the two
    # archives use the same file names for different utterances, so take every
    # test WAV from the streaming archive to keep WER meaningful.
    fixtures = STREAMING_DIR / "test_wavs"
    references = {}
    trans = fixtures / "trans.txt"
    if trans.exists():
        for line in trans.read_text(encoding="utf-8").splitlines():
            name, _, text = line.partition(" ")
            references[name] = text.strip()

    primary = fixtures / "0.wav"
    apm_wav, apm_reason = apm_variant(primary, AUDIO_OUT / "0_apm.wav")

    results = []
    with MemSampler() as mem:
        for threads in (1, 2, 4):
            results.append(
                launch(f"offline-ctc-{threads}t", "offline", OFFLINE_DIR, primary, threads)
            )
        results.append(
            launch("streaming-ctc-80ms-2t", "streaming", STREAMING_DIR, primary, 2)
        )
        if apm_wav is not None:
            results.append(
                launch("offline-ctc-4t-apm", "offline", OFFLINE_DIR, apm_wav, 4)
            )
        long_wav = fixtures / "1.wav"
        if long_wav.exists():
            results.append(
                launch("offline-ctc-4t-long", "offline", OFFLINE_DIR, long_wav, 4)
            )
        tegra = tegrastats_sample()

    ref_for = {"0.wav": references.get("0.wav", ""), "1.wav": references.get("1.wav", "")}
    ref_for["0_apm.wav"] = ref_for["0.wav"]
    for r in results:
        ref = ref_for.get(r["wav"], "")
        r["reference"] = ref
        r["wer"] = wer(ref, r["transcript"]) if ref else None

    report = {
        "gate": "F4 NVIDIA NeMo FastConformer CTC",
        "runtime": "sherpa-onnx 1.13.6 (bundled onnxruntime), CPU execution provider",
        "precision": "int8 quantized ONNX",
        "provider": "cpu",
        "note_gpu": (
            "CUDA execution provider not used: no JetPack-6 onnxruntime-gpu wheel is "
            "reachable from the allowlisted sandbox indexes, and no torch/NeMo stack "
            "is installed. CPU-only numbers below are therefore a floor, not a ceiling."
        ),
        "model_source": (
            "k2-fsa/sherpa-onnx GitHub release mirror of the NVIDIA NeMo exports "
            "(huggingface.co unreachable; NGC model API needs an auth key)"
        ),
        "audio_source": "LibriSpeech utterances shipped with the model archives (real speech)",
        "apm_cross_check": apm_reason,
        "warm_runs": WARM_RUNS,
        "system_min_mem_available_mib": mem.min_available_mib,
        "tegrastats_sample": tegra,
        "results": results,
    }

    payload = json.dumps(report, indent=2)
    LOG.write_text(payload, encoding="utf-8")
    print(payload)

    best = min((r for r in results if r["wav"] == "0.wav"), key=lambda r: r["rtf_warm"])
    if not best["transcript"].strip():
        raise SystemExit("FAIL: empty transcript")
    if best["wer"] is not None and best["wer"] > 0.15:
        raise SystemExit(f"FAIL: WER {best['wer']:.3f} above 0.15 on a clean utterance")
    if best["rtf_warm"] >= 1.0:
        raise SystemExit(f"FAIL: warm RTF {best['rtf_warm']:.3f} is not real time")
    print(
        f"F4 FastConformer ok: warm RTF {best['rtf_warm']:.3f}, "
        f"WER {best['wer']:.3f}, peak RSS {best['peak_rss_mib']:.0f} MiB"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--label", default="")
    ap.add_argument("--mode", default="offline", choices=("offline", "streaming"))
    ap.add_argument("--model-dir", default=str(OFFLINE_DIR))
    ap.add_argument("--wav", default=str(OFFLINE_DIR / "test_wavs" / "0.wav"))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    return run_worker(args) if args.worker else run()


if __name__ == "__main__":
    raise SystemExit(main())
