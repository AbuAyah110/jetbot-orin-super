#!/usr/bin/env python3
"""F5: two-stage neural TTS on-device (text -> mel -> HiFi-GAN waveform -> WAV).

The gate the stage asks for is FastPitch + HiFi-GAN. NGC ships NVIDIA's
FastPitch and HiFi-GAN only as `.nemo` torch archives, and nemo_toolkit[tts]
resolves to torch 2.13 plus a CUDA 13 wheel stack that is the wrong CUDA
generation for JetPack 6 and not a Tegra iGPU build, so the mel generator is
substituted and the vocoder stage is kept: Matcha-TTS (text -> mel) then
HiFi-GAN (mel -> waveform), both ONNX under sherpa-onnx. Rationale and the
exact URLs are recorded in docs/bringup/06-voice.md.

Each measured configuration runs in its own worker process so the reported peak
RSS belongs to that configuration alone and not to an accumulated sweep.

Playback is opt-in and never concurrent with capture. See safe_playback().
"""

from __future__ import annotations

import argparse
import json
import os
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

MODELS = ROOT / "data" / "models" / "f5"
ACOUSTIC_DIR = MODELS / "matcha-icefall-en_US-ljspeech"
AUDIO_OUT = ROOT / "data" / "audio" / "f5"
LOG = ROOT / "data" / "bringup" / "f5_matcha_hifigan_tts.json"

# Reused from F4 to check intelligibility without a listener.
ASR_DIR = (
    ROOT / "data" / "models" / "f4" / "sherpa-onnx-nemo-fast-conformer-ctc-en-24500-int8"
)
ASR_RATE = 16000

WARM_RUNS = 5

# Sentences are synthesized one at a time so playback can start early, then
# joined. Butt-splicing them degrades the first phoneme of the next sentence:
# concatenated with no gap, the plosive onset of "Stopping" collided with the
# tail of "detected." and the F4 recognizer heard "Sopping" in 4 of 8 runs. The
# same sentence synthesized alone was correct 12 of 12 times, so this is a
# splice artifact, not a model defect. 200 ms restored 8 of 8 and reads as
# normal sentence prosody.
INTER_SENTENCE_GAP_S = 0.20

# Speaker must stay at or below this. The SSS1629's hardware sidetone once
# produced a mic-to-speaker feedback loop, so playback level is capped and the
# 'Mic' PLAYBACK control is driven to 0% and muted before any aplay.
SPEAKER_MAX_PCT = 40
SPEAKER_PCT = 20

SENTENCES = {
    "robot_stop": "Obstacle detected. Stopping now.",
    "robot_dock": (
        "Battery at twelve percent. Returning to the charging dock. "
        "Please clear a path ahead."
    ),
}


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text.strip())]
    return [p for p in parts if p]


def write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes((pcm * 32767.0).astype(np.int16).tobytes())


def verify_wav(path: Path) -> dict:
    """Offline validation, so a WAV is never sent to a speaker unchecked."""
    with wave.open(str(path), "rb") as wf:
        nchan, sw, rate, nframes = (
            wf.getnchannels(),
            wf.getsampwidth(),
            wf.getframerate(),
            wf.getnframes(),
        )
        raw = wf.readframes(nframes)
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    peak = float(np.abs(samples).max()) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "channels": nchan,
        "sample_width": sw,
        "sample_rate": rate,
        "duration_s": nframes / rate if rate else 0.0,
        "peak": peak,
        "rms": rms,
        "clipped": bool(peak >= 0.999),
        "silent": bool(rms < 0.0005),
    }


def norm_text(s: str) -> list[str]:
    return re.sub(r"[^A-Z' ]+", " ", s.upper()).split()


def _edit_rate(r: list | str, h: list | str) -> float:
    if not r:
        return float("nan")
    prev = list(range(len(h) + 1))
    for i, rt in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, ht in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rt != ht))
        prev = cur
    return prev[len(h)] / len(r)


def wer(ref: str, hyp: str) -> float:
    return _edit_rate(norm_text(ref), norm_text(hyp))


def cer(ref: str, hyp: str) -> float:
    """Character error rate.

    The gate decides on CER, not WER: the robot phrases are deliberately short,
    and on a four-word utterance a single substitution is already WER 0.25, so
    WER cannot distinguish "one softened consonant" from "unintelligible". WER
    is still recorded for continuity with F4.
    """
    return _edit_rate(" ".join(norm_text(ref)), " ".join(norm_text(hyp)))


def peak_rss_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024.0
    return float("nan")


def cpu_seconds() -> float:
    parts = Path("/proc/self/stat").read_text().split()
    return (int(parts[13]) + int(parts[14])) / 100.0


# --------------------------------------------------------------------------
# worker: one (vocoder, threads) configuration, one JSON line on stdout
# --------------------------------------------------------------------------


def run_worker(args: argparse.Namespace) -> int:
    import sherpa_onnx

    vocoder = MODELS / args.vocoder
    text = SENTENCES[args.sentence]
    sentences = split_sentences(text)

    # max_num_sentences=1 makes the engine emit one sentence per generate call,
    # which is what lets first-audio latency be measured the way F6 will stream:
    # play sentence one while sentence two is still being synthesized.
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                acoustic_model=str(ACOUSTIC_DIR / "model-steps-3.onnx"),
                vocoder=str(vocoder),
                tokens=str(ACOUSTIC_DIR / "tokens.txt"),
                data_dir=str(ACOUSTIC_DIR / "espeak-ng-data"),
            ),
            provider="cpu",
            num_threads=args.threads,
        ),
        max_num_sentences=1,
    )
    if not cfg.validate():
        raise SystemExit(f"invalid TTS config for vocoder {args.vocoder}")

    load_t0 = time.perf_counter()
    tts = sherpa_onnx.OfflineTts(cfg)
    load_s = time.perf_counter() - load_t0
    rate = tts.sample_rate

    gap = np.zeros(int(rate * INTER_SENTENCE_GAP_S), dtype=np.float32)

    def synth(t: str) -> np.ndarray:
        audio = tts.generate(t, sid=0, speed=1.0)
        return np.asarray(audio.samples, dtype=np.float32)

    def join(chunks: list[np.ndarray]) -> np.ndarray:
        joined: list[np.ndarray] = []
        for i, c in enumerate(chunks):
            if i:
                joined.append(gap)
            joined.append(c)
        return np.concatenate(joined)

    # Cold: first call after load pays ONNX arena/kernel warmup.
    cold_t0 = time.perf_counter()
    first_cold = synth(sentences[0])
    first_audio_cold_s = time.perf_counter() - cold_t0

    full_t0 = time.perf_counter()
    full = join([synth(s) for s in sentences])
    full_cold_s = time.perf_counter() - full_t0

    first_warm, full_warm = [], []
    cpu0, wall0 = cpu_seconds(), time.perf_counter()
    for _ in range(WARM_RUNS):
        t0 = time.perf_counter()
        chunks = []
        for i, s in enumerate(sentences):
            chunks.append(synth(s))
            if i == 0:
                first_warm.append(time.perf_counter() - t0)
        full_warm.append(time.perf_counter() - t0)
        full = join(chunks)
    cpu_used = cpu_seconds() - cpu0
    wall_used = time.perf_counter() - wall0

    wav = AUDIO_OUT / f"{args.sentence}_{Path(args.vocoder).stem}_{args.threads}t.wav"
    write_wav(wav, full, rate)

    audio_s = len(full) / rate
    first_audio_s = len(first_cold) / rate
    full_median = float(np.median(full_warm))
    out = {
        "label": args.label,
        "sentence_key": args.sentence,
        "text": text,
        "num_sentences": len(sentences),
        "vocoder": args.vocoder,
        "threads": args.threads,
        "sample_rate": rate,
        "inter_sentence_gap_s": INTER_SENTENCE_GAP_S,
        "audio_s": audio_s,
        "first_sentence": sentences[0],
        "first_sentence_audio_s": first_audio_s,
        "load_s": load_s,
        "first_audio_cold_s": first_audio_cold_s,
        "first_audio_cold_total_s": load_s + first_audio_cold_s,
        "first_audio_warm_s_median": float(np.median(first_warm)),
        "total_cold_s": full_cold_s,
        "total_cold_total_s": load_s + first_audio_cold_s + full_cold_s,
        "total_warm_s_median": full_median,
        "total_warm_s_mean": float(np.mean(full_warm)),
        "total_warm_s_min": float(np.min(full_warm)),
        "total_warm_s_max": float(np.max(full_warm)),
        "rtf_cold": full_cold_s / audio_s,
        "rtf_warm": full_median / audio_s,
        "rtf_warm_best": float(np.min(full_warm)) / audio_s,
        "cpu_cores_busy": cpu_used / wall_used if wall_used else float("nan"),
        "host_loadavg_1m": float(Path("/proc/loadavg").read_text().split()[0]),
        "peak_rss_mib": peak_rss_mib(),
        "wav": verify_wav(wav),
    }
    print("@@RESULT@@" + json.dumps(out))
    return 0


# --------------------------------------------------------------------------
# parent: orchestrate configurations, sample system memory, optional playback
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


def launch(label: str, sentence: str, vocoder: str, threads: int) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--label", label,
        "--sentence", sentence,
        "--vocoder", vocoder,
        "--threads", str(threads),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@") :])
    raise SystemExit(
        f"{label}: worker produced no result\n--- stdout ---\n{proc.stdout}"
        f"\n--- stderr ---\n{proc.stderr}"
    )


def tegrastats_sample(seconds: int = 3) -> str | None:
    try:
        proc = subprocess.run(
            ["tegrastats", "--interval", "1000"],
            capture_output=True,
            text=True,
            timeout=seconds,
        )
        text = proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        raw = exc.stdout or b""
        text = raw.decode() if isinstance(raw, bytes) else raw
    except Exception:
        return None
    lines = text.strip().splitlines()
    return lines[-1] if lines else None


def asr_round_trip(results: list[dict]) -> dict:
    """Transcribe the synthesized WAVs with the F4 recognizer.

    Nobody can listen to the output in a headless sandbox, so "intelligible" is
    measured instead of asserted: if the F4 FastConformer reads the text back,
    the waveform carries the words. The 22050 Hz TTS output is resampled to the
    recognizer's 16 kHz -- the same conversion F6 will need for the AEC
    reference tap.
    """
    if not (ASR_DIR / "model.int8.onnx").exists():
        return {"available": False, "reason": f"F4 model missing at {ASR_DIR}"}
    try:
        import sherpa_onnx

        rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
            model=str(ASR_DIR / "model.int8.onnx"),
            tokens=str(ASR_DIR / "tokens.txt"),
            num_threads=2,
            provider="cpu",
        )
    except Exception as exc:
        return {"available": False, "reason": f"recognizer load failed: {exc!r}"}

    rows = []
    for r in results:
        path = AUDIO_OUT / r["wav"]["file"]
        with wave.open(str(path), "rb") as wf:
            src_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        n = int(round(len(x) * ASR_RATE / src_rate))
        y = np.interp(
            np.arange(n) * (src_rate / ASR_RATE), np.arange(len(x)), x
        ).astype(np.float32)
        stream = rec.create_stream()
        stream.accept_waveform(ASR_RATE, y)
        rec.decode_stream(stream)
        hyp = stream.result.text
        rows.append(
            {
                "wav": r["wav"]["file"],
                "reference": r["text"],
                "transcript": hyp,
                "wer": wer(r["text"], hyp),
                "cer": cer(r["text"], hyp),
            }
        )
    return {
        "available": True,
        "recognizer": ASR_DIR.name,
        "resampled": f"{results[0]['sample_rate']} -> {ASR_RATE} Hz (linear)",
        "rows": rows,
    }


def safe_playback(wavs: list[Path]) -> dict:
    """Play each WAV exactly once, sequentially, with sidetone forced off.

    The Waveshare/SSS1629 endpoint is resolved by ALSA *name*
    (plughw:CARD=<id>,DEV=0), never a card index, because index numbers move
    between boots. The card's 'Mic' PLAYBACK control is the hardware sidetone
    that previously caused a loud mic-to-speaker feedback loop; it is driven to
    0% and muted, and 'Speaker' is capped, before anything is played. No capture
    is started anywhere in this function -- F2 AEC is not wired to playback yet,
    so duplex stays forbidden until F6.
    """
    report: dict = {"performed": False, "reason": None, "played": []}

    if os.environ.get("JETBOT_F5_PLAYBACK", "0") != "1":
        report["reason"] = "not requested (set JETBOT_F5_PLAYBACK=1 to enable)"
        return report
    if not Path("/dev/snd").exists():
        report["reason"] = "/dev/snd not present (no ALSA access in this environment)"
        return report

    from jetbot_agent.hardware.audio_interface import (
        apply_safe_mixer_baseline,
        mixer_report,
        resolve_sss1629,
    )

    try:
        ident = resolve_sss1629()
    except Exception as exc:
        report["reason"] = f"name resolution failed: {exc!r}"
        return report

    play = ident["alsa_playback"]
    if not play.startswith("plughw:CARD="):
        report["reason"] = f"refusing non-name-resolved endpoint {play}"
        return report

    # Never talk to the speaker while something else is capturing.
    if subprocess.run(["pgrep", "-x", "arecord"], capture_output=True).returncode == 0:
        report["reason"] = "arecord is running; refusing concurrent playback"
        return report

    card = ident["card_index_ephemeral"]
    report["endpoint"] = play
    report["usb_name"] = ident["usb_name"]
    try:
        apply_safe_mixer_baseline(card)
        subprocess.run(
            ["amixer", "-c", str(card), "sset", "Speaker", f"{SPEAKER_PCT}%", "unmute"],
            check=True, capture_output=True, text=True,
        )
        subprocess.run(
            ["amixer", "-c", str(card), "sset", "Mic", "playback", "0%", "mute"],
            check=True, capture_output=True, text=True,
        )
        report["mixer_before"] = mixer_report(card)
    except Exception as exc:
        report["reason"] = f"mixer baseline failed: {exc!r}"
        return report

    try:
        for wav in wavs:
            info = verify_wav(wav)
            if info["clipped"] or info["silent"]:
                report["played"].append({"file": wav.name, "skipped": "failed verify"})
                continue
            subprocess.run(
                ["aplay", "-D", play, str(wav)],
                check=True, capture_output=True, text=True, timeout=30,
            )
            report["played"].append({"file": wav.name, "duration_s": info["duration_s"]})
            time.sleep(0.3)
        report["performed"] = True
    except Exception as exc:
        report["reason"] = f"aplay failed: {exc!r}"
    finally:
        # Kill any lingering aplay, then re-mute so an idle robot cannot ring.
        subprocess.run(["pkill", "-x", "aplay"], capture_output=True)
        subprocess.run(
            ["amixer", "-c", str(card), "sset", "Mic", "playback", "0%", "mute"],
            capture_output=True,
        )
        subprocess.run(
            ["amixer", "-c", str(card), "sset", "Speaker", "0%", "mute"],
            capture_output=True,
        )
        report["mixer_after"] = mixer_report(card)
        report["speaker_remuted"] = True

    return report


def run() -> int:
    if not (ACOUSTIC_DIR / "model-steps-3.onnx").exists():
        raise SystemExit(
            f"missing {ACOUSTIC_DIR}; run scripts/bringup/fetch_tts_models.sh first"
        )
    AUDIO_OUT.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)

    assert SPEAKER_PCT <= SPEAKER_MAX_PCT, "playback level exceeds the safety cap"

    results = []
    with MemSampler() as mem:
        for threads in (1, 2, 4):
            results.append(
                launch(f"hifigan-v2-{threads}t", "robot_stop", "hifigan_v2.onnx", threads)
            )
        results.append(launch("hifigan-v1-2t", "robot_stop", "hifigan_v1.onnx", 2))
        results.append(launch("hifigan-v2-2t-long", "robot_dock", "hifigan_v2.onnx", 2))
        tegra = tegrastats_sample()

    model_sizes = {
        "acoustic_matcha_onnx_mib": (ACOUSTIC_DIR / "model-steps-3.onnx").stat().st_size / 2**20,
        "vocoder_hifigan_v1_mib": (MODELS / "hifigan_v1.onnx").stat().st_size / 2**20,
        "vocoder_hifigan_v2_mib": (MODELS / "hifigan_v2.onnx").stat().st_size / 2**20,
    }

    round_trip = asr_round_trip(results)
    wavs = [AUDIO_OUT / r["wav"]["file"] for r in results]
    playback = safe_playback(wavs)

    report = {
        "gate": "F5 two-stage neural TTS (text -> mel -> HiFi-GAN -> WAV)",
        "architecture": "Matcha-TTS acoustic model + HiFi-GAN vocoder",
        "substitution": (
            "Stage F5 specifies NVIDIA FastPitch as the mel generator. NGC serves "
            "nvidia/nemo/tts_en_fastpitch 1.8.1 (187 MB) and nvidia/nemo/tts_hifigan "
            "1.0.0rc1 (315 MB) without a key, but each version contains exactly one "
            "file -- a .nemo torch archive with no ONNX or TensorRT export. Loading or "
            "exporting one needs nemo_toolkit[tts], which resolves to nemo-toolkit "
            "3.0.0 + torch 2.13.0 + a CUDA 13 wheel stack (cudnn-cu13, cublas 13, "
            "nccl-cu13, cusparselt-cu13, nvshmem-cu13): the wrong CUDA generation for "
            "JetPack 6 / L4T R36.4.4 (CUDA 12.6) and not a Tegra iGPU build, so it "
            "would not reach the GPU anyway. The HiFi-GAN vocoder stage is therefore "
            "kept and only the mel generator is substituted."
        ),
        "runtime": "sherpa-onnx 1.13.6 (bundled onnxruntime), CPU execution provider",
        "precision": "fp32 ONNX",
        "provider": "cpu",
        "note_gpu": (
            "CUDA execution provider not used: no JetPack-6 onnxruntime-gpu wheel is "
            "reachable from the allowlisted sandbox indexes. CPU-only numbers below "
            "are a floor, not a ceiling."
        ),
        "model_source": (
            "k2-fsa/sherpa-onnx GitHub release mirrors: tts-models/"
            "matcha-icefall-en_US-ljspeech (acoustic, LJSpeech) and "
            "vocoder-models/hifigan_v{1,2}.onnx (original jik876 HiFi-GAN weights)"
        ),
        "model_sizes": model_sizes,
        "warm_runs": WARM_RUNS,
        "system_min_mem_available_mib": mem.min_available_mib,
        "tegrastats_sample": tegra,
        "alsa_safety": {
            "endpoint_resolution": "by ALSA name plughw:CARD=<id>,DEV=0, never an index",
            "sidetone": "'Mic' PLAYBACK forced 0% and muted before and after playback",
            "speaker_cap_pct": SPEAKER_MAX_PCT,
            "speaker_used_pct": SPEAKER_PCT,
            "playback_mode": "each file once, sequential, never during capture",
            "post_playback": "pkill -x aplay, then speaker re-muted",
        },
        "live_playback": playback,
        "asr_round_trip": round_trip,
        "results": results,
    }

    payload = json.dumps(report, indent=2)
    LOG.write_text(payload, encoding="utf-8")
    print(payload)

    # --- gate assertions -------------------------------------------------
    for r in results:
        w = r["wav"]
        if w["silent"]:
            raise SystemExit(f"FAIL: {w['file']} is effectively silent")
        if w["clipped"]:
            raise SystemExit(f"FAIL: {w['file']} is clipped (peak {w['peak']:.3f})")
        if w["sample_rate"] != r["sample_rate"] or w["channels"] != 1:
            raise SystemExit(f"FAIL: {w['file']} is not mono at {r['sample_rate']} Hz")
    best = min(results, key=lambda r: r["rtf_warm"])
    if best["rtf_warm"] >= 1.0:
        raise SystemExit(f"FAIL: warm RTF {best['rtf_warm']:.3f} is not real time")
    if round_trip.get("available"):
        worst = max(round_trip["rows"], key=lambda r: r["cer"])
        if worst["cer"] > 0.15:
            raise SystemExit(
                f"FAIL: {worst['wav']} unintelligible to the F4 recognizer "
                f"(CER {worst['cer']:.3f}, WER {worst['wer']:.2f}): "
                f"{worst['transcript']!r}"
            )
        print(
            f"round-trip ASR worst CER {worst['cer']:.3f} "
            f"(WER {worst['wer']:.2f}) on {worst['wav']}"
        )
    print(
        f"F5 TTS ok: best warm RTF {best['rtf_warm']:.3f} ({best['label']}), "
        f"first audio {best['first_audio_warm_s_median']*1000:.0f} ms, "
        f"peak RSS {best['peak_rss_mib']:.0f} MiB, "
        f"live playback {'yes' if playback['performed'] else 'no'}"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--label", default="")
    ap.add_argument("--sentence", default="robot_stop", choices=tuple(SENTENCES))
    ap.add_argument("--vocoder", default="hifigan_v2.onnx")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    return run_worker(args) if args.worker else run()


if __name__ == "__main__":
    raise SystemExit(main())
