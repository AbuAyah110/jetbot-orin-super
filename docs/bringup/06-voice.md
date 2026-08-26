# Stage F — Voice installs and isolated gates

Do not connect voice to the agent loop during Stage F. Agent voice tools are Stage H ticket **I6**: one-shot ASR after F4, one-shot TTS after F5, duplex only after F6. If F1–F4 are still open, I6 stays a no-op stub. Validate each model in its supported NVIDIA runtime on the Jetson Orin Nano Super 8GB before attempting model-specific export or TensorRT optimization. TensorRT-Edge-LLM is not the ASR/TTS runtime.

For every inference gate, record the exact model, precision, software versions, input duration, cold and warm latency, real-time factor where applicable, and peak process plus system unified RAM/VRAM (`tegrastats` is suitable). Hardware measurements establish acceptance limits; do not choose thresholds from desktop results.

## F1 — ALSA identity and safe mixer baseline

1. Identify the Waveshare/Solid State System SSS1629 endpoint by USB/ALSA **device name** using `arecord -l`, `aplay -l`, and `arecord -L`. The observed bring-up endpoint was `plughw:2,0`, but card numbers can change and must never be hardcoded in production.
2. Set capture to a conservative baseline near 80% (hardware maximum is +31 dB), playback low, and hardware sidetone/monitoring **OFF**.
3. Record the resolved capture and playback names and mixer controls with the test evidence.
4. Run only sequential capture-then-playback tests until F2 AEC passes.

**Safety:** hardware sidetone previously caused a dangerous loud feedback loop. Never enable sidetone, microphone monitoring, or simultaneous untreated capture/playback. Start playback at low volume with a physical mute/disconnect available.

Pass: a 16 kHz mono WAV is captured from the name-resolved USB endpoint and played once at a safe level, with no sidetone or simultaneous capture.

## F2 — WebRTC APM AEC/NS/AGC/VAD

Use WebRTC Audio Processing Module as the primary front end. Feed 16 kHz mono audio in 10 ms frames through:

- high-pass filter;
- noise suppression;
- automatic gain control;
- voice activity detection; and
- acoustic echo cancellation using the time-aligned far-end/reference samples actually sent to the speaker.

Create or retain versioned test metadata for clean-speech, noisy-speech, and speaker-echo fixtures. Save unprocessed and processed outputs and the APM configuration. Compare signal level, clipping, speech preservation, noise reduction, echo reduction, CPU load, latency, and peak unified memory. AEC must be tested with the TTS/playback reference path, not with noise suppression alone.

Pass: all three fixtures process without frame discontinuities; speech remains intelligible; measured noise and playback echo are reduced relative to the unprocessed fixtures; no clipping or feedback occurs; resource and latency measurements are recorded.

### Probe 2026-08-25 (offline, no speaker)

Package: `pywebrtc-audio==0.1.0` + `numpy==2.2.6` in `.venv`. Gate: `./scripts/bringup/test_webrtc_apm.sh`. Artifacts under `data/audio/f2/` (gitignored).

| Fixture | Result |
| --- | --- |
| Noise-only (NS, AGC off) | RMS 0.200 → 0.024 (**8.2×** reduction) |
| Echo-only (AEC, AGC off, far-end reference) | RMS 0.140 → 0.00059 (**236×** reduction) |
| Clean speech (full APM) | RMS 0.192 → 0.465 (AGC), peak 0.97, not clipped; speech_probability 0.90 |
| Real-time factor | 0.005–0.016 (faster than real time) |
| Live duplex | **Not** run. Sequential ALSA still required until F6. |

Live capture/playback was not repeated in the sandbox; F1 ALSA identity/mixer from earlier this session still applies.

## F3 — Optional RNNoise A/B benchmark

RNNoise is optional residual denoising after APM. It does **not** cancel acoustic echo and does not replace F2. RNNoise expects 48 kHz audio and fixed 480-sample (10 ms) frames, so a 16 kHz capture pipeline needs resampling around it. Benchmark APM alone against APM + RNNoise using identical fixtures.

Record speech quality, residual noise, end-to-end latency, resampler cost, CPU load, and peak unified memory. Keep RNNoise disabled unless the residual-noise benefit justifies the resampling and latency cost.

Pass: reproducible A/B artifacts and measurements exist; adopting or rejecting RNNoise is documented. This gate is not required for F4 or F5.

## F4 — NVIDIA FastConformer ASR

Install a Jetson-compatible NVIDIA FastConformer model/runtime independently of the production agent. Transcribe one known 16 kHz mono WAV, first without APM and then with the F2 output where useful.

Pass: the expected speech produces a non-empty, intelligible transcript. Record model/precision/runtime, cold and warm latency, audio duration, real-time factor (`inference_seconds / audio_seconds`), CPU/GPU utilization, and peak unified RAM/VRAM. State from measured evidence whether sustained real-time use fits the 8 GB device alongside reserved system capacity; if it does not, keep optimization/model selection open rather than inventing a pass threshold.

### Result 2026-08-25 — PASS (CPU, int8 ONNX)

Gate: `./scripts/bringup/test_fastconformer_asr.sh` (fetches models, then runs the sweep). Report: `data/bringup/f4_fastconformer_asr.json` (local artifact — `data/bringup/**` is gitignored; re-run the gate to regenerate it). Each configuration runs in its own worker process so peak RSS is attributed to that configuration alone.

**Install path.** NeMo was not installed. `nemo_toolkit[asr]` requires PyTorch, and no torch is present in `.venv` or the system interpreter; the JetPack-6 aarch64 CUDA wheels for torch/onnxruntime-gpu are not on any index reachable from the bring-up sandbox. Rather than force a multi-gigabyte, likely-broken stack onto an 8 GB device, F4 uses the NVIDIA NeMo FastConformer encoders **already exported to ONNX**, executed by `sherpa-onnx==1.13.6` (single 4.2 MB wheel plus a 13.1 MB core wheel; bundles its own onnxruntime and the matching NeMo log-mel front end and CTC decoder).

**Model source.** `huggingface.co` is unreachable from the sandbox and the NGC model API returns 401/404 without a key, so the checkpoints come from the `k2-fsa/sherpa-onnx` GitHub release mirror of the NVIDIA exports:

| Model | Archive | ONNX |
| --- | --- | --- |
| `nemo-fast-conformer-ctc-en-24500` (offline) | 104 MB | 132 MB int8 |
| `nemo-streaming-fast-conformer-ctc-en-80ms` (cache-aware streaming) | 99 MB | 132 MB int8 |

**Audio.** Real LibriSpeech utterances shipped inside the archives, with reference transcripts, so word error rate is measured rather than asserted. The robot's **own microphone has not yet fed this gate** — a real captured utterance from the F1 ALSA endpoint is still needed before the F6 duplex pipeline is trusted end to end.

RTF is `warm_median / audio_seconds` over 5 warm runs. The median (not the mean) is the headline because the Jetson was shared during bring-up; `host_loadavg_1m` and min/max are recorded per row so contention stays visible.

| Config | Threads | Audio | Cold (load+infer) | Warm infer (median) | RTF | Cores busy | Load avg | Peak RSS | WER |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| offline CTC | 1 | 6.63 s | 2.36 s (1.92 + 0.45) | 0.438 s | **0.066** | 1.0 | 0.39 | 326 MiB | **0.00** |
| offline CTC | 2 | 6.63 s | 2.24 s (1.93 + 0.31) | 0.296 s | **0.045** | 2.0 | 0.84 | 324 MiB | **0.00** |
| offline CTC | 4 | 6.63 s | 2.48 s | 0.671 s | 0.101 † | 2.0 † | 1.74 | 325 MiB | **0.00** |
| offline CTC, long utterance | 4 | 16.72 s | 4.29 s | 1.667 s | 0.100 † | 2.1 † | 3.98 | 466 MiB | 0.02 |
| streaming CTC 80 ms | 2 | 6.63 s | 4.98 s (2.73 + 2.24) | 2.247 s | **0.339** | 1.9 | 2.48 | 326 MiB | 0.06 |

† **These two rows are not trustworthy as thread-scaling data.** Both 4-thread runs achieved only ~2 cores of actual parallelism and landed *slower* than the 2-thread run, while `host_loadavg_1m` climbed from 0.39 to 3.98 across the sweep — an unrelated workload was using roughly two cores of this shared board. On a quiet board earlier the same 4-thread configuration measured 0.207 s / RTF 0.031 with 3.9 cores busy. The 1- and 2-thread rows reproduced within ±3% across four separate sweeps and are the numbers to rely on; treat 4-thread scaling as unmeasured until it is re-run on an idle board.

Transcript sample (offline, WER 0.00 at every thread count):

```
ref: AFTER EARLY NIGHTFALL THE YELLOW LAMPS WOULD LIGHT UP HERE AND THERE THE SQUALID QUARTER OF THE BROTHELS
hyp: After early nightfall the yellow lamps would light up here and there the squalid quarter of the brothels.
```

The offline model emits punctuation and casing, so WER is computed on case-folded, punctuation-stripped words. The streaming model's 0.06 WER is a single substitution, `brothel` for `brothels`, at the very end of the clip — the tail of the audio is cut off before the cache-aware decoder emits the final token.

**8 GB fit.** Measured, not extrapolated. Peak RSS is ~325 MiB for a 6.6 s utterance and 466 MiB for 16.7 s. Model weights are constant, so the growth is activation and feature buffers proportional to utterance length — capping utterance length caps memory. System `MemAvailable` never fell below 3.86 GiB even with the foreign load present, and `tegrastats` reported 2269/7620 MB RAM, `GR3D_FREQ 0%`, and **0 MB of the 32 GiB swap touched**. Sustained real-time ASR fits with a wide margin.

**Runtime choice for the agent.** Use the **offline CTC model at 2 threads**: RTF 0.045 (22× faster than real time) on two cores, leaving four for the camera, motor watchdog, and agent loop. This is also the configuration that measured most reliably under real system load, which matters more for a robot than a best-case number.

The streaming model costs about 7.5× more per second of audio (RTF 0.339) because cache-aware chunking re-runs the encoder every 80 ms. Its value is partial hypotheses *during* speech, so it belongs in F6 duplex, not in one-shot ASR. For Stage H ticket I6, use the offline model.

**F2 APM cross-check.** Running the APM output through the recognizer gives WER 0.00 and RTF 0.032, identical to the unprocessed clean fixture — the expected result, confirming the APM does not damage clean speech ahead of the recognizer. Its actual value is on the noisy and echo fixtures measured in F2. This row is **absent from the committed JSON**: the working-tree copy of `jetbot_agent/audio/audio_preprocessor.py` is currently reverted to a `NotImplementedError` stub by an unrelated in-flight edit, so the gate records `apm_cross_check` as unavailable and skips the row rather than silently dropping it. The measurement above was taken against the committed (`HEAD`) F2 implementation. The row returns automatically once the working tree has a functional F2 module.

**Open items.**

- No CUDA execution provider: no JetPack-6 `onnxruntime-gpu` wheel is reachable from the allowlisted indexes. Every number here is CPU-only and therefore a floor, not a ceiling.
- The robot's own microphone has not fed this gate. Capture a real utterance from the F1 ALSA endpoint and re-run before trusting F6 end to end.
- Re-run the thread sweep on an idle board to get real 4-thread scaling.

Per the stage rules, optimization (TensorRT export, GPU provider) stays open rather than becoming a requirement — F4 already passes with margin at 2 threads.

## F5 — NVIDIA FastPitch + HiFi-GAN TTS

Run the stages explicitly: text → FastPitch mel spectrogram → HiFi-GAN waveform → WAV. Validate the generated file before playback, then perform one low-volume, one-shot `aplay` with capture stopped and sidetone off.

Pass: both model stages complete and the WAV is intelligible without clipping. Record model/precision/runtime, cold and warm text-to-WAV latency, generated audio duration, synthesis real-time factor (`inference_seconds / generated_audio_seconds`), CPU/GPU utilization, and peak unified RAM/VRAM. Decide feasibility on measured 8 GB Orin results; optimize/export only if supported and needed.

## F6 — Duplex voice pipeline and watchdog

Attempt duplex operation only after F2 passes. Route playback samples to both ALSA and the APM far-end/reference input with measured delay alignment. Pipeline:

`ALSA capture → WebRTC APM → VAD → FastConformer → agent boundary → FastPitch → HiFi-GAN → reference tap → ALSA playback`

Add a feedback watchdog that immediately mutes playback and stops capture/playback on sustained clipping, runaway level, missing/stale AEC reference, processing backlog, or operator stop. Begin with the speaker physically separated or at minimum safe volume.

Pass: repeated short turn-taking and interruption tests complete without feedback, runaway volume, clipping, stale-reference operation, or unbounded queues. Record end-to-end latency and peak unified RAM/VRAM under the combined pipeline; acceptance requires measured resource headroom on the 8 GB device, with the final budget set from these measurements.
