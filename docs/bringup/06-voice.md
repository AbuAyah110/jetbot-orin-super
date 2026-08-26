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

### Result 2026-08-25 — PASS with a documented substitution (CPU, fp32 ONNX)

Gate: `./scripts/bringup/test_matcha_hifigan_tts.sh` (fetches models, then runs the sweep). Report: `data/bringup/f5_matcha_hifigan_tts.json` (local artifact — `data/bringup/**` is gitignored; re-run the gate to regenerate it). Each configuration runs in its own worker process so peak RSS is attributed to that configuration alone.

**Substitution: the mel generator is Matcha-TTS, not FastPitch. The vocoder is genuinely HiFi-GAN.** The two-stage shape the gate asks for is preserved — text → mel spectrogram → HiFi-GAN → waveform — and only the acoustic model differs. This was not a silent swap; the FastPitch path was chased to the point of failure first:

| Attempt | Result |
| --- | --- |
| `nvidia/nemo/tts_en_fastpitch` 1.8.1 via `api.ngc.nvidia.com` | **Reachable, 200, 187,023,360 bytes.** No key needed. |
| `nvidia/nemo/tts_hifigan` 1.0.0rc1 via `api.ngc.nvidia.com` | **Reachable, 200, 315,386,678 bytes.** No key needed. |
| Either version's file listing | `totalFileCount: 1` — a single `.nemo` archive. **No ONNX and no TensorRT plan exists on NGC.** |
| `huggingface.co/nvidia/tts_en_fastpitch` | Blocked: `HTTP 403 from proxy after CONNECT`. |

So the weights are downloadable but not *runnable*: a `.nemo` is a torch checkpoint, and loading or exporting one requires `nemo_toolkit[tts]`. A dry-run resolve of that extra on this interpreter produces **161 packages**, headed by `nemo-toolkit 3.0.0`, `torch 2.13.0`, `triton 3.7.1`, and a complete **CUDA 13** wheel stack (`nvidia-cudnn-cu13`, `nvidia-cublas 13`, `nvidia-nccl-cu13`, `nvidia-cusparselt-cu13`, `nvidia-nvshmem-cu13`, `cuda-toolkit 13`). Summing the PyPI wheel sizes for just 26 of those 161 packages is **3.37 GB**. Two independent reasons not to install it:

- **Wrong CUDA generation.** JetPack 6 / L4T R36.4.4 is CUDA 12.6. These are CUDA 13 wheels built for SBSA discrete GPUs, not Tegra iGPU builds, so they would not reach the Orin's GPU even after the download — the same wall F4 hit looking for `onnxruntime-gpu`.
- **Version skew.** The checkpoint was published in 2022 against NeMo 1.8; the only resolvable toolkit is NeMo 3.0.

Spending 3.37 GB on an 8 GB device to obtain a CPU-only, version-skewed FastPitch is a worse engineering outcome than keeping the HiFi-GAN vocoder and substituting a mel generator that is already exported to ONNX. Revisit if a Jetson-index `nemo_toolkit` or an NVIDIA-published FastPitch ONNX becomes reachable.

**Install path.** Nothing new was installed. The gate reuses the `sherpa-onnx==1.13.6` already in `.venv` from F4, which supports Matcha-style TTS with an external vocoder and bundles the espeak-ng G2P front end. Checkpoints come from the `k2-fsa/sherpa-onnx` GitHub release mirrors:

| Stage | Model | ONNX size |
| --- | --- | --- |
| Acoustic (text → mel) | `matcha-icefall-en_US-ljspeech`, LJSpeech, 1 female speaker | 70.7 MiB |
| Vocoder (mel → wave) | `hifigan_v2.onnx` (original jik876 HiFi-GAN weights) | **3.6 MiB** |
| Vocoder alternative | `hifigan_v1.onnx` | 53.2 MiB |

Output is 22050 Hz mono. `first audio` is the time to synthesize the **first sentence only** (`max_num_sentences=1`), which is the number that matters for a robot: it is when the speaker can legitimately start talking while the rest is still being generated. RTF is `total_warm_median / generated_audio_seconds` over 5 warm runs; median, not mean, for the same shared-board reason as F4.

| Config | Vocoder | Threads | Text | Audio | Cold (load+all) | First audio (warm) | Total warm | RTF | Cores busy | Load avg | Peak RSS | Peak |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v2, 1 thread | hifigan_v2 | 1 | 2 sentences | 2.81 s | 4.38 s | 0.626 s | 1.061 s | 0.377 | 1.0 | 1.05 | 163 MiB | 0.49 |
| **v2, 2 threads** | hifigan_v2 | 2 | 2 sentences | 2.81 s | 3.62 s | **0.349 s** | 0.602 s | **0.214** | 2.1 | 1.20 | 163 MiB | 0.47 |
| v2, 4 threads | hifigan_v2 | 4 | 2 sentences | 2.81 s | 3.26 s | 0.220 s | 0.389 s | **0.138** | 4.4 | 1.43 | 162 MiB | 0.42 |
| v1, 2 threads | hifigan_v1 | 2 | 2 sentences | 2.81 s | 8.31 s | 2.060 s | 3.478 s | **1.237** ✗ | 2.0 | 1.64 | 241 MiB | 0.48 |
| v2, 2 threads, long | hifigan_v2 | 2 | 3 sentences | 6.40 s | 4.43 s | 0.428 s | 1.326 s | 0.207 | 2.1 | 1.62 | 175 MiB | 0.46 |

Test sentences, both robot-relevant, written to `data/audio/f5/` (gitignored):

- `robot_stop_*.wav` — "Obstacle detected. Stopping now." → 2.81 s
- `robot_dock_hifigan_v2_2t.wav` — "Battery at twelve percent. Returning to the charging dock. Please clear a path ahead." → 6.40 s

Every WAV was validated offline before any playback was considered: mono, 16-bit, 22050 Hz, peak 0.42–0.49 (**not clipped**, threshold 0.999), RMS 0.047–0.061 (**not silent**). The gate fails itself on a clipped, silent, or wrong-rate file.

**Intelligibility was measured, not asserted.** Nobody can listen in a headless sandbox, so the gate feeds each synthesized WAV back through the **F4 FastConformer recognizer** (resampled 22050 → 16000 Hz) and scores the transcript against the input text. Round-trip results:

| WAV | CER | WER | Transcript |
| --- | --- | --- | --- |
| `robot_stop_hifigan_v2_{1,2,4}t.wav` | **0.000** | **0.00** | `obstacle detected. Stopping now,` |
| `robot_dock_hifigan_v2_2t.wav` | **0.012** | 0.14 | `battery at twelve per cent. Returning to the charging dock. Please clear a path ahead,` |
| `robot_stop_hifigan_v1_2t.wav` | 0.067 | 0.25 | `obstacle detected. stotopping now,` |

Scores are computed on case-folded, punctuation-stripped words, as in F4, so the recognizer's trailing comma is not an error. The `robot_dock` row is the recognizer splitting `percent` into `per cent`; the audio is correct. That single token is 14% of a seven-word reference but 1.2% of its characters, which is why **the gate decides on CER (≤ 0.15) and records WER for continuity with F4** — on a four-word phrase like "Obstacle detected. Stopping now." one substitution is already WER 0.25, so WER cannot tell a softened consonant from unintelligible speech.

**Matcha sampling is stochastic** (`noise_scale`), so every invocation produces slightly different audio and the round-trip score moves; the table is one run. Across four sweeps the worst row per run ranged **CER 0.012–0.067**, the maximum being the `stotopping` sample above. Note that this row is **WER 0.25 while being plainly intelligible** — it would have failed a WER ≤ 0.2 gate. That is the concrete reason the threshold is on CER; do not tighten it back to WER without accounting for the four-word phrases.

**Sentences must not be butt-spliced — this was a real bug caught by the round-trip check.** Synthesizing per sentence and concatenating with no gap let the tail of "detected." collide with the plosive onset of "Stopping", and the recognizer heard **"Sopping" in 4 of 8 runs**. The same sentence synthesized alone was correct **12 of 12**, so it is a splice artifact, not a model defect. Inserting **200 ms of silence** between sentences restored 8 of 8 and reads as normal prosody; 100 ms was still wrong 2 of 8. `INTER_SENTENCE_GAP_S = 0.20` in the gate. F6 must keep this gap when it streams sentence-by-sentence, and the same silence has to be present in the AEC reference tap.

**The vocoder choice is the load-bearing result.** HiFi-GAN **v1 is unusable on this box: RTF 1.237, slower than real time**, and it costs 1.5× the peak RSS. v2 at the same thread count is **5.8× faster** for a model **15× smaller**, and v1 scored no better on the round trip, so it buys nothing measurable here. Anyone tempted to "upgrade" to v1 for quality should re-measure first. The 4-thread row is real scaling (4.4 cores actually busy), unlike F4's contended 4-thread rows.

**8 GB fit.** Measured, not extrapolated. Peak RSS is 162–175 MiB with v2 — roughly half of F4's ASR. Growth from 2.81 s to 6.40 s of output was 163 → 175 MiB, so memory tracks generated length and capping utterance length caps memory. System `MemAvailable` never fell below **5096 MiB**, and `tegrastats` reported `RAM 2243/7620MB`, `GR3D_FREQ 0%`, and **0 MB of the 32 GiB swap touched**. Sustained real-time TTS fits with a very wide margin.

**Runtime choice for the agent.** Use **`hifigan_v2` at 2 threads** for Stage H ticket I6: RTF 0.214 and first audio in **349 ms** on two cores. Paired with F4's offline CTC at 2 threads, ASR + TTS together occupy 4 of 6 cores and about 488 MiB, leaving two cores for the camera, motor watchdog, and agent loop. The 4-thread configuration is faster in isolation but would contend with the recognizer.

**ALSA safety rules used (F1 discipline, unchanged).** The SSS1629's `Mic` **playback** control is the hardware sidetone that previously caused a dangerous mic-to-speaker feedback loop, so the gate treats playback as the hazardous operation:

- The endpoint is resolved by **ALSA name** through `resolve_sss1629()` → `plughw:CARD=Device,DEV=0`. The card index (currently 2) is read only to address `amixer` and is never written into a device string; the helper refuses any endpoint that is not `plughw:CARD=`.
- `Mic` playback is driven to **0% and muted** before playback and again afterward.
- `Speaker` is capped at **40%** and actually set to **20%**; the gate asserts the cap before it starts.
- Each file is played **exactly once, sequentially**, never concurrently with capture — the gate checks `pgrep -x arecord` and refuses if a recorder is live. No `arecord` is ever started in this gate.
- Afterward: `pkill -x aplay` for any lingering player, then the speaker is **re-muted** so an idle robot cannot ring.
- Playback is opt-in via `JETBOT_F5_PLAYBACK=1` and self-disables if `/dev/snd` is absent.

**Live playback did NOT happen.** `/proc/asound/cards` confirms the card is present and name-resolvable (`2 [Device]: USB-Audio - USB PnP Audio Device`, `Solid State System Co.,Ltd.`), but `/dev/snd` is not exposed to the bring-up sandbox, so `aplay` could not run. The WAVs were verified offline instead, as recorded above. **The F5 pass is therefore on synthesis and file validation only; the "one low-volume `aplay`" half of the gate is still owed** and must be run on a session with `/dev/snd` before F5 is considered fully closed.

**Open items.**

- Run the one-shot low-volume `aplay` with `/dev/snd` available to close the playback half of the gate. Confirm intelligibility by ear.
- **Sample-rate mismatch ahead of F6.** TTS emits 22050 Hz; capture, the F2 APM, and F4 ASR are all 16 kHz. The F6 reference tap that feeds playback into the APM far-end input **must resample 22050 → 16000** and account for that resampler in the delay alignment. AEC silently fails on a mis-rated reference, which is exactly the condition that produces feedback.
- The voice is LJSpeech (single female speaker), not an NVIDIA voice. Acceptable for robot status phrases; revisit if a specific voice is required.
- No CUDA execution provider, same as F4. Every number here is CPU-only and therefore a floor, not a ceiling.

Per the stage rules, optimization (TensorRT export, GPU provider) stays open rather than becoming a requirement — F5 already passes with margin at 2 threads.

## F6 — Duplex voice pipeline and watchdog

Attempt duplex operation only after F2 passes. Route playback samples to both ALSA and the APM far-end/reference input with measured delay alignment. Pipeline:

`ALSA capture → WebRTC APM → VAD → FastConformer → agent boundary → FastPitch → HiFi-GAN → reference tap → ALSA playback`

Add a feedback watchdog that immediately mutes playback and stops capture/playback on sustained clipping, runaway level, missing/stale AEC reference, processing backlog, or operator stop. Begin with the speaker physically separated or at minimum safe volume.

Pass: repeated short turn-taking and interruption tests complete without feedback, runaway volume, clipping, stale-reference operation, or unbounded queues. Record end-to-end latency and peak unified RAM/VRAM under the combined pipeline; acceptance requires measured resource headroom on the 8 GB device, with the final budget set from these measurements.
