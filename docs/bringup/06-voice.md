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

## F3 — Optional RNNoise A/B benchmark

RNNoise is optional residual denoising after APM. It does **not** cancel acoustic echo and does not replace F2. RNNoise expects 48 kHz audio and fixed 480-sample (10 ms) frames, so a 16 kHz capture pipeline needs resampling around it. Benchmark APM alone against APM + RNNoise using identical fixtures.

Record speech quality, residual noise, end-to-end latency, resampler cost, CPU load, and peak unified memory. Keep RNNoise disabled unless the residual-noise benefit justifies the resampling and latency cost.

Pass: reproducible A/B artifacts and measurements exist; adopting or rejecting RNNoise is documented. This gate is not required for F4 or F5.

## F4 — NVIDIA FastConformer ASR

Install a Jetson-compatible NVIDIA FastConformer model/runtime independently of the production agent. Transcribe one known 16 kHz mono WAV, first without APM and then with the F2 output where useful.

Pass: the expected speech produces a non-empty, intelligible transcript. Record model/precision/runtime, cold and warm latency, audio duration, real-time factor (`inference_seconds / audio_seconds`), CPU/GPU utilization, and peak unified RAM/VRAM. State from measured evidence whether sustained real-time use fits the 8 GB device alongside reserved system capacity; if it does not, keep optimization/model selection open rather than inventing a pass threshold.

## F5 — NVIDIA FastPitch + HiFi-GAN TTS

Run the stages explicitly: text → FastPitch mel spectrogram → HiFi-GAN waveform → WAV. Validate the generated file before playback, then perform one low-volume, one-shot `aplay` with capture stopped and sidetone off.

Pass: both model stages complete and the WAV is intelligible without clipping. Record model/precision/runtime, cold and warm text-to-WAV latency, generated audio duration, synthesis real-time factor (`inference_seconds / generated_audio_seconds`), CPU/GPU utilization, and peak unified RAM/VRAM. Decide feasibility on measured 8 GB Orin results; optimize/export only if supported and needed.

## F6 — Duplex voice pipeline and watchdog

Attempt duplex operation only after F2 passes. Route playback samples to both ALSA and the APM far-end/reference input with measured delay alignment. Pipeline:

`ALSA capture → WebRTC APM → VAD → FastConformer → agent boundary → FastPitch → HiFi-GAN → reference tap → ALSA playback`

Add a feedback watchdog that immediately mutes playback and stops capture/playback on sustained clipping, runaway level, missing/stale AEC reference, processing backlog, or operator stop. Begin with the speaker physically separated or at minimum safe volume.

Pass: repeated short turn-taking and interruption tests complete without feedback, runaway volume, clipping, stale-reference operation, or unbounded queues. Record end-to-end latency and peak unified RAM/VRAM under the combined pipeline; acceptance requires measured resource headroom on the 8 GB device, with the final budget set from these measurements.
