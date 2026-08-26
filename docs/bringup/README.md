# Staged bring-up (Orin Nano Super)

Install and **verify one stage at a time**. Do not start the next stage until the gate command passes.

Issue tracker: [`TASKBOARD.md`](../../TASKBOARD.md) · spec: [`JETBOT_SPEC.md`](../../JETBOT_SPEC.md)

**Order after TensorRT:** agent integration **before** memory. Ticket IDs I1–I8 are agent **integration** slices (not Stage I). Stage I is memory, and it starts only after I8’s integration loop exists without memory tools.

| Stage | Doc | Gate |
| --- | --- | --- |
| A OS / env | [01-os.md](01-os.md) | `./scripts/diagnostics.sh` |
| B I2C / motors | [02-i2c-motors.md](02-i2c-motors.md) | `./scripts/bringup/probe_i2c.sh` then wheels-up `test_motors.py` |
| C CSI camera | [03-csi-camera.md](03-csi-camera.md) | `./scripts/bringup/test_csi_camera.sh` |
| D Audio | [04-audio.md](04-audio.md) | `./scripts/bringup/test_alsa.sh` |
| E Python skeleton | [05-python-skeleton.md](05-python-skeleton.md) | `./scripts/bringup/test_python_skeleton.sh` |
| F Voice | [06-voice.md](06-voice.md) · F3 evidence: [06b-f3-rnnoise.md](06b-f3-rnnoise.md) | safe ALSA → WebRTC APM → FastConformer + Matcha/HiFi-GAN → guarded duplex |
| G Model runtimes | [07-tensorrt.md](07-tensorrt.md) · G1 evidence: [07-tensorrt-g1.md](07-tensorrt-g1.md) · Edge-LLM evaluation: [07b-tensorrt-edge-llm.md](07b-tensorrt-edge-llm.md) | dummy I/O per runtime (isolated) |
| H Agent (I1–I8) | [08-agent.md](08-agent.md) · design notes: [09-agent-i1-i2.md](09-agent-i1-i2.md), [09b-agent-i5-navigation.md](09b-agent-i5-navigation.md), [09c-agent-i3-i4.md](09c-agent-i3-i4.md) | each I* ticket’s gate; LLM never PWM |
| I Memory | [09-memory.md](09-memory.md) | Chroma upsert/query + SQLite put/get |

Memory **tools** (Chroma/SQLite wrappers for the harness) are a later subtask after Stage I, not part of I1–I8.

Safety: `config/robot.yaml` stays `backend: mock` until Stage B is signed off. LLMs never set PWM. Motion only via the limited motor path (`/cmd_vel` or equivalent + watchdog). See [`docs/safety.md`](../safety.md).

## Status on this Jetson (2026-08-25)

Lightweight re-check only — no OS reinstall, no motor PWM.

| Stage | Result | Evidence |
| --- | --- | --- |
| A | **Pass with notes** | L4T R36.4.4, NVMe `/`, MAXN_SUPER, `multi-user.target`, 32 GiB swap. Swap path is `/ssd/32GB.swap` (not `/swapfile`); `vm.swappiness=60` (spec target 10). Reconcile the spec to the board rather than recreating swap — see [01-os.md](01-os.md). |
| B | **Pass (probe)** | I2C buses present; bus 7 shows `0x70`/`0x60`/`0x3c`. PWM **not** re-run. Prior motion: `notebooks/basic_motion/basic_motion.ipynb`. |
| C | **Pass** | `nvarguscamerasrc` 1-frame EOS; prior live preview: `notebooks/camera/csi_camera_test.ipynb`. |
| D | **Audio HW verified** | Waveshare SSS1629. Identify by ALSA **name**, never card index. Sidetone **off**. Sequential capture then playback until F2 AEC. |
| E | **Pass** | `.venv` via `virtualenv` (no `python3-venv` apt). PyYAML 6.0.3. `test_python_skeleton.sh` ok. |
| F | **F1, F2, F3, F4, F5 pass — only F6 open** | F1 name-resolved ALSA mixer. F2 `pywebrtc-audio` offline NS 8.2× / AEC 236×. **F3 verdict: reject RNNoise** — it denoises better and hears worse, raising ASR WER 0.006 → 0.253, so WebRTC APM stays the only front end. F4 FastConformer CTC int8 ONNX via `sherpa-onnx`, RTF 0.045 @ 2 threads, WER 0.00, 324 MiB peak. **F5 pass with a substitution:** Matcha-TTS mel + genuine HiFi-GAN v2 vocoder instead of FastPitch (no FastPitch ONNX exists on NGC), first audio 349 ms, RTF 0.214, 163 MiB peak; **one low-volume `aplay` still owed** because `/dev/snd` is not exposed to the sandbox. |
| G | **G1 pass; stage rescoped** | TensorRT 10.3.0.30 / CUDA 12.6.11 / cuDNN 9.3.0 healthy, three engines built and numerically verified. TensorRT-LLM absent with no Tegra wheel; **PyTorch absent and now a prerequisite ticket ahead of G3/G4/G5**; first VLM gate goes through **llama.cpp + GGUF**. **TensorRT Edge-LLM — the spec's original runtime — is real and on-matrix for this board** (G5), blocked on PyTorch and a `sm_87`/CUDA-12 CuTe DSL artifact. |
| H | **I1–I5 pass; I6–I8 open** | Harness state machine, structural tool-safety boundary, vision, search, and navigation tools all land as pure software with the mock backend. I6 needs F6 for duplex; I7 needs Stage G. |
| I | After H | Memory stores after the agent loop skeleton exists. |

Voice stack: **FastConformer** ASR + **Matcha-TTS / HiFi-GAN v2** TTS + **WebRTC APM** front end (required; **RNNoise rejected**). TensorRT is not the ASR/TTS runtime — see [07-tensorrt.md](07-tensorrt.md).

**Correction (2026-08-26):** this page previously stated there is "no NVIDIA product called 'TensorRT Edge-LLM.'" That was wrong. [`NVIDIA/TensorRT-Edge-LLM`](https://github.com/NVIDIA/TensorRT-Edge-LLM) is real, active, and lists this board's platform (Jetson Orin / JetPack 6.2+ / CUDA 12.6) as `Compatible`. It is a **different project from TensorRT-LLM**, which is the one genuinely absent here. Evidence and remaining blockers: [07b-tensorrt-edge-llm.md](07b-tensorrt-edge-llm.md).

**Ahead of F6:** TTS emits 22050 Hz while capture, the APM, and ASR are all 16 kHz, so the AEC reference tap **must resample 22050 → 16000** or echo cancellation fails silently. Keep the 200 ms inter-sentence gap F5 found, in both the playback stream and the reference tap.
