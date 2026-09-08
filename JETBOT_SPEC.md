# JETBOT_SPEC.md: Master Specification & Cursor Implementation Plan

Master specification for the **Autonomous Evolving JetBot** built on the **NVIDIA Jetson Orin Nano Super (8GB)**. This document contains the target architecture, directory layout, and hardware protocols.

**Bring-up vs production:** Follow [`docs/bringup/README.md`](docs/bringup/README.md) on this Jetson. Track work on [`TASKBOARD.md`](TASKBOARD.md). Historical ROS / Cosmos / EmbeddingGemma notes remain in [`PROJECT_PLAN.md`](PROJECT_PLAN.md) until they are explicitly merged.

**Bring-up order:** A OS → B I2C/motors → C CSI → D ALSA → E Python skeleton → F voice → G TensorRT dummy engines → **H agent integration (tickets I1–I8)** → **I memory**. Agent comes **before** memory. I1–I8 are integration slices (Stage H), not Stage I. Memory tools wait until after Stage I. Do not implement the full VLM/VLA loop until the gates for the tickets you are wiring have passed. Safety: LLMs never PWM ([`docs/safety.md`](docs/safety.md)).

---

## Architecture deltas (do not silently mix)

| Topic | This spec (target) | Current repo (as-is) |
| --- | --- | --- |
| Motors | PCA9685 `/dev/i2c-1` @ `0x40`, later `hardware/motor_controller.py` | Classic HAT often **bus 7**, addr **`0x70`/`0x60`**; ROS `jetbot_base` + watchdog |
| Safety | Direct I2C in the final tree | LLMs never PWM; `/cmd_vel` + limits ([`docs/architecture.md`](docs/architecture.md)) |
| Models | **llama.cpp + GGUF** Qwen2.5-VL-3B; smolvla and Nemotron embed as PyTorch — see [runtime reality](#measured-runtime-reality-2026-08-26) | llama.cpp Cosmos, Qwen3.5-0.8B, EmbeddingGemma |
| Voice | **Sherpa-ONNX Zipformer ASR + Piper VITS TTS**, int8 CPU models in one process; WebRTC APM front end **required**, RNNoise **rejected** | F1/F2/F3 historical gates retained; compact F4/F5 replacement passed; F6 open |
| Swap | 32 GB `/swapfile`, `vm.swappiness=10` | This Jetson: 32 GiB `/ssd/32GB.swap`, swappiness 60 — **adopt the as-is column**, see [Stage A notes](docs/bringup/01-os.md) |

**Bring-up rule:** probe I2C buses **1 and 7** and record the real address map. Do not assume `0x40` until it appears. Wheel tests only with wheels off the ground; every motion script must hard-stop on timeout.

---

## 1. System Hardware & Environment Configuration

* **Compute Unit:** NVIDIA Jetson Orin Nano Super (8 GB LPDDR5 Unified Memory)
* **Storage:** 1 TB NVMe SSD mounted as primary Root (`/`)
* **Virtual Memory:** 32 GB NVMe swap. The board uses `/ssd/32GB.swap` with `vm.swappiness=60`; this spec originally called for `/swapfile` with `vm.swappiness=10`. **Keep the board as it is and amend this spec** — the swap is already the right size on the right device and is in `fstab`, and re-running `scripts/setup_swap.sh` would add a second 32 GiB file plus a duplicate `fstab` entry. See [Stage A notes](docs/bringup/01-os.md).
* **Operating System:** Headless JetPack 6.x / 7.x (GUI disabled: `sudo systemctl set-default multi-user.target`)
* **Vision Sensor:** Raspberry Pi Camera v2.1 (IMX219) on CSI Port 0 (connected via 15-to-22 pin FPC adapter cable)
* **Audio Hardware:** Waveshare USB to Audio Module (SSS1629 ALSA codec)
  * **Mic Input:** Onboard MEMS / 3.5mm jack (16 kHz sampling via ALSA)
  * **Speaker Output:** Onboard header / 3.5mm audio jack (driven via ALSA `aplay`)
  * **Identity:** select the ALSA endpoint by its USB device name, never a fixed card index
* **Motor Control:** Spec target PCA9685 on `/dev/i2c-1` @ `0x40`. **This board (2026-08-25 probe):** bus **7** has `0x70` / `0x60` / OLED `0x3c`; bus **1** has kernel-claimed `UU` at `0x40` and `0x25`. Classic notebooks used bus 7 @ `0x70`.
* **Inference Engines:** base **TensorRT 10.3.0.30** on CUDA 12.6.11 / cuDNN 9.3.0, verified by an end-to-end engine build. There is **no NVIDIA product called "TensorRT Edge-LLM"**; earlier revisions of this spec named one and every reference has been corrected. TensorRT-LLM — the nearest real thing — is **not installed and has no Tegra aarch64 wheel**. The VLM runtime is **llama.cpp + GGUF**. Voice models are validated in their own runtimes first; TensorRT export stays a later, model-specific experiment rather than an assumed capability. Full inventory and smoke-test evidence: [Stage G1 notes](docs/bringup/07-tensorrt-g1.md).

### Voice architecture

The primary real-time audio front end is **WebRTC Audio Processing Module (APM)**. Process 16 kHz mono audio in 10 ms frames with high-pass filtering, noise suppression, automatic gain control, voice activity detection, and acoustic echo cancellation. AEC must receive the time-aligned far-end/reference audio sent to TTS playback because the robot's speaker is close to its microphone.

**RNNoise was benchmarked and rejected (F3, 2026-08-26). WebRTC APM is the only front end.** RNNoise wins every signal metric — 270× noise reduction against the APM's 8.2×, +4.97 dB segmental SNR against +0.76 dB — and still makes the robot worse at hearing: placed behind the APM as a residual denoiser it raises FastConformer WER on real noisy speech from **0.006 to 0.253**, peaking at **0.889** on the fan-hum-plus-motor-whine fixture that most resembles this robot's noise floor. It also costs 15× the APM's CPU and adds ~52 ms of buffering. Do not pin it in `jetbot_agent/requirements.txt`. It was never an AEC candidate and the measurements confirm it cannot be one: on the echo fixture the APM cancels 2137× against RNNoise's 4.5×. Evidence: [F3 notes](docs/bringup/06b-f3-rnnoise.md).

ASR and TTS are staged independently, proven one file at a time, with latency, real-time factor, and peak unified RAM/VRAM recorded on the 8 GB Orin. Full duplex is forbidden until AEC passes.

**Current ASR default:** `sherpa-onnx-zipformer-small-en-2023-06-26`, using its int8 encoder/decoder/joiner through `sherpa-onnx==1.13.6`, CPU only, 2 threads. The combined Zipformer+Piper gate transcribes its short generated fixture at WER 0.00 and RTF 0.04.

**Current TTS default:** `vits-piper-en_US-lessac-low-int8`, a single Piper VITS graph through the same Sherpa-ONNX runtime, CPU only, 2 threads. It emits 16 kHz audio directly, removing the old Matcha/HiFi-GAN two-stage path and its 22.05→16 kHz AEC-reference resampler.

The old FastConformer and Matcha/HiFi-GAN results remain historical evidence in the Stage F notes. Their model files and fetch/test scripts are no longer part of the active stack. No NeMo or PyTorch package was installed or retained for voice.

The measured one-process gate peaked at **less than 200 MiB RSS** with both C++ objects resident and reported **0 MiB GPU/VRAM use**. The exact decimal-MB value and run-to-run allocator variation are recorded in [the Stage F notes](docs/bringup/06-voice.md); do not restate this as a strict sub-200,000,000-byte guarantee.

### Measured runtime reality (2026-08-26)

Probed on device in Stage G1. This subsection is the authority on what exists; where it disagrees with an aspiration elsewhere in this spec, it wins. Full evidence: [Stage G1 notes](docs/bringup/07-tensorrt-g1.md).

**Installed and healthy:**

| Component | Version | Note |
| --- | --- | --- |
| TensorRT | **10.3.0.30** | built and ran three engines with numerics verified against a NumPy recompute |
| `trtexec` | v100300 | `/usr/src/tensorrt/bin/trtexec` — **not on the default `PATH`** |
| CUDA toolkit | **12.6.11** | `nvcc` 12.6.68, also **not on `PATH`**; GPU arch SM 87 |
| cuDNN | **9.3.0** | |

Engine *building* peaked at about **1.5 GB RSS for a 68 KB graph**, so on 8 GB shared memory it must be budgeted separately from engine *running* — build when nothing else large is resident. The repo `.venv` was created without system site packages and cannot see the apt TensorRT bindings; the gate bridges with `PYTHONPATH=/usr/lib/python3.10/dist-packages`.

**Absent, and each absence blocks something:**

| Missing | Consequence |
| --- | --- |
| **PyTorch / torchvision** | **Hard prerequisite for G3 and G4**, which are both blocked until it exists. It cannot come from PyPI — Jetson needs NVIDIA's wheel index or a `jetson-containers` image matched to CUDA 12.6 / L4T R36. This has its own ticket, sequenced **ahead of G3/G4**. |
| **TensorRT-LLM** | No apt package, no module, and **no Tegra aarch64 wheel** — the aarch64 wheels on `pypi.nvidia.com` are SBSA Grace-class and their TensorRT dependency fails with `TensorRT does not currently build wheels for Tegra systems`. Jetson support lives on the `v0.12.0-jetson` branch, which **targets JetPack 6.1 on AGX Orin 64 GB**, not this Orin Nano 8 GB. |
| **AutoAWQ kernels** | **INT4 AWQ will not load on Jetson aarch64.** TensorRT 10.3 does expose `DataType.INT4`, but that is weight-only quantization driven by explicit Q/DQ nodes in an ONNX graph — not an AWQ checkpoint loader, and it provides no LLM serving layer. Re-quantizing on-device is also out with no PyTorch, `transformers`, or `modelopt`. |
| `onnxruntime` (Python) | The voice stack runs ORT on **CPU** via the copy `sherpa-onnx` vendors, which has no CUDA, cuDNN, or TensorRT linkage at all. Stage F numbers are CPU floors, not ceilings. |
| OpenCV in `.venv` | `save_frame()` falls back to uncompressed PPM. Install the `.[vision]` extra. |

**Stage G is rescoped from "build three TensorRT engines" to "stand up a runtime that can execute these models at all,"** with TensorRT export as a separate later optimization. None of the three named models is published as a TensorRT engine:

* **G2 — Qwen2.5-VL-3B.** The decided path is **llama.cpp + GGUF**: a ~2.0 GB Q4 text backbone plus a **mandatory ~1.25 GB F16 `mmproj` vision encoder**, which is not quantizable without visible degradation, with the vision tower loaded on demand. A **CUDA-enabled llama.cpp for Tegra must be compiled locally** — no llama.cpp binary exists on this host — and vision support for this architecture has needed recent or forked llama.cpp, so pin and verify the build. Against ~4.8–5.3 GB available it fits, but **the real risk is co-residency with the voice stack on 8 GB shared memory, not the weights themselves.**
* **G3 — smolvla.** Ships **PyTorch safetensors** (~450M params) via LeRobot's `SmolVLAPolicy`, with **no published ONNX export or TensorRT engine**. Note that **the `smolvla-jetbot` fine-tune named in earlier revisions of this spec does not exist** — only `lerobot/smolvla_base` does, and a JetBot fine-tune is future work. ~0.9 GB in BF16, so it fits comfortably once torch exists. Dummy motor-token I/O only, no PWM.
* **G4 — llama-nemotron-embed-vl-1b-v2.** Real, but ships HF safetensors with no published engine, and **the "1b" is misleading: it is ~1.7B params, so FP16 weights are ~3.4 GB**, not ~2 GB. NVIDIA's optimized path is a NeMo Retriever NIM, which is x86-first and not a drop-in on Tegra. For Stage I memory, seriously consider a smaller text-only embedder — a 1.7B multimodal encoder is a heavy choice for a board that must also hold a VLM and the voice stack.

---

## 2. Project Directory Structure

Target production tree (scaffolded under `jetbot_agent/`; modules are stubs until their stage passes):

```text
jetbot_agent/
├── JETBOT_SPEC.md              # This file lives at repo root; copy/symlink optional
├── requirements.txt            # Python dependencies
├── config.yaml                 # Configuration parameters & system thresholds
├── setup_env.sh                # OS initialization, swap, ALSA & camera setup
├── main.py                     # Main orchestrator state machine & loop (not implemented yet)
│
├── hardware/                   # Hardware drivers (I2C, Camera, Audio)
│   ├── __init__.py
│   ├── motor_controller.py     # PCA9685 I2C motor driver (/dev/i2c-1 @ 0x40)
│   ├── csi_camera.py           # GStreamer / Isaac ROS Argus zero-copy CSI pipeline
│   └── audio_interface.py      # Waveshare ALSA wrapper for PyAudio & aplay
│
├── audio/                      # Voice Subsystem
│   ├── __init__.py
│   ├── audio_preprocessor.py   # WebRTC APM AEC/NS/AGC/HPF/VAD (RNNoise rejected in F3)
│   ├── zipformer_asr.py        # Zipformer transducer adapter (int8, CPU)
│   └── piper_tts.py            # Piper VITS adapter (int8, CPU)
│
├── engine/                     # Model execution — llama.cpp/GGUF for the VLM, PyTorch for the rest
│   ├── __init__.py
│   ├── trt_llm_vlm.py          # Qwen2.5-VL-3B via llama.cpp + GGUF (Q4 backbone + F16 mmproj)
│   ├── trt_vla_motor.py        # smolvla policy launcher — PyTorch safetensors, no engine published
│   └── trt_embedder.py         # llama-nemotron-embed-vl-1b-v2 (~1.7B, ~3.4 GB FP16) wrapper
│
├── memory/                     # 3-Tier Multimodal Memory Engine
│   ├── __init__.py
│   ├── chroma_db.py            # Local Vector Store (ChromaDB + Nemotron Embeddings)
│   ├── memory_compactor.py     # Asynchronous context summarizer & flusher
│   └── facts_db.py             # SQLite key-value user facts DB (Mem0 style)
│
└── agent/                      # Hermes Agent Harness & MCP Tools
    ├── __init__.py
    ├── hermes_harness.py       # Core decision-making & orchestration engine
    └── tools/                  # Executable Agent Tools (Stage H I1–I8)
        ├── __init__.py
        ├── navigation_tools.py # I5: smolvla / cmd_vel + watchdog (dummy first; never PWM)
        ├── search_tools.py     # I4: Tavily (API key may come later; fail closed)
        ├── vision_tools.py     # I3: OCR & visual grounding stubs via CSI camera
        ├── voice_tools.py      # I6: after F4/F5 one-shot; duplex after F6
        └── engine_tools.py     # I7: VLM/engine wrappers after Stage G
        # memory_tools.py       # after Stage I memory — not part of I1–I8
```

OS setup used during bring-up: [`setup_env.sh`](setup_env.sh) at repo root (also copied conceptually from this tree).

## 3. Staged bring-up (source of truth)

Live procedures: [`docs/bringup/README.md`](docs/bringup/README.md). Issues: [`TASKBOARD.md`](TASKBOARD.md).

| Stage | What | Agent tickets |
| --- | --- | --- |
| A | OS / NVMe / swap / headless / MAXN SUPER | — |
| B | I2C probe, then wheels-up motors | — |
| C | IMX219 CSI / Argus | I3 uses camera |
| D | Waveshare SSS1629 ALSA (name, not card index; sidetone off) | — |
| E | `jetbot_agent` import skeleton | — |
| F | WebRTC APM, Zipformer ASR, Piper VITS TTS, duplex (RNNoise rejected) | I6 waits on F4/F5/F6 |
| G | Model runtimes, dummy I/O only. **PyTorch install comes first**, then llama.cpp/GGUF VLM | I5 smolvla + I7 wait on G |
| **H** | **Agent integration** | **I1 harness → I2 safe tools → I3 vision → I4 search → I5 nav dummy → I6 voice → I7 VLM → I8 `main.py`** |
| **I** | **Memory (Chroma + SQLite)** | Memory tools *after* this stage |

Voice stack: **Zipformer** ASR, **Piper VITS** TTS, **WebRTC APM** front end (required; **RNNoise rejected in F3**), all CPU-only through one `sherpa-onnx` process. Identify the USB sound device by ALSA name, never a fixed card index.
