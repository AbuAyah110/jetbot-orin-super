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
| Models | TensorRT-Edge-LLM Qwen2.5-VL-3B, smolvla, Nemotron embed | llama.cpp Cosmos, Qwen3.5-0.8B, EmbeddingGemma |
| Voice | FastConformer ASR; FastPitch + HiFi-GAN TTS; WebRTC APM front end | Stage F stubs only |
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
* **Inference Engines:** TensorRT-Edge-LLM is limited to the compatible language/vision models. FastConformer, FastPitch, and HiFi-GAN must first be validated in their supported NVIDIA runtime on the Orin; export/optimization with TensorRT is a later, model-specific experiment, not an assumed capability of TensorRT-Edge-LLM.

### Voice architecture

The primary real-time audio front end is **WebRTC Audio Processing Module (APM)**. Process 16 kHz mono audio in 10 ms frames with high-pass filtering, noise suppression, automatic gain control, voice activity detection, and acoustic echo cancellation. AEC must receive the time-aligned far-end/reference audio sent to TTS playback because the robot's speaker is close to its microphone.

RNNoise is an optional residual-denoising experiment after APM, not an AEC replacement. Its native 48 kHz, fixed-frame processing introduces resampling and latency tradeoffs that must be measured before adoption. The first gate stays simple: capture 16 kHz mono, run WebRTC APM, and verify reduced noise and playback echo.

FastConformer ASR and FastPitch + HiFi-GAN TTS are staged independently. First prove one-file inference in a supported NVIDIA runtime and record latency, real-time factor, and peak unified RAM/VRAM on the 8 GB Orin. Only then evaluate export or TensorRT optimization where the model/runtime combination supports it. Full duplex is forbidden until AEC passes.

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
│   ├── audio_preprocessor.py   # WebRTC APM AEC/NS/AGC/HPF/VAD; optional RNNoise
│   ├── fastconformer_asr.py    # NVIDIA FastConformer ASR runtime adapter
│   └── fastpitch_hifigan_tts.py # NVIDIA FastPitch + HiFi-GAN TTS adapter
│
├── engine/                     # TensorRT-Edge-LLM & Model Execution
│   ├── __init__.py
│   ├── trt_llm_vlm.py          # C++ wrapper for Qwen2.5-VL-3B (INT4 AWQ)
│   ├── trt_vla_motor.py        # TensorRT engine launcher for smolvla-jetbot
│   └── trt_embedder.py         # llama-nemotron-embed-vl-1b-v2 engine wrapper
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
| F | WebRTC APM, optional RNNoise, FastConformer, FastPitch+HiFi-GAN, duplex | I6 waits on F4/F5/F6 |
| G | TensorRT engines, dummy I/O only | I5 smolvla + I7 wait on G |
| **H** | **Agent integration** | **I1 harness → I2 safe tools → I3 vision → I4 search → I5 nav dummy → I6 voice → I7 VLM → I8 `main.py`** |
| **I** | **Memory (Chroma + SQLite)** | Memory tools *after* this stage |

Voice stack: **FastConformer** ASR, **FastPitch + HiFi-GAN** TTS, **WebRTC APM** front end, optional **RNNoise**. Identify the USB sound device by ALSA name, never a fixed card index.
