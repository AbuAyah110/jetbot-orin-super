# JetBot Orin Super

Hardware build guide **and** agentic AI robotics stack for **NVIDIA Jetson Orin Nano Super 8GB**.

Target hardware/agent tree: **[`JETBOT_SPEC.md`](JETBOT_SPEC.md)**. Historical ROS/Cosmos plan: **[`PROJECT_PLAN.md`](PROJECT_PLAN.md)**.

## Two layers in one repo

| Layer | Status | Docs |
| --- | --- | --- |
| Classic JetBot hardware + Jupyter | Available | [docs/getting_started.md](docs/getting_started.md) |
| Agentic stack (Hermes / Qwen / Cosmos / memory / ROS 2) | **M0–M1 in progress** | [docs/architecture.md](docs/architecture.md), [docs/safety.md](docs/safety.md) |

> **Core rule:** LLMs never set PWM. Motion goes Hermes/Qwen → MCP → ROS 2 → `jetbot_base` (limits + watchdog) → motors.

## Current sprint

Staged **install-and-test** on this Orin (one gate at a time): **[`TASKBOARD.md`](TASKBOARD.md)** · **[`docs/bringup/README.md`](docs/bringup/README.md)**.

**Order:** A → B → C → D → E → F → G → **H agent (I1–I8)** → **I memory**. Agent is before memory.

- [x] Milestone 0–1 foundation (diagnostics, mock motors, ROS base package)
- [x] Milestone 2 mock camera (`perception`, fake/file/webcam, change detection)
- [x] Stage A OS (2026-08-25 re-check: L4T R36.4.4, NVMe `/`, MAXN_SUPER, headless; swap is `/ssd/32GB.swap` 32 GiB, swappiness 60)
- [x] Stage B I2C probe (bus 7: `0x70`/`0x60`/`0x3c`; no PWM this pass; prior `notebooks/basic_motion`)
- [x] Stage C CSI (Argus 1-frame + `notebooks/camera/csi_camera_test.ipynb`)
- [x] Stage D audio HW (Waveshare SSS1629; ALSA **name** not card index; sidetone **off**)
- [x] Stage E Python skeleton (`.venv` via `virtualenv`; PyYAML 6.0.3; import smoke pass)
- [x] Stage F2 WebRTC APM (`pywebrtc-audio` 0.1.0; offline NS/AEC fixtures; no live duplex)
- [x] Stage F3 RNNoise A/B — **rejected**; it denoises better and hears worse (ASR WER 0.006 → 0.253), so WebRTC APM stays the only front end
- [x] Stage F4/F5 default — Zipformer small int8 ASR + Piper Lessac-low int8 VITS in one `sherpa-onnx` CPU process; short-turn WER 0.00, 162.3 MiB peak, zero VRAM (5 s live mic: 192.2 MiB / 201.5 decimal MB)
- [ ] Stage F6 duplex after AEC+ASR+TTS — Piper now emits 16 kHz directly; keep the AEC reference time-aligned and preserve the feedback watchdog
- [x] Stage G1 TensorRT runtime (10.3.0.30 / CUDA 12.6.11 / cuDNN 9.3.0; three engines built and numerically verified)
- [ ] Stage G runtimes — **PyTorch install first**, then Qwen2.5-VL via **llama.cpp + GGUF**, then smolvla and the Nemotron embedder
- [x] Stage H I1–I5 (harness state machine, structural tool-safety boundary, vision, Tavily search, nav dummy + motion adapter)
- [ ] Stage H I6–I8 (voice tools after F6, VLM tools after G, `main.py` loop)
- [ ] Stage I memory (after agent); memory tools after that

```bash
./scripts/diagnostics.sh
PYTHONPATH=src python3 -m pytest tests/unit -q
PYTHONPATH=src python3 scripts/demo_camera.py --backend fake --out data/images/demo.jpg
```


## Quick start — laptop / CI (mock motors)

```bash
git clone https://github.com/AbuAyah110/jetbot-orin-super.git
cd jetbot-orin-super
python3 -m pip install -e ".[dev]"
python3 -m pytest tests/unit -q
./scripts/diagnostics.sh
```

## Quick start — Jetson (after SSH)

```bash
# SSH config on laptop:
# Host jetbot
#   HostName <JETSON_IP>
#   User <JETSON_USERNAME>
#   IdentityFile ~/.ssh/id_ed25519

./scripts/diagnostics.sh
# optional: sudo ./scripts/setup_swap.sh

# ROS 2 Humble (once installed on Jetson):
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select jetbot_base
source install/setup.bash
ros2 launch jetbot_base base.launch.py backend:=mock
# other terminal:
ros2 run jetbot_base teleop_keyboard
```

Keep `backend:=mock` until [docs/safety.md](docs/safety.md) and [docs/hardware_motors.md](docs/hardware_motors.md) checklist is done.

## Hardware JetBot guide

1. [Jetson setup (SSD + MAXN SUPER)](docs/jetson_setup.md)
2. [Bill of Materials](docs/bill_of_materials_orin.md)
3. [Hardware assembly](docs/hardware_setup.md)
4. Classic package: `python3 setup.py install` then Jupyter notebooks under `notebooks/`

Orin motor I2C defaults in `jetbot/`: **bus 1**, address **112 (`0x70`)**.

## Milestone roadmap

See [PROJECT_PLAN.md](PROJECT_PLAN.md) §35 and [docs/roadmap.md](docs/roadmap.md).

```text
M0 diagnostics → M1 ROS motors → M2 camera → M3 Cosmos → M4 memory
→ M5–M6 MCPs → M7 Hermes → M8 Qwen → M9 voice → M10 Nav2 → …
```

## License

MIT — see [LICENSE.md](LICENSE.md). Based on NVIDIA JetBot (MIT).
