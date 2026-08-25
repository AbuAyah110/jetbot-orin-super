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
| F Voice | [06-voice.md](06-voice.md) | safe ALSA → WebRTC APM → FastConformer + FastPitch/HiFi-GAN → guarded duplex |
| G TensorRT engines | [07-tensorrt.md](07-tensorrt.md) | dummy I/O per engine (isolated) |
| H Agent (I1–I8) | [08-agent.md](08-agent.md) | each I* ticket’s gate; LLM never PWM |
| I Memory | [09-memory.md](09-memory.md) | Chroma upsert/query + SQLite put/get |

Memory **tools** (Chroma/SQLite wrappers for the harness) are a later subtask after Stage I, not part of I1–I8.

Safety: `config/robot.yaml` stays `backend: mock` until Stage B is signed off. LLMs never set PWM. Motion only via the limited motor path (`/cmd_vel` or equivalent + watchdog). See [`docs/safety.md`](../safety.md).

## Status on this Jetson (2026-08-25)

Lightweight re-check only — no OS reinstall, no motor PWM.

| Stage | Result | Evidence |
| --- | --- | --- |
| A | **Pass with notes** | L4T R36.4.4, NVMe `/`, MAXN_SUPER, `multi-user.target`, 32 GiB swap. Swap path is `/ssd/32GB.swap` (not `/swapfile`); `vm.swappiness=60` (spec target 10). |
| B | **Pass (probe)** | I2C buses present; bus 7 shows `0x70`/`0x60`/`0x3c`. PWM **not** re-run. Prior motion: `notebooks/basic_motion/basic_motion.ipynb`. |
| C | **Pass** | `nvarguscamerasrc` 1-frame EOS; prior live preview: `notebooks/camera/csi_camera_test.ipynb`. |
| D | **Audio HW verified** | Waveshare SSS1629. Identify by ALSA **name**, never card index. Sidetone **off**. Sequential capture then playback until F2 AEC. |
| E–G | Not this check | Skeleton / voice models / TensorRT dummy I/O still ahead. |
| H | Not started | Split into I1–I8; starts after G (tools that need F/G wait on those gates). |
| I | After H | Memory stores after the agent loop skeleton exists. |

Voice stack target: **FastConformer** ASR + **FastPitch / HiFi-GAN** TTS + **WebRTC APM** (+ optional **RNNoise**). TensorRT-Edge-LLM is not the ASR/TTS runtime.
