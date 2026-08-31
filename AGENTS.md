# AGENTS.md — instructions for AI agents working in this repo

JetBot Orin Super: hardware build guide plus agentic robotics stack for the
NVIDIA Jetson Orin Nano Super 8GB. See [README.md](README.md) for layout and
[TASKBOARD.md](TASKBOARD.md) for the current staged bringup gate.

## Core rule

**LLMs never set PWM.** Motion goes Hermes/Qwen → MCP → ROS 2 → `jetbot_base`
(velocity limits + watchdog) → motors. Read [docs/safety.md](docs/safety.md)
before touching motor, GPIO, or autonomy code.

Keep `backend: mock` in `config/robot.yaml` until the on-device checklist in
[docs/hardware_motors.md](docs/hardware_motors.md) is done. Never enable
`jetbot_i2c` as a side effect of unrelated work.

## Safe defaults for any agent

```bash
PYTHONPATH=src python3 -m pytest tests/unit -q
./scripts/diagnostics.sh
```

Mock-backend tests, code edits, and docs are always safe. Anything under
`scripts/bringup/` touches real peripherals — see the gate rules below.

## Cursor Cloud specific instructions

**There is no hardware present in a cloud agent VM.** A Cursor cloud agent runs
in an ephemeral x86 Linux container with no connection to the Jetson. It has no
access to:

- `/dev/i2c-*` — no motor HAT, no OLED, no `i2cdetect`
- `nvargus` / `/dev/video0` — no CSI camera, no `nvarguscamerasrc`
- ALSA devices — no microphone or speaker
- TensorRT / Jetson CUDA — no engine build, no accelerated inference

Therefore, as a cloud agent you **must not**:

1. **Run any hardware gate in `scripts/bringup/*`** — `probe_i2c.sh`,
   `test_csi_camera.sh`, `test_alsa.sh`, `f1_alsa_baseline.sh`,
   `test_webrtc_apm.sh`, `test_zipformer_piper.sh`, `test_motors.py`, or
   equivalents. They will fail, hang, or produce meaningless output.
2. **Claim hardware verification.** Do not write "verified on device",
   "I2C scan passed", "camera captured", or tick a `TASKBOARD.md` /
   `docs/bringup/` stage as done based on a cloud run. Only an on-device run
   counts as evidence.
3. **Fabricate device output.** No invented `i2cdetect` grids, `v4l2-ctl`
   listings, RTF/WER numbers, or `tegrastats` lines.
4. **Touch motors at all** — no PWM, no GPIO, no enabling the I2C backend.

What you *should* do in the cloud: read and edit code, write docs, run
`tests/unit` with the mock backend, run linters, and reason about the bringup
scripts statically. When a change needs hardware proof, say so explicitly and
leave the gate unticked — for example: "Edited `test_csi_camera.sh`; not run, no
CSI camera in cloud. Needs an on-device run to verify."

Hardware gates can only be verified through an on-device **My Machines** worker
on the Jetson; see [docs/remote_access.md](docs/remote_access.md).

## Unattended and phone-driven sessions

Motor and PWM commands must never be issued from an unattended session,
including phone sessions on the on-device worker. The wheels-up rule applies:
motion testing needs an operator physically present, wheels off the ground, and
a hand on the power cut. Details in [docs/remote_access.md](docs/remote_access.md).

## Conventions

- Docs are Markdown under `docs/`, indexed by `mkdocs.yml`.
- Python package sources in `src/` and `jetbot/`; ROS 2 packages in `ros2_ws/`.
- Do not run `git config`; do not push without being asked.
- Do not use `sudo` — it needs an interactive password on this board.
- Run shell commands unsandboxed (`required_permissions: ["all"]`) from the
  first call, and do not ask per-command approval. GPU, CUDA, TensorRT, I2C,
  `systemctl --user`, builds, and `git push` all need it, and a sandboxed
  attempt fails in ways that are slow to tell apart from a real defect.
- Do not ask for confirmation of routine next steps; report what was done.
