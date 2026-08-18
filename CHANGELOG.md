# Changelog

## 0.5.0 — JetBot Orin Super fork

- Retarget documentation and package defaults for Jetson Orin Nano Super.
- Motor driver defaults: I2C bus 1, address 112 (0x70); fallback to 96 / SparkFun.
- Add Jetson SSD boot + MAXN SUPER official doc links.
- Add power placeholder for Super-capable supply.
- Add `PROJECT_PLAN.md` agentic architecture (Hermes / Qwen / Cosmos / memory).
- Milestone 0–1: diagnostics, `jetbot_control` mock motors, ROS 2 `jetbot_base` (cmd_vel, watchdog, e-stop, teleop), unit tests.
- Milestone 2 (laptop): `perception` camera service with fake/file/webcam backends, ring buffer, motion detect, CSI stub.

Upstream JetBot history: https://github.com/NVIDIA-AI-IOT/jetbot/blob/master/CHANGELOG.md
