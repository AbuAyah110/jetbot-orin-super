# JetBot Orin Super

Open-source **JetBot** for **NVIDIA Jetson Orin Nano Super**.

This documentation set is adapted from [jetbot.org](https://jetbot.org/master/) and the [NVIDIA-AI-IOT/jetbot](https://github.com/NVIDIA-AI-IOT/jetbot) project, with Orin Super setup, I2C motor defaults, and a roadmap for a talkable on-device AI robot.

## Start here

1. [Getting Started](getting_started.md)
2. [Jetson setup (SSD + MAXN SUPER)](jetson_setup.md)
3. [Bill of Materials](bill_of_materials_orin.md)
4. [Hardware Setup](hardware_setup.md)
5. [Software Setup (Orin native)](software_setup/orin_native.md)

## What's different from classic JetBot?

- Compute: **Orin Nano Super** (not Jetson Nano)
- Motors: I2C **bus 1**, address **112 (0x70)** by default — [details](reference/i2c_orin.md)
- OS: **JetPack** on SSD (no JetBot SD image)
- Power: **custom Super-capable supply** — [power.md](power.md)
- Future: mic, speaker, VLM, ASR, TTS, RAG — [roadmap](roadmap.md)

## It's educational

Same Jupyter path as JetBot: basic motion → teleop → collision avoidance → road / object following. Community Orin notebook: NanoOWL object following under `notebooks/object_following/`.

## Get involved

- File issues on this repository
- Upstream JetBot discussions: [GitHub Discussions](https://github.com/NVIDIA-AI-IOT/jetbot/discussions)
- [Jetson Developer Forums](https://forums.developer.nvidia.com/)
