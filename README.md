# JetBot Orin Super

An open-source **JetBot** build guide and software fork for **NVIDIA Jetson Orin Nano Super**.

This project starts from the official [NVIDIA JetBot](https://github.com/NVIDIA-AI-IOT/jetbot) platform and [jetbot.org](https://jetbot.org/master/) documentation, then adapts hardware notes and motor I2C defaults for Orin Nano / Orin Nano Super.

> **Phase 1 (this repo today):** mechanical JetBot + motion + classic notebooks on Orin Nano Super.  
> **Phase 2 (coming next):** conversational robot — microphone, speaker, VLM, ASR, TTS, and on-device RAG.

## Why this fork?

| Item | Original JetBot (Nano) | This project (Orin Nano Super) |
| --- | --- | --- |
| Compute | Jetson Nano | Jetson Orin Nano Super |
| Motor I2C bus | often `0` | **`1`** (40-pin header) |
| Motor chip address | `96` (`0x60`) | **`112` (`0x70`)** on many Orin expansion boards |
| Power | Nano-era packs | **Different** — see [Power](docs/power.md) (TBD for Super) |
| OS image | JetBot SD image | **JetPack** on NVMe/SSD (no official JetBot Orin image) |

Community reference for the I2C change: [NVIDIA Developer Forums — Using JetBot with Jetson Orin Nano](https://forums.developer.nvidia.com/t/using-jetbot-with-jetson-orin-nano-dev-kit/281686/8).

## Quick start

1. **Flash Jetson** to NVMe/SSD and enable **MAXN SUPER** — see [Jetson setup](docs/jetson_setup.md).
2. **Buy parts** — [Bill of Materials (Orin Super)](docs/bill_of_materials_orin.md).
3. **Print chassis** — STL files in [`assets/`](assets/) — [3D printing tips](docs/3d_printing.md).
4. **Assemble** — [Hardware setup](docs/hardware_setup.md) (same mechanical steps as JetBot; Orin carrier notes called out).
5. **Install software** — [Software setup](docs/software_setup/orin_native.md).
6. **Verify motors** — open `notebooks/basic_motion/basic_motion.ipynb`.

```bash
git clone https://github.com/AbuAyah110/jetbot-orin-super.git
cd jetbot-orin-super
python3 setup.py install
```

Confirm the motor driver is visible:

```bash
sudo i2cdetect -y -r 1
# Expect 0x70 (112) for Orin-style boards, or 0x60 (96) for classic Adafruit Motor HAT wiring
```

## Repository layout

```
assets/           # STL / CAD for chassis, camera mount, caster
docs/             # Full build + software guide (MkDocs-friendly)
jetbot/           # Python package (Orin I2C defaults)
notebooks/        # Basic motion, teleop, collision, road, object following
  object_following/live_demo_nanoowl_orin.ipynb   # Community NanoOWL demo
scripts/          # Jetson helper scripts
docker/           # Upstream Docker assets (Nano-era; not primary path on Orin)
```

## Official upstream docs

- [JetBot documentation](https://jetbot.org/master/)
- [NVIDIA-AI-IOT/jetbot](https://github.com/NVIDIA-AI-IOT/jetbot)
- [Orin BOM page](https://jetbot.org/master/bill_of_materials_orin.html)

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for the conversational stack (VLM + ASR + TTS + RAG), mic/speaker, and Super-capable power.

## License

MIT — see [LICENSE.md](LICENSE.md). Based on NVIDIA JetBot (MIT).
