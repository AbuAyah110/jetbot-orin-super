# Software setup — Jetson Orin Nano Super (native)

There is **no official JetBot OS image** for Orin Nano. Use JetPack, then install this package.

Upstream Docker containers target **Jetson Nano + JetPack 4.x** and are **not** the recommended path here. Keep `docker/` in the tree for reference only.

## Prerequisites

1. JetPack installed on NVMe/SSD — [Jetson setup](../jetson_setup.md).
2. Hardware assembled — [Hardware setup](../hardware_setup.md).
3. User in the `i2c` and `gpio` groups (re-login after):

```bash
sudo usermod -aG i2c,gpio $USER
```

4. System packages commonly needed:

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev i2c-tools cmake
```

## Install the `jetbot` package

```bash
git clone https://github.com/AbuAyah110/jetbot-orin-super.git
cd jetbot-orin-super
python3 setup.py install
# or: pip3 install -e .
```

Community note (same flow): install from source with `python3 setup.py install` when not using a JetBot image — [forum thread](https://forums.developer.nvidia.com/t/using-jetbot-with-jetson-orin-nano-dev-kit/281686/14).

## Verify I2C motors

On Orin Nano, the 40-pin header I2C is typically **bus 1**.

```bash
sudo i2cdetect -y -r 1
```

| Address | Meaning |
| --- | --- |
| `0x70` (112) | Orin-style / many custom JetBot motor boards (**default in this fork**) |
| `0x60` (96) | Classic Adafruit Motor HAT address |
| `0x3c` | PiOLED (SSD1306) often appears here |

Then in Python / Jupyter:

```python
from jetbot import Robot
robot = Robot()          # bus=1, prefers addr 112
robot.forward(0.2)
robot.stop()
```

Override if needed:

```python
robot = Robot(i2c_bus=1, i2c_address=96)   # classic Motor HAT
```

Details: [Orin I2C notes](../reference/i2c_orin.md).

## Jupyter notebooks

```bash
cd notebooks
jupyter lab
# or: jupyter notebook
```

Start with:

1. `basic_motion/basic_motion.ipynb`
2. `teleoperation/teleoperation.ipynb`
3. Classic AI demos under `collision_avoidance/`, `road_following/`, `object_following/`
4. Community Orin demo: `object_following/live_demo_nanoowl_orin.ipynb` (NanoOWL; requires extra deps)

## Camera

Use a **22-pin** CSI camera compatible with Orin Nano (for example Arducam IMX219 wide-angle from the BOM). Confirm with:

```bash
# Depends on JetPack / Argus tooling available on your image
nvgstcapture-1.0
# or open the Camera widget in a notebook: from jetbot import Camera
```

## Optional: display stats on PiOLED

Upstream `jetbot` includes a small stats app under `jetbot/apps/`. It already uses `i2c_bus=1` for the OLED.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| `No module named 'jetbot'` | Re-run `python3 setup.py install` with the same Python Jupyter uses |
| `No motor driver found` | `i2cdetect -y -r 1`; confirm SDA/SCL/GND/3V3; address 112 vs 96 |
| Motors spin backward | Swap motor wire polarity on the driver terminals |
| Camera fails | Ribbon orientation, 22-pin cable, overlays / jetson-io if required |

## Next

- [Examples overview](../examples/basic_motion.md)
- [Roadmap: talkable JetBot](../roadmap.md)
