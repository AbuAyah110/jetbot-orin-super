# Bill of Materials — Jetson Orin Nano Super JetBot

Based on the official [JetBot Orin BOM](https://jetbot.org/master/bill_of_materials_orin.html), updated for **Orin Nano Super** and this fork.

!!! note

    JetBot was originally designed for the discontinued Jetson Nano Developer Kit. This guide targets **Jetson Orin Nano Super** (8GB Developer Kit with Super / MAXN SUPER support).

    **Power is different** from the upstream Orin BOM — see [Power](power.md). Do not rely on the light PD pack for Super + AI demos.

Some parts are 3D printed. STL files are in [`../assets/`](../assets/) and mirrored under [`cad/`](cad/). See [3D printing](3d_printing.md).

## Common parts

| Part | Qty | Approx. cost | URL | Notes |
| --- | --: | --: | --- | --- |
| Jetson Orin Nano Super 8GB Developer Kit | 1 | — | [NVIDIA](https://store.nvidia.com/jetson/store/) | Prefer Super-capable module / flash config |
| NVMe SSD | 1 | — | vendor of choice | **Recommended** boot + model storage |
| Micro SD card (optional) | 1 | ~$15 | — | Only if not booting from NVMe |
| Motor (TT form factor) | 2 | ~$6 | [Adafruit](http://adafru.it/3777) | |
| Motor Driver (Adafruit Motor HAT / compatible) | 1 | ~$20 | [Adafruit](http://adafru.it/2927) | Orin boards often appear at I2C **0x70** |
| Caster ball (1-inch) | 1 | ~$11 | — | |
| USB cable pack (Type A to Micro, right angle) | 1 | ~$7 | — | Motor driver / accessory power as needed |
| PiOLED display | 1 | ~$13 | [Adafruit](http://adafru.it/3527) | |
| PiOLED header (right angle) | 1 | ~$8 | [Adafruit](http://adafru.it/1541) | |
| Chassis | 1 | — | [chassis.stl](../assets/chassis.stl) | 3D print |
| Camera mount | 1 | — | [camera_mount.stl](../assets/camera_mount.stl) | 3D print |

## Power source

| Part | Qty | Status |
| --- | --: | --- |
| Super-capable battery / PSU | 1 | **TBD** — [power.md](power.md) |
| Cabling for Jetson + motor rail | 1 | **TBD** |

Upstream Orin BOM listed a ~20W PD pack and recommended 7W mode. **This project will not use that as the final solution** for Orin Nano Super.

## Camera

Need a **22-pin** CSI camera for the Orin Nano carrier.

### Option 1 (default) — IMX219 175° FoV

| Part | Qty | Approx. cost | Notes |
| --- | --: | --: | --- |
| Arducam B0392 IMX219 175° | 1 | ~$20 | Wide FoV for JetBot |

## Wheels

### Option 1 — 60mm

| Part | Qty | Notes |
| --- | --: | --- |
| Wheel 60mm | 2 | TT shaft |
| Caster base / shroud | 1 each | [caster_base_60mm.stl](../assets/caster_base_60mm.stl), [caster_shroud_60mm.stl](../assets/caster_shroud_60mm.stl) |

### Option 2 — 65mm

| Part | Qty | Notes |
| --- | --: | --- |
| Wheel 65mm | 2 | TT shaft |
| Caster base / shroud | 1 each | 65mm STL variants in `assets/` |

## Assembly hardware

| Part | Qty per JetBot | Notes |
| --- | --: | --- |
| Adhesive poster strips | 2 | Battery / cable management |
| M2 self-tapping screws (~8mm) | ~20 | |
| M3 × 25mm screws | 4 | Motors |
| M3 nuts | 4 | |
| Female–female jumpers (~20cm) | 4 | OLED ↔ motor driver I2C |

## Phase 2 (not required for first motion)

| Part | Qty | Status |
| --- | --: | --- |
| USB microphone | 1 | Planned |
| Speaker / amp | 1 | Planned |

See [roadmap](roadmap.md).

## Purchasing tip

Prefer the detailed vendor links on the official page when shopping: [Bill of Materials (Orin)](https://jetbot.org/master/bill_of_materials_orin.html) — then substitute Super compute + this project's power solution.
