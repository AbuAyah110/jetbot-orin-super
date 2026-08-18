# Jetson Orin Nano Super — OS, SSD boot, and MAXN SUPER

This project expects a normal **JetPack** install on **Jetson Orin Nano Super**, not the legacy JetBot SD card image (that image targets Jetson Nano only).

## 1. Boot from NVMe SSD (recommended)

Orin Nano Developer Kit does **not** include storage in the box. Use a **microSD** or, preferably, an **NVMe SSD** for models, containers, and datasets.

Official setup paths:

| Method | Best for | Official guide |
| --- | --- | --- |
| **Jetson ISO** | First-time install without an Ubuntu host (JetPack 7.2+) | [Quick Start Guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html) |
| **SDK Manager** | Guided flash to NVMe, recovery, JetPack components | [BSP Setup](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/setup_bsp.html) |

High-level SSD path:

1. Install a compatible NVMe SSD on the Orin Nano carrier board.
2. Flash Jetson Linux to **NVMe** (Jetson ISO installer or SDK Manager — select NVMe as target storage).
3. Remove installer media when prompted and boot from the SSD.

> Prefer SSD over microSD for Phase 2 (VLM / ASR / TTS / RAG) — model weights and TensorRT engines need the space and bandwidth.

## 2. Enable MAXN SUPER power mode

**MAXN SUPER** is available when the module is flashed with the **super** configuration (for example `jetson-orin-nano-devkit-super`). It is an unconstrained experimental mode for max clocks; thermal throttling still applies if you exceed the module TDP.

Official references:

- [How-to: Change Power Mode (Orin Nano Dev Kit User Guide)](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/howto.html)
- [JetPack 6.2 Super Mode announcement](https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/)
- [Platform power & performance (Developer Guide)](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html)

### Desktop

1. Click the current power mode in the Ubuntu top bar.
2. Open **Power Mode**.
3. Select **MAXN SUPER**.

### Command line

List modes (IDs vary by JetPack / flash config):

```bash
sudo /usr/sbin/nvpmodel -q
```

Select the mode ID shown for `MAXN_SUPER` (example only — **verify on your board**):

```bash
sudo /usr/sbin/nvpmodel -m <mode_id>
sudo jetson_clocks   # optional: lock clocks for benchmarks
```

On some JetPack 6.2 Super flashes, Orin Nano MAXN SUPER was documented as:

```bash
sudo nvpmodel -m 2
```

Always confirm with `nvpmodel -q` after flashing.

### If MAXN SUPER is missing

ISO upgrade paths can retain an older power profile. Reflash with SDK Manager using the **super** BSP configuration so Super modes appear. See NVIDIA forum guidance on [power profile retention](https://forums.developer.nvidia.com/t/help-setting-up-new-jetson-orin-nano/375797).

## 3. Sanity checks after first boot

```bash
# JetPack / L4T release
head -n 1 /etc/nv_tegra_release
cat /etc/nv_tegra_release

# Power mode
sudo nvpmodel -q

# Storage
df -h
lsblk

# I2C (motors / OLED on bus 1)
sudo i2cdetect -y -r 1
```

## 4. Wi-Fi and remote Jupyter

1. Connect Wi-Fi (desktop UI or `nmcli`).
2. Note the IP address (`ip addr` or OLED once display container/app is running).
3. After JetBot software install, open notebooks from another machine on the LAN.

Upstream Wi-Fi notes still apply: [jetbot.org Wi-Fi setup](https://jetbot.org/master/software_setup/wifi_setup.html).

## Next

- [Bill of Materials](bill_of_materials_orin.md)
- [Hardware assembly](hardware_setup.md)
- [Software install (Orin native)](software_setup/orin_native.md)
- [Power (Orin Super — TBD)](power.md)
