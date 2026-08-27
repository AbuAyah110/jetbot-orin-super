# Stage G — Cosmos-Reason2-2B on Orin Nano (Edge-LLM)

Date: 2026-08-27 (cleanup)  
Branch: `stage-g-cosmos-nano`  
Ticket: [#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17) is now the **Cosmos-Reason2-2B** robot VLM (Qwen artifacts deleted on device). Related: [#37](https://github.com/AbuAyah110/jetbot-orin-super/issues/37).

This is the **locked robot VLM**: Cosmos-Reason2-2B via **TensorRT Edge-LLM v0.10.0**, in-process, INT4 LLM + FP16 ViT, **maxBatchSize 1**, **maxInputLen 3072**, **maxKVCacheCapacity 4096**. Do **not** quantize on this board. Do **not** install PyTorch. Do **not** start a Qwen rebuild. Do **not** load Hub **FP8 / NVFP4** Cosmos checkpoints (SM87 cannot run them).

Motors were **not** driven. Camera loop was **not** started. No engine was loaded; there is **no Cosmos tegrastats**.

## Qwen2.5-VL removed (2026-08-27)

Qwen is **not** the robot VLM. On-device artifacts were deleted so nothing can load the wrong engines:

| Target | Size before delete | Result |
| --- | --- | --- |
| `~/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-ModelOpt-INT4/` (onnx ~5.0 GiB + engines ~3.8 GiB) | **8.8G** | **Gone** |
| `~/opt/llama.cpp` (old GGUF / llama.cpp VLM path) | **217M** | **Gone** |
| FastConformer / Matcha / HiFi-GAN under `data/models/f4`, `f5` | empty `.gitkeep` only | left as placeholders |
| RNNoise under `data/models/f3` | **13M** (not huge) | **kept** |
| Jupyter leftover `~/.local/share/jupyter` + `~/.jupyter` | **24M** | **Gone** (repo `notebooks/` kept) |

`df -h /` after cleanup: **73G used / 34G avail (69%)** on 113G NVMe (was 82G used / 26G avail). Roughly **~9 GiB freed**. No other `Qwen2.5-VL*` trees under `$HOME`. Did **not** `pip uninstall` sherpa-onnx, numpy, or pywebrtc-audio.

## Cosmos ONNX: **absent** — waiting on workstation rsync

Canonical empty destination (created this session):

```
~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/
  onnx/llm/
  onnx/visual/
  engines/llm/
  engines/visual/
  logs/
  WAIT_FOR_WORKSTATION_ONNX.txt
```

Legacy empty stub still present (same layout): `~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4/`. Use **`Cosmos-Reason2-2B`** for rsync.

**Wait** for workstation artifacts (`TensorRT-Edge-LLM` **v0.10.0** export). Do **not** run `llm_build` until `onnx/llm/model.onnx` exists. Engines are built **on this Jetson** from that ONNX; do not copy x86 `.engine` files.

BGE-small is **CPU later** (orchestrator / LanceDB). It is **not on disk yet**. Do not fetch it in this loop.

### Workstation rsync (exact destination)

On the x86 workstation, after INT4 AWQ export (not FP8, not NVFP4):

```bash
# Export flags belong here, not on llm_build:
#   --externalize-weights int4_ffn --int4-gemm-plugin-version 1
rsync -a --info=progress2 \
  "$WORKSPACE_DIR/onnx/" \
  impulse110@<orin>:/home/impulse110/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/
```

Expected after rsync: `onnx/llm/model.onnx` (+ `.data` / sidecars) and `onnx/visual/model.onnx`. Prefer `onnx/SHA256SUMS`.

Hub IDs to **reject** on this Nano: `nvidia/Cosmos-Reason2-2B-FP8`, `nvidia/Cosmos-Reason2-2B-NVFP4`.

## Inventory (this Jetson, 2026-08-27)

| Item | Value |
| --- | --- |
| Product | WaveShare JetBot / Jetson Orin Nano Super (8 GB UMA, SM87) |
| L4T | R36.4.4 |
| TensorRT | **10.3** (JetPack) |
| TensorRT Edge-LLM | `/home/impulse110/Documents/_edgellm_ref/repo` tag **`v0.10.0`** (**kept**) |
| `llm_build` | `/home/impulse110/Documents/_edgellm_ref/repo/build/examples/llm/llm_build` (already built; can run **after** ONNX lands) |
| `visual_build` | `.../build/examples/multimodal/visual_build` |
| Disk `/` | NVMe **113G**, **73G used, 34G avail (69%)** after Qwen delete |
| Swap | `/ssd/32GB.swap` **32 GiB** (**kept**) |
| Voice (Stage F) | Zipformer + Piper in `jetbot-wt-voice-sherpa/data/models` + sherpa-onnx `.venv` (**kept**) |
| WebRTC APM / F2 | **kept** |
| jetbot I2C/OLED | **kept** (not wiped) |
| Qwen2.5-VL | **deleted** |
| llama.cpp | **deleted** (`~/opt/llama.cpp`) |
| BGE | **not on disk** (CPU later) |
| Motors | `jetbot.Robot` / PCA9685 **not opened** |

`llm_build --help` does **not** accept `--externalize-weights` or `--int4-gemm-plugin-version`. Those are **export** flags. This binary accepts `--onnxDir`, `--engineDir`, `--maxBatchSize`, `--maxInputLen`, `--maxKVCacheCapacity`.

## Memory budget (locked)

Cosmos-Reason2-2B is **Qwen3-VL-class**. NVIDIA Nano 8 GB published proxy: **Qwen3-VL-2B INT4 + FP16 ViT ≈ 4.38 GiB GPU mem**, not 2.1 GiB (2.1 GiB is weights, not a running engine). Budget **4.3–4.7 GiB** for the VLM. Usable RAM is **~7.2–7.5 GiB**.

If a loaded Cosmos engine shows **tegrastats ≥ 5.0 GiB**, **stop and report**. Think mode only when parked. Context **4096**, 4–6 text turns, current CSI JPEG 448² only.

Locked process (when artifacts exist): CSI JPEG → resize 448² → Edge-LLM Cosmos in-process; ASR (CPU) → orchestrator → BGE-small CPU → LanceDB; JSON `stop|drive|speak|wait|weather` → `jetbot.Robot` I2C + TTS (CPU). Clamp `vx` 0.22, `wz` 1.0, duration then stop.

Explicitly **not**: ROS2, Nav2, extra HTTP LLM server, MiniLM, sqlite-vec, BGE GPU TRT, TensorRT-LLM, FastPitch/NeMo/Riva, Hermes Agent, live Jina CLIP / SmolVLA / Gemma 4 audio / Qwen3-Embedding, PyTorch/`transformers` on this Nano, thinking while moving, Qwen2.5-VL.

## Memory tactics from the locked plan

Keep idle UMA small so Cosmos INT4 + FP16 ViT (~4.3–4.7 GiB) can load. **Do not** use Gemini’s 3.90 GB table. Abort if a loaded Cosmos engine shows **tegrastats RAM ≥ 5.0 GiB**.

| Tactic | Why |
| --- | --- |
| CSI **448² JPEG**, not 1080p ROS | ViT tokens and host copies stay small |
| **maxKVCacheCapacity 4096**, not 262k | KV dominates UMA after weights |
| **One** in-process Edge-LLM, **no** extra HTTP LLM server | Second server doubles engines |
| **BGE-small on CPU** later, never BGE GPU TRT | GPU stays for Cosmos |
| Think mode **parked only** | Extra decode while driving blows the budget |
| **No** live PyTorch / Jina / SmolVLA | Those stacks do not fit beside Cosmos |
| Zipformer + Piper **CPU** (~192 MiB) | Voice stays off the GPU |
| GUI **multi-user.target** | Desktop is hundreds of MiB to GiB |
| OS idle often **1.2–2.2 GiB** used without Cursor | Target **~5.5–6.5 GiB available** for Cosmos |
| 32 GiB `/ssd/32GB.swap` **kept**; **`vm.swappiness=10`** | Swap is OOM safety; Cosmos must not page |

Script (dry-run default; privileged steps need `sudo`):

```bash
./scripts/JETSON_IDLE_RAM.sh
sudo ./scripts/JETSON_IDLE_RAM.sh --apply
```

`--apply` stops/disables docker, `nv-l4t-usb-device-mode`, bluetooth, cups, snapd; writes `/etc/sysctl.d/99-jetbot-memory.conf`; rewrites `jetbot_oled.service` to run `oled_status.py` **by path** (no `python3 -m jetbot.apps…`). It does **not** kill `nvargus-daemon`, delete swap, uninstall JetPack, or `llm_build`.

Workstation ONNX (still running as of this write): quantized checkpoint on the PC is `~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4/quantized/` (~2.7 GB). After export, rsync to this Jetson: `~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/`. **Do not** `llm_build` until that tree has `onnx/llm/model.onnx`.

### Idle RAM this session (2026-08-27, waiting on rsync)

Measured **before** any stop (Cursor SSH still connected):

| Item | Value |
| --- | --- |
| `free -h` | **2.6 GiB used** / 2.2 GiB free / **4.6 GiB available** of 7.4 GiB |
| tegrastats | **2859 / 7620 MB**, SWAP 631 / 32768 MB (cached 116 MB), GR3D 0% |
| default target | **`multi-user.target`** (GUI already off; gdm inactive) |
| `nvpmodel -q` | **MAXN_SUPER** (left unchanged) |
| Swap | `/ssd/32GB.swap` **32 GiB**, **631 MiB used** — **not deleted**, not grown |
| `vm.swappiness` | **60** — cannot change without sudo; script documents `sysctl -w vm.swappiness=10` |
| Disk `/` | 73G used / 34G avail (69%); **no Qwen** trees |
| jupyter / ipykernel | **dead** |
| rviz / nav2 / firefox / chromium | **none** |
| CSI | `nvargus-daemon` PID 5550, **~1.4 MiB RSS**, no gst preview (daemon kept) |
| OLED | `python3 -m jetbot.apps.oled_status` PID 904, **~4.6 MiB RSS** / **3.0 MiB PSS** (not the historical 160+ MiB). Unit still `-m`; by-path fix is in git + `JETSON_IDLE_RAM.sh --apply` (needs sudo to restart the **system** unit). |
| docker | socket+service **active**, `docker ps` **empty**; dockerd ~12.5 MiB + containerd ~10 MiB RSS |
| snapd | **active**, ~37 MiB RSS |
| `nv-l4t-usb-device-mode` | **active** |
| bluetooth / cups / ros2 | inactive |
| Cursor `node` (editor) | **~1.9 GiB PSS** across ~15 procs; top RSS ~545 / 514 / 501 / 241 MiB. **Goes away when the editor disconnects.** |

**Without sudo this pass:** no docker/snapd/usb-gadget stop, no sysctl, no OLED restart. User `--apply` only confirmed leftover GUI/Jupyter processes were already gone. Expected extra reclaim after `sudo … --apply`: on the order of **~70–80 MiB** (snapd+docker+gadget), not GiB.

**Vs locked idle target 1.2–1.8 GiB used / 5.5–6.5 GiB available:** we are at **2.6 GiB used / 4.6 GiB available**. The gap is almost entirely **Cursor remote** (~1.5–1.9 GiB). OS+JetPack+OLED+nvargus with Cursor gone is already in the **~1.2–2.2 GiB used** band from the locked plan. Do not stop Cursor from this script.

OLED **delta this session: ~0 MiB** (already slim; the win is preventing a future `-m` restart from importing `jetbot/__init__.py`).

### Post-`sudo JETSON_IDLE_RAM.sh --apply` measurement

Measured 2026-08-27 at 12:05 CDT with Cursor still connected:

| Item | Post-apply value |
| --- | --- |
| `free -h` | **2.5 GiB used**, 2.1 GiB free, **4.7 GiB available** of 7.4 GiB |
| `tegrastats` | **2783–2784 / 7620 MB**, SWAP **507 / 32768 MB** (cached 99 MB), GR3D 0% |
| Swap | `/ssd/32GB.swap`, 32G total, **507.5M used**, priority -2 |
| `vm.swappiness` | **10**; persisted in `/etc/sysctl.d/99-jetbot-memory.conf` |
| docker | **inactive**, service disabled |
| snapd | **inactive**, service disabled |
| gdm / gdm3 | **inactive**; default target remains `multi-user.target` |
| OLED | **active**, 15.5M systemd memory; PID 369574 |
| OLED cmdline | `/usr/bin/python3 /home/impulse110/Documents/jetbot-orin-super/jetbot/apps/oled_status.py` |
| CSI daemon | `nvargus-daemon` active, ~1.4 MiB RSS; no capture pipeline started |
| Power mode | `MAXN_SUPER` |

The privileged cleanup reclaimed about 0.1 GiB used RAM and increased available
RAM from 4.6 to 4.7 GiB in this Cursor-connected session. The remaining large
resident is Cursor remote, not a robot service. No model or camera pipeline was
loaded and motors were not opened.

## Repository-local runtime layout

Code stays in `/home/impulse110/Documents/jetbot-orin-super`. Non-git payloads:

```text
third_party/tensorrt-edge-llm/  # ignored full v0.10.0 source/build clone
data/edgellm/cosmos/
  onnx/llm/
  onnx/visual/
  engines/llm/
  engines/visual/
  logs/
```

For compatibility, the old model destination may be a symlink:

```bash
mkdir -p /home/impulse110/Documents/jetbot-orin-super/data/edgellm/cosmos
ln -s /home/impulse110/Documents/jetbot-orin-super/data/edgellm/cosmos \
  ~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B
```

Do not commit the Edge-LLM clone, ONNX, external data, engines, weights, or
virtual environments. See `data/edgellm/README.md`.

### Exact workstation rsync reminder

Run on the workstation:

```bash
rsync -avP --checksum ~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4/onnx/ impulse110@192.168.50.65:~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/
```

## Exact on-device engine build (run only after ONNX arrives)

Export-time (workstation, already required in the ONNX tree): `--externalize-weights int4_ffn --int4-gemm-plugin-version 1`.

On this Jetson:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export REPO_ROOT=$HOME/Documents/jetbot-orin-super
export EDGELLM_ROOT=$REPO_ROOT/third_party/tensorrt-edge-llm
export EDGELLM_PLUGIN_PATH=$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so
export WORKSPACE_DIR=$REPO_ROOT/data/edgellm/cosmos

# Confirm pin
git -C "$EDGELLM_ROOT" describe --tags --always   # expect v0.10.0

cd "$EDGELLM_ROOT"
./build/examples/llm/llm_build \
  --onnxDir "$WORKSPACE_DIR/onnx/llm" \
  --engineDir "$WORKSPACE_DIR/engines/llm" \
  --maxBatchSize 1 \
  --maxInputLen 3072 \
  --maxKVCacheCapacity 4096

./build/examples/multimodal/visual_build \
  --onnxDir "$WORKSPACE_DIR/onnx/visual" \
  --engineDir "$WORKSPACE_DIR/engines" \
  --minImageTokens 8 \
  --maxImageTokens 2048 \
  --maxImageTokensPerImage 2048
```

Wrapper that **exits 2** if ONNX is still missing:

```bash
./scripts/bringup/llm_build_cosmos.sh
```

Build LLM then ViT **sequentially**, nothing else large resident (stop voice/agent). Sample `tegrastats` throughout. Do not copy x86 `.engine` files.

## Status

| Step | Result |
| --- | --- |
| HOME `/home/impulse110` | OK (Jetson) |
| Qwen2.5-VL on disk | **Deleted** |
| Cosmos-Reason2-2B rsync dir | **Ready** (`~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/`) |
| Cosmos-Reason2 ONNX rsynced | **No** |
| `llm_build` started | **No** (waiting on workstation ONNX) |
| Cosmos engine loaded / tegrastats | **N/A** |
| BGE | Not on disk (CPU later) |
| Idle RAM (Cursor still on, post-apply) | **2.5 GiB used / 4.7 GiB available**; tegrastats 2783–2784 / 7620 MB |

When ONNX lands: run `llm_build_cosmos.sh`, attach peak RAM/swap, and abort if tegrastats ≥ 5.0 GiB on a loaded inference engine.
