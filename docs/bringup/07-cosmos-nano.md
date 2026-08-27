# Stage G — Cosmos-Reason2-2B on Orin Nano (Edge-LLM)

Date: 2026-08-26  
Branch: `stage-g-cosmos-nano`  
Ticket: [#37](https://github.com/AbuAyah110/jetbot-orin-super/issues/37). Do not treat [#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17) / [#32](https://github.com/AbuAyah110/jetbot-orin-super/issues/32) as the Cosmos path — those are Qwen2.5-VL.

This is the **locked robot VLM**: Cosmos-Reason2-2B via **TensorRT Edge-LLM v0.10.0**, in-process, INT4 LLM + FP16 ViT, **4096 KV**, **3072 max input**. Do **not** quantize on this board. Do **not** load Hub **FP8 / NVFP4** Cosmos checkpoints (SM87 cannot run them). Do **not** treat existing Qwen2.5-VL INT4 engines as Cosmos.

Motors were **not** driven. Camera loop was **not** started. No engine was loaded; there is **no Cosmos tegrastats**.

## Cosmos ONNX: **absent**

Searched `$HOME/tensorrt-edgellm-workspace` and `$HOME` for `Cosmos-Reason2*` ONNX / `model.onnx` trees. **None present.**

What *is* on disk (unused for this loop):

| Path | What it is |
| --- | --- |
| `~/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-ModelOpt-INT4/{onnx,engines}` | Qwen2.5-VL-3B INT4 ONNX (5.0 GiB) + SM87 engines (3.8 GiB). **Not Cosmos.** Do not `llm_build` or load these for the robot VLM. |

Prepared empty rsync destination (this session):

```
~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4/
  onnx/llm/
  onnx/visual/
  engines/llm/
  engines/visual/
  logs/
  WAIT_FOR_WORKSTATION_ONNX.txt
```

**Wait** for workstation artifacts (`TensorRT-Edge-LLM` **v0.10.0** export). Do not run `llm_build` until `onnx/llm/model.onnx` exists.

### Workstation rsync (exact destination)

On the x86 workstation, after INT4 AWQ export (not FP8, not NVFP4):

```bash
# Export flags belong here, not on llm_build:
#   --externalize-weights int4_ffn --int4-gemm-plugin-version 1
rsync -a --info=progress2 \
  "$WORKSPACE_DIR/onnx/" \
  impulse110@<orin>:/home/impulse110/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4/onnx/
```

Expected after rsync: `onnx/llm/model.onnx` (+ `.data` / sidecars) and `onnx/visual/model.onnx`. Prefer `onnx/SHA256SUMS`.

Hub IDs to **reject** on this Nano: `nvidia/Cosmos-Reason2-2B-FP8`, `nvidia/Cosmos-Reason2-2B-NVFP4`.

## Inventory (this Jetson, 2026-08-26 ~18:40 local)

| Item | Value |
| --- | --- |
| Product | WaveShare JetBot / Jetson Orin Nano Super (8 GB UMA, SM87) |
| Device tree | `NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super` |
| TNSPEC | `3767-300-0005-T.1-1-0-jetson-orin-nano-devkit-super-` |
| L4T | R36.4.4 (`/etc/nv_tegra_release`, `nvidia-l4t-core 36.4.4-20250616085344`) |
| JetPack userspace | 6.2-class (L4T 36.4.4); Compatible, not Thor/JP7 Official |
| Kernel | `5.15.148-tegra` aarch64 |
| Power | `NV Power Mode: MAXN_SUPER` |
| GUI / boot | `systemctl get-default` → **`multi-user.target`** (`gdm3` inactive) |
| CUDA | 12.6.11 / toolkit 12.6.68 (`nvidia-smi` Driver 540.4.0) |
| TensorRT | **10.3.0.30** (`python3-libnvinfer`, `tensorrt` apt) |
| cuDNN | 9.3.0.75 (CUDA 12.6) |
| TensorRT Edge-LLM | `/home/impulse110/Documents/_edgellm_ref/repo` tag **`v0.10.0`** (`71dd1ba`) |
| `llm_build` | `/home/impulse110/Documents/_edgellm_ref/repo/build/examples/llm/llm_build` (already built) |
| `visual_build` | `.../build/examples/multimodal/visual_build` |
| Plugin | `.../build/libNvInfer_edgellm_plugin.so` |
| Disk `/` | NVMe `nvme0n1p1` **113G total, 80G used, 27G avail (76%)** |
| RAM (`free -h`) | **7.4 GiB total**, 2.6 GiB used, 3.7 GiB free, 4.5 GiB available |
| `/proc/meminfo` MemTotal | **7,802,736 kB (~7.44 GiB)** — not a clean 8 |
| tegrastats RAM (idle, 2 samples) | **2862–2865 / 7620 MB**; GR3D 0%; GPU ~48 C |
| Swap | `/ssd/32GB.swap` **32 GiB**, ~973 MiB used |
| Voice (Stage F) | `jetbot-wt-voice-sherpa` Zipformer+Piper sherpa-onnx: worktree **154 MiB**, models **64 MiB** (zipformer 28 MiB + piper 37 MiB). Present; not loaded for this inventory. |
| Motors | `jetbot.Robot` / PCA9685 **not opened**. Do not drive until orchestrator is ready. |

`llm_build --help` does **not** accept `--externalize-weights` or `--int4-gemm-plugin-version`. Those are **export** flags. This binary accepts `--onnxDir`, `--engineDir`, `--maxBatchSize`, `--maxInputLen`, `--maxKVCacheCapacity`.

## Memory budget (locked)

Cosmos-Reason2-2B is **Qwen3-VL-class**. NVIDIA Nano 8 GB published proxy: **Qwen3-VL-2B INT4 + FP16 ViT ≈ 4.38 GiB GPU mem**, not 2.1 GiB (2.1 GiB is weights, not a running engine). Budget **4.3–4.7 GiB** for the VLM. Usable RAM is **~7.2–7.5 GiB**.

If a loaded Cosmos engine shows **tegrastats ≥ 5.0 GiB**, **stop and report**. Think mode only when parked. Context **4096**, 4–6 text turns, current CSI JPEG 448² only.

Locked process (when artifacts exist): CSI JPEG → resize 448² → Edge-LLM Cosmos in-process; ASR (CPU) → orchestrator → BGE-small CPU → LanceDB; JSON `stop|drive|speak|wait|weather` → `jetbot.Robot` I2C + TTS (CPU). Clamp `vx` 0.22, `wz` 1.0, duration then stop.

Explicitly **not**: ROS2, Nav2, extra HTTP LLM server, MiniLM, sqlite-vec, BGE GPU TRT, TensorRT-LLM, FastPitch/NeMo/Riva, Hermes Agent, live Jina CLIP / SmolVLA / Gemma 4 audio / Qwen3-Embedding, PyTorch/`transformers` on this Nano, thinking while moving.

## Exact on-device engine build (run only after ONNX arrives)

Export-time (workstation, already required in the ONNX tree): `--externalize-weights int4_ffn --int4-gemm-plugin-version 1`.

On this Jetson:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export EDGELLM_ROOT=$HOME/Documents/_edgellm_ref/repo
export EDGELLM_PLUGIN_PATH=$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4

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
./scripts/bringup/g_cosmos_llm_build.sh
```

Build LLM then ViT **sequentially**, nothing else large resident (stop voice/agent). Sample `tegrastats` throughout. Do not copy x86 `.engine` files.

## Status

| Step | Result |
| --- | --- |
| HOME `/home/impulse110` | OK |
| Inventory | This file |
| Cosmos-Reason2 ONNX rsynced | **No** |
| `llm_build` started | **No** (waiting) |
| Cosmos engine loaded / tegrastats | **N/A** |
| Qwen2.5-VL on disk | Present, **unused** for this loop |

When ONNX lands: run `g_cosmos_llm_build.sh`, attach peak RAM/swap, and abort if tegrastats ≥ 5.0 GiB on a loaded inference engine.
