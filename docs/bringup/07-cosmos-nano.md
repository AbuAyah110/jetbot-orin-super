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

## Exact on-device engine build (run only after ONNX arrives)

Export-time (workstation, already required in the ONNX tree): `--externalize-weights int4_ffn --int4-gemm-plugin-version 1`.

On this Jetson:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
export EDGELLM_ROOT=$HOME/Documents/_edgellm_ref/repo
export EDGELLM_PLUGIN_PATH=$EDGELLM_ROOT/build/libNvInfer_edgellm_plugin.so
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace/Cosmos-Reason2-2B

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
| HOME `/home/impulse110` | OK (Jetson) |
| Qwen2.5-VL on disk | **Deleted** |
| Cosmos-Reason2-2B rsync dir | **Ready** (`~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/`) |
| Cosmos-Reason2 ONNX rsynced | **No** |
| `llm_build` started | **No** (waiting on workstation ONNX) |
| Cosmos engine loaded / tegrastats | **N/A** |
| BGE | Not on disk (CPU later) |

When ONNX lands: run `g_cosmos_llm_build.sh`, attach peak RAM/swap, and abort if tegrastats ≥ 5.0 GiB on a loaded inference engine.
