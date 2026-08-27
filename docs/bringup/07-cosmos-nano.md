# Stage G — Cosmos-Reason2-2B on Orin Nano (Edge-LLM)

Date: 2026-08-27 (cleanup)  
Branch: `stage-g-cosmos-nano`  
Ticket: [#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17) is now the **Cosmos-Reason2-2B** robot VLM (Qwen artifacts deleted on device). Related: [#37](https://github.com/AbuAyah110/jetbot-orin-super/issues/37).

This is the **locked robot VLM**: Cosmos-Reason2-2B via **TensorRT Edge-LLM v0.10.0**, in-process, INT4 LLM + FP16 ViT, **maxBatchSize 1**, **maxInputLen 3072**, **maxKVCacheCapacity 4096**. Do **not** quantize on this board. Do **not** install PyTorch. Do **not** start a Qwen rebuild. Do **not** load Hub **FP8 / NVFP4** Cosmos checkpoints (SM87 cannot run them).

**Engines are built and loaded as of 2026-08-27 13:24 CDT.** A later VLM pass (14:31 CDT) ran one 448² CSI JPEG through `llm_inference` in drive mode. Motors were **not** driven and no live camera loop was left running.

## Thin-stack path is the operator entry point

The thin-stack tree the operator runs from is:

```text
~/jetbot-thin-stack/jetbot_vlm_agent/scripts/JETSON_BUILD.sh
~/jetbot-thin-stack/cosmos-onnx/{llm,visual}
~/jetbot-thin-stack/cosmos-engines/{llm,visual}
```

Those are **compatibility symlinks**, not a second copy. `jetbot_vlm_agent`
resolves to the ignored `data/archive/legacy-thin-stack/`, and the model paths
resolve to ignored `data/edgellm/cosmos/{onnx,engines}`. No multi-GB payload is
duplicated or tracked. `~/TensorRT-Edge-LLM` is likewise a symlink to
`third_party/tensorrt-edge-llm`, so the documented default
`TENSORRT_EDGELLM_ROOT` resolves on this device.

**This repository stays canonical for source.** The tracked robot loop is
`jetbot_agent/robot_loop/`; the tracked builder is
`scripts/bringup/JETSON_BUILD.sh`, mirrored into the thin-stack `scripts/`
directory so either invocation runs identical flags. Do not treat the legacy
tree as a second source checkout.

## SM87 engine build — pass (2026-08-27)

Built on this Jetson from the x86 INT4 ONNX. `aarch64` confirmed; Edge-LLM pin
`v0.10.0`; TensorRT 10.3.0.30; CUDA 12.6 on `PATH`;
`EDGELLM_PLUGIN_PATH=third_party/tensorrt-edge-llm/build/libNvInfer_edgellm_plugin.so`.

Exact commands (the only flags `llm_build` v0.10.0 accepts for this model):

```bash
export TENSORRT_EDGELLM_ROOT="$HOME/TensorRT-Edge-LLM"
export COSMOS_ONNX_DIR="$HOME/jetbot-thin-stack/cosmos-onnx"
export COSMOS_ENGINE_DIR="$HOME/jetbot-thin-stack/cosmos-engines"
cd "$TENSORRT_EDGELLM_ROOT"

./build/examples/llm/llm_build \
  --onnxDir "$COSMOS_ONNX_DIR/llm" \
  --engineDir "$COSMOS_ENGINE_DIR/llm" \
  --maxBatchSize 1 \
  --maxInputLen 3072 \
  --maxKVCacheCapacity 4096

./build/examples/multimodal/visual_build \
  --onnxDir "$COSMOS_ONNX_DIR/visual" \
  --engineDir "$COSMOS_ENGINE_DIR" \
  --minImageTokens 64 \
  --maxImageTokens 280 \
  --maxImageTokensPerImage 280
```

`--externalize-weights int4_ffn` and `--int4-gemm-plugin-version 1` are
**export-time** flags. This binary rejects them (`unrecognized option`), and the
INT4 layout is already inside the ONNX sidecars. No FP8, no NVFP4, no
`--memPoolSize`, and no re-quantization on the Nano.

| Artifact | Path (canonical) | Size | SHA256 |
| --- | --- | --- | --- |
| LLM engine | `data/edgellm/cosmos/engines/llm/llm.engine` | 777 MiB | `fdc2c15e8f62fa0756845ae8d96e6e7ffce5a3b535a299fc239769a631b5b0f7` |
| Visual engine | `data/edgellm/cosmos/engines/visual/visual.engine` | 785 MiB | `3b5e8b8afff9677842e06026abaaeedabc47457d55cfb6a74fd7e6c72c8abf3b` |

Engine directory totals 2.7 GiB (engine files plus `embedding.safetensors`
594 MiB, `external_int4_ffn_weights.safetensors` 520 MiB, tokenizer, configs).
`df -h /` after the build: **78G used / 30G avail (73%)**.

Builder-recorded configuration, read back from the generated `config.json`:

| Key | LLM engine | Visual engine |
| --- | --- | --- |
| `edgellm_version` | `0.10.0` | — |
| `max_batch_size` | **1** | — |
| `max_input_len` | **3072** | — |
| `max_kv_cache_capacity` | **4096** | — |
| `max_kv_pool_pages` | 32 | — |
| `kv_cache_dtype` | **fp16** (not FP8) | — |
| `external_weight_files[].kind` | **`int4_ffn_weights`** | — |
| `min/max_image_tokens` | — | **64 / 280**, per-image **280** |
| `vision_config.dtype` | — | **float16** |

Build cost: LLM engine generation 103.5 s, peak TRT allocators CPU 2 MiB /
GPU 1732 MiB, peak host during build+serialize 6300 MiB. Visual engine
generation 49.1 s, peak allocators CPU 8 MiB / GPU 789 MiB, peak host
4441 MiB. TensorRT skipped a 3456 MiB tactic against ~1890 MiB free and fell
back; that is a warning, not a failure, and the build completed.

## Cosmos load RAM — 2.88 GiB delta, under the 5.0 GiB halt line

`llm_inference` loaded **only** the Cosmos LLM engine plus the visual engine
(`--multimodalEngineDir`), text prompt only, temperature 0. The action runner is
absent by design and its load failure is logged and ignored. tegrastats sampled
at 500 ms, baseline captured before the process started:

| Measure | Value |
| --- | --- |
| Baseline RAM (Cursor still connected) | **2487 / 7620 MB** |
| Peak RAM with Cosmos resident | **5441 / 7620 MB** (5.31 GiB system-wide) |
| **Cosmos delta over baseline** | **2954 MB = 2.88 GiB** |
| Swap peak | 1575 / 32768 MB |
| First load pass (independent) | baseline 2110 MB, peak 5403 MB, delta **3.22 GiB** |
| TRT execution-context GPU allocation | 1549 MiB |
| Shared execution-context memory | 161,483,264 B (base 161,483,264; vision 57,917,440) |

Cosmos sits **below** the 4.3–4.7 GiB planning band and well below the
**5.0 GiB abort threshold**, so the build was not halted and KV stays at 4096.
The system-wide 5.31 GiB peak still includes ~1.5–1.9 GiB of Cursor remote,
which disappears when the editor disconnects. Do not read 5.31 GiB as the
Cosmos footprint.

Functional sanity: a text request returned `ok` and a three-request drive-mode
batch (temperature 0, `max_generate_length` 96) returned 3/3 successful with
`finish_reason: end-of-sequence`.

Generate length is a **runtime** setting owned by the orchestrator, not a build
flag: drive mode 64–96 tokens at temperature 0, parked think mode 256–512.
Never think while `vx != 0`.

Logs (ignored, on-device): `data/edgellm/cosmos/logs/JETSON_BUILD.log`,
`JETSON_BUILD_verify.log`, `llm_inference_load.log`,
`llm_inference_drive.log`, `tegrastats-build.txt`,
`tegrastats-cosmos-load.txt`, `vlm_drive_*.json`, `vlm_drive_inference.log`,
`tegrastats-vlm-drive.txt`.

## VLM CSI JPEG smoke — pass (2026-08-27 14:31 CDT)

Resident PID 399034 (`llm_inference` writing `/tmp/cosmos_reason2_resident.fifo`) could not take another prompt. The FIFO was **not** drained; that process was killed and a fresh `llm_inference` loaded the same engines. One CSI frame was captured with a single pipeline (`nvarguscamerasrc` `num-buffers=1`, 1280×720 NVMM → `nvvidconv` 448² → `nvjpegenc`), not a second 1080p stream. Drive mode: **no think suffix**, `temperature` 0, `max_generate_length` 96, `batch_size` 1, `warmup` 0.

| Item | Result |
| --- | --- |
| Verdict | **Pass** — model returned text with `action":"stop"`; motors not opened |
| Image | `data/edgellm/cosmos/logs/csi448_vlm_test.jpg` (448×448, 17.8 KiB) |
| Prompt (user) | Drive-mode JSON-only turn: look at the current 448² frame and choose a safe action (`stop\|drive\|speak\|wait\|weather`). System message is the JetBot JSON schema. |
| Output | `{"action":"stop","vx":0.01,"wz":0.01,"duration_s":0.13,"say":"\\","goal":"avoid red object","reason":"The red object is in the path, causing a potential collision."}` |
| JSON | **Invalid** (`say` truncated / unescaped quote). Schema kind is still recognizable as `stop`. |
| finish_reason | `end-of-sequence`; 1/1 successful |
| Vision | 1 image, **196** image tokens, encoder **180 ms** |
| Prefill | 426 tokens, **174 ms**, 2449 tok/s |
| Generation | **54** tokens, **45.6 tok/s**, 21.9 ms/token |
| TTFT (approx.) | **~354 ms** (vision 180 + prefill 174). Runtime did not emit a named TTFT field. E2E request ~1.54 s. |
| tegrastats RAM | baseline after kill **2590 / 7620 MB**; peak **5613 / 7620 MB** (5.48 GiB board total); **delta 3023 MB = 2.95 GiB** |
| Swap | 1503 / 32768 MB (unchanged) |
| Edge-LLM peak UMA | 1101 MB (profiler) |

Board total crossed 5 GiB because Cursor remote is still connected (~2.4 GiB used before this load). The VLM-only delta stayed **~2.95 GiB**, under the 5.0 GiB abort. Did not halt.

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

## Cosmos ONNX: discovered in a stray tree; checksum rsync still required

The initial canonical-path check was accurate but incomplete: no ONNX existed
under `~/tensorrt-edgellm-workspace`. A workspace-wide cleanup later found a
complete-looking export under `/home/impulse110/jetbot-thin-stack/cosmos-onnx/`,
created at 11:40 CDT. It has been moved (not committed) to:

```
/home/impulse110/Documents/jetbot-orin-super/data/edgellm/cosmos/
  onnx/llm/
  onnx/visual/
  engines/llm/
  engines/visual/
  logs/
```

Inventory: 3.2 GiB total; LLM config says Edge-LLM `0.10.0`,
`int4_ffn_weights`, 28 layers, hidden size 2048, and FP16 vision. Key hashes:

- `llm/model.onnx`: `84a6f7bd42d3bc38e74b0fb3c532af8ce3fbab54483fe473b862497ca41d5780`
- `visual/model.onnx`: `6e8df49afccb939e99cf2d5a1a68c2cdd5f75f5b5e64f6cc88751435a175cb75`

The exact workstation `rsync --checksum` command below remains required to
validate/replace this tree. Engines are built on this Jetson; never copy x86
`.engine` files.

A local BGE-small CPU candidate was also found and moved to the ignored
`data/models/bge-small-en-v1.5-onnx/`: 127 MiB ONNX, 384-dimensional normalized
CLS output, plus a 696 KiB tokenizer. It was not imported or loaded.

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
| TensorRT Edge-LLM | `third_party/tensorrt-edge-llm/` tag **`v0.10.0`** (ignored full clone/build) |
| `llm_build` | `third_party/tensorrt-edge-llm/build/examples/llm/llm_build` (already compiled) |
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

Workstation export is reported complete in the migrated legacy notes. Use the
exact checksum rsync below to validate the relocated tree before the deliberate
build. Do not infer correctness from filenames alone.

### Idle RAM this session (2026-08-27)

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

For compatibility, the old workspace is a symlink into repository-local data:

```bash
mkdir -p /home/impulse110/Documents/jetbot-orin-super/data/edgellm/cosmos
ln -s /home/impulse110/Documents/jetbot-orin-super/data/edgellm \
  ~/tensorrt-edgellm-workspace
# data/edgellm/Cosmos-Reason2-2B is a local symlink to data/edgellm/cosmos
```

Legacy thin-stack scripts and notes still resolve at:

```text
~/jetbot-thin-stack/jetbot_vlm_agent/
```

That directory is a compatibility symlink to
`data/archive/legacy-thin-stack/`. The sibling legacy model names
`cosmos-onnx`, `cosmos-engines`, and `bge-small-en-v1.5-onnx` are symlinks to
the ignored canonical payload directories. They consume no additional model
storage. Production code remains under tracked `jetbot_agent/robot_loop/`.

Do not commit the Edge-LLM clone, ONNX, external data, engines, weights, or
virtual environments. See `data/edgellm/README.md`.

### Exact workstation rsync reminder

Run on the workstation:

```bash
rsync -avP --checksum ~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4/onnx/ impulse110@192.168.50.65:~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/
```

### Earlier interrupted build attempt

A first `llm_build` run started at 12:59:32 CDT from the thin-stack paths and
died at 13:01 when those paths were relocated mid-run, before any engine was
written. Its logs are retained under ignored
`data/archive/unexpected-build-attempt-2026-08-27/`. The successful build
documented above is a clean restart from the same ONNX after the compatibility
symlinks were in place.

## Rebuilding the engines

`scripts/bringup/JETSON_BUILD.sh` is the tracked builder and is mirrored to
`~/jetbot-thin-stack/jetbot_vlm_agent/scripts/JETSON_BUILD.sh`. It refuses
non-`aarch64`, exits 2 without `onnx/llm/model.onnx`, and exits 4 if the ONNX is
not INT4-FFN externalized or advertises FP8/NVFP4 KV. `SKIP_IF_ENGINES=1`
verifies without rebuilding. `scripts/bringup/llm_build_cosmos.sh` wraps it with
repository-local defaults.

```bash
export TENSORRT_EDGELLM_ROOT="$HOME/TensorRT-Edge-LLM"
export COSMOS_ONNX_DIR="$HOME/jetbot-thin-stack/cosmos-onnx"
export COSMOS_ENGINE_DIR="$HOME/jetbot-thin-stack/cosmos-engines"
bash ~/jetbot-thin-stack/jetbot_vlm_agent/scripts/JETSON_BUILD.sh
```

Build LLM then ViT **sequentially**, nothing else large resident (stop voice/agent). Sample `tegrastats` throughout. Do not copy x86 `.engine` files.

## Status

| Step | Result |
| --- | --- |
| HOME `/home/impulse110` | OK (Jetson) |
| Qwen2.5-VL on disk | **Deleted** |
| Cosmos-Reason2 ONNX | **Present**; INT4 FFN + FP16 vision, Edge-LLM 0.10.0 |
| `llm_build` (SM87, 3072 / 4096, batch 1) | **Pass** — `llm.engine` 777 MiB |
| `visual_build` (64 / 280 / 280, FP16 ViT) | **Pass** — `visual.engine` 785 MiB |
| Cosmos engine loaded / tegrastats | **Pass** — peak 5441/7620 MB, **delta 2.88 GiB**, under the 5.0 GiB abort |
| Text inference sanity | **Pass** — 3/3 drive-mode requests, temperature 0, 96 tokens |
| VLM + 448² CSI JPEG | **Pass** — 1 image, 196 vis tokens, text `action=stop`; JSON `say` invalid |
| Motors / camera loop | **Not started** (one-shot CSI JPEG only; no PWM) |
| BGE | 127 MiB local CPU ONNX candidate in ignored data; not loaded |

Remaining for Stage G: wire the Edge-LLM loader into
`jetbot_agent/robot_loop/` in-process (no HTTP sidecar). One-shot 448² CSI JPEG
inference already passed; a parked loop with duration-stop motors is still open.
