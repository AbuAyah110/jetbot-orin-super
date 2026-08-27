# Task board — locked Cosmos robot plan

GitHub: [repository](https://github.com/AbuAyah110/jetbot-orin-super) ·
[project](https://github.com/users/AbuAyah110/projects/2) ·
[issues](https://github.com/AbuAyah110/jetbot-orin-super/issues) ·
[milestones](https://github.com/AbuAyah110/jetbot-orin-super/milestones)

This file is the source of truth for the WaveShare JetBot / Jetson Orin Nano
Super deployment. Close an issue only after its verify gate passes on the
Jetson. No model may write PWM directly.

## Architecture of record

One Linux process owns the robot loop:

```text
CSI/Argus → one 448×448 JPEG pipeline → Cosmos-Reason2-2B Edge-LLM
Zipformer ASR (CPU) ──────────────────→ JSON action parser
                                              │
                     stop | drive | speak | wait | weather
                                              │
                jetbot.Robot I2C + Piper TTS (CPU)
                                              │
                         BGE-small CPU → LanceDB
```

Locked limits: `abs(vx) <= 0.22`, `abs(wz) <= 1.0`; every drive has a bounded
duration followed by stop. Extended thinking is parked-only. Cosmos uses one
in-process TensorRT Edge-LLM v0.10.0 runtime: INT4 LLM + FP16 ViT,
`maxBatchSize=1`, `maxInputLen=3072`, `maxKVCacheCapacity=4096`.

Explicitly not in the robot loop: PyTorch, `transformers`, TensorRT-LLM, Hermes
64k, ROS 2, Nav2, RViz, live Jina CLIP, live SmolVLA, Qwen2.5-VL, llama.cpp, an
extra HTTP LLM server, GPU BGE, or a second CSI pipeline. Qwen2.5-VL and
llama.cpp artifacts were deleted from this device.

## Current status — 2026-08-27

| Stage | State | Evidence / gate |
| --- | --- | --- |
| A–E hardware + Python | Pass with documented notes | [bringup index](docs/bringup/README.md) |
| F voice | Zipformer + Piper CPU branch exists; consolidate into this branch | [voice issue list](https://github.com/AbuAyah110/jetbot-orin-super/issues?q=is%3Aissue+stage-f) |
| G1 TensorRT | Pass | [#16](https://github.com/AbuAyah110/jetbot-orin-super/issues/16) |
| G-Cosmos export | In progress on workstation | [#37](https://github.com/AbuAyah110/jetbot-orin-super/issues/37) |
| G-Cosmos rsync | Ready; ONNX absent on Jetson | `test -f data/edgellm/cosmos/onnx/llm/model.onnx` |
| G-Cosmos Jetson build | Blocked only on rsync | `./scripts/bringup/llm_build_cosmos.sh` |
| One-process scaffold | Parser/camera/prompts/stubs implemented; no model loaded | `pytest tests/unit/test_robot_loop_actions.py` |
| Robot integration | Pending engines + safe stopped integration | Scripted parked episode; no direct PWM |
| Memory | Stub only; no BGE weights downloaded | CPU BGE vector round-trip in LanceDB |

Post-idle-script baseline with Cursor connected: `free -h` **2.5 GiB used /
4.7 GiB available**; tegrastats **2783–2784 / 7620 MB**, GR3D 0%; swap
507/32768 MB; swappiness 10. Docker, snapd, and gdm are inactive. See
[Cosmos Nano bring-up](docs/bringup/07-cosmos-nano.md).

## Ordered execution

### 1. Workstation — export Cosmos INT4 ONNX

- [ ] Pin NVIDIA TensorRT Edge-LLM **v0.10.0**.
- [ ] Quantize/export `nvidia/Cosmos-Reason2-2B` on the x86 workstation.
- [ ] Use `--externalize-weights int4_ffn --int4-gemm-plugin-version 1`.
- [ ] Reject FP8 and NVFP4 exports; Orin Nano is SM87.
- [ ] Produce `onnx/llm/model.onnx`, external data, and visual ONNX.
- [ ] Generate and verify checksums.

Tracking: [#37](https://github.com/AbuAyah110/jetbot-orin-super/issues/37).
The obsolete Qwen/llama.cpp prototype is tracked historically by
[#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17), not as a
deployment dependency.

### 2. Workstation → Jetson rsync

```bash
rsync -avP --checksum ~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4/onnx/ impulse110@192.168.50.65:~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/
```

The compatibility destination points to repository-local ignored data:
`data/edgellm/cosmos/onnx/`. Do not copy x86 TensorRT engines.

Gate:

```bash
test -f /home/impulse110/Documents/jetbot-orin-super/data/edgellm/cosmos/onnx/llm/model.onnx
```

### 3. Jetson — build SM87 engines

```bash
./scripts/bringup/llm_build_cosmos.sh
```

The script exits **2** while `onnx/llm/model.onnx` is absent. Once present it
runs Edge-LLM v0.10.0 `llm_build` with batch 1, input 3072, KV 4096. Export
flags are recorded by the script but are not passed to the v0.10.0 builder,
which does not accept them. Build sequentially with no voice/model resident.

Gate: engine files exist under `data/edgellm/cosmos/engines/`; attach peak RAM
and swap. Stop and report if a loaded Cosmos runtime reaches 5.0 GiB in
tegrastats.

### 4. One-process parked robot integration

- [x] One import-safe CSI JPEG class; one Argus/GStreamer pipeline, 448×448.
- [x] Strict JSON parser for `stop|drive|speak|wait|weather`; invalid → stop.
- [x] Velocity/duration clamps and explicit duration-then-stop contract.
- [x] Drive and parked-think prompt suffix helpers.
- [x] `CosmosRuntime` raises `StageNotReady`; no engine mapping.
- [x] Import-safe BGE/LanceDB stubs; no model fetch.
- [ ] Wire the Edge-LLM loader after engines pass isolated inference.
- [ ] Wire Zipformer/Piper CPU without adding another process.
- [ ] Wire only the bounded action executor to `jetbot.Robot`.
- [ ] Run first episode with wheels raised, then repeat parked.

### 5. CPU memory

- [ ] Add a locally verified small BGE ONNX only after Cosmos residency passes.
- [ ] Tokenize/embed on CPU; never install `transformers` or PyTorch.
- [ ] Persist vectors in repository-local ignored LanceDB data.
- [ ] Measure combined Cosmos + voice + BGE/LanceDB RAM before enabling recall.

## Repository layout

| Path | Git status | Purpose |
| --- | --- | --- |
| `jetbot_agent/robot_loop/` | tracked | Locked loop parser, prompts, camera, runtime/memory stubs |
| `scripts/JETSON_IDLE_RAM.sh` | tracked | Idempotent headless-memory preparation |
| `scripts/bringup/llm_build_cosmos.sh` | tracked | Guarded on-device builder |
| `docs/bringup/07-cosmos-nano.md` | tracked | Inventory, RAM, rsync/build evidence |
| `third_party/tensorrt-edge-llm/` | ignored | Full Edge-LLM v0.10.0 clone/build |
| `data/edgellm/cosmos/` | ignored | ONNX, external weights, SM87 engines, logs |

Never commit weights, ONNX, engines, GGUF, safetensors, Edge-LLM source/build,
virtual environments, or generated LanceDB tables.
