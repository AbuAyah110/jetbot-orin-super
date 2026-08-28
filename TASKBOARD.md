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

Legacy thin-stack tools expect `~/jetbot-thin-stack/jetbot_vlm_agent/`. That
path is retained as a compatibility symlink into the repository's ignored
archive. Its Cosmos/BGE model paths are sibling symlinks into ignored
repository-local data, so no multi-GB ONNX tree is duplicated or tracked.

## Current status — 2026-08-27

| Stage | State | Evidence / gate |
| --- | --- | --- |
| A–E hardware + Python | Pass with documented notes | [bringup index](docs/bringup/README.md) |
| F voice | Zipformer + Piper CPU integrated; no model loaded in this pass | [voice issue list](https://github.com/AbuAyah110/jetbot-orin-super/issues?q=is%3Aissue+stage-f) |
| G1 TensorRT | Pass | [#16](https://github.com/AbuAyah110/jetbot-orin-super/issues/16) |
| G-Cosmos export | Reported complete in migrated workstation notes | [#37](https://github.com/AbuAyah110/jetbot-orin-super/issues/37) |
| G-Cosmos rsync | ONNX present on device; INT4 FFN + FP16 vision, Edge-LLM 0.10.0 | `rsync -avP --checksum ...` for revalidation |
| G-Cosmos Jetson build | **Pass** — SM87 `llm.engine` 777 MiB + `visual.engine` 785 MiB | `bash ~/jetbot-thin-stack/jetbot_vlm_agent/scripts/JETSON_BUILD.sh` |
| G-Cosmos load / RAM | **Pass** — peak 5441/7620 MB, Cosmos delta **2.88 GiB**, under the 5.0 GiB abort | `tegrastats-cosmos-load.txt` |
| One-process scaffold | Parser/camera/prompts + look-then-log resident loader | `pytest tests/unit/test_look_then_log.py` |
| Robot integration | **Talk-and-drive live** — VAD listen, no beep; PWM via `jetbot.Robot` | `scripts/bringup/talk_and_drive.py` |
| Conversation | **Live** — general Q&A, fresh-frame visual follow-ups, five-exchange persistent text memory | [12-natural-conversation.md](docs/bringup/12-natural-conversation.md) |
| Five demos | **Wired** — show-and-tell, occupancy creep, deictic refuse, parked think, eyes-first where-is, text places | [13-five-demos.md](docs/bringup/13-five-demos.md) |
| Motion request routing | **Fixed** — motion verbs veto parked question routes; speak-only replies can no longer claim movement | [12-natural-conversation.md](docs/bringup/12-natural-conversation.md) |
| Go-around detour | **Partial** — colour-grounded targets only; honest refusal otherwise | `scripts/bringup/talk_and_drive.py` |
| Monocular path gate | **Does not work** — every prompt wording is a constant; blocked on ToF/bumper | `scripts/bringup/probe_path_gate.py` |
| Resume after power | User unit enabled; 20 s delay then same loop | [11-resume-after-power.md](docs/bringup/11-resume-after-power.md) |
| Memory | **Live** — 32.5 MiB CPU INT8 BGE, LanceDB float16 vectors, explicit teach + restart recall passed | [09-memory.md](docs/bringup/09-memory.md) |

Cosmos residency measured 2026-08-27: baseline **2487 / 7620 MB**, peak with the
LLM + visual engines resident **5441 / 7620 MB**, so the Cosmos delta is
**2.88 GiB** — below the 4.3–4.7 GiB planning band and below the 5.0 GiB abort
threshold, so KV stays at 4096. The system-wide peak still includes ~1.5–1.9 GiB
of Cursor remote. Swap peak 1575/32768 MB. See
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
The obsolete Qwen/llama.cpp prototype is closed as superseded:
[#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17).

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

### 3. Jetson — build SM87 engines — **done 2026-08-27**

```bash
export TENSORRT_EDGELLM_ROOT="$HOME/TensorRT-Edge-LLM"
export COSMOS_ONNX_DIR="$HOME/jetbot-thin-stack/cosmos-onnx"
export COSMOS_ENGINE_DIR="$HOME/jetbot-thin-stack/cosmos-engines"
bash ~/jetbot-thin-stack/jetbot_vlm_agent/scripts/JETSON_BUILD.sh
```

`scripts/bringup/JETSON_BUILD.sh` is the tracked builder, mirrored into the
thin-stack `scripts/` directory. It runs Edge-LLM v0.10.0 `llm_build` with
batch 1, input 3072, KV 4096, then `visual_build` at 64/280/280 with FP16 ViT.
`--externalize-weights` and `--int4-gemm-plugin-version` are export-time flags
that this builder rejects; INT4 is already in the ONNX. No FP8, no NVFP4, no
`--memPoolSize`, no on-device re-quantization. It exits 2 without ONNX and 4 if
the ONNX is not INT4-FFN externalized.

Gate **passed**: `data/edgellm/cosmos/engines/llm/llm.engine` (777 MiB) and
`engines/visual/visual.engine` (785 MiB); LLM engine generation 103.5 s, visual
49.1 s; Cosmos load delta 2.88 GiB against the 5.0 GiB abort threshold.

Generate length is a runtime setting, not a build flag: drive 64–96 tokens at
temperature 0, parked think 256–512, never think while `vx != 0`.

### 4. One-process parked robot integration

- [x] One import-safe CSI JPEG class; one Argus/GStreamer pipeline, 448×448.
- [x] Strict JSON parser for `stop|drive|speak|wait|weather`; invalid → stop.
- [x] Velocity/duration clamps and explicit duration-then-stop contract.
- [x] Drive and parked-think prompt suffix helpers.
- [x] `CosmosRuntime` refuses in-process TRT map; look-then-log uses `cosmos_resident`.
- [x] Import-safe BGE/LanceDB stubs; no model fetch.
- [x] Wire a resident Edge-LLM generate loop (file protocol, no HTTP) after isolated inference.
- [ ] Wire Zipformer/Piper CPU without adding another process.
- [ ] Wire only the bounded action executor to `jetbot.Robot`.
- [x] Run look-then-log (4 ticks, stop held, no motors); first driving episode still open.

### 5. CPU memory

- [x] Consolidate the existing 127 MiB BGE-small ONNX candidate under ignored `data/models/` without loading it.
- [ ] Verify that BGE ONNX on CPU only after Cosmos residency passes.
- [ ] Tokenize/embed on CPU; never install `transformers` or PyTorch.
- [ ] Persist vectors in repository-local ignored LanceDB data.
- [ ] Measure combined Cosmos + voice + BGE/LanceDB RAM before enabling recall.

## Repository layout

| Path | Git status | Purpose |
| --- | --- | --- |
| `jetbot_agent/robot_loop/` | tracked | Locked loop parser, prompts, camera, runtime/memory stubs |
| `scripts/JETSON_IDLE_RAM.sh` | tracked | Idempotent headless-memory preparation |
| `scripts/bringup/JETSON_BUILD.sh` | tracked | Locked-flag SM87 engine builder (mirrored into thin-stack) |
| `scripts/bringup/llm_build_cosmos.sh` | tracked | Repo-default wrapper around `JETSON_BUILD.sh` |
| `docs/bringup/07-cosmos-nano.md` | tracked | Inventory, RAM, rsync/build evidence |
| `third_party/tensorrt-edge-llm/` | ignored | Full Edge-LLM v0.10.0 clone/build |
| `data/edgellm/cosmos/` | ignored | ONNX, external weights, SM87 engines, logs |
| `~/jetbot-thin-stack/jetbot_vlm_agent/` | external symlink | Legacy thin-stack compatibility path; production source is not here |
| `~/jetbot-thin-stack/{cosmos-onnx,cosmos-engines,bge-small-en-v1.5-onnx}` | external symlinks | Compatibility aliases into ignored repo data; no duplicate bytes |

Never commit weights, ONNX, engines, GGUF, safetensors, Edge-LLM source/build,
virtual environments, or generated LanceDB tables.
