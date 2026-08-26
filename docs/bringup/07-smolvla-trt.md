# Stage G3 — SmolVLA TensorRT path (hypothesis vs this board)

**Verdict: Gemini's plan is not viable as written. Partial pieces are real (FP16 is allowed, VIC resize exists, TRT Python works); the architecture, I/O, runtime stack, and `--memPoolSize` claim are wrong.**

The least-memory path that is **actually reachable** on this Orin Nano Super 8 GB is still **eager PyTorch `SmolVLAPolicy` after issue #30**, dummy tensors, no PWM. A TensorRT FP16 pair of subgraphs (prefix VLM + denoise expert, Python Euler loop, ctypes `libcudart`) is the highest-leverage *future* memory win **if** ONNX export of those two graphs ever works — it was not demonstrated here. INT8 PTQ is later. Do not close [#18](https://github.com/AbuAyah110/jetbot-orin-super/issues/18) until a dummy forward actually runs.

This is a separate lane from Qwen / llama.cpp. It does not change VLM modules.

Sources: LeRobot `modeling_smolvla.py` / `configuration_smolvla.py` (huggingface/lerobot `main`, 2026-08-26), `lerobot/smolvla_base` `config.json` (fetched, weights not downloaded), G1 evidence [07-tensorrt-g1.md](07-tensorrt-g1.md), TensorRT **10.3.0** Python bindings probed on this board, `trtexec --help` for `--memPoolSize`.

## Gemini claim vs reality

| # | Gemini | This board / upstream |
| --- | --- | --- |
| 1 | `torch.onnx.export` the **whole** policy with dummy `(image 1×3×224×224, text ids 1×16)` and dynamic batch/seq | **False.** `smolvla_base` takes **three** cameras at **3×256×256**, `resize_imgs_with_padding = [512, 512]`, language **length 48**, state dim 6 padded to 32, and emits a **(50, 6) action chunk**. Inference is `sample_actions` → prefix VLM + **10 Euler steps** of `denoise_step`, not one feed-forward. A two-tensor "image + tokens → wheel speed" export is architecturally false. Stale comment in `prepare_images` still says 224²; **config.json wins (512)**. |
| 2 | Transfer ONNX, build FP16 on Jetson in 10–20 min | **Unmeasured / premature.** No published ONNX (G1). Torch is **absent** (#30). Builder for a **68 KB** graph already peaked **~1.5 GB RSS** (G1); a 450M-param graph will be worse. If export ever exists, **build on-device with the VLM unloaded**, or export ONNX on x86 and only run `trtexec` here. Do not co-resident torch + weights + ONNX proto + TRT builder + Qwen. |
| 3 | GStreamer VIC resize to 224², skip OpenCV | **Partial.** `nvvidconv` is real VIC hardware and is already in the working CSI pipeline. **224 is the historic JetBot SSD size, not SmolVLA.** Policy resize is 512 with pad. Working strings still use **`videoconvert` + OpenCV `appsink`** after VIC. Skipping OpenCV is optional later, not a G3 gate. |
| 4 | Runtime = **pycuda + tensorrt**, PyTorch never inits its 500 MB allocator, resident **~1.3 GB**, latency **&lt;30 ms** | **Mostly false or unmeasured.** `pycuda` is **absent**; G1 used **ctypes `libcudart`**. `pip install pycuda tensorrt` from PyPI is the wrong ABI (Tegra vs SBSA). Apt `python3-libnvinfer` + `PYTHONPATH=/usr/lib/python3.10/dist-packages` is the proven path. **1.3 GB and &lt;30 ms were not measured.** One VLM prefix (SmolVLM2-500M, 16 layers, 512²) plus **10** expert steps on Orin Nano is unlikely to beat 30 ms end-to-end; mark unmeasured. Dropping torch's caching allocator is a real *hypothesis* for memory, not a number. |

## Real I/O (`lerobot/smolvla_base`)

Checkpoint named in the old spec (`smolvla-jetbot`) **does not exist**. Only `lerobot/smolvla_base` (~450M params, ~0.9 GB BF16 safetensors). Tiny config copy: [`scripts/bringup/smolvla_base.config.json`](../../scripts/bringup/smolvla_base.config.json).

| Tensor | Shape (B=1) | Notes |
| --- | --- | --- |
| `observation.images.camera1` | `1×3×256×256` then pad to `512×512` | SigLIP-style `[-1, 1]` after `*2-1` in `prepare_images` |
| `observation.images.camera2` | same | Pretrain has **three** cameras; JetBot has **one** IMX219 CSI0 |
| `observation.images.camera3` | same | `empty_cameras=0` — missing cams are omitted, not dummy-filled, unless that flag is raised |
| `observation.state` | `1×6` → pad `1×32` | MEAN_STD norm. Not wheel encoders in the base SO-100 layout |
| `observation.language_tokens` | `1×48` | `tokenizer_max_length=48`, `pad_language_to=max_length` |
| `observation.language_attention_mask` | `1×48` | Required alongside tokens |
| `action` (output) | chunk `1×50×6` (pad dim 32) | **Not** left/right PWM. Map to `cmd_vel` later via I5 seam |

Flow-matching internals (not a single ONNX I/O):

* `num_steps=10` Euler steps (`euler_integrate`, `dt = -1/num_steps`)
* `use_cache=true`: prefix KV computed once; `denoise_step` consumes `past_key_values` and **crops** them back to prefix length each step
* VLM: `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`, `num_vlm_layers=16`, action expert width `0.75×`

## Export feasibility

**Blocked on #30.** This investigation did not install torch and did not download weights.

| Attempt | Feasible? | Why |
| --- | --- | --- |
| Whole `SmolVLAPolicy` / `sample_actions` as one ONNX | **No** | Python `for` over denoising; KV cache object + `.crop()`; list of image tensors; transformers SmolVLM internals; RTC optional branch |
| Training `forward` (noise interpolation + MSE) | Wrong graph | That is the train loss, not inference |
| **Prefix** (`embed_prefix` + VLM `use_cache`) | Maybe later | Static-ish, but SmolVLM/SigLIP ONNX is its own project (attention, dynamic seq) |
| **Action expert `denoise_step`** | Maybe later | Right unit to TRT; still needs KV cache as inputs or a TRT data-dependent cache story |
| INT8 PTQ | Not now | G1: `platform_has_fast_int8` is true. PTQ needs a calibration set of images, tokens, state, noisy actions, timesteps. No dataset on this robot. FP16 first if any engine is built |

**Where to export:** not on-device next to Qwen. Either x86 with CUDA for `torch.onnx.export`, or a dedicated Jetson session with torch + model only. **Where to build the engine:** on this Jetson (`trtexec` at `/usr/src/tensorrt/bin/trtexec`, SM 87, TRT 10.3). Transfer the ONNX, not a desktop-built engine.

Script: [`scripts/bringup/export_smolvla_onnx.py`](../../scripts/bringup/export_smolvla_onnx.py) — **refuses** until torch+lerobot exist; always refuses `--graph policy`.

## `--memPoolSize=workspace:256` is not an inference VRAM cap

G1 already passes `--memPoolSize=workspace:256M` and `IBuilderConfig.set_memory_pool_limit(WORKSPACE, 256 MiB)`.

TensorRT 10 `trtexec --help` on this board: pool is `workspace` (also DLA / tacticSharedMem). Suffixes `B|K|M|G`; **default unit is MiB** if omitted, so Gemini's `workspace:256` is 256 MiB, same order as G1's `256M`.

NVIDIA's own FAQ (TRT issue #4211 / "How TensorRT Works"): `setMemoryPoolLimit(WORKSPACE)` caps **scratch/tactic workspace the builder may consider**, and at runtime TRT allocates **no more workspace than needed**, typically less than the cap. It does **not** include weights, activations (enqueue / device memory), I/O buffers, or CUDA/TRT infrastructure. It does **not** "prevent TensorRT from reserving excessive VRAM during inference" in the sense of a 256 MB process cap.

On this SoC memory is **unified**; `jtop` "VRAM" is not a separate pool. `ICudaEngine.device_memory_size_v2` is the number to read once an engine exists.

A 256 MiB workspace cap on a 450M-param VLM+expert can also **drop fast tactics** ("some tactics do not have sufficient workspace"). Do not copy the flag as a memory-saving slogan without a build log.

## Runtime: pycuda vs G1 ctypes vs TensorRT 10 Python

| Piece | Status |
| --- | --- |
| `python3-libnvinfer` 10.3.0.30 | Installed; `.venv` needs `PYTHONPATH=/usr/lib/python3.10/dist-packages` |
| `pycuda` | **Absent.** No apt `python3-pycuda` installed. Do not `pip install pycuda` from PyPI on Tegra |
| `cuda-python` | Absent (G1) |
| G1 path | ctypes `libcudart.so.12` + `set_tensor_address` + `execute_async_v3` — **proven** on a tiny graph |
| Recommendation | Keep G1's ctypes path. Adding pycuda is extra native build work for no memory win |

KV cache is **not** N/A for SmolVLA: the VLM prefix cache is real (`use_cache=true`). Gemini's "no KV" applies to a feed-forward CNN, not this policy. Activations + expert steps still cost.

## TensorRT 8 APIs in the pasted snippet that **will not run** on TRT 10.3

Probed 2026-08-26 with `python3-libnvinfer` 10.3.0:

| Pasted (TRT 8) | On this board (TRT 10.3) |
| --- | --- |
| `engine.get_binding_shape(...)` | **Missing** on `ICudaEngine` |
| `engine.max_batch_size` | **Missing** |
| `engine.num_bindings` | **Missing** |
| `engine.get_binding_index(...)` | **Missing** |
| `context.set_binding_shape(...)` | **Missing** |
| `context.execute_async_v2(...)` | **Missing** (`execute_async_v3` exists; `execute_v2` exists but is the old bindings enqueue) |
| `engine[binding]` as a buffer index / allocation key | `__getitem__(int) -> str` (**tensor name**, not a pointer). `ICudaEngine.__len__` is **false** — `for b in engine` is not a binding loop |
| implicit-batch `max_batch_size` engines | Explicit-batch / named tensors. G1: `num_io_tensors`, `get_tensor_name`, `get_tensor_mode`, `get_tensor_dtype`, `get_tensor_shape`, `set_input_shape`, `set_tensor_address`, `execute_async_v3(stream_handle=...)` |

Implementing the Gemini snippet here would fail at import/attribute time. The stub in `jetbot_agent/engine/trt_vla_motor.py` uses the G1/TRT-10 names and does not import pycuda.

## Camera / VIC (IMX219 CSI0)

`nvargus-daemon` ~180–190 MB is already in the OS baseline. Do not treat VIC preprocess as "free RAM."

Board-proven fragments (do not invent a new gst string for the dummy gate):

```text
# docs/bringup/03-csi-camera.md (1-buffer EOS)
nvarguscamerasrc sensor-id=0 num-buffers=1
  ! video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1
  ! nvvidconv ! video/x-raw,format=I420 ! fakesink
```

```text
# src/perception/camera.py GstCsiCamera (working Python path)
nvarguscamerasrc sensor-id={sensor} !
  video/x-raw(memory:NVMM), width={cw}, height={ch}, format=NV12, framerate={fps}/1 !
  nvvidconv ! video/x-raw, width={w}, height={h}, format=BGRx !
  videoconvert ! video/x-raw, format=BGR ! appsink drop=1
```

`jetbot/camera/opencv_gst_camera.py` still resizes to **224²** (JetBot default). That is the wrong size for SmolVLA. A later VIC path should resize toward **512** (or 256, then pad in numpy/LeRobot). `videoconvert` is CPU; VIC did the NVMM scale.

## Memory ranking (lowest reachable first)

Numbers labeled **unmeasured** were not collected in this worktree.

1. **Eager PyTorch after #30** — only path that can satisfy #18's dummy forward. Weights ~0.9 GB BF16. Stage memory budget currently charges **1.46–1.86 GiB** for VLA, attributed mostly to the **PyTorch/CUDA runtime**, not the tensors. **Not re-measured here.** Fits isolated; co-resident with Qwen+voice is the real 8 GB fight.
2. **TRT FP16 of prefix + denoise expert, Python Euler loop, ctypes CUDA** — theoretically drops torch's caching allocator and can use packed FP16 weights. **Unmeasured.** Blocked on export. Build when nothing else large is resident (G1 builder ~1.5 GB RSS even for a toy graph).
3. **INT8 engine** — Orin can build INT8; PTQ calibration does not exist. Skip.
4. **Gemini one-graph pycuda 224/16 &lt;30 ms @ 1.3 GB** — not a plan for this model or this TRT.

## What is blocked on #30

* Loading `SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")`
* Any `torch.onnx.export`, including subgraphs
* Measuring torch RSS vs TRT RSS on this SoC
* Dummy motor-**token** I/O that is a real forward (the #18 gate)

This ticket delivered the I/O map, the TRT-10 stub (import-safe, no engine), the export script that **refuses** cleanly, and this evidence record. `./scripts/bringup/test_trt_smolvla.sh` runs the unit tests then **exits 1** on purpose so #18 stays open.

Safety: no PCA9685, no I2C `0x40`, no `/dev/snd`, no PWM from this lane.
