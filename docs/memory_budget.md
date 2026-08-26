# Memory budget — Jetson Orin Nano Super, 8 GB unified

Authoritative footprint accounting for the JetBot stack on this board, rebuilt 2026-08-26 against
the **architecture of record** rather than the spec as originally written. It supersedes the
previous revision, which still costed the VLM as llama.cpp Q4 GGUF + F16 `mmproj` and the voice
stack as FastConformer + Matcha/HiFi-GAN. Both of those are retired.

Interactive companion (toggle components, see headroom):
`~/.cursor/projects/home-impulse110-Documents/canvases/jetbot-memory-budget.canvas.tsx`.

## Verdict

**The stack still does not fit all-resident on 8 GB. It now fits comfortably in two of the three
operating modes, and the third is the one that has to be scheduled rather than assumed.**

| Configuration | Total | vs 7620 MiB pool | Verdict |
| --- | ---: | ---: | --- |
| **All-resident** — VLM + VLA + embedder + stores + voice + OS | **8751–10683 MiB (8.54–10.43 GiB)** | **−1131 … −3063 MiB** | **Does not fit** |
| **Driving only** — VLA + voice + OS | **3932–4398 MiB (3.84–4.29 GiB)** | **+3222 … +3688 MiB** | **Fits, wide margin** |
| **Retrieval only** — VLM + embedder + stores + voice + OS | **7256–8778 MiB (7.08–8.57 GiB)** | **+364 … −1158 MiB** | **Does not fit** — the optimistic bound is already inside the 400 MiB reserve |

Two things changed since the last revision and they pull in opposite directions:

- **Voice got much cheaper.** Zipformer + Piper in one Sherpa-ONNX process is **measured** at
  162.3–192.2 MiB, against 520–673 MiB for the retired two-process FastConformer + Matcha stack.
  That is a real, banked saving of roughly 330–450 MiB.
- **The VLM got more honest and no smaller.** The path of record is now TensorRT Edge-LLM INT4
  AWQ, and a genuine INT4 AWQ checkpoint is **~3.2 GiB of weights** because the vision tower is
  left FP16. That is the same weight class as the Q4 + `mmproj` estimate it replaces, so nothing
  was recovered — only the uncertainty moved from "which runtime" to "how much resident overhead."

The verdict does not depend on resolving that uncertainty. Even at the optimistic bound the
all-resident set overruns the pool by 1.10 GiB.

## How to read the numbers

Every figure carries a basis:

- **Measured** — a number this board produced, from a gate whose artifact exists. Trustworthy to a
  few MiB.
- **Derived** — arithmetic on a published, verified fact (parameter count, layer count, Hub file
  size). No measurement, but no guesswork either.
- **Estimated** — an informed guess. **The single largest line item in this budget is estimated,
  and it is the one that decides every marginal row.**

Units are **MiB (1024²)** throughout, matching `tegrastats` and `/proc/meminfo`. GiB is 1024 MiB.
Hub file sizes quoted in decimal bytes are converted before use.

Orin Nano has **no discrete VRAM**. CPU and GPU share the same pages, so "free GPU memory" is not
a separate quantity and a CPU-only model competes with a GPU model for the same bytes. Any table
that reports this stack in "GB VRAM" is using a machine model that does not describe this SoC.

## Architecture of record

This is what the budget costs. Anything not on this list is not charged.

| Subsystem | Checkpoint / runtime | Precision | Where it runs |
| --- | --- | --- | --- |
| **VLM** | `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` via **TensorRT Edge-LLM v0.10.0** | **INT4 AWQ text + FP16 vision tower** | GPU. ONNX exported on an x86 host; **engine built on the Orin** (Edge-LLM build location is Device). C++ `llm_inference` decode. |
| **VLA** | `lerobot/smolvla_base` | BF16 weights under **eager PyTorch** | GPU. **No TensorRT split exists yet.** |
| **Embedder** | `jinaai/jina-clip-v2`, Hub `onnx/model_int8.onnx` | INT8 ONNX | GPU or CPU via ORT; no TRT engine built. |
| **Voice** | Sherpa-ONNX Zipformer (ASR) + Piper VITS (TTS), **one process** | INT8 ONNX | **CPU only. Zero GPU memory.** |
| **Stores** | Chroma HNSW + SQLite facts | — | CPU. |
| **OS baseline** | L4T R36.4.4 / JetPack 6.2.1, incl. `nvargus-daemon` | — | Measured, always charged. |

Explicitly **not** in this stack, and therefore not charged:

- **No FP8, no NVFP4.** SM 87 has no native FP8 Tensor Cores and the Edge-LLM Orin support matrix
  is **FP16, INT8, INT4 only** — "Jetson Orin does not run FP8 or FP4 model engines." The
  `nvidia/llama-nemotron-embed-vl-1b-v2-fp8` export is a real NVIDIA artifact and is still
  unusable here. FP8 is not an intermediate step to anything on this board.
- **No `smolvla-jetbot`.** That checkpoint does not exist. Only `lerobot/smolvla_base` does.
- **No ROS 2, no Isaac ROS, no NITROS.** Not installed. The real camera cost is `nvargus-daemon`
  at 181 MiB, and that is **already inside the measured OS baseline** — charging a NITROS line on
  top double-counts it.
- **No llama.cpp / GGUF.** Retired as the VLM path of record in favour of Edge-LLM.
- **No `nvidia/llama-nemotron-embed-vl-1b-v2`**, and **no SigLIP 2 Base**. Neither is the RAG
  default.

**Licensing note:** `jina-clip-v2` is **CC BY-NC 4.0**. Acceptable for this research robot; **not**
a drop-in for a commercial product without a commercial licence from Jina.

## The pool

| Metric | Value | Source |
| --- | --- | --- |
| `MemTotal` | 7802736 kB (~7.44 GiB) | `/proc/meminfo` |
| `tegrastats` RAM total | **7620 MiB** | G1 ([07-tensorrt-g1.md](bringup/07-tensorrt-g1.md)) |
| `cudaMemGetInfo` total | 7990 MB | G1 |
| Swap | 32 GiB at `/ssd/32GB.swap`, `vm.swappiness=60`, **0 B used through every gate** | G1, F4, F5, [01-os.md](bringup/01-os.md) |

The 7620 / 7990 spread is carveout. **7620 MiB is the pool throughout this document** because it is
the conservative figure and the one `tegrastats` reports during a run.

### OS baseline: 2243–2269 MiB, measured

`tegrastats` reported `RAM 2243/7620MB` and `RAM 2269/7620MB` at idle during the F5 and F4 voice
gates. This baseline **already includes `nvargus-daemon`**, an always-active systemd service
measured live at **185852 kB RSS (181 MiB)** with the camera stack up.

G1 recorded a higher `RAM 2824/7620MB` sample, but that was taken during engine-build activity
rather than at idle, so the F4/F5 figures are used here.

`MemAvailable` runs ~1 GB below `MemTotal − tegrastats RAM` because the kernel conservatively
discounts cache it is unsure it can reclaim. Both are recorded; neither is silently preferred.

**Model budget = 7620 − 2269 … 7620 − 2243 = 5351–5377 MiB (5.22–5.25 GiB).**

A **400 MiB reserve** is held back from the pool for page cache and NVMM contiguity. "Fits" below
means the **pessimistic** bound still clears that reserve.

## Measured line items

These are the only figures in this budget carrying hard evidence.

| Line item | Footprint | Gate | Notes |
| --- | ---: | --- | --- |
| **Voice — Zipformer + Piper, one process, CPU** | **162.3 MiB** offline / **192.2 MiB** live | **F4/F5 rewrite** (`stage-f-zipformer-piper`), `data/bringup/zipformer_piper.json` | 166220 KiB peak RSS for a short offline turn; 196776 KiB with a 5 s live USB-microphone capture and Piper resident in the same process. `sherpa-onnx==1.13.6`, 2 threads. **GPU use 0 MiB.** Both models are Sherpa-ONNX C++ objects in a single Python process — no second inference framework, no CUDA provider, no PyTorch, no NeMo. |
| Agent harness process | **32 MiB** | measured for this document | RSS after importing `jetbot_agent`, `numpy`, PyYAML and `sherpa_onnx`, before any model loads. |
| TensorRT engine build, **transient** | **1491 MiB** peak RSS | **G1**, `data/bringup/g1_runtime.json` | For a **68 KB** ONNX graph. `libnvinfer_builder_resource.so` alone is 152 MB and tactic search is not free. `cudaMemGetInfo` free moved 4882 → 4012 MB across the run. **This is a floor, not a forecast** — see the `llm_build` risk below. |
| `nvargus-daemon` resident | **181 MiB** | live `ps` 2026-08-26 | Inside the OS baseline; **do not charge it twice.** |

**Voice total, both models plus the harness: 194–224 MiB.** This replaces the 520–673 MiB charged
in the previous revision. Growth tracks utterance and generation length and nothing else, so
capping turn length is a hard cap on this line; the 6.6 s bundled fixture plus a concurrent
synthesis can exceed 200 MiB.

The **`trtexec` RSS rows in `g1_runtime.json` (~43 MB) are an artifact** and are deliberately not
used — that run sampled the wrapper process, not the `trtexec` child. The 1491 MiB figure comes
from the Python API path, which was sampled correctly.

## Derived line items

| Line item | Footprint | Basis |
| --- | ---: | --- |
| VLM weights — `Qwen2.5-VL-3B-Instruct-AWQ` | **3244 MiB (~3.2 GiB)** | Hub `model.safetensors` = 3,401,785,760 B. INT4 text **plus an FP16 vision tower**: `modules_to_not_convert: ["visual"]`, so ~1.87 GiB of the file is still F16. That is why a "3B INT4" file is 3.2 GiB and not ~1.8 GiB. |
| VLM KV cache | **72 MiB** @ 2048 tok · **144 MiB** @ 4096 · **288 MiB** @ 8192 | Qwen2.5-VL-3B: 2 (K,V) × 36 layers × 2 KV heads × 128 head_dim × 2 B = **36 KiB/token**. |
| VLA weights — `lerobot/smolvla_base` BF16 | **900 MiB** | ~450M params × 2 B. |
| Embedder weights — `jina-clip-v2` INT8 | **825 MiB** derived / **834 MiB** Hub file | 865M params × 1 B = 825 MiB; Hub `onnx/model_int8.onnx` is 874,350,932 B = 834 MiB. Dual encoder: Jina XLM-RoBERTa 561M text + EVA02-L14 304M vision. Listed via the Hub tree API; **not downloaded**. |

**Matryoshka 256-d is a Chroma-side setting only.** Truncating 1024 → 256 makes stored vectors 4×
smaller (50k chunks: ~49 MiB instead of ~196 MiB) and distance math 4× cheaper. It does **not**
shrink the 834 MiB INT8 graph or any engine built from it. The encoder stays fully resident while
RAG is on.

## Estimated line items

| Line item | Footprint | Why it is only an estimate |
| --- | ---: | --- |
| **VLM resident — Edge-LLM INT4 AWQ engine, KV @ 4096, runtime buffers, activations** | **3488–3838 MiB (3.41–3.75 GiB)** | **The largest uncertainty in this budget.** No ONNX tree and no SM 87 engine exist on this host; nothing has been loaded. 3244 MiB of weights is the floor; the range adds KV @ 4096 plus TRT runtime and activation arenas. For scale, this checkpoint reports 3334 MB of GPU memory on an A6000 — a machine with discrete VRAM — and NVIDIA's own Nano 2B-VLM benchmarks land at 4.4–4.6 GB, which is a *small*-VLM floor rather than a 3B proof. **The pessimistic end of that evidence sits above this range, not inside it.** |
| **VLA resident — eager PyTorch** | **1495–1905 MiB (1.46–1.86 GiB)** | 900 MiB of BF16 weights plus 600–1000 MiB of PyTorch + CUDA context + cuDNN/cuBLAS kernel libraries, charged **once per GPU process**. **PyTorch is not installed on this board.** Anchored on the one hard datapoint available: a TensorRT process needed 1491 MiB to build a 68 KB graph, so GPU framework overhead here is measured in hundreds of MiB, not tens. **This line stands until a TensorRT split exists** — SmolVLA cannot be exported as one graph (Python Euler loop over 10 denoise steps, croppable KV cache object, three camera inputs), so the reachable path is prefix + denoise-expert subgraphs, and neither has been exported. |
| Embedder resident — `jina-clip-v2` INT8 | **1126–2048 MiB (1.1–2.0 GiB)** | 834 MiB of graph plus one or two engines and 512×512 activations. **Unmeasured on this SoC.** EVA-02-L/14 uses SwiGLU, 2D RoPE and xFormers attention, which often need mixed precision or careful calibration — the Hub INT8 file proves *an* INT8 graph exists, not a working Orin engine. |
| Chroma HNSW index + SQLite | **205–399 MiB (0.20–0.39 GiB)** | Stage I stores are not built and no gate has run. A placeholder that must be measured. Cap Stage I to tens of thousands of chunks until it is. |
| `llm_build` for a 3B VLM, **transient** | **unmeasured, ≫ 1491 MiB** | The measured 1491 MiB was a 68 KB toy graph. A 3B builder is a different order of magnitude and is absent from NVIDIA's Nano 8 GB benches. **This is the highest OOM risk in the whole plan.** |

## The three planned configurations

Arithmetic in MiB, low bound / high bound. The OS baseline and the voice stack are charged in
every row because both are always resident by design.

| Component | Low | High |
| --- | ---: | ---: |
| OS baseline | 2243 | 2269 |
| Voice (Zipformer + Piper + harness) | 194 | 224 |
| VLM resident (Edge-LLM INT4 AWQ, KV @ 4096) | 3488 | 3838 |
| VLA resident (`smolvla_base`, eager torch) | 1495 | 1905 |
| Embedder resident (`jina-clip-v2` INT8) | 1126 | 2048 |
| Chroma + SQLite | 205 | 399 |

| Configuration | Components | Total (MiB) | Total (GiB) | Headroom vs 7620 | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| **Driving only** | OS + voice + VLA | **3932–4398** | **3.84–4.29** | **+3222 … +3688** | **Fits**, wide margin |
| **Retrieval only** | OS + voice + VLM + embedder + stores | **7256–8778** | **7.08–8.57** | **+364 … −1158** | **Does not fit** |
| **All-resident** | everything above | **8751–10683** | **8.54–10.43** | **−1131 … −3063** | **Does not fit** |

Retrieval-only deserves a second look because it is the row people will want to argue with. Its
optimistic bound leaves 364 MiB — **less than the 400 MiB reserve** — and that bound assumes the
VLM resides at 3488 MiB and the embedder at 1126 MiB simultaneously, which is the best case of two
unmeasured estimates at once. Treating it as "nearly fits" is how a board OOMs.

## Co-residency: what actually fits

The decision this budget exists to support is not the total but **what can be resident at once.**
Voice and the OS baseline are included in every row.

| Co-residency set | Total (MiB) | Headroom vs 7620 | Verdict |
| --- | ---: | ---: | --- |
| Voice only | 2437–2493 | +5127 … +5183 | **Fits** |
| Voice + toy engine build (68 KB graph, measured) | 3928–3984 | +3636 … +3692 | **Fits** |
| Voice + VLA — **driving mode** | 3932–4398 | +3222 … +3688 | **Fits** |
| Voice + embedder + stores (RAG, no VLM) | 3768–4940 | +2680 … +3852 | **Fits** |
| Voice + VLA + embedder + stores | 5263–6845 | +775 … +2357 | **Fits** |
| Voice + VLM — **reasoning mode** | 5925–6331 | +1289 … +1695 | **Fits** |
| Voice + VLM + embedder + stores — **retrieval mode** | 7256–8778 | +364 … −1158 | **Does not fit** |
| Voice + VLM + VLA | 7420–8236 | +200 … −616 | **Does not fit** |
| **Everything resident** | **8751–10683** | **−1131 … −3063** | **Does not fit** |
| Any set + `llm_build` for a 3B VLM | unbounded | unknown | **Forbidden** — see below |

Four conclusions follow directly:

1. **One large GPU tenant at a time.** Voice plus any single large model fits with room to spare.
   Voice plus any *two* of {VLM, VLA, embedder} is at or past the reserve. There is no arrangement
   in which the VLM, the VLA policy, the embedder and the voice stack are all resident.
2. **Voice is free and stays up.** At 194–224 MiB it is the cheapest thing on the board, it is
   CPU-only, and it never contends for the GPU. It should remain resident across every mode so the
   robot can always be spoken to, including while a model is being swapped.
3. **Retrieval mode needs the embedder and the VLM to take turns.** Encode-then-unload before
   invoking the VLM, or accept that RAG and reasoning are two passes rather than one resident set.
4. **Engine builds need a nearly empty board.** Voice may stay up during a build. Nothing else may.

## Rule: never `llm_build` alongside another large model

The measured 1491 MiB builder cost was for a **68 KB** graph. The Edge-LLM `llm_build` for
Qwen2.5-VL-3B is unmeasured, certainly far larger, and absent from NVIDIA's published Orin Nano
8 GB benchmarks. Edge-LLM requires the engine to be built **on the device**, so this cost cannot be
moved to the workstation the way the ONNX export can.

**Give every engine build a dedicated window with the voice stack as its only other tenant.** Never
build while the VLM, the VLA policy, the embedder, or any PyTorch process is loaded. This is the
single most likely way to OOM this board, ahead of any steady-state residency question.

## Rejected: the 3.90 GiB and 4.35 GB all-resident tables

Both Gemini-authored tables are **false** and must not be used to size anything. They share one
root error — treating this SoC as if it had discrete VRAM and a 0.30 GiB operating system — and
several independent ones.

| Claimed row | Claimed | Actual | Why the claim fails |
| --- | ---: | ---: | --- |
| OS / NITROS | 0.30 GiB | **2.19–2.22 GiB** OS; NITROS **0** | `tegrastats` measured 2243–2269 MiB at F4/F5 idle. ROS 2, Isaac ROS and NITROS are **not installed**, so the NITROS line is both speculative and, via `nvargus-daemon`, double-counted. Wrong by ~1.9 GiB. |
| Qwen2.5-VL-3B INT4 | 1.80 GiB | **3.41–3.75 GiB** resident (est.) | Ignores the FP16 vision tower that every real AWQ export leaves unquantized — weights alone are 3.2 GiB — then also omits KV cache, TRT runtime and activations. |
| `smolvla-jetbot` FP16/INT8 | 1.20–1.35 GiB | **1.46–1.86 GiB** | The checkpoint **does not exist**. `lerobot/smolvla_base` weights are ~0.88 GiB, so the weights claim was generous; what is missing is the PyTorch and CUDA runtime around them. |
| Embedder INT8 | 0.40 GiB (SigLIP 2) / 0.90 GB (Jina) | **1.1–2.0 GiB** resident | Weights-only accounting. 865M × 1 B = 825 MiB is the *graph*, not the resident set; engines and 512² activations are extra. |
| Audio | 0 | **0.19–0.22 GiB** | CPU-only is not free on unified memory. It is cheap — but it is not zero. |
| PyCUDA runtime on Tegra | assumed | **not available** | G1 used ctypes `libcudart` and apt `python3-libnvinfer`. PyPI `pycuda`/`tensorrt` wheels are the wrong ABI for Tegra. |
| **Total vs "5.2 GB usable"** | **3.90 / 4.35** | **8.54–10.43 GiB** | The 5.2 GiB figure is the **model** budget *after* subtracting the OS. Charging a 0.30 GiB OS line against it counts the operating system twice. |

Swapping the embedder from 0.40 to 0.90 does not repair the other rows. Neither table is a
starting point for negotiation.

## What is still unmeasured

**The largest line item in this budget has never been loaded on this board.** Closing that is the
only way to tighten the total.

| Unmeasured | Estimate | Blocked by | What would close it |
| --- | ---: | --- | --- |
| Qwen2.5-VL-3B via Edge-LLM INT4 AWQ | 3.41–3.75 GiB resident | **G2** — no ONNX tree exported, no SM 87 engine built, no x86 export host used | Peak process RSS and a `tegrastats` trough from a real decode, **with and without the vision tower resident**. Report KV separately from weights. |
| `llm_build` peak for a 3B graph | **unknown, ≫ 1491 MiB** | **G2** | The build itself, on an otherwise idle board, with `tegrastats` sampled throughout. Do this **first** — it gates whether the VLM is reachable at all. |
| `lerobot/smolvla_base` under eager PyTorch | 1.46–1.86 GiB | **#30 / G3** — torch not installed, weights not downloaded | RSS of a dummy forward. Separately: RSS of a bare `import torch` plus a CUDA context, before any weights, to split the framework cost from the model cost. |
| SmolVLA prefix + denoise-expert TRT split | would replace the line above | **#18** — no ONNX export of either subgraph | Export both subgraphs, build FP16 engines on-device with the VLM unloaded, measure. INT8 PTQ needs a calibration set this robot does not have. |
| `jinaai/jina-clip-v2` INT8 | 1.1–2.0 GiB resident | **G4 / Stage I** — Hub ONNX listed, **not** fetched, no TRT engine | Peak RSS of ORT or `trtexec` at 512×512, batch 1; confirm INT8 accuracy against the FP16 graph before trusting it. |
| Chroma + SQLite | 0.20–0.39 GiB | Stage I not started | RSS of the store with a representative index loaded at 256-d. |
| Real `trtexec` build RSS | unknown | G1 sampling bug, fixed but not re-run | Re-run `./scripts/bringup/g1_tensorrt_smoke.sh`; the `tracked_pid` field marks output from the fixed version. |

**G2, G3 and G4 must report peak process RSS and a `tegrastats` trough the way the voice gates do.**
Until they do, the biggest number in this budget is the one that cannot be defended.

## Reproduce

```bash
# Pool, baseline, swap
grep -E "MemTotal|MemAvailable|SwapTotal|SwapFree" /proc/meminfo
tegrastats --interval 1000 | head -3
ps -o rss=,comm= -C nvargus-daemon

# Voice, measured (branch stage-f-zipformer-piper)
./scripts/bringup/test_zipformer_piper.sh          # -> data/bringup/zipformer_piper.json
./scripts/bringup/test_zipformer_piper.sh --live-capture

# TensorRT builder floor
./scripts/bringup/g1_tensorrt_smoke.sh             # G1 -> data/bringup/g1_runtime.json
```

Artifacts under `data/bringup/` are gitignored; re-run the gates to regenerate them.

## Source documents

This branch is cut from `main`, so companion documents that live on other feature branches are
linked absolutely. Merge order will collapse these into relative links.

On `main`: [07-tensorrt-g1.md](bringup/07-tensorrt-g1.md) and [07-tensorrt.md](bringup/07-tensorrt.md) (G1),
[01-os.md](bringup/01-os.md) (Stage A, swap), [JETBOT_SPEC.md](../JETBOT_SPEC.md).

On feature branches:

- Voice measurements — [`06-voice.md` on `stage-f-zipformer-piper`](https://github.com/AbuAyah110/jetbot-orin-super/blob/stage-f-zipformer-piper/docs/bringup/06-voice.md)
- Edge-LLM export/build path — [`07-edgellm-workstation-quant.md` on `stage-g-edgellm-workstation`](https://github.com/AbuAyah110/jetbot-orin-super/blob/stage-g-edgellm-workstation/docs/bringup/07-edgellm-workstation-quant.md)
- SmolVLA TensorRT feasibility — [`07-smolvla-trt.md` on `stage-g-smolvla-trt`](https://github.com/AbuAyah110/jetbot-orin-super/blob/stage-g-smolvla-trt/docs/bringup/07-smolvla-trt.md)
- Embedder decision — [`jina_clip_v2.md` on `stage-i-jina-clip`](https://github.com/AbuAyah110/jetbot-orin-super/blob/stage-i-jina-clip/docs/jina_clip_v2.md)
- Rejected tables — [`gemini_architecture_audit.md` on `stage-i-jina-clip`](https://github.com/AbuAyah110/jetbot-orin-super/blob/stage-i-jina-clip/docs/gemini_architecture_audit.md)

`main`'s `docs/bringup/06-voice.md` still describes the retired FastConformer + Matcha stack, and
`docs/bringup/07-tensorrt.md` still says no NVIDIA product is named TensorRT Edge-LLM. Both are
**stale**; the branch documents above supersede them.
