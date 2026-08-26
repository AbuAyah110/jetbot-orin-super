# Memory budget — Jetson Orin Nano Super, 8 GB unified

Authoritative footprint accounting for the JetBot stack on this board, rebuilt 2026-08-26 against
the **architecture of record** rather than the spec as originally written. It supersedes the
previous revision, which still costed the VLM as llama.cpp Q4 GGUF + F16 `mmproj` and the voice
stack as FastConformer + Matcha/HiFi-GAN. Both of those are retired.

**This revision replaces the VLM estimate with measured evidence.** The operating rows below are
now priced under an explicit, optimistic premise — that **the INT4 AWQ Qwen runtime builds and
runs on this Orin** — and the number that premise implies is taken from NVIDIA's own published
TensorRT Edge-LLM Orin benchmarks rather than from a guess. The `llm_build` OOM risk is no longer
charged against the operating rows; it lives in a separate build-time row where nothing else is
resident. See [Scenario: the INT4 Qwen runtime succeeds](#scenario-the-int4-qwen-runtime-succeeds).

Interactive companion (toggle components, see headroom):
`~/.cursor/projects/home-impulse110-Documents/canvases/jetbot-memory-budget.canvas.tsx`.

## Verdict

**Driving fits with a wide margin. Retrieval does not fit. All-resident does not fit, and is now
further out than the previous revision thought.** Granting that the INT4 Qwen runtime works does
not rescue the marginal rows — it makes them worse, because a *working* INT4 AWQ VLM on Orin
measures larger than this budget previously estimated.

| Configuration | Contents | Total | vs 7620 MiB pool | Verdict |
| --- | --- | ---: | ---: | --- |
| **Driving** | OS + voice + SmolVLA. **No VLM, no LanceDB.** | **3932–4398 MiB (3.84–4.29 GiB)** | **+3222 … +3688 MiB** | **Fits, wide margin** |
| **Retrieval** | OS + voice + Qwen + Jina CLIP v2 + LanceDB. VLA paused. | **8742–10389 MiB (8.54–10.15 GiB)** | **−1122 … −2769 MiB** | **Does not fit** |
| **All-resident** | everything above at once | **10237–12294 MiB (10.00–12.01 GiB)** | **−2617 … −4674 MiB** | **Does not fit** |

Three things changed since the last revision:

- **Voice got much cheaper, and it is banked.** Zipformer + Piper in one Sherpa-ONNX process is
  **measured** at 162.3–192.2 MiB, against 520–673 MiB for the retired two-process
  FastConformer + Matcha stack — a real saving of roughly 330–450 MiB.
- **The VLM got measured, and it got bigger.** Assuming the runtime works, Qwen2.5-VL-3B INT4 AWQ
  with an FP16 vision tower costs **5129–5718 MiB (5.01–5.58 GiB) resident**, not the
  3488–3838 MiB this budget previously estimated. That is **+1641 … +1880 MiB (+1.60 … +1.84 GiB)**
  on the single line that decides every marginal row — and the entire delta lands there.
- **The `llm_build` OOM risk left the operating rows.** It is a real risk and it is still the first
  thing that can kill this plan, but it is a *build-time* cost with nothing else resident, so
  charging it against steady-state residency was double-counting a scheduling problem.
- **Stage I storage got cheaper and simpler.** One LanceDB table replaces Chroma HNSW + SQLite
  facts: **50–130 MiB resident** (−155 … −269 MiB vs 205–399). Retrieval still includes it;
  driving does not load it. It does not change any fit verdict.

The correction to the VLM line, +1.60 to +1.84 GiB, is almost exactly the size of the entire
1.80 GiB that the rejected Gemini table claimed for that subsystem.

### Where the pessimism went, and where it did not

Dropped from the planned operating rows, because they were speculative:

- **"The VLM might be 6 GB."** It is not. The bench-anchored band tops out at 5718 MiB, and the
  largest INT4 AWQ VLM NVIDIA publishes on *any* Orin — Qwen3-VL-8B-Instruct — is 9065 MiB, so
  the scaling that produces 6 GB for a 3B model does not exist.
- **`llm_build` peak.** Moved to its own row. It never was a steady-state cost.

Not dropped, because they are now evidence rather than fear:

- **A working 3B INT4 AWQ VLM does not co-reside with anything else large on this board.** At
  5129–5718 MiB it exceeds the 4727 MiB safe single-tenant ceiling on its own.
- **1.80 GiB remains fiction.** No published Edge-LLM row on any device puts a VLM of this class
  below 2576 MiB, and that row is a 0.8B model.

## How to read the numbers

Every figure carries a basis:

- **Measured** — a number this board produced, from a gate whose artifact exists. Trustworthy to a
  few MiB.
- **Derived** — arithmetic on a published, verified fact (parameter count, layer count, Hub file
  size). No measurement, but no guesswork either.
- **Bench-anchored** — interpolated between numbers NVIDIA measured on Orin hardware with the same
  runtime and the same precision, on a curve that fits their own datapoints to ±11 MiB. Not a
  measurement of *our* model, but not an opinion either. **The largest line item in this budget is
  now bench-anchored rather than estimated, which is the main change in this revision.**
- **Estimated** — an informed guess. Still the basis for SmolVLA, the embedder and the stores.

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
| **Stores** | **One LanceDB** (vectors + facts in one table; replaces Chroma + `facts_db` SQLite) | IVF-PQ on disk, mmap | CPU. Embedded Python (`lancedb` 0.37.1 `cp310-abi3-manylinux_2_28_aarch64`). |
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

**Matryoshka 256-d is a store-side setting only.** Truncating 1024 → 256 makes stored vectors 4×
smaller (50k chunks: ~49 MiB of float32 on disk instead of ~196 MiB) and distance math 4× cheaper.
It does **not** shrink the 834 MiB INT8 graph or any engine built from it. The encoder stays fully
resident while RAG is on.

## Estimated line items

| Line item | Footprint | Why it is only an estimate |
| --- | ---: | --- |
| **VLA resident — eager PyTorch** | **1495–1905 MiB (1.46–1.86 GiB)** | 900 MiB of BF16 weights plus 600–1000 MiB of PyTorch + CUDA context + cuDNN/cuBLAS kernel libraries, charged **once per GPU process**. **PyTorch is not installed on this board.** Anchored on the one hard datapoint available: a TensorRT process needed 1491 MiB to build a 68 KB graph, so GPU framework overhead here is measured in hundreds of MiB, not tens. **This line stands until a TensorRT split exists** — SmolVLA cannot be exported as one graph (Python Euler loop over 10 denoise steps, croppable KV cache object, three camera inputs), so the reachable path is prefix + denoise-expert subgraphs, and neither has been exported. |
| Embedder resident — `jina-clip-v2` INT8 | **1126–2048 MiB (1.1–2.0 GiB)** | 834 MiB of graph plus one or two engines and 512×512 activations. **Unmeasured on this SoC.** EVA-02-L/14 uses SwiGLU, 2D RoPE and xFormers attention, which often need mixed precision or careful calibration — the Hub INT8 file proves *an* INT8 graph exists, not a working Orin engine. |
| **LanceDB — one embedded table, vectors + facts** | **50–130 MiB (0.05–0.13 GiB) resident** | Replaces Chroma HNSW + SQLite `facts_db`. **Not zero RAM.** See [Stage I store: LanceDB](#stage-i-store-lancedb). |

## Scenario: the INT4 Qwen runtime succeeds

This is the premise of every operating row below. **Assume the Edge-LLM INT4 AWQ export, the
on-device `llm_build`, and the C++ `llm_inference` decode all work** — no OOM, no plugin failure,
no accuracy regression that forces a fallback. The question is then narrow and answerable: *what
does a working one cost?*

### The evidence: NVIDIA measured this, on this board family

TensorRT Edge-LLM ships a performance benchmark table with an Orin Nano 8 GB section. Its
`GPU Mem (MB)` column is defined as "peak GPU memory usage during inference," and on Jetson it is
**peak process RSS**: `examples/utils/memoryMonitor.cpp` logs *"iGPU detected, monitoring unified
memory through RSS"*, `getPeakUnifiedMemory()` returns `getrusage(RUSAGE_SELF).ru_maxrss`, and
`getPeakGpuMemory()` returns 0 on an integrated GPU. `toMB()` divides by 1024², so the column is
**MiB and directly comparable to `tegrastats`**.

It is a *process-local* metric, not a system total — the same Qwen3-VL-2B-Instruct engine reports
**4344 MiB on Orin NX 16 GB and 4380 MiB on Orin Nano 8 GB**, a 36 MiB spread across boards with
8× different memory. It therefore excludes the OS baseline, which this budget charges separately.

**NVIDIA's Orin Nano 8 GB table has no Qwen2.5-VL-3B row.** It has no VLM larger than 2B at all,
and its single largest entry of any kind is 4621 MiB. That omission is the whole problem, and it
is not an oversight — see below.

v0.9.0, INT4 AWQ text / FP16 vision tower, vanilla decode, batch 1, `--maxInputLen 2048`,
`--maxKVCacheCapacity 2200`, ViT at 265 image tokens, JetPack 7.2:

| Device | Model | Params | Peak RSS (MiB) |
| --- | --- | ---: | ---: |
| **Orin Nano 8 GB** | Qwen3.5-0.8B | 0.8B | **2603** |
| **Orin Nano 8 GB** | Qwen3-VL-2B-Instruct | 2B | **4380** |
| **Orin Nano 8 GB** | Qwen3.5-2B | 2B | **4621** ← largest Nano row of any kind |
| Orin NX 16 GB | Qwen3-VL-2B-Instruct | 2B | 4344 |
| Orin NX 16 GB | Qwen3-VL-4B-Instruct | 4B | 5903 |
| Orin NX 16 GB | Qwen3-VL-8B-Instruct | 8B | 9065 |
| Orin NX 16 GB | **Qwen2.5-VL-7B-Instruct**, EAGLE3 | 8.29B | 9161 |

### Interpolating to a 3B

The Qwen3-VL rows are dense VLMs with an INT4 AWQ backbone and an FP16 ViT — the same shape as our
checkpoint — and they are almost perfectly linear in parameter count:

```text
peak RSS (MiB) ≈ 2774 + 785 × params_in_billions
    2B → 4344 predicted vs 4344 measured   (+0)
    4B → 5914 predicted vs 5903 measured  (−11)
    8B → 9054 predicted vs 9065 measured  (+11)
```

A ±11 MiB fit across a 4× parameter range is not a coincidence. Evaluating it for
`Qwen2.5-VL-3B-Instruct-AWQ`:

| Route | Result |
| --- | ---: |
| Qwen3-VL curve at the **nominal** 3B | **5129 MiB** |
| Qwen3-VL curve at the **actual** 3.75B total (3.09B LLM + ~0.67B ViT) | **5718 MiB** |
| Independent check — anchored on the **Qwen2.5-VL-7B** row (9161 MiB EAGLE3 − 455 MiB measured EAGLE3 overhead ≈ 8706 MiB vanilla at 8.29B), same slope | **5142 MiB** |

Two independent routes land within 13 MiB of each other at ~5.13 GiB. The third route is the
conservative one, because the "3B" in this checkpoint's name counts only the language backbone
while the FP16 vision tower adds another ~0.67B parameters that the runtime must also hold.

**Planned VLM resident: 5129–5718 MiB (5.01–5.58 GiB), bench-anchored.** Bracketed below by a
measured 4380 MiB for a 2B on this exact board, and above by a measured 5903 MiB for a 4B on the
same SM 87 architecture.

| Line item | Footprint | Basis |
| --- | ---: | --- |
| **VLM resident — Edge-LLM INT4 AWQ engine + FP16 ViT engine, KV @ 2200, activations, C++ `llm_inference` runtime, CUDA context** | **5129–5718 MiB (5.01–5.58 GiB)** | **Bench-anchored.** Includes everything in one process: both engines' weights, KV, ViT and LLM activation arenas, TensorRT + cuteDSL runtime, and the CUDA context. Peak, not steady state. |

The ~3.2 GiB of AWQ weights are inside that figure, not additional to it: 3244 MiB of weights plus
a **bounded** runtime add-on of **1885–2474 MiB** for KV @ 2200 (77 MiB derived), the FP16 ViT and
LLM activation arenas, the TensorRT 10.3 and cuteDSL plugin runtime, the CUDA primary context, and
the engine-deserialization staging that `ru_maxrss` captures at load. That add-on is measured by
difference against NVIDIA's curve rather than guessed, which is the only reason it can be bounded
at all.

### Why NVIDIA's Nano table stops at 2B

The board's arithmetic explains the gap in their table. Charging the measured OS baseline and the
measured voice stack leaves a **safe single-VLM ceiling of 7620 − 400 reserve − 2493 = 4727 MiB**:

| Candidate | Peak RSS | vs the 4727 MiB ceiling |
| --- | ---: | --- |
| Qwen3.5-0.8B (measured, Nano) | 2603 | **−2124, comfortable** |
| Qwen3-VL-2B-Instruct (measured, Nano) | 4380 | **−347, fits** |
| Qwen3.5-2B (measured, Nano) | 4621 | **−106, fits by a hair** |
| **Qwen2.5-VL-3B-Instruct-AWQ (bench-anchored)** | **5129–5718** | **+402 … +991, over the ceiling** |
| Qwen3-VL-4B-Instruct (measured, Orin NX — **absent from the Nano table**) | 5903 | +1176, over the pool once the OS is charged |

NVIDIA's Nano 8 GB coverage stops at 2B because 4B does not fit an 8 GB Orin with an operating
system on it. A 3B sits exactly in the gap they declined to publish, on the wrong side of the
ceiling. **The honest reading of "the INT4 Qwen runtime succeeds" is that it succeeds as the only
large tenant on the board, and only if the vision tower can be evicted between visual turns.**

### The one lever that works

The FP16 vision tower is ~0.67B parameters, so its engine weights are ~1286 MiB — a far larger
share of this checkpoint than in the benched Qwen3.5 family, where dropping the ViT path is worth
only 190–476 MiB (`Qwen3.5-2B` 4621 → `Qwen3.5-2B-LLM` 4145). Evicting the ViT engine between
visual turns is therefore worth roughly **1.26 GiB here**, and it is the difference between
"reasoning mode does not fit" and "reasoning mode fits with 0.7–1.3 GiB to spare":

| Reasoning mode | Total (MiB) | Total (GiB) | Headroom | Verdict |
| --- | ---: | ---: | ---: | --- |
| OS + voice + VLM, **ViT resident** | 7566–8211 | 7.39–8.02 | +54 … −591 | **Does not fit** |
| OS + voice + VLM, **ViT evicted** (−1286 MiB) | 6280–6925 | 6.13–6.76 | +695 … +1340 | **Fits** |

Levers that do **not** work, so nobody re-proposes them: shrinking the KV cache (2200 → 1024 saves
~40 MiB), shrinking the image profile (NVIDIA's benched Nano ViT engine is already
`--maxImageTokens 2048 --maxImageTokensPerImage 2048`, so our recipe is on their profile and there
is nothing to recover), and `--externalize-weights int4_ffn` (real, required on Orin, and already
in effect in every benched row above — it cuts *build* host memory, not runtime residency).

## Stage I store: LanceDB

**One LanceDB replaces both Chroma and the SQLite facts DB.** Vectors (Jina CLIP v2, Matryoshka
256-d) and key-value facts live in the same table — extra scalar columns on the same rows, not a
second process. This is a Stage I architecture change, not a measurement. Nothing has been
installed on this board.

PyPI `lancedb==0.37.1` ships a **`cp310-abi3-manylinux_2_28_aarch64`** wheel. This board is
CPython 3.10.12 / Ubuntu 22.04 (glibc 2.35), so the wheel is ABI-legal. That is not the same as
"imported here": PyArrow (a runtime dep) and the Rust extension have never been loaded on this
SoC. Do not treat aarch64 wheels as a Jetson gate.

Lance is a **disk-native, mmap** store (Lance columnar format, default index **IVF-PQ**, optional
HNSW). Chroma's default HNSW graph is RAM-resident. Published third-party RSS after index build
on 1536-d float32 (Kanopy, M2, not this board):

| Scale | LanceDB RSS | Chroma RSS |
| ---: | ---: | ---: |
| 100k × 1536-d | ~120 MB | ~800 MB |
| 1M × 1536-d | ~400 MB | ~6.5 GB |

Stage I is **tens of thousands of 256-d** vectors, not millions of 1536-d. Raw payload at the
cap: 50k × 256 × 4 B = **49 MiB** of float32 on disk. IVF-PQ compresses that further. Facts
columns are kilobytes. Disk at the Stage I cap is **~50–200 MiB** for the Lance directory
(fragments + IVF-PQ + versions), well under a gigabyte.

Resident RAM is **not** that disk size, and it is **not** zero:

| Piece | Charge |
| --- | ---: |
| Python import tax (`lancedb` + PyArrow + the Rust `.so`) | **~40–80 MiB** RSS even on an empty connect |
| IVF-PQ / table pages faulted into RSS | **~10–40 MiB** at 50k × 256-d |
| mmap / page cache of the Lance directory (shows in `tegrastats`, not always in process RSS) | **0–30 MiB** working set if the kernel has not reclaimed it |

**Planned LanceDB resident: 50–130 MiB (0.05–0.13 GiB).** Process RSS plus a bounded mmap working
set. Low bound is an empty-ish connect plus a small table; high bound is a 50k-row IVF-PQ index
with a query working set faulted in. **Do not use HNSW or `IVF_HNSW_PQ` on this board** — Lance
maintainers treat HNSW index build as memory-intensive (OOM reports at tens of millions of
vectors even on 32 GB hosts). Stay on default IVF-PQ.

Versus the previous Chroma HNSW + SQLite facts line (**205–399 MiB**): **−155 … −269 MiB
(−0.15 … −0.26 GiB)**. Real, but two orders of magnitude smaller than the VLM correction. It
does not change any fit/no-fit verdict.

**Driving does not load LanceDB.** Retrieval and all-resident do.

## The planned configurations

Arithmetic in MiB, low bound / high bound. The OS baseline and the voice stack are charged in
every row because both are always resident by design.

| Component | Low | High | Basis |
| --- | ---: | ---: | --- |
| OS baseline | 2243 | 2269 | Measured |
| Voice (Zipformer + Piper + harness) | 194 | 224 | Measured |
| VLM resident (Edge-LLM INT4 AWQ + FP16 ViT, KV @ 2200) | 5129 | 5718 | Bench-anchored |
| VLA resident (`smolvla_base`, eager torch) | 1495 | 1905 | Estimated |
| Embedder resident (`jina-clip-v2` INT8) | 1126 | 2048 | Estimated |
| LanceDB (one table, vectors + facts) | 50 | 130 | Estimated |

**"Driving" means OS + voice + SmolVLA, with no VLM and no LanceDB resident.** That is the
definition this budget uses, and it is load-bearing: the VLA policy is the wheel-rate control
loop, and neither the VLM nor the RAG store is in it. Both driving variants are priced below,
because "does driving fit" has two different answers depending on which one is meant.

| Configuration | Components | Total (MiB) | Total (GiB) | Headroom vs 7620 | Verdict |
| --- | --- | ---: | ---: | ---: | --- |
| **Driving — reflex** | OS + voice + VLA | **3932–4398** | **3.84–4.29** | **+3222 … +3688** | **Fits**, wide margin |
| **Driving — VLM in the loop** | OS + voice + VLA + VLM | **9061–10116** | **8.85–9.88** | **−1441 … −2496** | **Does not fit** |
| **Retrieval** | OS + voice + VLM + embedder + LanceDB; VLA paused | **8742–10389** | **8.54–10.15** | **−1122 … −2769** | **Does not fit** |
| **All-resident** | OS + voice + VLM + VLA + embedder + LanceDB | **10237–12294** | **10.00–12.01** | **−2617 … −4674** | **Does not fit** |

### Delta against the previous revision

Two line items moved. The VLM dominates; LanceDB is a rounding error next to it.

| Row | Pre-INT4 (Chroma+SQLite) | INT4 succeeds + Chroma | **This: INT4 succeeds + LanceDB** | Net vs pre-INT4 |
| --- | ---: | ---: | ---: | ---: |
| VLM resident | 3488–3838 | 5129–5718 | **5129–5718** | **+1641 … +1880 MiB** |
| Store | 205–399 Chroma+SQLite | 205–399 | **50–130 LanceDB** | **−155 … −269 MiB** |
| Driving — reflex | 3932–4398 | 3932–4398 | **3932–4398** | **0** — no VLM, no store |
| Retrieval | 7256–8778 | 8897–10658 | **8742–10389** | **+1486 … +1611 MiB** |
| All-resident | 8751–10683 | 10392–12563 | **10237–12294** | **+1486 … +1611 MiB** |

LanceDB vs the INT4+Chroma intermediate: retrieval and all-resident each drop **155–269 MiB**.
That does not pull either row under the pool. Retrieval's optimistic bound is still **−1122 MiB**.
Even dropping the embedder leaves OS + voice + VLM + LanceDB at **7616–8341 MiB (−721 … +4)**.

The direction of the VLM delta is worth stating plainly, because it is the opposite of what
"assume it works" usually buys. Granting success did not make the VLM cheaper. It replaced a
**band** — 3.41–3.75 GiB, whose real width was unbounded because nothing constrained the top —
with a **measurement-anchored value** that is 1.6–1.8 GiB higher and roughly 0.6 GiB wide. Lance
then clawed back a quarter-gigabyte. The estimate got better and the news stayed worse.

## Co-residency: what actually fits

The decision this budget exists to support is not the total but **what can be resident at once.**
Voice and the OS baseline are included in every row.

| Co-residency set | Total (MiB) | Total (GiB) | Headroom vs 7620 | Verdict |
| --- | ---: | ---: | ---: | --- |
| Voice only | 2437–2493 | 2.38–2.43 | +5127 … +5183 | **Fits** |
| Voice + embedder + LanceDB (RAG, no VLM) | 3613–4671 | 3.53–4.56 | +2949 … +4007 | **Fits** |
| Voice + VLA — **driving mode** (no LanceDB) | 3932–4398 | 3.84–4.29 | +3222 … +3688 | **Fits** |
| Voice + toy engine build (68 KB graph, measured) | 3928–3984 | 3.84–3.89 | +3636 … +3692 | **Fits** |
| Voice + VLA + embedder + LanceDB | 5108–6576 | 4.99–6.42 | +1044 … +2512 | **Fits** |
| Voice + VLM, **ViT evicted** — text reasoning | 6280–6925 | 6.13–6.76 | +695 … +1340 | **Fits** |
| Voice + VLM — **reasoning mode, ViT resident** | 7566–8211 | 7.39–8.02 | +54 … −591 | **Does not fit** |
| Voice + VLM + LanceDB (no embedder) | 7616–8341 | 7.44–8.15 | +4 … −721 | **Does not fit** |
| Voice + VLM + embedder + LanceDB — **retrieval mode** | 8742–10389 | 8.54–10.15 | −1122 … −2769 | **Does not fit** |
| Voice + VLM + VLA — **driving with the VLM in the loop** | 9061–10116 | 8.85–9.88 | −1441 … −2496 | **Does not fit** |
| **Everything resident** | **10237–12294** | **10.00–12.01** | **−2617 … −4674** | **Does not fit** |
| Any set + `llm_build` for a 3B VLM | build-time only | — | — | **Forbidden as a co-residency set** — see below |

Five conclusions follow directly:

1. **The VLM is now the only large tenant, not one of several.** At 5129–5718 MiB it does not
   co-reside with the VLA, the embedder, or LanceDB. The previous revision could still describe
   "one large GPU tenant at a time" as a policy; it is now an arithmetic fact with only one
   satisfying assignment.
2. **Evicting the FP16 vision tower is mandatory, not an optimisation.** It is the only lever
   worth more than 100 MiB, and without it even reasoning-mode-alone fails.
3. **Voice is free and stays up.** At 194–224 MiB it is the cheapest thing on the board, it is
   CPU-only, and it never contends for the GPU. It should remain resident across every mode so the
   robot can always be spoken to, including while a model is being swapped.
4. **Retrieval is two passes, permanently.** Not "encode-then-unload if convenient" — the VLM and
   the embedder overrun the pool by 1.1–2.7 GiB together even with a cheap Lance store, so there
   is no resident set that serves RAG and reasoning in one pass. The harness needs a real unload
   path, not a hope.
5. **Driving is safe and should stay VLM-free and store-free.** It fits with +3.1 to +3.6 GiB of
   headroom precisely because neither the VLM nor LanceDB is in it. Putting the VLM in the control
   loop costs 1.4–2.4 GiB more than the board has.

### If a 3B is not required: the measured alternative

Every 3B figure above is an interpolation. Two Orin Nano rows are actual measurements, and
switching to one turns the largest line in this budget from bench-anchored into measured:

| Alternative | Reasoning mode (OS + voice + VLM) | Retrieval mode | Note |
| --- | ---: | ---: | --- |
| `Qwen3-VL-2B-Instruct`, 4380 MiB **measured on Nano** | **6817–6873 (+747 … +803)** — **fits** | 7993–9051 (−373 … −1431) — no | Supported Edge-LLM ID; ~750 MiB cheaper than the 3B's optimistic bound |
| `Qwen3.5-0.8B`, 2603 MiB **measured on Nano** | 5040–5096 (+2524 … +2580) — fits wide | 6216–7274 (+346 … +1404) — **marginal** | The only VLM that comes close to co-residing with the embedder, and even it does not clear the reserve at the pessimistic bound |

`Qwen3-VL-2B-Instruct` is the recommendation if reasoning quality permits: it removes the
extrapolation, removes the `llm_build` unknown for a 3B graph, and is the largest VLM NVIDIA
actually publishes on this board.

## Build time: `llm_build`, with nothing else resident

**This is not an operating row and is not charged against any configuration above.** It is a
scheduled window.

| Metric | Value |
| --- | ---: |
| Budget available with only OS + voice up | **7620 − 2269 − 224 = 5127 MiB** |
| Measured builder peak, 68 KB toy graph (G1) | **1491 MiB** |
| Builder peak for a 3B VLM ONNX graph | **unmeasured** |

The measured 1491 MiB was for a **68 KB** graph, so it is a floor and not a forecast. Edge-LLM
requires the engine to be built **on the device**, so unlike the ONNX export this cost cannot be
moved to the workstation. `--externalize-weights int4_ffn` was added in 0.8.0 specifically to cut
engine-build host memory and is required for Orin INT4 — use it, and build the LLM and ViT engines
**sequentially**, never concurrently.

**Give every engine build a dedicated window with the voice stack as its only other tenant.** Never
build while the VLM, the VLA policy, the embedder, or any PyTorch process is loaded. This remains
the single most likely way to OOM this board — but it is a *scheduling* failure, not a residency
one, which is why it no longer inflates the steady-state rows.

## Rejected: the 3.90 GiB and 4.35 GB all-resident tables

Both Gemini-authored tables are **false** and must not be used to size anything. They share one
root error — treating this SoC as if it had discrete VRAM and a 0.30 GiB operating system — and
several independent ones.

| Claimed row | Claimed | Actual | Why the claim fails |
| --- | ---: | ---: | --- |
| OS / NITROS | 0.30 GiB | **2.19–2.22 GiB** OS; NITROS **0** | `tegrastats` measured 2243–2269 MiB at F4/F5 idle. ROS 2, Isaac ROS and NITROS are **not installed**, so the NITROS line is both speculative and, via `nvargus-daemon`, double-counted. Wrong by ~1.9 GiB. |
| Qwen2.5-VL-3B INT4 | 1.80 GiB | **5.01–5.58 GiB** resident (bench-anchored) | Ignores the FP16 vision tower — weights alone are 3.2 GiB — then also omits KV, TRT runtime, activations, and the C++ `llm_inference` process. NVIDIA's own Nano 2B VLM rows are already 4.38–4.62 GiB. |
| `smolvla-jetbot` FP16/INT8 | 1.20–1.35 GiB | **1.46–1.86 GiB** | The checkpoint **does not exist**. `lerobot/smolvla_base` weights are ~0.88 GiB, so the weights claim was generous; what is missing is the PyTorch and CUDA runtime around them. |
| Embedder INT8 | 0.40 GiB (SigLIP 2) / 0.90 GB (Jina) | **1.1–2.0 GiB** resident | Weights-only accounting. 865M × 1 B = 825 MiB is the *graph*, not the resident set; engines and 512² activations are extra. |
| Audio | 0 | **0.19–0.22 GiB** | CPU-only is not free on unified memory. It is cheap — but it is not zero. |
| PyCUDA runtime on Tegra | assumed | **not available** | G1 used ctypes `libcudart` and apt `python3-libnvinfer`. PyPI `pycuda`/`tensorrt` wheels are the wrong ABI for Tegra. |
| **Total vs "5.2 GB usable"** | **3.90 / 4.35** | **10.00–12.01 GiB** all-resident | The 5.2 GiB figure is the **model** budget *after* subtracting the OS. Charging a 0.30 GiB OS line against it counts the operating system twice. |

Swapping the embedder from 0.40 to 0.90 does not repair the other rows. Neither table is a
starting point for negotiation.

## What is still unmeasured

**The largest line item in this budget has never been loaded on this board.** Closing that is the
only way to tighten the total.

| Unmeasured | Estimate | Blocked by | What would close it |
| --- | ---: | --- | --- |
| Qwen2.5-VL-3B via Edge-LLM INT4 AWQ | **5.01–5.58 GiB** resident (bench-anchored) | **G2** — no ONNX tree exported, no SM 87 engine built, no x86 export host used | Peak process RSS and a `tegrastats` trough from a real decode, **with and without the vision tower resident**. Report KV separately from weights. NVIDIA's Nano 8 GB table **omits** this 3B VLM. |
| `llm_build` peak for a 3B graph | **unknown, ≫ 1491 MiB** | **G2** | The build itself, on an otherwise idle board, with `tegrastats` sampled throughout. Do this **first** — it gates whether the VLM is reachable at all. |
| `lerobot/smolvla_base` under eager PyTorch | 1.46–1.86 GiB | **#30 / G3** — torch not installed, weights not downloaded | RSS of a dummy forward. Separately: RSS of a bare `import torch` plus a CUDA context, before any weights, to split the framework cost from the model cost. |
| SmolVLA prefix + denoise-expert TRT split | would replace the line above | **#18** — no ONNX export of either subgraph | Export both subgraphs, build FP16 engines on-device with the VLM unloaded, measure. INT8 PTQ needs a calibration set this robot does not have. |
| `jinaai/jina-clip-v2` INT8 | 1.1–2.0 GiB resident | **G4 / Stage I** — Hub ONNX listed, **not** fetched, no TRT engine | Peak RSS of ORT or `trtexec` at 512×512, batch 1; confirm INT8 accuracy against the FP16 graph before trusting it. |
| LanceDB 50k × 256-d + facts columns | 0.05–0.13 GiB resident; ~50–200 MiB disk | Stage I not started; package never imported here | RSS of `import lancedb` then of a representative IVF-PQ query; `smaps` to split RSS from mmap. Do **not** build HNSW. |
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
