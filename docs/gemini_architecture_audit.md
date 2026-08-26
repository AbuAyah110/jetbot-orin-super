# Gemini “Zero-Bloat JetBot” architecture audit

Evidence-only. **No stack was implemented in this pass.** Numbers are MiB
(1024²) unless labeled GiB. Host: Jetson Orin Nano Super 8 GB unified,
L4T R36.4.4 / JetPack 6.2.1, SM 87, TensorRT 10.3. Interactive companion:
`~/.cursor/projects/home-impulse110-Documents/canvases/jetbot-memory-budget.canvas.tsx`.

Primary sources (some live on other branches, not copied here):
`docs/memory_budget.md` on `stage-g-membudget`;
`docs/bringup/07-edgellm-workstation-quant.md` on `stage-g-edgellm-workstation`;
`docs/bringup/07-smolvla-trt.md` on `stage-g-smolvla-trt`;
[`docs/bringup/07-tensorrt-g1.md`](bringup/07-tensorrt-g1.md)
and [`docs/bringup/06-voice.md`](bringup/06-voice.md) on this branch; local
TensorRT-Edge-LLM v0.10.0 clone at `/home/impulse110/Documents/_edgellm_ref/repo`;
NVIDIA [support matrix](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html)
and [v0.9.0 Orin Nano 8 GB benches](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html).

G1’s line “there is no NVIDIA product called TensorRT Edge-LLM” is **stale**.
The workstation-quant note already corrected that; this audit treats Edge-LLM
v0.10.0 as real.

---

## Scoreboard

| Claim | Verdict | Evidence | Corrected number |
| --- | --- | --- | --- |
| AWQ INT4 “does not work” is an incorrect assumption | **half** | AutoAWQ **on Jetson** still does not work (community card + G1). Edge-LLM **INT4 AWQ plugins** on Orin **are** in the NVIDIA matrix (Compatible JP6.2+, FP16/INT8/INT4 only, **build on device**). Those are different pipelines. | Use Edge-LLM export + `llm_build` on device, not AutoAWQ-on-Tegra |
| NVIDIA demos Orin Nano 8 GB INT4 AWQ | **half** | v0.9.0 Nano 8 GB table has INT4 AWQ for **Qwen3-0.6B / 1.7B** and **Qwen3-VL-2B / Qwen3.5-2B**. **No Qwen2.5-VL-3B row** on Nano 8 GB. Jetson AI Lab’s “Qwen3-4B → ~2 GB” is a **text** LLM. | Do not treat a 4B text demo as a 3B VLM footprint |
| sm_87 native weight-only INT4 in Edge-LLM | **half** | INT4 is **legal** on Orin. Runtime is Edge-LLM **INT4 groupwise GEMM plugins**, not Hopper-style native FP8 Tensor Cores. `DataType.INT4` in TRT 10.3 is Q/DQ WOQ, not an AWQ loader by itself. | Plugin INT4 WOQ, SM 87; FP8 engines forbidden |
| 4B params → ~2 GB (applied to Qwen2.5-VL-3B) | **false** | Official `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` is **~3.18 GiB** on disk; community AWQ dump **3244 MiB**. Both leave **`visual` in FP16**. 3B INT4 **text** ≈ 1.5–1.8 GiB; the rest is the vision tower. | Weights **~3.2 GiB**, not 1.80 |
| Quantize with `tensorrt-edgellm-quantize … --quantization int4_awq` | **true** (flag) | Real CLI choice in v0.10.0 `quantize.py`. Subcommand is `llm`, not a bare model-name first arg. `--visual_quantization` is **`fp8` only**; unset → vision **FP16**. Prefer **CPU export** of official AWQ and **skip** re-quantize. | Flag exists; do not pass `--visual_quantization` on Orin |
| Official `Qwen/Qwen2.5-VL-3B-Instruct` supported | **true** | Edge-LLM supported-models list + Orin VLM CI `Qwen2.5-VL-3B-Instruct-int4_awq`. CI is Orin-family, **not** an 8 GB memory guarantee. | Supported ID; 8 GB fit **unmeasured** |
| Vision typically stays FP16 | **true** | Official AWQ `modules_to_not_convert: ["visual"]`; NVIDIA benches labeled `INT4 AWQ / FP16`. | Charge FP16 ViT separately from INT4 LLM |
| Encoder is not Edge-LLM (SigLIP 2 via TRT) | **true** | SigLIP 2 is a dual encoder; Edge-LLM is the LLM/VLM decode path. | Keep it on TRT / ONNX |
| ONNX + ModelOpt PTQ INT8 + `trtexec` | **half** | Community ONNX exists for some SigLIP 2 So400m IDs; **this board has never exported or built it**. INT8 PTQ **needs calibration images**. ModelOpt is **absent on Jetson** (quantize on workstation). | Unmeasured on this SoC |
| Base 0.4B → ~400 MB INT8; Large ~900 MB; So400m ~1.0 GB | **half** / optimistic | Those look like **weights-only** (`N params × 1 byte`) and mix **Base** with **So400m (~400M ViT)**. Dual encoder = image **plus** text towers. Activations, TRT runtime, and I/O buffers are extra. | So400m-class INT8 resident **~0.6–1.2 GiB** estimated; Base is smaller if it is actually ViT-B |
| PyCUDA as the Jetson TRT runtime | **false** | G1: `pycuda` **absent**; PyPI `pycuda`/`tensorrt` is the **wrong ABI** (Tegra vs SBSA). Proven path: apt `python3-libnvinfer` + ctypes `libcudart`. | No PyCUDA |
| All-resident **3.90 GiB** vs 5.2 usable | **false** | 5.2 GiB **model budget** (7620 − OS) is the one good number in the pitch. The **0.30 GiB OS/NITROS** row is wrong by ~2 GiB. Qwen 1.80 omits vision+KV+runtime. `smolvla-jetbot` **does not exist**. | See honest range below |
| `smolvla-jetbot` 1.20 GiB | **false** (name) / **unmeasured** (TRT size) | Only `lerobot/smolvla_base` (~450M, **~900 MiB BF16**). Not a one-shot ONNX (`07-smolvla-trt.md`). Torch still required until export exists (#30 / #18). | Weights **~900 MiB**; with torch **1.46–1.86 GiB**; TRT **unmeasured** |
| Zipformer + Piper &lt; 0.20 GiB | **unmeasured** | Current shipped voice is FastConformer CTC + Matcha/HiFi-GAN v2: **520–673 MiB** (F4+F5). Sherpa-ONNX **already is** the runtime; Zipformer/Piper is a **model swap**, not a new stack. | Keep **0.52–0.67 GiB** until F-zipformer is gated |
| OS / NITROS 0.30 GiB | **false** | `tegrastats` **2243–2269 MiB** at F4/F5 idle, including `nvargus-daemon` (~181 MiB). ROS 2 / Isaac / NITROS **not installed**. G1 idle sample during build was **2824 MiB**. | OS **2.19–2.21 GiB**; NITROS **0** |
| Workstation: torch, ModelOpt, transformers | **true** (spirit) | Matches Edge-LLM install: export/quantize on **x86**; quantize needs CC 8.0+ GPU. | Keep heavy Python off the Jetson |
| Jetson: Edge-LLM C++, TRT, PyCUDA, Sherpa-ONNX | **half** | C++ `llm_inference` + TRT + Sherpa-ONNX: right direction. **PyCUDA false.** SmolVLA TRT **not ready**. Thin Python harness is still expected. | ctypes TRT; Sherpa already in `.venv` |
| Dynamic unload VLA vs SigLIP (pause motors to search) | **half** (ops, not RAM arithmetic) | Memory budget already requires **one large GPU tenant**. Pausing motion for RAG is acceptable **only** with an explicit mode switch and watchdog (`cmd_vel=0`). It is **not** a substitute for a 3.90 GiB all-resident table. | Mode switch, not co-residency |

---

## 1. INT4 AWQ on Orin Nano Super via Edge-LLM

**What is true.** Jetson Orin is **Compatible** on JetPack 6.2+ / CUDA 12.6 with
**FP16, INT8, and INT4 only**. “Jetson Orin does not run FP8 or FP4 model
engines.” Official IDs include `Qwen/Qwen2.5-VL-3B-Instruct` and
`Qwen/Qwen2.5-VL-3B-Instruct-AWQ`. `--quantization int4_awq` is a real
`tensorrt-edgellm-quantize` choice. Engines **build on the device** (wrong SM
if you copy an x86 `.engine`).

**What Gemini collapsed.** NVIDIA’s Nano **8 GB** published inference rows are
smaller VLMs at **INT4 LLM / FP16 ViT**: Qwen3-VL-2B **4380 MB** GPU mem,
Qwen3.5-2B VLM **4621 MB**. That is already most of the unified pool **for the
VLM alone**, and it is **not** Qwen2.5-VL-3B. A 4B **text** INT4 checkpoint
near 2 GB is not a 3B **VLM** with an FP16 tower.

**Weights vs resident.** Disk AWQ ≈ **3244 MiB**. Resident still adds KV
(36 KiB/token → 72 / 144 / 288 MiB at 2k / 4k / 8k), visual tokens, TRT
runtime, and builder peak. G1 measured **1491 MiB RSS** building a **68 KB**
ONNX graph. `llm_build` OOM on 8 GB remains the ranked #1 risk in
`07-edgellm-workstation-quant.md`. Community AutoAWQ safetensors are **not** a
drop-in Edge-LLM engine.

---

## 2. SigLIP 2 INT8 via TensorRT

**Direction is right.** Replacing `llama-nemotron-embed-vl-1b-v2` (~1.7B,
**1700 MiB INT8 aspirational / 3400 MiB FP16**) with a SigLIP 2–class dual
encoder is the same “smaller embedder” move already recommended in the memory
budget (and G1 §G4). That is the highest-leverage RAG change.

**Sizes are optimistic.** ~400 MB INT8 is **weights-only** for a ~0.4B **single**
tower. SigLIP 2 So400m is a ~400M **ViT plus** a text encoder; INT8 PTQ needs a
calibration set; TRT activations and runtime are extra; two embeddings (image +
text) both have to exist at query time. ONNX export is **plausible** (community
ONNX repos exist for some So400m IDs; NVIDIA TAO also documents a SigLIP 2
ONNX/`trtexec` path) but **unmeasured here**. Do not `pip install pycuda` on
Tegra.

---

## 3. The 3.90 GiB all-resident table

There is **no discrete VRAM**. CPU voice and GPU engines share `tegrastats`
**7620 MiB**.

| Row | Gemini GiB | Honest GiB | Why |
| --- | ---: | ---: | --- |
| OS / NITROS | 0.30 | **2.19–2.21** (OS); NITROS **0** | F4/F5 idle RAM 2243–2269 MiB |
| Qwen2.5-VL-3B INT4 | 1.80 | **3.4–5.5** resident (est.) | 3.2 GiB weights + KV + runtime; Nano 2B VLM benches 4.4–4.6 GB GPU mem as a floor for “small VLM,” not a 3B proof |
| SigLIP 2 Base INT8 | 0.40 | **0.5–1.2** | Weights-only vs dual encoder + TRT |
| smolvla-jetbot | 1.20 | **0.88** weights / **1.46–1.86** with torch / TRT **unmeasured** | Fictitious checkpoint name; no ONNX |
| Zipformer + Piper | &lt;0.20 | **0.20 unmeasured** or **0.52–0.67** current | Sherpa-ONNX already running |
| **Sum vs 7.44 GiB pool** | **3.90 vs 5.2 “usable”** | **~7.2–11.2** | 5.2 is **model** headroom after OS, not a place to hide a 0.30 OS line |

**Torch-not-on-device** is the right *strategy* for VLM **if** Edge-LLM export
works. It is **not** true for SmolVLA until a real ONNX graph exists: PyTorch
on Jetson remains the #18 dummy-forward path (#30).

---

## 4. Library split

| Piece | Gemini | This repo |
| --- | --- | --- |
| Workstation torch / ModelOpt / transformers | Yes | Yes for Edge-LLM quant/export |
| Jetson Edge-LLM C++ decode | Yes | Yes, after ONNX copy + on-device build |
| Jetson TensorRT | Yes | G1 passed on TRT 10.3 |
| Jetson PyCUDA | Yes | **No** — ctypes + apt bindings |
| SmolVLA TRT | Implied ready | **Not ready** |
| Sherpa-ONNX | New install | **Already** F4/F5 runtime |

---

## If the user actually makes those changes, does all-resident fit?

**No.** Give a range, not 3.90.

Assume the architectural moves **succeed**: official Qwen AWQ exported, Edge-LLM
engines built, SigLIP 2 INT8 TRT, SmolVLA someday TRT, Zipformer+Piper, **no**
ROS/NITROS, **no** 1.7B Nemotron embedder.

| Bound | Total vs 7620 MiB pool | Headroom |
| --- | ---: | ---: |
| Optimistic (min OS, Hub-like VLM + small KV, tiny SigLIP, BF16-weight VLA, Zipformer 200 MiB) | **~7.2 GiB** | **&lt; 0.3 GiB** — below the 400 MiB reserve |
| Typical (Nano 2B-VLM GPU-mem proxy ~4.4 GiB **plus** OS 2.2, current voice, SigLIP, VLA) | **~8.5–9.5 GiB** | **Does not fit** |
| Pessimistic (3B VLM mid-5 GiB + torch VLA + current voice + larger SigLIP) | **~10–11 GiB** | **Does not fit** |

The 5.2 GiB “usable” figure **already subtracted OS**. Gemini then charged OS
again at 0.30 and understated every GPU tenant. Even the optimistic stack is
**marginal at best** and depends on unmeasured Qwen-3B **inference** (not just
weights) and an ONNX SmolVLA that does not exist.

**Highest-risk item:** `llm_build` **OOM** on 8 GB unified RAM for
Qwen2.5-VL-3B (absent from Nano 8 GB benches; builder ≫ G1’s 1.5 GiB toy).
**Second:** SmolVLA **ONNX export** of prefix + denoise expert (#30 / #18).
SigLIP INT8 PTQ is third (calibration + TRT), not the blocker.

**Pause-motors-to-search:** acceptable as an **explicit** reasoning/RAG mode
with the motor watchdog holding stop. It does **not** make 3.90 GiB true; it
is how you **avoid** all-resident in the first place (already rec. 4 in the
memory budget).

---

## Optional issue notes

**#17 (G2 VLM).** INT4 AWQ via Edge-LLM is a **real NVIDIA path**; AutoAWQ-on-Jetson
is the wrong objection. The ticket should not be closed by this audit: no ONNX
tree and no SM87 engine exist on this host. llama.cpp + GGUF remains the path
that does not wait on a PC export or a multi-hour builder.

**#18 / #30 (SmolVLA).** TensorRT is a **future** memory win **if** subgraph
export works. Until torch is installed and a dummy forward runs, G3 stays
eager PyTorch, not Gemini’s one-graph pycuda plan.
