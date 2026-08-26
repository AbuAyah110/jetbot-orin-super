# G2 workstation path — Edge-LLM INT4 export of Qwen2.5-VL-3B

Evidence record. **Not a plan to implement on this Jetson in this session.** No HF 7 GB download, no engine build, no torch in `.venv`.

Verdict: **conditional yes.** A user can produce INT4 AWQ ONNX for `Qwen/Qwen2.5-VL-3B-Instruct` on an x86 Ubuntu workstation (CPU export of a supported AWQ checkpoint; GPU only if re-quantizing), copy **that ONNX tree** here, and compile SM87 TensorRT engines with the C++ builders. The decode path is C++ `llm_inference`. Do **not** request FP8 or NVFP4. The most likely failure is **Orin Nano 8 GB OOM during `llm_build`**, not the workstation export.

Pinned source: local clone `/home/impulse110/Documents/_edgellm_ref/repo` = NVIDIA/TensorRT-Edge-LLM **v0.10.0** (`bb29145`, `tensorrt_edgellm._version.__version__ = "0.10.0"`). Docs: [latest site](https://nvidia.github.io/TensorRT-Edge-LLM/latest/).

This board (do not re-derive): Jetson Orin Nano Super 8 GB, L4T R36.4.4 / JetPack 6.2.1, CUDA 12.6, TensorRT 10.3, SM **87**.

Parent should later cross-link this file from `docs/bringup/07-tensorrt.md` and `JETBOT_SPEC.md`. Those files still say there is no NVIDIA product named TensorRT Edge-LLM; that is **stale** (see below). They were left untouched because they are dirty in other worktrees.

---

## Gemini's three-step claim, claim by claim

Gemini hypothesized: (1) x86 or Thor: HF → ONNX + quantize INT4 AWQ or FP8; (2) copy ONNX to Orin, C++ builder compiles a device-specific engine; (3) native C++ `llm_inference`, no Python on the decode path. Also: chunked prefill, EAGLE-3, FP16/FP8/INT4 AWQ/GPTQ/NVFP4 with FP8 on SM89+ and NVFP4 on SM100+.

| Claim | Result | NVIDIA source |
| --- | --- | --- |
| 3-step: workstation export → copy ONNX → device engine + C++ runtime | **Confirmed**, with one split: **quantize is optional** if you start from a supported pre-quantized checkpoint. Export is a separate CLI from quantize. | [Installation](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/installation.html) Part 1 vs Part 2; [Quick Start](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/quick-start-guide.html) |
| Quantize on x86 **or Thor** | **Partially false.** Installation: export/quantize run on an **x86 host**; C++ runtime is the edge device (Thor/Orin/DRIVE/Spark). Thor can *build engines and run*, but the documented quantize GPU is the x86 host, CC 8.0+. | Installation: “Export and quantization (runs on an x86 host…)” |
| INT4 AWQ **or FP8** for this Orin | **False for FP8.** Orin: **FP16, INT8, INT4 only.** “Jetson Orin does not run FP8 or FP4 model engines.” Request **INT4 AWQ** (or INT8 SQ). | [Support matrix](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/support-matrix.html) |
| Copy ONNX; C++ builder is device-specific | **Confirmed.** Matrix **Build location: Device** for Jetson Orin. Engines are TensorRT + SM specific; x86 developer engines are not SM87. ONNX is also **not portable across Edge-LLM or TensorRT versions** — pin the same git tag on both machines. | Support matrix; [Engine builder](https://nvidia.github.io/TensorRT-Edge-LLM/latest/developer_guide/software-design/engine-builder.html) version warning |
| Native C++ `llm_inference`, no Python on decode | **Mostly confirmed.** Binary is `./build/examples/llm/llm_inference`. Overview: deployment runtime is free of Python. A **thin Python wrapper is still fine** for the agent harness; the experimental OpenAI server is Python over the C++ binding. Do not claim “zero Python on the Jetson.” | [Overview](https://nvidia.github.io/TensorRT-Edge-LLM/latest/overview.html); Quick Start §3–4 |
| Chunked prefill | **Present as a plugin contract**, not a user-facing “enable EAGLE” switch. `KVCacheStartIndex` documents chunked prefill. | `docs/source/developer_guide/customization/tensorrt-plugins.md` |
| EAGLE-3 | **Real, not for this Nano 8 GB plan.** Extra draft engine + memory. Orin test list exists (`tests/test_lists/l1_pipeline_orin_eagle.yml`) but NVIDIA’s Orin Nano 8 GB published benches omit Qwen2.5-VL-3B EAGLE. Skip. | Support matrix KV-reuse; performance benches |
| FP16 / FP8 / INT4 AWQ / GPTQ / NVFP4; FP8 = SM89+; NVFP4 = SM100+ | **Confirmed as product-wide.** This SM87 board: FP16, INT8, INT4 only. | [Performance precision key](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/performance/performance-benchmarks.html) |

---

## 1. Does the export/quantize path run on x86 Ubuntu with a dGPU?

**Yes**, that is the documented **Developer** row: `x86-64 Linux GPU`, Ubuntu 22.04 / 24.04, CUDA **12.x or 13.x**, TensorRT “compatible user package”, build location **Workstation**, purpose “Development and validation.” That row is **not** an edge deploy target.

Split the two Python CLIs:

| Step | GPU? | TensorRT on the PC? | Requirements |
| --- | --- | --- | --- |
| `tensorrt-edgellm-export` | **No.** “Export runs on CPU.” | **Not required.** | x86-64 Linux, Python **3.10+**, CPU RAM **≥ 1.5× checkpoint size**. Pin `pip install -e .` from **v0.10.0**. `pyproject.toml` pins `torch==2.13.0`, `transformers==5.14.1`, `onnx==1.19.0`. |
| `tensorrt-edgellm-quantize` | **Yes.** NVIDIA GPU **Compute Capability 8.0+** (Ampere+). CUDA 12.x or 13.x. | Not the C++ TRT package; ModelOpt uses CUDA/PyTorch. | GPU memory **at least the FP16 checkpoint size**. Extra: `pip install -e ".[tools]"` (`nvidia-modelopt==0.45.0`). Device default is `cuda`. |
| C++ `llm_build` on the **PC** | Optional developer path | Yes, matching CUDA/TRT | Produces **PC SM** engines. **Do not copy those to Orin.** |

**CPU-only x86:** export of a **pre-quantized** checkpoint is documented to work. **Quantize will not** (needs CUDA GPU). Prefer skipping quantize: `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` is an official supported ID.

**Thor vs workstation:** Thor is an Official **device** (JetPack 7.x, CUDA 13.x, build on device). It is not a substitute for the x86 quantize host in the installation guide. This Jetson Orin Nano Super is **Compatible** (JetPack 6.2+ / CUDA 12.6), not the Official Orin row (Official Orin is JetPack **7.2** / CUDA **13.2**). Compatible is still the row that matches this board.

Recommended PC container from Installation: `nvcr.io/nvidia/pytorch:25.12-py3`.

---

## 2. Is Qwen2.5-VL-3B-Instruct an official checkpoint ID?

**Yes, on current v0.10.0.** [Supported Models](https://nvidia.github.io/TensorRT-Edge-LLM/latest/user_guide/getting_started/supported-models.html) lists:

- `Qwen/Qwen2.5-VL-3B-Instruct`
- `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`
- (and 7B + NVIDIA FP8/NVFP4 7B, which this Orin cannot run as FP8/FP4 engines)

Exporter `model_type` includes `qwen2_5_vl`. C++ `Qwen25VLViTRunner` is in `cpp/multimodal/qwen25vlViTRunner.h`.

Qwen3-VL is **also** current; it did **not** replace 2.5. First listed in CHANGELOG **0.1.0** (“Qwen2.5-VL 3B”). **Do not pin an old tag** to keep 2.5 — use **v0.10.0 on both workstation and Orin**.

Orin VLM CI explicitly builds `Qwen2.5-VL-3B-Instruct-int4_awq` (`tests/test_lists/l1_pipeline_orin_vlm.yml`). That list is not “Nano 8 GB only”; treat it as Orin-family, not a memory guarantee for 8 GB unified.

HF sizes (API `usedStorage`, no download on this board):

| Checkpoint | Disk |
| --- | --- |
| `Qwen/Qwen2.5-VL-3B-Instruct` | **~7.00 GiB** (`7509337976`) |
| `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` | **~3.18 GiB** (`3413207823`) — `quantization_config.modules_to_not_convert: ["visual"]` |

Community `Azaz666/Qwen2.5-VL-3B-Instruct-AWQ-INT4` (~3244 MiB, vision FP16) is **not** on NVIDIA’s supported list. Prefer the official Qwen AWQ ID so the exporter’s AWQ layout parser is the one NVIDIA tests.

---

## 3. INT4 AWQ: what the package actually emits

`tensorrt-edgellm-quantize --quantization int4_awq` uses ModelOpt `mtq.INT4_AWQ_CFG` (`tensorrt_edgellm/quantization/quantization_configs.py`). The exporter then **repacks** tensors for TensorRT Edge-LLM:

- Runtime names: `int4_awq` (column-packed int32) and `int4_awq_modelopt` (packed uint8).
- ONNX uses **INT4 groupwise GEMM plugins**, not a generic AutoAWQ/Marlin loader. Default export is plugin **V2** (cuteDSL `Int4GroupwiseGemmPluginV2`, fragment layout). Limitations 0.10.0: V2 **may degrade accuracy**; fallback `--int4-gemm-plugin-version 1` (legacy AWQ-swizzled `Int4GroupwiseGemmPlugin`).
- GPTQ is **load-only**; the quantize package does not create GPTQ.

Calibration (quantize CLI):

| Modality | Flag | Default | Used for |
| --- | --- | --- | --- |
| Text | `--text_dataset` | `cnn_dailymail` | LLM / LM head / KV |
| Image | `--image_dataset` | `mmmu` | **only** with `--visual_quantization` |
| Samples | `--num_samples` | 512 | |

**Visual tower:** `--visual_quantization` choices are **`fp8` only**. Unset → vision stays **FP16**. On Orin, do **not** set visual FP8. INT4 AWQ applies to the **LLM backbone**; vision remains FP16 in both official Qwen AWQ and NVIDIA’s VLM benches (`INT4 AWQ / FP16`).

**Workstation peak VRAM (quantize):** Installation: GPU memory **≥ FP16 checkpoint size** ≈ **7 GB** for this 3B VLM, plus calibration activations. Practical:

| PC GPU | Quantize BF16 → INT4 AWQ | CPU export of official AWQ |
| --- | --- | --- |
| CPU-only | No | Yes (need ~5+ GB RAM; 1.5× 3.18 GiB) |
| 8 GB | Borderline / likely tight | Yes |
| 12 GB | Likely OK | Yes |
| 16–24 GB | Comfortable | Yes |

**Recommended:** skip quantize; export `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` on CPU. Only quantize if you need a ModelOpt unified checkpoint different from Qwen’s AWQ.

---

## 4. What actually transfers to the Jetson

Copy the **complete** `tensorrt-edgellm-export` output directory (Quick Start `rsync -a .../onnx/`). For this VLM that is at least:

```
onnx/
  llm/          # model.onnx + config / tokenizer sidecars
  visual/       # model.onnx + preprocessor_config.json
```

Plus, if used: `external_int4_ffn_weights.safetensors` (and related) next to the LLM graph.

**Not** required on the Jetson after a successful export: the original **BF16** 7 GB HF shards. Keep tokenizer/processor files that the exporter wrote into `llm/` (do not assume “ONNX only”).

Approximate disk (order of magnitude, not measured here):

| Artifact | Size |
| --- | --- |
| Official AWQ HF (stay on the PC) | ~3.2 GiB |
| BF16 HF (PC only if re-quantizing) | ~7.0 GiB |
| Exported ONNX tree (INT4 LLM + FP16 ViT) | expect **~4–10 GiB** (weights live in ONNX + sidecars) |
| Device engines + ONNX | Installation budgets **~20–50 GB** disk for the general case; 3B INT4 should land well below the top of that range if profiles are small |

Do **not** download the 7 GB BF16 checkpoint onto this Orin.

---

## 5. Engine build on Orin (SM87)

**Build location: Device** means: run `llm_build` and `visual_build` **on this Jetson**. TensorRT serializes kernels for the **builder GPU**. An x86 engine is the wrong SM; copying it is unsupported. Same Edge-LLM **v0.10.0** + this board’s TensorRT **10.3** on both ends of the ONNX contract.

CMake for this board (Installation, JetPack 6.2+ Orin):

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCUDA_CTK_VERSION=12.6 \
    -DENABLE_CUTE_DSL=ALL
```

Binaries: `./build/examples/llm/llm_build`, `./build/examples/multimodal/visual_build`, `./build/examples/llm/llm_inference`.

There is **no** `llm_build --workspaceMB` flag. Memory is controlled by **profiles**, **externalized INT4 weights**, and what else is resident:

- `llm_build --maxBatchSize 1 --maxInputLen … --maxKVCacheCapacity …` (example default maxBatchSize is **4** — too large for Nano; NVIDIA Orin Nano benches use **batch 1**).
- `visual_build` Orin Nano spec from the benchmark doc: `--minImageTokens 8 --maxImageTokens 2048 --maxImageTokensPerImage 2048` (not Thor’s 16384).
- Export with `--externalize-weights int4_ffn` for Orin INT4 (dense). Added in 0.8.0 specifically to **cut engine-build host memory**.

G1 measured **~1.5 GB RSS** building a **68 KB** toy ONNX. That does **not** scale linearly, but it proves builder peak ≫ engine file size. A 3B VLM ONNX build will need **several GB** of unified RAM on top of graph parse.

**Fits in 8 GB with nothing else large resident?** **Unproven for Qwen2.5-VL-3B.** NVIDIA’s **v0.9.0 Orin Nano 8 GB** table has **no** Qwen2.5-VL-3B row. Closest published **inference** (not build) footprints, INT4 AWQ / FP16 ViT, batch 1:

| Model | GPU mem (MB) |
| --- | --- |
| Qwen3-VL-2B-Instruct | 4,380 |
| Qwen3.5-2B VLM | 4,621 |

A 3B Qwen2.5-VL with FP16 ViT will sit **above** those inference numbers. Inference in the mid-5 GB range could still fit **alone**; **builder + ONNX parse + FP16 ViT graph** is the squeeze. Plan: MAXN SUPER, stop voice/agent, confirm 32 GiB swap, small profiles, `int4_ffn` externalize, build LLM and ViT **sequentially**. Fail the ticket on OOM rather than leaving swap thrash overnight.

Experimental `tensorrt-edgellm-build` (ONNX-less) still compiles **on the machine that has TensorRT** — if run on x86 it is still the wrong SM. Keep the supported ONNX + C++ builder path.

---

## 6. Python vs C++ on device

| Role | Python? |
| --- | --- |
| Decode / ViT forward | **No** — `llm_inference` + `Qwen25VLViTRunner` |
| Agent harness | **Yes, thin wrapper** (JSON in/out) is OK |
| Experimental server | Python `experimental.server` + pybind `_edgellm_runtime` |
| Export/quantize | x86 only |

---

## 7. FP8 / NVFP4 / EAGLE

Out of scope for this Orin: no FP8/FP4 engines; EAGLE adds a second engine and KV. Do not export `--visual_quantization fp8` or `--quantization fp8|nvfp4|mxfp8`.

---

## 8. Workstation recipe (official commands) vs blockers

Pin **v0.10.0** on the PC (same as the clone here). Do not mix older `tensorrt_edgellm` with this C++ tree.

### Preferred (no quantize GPU)

```bash
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout v0.10.0
git submodule update --init --recursive
python3 -m venv venv && source venv/bin/activate
pip install -e .

export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-AWQ
mkdir -p "$WORKSPACE_DIR"

# CPU export of official AWQ; vision stays FP16 in the checkpoint metadata
tensorrt-edgellm-export \
  Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
  "$WORKSPACE_DIR/onnx" \
  --externalize-weights int4_ffn \
  --int4-gemm-plugin-version 1
```

V1 plugin is the 0.10.0 accuracy fallback; try V2 (default) only if V1 engines are healthy and you want the cuteDSL kernel.

```bash
rsync -a "$WORKSPACE_DIR/onnx/" <user>@orin:~/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-AWQ/onnx/
```

On Orin, after C++ build (commands above):

```bash
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-AWQ

./build/examples/llm/llm_build \
  --onnxDir "$WORKSPACE_DIR/onnx/llm" \
  --engineDir "$WORKSPACE_DIR/engines/llm" \
  --maxBatchSize 1 \
  --maxInputLen 2048 \
  --maxKVCacheCapacity 2200

./build/examples/multimodal/visual_build \
  --onnxDir "$WORKSPACE_DIR/onnx/visual" \
  --engineDir "$WORKSPACE_DIR/engines" \
  --minImageTokens 8 \
  --maxImageTokens 2048 \
  --maxImageTokensPerImage 2048

./build/examples/llm/llm_inference \
  --engineDir "$WORKSPACE_DIR/engines/llm" \
  --multimodalEngineDir "$WORKSPACE_DIR/engines" \
  --inputFile "$WORKSPACE_DIR/input.json" \
  --outputFile "$WORKSPACE_DIR/output.json"
```

KV/input lengths follow the **benchmark** VLM row (2048 / 2200), not Thor CI 8192.

### Optional re-quantize (needs Ampere+ GPU, ~7+ GB VRAM)

```bash
pip install -e ".[tools]"
tensorrt-edgellm-quantize llm \
  --model_dir Qwen/Qwen2.5-VL-3B-Instruct \
  --output_dir /tmp/qwen25vl3b_int4_awq \
  --quantization int4_awq
# do not pass --visual_quantization (would be FP8)
tensorrt-edgellm-export /tmp/qwen25vl3b_int4_awq "$WORKSPACE_DIR/onnx" \
  --externalize-weights int4_ffn \
  --int4-gemm-plugin-version 1
```

### Blockers / risks (ranked)

1. **Most likely: `llm_build` OOM on 8 GB unified RAM** (builder ≫ G1’s 1.5 GB toy; 3B VLM absent from Nano 8 GB published tables).
2. JetPack **6.2.1 is Compatible, not Official** (Official Orin is JP 7.2 / CUDA 13.2 / newer TRT). Expect extra plugin/CuTe DSL friction on TRT 10.3.
3. INT4 GEMM **V2 accuracy** (documented 0.10.0 limitation).
4. Exporting **FP8/NVFP4** by following Thor examples — engines will not run on SM87.
5. Copying an **x86 `.engine`**.
6. Feeding **community AutoAWQ** weights without going through `tensorrt-edgellm-export` (layout mismatch vs groupwise plugin).
7. Co-residency with Stage F voice during **build**.

---

## Relation to G2 / issue #17

[#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17) currently gates llama.cpp + GGUF. This note does **not** close that ticket. It records that **TensorRT Edge-LLM is a real NVIDIA product** (v0.10.0) whose **INT4 AWQ VLM path is the official Orin precision**, and that AutoAWQ-on-Jetson is the wrong objection — quantization belongs on the **workstation**. llama.cpp remains the on-device path that does not wait on a PC GPU or a multi-hour SM87 engine build.

Follow-up ticket: workstation export of official AWQ ONNX, then a **later** Orin `llm_build` with nothing else large resident.
