# Stage G — TensorRT Edge-LLM support-matrix evaluation

Tickets: [#34 — build the C++ runtime on this board](https://github.com/AbuAyah110/jetbot-orin-super/issues/34)
(device side, and the CuTe DSL blocker) and
[#32 — workstation INT4 AWQ export](https://github.com/AbuAyah110/jetbot-orin-super/issues/32)
(the ONNX producer).
Evaluated 2026-08-26 against `NVIDIA/TensorRT-Edge-LLM` at tag **`v0.10.0`**
(and `main` @ `bb29145`, "TensorRT Edge-LLM 0.10.0 Documentation Update").

> **Independently corroborated.** [#32](https://github.com/AbuAyah110/jetbot-orin-super/issues/32)
> was filed from a separate investigation (branch `stage-g-edgellm-workstation`,
> `docs/bringup/07-edgellm-workstation-quant.md`) and independently arrived at the
> same pins: v0.10.0, INT4 AWQ with an FP16 vision tower, the official
> `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` checkpoint, ONNX rather than engines over the
> wire, and no FP8. The CuTe DSL gap below is the one finding that ticket does not
> cover.

> **This document corrects a factual error in the Stage G1 record.** G1
> concluded that "there is no NVIDIA product called 'TensorRT Edge-LLM'." That
> is false. See [Correcting the G1 record](#correcting-the-g1-record) below.
> G1's *measurements* are unaffected and still stand — see
> [07-tensorrt-g1.md](07-tensorrt-g1.md). The stage verdict and the ticket list
> live in [07-tensorrt.md](07-tensorrt.md); the long tables stay here.

## Verdict

**This board is a supported target.** `NVIDIA/TensorRT-Edge-LLM` v0.10.0 lists
Jetson Orin on JetPack 6.2+ / CUDA 12.6 as a **`Compatible`** platform, ships a
documented CMake invocation for exactly that configuration, and maps
`EMBEDDED_TARGET=jetson-orin` to **`sm_87`**. Jetson **Orin Nano 8 GB** is one
of the five platforms NVIDIA publishes benchmarks for, and `INT4 AWQ` is
declared valid on **all platforms**. **Qwen2.5-VL-3B-Instruct** is in the
supported-models matrix, has a dedicated model package and C++ ViT runner, and
appears in NVIDIA's own Orin L1 test list at `fp16`, `int4_awq`, and `int8_sq`.

So the original `JETBOT_SPEC.md` runtime choice — TensorRT Edge-LLM running
Qwen2.5-VL-3B INT4 AWQ — is **coherent and on-matrix**, not fictitious.

**It is not, however, installable end-to-end today.** Two concrete gates block
it, neither of which is "the platform is unsupported":

1. **No `sm_87` + CUDA-12 CuTe DSL artifact is shipped.** The source tree
   carries only CUDA-13 prebuilts (JetPack 7). CMake will hard-fail at configure
   time on JetPack 6. See [The CuTe DSL CUDA-12 gap](#the-cute-dsl-cuda-12-gap).
2. **The ONNX export step requires PyTorch** (`torch==2.13.0`), which is
   [#30](https://github.com/AbuAyah110/jetbot-orin-super/issues/30)'s lane and
   is not installed. See [Build and install requirements](#build-and-install-requirements).

Neither gate is a dead end, but both are real work. `llama.cpp + GGUF` should
stay the **primary** VLM path for Stage G2 and Edge-LLM should be tracked as the
**higher-performance successor**, on the strength of the numbers in
[Orin Nano 8 GB benchmarks](#orin-nano-8-gb-benchmarks-nvidias-own).

## Correcting the G1 record

G1 searched the board for `*tensorrt_llm*`, `*tensorrt-llm*`, and `*edge*llm*`,
found nothing, and correctly reported that **TensorRT-LLM is not installed**.
It then generalised that to "the product named in the spec does not exist."
That inference does not follow:

**TensorRT-Edge-LLM and TensorRT-LLM are different projects.** The absence of a
`tensorrt_llm` package says nothing at all about Edge-LLM. Nor does the absence
of an `*edge*llm*` path on disk — Edge-LLM has no apt package and no wheel on
any index; it is a **local source build**, so "not on disk" is its expected
state before installation.

| | TensorRT-LLM | TensorRT Edge-LLM |
| --- | --- | --- |
| Repo | `NVIDIA/TensorRT-LLM` | `NVIDIA/TensorRT-Edge-LLM` |
| Target | datacentre / SBSA Grace | **Jetson**, DRIVE, DGX Spark |
| Jetson Orin JetPack 6 | `v0.12.0-jetson` branch, AGX Orin 64 GB | **`Compatible`** in the official matrix |
| Distribution | wheels (no Tegra build) | local CMake source build |
| Python package | `tensorrt_llm` | `tensorrt_edgellm` |

Everything else G1 measured stands: TensorRT 10.3.0.30 healthy, ~1.5 GB builder
peak for a 68 KB graph, no PyTorch anywhere, the `.venv` needing
`PYTHONPATH=/usr/lib/python3.10/dist-packages`, and CUDA being unusable inside
the agent sandbox.

## Project provenance

Established with `gh api` against the live repo, not from rendered HTML.

| Field | Value |
| --- | --- |
| Repository | `NVIDIA/TensorRT-Edge-LLM` |
| Description | "High-performance, light-weight C++ LLM and VLM Inference Software for Physical AI" |
| Created | 2025-10-02 |
| Last push | 2026-08-18 |
| Stars | 522 |
| Licence | Apache-2.0 |
| Primary language | Python (with a C++ runtime) |
| Latest release | **v0.10.0**, published 2026-08-12 |
| Release history | v0.4.0 … v0.10.0, ten releases since 2026-01-05 |
| Release assets | **none** — every release is source-only |

## The Official Support Matrix

`docs/source/user_guide/getting_started/support-matrix.md` @ `v0.10.0`,
"Platforms" table. The two Orin rows, verbatim:

| Platform | Level | OS / SDK | CUDA Toolkit | TensorRT | Build location | Precision constraint |
| --- | --- | --- | --- | --- | --- | --- |
| Jetson Orin | Official | JetPack 7.2 | 13.2 | JetPack package | Device | FP16, INT8, and INT4 only |
| Jetson Orin | **Compatible** | **JetPack 6.2+** | **12.6** | JetPack package | Device | FP16, INT8, and INT4 only |

The page defines the levels: "`Official` combinations are release-tested
deployment targets. `Compatible` combinations are **expected to work with the
stated constraints**." It also states plainly: "Jetson Orin does not run FP8 or
FP4 model engines."

### Mapping the matrix onto this board

| Requirement | Matrix row | This board | Match |
| --- | --- | --- | --- |
| Platform | Jetson Orin | Orin Nano 8 GB (Super), SM 87 | yes |
| OS / SDK | JetPack 6.2+ | L4T **R36.4.4** = JetPack **6.2.1** | yes |
| CUDA Toolkit | 12.6 | **12.6.11** | yes |
| TensorRT | "JetPack package" | **10.3.0.30-1+cuda12.5** | yes, by implication |
| Arch | aarch64 Tegra | aarch64, `jetson-orin` target | yes |
| CuTe DSL artifact | `sm_87` | `sm_87` inferred from the target | yes |
| Precision | FP16 / INT8 / INT4 only | INT4 AWQ requested | yes |

**The JetPack 7 risk did not materialise.** The README's featured links do point
at JetPack 7.1 and Jetson T4000, and the 0.10.0 news is all Thor-class models,
but the JetPack 6.2+ Orin row is present in the **current** v0.10.0 matrix, and
`installation.md` still ships a **"JetPack 6.2+ Orin"** CMake block. JetPack 6
has not been dropped, and **no upgrade to JetPack 7 is required.**

**So the newest release supporting JetPack 6 is the newest release, `v0.10.0`.**
There is no need to pin an older tag. `support-matrix.md` does not exist at
`v0.4.0`–`v0.9.1` — the consolidated matrix page is new in `v0.10.0` — so
`v0.10.0` is also the only tag where the JetPack 6 guarantee is stated in this
form.

### What I could not verify

- **No document names TensorRT 10.3 explicitly.** The matrix says the TensorRT
  version comes from the "JetPack package", and JetPack 6.2.x ships 10.3.0.30,
  so 10.3 is the implied version for that row. But the only TensorRT versions
  named anywhere in the docs are 10.13.3.9 (JetPack 7.1) and 10.15
  (`limitations.md` @ 0.6.0). Treat "TensorRT 10.3 works" as *implied by the
  matrix*, not as an explicitly tested claim.
- The C++ source does at least **contemplate** pre-10.7 TensorRT.
  `cpp/common/trtUtils.cpp:469` gates `deserializeCudaEngine` on
  `NV_TENSORRT_MINOR >= 7` and provides an `#else` mmap fallback; `kFP4` and
  `kE8M0` enum cases are gated at `>= 10.8` / `>= 10.12`. On TensorRT 10.3 all
  of those take the older branch, and those branches exist. That is
  encouraging, not proof.
- **No Orin Nano row is benchmarked under JetPack 6.** Every published
  benchmark section states **JetPack 7.2**. The Orin Nano throughput and memory
  numbers below are therefore JetPack 7.2 figures; JetPack 6.2 performance on
  the same board is unmeasured.

## Orin Nano 8 GB specifically

This is not merely "the Orin architecture is supported."
`docs/source/user_guide/performance/performance-benchmarks.md` names
**"Jetson Orin Nano 8GB"** as one of five benchmarked platforms (line 3, and
again in the v0.9.0 and v0.8.0 section headers), and gives it its own results
table and its own build-parameter row:

| Engine | `visual_build` parameters |
| --- | --- |
| VLM visual engine (general) | `--minImageTokens 8` `--maxImageTokens 16384` `--maxImageTokensPerImage 2048` |
| **Orin NX / Orin Nano** VLM visual engine | `--minImageTokens 8` **`--maxImageTokens 2048`** `--maxImageTokensPerImage 2048` |

Two more Orin-Nano-relevant instructions:

- "Jetson Orin NX and Orin Nano rows are generally batch `1`."
- "For INT4 runs on Orin, follow the export docs but use externalized INT4
  weights: **`--externalize-weights int4_ffn`** for dense checkpoints."

**No minimum-memory figure is documented anywhere.** Instead of a floor, the
docs scope by device: "Jetson AGX Orin, Orin NX, and Orin Nano run the
externalized INT4 entries **supported by each memory target**." The 8 GB budget
is expressed through the reduced `--maxImageTokens` and batch-1 guidance above,
not as a stated minimum.

### Orin Nano 8 GB benchmarks (NVIDIA's own)

From the **v0.9.0** results section, `#### Jetson Orin Nano (8GB)`. JetPack 7.2,
batch 1. "GPU Mem" is peak GPU memory during inference, in MB.

| Model | Kind | Mode | Precision | Prefill (tok/s) | ViT (tok/s) | Decode (tok/s) | GPU Mem (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3-0.6B | LLM | Vanilla | INT4 AWQ | 2,133 | – | **72.8** | **1,889** |
| Qwen3-1.7B | LLM | Vanilla | INT4 AWQ | 970.5 | – | 37.4 | 3,234 |
| Qwen3-1.7B | LLM | EAGLE3 | INT4 AWQ | 967.8 | – | 41.2 | 3,328 |
| **Qwen3-VL-2B-Instruct** | **VLM** | Vanilla | **INT4 AWQ / FP16** | 1,532.2 | 1,756.4 | **36.9** | **4,380** |
| Qwen3.5-0.8B | VLM | Vanilla | INT4 AWQ / FP16 | 1,176.8 | 5,400.2 | 55.3 | 2,603 |
| Qwen3.5-0.8B | VLM | MTP | INT4 AWQ / INT4 AWQ / FP16 | 1,177.4 | 5,404.1 | 45.0 | 3,202 |
| Qwen3.5-2B | VLM | Vanilla | INT4 AWQ / FP16 | 822.6 | 1,853.8 | 29.9 | 4,621 |
| Qwen3.5-2B | LLM | Vanilla | INT4 AWQ | 621.0 | – | 30.0 | 4,176 |

**The Qwen3-VL-2B row is the one to reason from.** A ~2B VLM at INT4 AWQ with an
FP16 vision tower runs at **36.9 tok/s in 4,380 MB** on this exact board class.
Against G1's measured ~4.8–5.3 GB `MemAvailable`, that fits. Qwen2.5-VL-3B is
larger, so budget roughly **5.0–5.5 GB** and expect **~25–35 tok/s** — but note
that is an extrapolation across both model family and JetPack version, not a
published number. No Orin Nano row for Qwen2.5-VL-3B exists.

For comparison, this is materially better than the llama.cpp GGUF envelope G1
projected (~2.0 GB Q4 backbone + a mandatory ~1.25 GB **F16** `mmproj`), because
Edge-LLM's INT4 AWQ / FP16 split covers weights *and* vision tower inside the
single peak figure above.

## Qwen2.5-VL-3B support

**Yes, at v0.10.0, and it is not a legacy leftover.**

`docs/source/user_guide/getting_started/supported-models.md` @ `v0.10.0`, under
"Qwen vision-language families" → "**Qwen2.5-VL:**" (lines 92–96):

- `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen2.5-VL-7B-Instruct`
- **`Qwen/Qwen2.5-VL-3B-Instruct-AWQ`**, `Qwen/Qwen2.5-VL-7B-Instruct-AWQ`
- `nvidia/Qwen2.5-VL-7B-Instruct-FP8`, `nvidia/Qwen2.5-VL-7B-Instruct-NVFP4`

Corroborated by three independent signals in the same tree:

| Signal | Path |
| --- | --- |
| Dedicated Python model package | `tensorrt_edgellm/models/qwen2_5_vl/` (`modeling_qwen2_5_vl_text.py`, `..._visual.py`) |
| Dedicated C++ ViT runner | `cpp/multimodal/qwen25vlViTRunner.cpp` / `.h` |
| **Orin** L1 test coverage | `tests/test_lists/l1_pipeline_orin_vlm.yml` |

That test list is titled "L1 Pipeline Tests - Jetson Orin (Ampere device)" with
"Focus: fp16/int4_awq VLM and vocab reduction on Orin", and it exercises
Qwen2.5-VL-3B-Instruct in **three precisions**:

| Test group in `l1_pipeline_orin_vlm.yml` | Precision |
| --- | --- |
| `Qwen2.5-VL-3B-Instruct-fp16-mxsl8192-mxbs2-...` | `fp16` |
| `Qwen2.5-VL-3B-Instruct-int4_awq-mxsl8192-mxbs2-...` | **`int4_awq`** |
| `Qwen2.5-VL-3B-Instruct-int8_sq-mxsl8192-mxbs2-...` | `int8_sq` |

Each group runs `test_engine_build` plus `test_e2e_bench` on `coco` (bs1/bs2),
`mmmu`, and `mmmu_pro_4`. So Qwen2.5-VL-3B **INT4 AWQ engine build and
end-to-end VLM benchmarking on Orin is part of NVIDIA's own release gating for
0.10.0.**

Caveat on which Orin: the same file also gates Qwen2.5-VL-7B, Qwen3-VL-8B,
Cosmos-Reason2-8B, and Qwen3.5-9B at `int4_awq`, which do not fit in 8 GB. That
CI Orin is almost certainly an AGX Orin 64 GB. The list proves **Orin/SM 87
correctness coverage**, not that every row fits an 8 GB Nano.

## INT4 AWQ — real, and the right choice here

The spec asked for INT4 AWQ specifically. That was correct, and it is the
*recommended* precision for this board rather than merely a tolerated one.

`performance-benchmarks.md` "Precision Key" is explicit about platform reach:

| Precision | Platform Requirement | Available on SM 87? |
| --- | --- | --- |
| FP16 | All platforms | yes |
| **INT4 AWQ** | **All platforms** | **yes** |
| INT4 GPTQ | All platforms | yes |
| FP8 | SM89+ (Ada and newer) | **no** |
| NVFP4 | SM100+ (Blackwell and newer) | **no** |

This is the answer to the NVFP4 question. NVFP4 dominates the 0.10.0 news and
every Thor / DGX Spark benchmark, **but it is unavailable on Orin.** The support
matrix says the same thing twice ("FP16, INT8, and INT4 only"; "Jetson Orin does
not run FP8 or FP4 model engines") and so does `installation.md`: "Jetson Orin
does not support FP8, MXFP8, FP4, or NVFP4 runtime precision in this release.
Use FP16, INT8, or INT4 checkpoints for Orin."

**Recommended precision for a 3B VLM on Orin: `int4_awq` backbone + `int4_awq`
LM head + FP16 vision tower.** That is the shape of every Orin Nano VLM
benchmark row and of the Orin L1 test list. The FP8 visual-tower option is not
usable here (SM 89+).

`docs/source/user_guide/features/quantization.md`, "Supported Methods":

| Component | Methods | Usable on SM 87 |
| --- | --- | --- |
| Backbone | `fp8`, **`int4_awq`**, `nvfp4`, `mxfp8`, `int8_sq` | `int4_awq`, `int8_sq` |
| LM head | `fp8`, **`int4_awq`**, `nvfp4`, `mxfp8` | `int4_awq` |
| KV cache | `fp8` | no — use FP16 KV |
| Visual tower | `fp8` | no — use FP16 ViT |

### Quantization can be skipped entirely

This matters a lot given the constraints on this board, because quantization is
the one step that genuinely needs a big GPU.

- `installation.md`: "**Quantization requires an NVIDIA GPU**" with "Compute
  Capability 8.0+", "GPU memory at least equal to the FP16 checkpoint size", and
  it is documented as an **x86-64 host** step.
- But `quantization.md` opens with: "**Skip this step when you already have a
  supported pre-quantized HuggingFace checkpoint.**"
- And `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` **is** such a checkpoint — it is listed
  in the supported-models matrix.

**So AWQ checkpoints are directly consumable and no calibration run is needed.**
Pull the pre-quantized AWQ checkpoint, export it, and build. That removes the
GPU-host quantization requirement completely.

Note that this also refutes the G1/TASKBOARD claim that "INT4 AWQ will not
load." That claim was about **AutoAWQ's** GEMM/Triton kernels being unavailable
on Jetson aarch64, which is true but irrelevant here: Edge-LLM does not use
AutoAWQ. It reads the AWQ checkpoint in its own exporter and emits an
`Int4GroupwiseGemmPlugin` TensorRT engine.

### Two INT4 caveats to carry into the build

- `limitations.md` @ **0.10.0**: "`Int4GroupwiseGemmPluginV2` may cause accuracy
  degradation. For ONNX export, use **`--int4-gemm-plugin-version 1`** to select
  the V1 plugin as a fallback." Plan on validating output quality, and keep that
  flag in reach.
- `performance-benchmarks.md`: INT4 on Orin wants
  **`--externalize-weights int4_ffn`**.

## Build and install requirements

The project splits cleanly in two, and the split is what makes this awkward on a
single-Jetson setup.

### Part 1 — export (and optional quantization)

`installation.md` Part 1 is explicitly headed "**(x86 Host)**":

| Requirement | Value | This board |
| --- | --- | --- |
| Platform | x86-64 Linux | **aarch64 Tegra** — off-matrix for this step |
| Python | 3.10+ | 3.10.12, fine |
| `torch` | **`==2.13.0`** (hard pin, `pyproject.toml`) | **absent** — [#30](https://github.com/AbuAyah110/jetbot-orin-super/issues/30) |
| `transformers` | `==5.14.1` | absent |
| `onnx` / `onnxscript` | `1.19.0` / `0.7.1` | absent |
| `numpy` | **`==2.2.6`** | **`.venv` already has 2.2.6 — no conflict** |
| GPU | **not required** — "Export runs on CPU" | n/a |
| Memory | "at least 1.5 times the checkpoint size in **CPU memory**" | ~3 GB for a 3B AWQ checkpoint — fits |
| Install form | `pip install -e .` from source | no wheel, no apt package, no container |

Two useful consequences:

- **Export needs no GPU and no quantization toolkit** — just CPU torch plus
  `transformers`/`onnx`. The `[tools]` extra (which pulls `nvidia-modelopt`,
  `datasets`, `torchvision`) is only needed to *create* a quantized checkpoint,
  which the pre-quantized AWQ checkpoint lets us skip.
- **Edge-LLM's `numpy` pin is exactly the `.venv`'s 2.2.6.** That is a genuine
  piece of luck: the Stage F voice gates' pin is not threatened by this
  dependency set. The `torch==2.13.0` pin is the thing to reconcile with #30.

The docs allow splitting the machines: "If export and inference use different
machines, copy the complete output directory to the target." So exporting on an
x86 host and `rsync`-ing the ONNX to the JetBot is the documented, lowest-risk
route. There is also an **experimental direct engine builder** that skips ONNX
(`tensorrt-edgellm-build`), but it is Python and still checkpoint-driven, so it
does not remove the torch dependency.

### Part 2 — the C++ runtime (on device)

| Requirement | Value | This board |
| --- | --- | --- |
| Toolchain | `cmake`, `build-essential`, `git` | present |
| CUDA / TensorRT | "from the target JetPack" | 12.6.11 / 10.3.0.30 |
| Disk | "~20-50 GB for ONNX files and TensorRT engines" | check `/ssd` before starting |
| Submodules | `googletest`, `nlohmannJson`, `NVTX` | small, shallow-clonable |
| Stated build time | "~1-2 minutes depending on hardware" | optimistic for Orin Nano; assume much longer |

The documented invocation for this board, verbatim from `installation.md`
("**JetPack 6.2+ Orin**"):

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR=/usr \
    -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
    -DEMBEDDED_TARGET=jetson-orin \
    -DCUDA_CTK_VERSION=12.6 \
    -DENABLE_CUTE_DSL=ALL
```

Corroborated in the CMake itself, so this is not a stale doc:

- `cmake/aarch64_linux_toolchain.cmake:71-74` — `jetson-orin` defaults
  `CUDA_CTK_VERSION` to **12.6** and *rejects* `>= 13.0`. The JetPack 6 path is
  the target's default, not an afterthought.
- `cmake/CuteDsl.cmake:128` — `jetson_orin` infers artifact tag **`sm_87`**.
- `cmake/XQACubins.cmake:26` — **87** is in the supported XQA SM list.
- `cmake/CuteDsl.cmake` shim comment — a `cudaLibrary*` → `cu*` shim exists
  specifically for "some **12.0–12.6** embedded" runtimes. Someone built this
  for CUDA 12.6 Tegra on purpose.

### The CuTe DSL CUDA-12 gap

**This is the blocker that stops a build today, and it is JetPack-6-specific.**

`cmake/CuteDsl.cmake` resolves a prebuilt kernel archive named
`cutedsl_{arch}_{tag}_cuda{MAJOR}.tar.gz` from `kernelSrcs/cuteDSLPrebuilt/`.
With `CUDA_CTK_VERSION=12.6` and `EMBEDDED_TARGET=jetson-orin` it looks for
`cutedsl_aarch64_sm_87_cuda12.tar.gz`. What the tree actually ships:

| Tarball in `kernelSrcs/cuteDSLPrebuilt/` | Size | CUDA | Use |
| --- | --- | --- | --- |
| `cutedsl_aarch64_sm_110_cuda13.tar.gz` | 5.0 MB | 13 | Thor, JetPack 7 |
| `cutedsl_aarch64_sm_121_cuda13.tar.gz` | 1.9 MB | 13 | DGX Spark GB10 |
| `cutedsl_aarch64_sm_87_cuda13.tar.gz` | 0.8 MB | **13** | Orin on **JetPack 7.2** |
| `cutedsl_aarch64_sm_87_cuda12.tar.gz` | **missing** | 12 | **Orin on JetPack 6.2 — what we need** |

`kernelSrcs/README.md` lists `cutedsl_aarch64_sm_87_cuda12.tar.gz` in the Docker
builder's default matrix, so the artifact is a supported configuration — it is
just **not committed**, and **GitHub releases carry no assets**, so there is
nowhere to download it from. CMake will emit
`CuTe DSL: no prebuilt artifact found for aarch64/sm_87/cuda12` and then
`FATAL_ERROR: Prebuilt CuTe DSL library not found`.

`-DENABLE_CUTE_DSL=OFF` is **not** a clean escape.
`cpp/kernels/contextAttentionKernels/cuteDslFMHAV2Runner.cpp` — the Ampere FMHA
path, 792 lines — has **no preprocessor guard at all**, so it compiles
unconditionally and needs the artifact's headers and symbols. Only the Blackwell
overlay (`cuteDslFMHARunner.cpp`) is guarded, by
`CUTE_DSL_FMHA_BLACKWELL_ENABLED`. `kernelSrcs/README.md` confirms the intent:
"Every selection implicitly includes `fmha`, because the attention runner is
compiled unconditionally."

Two ways out, both real work:

1. **Generate the artifact locally.** `python kernelSrcs/build_cutedsl.py
   --gpu_arch sm_87 --arch aarch64 --cuda-version 12`, needing
   `nvidia-cutlass-dsl[cu12]==4.6.1` (confirmed a `py3-none-any` wheel, so
   aarch64-installable), `cupy-cuda12x==12.3.0`, and `cuda-python`. It requires
   a **visible GPU**, so it must run unsandboxed, and the README warns to use
   `-j 1` "if GPU memory is limited" — which is the case here. Must go in an
   isolated venv, never the shared `.venv`.
2. **Ask upstream to publish it**, or build it in the project's own
   `kernelSrcs/Dockerfile.cutedsl` container, whose default matrix already
   includes the `aarch64 / sm_87 / cuda12` target.

## KV cache reuse

Relevant because it is the cheapest throughput win for a robot that re-sends a
similar prompt each tick. From `support-matrix.md`, "KV Cache Reuse Support":

| Scenario | Generalized reuse | Requirement |
| --- | --- | --- |
| **Image-input VLM, vanilla decoding** | **Yes** | FP16 KV cache plus any recurrent snapshot pools the model needs |
| Text, attention-only, vanilla decoding | Yes | FP16 KV cache |
| FP8 KV cache | No | reuse requires FP16 KV pages |
| MTP / DFlash / DSpark / Gemma4 MTP | No | run without context reuse |

FP16 KV is the only option on Orin anyway, so **image-input VLM KV reuse is
available to us** — the FP8-KV exclusion costs us nothing.

## Recommendation

1. **Keep `llama.cpp + GGUF` as the primary G2 path.** It has fewer gates: no
   torch, no CuTe DSL artifact, no x86 host. Agent 1's work should continue.
2. **Track TensorRT Edge-LLM as the intended successor, not a rejected idea.**
   NVIDIA's own Orin Nano 8 GB numbers (36.9 tok/s for a 2B VLM at INT4 AWQ in
   4.4 GB) are the best published evidence we have for what this board can do,
   and they are on the spec's original path.
3. **Sequence the export behind [#30](https://github.com/AbuAyah110/jetbot-orin-super/issues/30) (PyTorch),
   or do it off-device via [#32](https://github.com/AbuAyah110/jetbot-orin-super/issues/32).**
   Export needs CPU torch only, so #30 unblocks it without needing a CUDA torch
   build — and the docs explicitly allow exporting on an x86 host and
   `rsync`-ing the ONNX across, which is what #32 plans.
4. **Resolve the CuTe DSL `sm_87`/`cuda12` artifact before budgeting a build**
   ([#34](https://github.com/AbuAyah110/jetbot-orin-super/issues/34)). Until that
   archive exists, CMake cannot even configure. This is a self-contained,
   testable sub-task worth doing on its own, and it is independent of the export
   work.
5. **Use `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`.** Pre-quantized, on the supported
   list, and it removes the GPU-host quantization step entirely.
6. **Do not plan for NVFP4, FP8, or FP8 KV cache on this board.** FP16 vision
   tower, INT4 AWQ backbone and LM head, FP16 KV cache.

## Reproducing this evaluation

```bash
./scripts/bringup/g5_edgellm_probe.sh
```

The probe is read-only: it clones the upstream repo at a pinned tag, verifies the
support-matrix rows, the Qwen2.5-VL entries, the precision key, and the CuTe DSL
prebuilt inventory, then reports whether this board matches the matrix. It
installs nothing and touches no GPU.

Machine-readable results: `data/bringup/g5_edgellm.json`.
