# Memory budget — Jetson Orin Nano Super, 8 GB unified

Authoritative footprint accounting for the JetBot stack on this board, assembled 2026-08-26 from
the bring-up gates that have actually run. It replaces the working estimate of
"~4.15 GB active footprint / 5.2 GB, ~1.05 GB LPDDR5 free."

Interactive companion (toggle components, see headroom):
`~/.cursor/projects/home-impulse110-Documents/canvases/jetbot-memory-budget.canvas.tsx`.

## Verdict

**The stack as specified does not fit all-resident on 8 GB, and the shortfall is larger than a
first-pass review of ~1.5 GB suggested.**

| | Value |
| --- | --- |
| Claimed footprint for the six named subsystems | 4250 MiB (4.15 GiB) |
| Corrected footprint for the same six subsystems | **7182–10285 MiB (7.01–10.04 GiB)** |
| Understatement | **2.86–5.89 GiB** |
| Real model budget (pool minus measured OS baseline) | 5351–5377 MiB (5.22–5.25 GiB) |
| Overrun | **1.76–4.82 GiB** |

The **ceiling** in the original estimate is sound and deserves credit: 7620 MiB total minus a
measured 2243–2269 MiB OS baseline really does leave about 5.2 GiB for models. What fills that
ceiling is what is wrong. Even the optimistic end of the corrected range — INT8 embedder, low
PyTorch overhead, ROS 2 not installed — overruns the budget by 1.76 GiB, so the verdict does not
depend on resolving the uncertainty in the estimates.

## How to read the numbers

Every figure carries a basis:

- **Measured** — a number this board produced, from a gate whose artifact exists. Trustworthy to a
  few MiB.
- **Derived** — arithmetic on a published, verified fact (parameter count, layer count). No
  measurement, but no guesswork either.
- **Estimated** — an informed guess. Everything over 1.5 GiB in this budget is in this category,
  and that is the honest shape of the uncertainty: **the two largest line items are the two least
  measured.**

Units are **MiB (1024²)** throughout, matching `tegrastats` and `/proc/meminfo`. The original
table's "GB" are read as GiB.

## The pool

| Metric | Value | Source |
| --- | --- | --- |
| `MemTotal` | 7802736 kB (~7.44 GiB) | `/proc/meminfo` |
| `tegrastats` RAM total | 7620 MB | G1 ([07-tensorrt-g1.md](bringup/07-tensorrt-g1.md)) |
| `cudaMemGetInfo` total | 7990 MB | G1 |
| Swap | 32 GiB at `/ssd/32GB.swap`, `vm.swappiness=60`, **0 B used through every gate** | G1, F4, F5, [01-os.md](bringup/01-os.md) |

The 7620 / 7990 spread is carveout. **7620 MiB is used as the pool throughout this document**
because it is the conservative figure and the one `tegrastats` reports during a run.

Orin Nano has **no discrete VRAM**. CPU and GPU share these pages, so "free GPU memory" is not a
separate quantity and a CPU-only model competes with a GPU model for the same bytes. Both voice
models are CPU-only — no JetPack 6 `onnxruntime-gpu` wheel is reachable (F4/F5 open items, G1
§onnxruntime) — and they still consume the same pool.

### OS baseline: 2243–2269 MiB, measured

`tegrastats` reported `RAM 2243/7620MB` during F5 and `RAM 2269/7620MB` during F4
([06-voice.md](bringup/06-voice.md)). This baseline **already includes `nvargus-daemon`**, which is
an always-active systemd service and was measured live at **185852 kB RSS (181 MiB)** on
2026-08-26 with the camera stack up.

G1 recorded a higher `RAM 2824/7620MB` sample, but that was taken during engine-build activity
rather than at idle, so the F4/F5 figures are the ones used here.

`MemAvailable` runs ~1 GB below `MemTotal - tegrastats RAM` because the kernel conservatively
discounts cache it is unsure it can reclaim (G1 §Memory headroom). Both are recorded; neither is
silently preferred.

**Model budget = 7620 − 2269 … 7620 − 2243 = 5351–5377 MiB.**

## Measured line items

These are the only figures in this budget that carry hard evidence.

| Line item | Footprint | Gate | Notes |
| --- | --- | --- | --- |
| ASR — FastConformer CTC, INT8 ONNX via `sherpa-onnx`, **CPU** | **325–466 MiB** peak RSS | **F4**, `data/bringup/f4_fastconformer_asr.json` | 325.8 MiB at 1 thread and 324.5 MiB at 2 threads for a 6.63 s utterance; **466.5 MiB** for 16.72 s. Weights are constant, so growth is activation and feature buffers proportional to utterance length — **capping utterance length caps this line.** `MemAvailable` floor 3.86 GiB, swap untouched. |
| TTS — Matcha-TTS + HiFi-GAN **v2**, **CPU** | **163–175 MiB** peak RSS | **F5**, `data/bringup/f5_matcha_hifigan_tts.json` | 162.9 MiB for 2.81 s of audio, 174.9 MiB for 6.40 s. `MemAvailable` floor 5096 MiB, swap untouched. **HiFi-GAN v1 costs 240.6 MiB and is unusable anyway** at RTF 1.237. |
| TensorRT engine build, **transient** | **1491 MiB** peak RSS | **G1**, `data/bringup/g1_runtime.json` | For a **68 KB** ONNX graph. `libnvinfer_builder_resource.so` alone is 152 MB and the tactic search is not free. `cudaMemGetInfo` free moved 4882 → 4012 MB across the run. Engine builds happen on-device per NVIDIA's support matrix. |
| `nvargus-daemon` resident | **181 MiB** | live `ps` 2026-08-26 | Inside the OS baseline above; **do not charge it twice.** |
| Agent harness process | **32 MiB** | measured for this document | RSS after importing `jetbot_agent`, `numpy`, PyYAML and `sherpa_onnx`, before any model loads. The one omission from the original table that genuinely does not matter. |

The **`trtexec` RSS rows in `g1_runtime.json` (~43 MB) are an artifact** and are deliberately not
used here — that run sampled the wrapper process, not the `trtexec` child. The 1491 MiB figure
comes from the Python API path, which was sampled correctly. See the caveat in
[07-tensorrt-g1.md](bringup/07-tensorrt-g1.md).

Voice total, both models plus the harness: **520–673 MiB.**

## Derived line items

| Line item | Footprint | Basis |
| --- | --- | --- |
| VLA policy — `lerobot/smolvla_base` BF16 weights | **900 MiB** | ~450M params × 2 B, per G1 §G3. |
| Embedder — **retired default** `nvidia/llama-nemotron-embed-vl-1b-v2` | **1700 MiB** INT8 / **3400 MiB** FP16 | **~1.7B params, not 1B** (Llama 3.2 1B + SigLip2 400M), per G1 §G4. Kept as the spec-as-written row. |
| Embedder — **Stage I default** `jinaai/jina-clip-v2` INT8 | **825 MiB weights** (865M × 1 B) / Hub `model_int8.onnx` **834 MiB** / resident **~1126–2048 MiB** | Dual encoder 561M text + 304M EVA02-L14. N×1-byte ≈ 0.87 GB decimal. **Two TRT engines + 512² activations are extra** — Matryoshka 256-d does not shrink this. See [`jina_clip_v2.md`](jina_clip_v2.md). **Unmeasured** on this board. |
| VLM KV cache | **72 MiB** @ 2048 tok · **144 MiB** @ 4096 · **288 MiB** @ 8192 | Qwen2.5-VL-3B: 2 (K,V) × 36 layers × 2 KV heads × 128 head_dim × 2 B = **36 KiB/token**. |

## Estimated line items

| Line item | Footprint | Why it is only an estimate |
| --- | --- | --- |
| VLM text backbone — Qwen2.5-VL-3B Q4_K_M GGUF | **2000–2200 MiB** | G1's feasibility analysis. **No llama.cpp binary exists on this host and no GGUF has been fetched**; nothing has been loaded on-device. |
| VLM vision tower — F16 `mmproj` | **1250 MiB** | Same. Mandatory: the vision tower **cannot be quantized to INT4 without visible degradation**, so this is a fixed cost whenever vision is resident. |
| llama.cpp compute buffers | **100–450 MiB** on top of KV | Depends on batch and image token count, neither of which has been exercised here. |
| PyTorch + CUDA context + cuDNN/cuBLAS kernel libraries | **600–1000 MiB**, charged **once per GPU process** | **PyTorch is not installed at all** (ticket G1a). Anchored on the one hard datapoint available: a TensorRT process needed 1491 MiB to build a 68 KB graph, so GPU framework overhead on this board is measured in hundreds of MiB, not tens. `smolvla` and the embedder can share one process; two processes pay it twice. |
| Chroma HNSW index + SQLite | **200–400 MiB** | Stage I memory stores are not built and no gate has run. A placeholder that must be measured. |
| ROS 2 + Isaac ROS graph | **0 or 300 MiB** | **ROS 2 is not installed on this board.** The 300 MiB is the spec's own guess, carried forward unverified. |

## Verdict on the proposed table, row by row

| Subsystem | Claimed | Corrected | Delta | Assessment |
| --- | --- | --- | --- | --- |
| High-Level VLM — Qwen2.5-VL-3B INT4 AWQ | 1.80 GiB | **3.41–3.95 GiB** | **+1.61 … +2.15** | **Wrong, and the runtime named cannot exist here.** INT4 AWQ will not load: AWQ checkpoints need AutoAWQ GEMM/Triton kernels unavailable on Jetson aarch64, and there is no Tegra TensorRT-LLM wheel (G1 §G2). The decided path is llama.cpp + GGUF, which costs a ~2.0–2.2 GB Q4 backbone **plus a mandatory ~1.25 GB F16 `mmproj`** plus KV and compute buffers. |
| Reactive Motor VLA — `smolvla-jetbot` FP16/INT8 | 1.20 GiB | **1.46–1.86 GiB** | **+0.26 … +0.66** | **Model name is fictitious; the number is close for the wrong reason.** The `smolvla-jetbot` fine-tune **does not exist** — only `lerobot/smolvla_base` does, at ~0.88 GiB BF16, so the weights claim is actually generous. What is missing is the PyTorch and CUDA runtime wrapped around them. |
| Multimodal RAG — `llama-nemotron-embed` INT8 | 0.45 GiB | **1.66–3.32 GiB** | **+1.21 … +2.87** | **The largest single error in the original table.** ~1.7B params, FP16 safetensors, no on-device quantizer. **No longer the RAG default.** |
| Multimodal RAG — **jina-clip-v2 INT8** (new default) | 0.90 GiB (Gemini) | **0.81 GiB weights / 1.1–2.0 GiB resident** | **+0.2 … +1.1** vs 0.90 | Weights-only 0.90 is close; resident is engines + activations. **Larger than SigLIP 2 Base**, smaller than Nemotron. |
| ASR — FastConformer CTC FP16 | 0.25 GiB | **0.32–0.46 GiB** | **+0.07 … +0.21** | **Sound in magnitude — the best row in the table.** Wrong on details: it runs INT8 on **CPU**, not FP16 on GPU. Use 0.32 GiB with a 0.46 GiB cap for long utterances, and cap utterance length to hold the cap. |
| TTS — FastPitch + HiFi-GAN mixed | 0.15 GiB | **0.16–0.17 GiB** | **+0.01 … +0.02** | **The number is right.** The model is not: **FastPitch was a measured dead end** — no ONNX or TensorRT plan exists on NGC and `nemo_toolkit[tts]` resolves to 161 packages / 3.37 GB of the wrong CUDA generation (F5). The shipped path is **Matcha-TTS + HiFi-GAN v2**. |
| ROS 2 Middleware — Isaac ROS Argus zero-copy | 0.30 GiB | **0 … 0.29 GiB** | **−0.30 … −0.01** | **Should not be charged as written.** ROS 2 is not installed, so the figure is speculative. The real camera cost is `nvargus-daemon` at 181 MiB, and that is **already inside the measured OS baseline** — charging 0.30 GiB on top double-counts it. |

## Line items the table omits entirely

Together these exceed the largest row in the original table.

| Missing line item | Footprint | Basis | Why it belongs |
| --- | --- | --- | --- |
| TensorRT engine build, transient | **1.46 GiB** | Measured (G1) | Bigger than every claimed row except the VLM, and completely absent. A real graph will cost more than the 68 KB toy that produced this number. |
| VLM KV cache + llama.cpp compute buffers | 0.24–0.58 GiB | Derived | Weights are not the whole cost of serving a VLM. |
| PyTorch + CUDA context + kernel libraries | 0.59–0.98 GiB | Estimated | Charged once per GPU process; both `smolvla` and the embedder need it. |
| Chroma HNSW index + SQLite | 0.20–0.39 GiB | Estimated | Stage I is in the plan, so its memory belongs in the budget. |
| Agent harness process | 0.03 GiB | Measured | Included for completeness; genuinely cheap. |

## Co-residency: what actually fits

The decision this budget exists to support is not the total but **what can be resident at once.**
The table below holds the embedder at INT8 and the VLM context at 4096 tokens — the most
favourable assumptions available — and includes the always-charged OS baseline. "Fits" means the
**pessimistic** bound still leaves 400 MiB of slack for page cache and NVMM contiguity.

| Co-residency set | Total (MiB) | Headroom vs 7620 (MiB) | Verdict |
| --- | --- | --- | --- |
| Voice only (ASR + TTS + harness) | 2763–2942 | +4678 … +4857 | **Fits** |
| Voice + VLM, vision resident | 6257–6986 | +634 … +1363 | **Fits** |
| Voice + VLM, vision demand-loaded | 5007–5736 | +1884 … +2613 | **Fits** |
| Voice + `smolvla` | 4263–4842 | +2778 … +3357 | **Fits** |
| Voice + engine build | 4254–4433 | +3187 … +3366 | **Fits** |
| Voice + `smolvla` + engine build | 5754–6333 | +1287 … +1866 | **Fits** |
| Voice + `smolvla` + embedder INT8 + stores (no VLM) | 6163–6942 | +678 … +1457 | **Fits** |
| Voice + VLM text-only + engine build | 6498–7227 | +393 … +1122 | Marginal |
| Voice + VLM text-only + `smolvla` | 6507–7636 | −16 … +1113 | Optimistic bound only |
| Voice + VLM text-only + embedder INT8 + stores | 7507–8836 | −1216 … +113 | Optimistic bound only |
| Voice + VLM (vision) + `smolvla` | 7757–8886 | −1266 … −137 | **Does not fit** |
| Voice + VLM (vision) + engine build | 7748–8477 | −857 … −128 | **Does not fit** |
| Voice + VLM (vision) + embedder INT8 | 8557–9686 | −2066 … −937 | **Does not fit** |
| Voice + VLM (vision) + embedder FP16 | 10257–11386 | −3766 … −2637 | **Does not fit** |
| **Spec as written, everything resident** | **9957–11286** | **−3666 … −2337** | **Does not fit** |

Three conclusions follow directly:

1. **One large subsystem at a time.** Voice plus any single large model fits comfortably. Voice plus
   the full VLM plus `smolvla` is over the pool even at its optimistic bound. There is **no**
   arrangement in which the VLM, the VLA policy, the embedder and the voice stack are all resident.
2. **The vision tower must be demand-loaded.** 1250 MiB of F16 `mmproj` is 23% of the model budget
   and only earns its place on frames that need visual grounding. Evicting it is what moves
   "reasoning + navigation" from *does not fit* to merely marginal.
3. **Engine builds need a nearly empty board.** Voice may stay up during a build; the VLM with its
   vision tower may not.

## Recommendations

### 1. Stage I RAG default is Jina CLIP v2 (not Nemotron, not SigLIP 2)

Do **not** load `llama-nemotron-embed-vl-1b-v2`. The multimodal default is
**`jinaai/jina-clip-v2`**: 865M params, Hub INT8 ONNX **834 MiB**, honest resident
**~1.1–2.0 GiB**, Matryoshka **256-d** in Chroma ([`jina_clip_v2.md`](jina_clip_v2.md)).
That recovers **~0.6–2.2 GiB vs Nemotron** but **spends more than SigLIP 2 Base**.
A text-only EmbeddingGemma-class encoder (~0.2–0.6 GiB) remains the cheapest Stage I
option if image retrieval can wait. 256-d Chroma **does not** reduce encoder RAM.

`JETBOT_SPEC.md` is **not** edited on this branch (`stage-g-edgellm` has it dirty).

### 2. Treat `mmproj` demand-loading as mandatory, not an optimization

The F16 vision tower cannot be quantized without visible degradation, so its 1250 MiB is fixed and
the only lever is residency: load per visual query, evict after. The Q4 backbone can stay resident,
because llama.cpp mmaps it and the kernel can reclaim those pages without an explicit unload.

### 3. Nothing large may be co-resident with an engine build

Measured 1491 MiB for a trivial graph. Give builds a dedicated window with the voice stack as the
only other tenant (4254–4433 MiB total). **Never** build while the VLM's vision tower, the
embedder, or a PyTorch process is loaded.

### 4. Make the mode switch explicit in the agent design

Reasoning mode (voice + VLM) and navigation mode (voice + `smolvla`) each fit with room to spare;
their union does not. That requires a model-residency manager in the harness with a real unload
path, not an implicit hope. Voice stays resident across both modes — it is the cheapest thing on
the board at 520–673 MiB and it is CPU-only, so it never contends for the GPU.

### 5. Do not spend the 32 GiB swap on model weights

Swap has been touched for **exactly 0 bytes** through F4, F5 and G1 at `vm.swappiness=60`. Keep it
that way. A paged-out policy network means a stalled control loop on a robot with a motor watchdog,
so swap is OOM insurance rather than capacity. `PROJECT_PLAN.md` already says this; the budget now
has the numbers behind it.

### 6. Cap utterance and generation length

Both voice models grow with input/output length and nothing else: ASR 325 → 466 MiB from 6.6 s to
16.7 s, TTS 163 → 175 MiB from 2.8 s to 6.4 s. Capping length is a hard cap on those lines.

## User-supplied quantized checkpoints (2026-08-26)

One Hugging Face VLM checkpoint was offered as a rebuttal to the all-resident
verdict. Hub file sizes for that VLM were taken from the Hub tree API (LFS
`size` fields); the weights were **not** downloaded. It does **not** reverse
the verdict.

**Rejected — do not use to shrink any line.**
`nvidia/llama-nemotron-embed-vl-1b-v2-fp8` is a real NVIDIA ModelOpt/vLLM FP8
export, but it is not a tenant on this board: TensorRT-Edge-LLM's Orin support
matrix is **FP16, INT8, and INT4 only** ("Jetson Orin does not run FP8 or FP4
model engines"), and SM 87 has no native FP8 Tensor Cores. The embedder line
stays the prior **1700 MiB INT8 (aspirational) / 3400 MiB FP16** figures, or a
smaller text-only alternative (EmbeddingGemma-class, ~0.2–0.6 GiB), or the new
default **jina-clip-v2 INT8 (~825 MiB weights, ~1.1–2.0 GiB resident)**. FP8 is not
an intermediate. SigLIP 2 is **not** the RAG default.

### `Azaz666/Qwen2.5-VL-3B-Instruct-AWQ-INT4`

| Field | Value |
| --- | --- |
| Repo | https://huggingface.co/Azaz666/Qwen2.5-VL-3B-Instruct-AWQ-INT4 |
| Pinned revision | `a945acad73cf25d1c84bb4b8a79923aa7ecff876` (tree the user linked) |
| Publisher | Community (`Azaz666`), not Qwen and not NVIDIA |
| License | Apache-2.0 |
| Declared runtime | **AutoAWQ** (`library_name: autoawq`, `quant_method: awq`, `version: gemm`) |
| `model.safetensors` | **3,401,801,720 B = 3244 MiB** |
| Hub dtype mix | 2,774,532,096 I32-packed 4-bit params + **980,090,880 F16 params** |

`config.json` at that SHA: `hidden_size` 2048, `num_hidden_layers` 36,
`num_key_value_heads` 2, vision tower present (`vision_config.depth` 32,
`hidden_size` 1280). Quantization is AWQ INT4, **group size 128**, zero-point
true. **`modules_to_not_convert: ["visual"]` — the vision encoder is left FP16.**
That is why a "3B INT4" file is 3.17 GiB rather than ~1.8 GiB: roughly 1.87 GiB
of the file is still F16.

The card's own Jetson note (README on `main`; not present at the pinned SHA) is
explicit: AutoAWQ CUDA/Triton kernels **are not available on Jetson aarch64**,
and the author points at a different INT8 method for this board. Measured GPU
memory on an A6000 for this checkpoint is **3334 MB** — already most of this
board's 5.2 GiB model budget, on a machine that has discrete VRAM.

**Can it load here?**

| Runtime | Loadable on Orin Nano Super SM 87? |
| --- | --- |
| AutoAWQ / Triton GEMM / transformers-with-AutoAWQ | **No.** Publisher says so; aarch64 Jetson lacks those kernels. |
| llama.cpp | **No, not this file.** This is AutoAWQ safetensors, not GGUF. |
| TensorRT-Edge-LLM | **Not this repo.** Edge-LLM **does** ingest AutoAWQ packing (`quant_method == awq` → `QUANT_INT4_AWQ`, `repack_awq_to_plugin`) and lists **`Qwen/Qwen2.5-VL-3B-Instruct-AWQ`** as a supported pre-quantized checkpoint. It does **not** list `Azaz666/...`. Official Qwen AWQ `model.safetensors` is 3,401,785,760 B (16 KB smaller) — same weight class, still includes an FP16 visual. Orin is **Compatible** on JetPack 6.2+ / CUDA 12.6 for **FP16, INT8, INT4 only**; engines **build on device**. An unlisted community export is not a supported input. Even the official AWQ still has to be exported and built; that path has never been run on this host. |

INT4 groupwise GEMM existing in Edge-LLM ≠ this safetensors repo is a drop-in
engine. Weights-on-disk also ≠ resident footprint: KV cache, TRT runtime, and
activations remain separate line items (still unmeasured for a 3B VLM).

### Recalculated VLM line (weights vs resident)

Previous estimates are **not** replaced. The AWQ Hub size is an alternate
**weight** column only. The embedder is **not** recalculated from the rejected
FP8 repo.

| Line | Original claim | Previous correction | User-supplied VLM (Hub file) | Still unmeasured on top |
| --- | --- | --- | --- | --- |
| VLM | 1.80 GiB INT4 AWQ | 3.41–3.95 GiB Q4 GGUF + F16 mmproj + KV | **3244 MiB** AWQ safetensors (INT4 text + FP16 visual in one file) | KV (72/144/288 MiB) + Edge-LLM/llama.cpp runtime + activations. A6000 card reports 3334 MB for the VLM alone. |
| Embedder | 0.45 GiB INT8 Nemotron | **Nemotron 1.66 / 3.32 GiB** still valid as the old row; **Jina default 825 MiB weights / 1.1–2.0 GiB resident** | Jina Hub INT8 ONNX 834 MiB (tree API; not downloaded) | ORT/TRT resident unmeasured; Chroma 256-d is store-only. |

KV cache formula is unchanged: Qwen2.5-VL-3B still 36 KiB/token.

### Does the INT4 AWQ VLM make all-resident fit?

**No.** It does not reverse the verdict. It only confirms that a real INT4 AWQ
file is ~3.2 GiB because the vision tower is still FP16 — the same order as the
llama.cpp Q4+mmproj estimate, not the claimed 1.80 GiB.

| Set | MiB | vs 5351–5377 MiB model budget |
| --- | --- | --- |
| AWQ VLM weights only | 3244 | Fits, leaves ~2.1 GiB |
| AWQ VLM + KV@4096 + runtime buffers | 3488–3838 | Fits as a single tenant |
| Voice + AWQ VLM + KV/runtime | 4008–4511 | Fits |
| Voice + AWQ VLM + `smolvla` 900 + one GPU runtime 600–1000 + KV@4096 | **5508–6411** | **Does not fit** (over even at the low bound) |
| Voice + AWQ VLM + embedder **FP16** 3400 + runtime + KV@4096 | **8008–8911** | **Does not fit** |
| Spec-shaped: voice + AWQ VLM + smolvla + embedder FP16 + stores + KV | **9108–10311** | **Does not fit** (same conclusion as the previous 9957–11286 table) |

What the AWQ checkpoint **does** establish:

1. The original 1.80 GiB VLM claim ignored an FP16 vision tower that both this
   AutoAWQ export and official `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` leave
   unquantized.
2. Relative to the **previous correction**, 3244 MiB of Hub weights is not a
   savings versus Q4+mmproj (~3.2 GiB weights either way).
3. AutoAWQ kernels will not run on this SoC; Edge-LLM INT4 is a different
   pipeline (official Qwen AWQ, export, on-device engine build) and has never
   been measured here.

Until G2/G4 actually load a legal checkpoint on this board, runtime overhead
stays estimated.

## What is still unmeasured

**The two largest line items in this budget have never been loaded on this board.** Closing that is
the only way to tighten the total.

| Unmeasured | Estimate | Blocked by | What would close it |
| --- | --- | --- | --- |
| Qwen2.5-VL-3B via llama.cpp | 3.41–3.95 GiB | **G2** — no llama.cpp binary on this host, no GGUF fetched, CUDA build for Tegra must be compiled locally | Peak process RSS and a `tegrastats` trough from a real forward pass, **with and without the vision tower loaded**. Also measure how much of the mmap-backed Q4 backbone is genuinely resident under pressure, which could move the number either way. |
| `llama-nemotron-embed-vl-1b-v2` | 1.66–3.32 GiB | **G4** (retired default) | Do not spend G1a on this checkpoint. |
| `jinaai/jina-clip-v2` INT8 | 1.1–2.0 GiB resident (est.) | **G4 / Stage I** — Hub ONNX listed, **not** fetched, **no** TRT engine | Peak RSS of ORT or `trtexec` at 512×512, batch 1; confirm INT8 accuracy. |
| PyTorch + CUDA context on Tegra | 0.59–0.98 GiB | **G1a** | RSS of a bare `import torch` plus a CUDA context on this board, before any weights. |
| Chroma + SQLite | 0.20–0.39 GiB | Stage I not started | RSS of the store with a representative index loaded. |
| ROS 2 / Isaac ROS graph | 0 or 0.30 GiB | not installed | Only if ROS 2 is actually adopted. |
| Real `trtexec` build RSS | unknown | G1 sampling bug, fixed but not re-run | Re-run `./scripts/bringup/g1_tensorrt_smoke.sh`; the `tracked_pid` field marks output from the fixed version. |

**G2 and G4 must report peak process RSS and a `tegrastats` trough the way F4 and F5 do.** Until
they do, the two biggest numbers in this budget are the two that cannot be defended.

## Reproduce

```bash
# Pool, baseline, swap
grep -E "MemTotal|MemAvailable|SwapTotal|SwapFree" /proc/meminfo
tegrastats --interval 1000 | head -3
ps -o rss=,comm= -C nvargus-daemon

# Source gates (artifacts under data/bringup/ are gitignored; re-run to regenerate)
./scripts/bringup/test_fastconformer_asr.sh     # F4 -> f4_fastconformer_asr.json
./scripts/bringup/test_matcha_hifigan_tts.sh    # F5 -> f5_matcha_hifigan_tts.json
./scripts/bringup/g1_tensorrt_smoke.sh          # G1 -> g1_runtime.json
```

Source documents: [06-voice.md](bringup/06-voice.md) (F1–F5),
[07-tensorrt.md](bringup/07-tensorrt.md) and [07-tensorrt-g1.md](bringup/07-tensorrt-g1.md) (G1),
[01-os.md](bringup/01-os.md) (Stage A, swap), [`jina_clip_v2.md`](jina_clip_v2.md),
[`gemini_architecture_audit.md`](gemini_architecture_audit.md), [`JETBOT_SPEC.md`](../JETBOT_SPEC.md).
