# Jina CLIP v2 — Stage I RAG embedder (not SigLIP 2)

Decision for this branch: **`jinaai/jina-clip-v2` is the multimodal RAG default**, replacing
`nvidia/llama-nemotron-embed-vl-1b-v2`. It is **not** a shrink versus **SigLIP 2 Base**.
SigLIP 2 is no longer the architecture default. Weights were **not** downloaded; no TensorRT
engine was built.

Companion audit: [`gemini_architecture_audit.md`](gemini_architecture_audit.md). Budget:
[`memory_budget.md`](memory_budget.md). Canvas:
`~/.cursor/projects/home-impulse110-Documents/canvases/jetbot-memory-budget.canvas.tsx`.

Primary sources (2026-08-26, Hub tree + `config.json` + [arXiv:2412.08802](https://arxiv.org/abs/2412.08802) + model card):

- https://huggingface.co/jinaai/jina-clip-v2
- Hub `model.safetensors` **1,730,688,642 B = 1650 MiB** F16 (implies ~865M params × 2 B)
- Hub `onnx/` listed via the tree API — **not** pulled

---

## Verdict for this Orin

**Sound replacement for Nemotron-embed (~1.7B, ~3.4 GiB FP16 / 1.7 GiB INT8 aspirational).**
About **half the parameters**, official **INT8 ONNX already on the Hub**, Matryoshka 256-d
vectors, native **512×512** images. Dual-encoder CLIP is **not** an Edge-LLM tenant.

**Not a sound shrink versus SigLIP 2 Base.** INT8 weights for 865M params are **~825 MiB**
(0.81 GiB), vs Gemini’s SigLIP 2 Base **~400 MB** weights-only line. Jina is closer to
Gemini’s “Large ~900 MB” bucket.

**Gemini’s 4.35 GB all-resident stack does not become true.** Same accounting errors as the
Zero-Bloat 3.90 GiB table (no discrete VRAM, OS ~2.2 GiB not 0.30, Qwen INT4+FP16 vision
~3.2 GiB weights, fictitious `smolvla-jetbot`, PyCUDA-on-Tegra false, NITROS not installed).
Swapping the embedder from 0.40 → 0.90 does not fix those rows. See the architecture audit;
do not re-litigate them here.

**License:** **CC BY-NC 4.0** on the Hub card. Fine for this research robot; **not** a
drop-in for a commercial product without Jina’s commercial channel (API / cloud / license).

---

## Claim scoreboard

| Gemini claim | Verdict | Evidence | Corrected |
| --- | --- | --- | --- |
| Dual encoder: Jina XLM-RoBERTa **561M** + EVA02-L14 **304M** = **865M** | **true** | Card table, paper Fig. 1 / Table 1, `config.json` dual towers | 865M; Hub F16 file 1650 MiB matches ~865M×2 B |
| INT8 VRAM **~0.90 GB** | **half** | 865×10^6 bytes ≈ **825 MiB (0.81 GiB)** / **0.87 GB** decimal. Hub `onnx/model_int8.onnx` is **874,350,932 B = 834 MiB**. That is **weights/graph**, not unified-memory resident | Resident **~1.1–2.0 GiB** estimated (engines + 512² activations + TRT). **Unmeasured** on this SoC |
| Matryoshka 1024→64; truncate to **256-d** with **&lt;1%** drop; 75% less Chroma storage | **true** (paper) / **true** (store bytes) / **false** (engine size) | Paper §1.3 + Table 5: 1024→256 typically &lt;1 pp on their CLIP/MTEB slices (EN retrieval 49.33→48.67 nDCG@10 is ~1.3% relative). 256/1024 = **4× smaller vectors**. `config.json` `matryoshka_dimensions`: 32, 64, 128, 256, 512, 768, 1024 | Use **dim=256** in Chroma. Does **not** shrink the 834 MiB INT8 ONNX / TRT engines |
| Bypass Edge-LLM; **two ONNX files**; ModelOpt INT8 PTQ; `trtexec` on Orin; VIC **512×512**; PyCUDA | **half** / **false** pieces | Dual encoder **is** outside Edge-LLM (**true**). Native image **512×512** (`vision_config.image_size`, paper). Hub ONNX is **one fused** `model.onnx` + `model.onnx_data` (3.45 GB FP32 payload) plus **quantized siblings** (`model_int8.onnx`, `model_uint8.onnx`, `model_fp16.onnx`, q4, …) — **not** separate text/vision ONNX, **no** `.engine`. ModelOpt PTQ **not** published; Optimum-style INT8 ≠ TRT PTQ. PyCUDA **false** (G1) | ctypes + apt `python3-libnvinfer`; one or two **TRT** engines if we split later; **unmeasured** `trtexec` |
| Qwen 1.80 + smolvla 1.35 + Jina 0.90 + NITROS 0.30 + audio 0 = **4.35 GB** under **5.2 GB** | **false** | Reuses the discredited 3.90 GiB method | Honest all-resident still **~7–11 GiB**; 5.2 GiB is **model** headroom after OS |

### INT8 of EVA-02 vs RoBERTa

**Plausible, unmeasured on SM 87.** XLM-RoBERTa-class INT8 PTQ is routine. EVA-02-L/14 uses
**SwiGLU**, **2D RoPE**, and xFormers attention (`naive_swiglu`, `rope_embeddings` in
`vision_config`). Those layers often need mixed precision or careful calibration. Hub
`model_int8.onnx` is an existence proof of **some** INT8 graph, not a measured Orin engine
and not ModelOpt PTQ. Do not assume TRT INT8 accuracy until a calibration set and an
on-device build exist.

No Hub TensorRT engines. FlashAttention2 / xFormers in the PyTorch card are **training/eager**
hints, not the Jetson path.

---

## vs Nemotron vs SigLIP 2 Base

| Embedder | Params | Native image | Out dim | INT8 weights (N×1 B) | Honest resident (est.) | Notes |
| --- | ---: | --- | ---: | ---: | --- | --- |
| Nemotron-embed-vl-1b-v2 | ~1.7B | SigLIP2 tower | 2048 | ~1.66 GiB aspirational | **1.66–3.32 GiB** | No Tegra quantizer; FP8 Hub export illegal on Orin |
| SigLIP 2 Base (Gemini) | ~0.4B (mixed with So400m) | often 256/384 | 768-class | ~400 MB claimed | **0.5–1.2 GiB** | Previous Gemini default; **smaller** than Jina |
| **jina-clip-v2 (default)** | **865M** | **512×512** | **1024**, MRL to 64 | **825 MiB** | **1.1–2.0 GiB** | Official INT8 ONNX; CC BY-NC |

**RAG default = Jina CLIP v2.** Keep EmbeddingGemma-class **text-only** (~0.2–0.6 GiB) if
Stage I drops image-to-image search. Keep SigLIP 2 Base only if a future gate proves a
**smaller** English-only dual encoder on this board.

---

## How to configure Chroma (256-d)

This only caps **vector-store disk/RAM**, not the encoder.

Issue [#28](https://github.com/AbuAyah110/jetbot-orin-super/issues/28) (persist upsert/query)
should create the collection with **fixed dimensionality** and **cosine** space because Jina’s
ONNX path already emits **L2-normalized** embeddings.

Recommended collection:

```python
collection = client.get_or_create_collection(
    name="jetbot_rag",
    metadata={
        "hnsw:space": "cosine",
        "hnsw:M": 16,                 # default-ish; raise to 32 only if recall is weak
        "hnsw:construction_ef": 100,  # build quality
        "hnsw:search_ef": 64,         # query recall vs latency; 40–100 is the usual band
        "embedding_model": "jinaai/jina-clip-v2",
        "embedding_dim": 256,
        "matryoshka": "truncate_first_256",
    },
)
```

Encode with `truncate_dim=256` (Transformers / SentenceTransformer) **or** slice the first
256 dims of the normalized 1024-d vector and **re-L2-normalize**. Store `float32` 256-d.

**Do not** set Chroma `max_elements` as a substitute for a memory budget. HNSW RAM scales
with **N × M × (pointer overhead)** plus **N × dim × 4** bytes of vectors. Example: 50k
chunks × 256 × 4 B ≈ **49 MiB** of raw vectors vs **196 MiB** at 1024-d. Graph overhead is
similar at both dims; **distance math** is 4× cheaper at 256. The **834 MiB INT8 ONNX /
TRT engines stay fully resident** while RAG is on.

Cap Stage I to tens of thousands of chunks until `tegrastats` is measured. SQLite facts
([#29](https://github.com/AbuAyah110/jetbot-orin-super/issues/29)) stay orthogonal.

---

## What this pass did not do

No multi-GB weight download, no `trtexec`, no ModelOpt, no PyCUDA. G4
([#19](https://github.com/AbuAyah110/jetbot-orin-super/issues/19)) should retarget dummy I/O
to Jina CLIP v2 (ONNX Runtime CPU/GPU or later TRT), not Nemotron.
