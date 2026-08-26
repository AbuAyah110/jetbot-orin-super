# Stage G — model runtimes (isolated)

Do not connect any runtime to motors or the camera loop until each dummy-I/O ticket passes.

This stage was originally written as "build three TensorRT engines." **It has been rescoped to "stand up a runtime that can execute these models at all,"** with TensorRT export as a separate later optimization. The reason is measured, not editorial: base TensorRT is healthy, but **none of the three models named for G2/G3/G4 is published as a TensorRT engine**, and two of them need PyTorch, which is not installed.

Full inventory, smoke-test numbers, and per-model feasibility analysis: **[07-tensorrt-g1.md](07-tensorrt-g1.md)**.

## Baseline

```bash
./scripts/diagnostics.sh              # TensorRT / CUDA sections
./scripts/bringup/g1_tensorrt_smoke.sh   # G1 gate: inventory + real engine build
```

## G1 result — PASS (2026-08-26)

TensorRT **10.3.0.30** on CUDA **12.6.11** / cuDNN **9.3.0**, and the gate went past reporting a version to building and running three engines from a 68 KB ONNX graph, checked against a full NumPy recompute rather than an exit code:

| Path | Build | Engine | Inference (median) | Max rel. error |
| --- | --- | --- | --- | --- |
| `trtexec` FP32 | 1.82 s | 121.3 KB | 0.0415 ms | 1.7e-06 |
| `trtexec` FP16 | 3.92 s | 121.3 KB | 0.0417 ms | 1.7e-06 |
| TensorRT Python API | 3.80 s | 121.3 KB | 0.1141 ms | 3.3e-07 |

Three findings carry forward into every later ticket:

- **The builder peaked at ~1.5 GB RSS for a 68 KB graph.** Engine *building* must be budgeted separately from engine *running*; on 8 GB shared memory, build when nothing else large is resident.
- **`trtexec` and `nvcc` are installed but not on `PATH`** (`/usr/src/tensorrt/bin/trtexec`), and the repo `.venv` was created without system site packages so it cannot see the apt TensorRT bindings. The gate bridges with `PYTHONPATH=/usr/lib/python3.10/dist-packages` rather than recreating the `.venv` underneath in-flight Stage F work.
- **Anything touching the GPU must run unsandboxed.** The Tegra device nodes are not visible in the agent sandbox, so `cuInit` fails there. Inventory, `dpkg`, and `tegrastats` work sandboxed.

**There is no NVIDIA product called "TensorRT Edge-LLM."** The old ticket 1 title said there was. The nearest real thing, TensorRT-LLM, is **absent with no Tegra aarch64 wheel** — the aarch64 wheels on `pypi.nvidia.com` are SBSA Grace-class, and the `v0.12.0-jetson` branch targets JetPack 6.1 on **AGX Orin 64 GB**, not this Orin Nano 8 GB.

## Separate tickets (dummy I/O only)

| # | Ticket | State |
| --- | --- | --- |
| G1 | TensorRT runtime present and building engines | **Done** — see above |
| G1a | **Install PyTorch for JetPack 6 / CUDA 12.6 aarch64** | **Prerequisite, sequenced ahead of G3/G4** |
| G2 | Qwen2.5-VL-3B **via llama.cpp + GGUF**, one dummy vision+text forward | Open |
| G3 | smolvla dummy motor-token I/O (no PWM) | Blocked on G1a |
| G4 | llama-nemotron-embed-vl-1b-v2 dummy vector out | Blocked on G1a |

**G1a — PyTorch.** G3 and G4 are both blocked on it and no stage previously owned it. It cannot come from PyPI; Jetson needs NVIDIA's wheel index or a `jetson-containers` image matched to CUDA 12.6 / L4T R36.

**G2 — llama.cpp + GGUF is the decided path.** A ~2.0 GB Q4 text backbone plus a **mandatory ~1.25 GB F16 `mmproj` vision encoder** (not quantizable without visible degradation), with the vision tower loaded on demand. A **CUDA-enabled llama.cpp for Tegra must be compiled locally** — no llama.cpp binary exists on this host — and vision support for this architecture has needed recent or forked llama.cpp, so pin and verify the build. INT4 AWQ is off the table: AWQ checkpoints need AutoAWQ kernels that are unavailable on Jetson aarch64, and TensorRT's `INT4` flag is ONNX Q/DQ weight-only quantization, not an AWQ loader, and brings no serving layer. Against ~4.8–5.3 GB available the weights fit; **the real risk is co-residency with the Stage F voice stack on 8 GB shared memory.**

**G3 — smolvla** ships PyTorch safetensors (~450M params, ~0.9 GB BF16) via LeRobot's `SmolVLAPolicy`, with no published ONNX or engine. The **`smolvla-jetbot` fine-tune named in the old spec does not exist**; only `lerobot/smolvla_base` does. Run it eagerly for the dummy gate (blocked on G1a / #30) and keep TensorRT export as a later optimization. A Gemini "one ONNX graph, pycuda, 224², `--memPoolSize` caps VRAM" plan was checked and **rejected as written** — I/O, flow-matching loop, and TRT 10 APIs: **[07-smolvla-trt.md](07-smolvla-trt.md)**.

**G4 — llama-nemotron-embed-vl-1b-v2** ships HF safetensors with no published engine, and **the "1b" is misleading: ~1.7B params, ~3.4 GB in FP16.** NVIDIA's optimized path is a NeMo Retriever NIM, which is x86-first and not a drop-in on Tegra. Consider a smaller text-only embedder for Stage I memory.

Pass per ticket: process exits 0 with a logged output shape / token count. Fail: OOM — reduce batch, confirm 32 GB swap, MAXN SUPER.

Agent **I7** (VLM/engine tools) and real smolvla inside **I5** wait on these dummy tickets. The agent stage itself (I1–I8) is [08-agent.md](08-agent.md), **before** memory.
