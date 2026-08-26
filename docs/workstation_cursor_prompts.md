# Cursor prompts for the x86 model-export workstation

These prompts are for a separate Cursor chat on an **x86-64 Ubuntu
workstation**, not on the Jetson. Paste one complete fenced block into Cursor
and let that agent inspect the workstation before it installs or downloads
anything.

The TensorRT Edge-LLM commands below come from NVIDIA TensorRT Edge-LLM
**v0.10.0** documentation. In that release, ONNX export runs on CPU;
quantization requires an NVIDIA GPU. TensorRT engines are target-specific and
must be built later on the Jetson Orin (SM87).

## Prerequisites checklist

- [ ] x86-64 Ubuntu 22.04 or 24.04.
- [ ] Python 3.10 or newer.
- [ ] For CPU export of the official AWQ checkpoint: CPU RAM at least 1.5 times
      the checkpoint size; no GPU is required.
- [ ] Only for BF16-to-INT4 re-quantization: NVIDIA Ampere-or-newer GPU
      (compute capability 8.0+) with GPU memory at least as large as the FP16
      checkpoint. The 3B VLM checkpoint is about 7 GiB; 8 GiB is tight and
      12 GiB or more is the practical target.
- [ ] CUDA 12.x or 13.x for quantization, verified with `nvcc --version` and
      `nvidia-smi`.
- [ ] Enough local disk for checkpoints, ONNX external data, calibration
      artifacts, and a retained copy for transfer. Reserve at least 30 GiB for
      these jobs; NVIDIA's general ONNX-plus-engine deployment allowance is
      20–50 GiB.
- [ ] Clone `https://github.com/NVIDIA/TensorRT-Edge-LLM.git`, check out the
      exact `v0.10.0` tag, and initialize submodules. Do not use `main`.
- [ ] Use a fresh virtual environment for this checkout. Install `pip install
      -e .` for export; install `pip install -e ".[tools]"` only when
      quantization is actually needed.
- [ ] Plan to copy complete ONNX/artifact directories to the Jetson. Do not
      build or transfer workstation `.engine` files.

## Job 1 — Export official Qwen2.5-VL-3B AWQ to Edge-LLM ONNX

### What success looks like

A complete `Qwen2.5-VL-3B-Instruct-AWQ/onnx/` tree containing at least
`llm/` and `visual/`, plus every tokenizer, processor, config, external-weight,
and ONNX sidecar produced by the exporter. Copy the **whole** `onnx/`
directory to the Jetson. The language backbone is INT4 AWQ and the visual
tower remains FP16.

### Prompt to paste into Cursor

```text
You are operating on my x86-64 Ubuntu workstation, not on my Jetson. Complete
the CPU export of the official supported checkpoint
Qwen/Qwen2.5-VL-3B-Instruct-AWQ with NVIDIA TensorRT Edge-LLM v0.10.0.

Work autonomously, but do not hide failures. First inspect Ubuntu, free disk,
RAM, Python, git, and any existing TensorRT-Edge-LLM checkout. Record commands,
versions, and failures in a short EXPORT_REPORT.md beside the output.

Use this exact supported workflow:

  git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
  cd TensorRT-Edge-LLM
  git checkout v0.10.0
  git submodule update --init --recursive
  python3 -m venv venv
  source venv/bin/activate
  python -m pip install --upgrade pip
  pip install -e .

Verify:

  git describe --tags --exact-match
  tensorrt-edgellm-export --help

Set:

  export WORKSPACE_DIR="$HOME/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-AWQ"
  mkdir -p "$WORKSPACE_DIR"

Then run the v0.10.0 exporter exactly as follows:

  tensorrt-edgellm-export \
    Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
    "$WORKSPACE_DIR/onnx" \
    --externalize-weights int4_ffn \
    --int4-gemm-plugin-version 1

These flags are deliberate. NVIDIA documents --externalize-weights int4_ffn
for dense INT4 runs on Orin. Edge-LLM 0.10.0 documents possible accuracy
degradation with the default INT4 GEMM V2, so
--int4-gemm-plugin-version 1 selects the V1 fallback. Do not remove either flag
without recording why.

Do not run tensorrt-edgellm-quantize for this job: the input is already an
official pre-quantized AWQ checkpoint. Do not select FP8, NVFP4, MXFP8, FP8 KV
cache, or --visual_quantization. Confirm from checkpoint metadata that the
visual modules are excluded from AWQ and remain FP16.

After export:
1. List the complete output tree and file sizes.
2. Confirm that onnx/llm and onnx/visual exist.
3. Run non-mutating ONNX structural checks available in the installed
   environment. Do not simplify or rewrite the graphs.
4. Save SHA-256 checksums for every output file to
   "$WORKSPACE_DIR/onnx/SHA256SUMS".
5. Write "$WORKSPACE_DIR/EXPORT_REPORT.md" with the git tag/commit, Python and
   package versions, exact command, elapsed time, checkpoint quantization
   metadata, output sizes, checks performed, and any warnings.
6. Print an rsync command that copies the complete directory to:
   <user>@<jetson>:~/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-AWQ/onnx/

Do not build a TensorRT .engine on this workstation. The deployment engine must
be built on the Jetson Orin so it targets SM87 and the Jetson TensorRT version.
Do not install or modify anything on the Jetson.
```

## Job 2 — Optional BF16-to-INT4 AWQ quantization and export

Run this only if the official AWQ checkpoint in Job 1 cannot be used or a
ModelOpt-generated unified checkpoint is explicitly required. It needs a CUDA
GPU. It is not the preferred path.

### What success looks like

A unified Hugging Face-style INT4 AWQ checkpoint under
`Qwen2.5-VL-3B-Instruct-ModelOpt-INT4/quantized/`, followed by a complete
`onnx/llm` + `onnx/visual` export tree. Copy the complete `onnx/` tree to the
Jetson; retain the quantized checkpoint and calibration report on the PC.
Vision must remain FP16.

### Prompt to paste into Cursor

```text
You are operating on my x86-64 Ubuntu CUDA workstation, not on my Jetson.
Re-quantize Qwen/Qwen2.5-VL-3B-Instruct from BF16 to Edge-LLM INT4 AWQ only if
the machine passes the prerequisite checks, then export it to ONNX with NVIDIA
TensorRT Edge-LLM v0.10.0.

Before downloading the model, inspect and record:
  uname -m
  lsb_release -a
  python3 --version
  nvcc --version
  nvidia-smi
  df -h "$HOME"

Require x86_64 Linux, Python 3.10+, CUDA 12.x or 13.x, an NVIDIA GPU with
compute capability 8.0+, and enough free VRAM for the roughly 7 GiB BF16
checkpoint plus calibration activations. If VRAM is only 8 GiB, stop and
report that it is borderline instead of forcing an OOM. Do not silently use
CPU quantization.

Use a clean checkout and environment:

  git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
  cd TensorRT-Edge-LLM
  git checkout v0.10.0
  git submodule update --init --recursive
  python3 -m venv venv
  source venv/bin/activate
  python -m pip install --upgrade pip
  pip install -e ".[tools]"

Verify:

  git describe --tags --exact-match
  tensorrt-edgellm-quantize --help
  tensorrt-edgellm-quantize llm --help
  tensorrt-edgellm-export --help

Set:

  export WORKSPACE_DIR="$HOME/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-ModelOpt-INT4"
  mkdir -p "$WORKSPACE_DIR"

Run the documented quantizer:

  tensorrt-edgellm-quantize llm \
    --model_dir Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir "$WORKSPACE_DIR/quantized" \
    --quantization int4_awq

Do not pass --visual_quantization: in v0.10.0 that option is FP8-only, while
the Jetson Orin deployment requires the vision tower to remain FP16. Do not
use FP8, NVFP4, MXFP8, or FP8 KV cache. Keep the default documented text
calibration unless there is a concrete failure; the defaults are
--text_dataset cnn_dailymail and --num_samples 512. If network or dataset
access fails, report the blocker rather than inventing calibration data.

After quantization, inspect config.json, hf_quant_config.json or embedded
quantization_config and safetensors metadata. Prove that the LLM backbone is
INT4 AWQ and visual modules were not quantized. Then export:

  tensorrt-edgellm-export \
    "$WORKSPACE_DIR/quantized" \
    "$WORKSPACE_DIR/onnx" \
    --externalize-weights int4_ffn \
    --int4-gemm-plugin-version 1

NVIDIA documents --externalize-weights int4_ffn for dense INT4 on Orin.
Edge-LLM 0.10.0 documents possible accuracy degradation with default INT4
GEMM V2; version 1 is the documented fallback.

Validate that onnx/llm and onnx/visual exist, inventory all sidecars and
external weights, run available non-mutating ONNX checks, and create
onnx/SHA256SUMS. Write EXPORT_REPORT.md containing hardware and software
versions, exact commands, calibration dataset/sample count, elapsed times,
peak GPU memory if observable, metadata evidence, output sizes, checks, and
warnings.

Print an rsync command for copying the complete onnx/ directory to the Jetson.
Do not build or copy a .engine from this x86 workstation; SM87 engines are
built later on the Jetson.
```

## Job 3 — Export Jina CLIP v2 text and vision ONNX graphs

The repository's Jina-specific audit was not committed when this prompt was
written. Therefore this job must verify the current Hugging Face model card
and upstream implementation before choosing export inputs or a PTQ API. A
single combined graph is not acceptable.

### What success looks like

Two independently runnable graphs, `jina-clip-v2/onnx/text/model.onnx` and
`jina-clip-v2/onnx/vision/model.onnx`, with tokenizer and image-preprocessor
assets, an I/O contract, ONNX checks, and reference embeddings/cosine
similarity. If a supported calibration-based INT8 PTQ route is confirmed,
also produce separate INT8 Q/DQ graphs and a calibration manifest. Copy ONNX,
tokenizer/processor assets, reference vectors, and reports to the Jetson—not
an x86 `.engine`.

### Prompt to paste into Cursor

```text
You are operating on my x86-64 Ubuntu workstation, not on my Jetson. Investigate
and export jinaai/jina-clip-v2 as two ONNX graphs: one text encoder and one
vision encoder. Do not create a monolithic graph and do not build TensorRT
engines on this PC.

The local JetBot repository did not contain a landed Jina export audit, so
start from primary sources. Before installing large packages or downloading
weights:
1. Read the current Hugging Face model card, config, custom-code files, license,
   intended preprocessing, embedding dimensions, pooling, normalization, and
   trust_remote_code requirements for jinaai/jina-clip-v2.
2. Search the model repository and Jina documentation for an official ONNX
   export path or already-published ONNX artifacts.
3. Inspect the installed/exporter CLI --help and source before choosing flags.
   Do not guess model inputs, opset, image size, tokenizer length, output names,
   dynamic axes, or a ModelOpt command.
4. Write FINDINGS.md with source URLs and the exact revision/commit used.

Create a clean Python virtual environment. Pin the model revision and save all
dependency versions. Use the model card's own preprocessing and public
text/image encoding methods to establish eager reference embeddings first.

Export two separate graphs:
  jina-clip-v2/onnx/text/model.onnx
  jina-clip-v2/onnx/vision/model.onnx

The text graph must accept the card-defined token inputs and produce the final
text embedding. The vision graph must accept the card-defined normalized image
tensor and produce the final image embedding. Preserve the model's documented
pooling and L2 normalization semantics. If remote custom code prevents a stable
export, isolate thin nn.Module wrappers around the two encoders; do not trace a
Python convenience method that performs tokenization, PIL transforms, or
NumPy conversion inside the graph.

Use the exporter recommended by the current upstream card/code. Prefer
torch.onnx.export with dynamo=True only if there is no official exporter and
the installed PyTorch supports it. Select the opset based on operator support
and record the reason. Use dynamic batch; make sequence or image dimensions
dynamic only if upstream supports them and ONNX Runtime parity passes.

Validate each FP32/FP16 ONNX graph with onnx.checker and ONNX Runtime against
the eager model on at least:
- three varied text strings,
- three representative RGB images,
- matched and deliberately mismatched text/image pairs.

Report maximum absolute/relative embedding error and cosine-similarity/ranking
parity. Save preprocessed test inputs and eager/ONNX reference embeddings.

Then investigate INT8 PTQ. This must be real calibration-based PTQ, not blind
weight casting. Prefer NVIDIA ModelOpt only if its currently installed
documentation explicitly supports these encoder graphs and you can cite the
API/CLI used. Calibrate text and vision separately with representative data,
keep numerically sensitive normalization/output operations in floating point
when the tool recommends it, and produce:
  jina-clip-v2/onnx-int8/text/model.onnx
  jina-clip-v2/onnx-int8/vision/model.onnx
  jina-clip-v2/calibration/MANIFEST.json

If no documented compatible PTQ route exists, stop after the verified dual
FP graph export and record INT8 as BLOCKED with the exact reason. Do not invent
a ModelOpt CLI, use dynamic quantization as a substitute, or claim TensorRT
compatibility without evidence.

Finish with:
- tokenizer and image processor assets beside the graphs,
- IO_CONTRACT.md with names, dtypes, shapes, dynamic dimensions, preprocessing,
  pooling, normalization, and embedding dimension,
- VALIDATION_REPORT.md with exact commands and parity numbers,
- SHA256SUMS for all transfer artifacts,
- a printed rsync command to copy the complete jina-clip-v2 artifact directory
  to the Jetson.

Do not build or transfer an x86 TensorRT .engine. The Jetson will build its own
SM87 engines later.
```

## Job 4 — Optional SmolVLA export-feasibility inspection only

### What success looks like

A short evidence report describing whether the prefix VLM and repeated
`denoise_step` can be exported as separate ONNX units, their real inputs and
KV-cache contract, and the smallest next experiment. A successful result is
allowed to be “not currently exportable.” No whole-policy fake graph and no
day-long export attempt.

### Prompt to paste into Cursor

```text
You are operating on my x86 workstation. Inspect SmolVLA ONNX/TensorRT export
feasibility only. Time-box this investigation to 90 minutes and do not start a
large model download until source inspection establishes a plausible subgraph.

Use primary upstream LeRobot source and the actual lerobot/smolvla_base config.
Confirm the current model revision and real inference flow before running any
export. The known architecture is iterative: compute a multimodal prefix/KV
cache, then run a denoise expert repeatedly in a 10-step Euler loop. It is not
a single image-plus-token feed-forward graph.

Explicitly reject and do not implement this false path:
- torch.onnx.export of the whole SmolVLAPolicy/sample_actions,
- a dummy 224x224 image with 16 tokens,
- a graph claimed to map directly to wheel PWM.

Verify current source values rather than trusting stale comments. Document the
actual camera count and image preprocessing, tokenizer length and attention
mask, state padding, noisy action/timestep inputs, action chunk shape, number
of denoise steps, use_cache behavior, and past_key_values crop/update behavior.

Assess exactly two possible export boundaries:
1. Prefix graph: image(s), language, and state to reusable prefix
   embeddings/KV-cache.
2. Denoise-step graph: noisy action, timestep, prefix/KV-cache and required
   masks/state to one denoising update.

For each boundary, list tensor names, dtypes, static/dynamic shapes, cache
layout, unsupported Python/custom-object behavior, and ONNX operator risks.
Inspect whether Hugging Face SmolVLM/SigLIP components have an upstream ONNX
export path that can be reused.

Only run a tiny no-weight or config-only proof if it answers a specific
question quickly. Do not spend the session forcing torch.onnx.export through
the whole policy. Do not perform INT8 PTQ; calibration would require images,
tokens, state, noisy actions, and timesteps. FP16 subgraphs come first if
export ever becomes viable.

Write SMOLVLA_EXPORT_FEASIBILITY.md with source links/revisions, findings,
blockers, the proposed two-graph I/O contracts, and one smallest next
experiment with a stop condition. Do not build an x86 .engine and do not claim
Jetson latency or memory numbers without measurement.
```

## Do not

- Do not install or use PyCUDA on the Jetson for these paths; use the Jetson's
  supported TensorRT/CUDA stack and the repository's C++ or ctypes runtime.
- Do not export FP8, NVFP4, MXFP8, FP8 KV cache, or FP8 vision for this Orin.
- Do not copy any workstation-built `.engine` to the Jetson. Copy ONNX and its
  complete sidecar/external-weight tree, then build SM87 engines on-device.
- Do not install PyTorch on the Jetson for the Edge-LLM Qwen export path.
- Do not re-quantize the official Qwen AWQ checkpoint.
- Do not treat a single whole-policy SmolVLA ONNX graph as a valid shortcut.
