# Edge-LLM Qwen2.5-VL-3B INT4 build on Orin Nano

Date: 2026-08-26

## Result

**Pass.** TensorRT Edge-LLM v0.10.0 built SM87 TensorRT engines for the
ModelOpt INT4 AWQ language model and FP16 visual encoder. A C++
`llm_inference` image-and-text request completed successfully and returned:

> This is a red panda.

No motors, camera, audio, or I2C devices were used.

## Platform and input

- Jetson Orin Nano Super, `aarch64`, SM87, MAXN_SUPER
- L4T R36.4.4 / JetPack 6.2-compatible userspace
- CUDA 12.6.68
- TensorRT 10.3.0.30
- TensorRT Edge-LLM tag `v0.10.0` (`71dd1ba`)
- ONNX: `~/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-ModelOpt-INT4/onnx`
- Export contract: INT4 AWQ LLM, FP16 visual, externalized `int4_ffn`,
  INT4 GEMM plugin v1

All 13 entries passed:

```bash
cd ~/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-ModelOpt-INT4/onnx
sha256sum -c SHA256SUMS
```

The ONNX tree was 5.0 GiB. The NVMe initially had 32 GiB free and had 29 GiB
free after the C++ build and LLM engine.

## C++ runtime build

The official JetPack 6.2+ Orin configuration was used. Compilation was capped
at four jobs and lowered to nice level 10:

```bash
export PATH=/usr/local/cuda-12.6/bin:$PATH
cd ~/Documents/_edgellm_ref/repo
git checkout v0.10.0
git submodule update --init --recursive

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=ALL
nice -n 10 cmake --build build -j4
```

There is a v0.10.0 packaging gap on this platform: the tag contains
`cutedsl_aarch64_sm_87_cuda13.tar.gz`, but not the CUDA 12 artifact selected by
the documented JetPack 6.2 command. Using that CUDA 13 artifact allowed the
host build and engine builds, but inference failed when
`fmha_v2_vit_d80` returned `cudaErrorUnknown`.

A separate venv outside the Jetbot `.venv` generated the matching CUDA 12.6
SM87 FMHA artifact, after which the runtime was rebuilt:

```bash
virtualenv ~/tensorrt-edgellm-workspace/cutedsl-venv
~/tensorrt-edgellm-workspace/cutedsl-venv/bin/pip install \
  'nvidia-cutlass-dsl==4.6.1' 'cupy-cuda12x==12.3.0' cuda-python

~/tensorrt-edgellm-workspace/cutedsl-venv/bin/python \
  kernelSrcs/build_cutedsl.py \
  --kernels fmha --gpu_arch sm_87 --arch aarch64 \
  --cuda-version 12.6 --output_dir \
  ~/tensorrt-edgellm-workspace/cutedsl-cuda12 -j2

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=fmha \
  -DCUTE_DSL_ARTIFACT_TAG=sm_87
nice -n 10 cmake --build build -j4
```

The initial full compile took 580 seconds. Rebuilding against the generated
CUDA 12.6 FMHA archive took 547 seconds.

## Engine builds

The plugin path must be absolute unless the command is launched from the
Edge-LLM repository root:

```bash
export EDGELLM_PLUGIN_PATH=$HOME/Documents/_edgellm_ref/repo/build/libNvInfer_edgellm_plugin.so
export WORKSPACE_DIR=$HOME/tensorrt-edgellm-workspace/Qwen2.5-VL-3B-Instruct-ModelOpt-INT4

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
```

| Component | Result | Wall time | Engine size | Process-tree peak RSS | Peak system RAM | Peak swap |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| LLM INT4 | Pass | 166.2 s | 815,300,996 bytes (778 MiB) | 5.450 GiB | 7,430 / 7,620 MB | 4,168 MB |
| Visual FP16 | Pass | 95.1 s | 1,351,559,300 bytes (1.26 GiB) | 4.517 GiB | 7,462 / 7,620 MB | 2,130 MB |

The complete engine bundle is 3.8 GiB because it also contains the tokenizer,
594 MiB embedding sidecar, and 1.2 GiB external INT4 FFN sidecar. Both builds
used swap heavily but completed without an OOM kill.

## Dummy VLM forward

`examples/multimodal/pics/red_panda.jpeg` and the prompt “What animal is this?
Answer in one short sentence.” were run at batch 1 with at most 16 generated
tokens:

```bash
./build/examples/llm/llm_inference \
  --engineDir "$WORKSPACE_DIR/engines/llm" \
  --multimodalEngineDir "$WORKSPACE_DIR/engines" \
  --inputFile "$WORKSPACE_DIR/input.json" \
  --outputFile "$WORKSPACE_DIR/output.json" \
  --maxGenerateLength 16 \
  --dumpProfile
```

Results:

- Output: `This is a red panda.`
- Exact streaming TTFT: **1,500.7 ms**
- Decode: **40.9 tokens/s** from the profiler (7 generated tokens)
- Streaming end-to-end rate: 4.2 tokens/s including image encode and prefill
- Vision encoder: 750.07 ms for 972 image tokens
- LLM prefill: 786.17 ms for 1,004 tokens (1,277.1 tokens/s)
- End-to-end wall time including engine loading: 12.2 s
- Peak tegrastats RAM: 6,959 / 7,620 MB; peak swap: 1,012 MB
- Edge-LLM reported peak unified allocation: 1,416.48 MB

The process-tree RSS sampler reported only 0.900 GiB because CUDA unified
allocations are not fully charged to the process RSS counters it reads.
Tegrastats is authoritative here: RAM rose from 2,802 MB at startup to
6,959 MB, a 4,157 MB (4.06 GiB) inference delta. That is 0.95–1.52 GiB below
the 5.01–5.58 GiB model-residency interpolation, but total board headroom at
peak was only 661 MB because Cursor and system services were also resident.

Logs and generated artifacts are under
`~/tensorrt-edgellm-workspace/logs/` and
`~/tensorrt-edgellm-workspace/cutedsl-cuda12/`.

## Next step

Keep the CUDA 12.6 SM87 FMHA artifact with this v0.10.0 build tree. Before
agent integration, repeat inference after stopping development services and
measure co-residency with the voice stack; the isolated VLM gate passes, but
661 MB total-system headroom is not enough evidence for safe concurrent use.
