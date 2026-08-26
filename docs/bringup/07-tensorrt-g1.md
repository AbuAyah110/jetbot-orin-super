# Stage G1 — TensorRT runtime inventory and smoke test

Ticket: [G1: TensorRT / Edge-LLM runtime present](https://github.com/AbuAyah110/jetbot-orin-super/issues/16).
Probed on device 2026-08-26.

> **This is the G1 evidence record.** The verdict, the rescoped ticket list, and
> the llama.cpp + GGUF decision now live in [07-tensorrt.md](07-tensorrt.md),
> which is the doc to read first; the measurements, per-model feasibility
> analysis, and reproduction details stay here rather than bloating the stage
> doc. [JETBOT_SPEC.md](../../JETBOT_SPEC.md) §"Measured runtime reality"
> carries the same conclusions.

**Headline: base TensorRT is installed, healthy, and passed a real
end-to-end engine build + inference. TensorRT-LLM is not installed, and there
is no NVIDIA product called "TensorRT Edge-LLM." No PyTorch is installed
anywhere on this board.** G2/G3/G4 as originally written could not be executed
by what is here, so the spec and [07-tensorrt.md](07-tensorrt.md) have since
been reconciled: Stage G is rescoped to standing up runtimes, the VLM path is
llama.cpp + GGUF, and PyTorch became its own prerequisite ticket ahead of
G3/G4.

Reproduce with:

```bash
./scripts/bringup/g1_tensorrt_smoke.sh
```

Machine-readable results: `data/bringup/g1_runtime.json`.
Engines, ONNX graph, and `trtexec` logs: `data/bringup/g1/`.

## Runtime inventory

| Component | Version | Path / notes |
| --- | --- | --- |
| L4T | R36.4.4 (GCID 41062509) | `nvidia-l4t-core 36.4.4-20250616085344` |
| Kernel / arch | 5.15.148-tegra, aarch64 | Ubuntu 22.04.5 LTS |
| Power mode | `MAXN_SUPER` | `nvpmodel -q` |
| CUDA toolkit | 12.6.11 | `/usr/local/cuda` → `cuda-12.6` |
| `nvcc` | 12.6.68 | `/usr/local/cuda/bin/nvcc` — **not on the default `PATH`** |
| CUDA runtime / driver | 12.6 / 12.6 | via `cudaRuntimeGetVersion` / `cudaDriverGetVersion` |
| GPU arch | SM 87 | Orin, `__nvcc_device_query` |
| TensorRT | **10.3.0.30-1+cuda12.5** | apt meta-package `tensorrt` |
| `libnvinfer.so.10` | 10.3.0 (232 MB) | `/usr/lib/aarch64-linux-gnu/` |
| `trtexec` | TensorRT v100300 | `/usr/src/tensorrt/bin/trtexec` — **not on the default `PATH`** |
| cuDNN | 9.3.0 | `libcudnn9-cuda-12 9.3.0.75-1` |
| cuBLAS | 12.6.1.4 | `libcublas-12-6`, under the CUDA target lib dir |
| Python `tensorrt` | 10.3.0 | `/usr/lib/python3.10/dist-packages/tensorrt/` (`python3-libnvinfer`) |

The full apt package list (`libnvinfer*`, `libnvonnxparsers*`,
`python3-libnvinfer*`) is in the JSON. `libnvinfer-samples`,
`libnvinfer-plugin`, the lean and dispatch runtimes, and the ONNX parser are all
installed, so this is a complete TensorRT install rather than a runtime-only one.

### Python bindings: the `.venv` cannot see them

The TensorRT bindings arrive as an apt package into
`/usr/lib/python3.10/dist-packages`. The repo `.venv` was created by
`virtualenv` with `include-system-site-packages = false`, so `import tensorrt`
fails inside it.

| Interpreter | `import tensorrt` |
| --- | --- |
| `/usr/bin/python3` | works — 10.3.0 |
| `.venv/bin/python3` as-is | `ModuleNotFoundError` |
| `.venv/bin/python3` with `PYTHONPATH=/usr/lib/python3.10/dist-packages` | works — 10.3.0 |

`scripts/bringup/g1_tensorrt_smoke.sh` takes the third option. The `.venv`
interpreter is the same CPython 3.10.12 as `/usr/bin/python3`, so the `cp310`
bindings load unchanged, and the Stage F voice gates that depend on this shared
`.venv` are left untouched. Recreating the `.venv` with
`--system-site-packages` would be the tidier long-term fix, but that is a
Stage E change and would disturb in-flight Stage F work.

### What is missing

| Thing | Status |
| --- | --- |
| **TensorRT-LLM** | **absent** — no apt package, no Python module, nothing on disk |
| "TensorRT Edge-LLM" | **does not exist as an NVIDIA product** under that name |
| PyTorch / torchvision | **absent in both interpreters** |
| Python `onnxruntime` | absent in both interpreters |
| `onnx` | absent; G1 installs it into a repo-local dir (see below) |
| `pycuda`, `cuda-python` | absent — G1 talks to `libcudart` through `ctypes` instead |
| `modelopt` / `ammo` (quantization) | absent |
| Polygraphy, `onnx-graphsurgeon` | absent |
| `llama.cpp`, MLC | no binary or module on the host |
| Docker | daemon **active**, but zero images pulled |
| `jetson-containers` | checked out at `/home/impulse110/jetson-containers` (Oct 2025); has `awq`, `mlc`, `llama_cpp`, `tensorrt_optimizer` — **no `tensorrt_llm` package** |

`numpy` differs between interpreters: 1.21.5 system, 2.2.6 in the `.venv`.

### onnxruntime

There is no Python `onnxruntime` package, so there are no execution providers to
report from Python. Stage F4/F5 does not use it: `sherpa-onnx` vendors its own
native `libonnxruntime.so` (34 MB, version string 1.27.1) at
`.venv/lib/python3.10/site-packages/sherpa_onnx/lib/`. Its `ldd` shows **no
CUDA, cuDNN, or TensorRT linkage at all** — only libc, libm, libdl, librt, and
libpthread. The provider name strings visible in the binary are ORT's compiled-in
registry enumeration, not evidence of an available provider.

**The voice stack is running ONNX Runtime on CPU.** ORT's presence in the repo
does not give Stage G a GPU ONNX path, and adding one would mean building ORT
with the CUDA/TensorRT EPs for Tegra or pulling a Jetson-specific wheel — PyPI's
aarch64 `onnxruntime-gpu` wheels are not built for Tegra.

## TensorRT smoke test — PASSED

A 68 KB ONNX graph built with the `onnx` helper API, no downloaded weights:

```text
input [1,3,32,32] -> Conv 3x3 (3->8, pad 1) -> Relu
                  -> Conv 3x3 s2 (8->4, pad 1) -> Relu
                  -> Flatten -> Gemm (1024->16) -> output [1,16]
```

Weights are seeded (`20260826`) and the whole graph is recomputed in NumPy, so
the check is against a real reference rather than just an exit code. Three
independent paths were run:

| Path | Build | Engine | Inference (median) | Max rel. error | Result |
| --- | --- | --- | --- | --- | --- |
| `trtexec` FP32 | 1.82 s | 121.3 KB | 0.0415 ms | 1.7e-06 | pass |
| `trtexec` FP16 (`--fp16`) | 3.92 s | 121.3 KB | 0.0417 ms | 1.7e-06 | pass |
| TensorRT Python API | 3.80 s | 121.3 KB | 0.1141 ms | 3.3e-07 | pass |

- Output shape `[1,16]`, all values finite, none all-zero, matching the NumPy
  reference to ~1e-6 relative. `trtexec` reported ~21,400 qps.
- The FP16 run produced a byte-identical engine size and identical numerics to
  FP32. `--fp16` only *permits* reduced precision; on a graph this small
  TensorRT's tactic search kept FP32 kernels. This is not evidence that FP16 is
  unavailable — `builder.platform_has_fast_fp16` and
  `platform_has_fast_int8` are both **true** on this board.
- The Python path is slower per call because it measures wall clock around
  `execute_async_v3` plus a full stream synchronize, which for a 0.04 ms kernel
  is dominated by launch and sync overhead. It is not a like-for-like number
  against `trtexec`'s GPU-compute timing.
- No `pycuda` or `cuda-python` is installed, so device buffers are allocated
  through a small `ctypes` binding over `libcudart.so.12`. That worked, and it is
  worth knowing that a TensorRT Python pipeline here needs either that shim or a
  new dependency.

Memory during the smoke test:

| Phase | Peak process RSS | Min system MemAvailable | Swap used |
| --- | --- | --- | --- |
| `trtexec` FP32 build + run | not measured (see below) | 3832 MB | 0 |
| `trtexec` FP16 build + run | not measured (see below) | 3821 MB | 0 |
| Python API build | 1491 MB | 3843 MB | 0 |
| Python API inference | 1496 MB | 3839 MB | 0 |

**The ~1.5 GB builder peak for a 68 KB graph is the most important number here.**
`libnvinfer_builder_resource.so` alone is 152 MB, and the tactic search is not
free. Engine *building* must be budgeted separately from engine *running*, and on
an 8 GB board that means building engines when nothing else large is resident.
`cudaMemGetInfo` moved from 4882 MB free to 4012 MB free across the Python run,
against a 7990 MB total pool. Swap was never touched by any phase.

> **Caveat on the `trtexec` RSS figures.** The run that produced the current
> `data/bringup/g1_runtime.json` sampled `/proc/self/status`, which for the
> `trtexec` phases measured the wrapper Python process rather than the `trtexec`
> child — so the ~43 MB in that JSON is an artifact, not `trtexec`'s cost.
> `scripts/bringup/g1_tensorrt_smoke.py` has since been fixed to sample the
> child's RSS via `run_watched()`, but the fix has **not been re-run** (the shell
> session died before it could be). **Re-run the gate to populate real `trtexec`
> memory numbers**; a `tracked_pid` field in each `memory` block marks output from
> the fixed version. Everything else in the table and the JSON — build times,
> latencies, engine sizes, accuracy, and the system-wide MemAvailable troughs —
> is unaffected, since those were never measured via `/proc/self`.

### Sandbox note

The Tegra device nodes (`/dev/nvmap`, `/dev/nvhost-*`) are not visible inside the
agent sandbox, so CUDA cannot initialize there:
`NvRmMemInitNvmap failed with No such file or directory`, and `cuInit` returns
`CUDA_ERROR_UNKNOWN`. `nvidia-smi` and `__nvcc_device_query` fail the same way.
Inventory, `dpkg`, and `tegrastats` all work sandboxed; **anything that touches
the GPU must run unsandboxed.** No `sudo` was used and no system state was
changed.

## Memory headroom

| Metric | Value |
| --- | --- |
| Unified memory total | 7802736 kB (~7.4 GiB); `cudaMemGetInfo` reports 7990 MB, `tegrastats` 7620 MB |
| MemAvailable (idle, end of run) | ~4.8–5.3 GB across runs |
| `tegrastats` RAM at idle | ~2.4–2.8 GB used of 7620 MB |
| Swap | 32 GiB, `/ssd/32GB.swap`, priority -2, **0 B used** |
| `vm.swappiness` | **60** |

Orin has no discrete VRAM — CPU and GPU share these pages, so "free GPU memory"
is not a separate quantity.

`MemTotal - MemAvailable` reads ~3.8 GB while `tegrastats` reports ~2.8 GB
resident. The kernel's `MemAvailable` conservatively discounts cache it is not
sure it can reclaim, so it under-reports headroom. Both are recorded in the JSON
rather than picking one.

**Swap, recorded not changed** (per the ticket): the board has 32 GiB at
`/ssd/32GB.swap` with `vm.swappiness=60`, in `fstab` as
`/ssd/32GB.swap swap swap defaults 0 0`. The spec originally asked for
`/swapfile` with `vm.swappiness=10`. This matches the deviation already accepted
in [Stage A notes](01-os.md) and [JETBOT_SPEC.md](../../JETBOT_SPEC.md): the swap
is the right size on the right device, and re-running `scripts/setup_swap.sh`
would add a second 32 GiB file plus a duplicate `fstab` entry. Left alone.

### GPU memory reporting on Tegra

- `tegrastats` `RAM`/`SWAP` fields — works unprivileged, the practical choice.
- `cudaMemGetInfo()` — reports the shared pool, not a private GPU heap.
- `nvidia-smi` — the binary exists at `/usr/sbin/nvidia-smi` and returns 0
  unsandboxed, but it is not the useful per-process memory tool it is on
  discrete GPUs.
- `/sys/kernel/debug/nvmap` — root only, not readable here.

## Feasibility for G2 / G3 / G4

The spec's Stage G assumes three TensorRT engines. **None of these three models
is published as a TensorRT engine, and all three would need PyTorch, which is
not installed.** Base TensorRT being healthy does not carry Stage G.

### G2 — Qwen2.5-VL-3B INT4 AWQ

**Not feasible as specified.** The runtime named in the spec is not here.

- TensorRT-LLM is not installed, and NVIDIA publishes **no Tegra aarch64 wheel**.
  The aarch64 wheels on `pypi.nvidia.com` are SBSA (Grace-class servers); their
  TensorRT dependency fails outright with
  `TensorRT does not currently build wheels for Tegra systems`.
- Jetson support lives on the **`v0.12.0-jetson`** branch, built from source with
  `--cuda_architectures 87`. That branch targets **JetPack 6.1 on Jetson AGX Orin
  64 GB**; this board is L4T R36.4.4 on an **Orin Nano 8 GB**. It does pair with
  TensorRT 10.3, which matches. Expect a multi-hour compile leaning on the 32 GiB
  swap, and treat it as unproven on this board.
- **INT4 AWQ specifically will not load.** AWQ checkpoints depend on AutoAWQ GEMM
  / Triton kernels that are not available on Jetson aarch64. TensorRT 10.3 does
  expose `DataType.INT4` and `BuilderFlag.INT4`, but that is weight-only
  quantization driven by explicit Q/DQ nodes in an ONNX graph — it is not an AWQ
  checkpoint loader, and it gives you no LLM serving layer (no paged KV cache, no
  in-flight batching). Re-quantizing on-device is also out: no PyTorch, no
  `transformers`, no `modelopt`.

Realistic paths, best first:

1. **llama.cpp with a Qwen2.5-VL GGUF pair.** A Q4_K_M text backbone (~2.0 GB)
   plus a mandatory F16 `mmproj` vision encoder (~1.25 GB, not quantizable
   without visible degradation). Needs a CUDA build of llama.cpp; no llama.cpp
   binary exists on this host. Vision support for this architecture has needed
   recent or forked llama.cpp, so pin and verify the build.
2. **MLC-LLM via `jetson-containers`.** The daemon is active and the checkout is
   present; no images are pulled yet.
3. **Build TensorRT-LLM `v0.12.0-jetson` from source** — highest cost, highest
   risk, only worth it if the spec's C++ runtime is a hard requirement.

Memory envelope: ~2.0–2.2 GB of Q4 weights + ~1.25 GB F16 vision tower + KV
cache and activations, against ~4.8–5.3 GB MemAvailable. It fits, but only with
the vision tower loaded on demand and little else resident. **The real risk is
co-residency with the Stage F voice stack on 8 GB shared memory, not the weights
themselves.**

### G3 — smolvla-jetbot

**Not feasible as a TensorRT engine. Feasible in PyTorch once torch exists.**

- `lerobot/smolvla_base` ships **PyTorch safetensors** (~450M params, F32/BF16),
  loaded via LeRobot's `SmolVLAPolicy`. There is no published ONNX export or
  TensorRT engine.
- PyTorch is absent, and torch for Jetson cannot come from PyPI — it needs
  NVIDIA's Jetson wheel index or a `jetson-containers` image matched to CUDA 12.6
  / L4T R36.
- A TensorRT engine would first require exporting the flow-matching action expert
  to ONNX. That export does not exist upstream; it is its own project, not a
  bring-up gate.
- The `smolvla-jetbot` checkpoint the spec names **does not exist yet**. Only the
  `lerobot/smolvla_base` starting point does; a JetBot fine-tune is future work.

Path: install Jetson PyTorch + LeRobot and run `SmolVLAPolicy` eagerly for the
dummy motor-token gate. 450M params in BF16 is ~0.9 GB, which fits comfortably.
Keep the TensorRT export as a later optimization ticket. The safety rule is
unaffected — dummy motor-token I/O only, no PWM.

### G4 — llama-nemotron-embed-vl-1b-v2

**Not feasible as a TensorRT engine. Feasible in PyTorch, but memory-expensive.**

- `nvidia/llama-nemotron-embed-vl-1b-v2` is real and ships HF safetensors: a
  transformer encoder pairing Llama 3.2 1B with a SigLip2 400M vision encoder,
  mean-pooled to a single 2048-dim embedding.
- **The "1b" in the name is misleading — it is ~1.7B params**, so FP16 weights are
  ~3.4 GB, not ~2 GB. That is a large slice of this board.
- NVIDIA's optimized path for it is a **NeMo Retriever NIM**, not a hand-built
  TensorRT engine. NIM containers are x86-first and not a drop-in on Tegra.
- Blocked on the same missing PyTorch/`transformers`.

Path: install Jetson PyTorch + `transformers` and run it eagerly for the
dummy-vector gate. For Stage I memory, seriously consider a smaller text-only
embedder — a 1.7B multimodal encoder is a heavy choice for a JetBot that also
has to hold a VLM and the voice stack in 8 GB.

## Spec mismatches to resolve

1. **"NVIDIA TensorRT Edge-LLM (C++ Runtime)" is not a real product name.** The
   nearest real thing is TensorRT-LLM, and it is not installed. The spec and
   `07-tensorrt.md` ticket 1 ("TensorRT-Edge-LLM runtime present") should be
   renamed and rescoped.
2. **"NeMo TensorRT Engines" is not an installed capability.** Only base TensorRT
   10.3 is installed. There is no NeMo, no TensorRT-LLM, and no ONNX/TensorRT
   export of any of the three named models. `JETBOT_SPEC.md` §1 already hedges
   this for the voice models; the same hedge applies to the Stage G models.
3. **Stage G's ticket list assumes three TensorRT engines.** None of the three
   models is published in that form. Stage G should be rescoped from "build these
   three engines" to "stand up a runtime that can execute these models at all,"
   with TensorRT export as a separate later optimization.
4. **PyTorch is a hard prerequisite that no stage currently owns.** G3 and G4 are
   both blocked on it, and installing it on Jetson is non-trivial (NVIDIA wheel
   index or container, matched to CUDA 12.6 / L4T R36). This deserves its own
   ticket ahead of G3.
5. **The `.venv` cannot see the TensorRT bindings.** Either recreate it with
   `--system-site-packages` or keep the `PYTHONPATH` bridge. Worth deciding
   deliberately rather than rediscovering per ticket.

## G1 gate status

Issue #16 asks that "the TensorRT section reports the installed version and the
runtime imports without error."

**Met, and then some** — TensorRT 10.3.0.30 is reported, the runtime imports in
system `python3` and in the bridged `.venv`, and it additionally built and ran
three engines with verified numerics. Note that `./scripts/diagnostics.sh`, the
command in the ticket, was not the vehicle: G1 used
`scripts/bringup/g1_tensorrt_smoke.sh`, which goes considerably further. Wiring
the diagnostics script to surface these same fields is a small follow-up.

The **"TensorRT-Edge-LLM runtime present"** half of the ticket is **not met and
cannot be met by installation alone** — see the mismatches above.

One follow-up before this is folded into `07-tensorrt.md`: re-run
`./scripts/bringup/g1_tensorrt_smoke.sh` to regenerate `g1_runtime.json` with the
fixed child-process RSS sampling, so the `trtexec` memory rows stop reading
"not measured". Nothing else about the gate result depends on it.

## Repo notes

- `data/bringup/**` is already ignored by `.gitignore`, and `**` spans
  subdirectories, so `data/bringup/g1/` needs no new rule. `*.onnx` and
  `*.engine` are ignored globally as well. **No `.gitignore` change was needed.**
- `onnx`, `protobuf`, and `ml_dtypes` are installed into
  `data/bringup/g1/pylibs` with `pip install --no-deps --target` (~89 MB,
  ignored). This is deliberate: installing them into the shared `.venv` would let
  pip's resolver move `numpy` off the version the Stage F voice gates are pinned
  against. The wrapper script re-installs them if the directory is missing.
