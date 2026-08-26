#!/usr/bin/env python3
"""G1: inventory the NVIDIA inference runtime and prove TensorRT end-to-end.

Two independent build paths are exercised on the same tiny ONNX graph so a
failure can be attributed to either the C++ tooling or the Python bindings:

1. ``trtexec`` (C++), FP32 and FP16 — build time, engine size, GPU compute latency.
2. The TensorRT Python API + a ctypes shim over ``libcudart`` — this board has
   neither ``pycuda`` nor ``cuda-python``, so device buffers are allocated
   directly through the CUDA runtime rather than through a wrapper package.

Outputs are checked against a NumPy reference for the whole graph, so a build
that silently produces garbage fails instead of passing on exit code alone.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "bringup"
ARTIFACT_DIR = OUT_DIR / "g1"
JSON_OUT = OUT_DIR / "g1_runtime.json"

TRTEXEC = Path("/usr/src/tensorrt/bin/trtexec")
CUDA_HOME = Path("/usr/local/cuda")
SEED = 20260826

IN_SHAPE = (1, 3, 32, 32)
IN_NAME = "input"
OUT_NAME = "output"

# CUDA runtime memcpy kinds.
H2D = 1
D2H = 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def sh(cmd: list[str] | str, timeout: int = 120) -> dict:
    """Run a command, never raise, and return a JSON-friendly result."""
    shell = isinstance(cmd, str)
    try:
        p = subprocess.run(
            cmd,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "cmd": cmd if shell else " ".join(cmd),
            "rc": p.returncode,
            "stdout": p.stdout.strip(),
            "stderr": p.stderr.strip(),
        }
    except FileNotFoundError:
        return {"cmd": cmd if shell else " ".join(cmd), "rc": 127, "stdout": "", "stderr": "not found"}
    except subprocess.TimeoutExpired:
        return {"cmd": cmd if shell else " ".join(cmd), "rc": 124, "stdout": "", "stderr": "timeout"}


def first_line(res: dict) -> str | None:
    text = res["stdout"] or res["stderr"]
    return text.splitlines()[0].strip() if text else None


def meminfo_kb() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts:
            out[key] = int(parts[0])
    return out


def proc_status_kb(key: str, pid: str = "self") -> int:
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
    except OSError:  # the process exited between sampling and reading
        return 0
    for line in lines:
        if line.startswith(key + ":"):
            return int(line.split()[1])
    return 0


class MemWatch:
    """Sample unified-memory pressure while a build or inference runs.

    Tegra has no discrete VRAM and no working nvidia-smi, so the only honest
    signals are the tracked process's high-water RSS and the system-wide
    MemAvailable trough. Both are recorded rather than one synthesised "GPU
    memory" number.

    ``pid`` tracks a child process instead of this one. Without it, wrapping a
    subprocess would report the wrapper's RSS and badly understate the real cost.
    """

    def __init__(self, interval: float = 0.05, pid: str = "self") -> None:
        self.interval = interval
        self.pid = pid
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.min_avail_kb = meminfo_kb()["MemAvailable"]
        self.max_rss_kb = proc_status_kb("VmRSS", pid)
        self.max_swap_used_kb = 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            mi = meminfo_kb()
            self.min_avail_kb = min(self.min_avail_kb, mi["MemAvailable"])
            self.max_swap_used_kb = max(
                self.max_swap_used_kb, mi["SwapTotal"] - mi["SwapFree"]
            )
            self.max_rss_kb = max(self.max_rss_kb, proc_status_kb("VmRSS", self.pid))
            self._stop.wait(self.interval)

    def __enter__(self) -> "MemWatch":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def report(self) -> dict:
        return {
            "tracked_pid": self.pid,
            "peak_process_rss_mb": round(self.max_rss_kb / 1024, 1),
            "min_system_mem_available_mb": round(self.min_avail_kb / 1024, 1),
            "peak_swap_used_mb": round(self.max_swap_used_kb / 1024, 1),
        }


def run_watched(cmd: list[str], timeout: int) -> tuple[dict, dict, float]:
    """Run a child process while sampling *its* RSS, not this process's."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    with MemWatch(pid=str(proc.pid)) as watch:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            rc = 124
    wall = time.perf_counter() - t0
    res = {"cmd": " ".join(cmd), "rc": rc, "stdout": stdout.strip(), "stderr": stderr.strip()}
    return res, watch.report(), wall


# --------------------------------------------------------------------------- #
# inventory
# --------------------------------------------------------------------------- #
def probe_python(label: str, argv: list[str], extra_env_path: str | None = None) -> dict:
    """Ask another interpreter which inference modules it can import."""
    probe = r"""
import importlib, json, sys
mods = ["tensorrt","onnx","onnxruntime","torch","torchvision","numpy",
        "pycuda","cuda","polygraphy","onnx_graphsurgeon","tensorrt_llm"]
out = {"executable": sys.executable, "version": sys.version.split()[0], "modules": {}}
for m in mods:
    try:
        mod = importlib.import_module(m)
        out["modules"][m] = {
            "present": True,
            "version": str(getattr(mod, "__version__", "unknown")),
            "path": getattr(mod, "__file__", None),
        }
    except Exception as e:
        out["modules"][m] = {"present": False, "error": type(e).__name__}
try:
    import onnxruntime as ort
    out["onnxruntime_providers"] = ort.get_available_providers()
except Exception:
    out["onnxruntime_providers"] = None
print(json.dumps(out))
"""
    env = dict(os.environ)
    if extra_env_path:
        env["PYTHONPATH"] = extra_env_path
    else:
        env.pop("PYTHONPATH", None)
    try:
        p = subprocess.run(argv + ["-c", probe], capture_output=True, text=True, timeout=120, env=env)
        data = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001 - a broken interpreter is a finding, not a crash
        return {"label": label, "ok": False, "error": f"{type(e).__name__}: {e}"}
    data.update({"label": label, "ok": True, "pythonpath": extra_env_path})
    return data


def bundled_onnxruntime() -> dict:
    """sherpa-onnx (Stage F4/F5) vendors its own libonnxruntime.so.

    The Python ``onnxruntime`` package is a separate thing; report the vendored
    library's real linkage instead of assuming the voice stack implies a GPU EP.
    """
    lib = ROOT / ".venv/lib/python3.10/site-packages/sherpa_onnx/lib/libonnxruntime.so"
    info: dict = {"path": str(lib), "present": lib.exists()}
    if not lib.exists():
        return info
    info["size_mb"] = round(lib.stat().st_size / 1e6, 1)
    strings = sh(f"strings {lib} 2>/dev/null | grep -E '^1\\.[0-9]+\\.[0-9]+$' | sort -u")
    info["version_candidates"] = strings["stdout"].splitlines()[:5]
    ldd = sh(["ldd", str(lib)])
    info["links_cuda"] = bool(re.search(r"libcud|libnvinfer", ldd["stdout"]))
    info["ldd"] = ldd["stdout"].splitlines()
    info["effective_providers"] = ["CPUExecutionProvider"] if not info["links_cuda"] else ["unknown"]
    return info


def inventory() -> dict:
    nvcc = CUDA_HOME / "bin" / "nvcc"
    cuda_version_json = CUDA_HOME / "version.json"
    inv: dict = {
        "host": {
            "hostname": platform.node(),
            "kernel": platform.release(),
            "arch": platform.machine(),
            "os": first_line(sh("grep PRETTY_NAME /etc/os-release")),
            "l4t": first_line(sh("cat /etc/nv_tegra_release")),
            "l4t_core_pkg": first_line(sh("dpkg-query -W -f='${Version}' nvidia-l4t-core")),
            "power_mode": first_line(sh("nvpmodel -q")),
        },
        "cuda": {
            "nvcc_path": str(nvcc) if nvcc.exists() else None,
            "nvcc_on_PATH": shutil.which("nvcc"),
            "nvcc_version": None,
            "toolkit_version": None,
            "sm_arch": None,
            "libcudart": first_line(sh("ldconfig -p | grep 'libcudart.so.12'")),
            "libcuda": first_line(sh("ldconfig -p | grep 'libcuda.so.1'")),
        },
        "tensorrt": {},
        "cudnn": {},
        "cublas": {},
        "tensorrt_llm": {},
        "onnxruntime": {},
        "python_interpreters": [],
    }

    if nvcc.exists():
        res = sh([str(nvcc), "--version"])
        m = re.search(r"release ([\d.]+), V([\d.]+)", res["stdout"])
        if m:
            inv["cuda"]["nvcc_version"] = m.group(2)
    if cuda_version_json.exists():
        try:
            inv["cuda"]["toolkit_version"] = json.loads(cuda_version_json.read_text())["cuda"]["version"]
        except Exception:  # noqa: BLE001
            pass
    q = sh([str(CUDA_HOME / "bin" / "__nvcc_device_query")])
    if q["rc"] == 0 and q["stdout"].strip().isdigit():
        inv["cuda"]["sm_arch"] = int(q["stdout"].strip())
    else:
        inv["cuda"]["sm_arch_error"] = (q["stdout"] or q["stderr"])[:400]

    # TensorRT
    pkgs = sh("dpkg-query -W -f='${Package} ${Version}\\n' 'libnvinfer*' 'tensorrt*' "
              "'python3-libnvinfer*' 'libnvonnxparsers*' 2>/dev/null")
    trt_pkgs = {}
    for line in pkgs["stdout"].splitlines():
        parts = line.split()
        if len(parts) == 2 and not parts[1].startswith("<"):
            trt_pkgs[parts[0]] = parts[1]
    so = Path("/usr/lib/aarch64-linux-gnu/libnvinfer.so.10")
    inv["tensorrt"] = {
        "dpkg_packages": trt_pkgs,
        "meta_package_version": trt_pkgs.get("tensorrt"),
        "libnvinfer_so": str(so.resolve()) if so.exists() else None,
        "libnvinfer_so_size_mb": round(so.resolve().stat().st_size / 1e6, 1) if so.exists() else None,
        "trtexec_path": str(TRTEXEC) if TRTEXEC.exists() else None,
        "trtexec_on_PATH": shutil.which("trtexec"),
        "samples_dir": "/usr/src/tensorrt" if Path("/usr/src/tensorrt").exists() else None,
    }
    if TRTEXEC.exists():
        ver = sh([str(TRTEXEC), "--version"], timeout=60)
        m = re.search(r"TensorRT\.trtexec \[TensorRT v(\d+)\]", ver["stdout"])
        inv["tensorrt"]["trtexec_build"] = m.group(1) if m else None

    # cuDNN / cuBLAS
    cudnn_hdr = Path("/usr/include/cudnn_version.h")
    cudnn_ver = None
    if cudnn_hdr.exists():
        txt = cudnn_hdr.read_text()
        nums = [
            re.search(rf"#define CUDNN_{k}\s+(\d+)", txt)
            for k in ("MAJOR", "MINOR", "PATCHLEVEL")
        ]
        if all(nums):
            cudnn_ver = ".".join(n.group(1) for n in nums)
    inv["cudnn"] = {
        "version": cudnn_ver,
        "dpkg": {
            k: v
            for k, v in (
                line.split()
                for line in sh("dpkg-query -W -f='${Package} ${Version}\\n' 'libcudnn*' 2>/dev/null")["stdout"].splitlines()
                if len(line.split()) == 2 and not line.split()[1].startswith("<")
            )
        },
        "libcudnn_so": first_line(sh("readlink -f /usr/lib/aarch64-linux-gnu/libcudnn.so.9")),
    }
    cublas = sorted(Path("/usr/local/cuda/targets/aarch64-linux/lib").glob("libcublas.so.*.*"))
    inv["cublas"] = {
        "so": str(cublas[-1]) if cublas else None,
        "version": cublas[-1].name.replace("libcublas.so.", "") if cublas else None,
        "dpkg": first_line(sh("dpkg-query -W -f='${Package} ${Version}' libcublas-12-6")),
    }

    # TensorRT-LLM / "Edge-LLM": look, do not assume.
    search = sh(
        "find /opt /usr/local /usr/src /usr/lib/python3.10 /home/impulse110 -maxdepth 4 "
        "\\( -iname '*tensorrt_llm*' -o -iname '*tensorrt-llm*' -o -iname '*edge*llm*' \\) "
        "-print 2>/dev/null",
        timeout=180,
    )
    inv["tensorrt_llm"] = {
        "filesystem_hits": [h for h in search["stdout"].splitlines() if h][:20],
        "dpkg": first_line(sh("dpkg-query -W -f='${Package} ${Version}' 'tensorrt-llm*' 2>/dev/null")),
        "importable_system_python": sh([sys.executable, "-c", "import tensorrt_llm"])["rc"] == 0,
        "present": False,
    }
    inv["tensorrt_llm"]["present"] = bool(inv["tensorrt_llm"]["filesystem_hits"])

    inv["onnxruntime"] = {
        "python_package": "see python_interpreters",
        "vendored_by_sherpa_onnx": bundled_onnxruntime(),
    }

    venv_py = ROOT / ".venv" / "bin" / "python3"
    system_dist = "/usr/lib/python3.10/dist-packages"
    local_libs = str(ARTIFACT_DIR / "pylibs")
    inv["python_interpreters"] = [
        probe_python("system python3", ["/usr/bin/python3"]),
        probe_python("repo .venv (as-is)", [str(venv_py)]) if venv_py.exists() else {"label": "repo .venv", "ok": False, "error": "missing"},
        probe_python(
            "repo .venv + system dist-packages on PYTHONPATH",
            [str(venv_py)],
            extra_env_path=f"{system_dist}:{local_libs}",
        ) if venv_py.exists() else {"label": "bridged .venv", "ok": False, "error": "missing"},
    ]
    return inv


# --------------------------------------------------------------------------- #
# tiny ONNX graph + NumPy reference
# --------------------------------------------------------------------------- #
def make_weights() -> dict:
    rng = np.random.default_rng(SEED)
    return {
        "w1": (rng.standard_normal((8, 3, 3, 3)) * 0.10).astype(np.float32),
        "b1": (rng.standard_normal(8) * 0.05).astype(np.float32),
        "w2": (rng.standard_normal((4, 8, 3, 3)) * 0.10).astype(np.float32),
        "b2": (rng.standard_normal(4) * 0.05).astype(np.float32),
        "w3": (rng.standard_normal((1024, 16)) * 0.03).astype(np.float32),
        "b3": (rng.standard_normal(16) * 0.05).astype(np.float32),
        "x": (rng.standard_normal(IN_SHAPE) * 0.5).astype(np.float32),
    }


def conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, pad: int) -> np.ndarray:
    n, _, h, ww = x.shape
    f, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    oh = (h + 2 * pad - kh) // stride + 1
    ow = (ww + 2 * pad - kw) // stride + 1
    out = np.zeros((n, f, oh, ow), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            patch = xp[:, :, i : i + oh * stride : stride, j : j + ow * stride : stride]
            out += np.einsum("nchw,fc->nfhw", patch, w[:, :, i, j], optimize=True)
    return out + b.reshape(1, f, 1, 1)


def numpy_reference(t: dict) -> np.ndarray:
    h = np.maximum(conv2d(t["x"], t["w1"], t["b1"], stride=1, pad=1), 0.0)
    h = np.maximum(conv2d(h, t["w2"], t["b2"], stride=2, pad=1), 0.0)
    h = h.reshape(h.shape[0], -1)
    return (h @ t["w3"] + t["b3"]).astype(np.float32)


def build_onnx(t: dict, path: Path) -> None:
    from onnx import TensorProto, helper, numpy_helper, save

    init = [
        numpy_helper.from_array(t["w1"], "w1"),
        numpy_helper.from_array(t["b1"], "b1"),
        numpy_helper.from_array(t["w2"], "w2"),
        numpy_helper.from_array(t["b2"], "b2"),
        numpy_helper.from_array(t["w3"], "w3"),
        numpy_helper.from_array(t["b3"], "b3"),
    ]
    nodes = [
        helper.make_node("Conv", [IN_NAME, "w1", "b1"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1]),
        helper.make_node("Relu", ["c1"], ["r1"]),
        helper.make_node("Conv", ["r1", "w2", "b2"], ["c2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[2, 2]),
        helper.make_node("Relu", ["c2"], ["r2"]),
        helper.make_node("Flatten", ["r2"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "w3", "b3"], [OUT_NAME], alpha=1.0, beta=1.0, transB=0),
    ]
    graph = helper.make_graph(
        nodes,
        "g1_smoke",
        [helper.make_tensor_value_info(IN_NAME, TensorProto.FLOAT, list(IN_SHAPE))],
        [helper.make_tensor_value_info(OUT_NAME, TensorProto.FLOAT, [1, 16])],
        initializer=init,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    path.parent.mkdir(parents=True, exist_ok=True)
    save(model, str(path))


def compare(got: np.ndarray, ref: np.ndarray, tol: float) -> dict:
    got = np.asarray(got, dtype=np.float32).reshape(ref.shape)
    abs_err = float(np.max(np.abs(got - ref)))
    denom = float(np.max(np.abs(ref))) or 1.0
    finite = bool(np.all(np.isfinite(got)))
    nonzero = bool(np.any(np.abs(got) > 1e-8))
    return {
        "max_abs_err": abs_err,
        "max_rel_err": abs_err / denom,
        "all_finite": finite,
        "not_all_zero": nonzero,
        "tolerance": tol,
        "numerically_sane": finite and nonzero and abs_err / denom < tol,
        "output_shape": list(got.shape),
        "first_values": [round(float(v), 5) for v in got.reshape(-1)[:6]],
        "reference_first_values": [round(float(v), 5) for v in ref.reshape(-1)[:6]],
    }


# --------------------------------------------------------------------------- #
# path 1: trtexec
# --------------------------------------------------------------------------- #
def run_trtexec(onnx_path: Path, x: np.ndarray, ref: np.ndarray, fp16: bool) -> dict:
    label = "fp16" if fp16 else "fp32"
    engine = ARTIFACT_DIR / f"g1_smoke_{label}.engine"
    raw_in = ARTIFACT_DIR / "g1_input.dat"
    out_json = ARTIFACT_DIR / f"g1_trtexec_{label}_out.json"
    log_path = ARTIFACT_DIR / f"g1_trtexec_{label}.log"
    raw_in.write_bytes(x.tobytes())

    cmd = [
        str(TRTEXEC),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine}",
        f"--loadInputs={IN_NAME}:{raw_in}",
        f"--exportOutput={out_json}",
        "--memPoolSize=workspace:256M",
        "--warmUp=200",
        "--iterations=100",
        "--avgRuns=50",
        "--noDataTransfers",
    ]
    if fp16:
        cmd.append("--fp16")

    res, mem, wall = run_watched(cmd, timeout=900)
    log_path.write_text(f"$ {' '.join(cmd)}\n\n{res['stdout']}\n{res['stderr']}\n")

    out: dict = {
        "precision": label,
        "cmd": " ".join(cmd),
        "returncode": res["rc"],
        "passed": False,
        "wall_time_s": round(wall, 2),
        "log": str(log_path.relative_to(ROOT)),
        "memory": mem,
    }
    text = res["stdout"] + "\n" + res["stderr"]
    m = re.search(r"Engine built in ([\d.]+) sec", text)
    out["engine_build_s"] = float(m.group(1)) if m else None
    m = re.search(r"GPU Compute Time: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms", text)
    if m:
        out["gpu_compute_ms"] = {
            "min": float(m.group(1)),
            "max": float(m.group(2)),
            "mean": float(m.group(3)),
            "median": float(m.group(4)),
        }
    m = re.search(r"Throughput: ([\d.]+) qps", text)
    out["throughput_qps"] = float(m.group(1)) if m else None
    if engine.exists():
        out["engine_bytes"] = engine.stat().st_size
        out["engine_kb"] = round(engine.stat().st_size / 1024, 1)
        out["engine_path"] = str(engine.relative_to(ROOT))

    if out_json.exists():
        try:
            dumped = json.loads(out_json.read_text())
            values = None
            for entry in dumped if isinstance(dumped, list) else [dumped]:
                if entry.get("name") == OUT_NAME or values is None:
                    values = entry.get("values")
            if values is not None:
                # FP16 accumulation on a graph this small still tracks FP32 closely.
                out["accuracy"] = compare(np.array(values, dtype=np.float32), ref, tol=0.05 if fp16 else 1e-3)
        except Exception as e:  # noqa: BLE001
            out["accuracy_error"] = f"{type(e).__name__}: {e}"

    out["passed"] = bool(
        res["rc"] == 0
        and out.get("engine_bytes")
        and out.get("accuracy", {}).get("numerically_sane")
    )
    return out


# --------------------------------------------------------------------------- #
# path 2: TensorRT Python API + ctypes CUDA runtime
# --------------------------------------------------------------------------- #
class Cudart:
    """Minimal ctypes binding: no pycuda / cuda-python is installed on this board."""

    def __init__(self) -> None:
        self.lib = ctypes.CDLL("libcudart.so.12")
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.lib.cudaRuntimeGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]

    def check(self, rc: int, what: str) -> None:
        if rc != 0:
            raise RuntimeError(f"{what} failed with CUDA error {rc}")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
        return ptr

    def to_device(self, ptr: ctypes.c_void_p, arr: np.ndarray) -> None:
        self.check(
            self.lib.cudaMemcpy(ptr, arr.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(arr.nbytes), H2D),
            "cudaMemcpy H2D",
        )

    def to_host(self, arr: np.ndarray, ptr: ctypes.c_void_p) -> None:
        self.check(
            self.lib.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ptr, ctypes.c_size_t(arr.nbytes), D2H),
            "cudaMemcpy D2H",
        )

    def stream(self) -> ctypes.c_void_p:
        s = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        return s

    def sync(self, s: ctypes.c_void_p) -> None:
        self.check(self.lib.cudaStreamSynchronize(s), "cudaStreamSynchronize")

    def mem_info_mb(self) -> dict:
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        self.check(self.lib.cudaMemGetInfo(ctypes.byref(free), ctypes.byref(total)), "cudaMemGetInfo")
        return {"free_mb": round(free.value / 1e6, 1), "total_mb": round(total.value / 1e6, 1)}

    def versions(self) -> dict:
        rt, drv = ctypes.c_int(), ctypes.c_int()
        self.lib.cudaRuntimeGetVersion(ctypes.byref(rt))
        self.lib.cudaDriverGetVersion(ctypes.byref(drv))
        fmt = lambda v: f"{v // 1000}.{(v % 1000) // 10}"  # noqa: E731
        return {"runtime": fmt(rt.value), "driver": fmt(drv.value)}


def run_trt_python(onnx_path: Path, x: np.ndarray, ref: np.ndarray) -> dict:
    out: dict = {"passed": False}
    try:
        import tensorrt as trt
    except Exception as e:  # noqa: BLE001
        out["error"] = f"tensorrt import failed: {type(e).__name__}: {e}"
        return out

    out["tensorrt_version"] = trt.__version__
    out["tensorrt_module"] = trt.__file__
    cudart = Cudart()
    out["cuda_versions"] = cudart.versions()
    out["unified_mem_before_mb"] = cudart.mem_info_mb()

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    with MemWatch() as watch:
        t0 = time.perf_counter()
        builder = trt.Builder(logger)
        network = builder.create_network()
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_path.read_bytes()):
            out["error"] = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            return out
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 256 << 20)
        serialized = builder.build_serialized_network(network, config)
        build_s = time.perf_counter() - t0

    if serialized is None:
        out["error"] = "build_serialized_network returned None"
        return out

    engine_path = ARTIFACT_DIR / "g1_smoke_python.engine"
    engine_path.write_bytes(bytes(serialized))
    out.update(
        {
            "platform_has_fast_fp16": bool(builder.platform_has_fast_fp16),
            "platform_has_fast_int8": bool(builder.platform_has_fast_int8),
            "engine_build_s": round(build_s, 2),
            "engine_bytes": engine_path.stat().st_size,
            "engine_kb": round(engine_path.stat().st_size / 1024, 1),
            "engine_path": str(engine_path.relative_to(ROOT)),
            "build_memory": watch.report(),
        }
    )

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(serialized))
    ctx = engine.create_execution_context()

    io = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        io.append(
            {
                "name": name,
                "mode": str(engine.get_tensor_mode(name)),
                "dtype": str(engine.get_tensor_dtype(name)),
                "shape": list(engine.get_tensor_shape(name)),
            }
        )
    out["io_tensors"] = io

    host_out = np.zeros((1, 16), dtype=np.float32)
    d_in = cudart.malloc(x.nbytes)
    d_out = cudart.malloc(host_out.nbytes)
    stream = cudart.stream()
    try:
        cudart.to_device(d_in, np.ascontiguousarray(x))
        ctx.set_tensor_address(IN_NAME, d_in.value)
        ctx.set_tensor_address(OUT_NAME, d_out.value)

        for _ in range(20):
            ctx.execute_async_v3(stream_handle=stream.value)
        cudart.sync(stream)

        with MemWatch() as watch2:
            samples = []
            for _ in range(200):
                t1 = time.perf_counter()
                ok = ctx.execute_async_v3(stream_handle=stream.value)
                cudart.sync(stream)
                samples.append((time.perf_counter() - t1) * 1000.0)
                if not ok:
                    out["error"] = "execute_async_v3 returned False"
                    return out
        cudart.to_host(host_out, d_out)
        out["unified_mem_after_mb"] = cudart.mem_info_mb()
    finally:
        cudart.lib.cudaStreamDestroy(stream)
        cudart.lib.cudaFree(d_in)
        cudart.lib.cudaFree(d_out)

    arr = np.array(samples)
    out["inference_ms"] = {
        "min": round(float(arr.min()), 4),
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(arr.mean()), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(arr.max()), 4),
        "iterations": int(arr.size),
        "note": "wall clock per execute_async_v3 + stream sync, H2D/D2H outside the loop",
    }
    out["inference_memory"] = watch2.report()
    out["accuracy"] = compare(host_out, ref, tol=1e-3)
    out["passed"] = bool(out["accuracy"]["numerically_sane"])
    return out


# --------------------------------------------------------------------------- #
# headroom
# --------------------------------------------------------------------------- #
def headroom() -> dict:
    mi = meminfo_kb()
    tegrastats = sh("timeout 4 tegrastats --interval 1000 | head -2", timeout=20)
    return {
        "unified_memory": {
            "note": "Orin has no discrete VRAM; CPU and GPU share these pages.",
            "mem_total_mb": round(mi["MemTotal"] / 1024, 1),
            "mem_free_mb": round(mi["MemFree"] / 1024, 1),
            "mem_available_mb": round(mi["MemAvailable"] / 1024, 1),
            "cached_mb": round(mi["Cached"] / 1024, 1),
        },
        "swap": {
            "swap_total_mb": round(mi["SwapTotal"] / 1024, 1),
            "swap_free_mb": round(mi["SwapFree"] / 1024, 1),
            "devices": sh("cat /proc/swaps")["stdout"].splitlines(),
            "fstab": sh("grep -i swap /etc/fstab")["stdout"].splitlines(),
            "swappiness": int(Path("/proc/sys/vm/swappiness").read_text().strip()),
            "spec_expectation": "/swapfile with vm.swappiness=10",
            "observed": "/ssd/32GB.swap with vm.swappiness=60 (recorded only, not changed)",
        },
        "gpu_memory_reporting": {
            "nvidia_smi_usable": sh("nvidia-smi -L")["rc"] == 0,
            "tegra_methods": [
                "tegrastats RAM/SWAP fields (works unprivileged)",
                "cudaMemGetInfo() — reports the shared pool, not a private GPU heap",
                "/sys/kernel/debug/nvmap — root only, not readable here",
            ],
            "tegrastats_sample": tegrastats["stdout"].splitlines()[:2],
        },
    }


# --------------------------------------------------------------------------- #
# downstream feasibility
# --------------------------------------------------------------------------- #
def feasibility(inv: dict, smoke: dict, head: dict) -> dict:
    """What G2/G3/G4 could actually execute here, based on probes rather than the spec.

    Every "blocker" below is something this run observed to be absent, not
    something inferred from the JETBOT_SPEC.md architecture table.
    """
    jc = Path("/home/impulse110/jetson-containers")
    bridged = next(
        (p for p in inv["python_interpreters"] if p.get("ok") and "PYTHONPATH" in p.get("label", "")),
        {},
    )
    mods = bridged.get("modules", {})
    torch_present = mods.get("torch", {}).get("present", False)
    trtllm_present = inv["tensorrt_llm"]["present"]
    py_api = smoke.get("python_api", {})

    hosts = {
        "torch_installed_anywhere": torch_present,
        "tensorrt_llm_installed": trtllm_present,
        "onnxruntime_python_installed": mods.get("onnxruntime", {}).get("present", False),
        "llama_cpp_binary": bool(shutil.which("llama-cli") or shutil.which("llama-qwen2vl-cli")),
        "mlc_installed": mods.get("mlc_llm", {}).get("present", False) if "mlc_llm" in mods else False,
        "docker_daemon_active": sh("systemctl is-active docker")["stdout"] == "active",
        "jetson_containers_checkout": str(jc) if jc.exists() else None,
        "jetson_containers_has_tensorrt_llm_pkg": (jc / "packages/llm/tensorrt_llm").exists(),
        "quantization_toolkit_installed": any(
            sh([sys.executable, "-c", f"import {m}"])["rc"] == 0
            for m in ("modelopt", "ammo")
        ),
    }

    unified_total_mb = smoke.get("python_api", {}).get("unified_mem_before_mb", {}).get("total_mb")
    # Use the idle reading taken before the smoke test, not a fresh one: by the
    # time this runs, this very process is still holding ~1.5 GB of builder RSS,
    # which would understate the headroom a downstream model would actually see.
    avail_mb = head["unified_memory"]["mem_available_mb"]
    total_mb = head["unified_memory"]["mem_total_mb"]

    return {
        "environment": hosts,
        "memory_envelope": {
            "unified_total_mb": unified_total_mb,
            "mem_total_mb": total_mb,
            "idle_mem_available_mb": avail_mb,
            "observed_trt_builder_peak_rss_mb": py_api.get("build_memory", {}).get("peak_process_rss_mb"),
            "note": (
                "The TensorRT builder alone peaked near 1.5 GB RSS for a 68 KB graph; "
                "libnvinfer_builder_resource.so is 152 MB and the tactic search is not free. "
                "Budget builder cost separately from engine runtime cost."
            ),
        },
        "g2_qwen25_vl_3b_int4_awq": {
            "spec_claim": "TensorRT Edge-LLM (C++ runtime) executes Qwen2.5-VL-3B INT4 AWQ",
            "runtime_present_for_that_claim": False,
            "blockers": [
                "TensorRT-LLM is not installed and NVIDIA publishes no Tegra aarch64 wheel; "
                "pypi.nvidia.com aarch64 wheels are SBSA (Grace) only and their TensorRT dep "
                "refuses to build on Tegra.",
                "Jetson TensorRT-LLM support lives on the v0.12.0-jetson branch, which must be "
                "built from source with --cuda_architectures 87; it targets JetPack 6.1 and "
                "Jetson AGX Orin 64 GB, while this board is L4T R36.4.4 / Orin Nano 8 GB.",
                "AWQ INT4 checkpoints depend on AutoAWQ GEMM / Triton kernels that are not "
                "available on Jetson aarch64, so an AWQ safetensors checkout is not directly runnable.",
                "No PyTorch, no transformers, no quantization toolkit (modelopt) installed, so "
                "re-quantizing on-device is also not currently possible.",
            ],
            "int4_support_in_installed_stack": {
                "tensorrt_datatype_int4": "trt.DataType.INT4 and BuilderFlag.INT4 exist in TensorRT 10.3",
                "caveat": (
                    "Plain TensorRT INT4 is weight-only quantization expressed through explicit "
                    "Q/DQ nodes in the ONNX graph. It is not an AWQ checkpoint loader, and there "
                    "is no LLM-shaped serving layer (paged KV cache, in-flight batching) without "
                    "TensorRT-LLM."
                ),
            },
            "realistic_paths": [
                "llama.cpp with a Qwen2.5-VL GGUF pair (Q4_K_M text backbone ~2.0 GB plus a "
                "mandatory F16 mmproj vision encoder ~1.25 GB). Needs a CUDA build of llama.cpp; "
                "no llama.cpp binary is present on this board.",
                "MLC-LLM via jetson-containers (docker daemon is active, but no images are pulled "
                "and this checkout has no tensorrt_llm package).",
                "Build TensorRT-LLM v0.12.0-jetson from source — a multi-hour compile that will "
                "lean hard on the 32 GiB swap and is unproven on Orin Nano 8 GB.",
            ],
            "expected_memory_envelope": (
                "INT4/Q4 weights ~2.0-2.2 GB plus an unquantized F16 vision encoder ~1.25 GB plus "
                f"KV cache and activations, against ~{avail_mb} MB MemAvailable at idle. It fits "
                "only with the vision tower loaded on demand and nothing else resident; running it "
                "concurrently with the Stage F voice stack on 8 GB shared memory is the real risk, "
                "not the weights themselves."
            ),
            "verdict": "NOT feasible as specified; feasible via llama.cpp GGUF after a CUDA build.",
        },
        "g3_smolvla_jetbot": {
            "spec_claim": "smolvla-jetbot runs as a TensorRT engine",
            "runtime_present_for_that_claim": False,
            "available_format": (
                "lerobot/smolvla_base ships PyTorch safetensors (~450M params, F32/BF16) and is "
                "loaded through LeRobot's SmolVLAPolicy. No TensorRT engine or ONNX export is published."
            ),
            "blockers": [
                "PyTorch is not installed on this board at all, in either interpreter.",
                "torch for Jetson cannot come from PyPI; it needs NVIDIA's Jetson wheel index or a "
                "jetson-containers image, matched to CUDA 12.6 / L4T R36.",
                "A TensorRT engine would require exporting the flow-matching action expert to ONNX "
                "first; that export does not exist upstream and is its own project.",
                "The 'smolvla-jetbot' checkpoint named in the spec is a fine-tune that does not "
                "exist yet — only the lerobot/smolvla_base starting point does.",
            ],
            "realistic_paths": [
                "Install Jetson PyTorch + LeRobot and run SmolVLAPolicy in eager mode for the "
                "dummy motor-token I/O gate. 450M params in BF16 is ~0.9 GB, which fits.",
                "Treat the TensorRT export as a later optimization ticket, not a Stage G gate.",
            ],
            "verdict": "NOT feasible as a TensorRT engine; feasible in PyTorch once torch is installed.",
        },
        "g4_nemotron_embedder": {
            "spec_claim": "llama-nemotron-embed-vl-1b-v2 runs as a TensorRT engine",
            "runtime_present_for_that_claim": False,
            "available_format": (
                "nvidia/llama-nemotron-embed-vl-1b-v2 ships HF safetensors and is a transformer "
                "encoder of ~1.7B params (Llama 3.2 1B text + SigLip2 400M vision), emitting a "
                "single 2048-dim embedding. NVIDIA's optimized path for it is a NeMo Retriever NIM, "
                "not a hand-built TensorRT engine."
            ),
            "blockers": [
                "No PyTorch / transformers to load the safetensors.",
                "NIM containers are x86-first and are not a drop-in on Tegra.",
                "The '1b' in the name is misleading: it is ~1.7B params, so FP16 weights are "
                "~3.4 GB, not ~2 GB.",
            ],
            "realistic_paths": [
                "Install Jetson PyTorch + transformers and run it eagerly for the dummy-vector gate.",
                "For Stage I memory, a much smaller text-only embedder would leave far more of the "
                "8 GB for the VLM; the multimodal 1.7B encoder is a heavy choice for a JetBot.",
            ],
            "verdict": "NOT feasible as a TensorRT engine; feasible in PyTorch, but memory-expensive.",
        },
        "spec_mismatches": [
            "JETBOT_SPEC.md names 'NVIDIA TensorRT Edge-LLM (C++ Runtime)'. No NVIDIA product ships "
            "under that name. The nearest real thing is TensorRT-LLM, and it is not installed here.",
            "The spec treats 'NeMo TensorRT Engines' as an installed capability. Only base TensorRT "
            "10.3 is installed; there is no NeMo, no TensorRT-LLM, and no ONNX/TensorRT export of any "
            "of the named models.",
            "Stage G's ticket list assumes all three models are TensorRT engines. None of them are "
            "published in that form; all three would need PyTorch, which is absent.",
            "The base TensorRT that IS installed is healthy and fast, so Stage G should be rescoped "
            "from 'build these three engines' to 'stand up a runtime that can execute these models "
            "at all', with TensorRT export as a later optimization.",
        ],
    }


# --------------------------------------------------------------------------- #
def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "ticket": "G1",
        "issue": "AbuAyah110/jetbot-orin-super#16",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runner": {"executable": sys.executable, "python": sys.version.split()[0], "numpy": np.__version__},
    }

    print("== inventory ==")
    report["inventory"] = inventory()
    trt_pkg = report["inventory"]["tensorrt"].get("meta_package_version")
    print(f"  CUDA {report['inventory']['cuda']['toolkit_version']}  TensorRT {trt_pkg}")
    print(f"  TensorRT-LLM present: {report['inventory']['tensorrt_llm']['present']}")

    print("== headroom ==")
    report["headroom"] = headroom()
    print(f"  MemAvailable {report['headroom']['unified_memory']['mem_available_mb']} MB, "
          f"swap {report['headroom']['swap']['swap_total_mb']} MB")

    print("== tiny ONNX graph ==")
    t = make_weights()
    onnx_path = ARTIFACT_DIR / "g1_smoke.onnx"
    smoke: dict = {}
    try:
        build_onnx(t, onnx_path)
        ref = numpy_reference(t)
        smoke["onnx"] = {
            "path": str(onnx_path.relative_to(ROOT)),
            "bytes": onnx_path.stat().st_size,
            "opset": 17,
            "graph": "Conv3x3(3->8) -> Relu -> Conv3x3s2(8->4) -> Relu -> Flatten -> Gemm(1024->16)",
            "input_shape": list(IN_SHAPE),
            "output_shape": [1, 16],
        }
        print(f"  {onnx_path.name} ({onnx_path.stat().st_size} B), reference computed in NumPy")
    except Exception as e:  # noqa: BLE001
        smoke["onnx_error"] = f"{type(e).__name__}: {e}"
        report["smoke_test"] = smoke
        JSON_OUT.write_text(json.dumps(report, indent=2) + "\n")
        print("FAIL: could not build the ONNX graph")
        return 1

    print("== trtexec (C++) ==")
    smoke["trtexec"] = {}
    for fp16 in (False, True):
        if not TRTEXEC.exists():
            smoke["trtexec"]["error"] = "trtexec not found"
            break
        r = run_trtexec(onnx_path, t["x"], ref, fp16=fp16)
        smoke["trtexec"][r["precision"]] = r
        lat = (r.get("gpu_compute_ms") or {}).get("median")
        print(f"  {r['precision']}: passed={r['passed']} build={r.get('engine_build_s')}s "
              f"engine={r.get('engine_kb')} KB median={lat} ms")

    print("== TensorRT Python API ==")
    smoke["python_api"] = run_trt_python(onnx_path, t["x"], ref)
    p = smoke["python_api"]
    print(f"  passed={p['passed']} build={p.get('engine_build_s')}s "
          f"engine={p.get('engine_kb')} KB median={(p.get('inference_ms') or {}).get('median')} ms")
    if p.get("error"):
        print(f"  error: {p['error']}")

    report["smoke_test"] = smoke

    print("== downstream feasibility ==")
    report["feasibility"] = feasibility(report["inventory"], smoke, report["headroom"])
    for key in ("g2_qwen25_vl_3b_int4_awq", "g3_smolvla_jetbot", "g4_nemotron_embedder"):
        print(f"  {key}: {report['feasibility'][key]['verdict']}")

    passed = [
        smoke["trtexec"].get("fp32", {}).get("passed"),
        smoke["trtexec"].get("fp16", {}).get("passed"),
        smoke["python_api"].get("passed"),
    ]
    report["verdict"] = {
        "trtexec_fp32": passed[0],
        "trtexec_fp16": passed[1],
        "python_api": passed[2],
        "tensorrt_runtime_usable": bool(any(passed)),
        "tensorrt_llm_present": report["inventory"]["tensorrt_llm"]["present"],
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {JSON_OUT.relative_to(ROOT)}")
    print(f"verdict: {report['verdict']}")
    return 0 if report["verdict"]["tensorrt_runtime_usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
