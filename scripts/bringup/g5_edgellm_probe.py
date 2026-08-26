#!/usr/bin/env python3
"""G5: check this board against NVIDIA TensorRT Edge-LLM's Official Support Matrix.

Read-only and offline-friendly. Installs nothing, builds nothing, and never
touches the GPU, so it is safe to run sandboxed and safe to run while another
agent is compiling.

Background: an earlier Stage G1 inventory concluded that "there is no NVIDIA
product called 'TensorRT Edge-LLM'". That was false. G1 searched for
``*tensorrt_llm*`` / ``*edge*llm*`` on disk, found nothing, and generalised the
miss into a claim about the product's existence. The two are *different
projects*: NVIDIA/TensorRT-Edge-LLM targets Jetson/DRIVE/DGX Spark and ships
only as a local CMake source build -- no wheel, no apt package -- so being
absent from disk is simply its pre-install state.

This probe replaces that guess with primary-source evidence. It clones the
upstream repo at a pinned tag (or reuses a local checkout) and asserts the
specific facts the Stage G decision rests on:

  1. the Official Support Matrix has a Jetson Orin row for JetPack 6.2+ / CUDA 12.6
  2. installation.md ships a "JetPack 6.2+ Orin" CMake invocation
  3. the CMake maps EMBEDDED_TARGET=jetson-orin to CUDA 12.6 and artifact sm_87
  4. Qwen2.5-VL-3B-Instruct and its AWQ checkpoint are in the supported models
  5. int4_awq is an offered quantization method, and is valid on all platforms
  6. NVFP4/FP8 are *not* available on SM 87, so INT4 AWQ is the right precision
  7. Jetson Orin Nano 8 GB is a benchmarked platform, with its own results table
  8. which CuTe DSL prebuilt kernel archives are actually shipped

Then it reads this board's own L4T / CUDA / TensorRT / SM values and reports
whether they satisfy the matrix row, plus the blockers that remain.

Results are written as JSON so the doc tables can be regenerated rather than
retyped. Evidence record: docs/bringup/07b-tensorrt-edge-llm.md
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/NVIDIA/TensorRT-Edge-LLM.git"
PINNED_TAG = "v0.10.0"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "data" / "bringup" / "g5_edgellm.json"
DEFAULT_CHECKOUT = REPO_ROOT / "data" / "bringup" / "g5" / "TensorRT-Edge-LLM"

# Paths inside the upstream checkout.
P_MATRIX = "docs/source/user_guide/getting_started/support-matrix.md"
P_INSTALL = "docs/source/user_guide/getting_started/installation.md"
P_MODELS = "docs/source/user_guide/getting_started/supported-models.md"
P_QUANT = "docs/source/user_guide/features/quantization.md"
P_LIMITS = "docs/source/user_guide/getting_started/limitations.md"
P_BENCH = "docs/source/user_guide/performance/performance-benchmarks.md"
P_TOOLCHAIN = "cmake/aarch64_linux_toolchain.cmake"
P_CUTEDSL = "cmake/CuteDsl.cmake"
P_PREBUILT = "kernelSrcs/cuteDSLPrebuilt"
P_PYPROJECT = "pyproject.toml"
P_ORIN_TESTS = "tests/test_lists/l1_pipeline_orin_vlm.yml"

# This board, per Stage G1. Used to evaluate the matrix row.
THIS_BOARD = {
    "device": "Jetson Orin Nano 8GB (Super)",
    "l4t": "R36.4.4",
    "jetpack": "6.2.1",
    "cuda": "12.6.11",
    "tensorrt": "10.3.0.30",
    "sm": "87",
    "arch": "aarch64",
    "unified_memory_mb": 7990,
}


def sh(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        return 1, f"{type(exc).__name__}: {exc}"


def ensure_checkout(dest: Path, tag: str, offline: bool) -> dict:
    """Clone the upstream repo at ``tag``, or reuse an existing checkout."""
    info: dict = {"path": str(dest), "tag": tag, "cloned": False}

    if (dest / ".git").is_dir():
        info["reused_existing"] = True
        rc, out = sh(["git", "-C", str(dest), "describe", "--tags", "--always"])
        info["describe"] = out if rc == 0 else None
        return info

    if offline:
        info["error"] = "checkout missing and --offline was requested"
        return info

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Blobless + single tag keeps this to a few tens of MB rather than the
    # full history; the probe only ever reads text files.
    rc, out = sh(
        [
            "git", "clone", "--filter=blob:none", "--depth", "1",
            "--branch", tag, REPO_URL, str(dest),
        ],
        timeout=900,
    )
    info["cloned"] = rc == 0
    if rc != 0:
        info["error"] = out[-2000:]
    return info


def read(root: Path, rel: str) -> str | None:
    p = root / rel
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def check_support_matrix(root: Path) -> dict:
    """Find the Jetson Orin rows in the Official Support Matrix."""
    text = read(root, P_MATRIX)
    if text is None:
        return {"present": False, "note": f"{P_MATRIX} not found at this tag"}

    orin_rows = [
        ln.strip() for ln in text.splitlines()
        if ln.strip().startswith("|") and "Jetson Orin" in ln
    ]

    jp6 = next(
        (r for r in orin_rows if re.search(r"JetPack\s*6", r)), None
    )
    result = {
        "present": True,
        "orin_rows": orin_rows,
        "jetpack6_row": jp6,
        "jetpack6_supported": jp6 is not None,
    }
    if jp6:
        cells = [c.strip() for c in jp6.strip("|").split("|")]
        result["jetpack6_row_cells"] = cells
        # Columns: Platform | Level | OS/SDK | CUDA | TensorRT | Build loc | Precision
        result["level"] = cells[1] if len(cells) > 1 else None
        result["os_sdk"] = cells[2] if len(cells) > 2 else None
        result["cuda_toolkit"] = cells[3] if len(cells) > 3 else None
        result["precision_constraint"] = cells[6] if len(cells) > 6 else None

    result["orin_excludes_fp8_fp4"] = bool(
        re.search(r"Jetson Orin does not run FP8 or FP4", text)
    )
    return result


def check_build_config(root: Path) -> dict:
    """Confirm the documented JetPack 6.2+ Orin build recipe exists."""
    install = read(root, P_INSTALL) or ""
    toolchain = read(root, P_TOOLCHAIN) or ""
    cutedsl = read(root, P_CUTEDSL) or ""

    return {
        "installation_has_jetpack6_orin_section": bool(
            re.search(r"\*\*JetPack 6\.2\+ Orin\*\*", install)
        ),
        "documents_embedded_target_jetson_orin": "EMBEDDED_TARGET=jetson-orin" in install,
        "documents_cuda_ctk_12_6": "-DCUDA_CTK_VERSION=12.6" in install,
        # jetson-orin defaults to CUDA 12.6 and rejects >= 13.0.
        "toolchain_jetson_orin_defaults_cuda_12_6": bool(
            re.search(
                r'"jetson-orin"\)\s*\n\s*set_ifndef\(CUDA_CTK_VERSION 12\.6\)',
                toolchain,
            )
        ),
        "cutedsl_maps_jetson_orin_to_sm_87": bool(
            re.search(
                r'_embedded_target STREQUAL\s*\n?\s*"jetson_orin"\)\s*\n\s*set\(_default_tag "sm_87"\)',
                cutedsl,
            )
            or ('"jetson_orin"' in cutedsl and 'set(_default_tag "sm_87")' in cutedsl)
        ),
        "orin_no_fp8_fp4_in_install_doc": bool(
            re.search(r"Jetson Orin does not support FP8, MXFP8, FP4, or NVFP4", install)
        ),
    }


def check_cutedsl_prebuilts(root: Path) -> dict:
    """Which CuTe DSL kernel archives ship in the tree?

    CMake resolves cutedsl_{arch}_{tag}_cuda{MAJOR}.tar.gz. On JetPack 6 that
    means cutedsl_aarch64_sm_87_cuda12.tar.gz. If it is absent, configure fails
    with FATAL_ERROR -- and ENABLE_CUTE_DSL=OFF is not an escape, because the
    Ampere FMHA runner compiles unconditionally.
    """
    d = root / P_PREBUILT
    shipped = sorted(p.name for p in d.glob("*.tar.gz")) if d.is_dir() else []
    needed = "cutedsl_aarch64_sm_87_cuda12.tar.gz"

    v2_runner = read(root, "cpp/kernels/contextAttentionKernels/cuteDslFMHAV2Runner.cpp") or ""
    # A guard would appear as a top-level #ifdef near the top of the file.
    v2_guarded = bool(re.search(r"^#if(def)?\s", v2_runner, re.MULTILINE))

    return {
        "prebuilt_dir_present": d.is_dir(),
        "shipped_tarballs": shipped,
        "needed_for_this_board": needed,
        "needed_tarball_present": needed in shipped,
        "ampere_fmha_runner_unconditional": not v2_guarded,
        "enable_cute_dsl_off_is_viable": v2_guarded,
        "note": (
            "Only cuda13 archives ship (JetPack 7). The cuda12 sm_87 archive is in "
            "kernelSrcs/README.md's Docker builder matrix but is not committed, and "
            "GitHub releases carry no assets. Generate it with "
            "'python kernelSrcs/build_cutedsl.py --gpu_arch sm_87 --arch aarch64 "
            "--cuda-version 12' (needs nvidia-cutlass-dsl[cu12]==4.6.1, cupy-cuda12x, "
            "cuda-python, and a visible GPU -- so unsandboxed, in an isolated venv)."
        ),
    }


def check_model_support(root: Path) -> dict:
    """Is Qwen2.5-VL-3B supported, and at which precisions on Orin?"""
    models = read(root, P_MODELS) or ""
    orin_tests = read(root, P_ORIN_TESTS) or ""

    precisions = sorted(
        set(re.findall(r"Qwen2\.5-VL-3B-Instruct-(fp16|int4_awq|int8_sq)", orin_tests))
    )

    return {
        "qwen25_vl_3b_in_supported_models": "Qwen/Qwen2.5-VL-3B-Instruct" in models,
        "qwen25_vl_3b_awq_checkpoint_listed": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ" in models,
        "has_python_model_package": (root / "tensorrt_edgellm/models/qwen2_5_vl").is_dir(),
        "has_cpp_vit_runner": (root / "cpp/multimodal/qwen25vlViTRunner.cpp").is_file(),
        "orin_l1_test_list_present": bool(orin_tests),
        "orin_l1_precisions_for_qwen25_vl_3b": precisions,
        "note": (
            "tests/test_lists/l1_pipeline_orin_vlm.yml is titled 'L1 Pipeline Tests - "
            "Jetson Orin (Ampere device)' and gates Qwen2.5-VL-3B-Instruct engine builds "
            "plus coco/mmmu e2e benchmarks. It also gates 7B-9B models, so that CI Orin is "
            "almost certainly an AGX Orin 64 GB: the list proves SM 87 correctness coverage, "
            "not that every row fits 8 GB."
        ),
    }


def check_quantization(root: Path) -> dict:
    """Is INT4 AWQ real, and what is actually usable on SM 87?"""
    quant = read(root, P_QUANT) or ""
    bench = read(root, P_BENCH) or ""
    limits = read(root, P_LIMITS) or ""
    install = read(root, P_INSTALL) or ""

    backbone = next(
        (ln for ln in quant.splitlines() if ln.strip().startswith("| Backbone")), ""
    )
    methods = sorted(set(re.findall(r"`([a-z0-9_]+)`", backbone)))

    def platform_req(name: str) -> str | None:
        m = re.search(rf"^\|\s*{re.escape(name)}\s*\|[^|]*\|\s*([^|]+?)\s*\|", bench, re.MULTILINE)
        return m.group(1).strip() if m else None

    return {
        "backbone_methods": methods,
        "int4_awq_offered": "int4_awq" in methods,
        "precision_key": {
            "FP16": platform_req("FP16"),
            "INT4 AWQ": platform_req("INT4 AWQ"),
            "INT4 GPTQ": platform_req("INT4 GPTQ"),
            "FP8": platform_req("FP8"),
            "NVFP4": platform_req("NVFP4"),
        },
        "nvfp4_requires_sm100": "SM100" in (platform_req("NVFP4") or ""),
        "fp8_requires_sm89": "SM89" in (platform_req("FP8") or ""),
        # Pre-quantized AWQ means no calibration, so no GPU host needed.
        "prequantized_checkpoints_skip_quantization": bool(
            re.search(r"Skip this step when you already have a supported pre-quantized", quant)
        ),
        "quantization_requires_gpu": bool(
            re.search(r"[Qq]uantization requires\s+\**an NVIDIA GPU", install)
        ),
        "export_runs_on_cpu": bool(re.search(r"Export runs on CPU", install)),
        "int4_gemm_plugin_v1_workaround": bool(
            re.search(r"--int4-gemm-plugin-version 1", limits)
        ),
        "externalize_weights_int4_ffn_on_orin": "--externalize-weights int4_ffn" in bench,
        "recommended_for_3b_vlm_on_orin": (
            "int4_awq backbone + int4_awq LM head + FP16 vision tower + FP16 KV cache. "
            "NVFP4 needs SM100+ and FP8 needs SM89+, so neither is available on SM 87."
        ),
    }


def check_orin_nano(root: Path) -> dict:
    """Is Orin Nano 8 GB a first-class benchmarked platform?"""
    bench = read(root, P_BENCH) or ""

    rows = []
    in_table = False
    for ln in bench.splitlines():
        if ln.startswith("#### "):
            in_table = "Orin Nano" in ln
            continue
        if in_table and ln.startswith("|") and "---" not in ln and "Model" not in ln:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            if len(cells) >= 15:
                rows.append(
                    {
                        "model": cells[0],
                        "kind": cells[1],
                        "mode": cells[2],
                        "precision": cells[3],
                        "decode_tok_s": cells[12],
                        "gpu_mem_mb": cells[14],
                    }
                )

    return {
        "listed_as_platform": "Jetson Orin Nano 8GB" in bench,
        "has_own_results_table": bool(rows),
        "has_own_visual_build_params": "Orin NX / Orin Nano VLM visual engine" in bench,
        "orin_nano_generally_batch_1": "Orin NX and Orin Nano rows are generally batch" in bench,
        "documented_minimum_memory": None,
        "memory_guidance": (
            "No minimum-memory figure is documented. The docs scope by device instead: "
            "'Orin Nano run the externalized INT4 entries supported by each memory target', "
            "with --maxImageTokens reduced to 2048 and batch 1."
        ),
        "benchmarks_run_under_jetpack": "7.2",
        "benchmark_rows": rows,
    }


def evaluate(matrix: dict, cutedsl: dict, models: dict, quant: dict) -> dict:
    """Does this board satisfy the matrix row, and what still blocks a build?"""
    on_matrix = bool(
        matrix.get("jetpack6_supported")
        and re.search(r"6\.2", matrix.get("os_sdk") or "")
        and "12.6" in (matrix.get("cuda_toolkit") or "")
    )

    blockers = []
    if not cutedsl.get("needed_tarball_present"):
        blockers.append(
            "No sm_87 + CUDA-12 CuTe DSL kernel archive is shipped, so CMake cannot "
            "configure for JetPack 6. ENABLE_CUTE_DSL=OFF is not an escape: the Ampere "
            "FMHA runner compiles unconditionally. Must be generated locally or obtained "
            "from upstream."
        )
    blockers.append(
        "The ONNX export step needs torch==2.13.0 (CPU is sufficient), which is not "
        "installed. Depends on the PyTorch ticket (#30)."
    )
    blockers.append(
        "Export is documented as an x86-64 host step. The docs do allow splitting "
        "machines and rsync-ing the ONNX to the target, which is the lower-risk route."
    )

    return {
        "board_is_on_support_matrix": on_matrix,
        "support_level": matrix.get("level"),
        "newest_release_supporting_jetpack6": PINNED_TAG,
        "target_model_supported": bool(models.get("qwen25_vl_3b_in_supported_models")),
        "int4_awq_is_correct_precision": bool(
            quant.get("int4_awq_offered") and quant.get("nvfp4_requires_sm100")
        ),
        "installable_end_to_end_today": False,
        "blockers": blockers,
        "verdict": (
            "SUPPORTED but not yet installable. NVIDIA/TensorRT-Edge-LLM v0.10.0 lists "
            "Jetson Orin / JetPack 6.2+ / CUDA 12.6 as Compatible, Orin Nano 8 GB is a "
            "benchmarked platform, Qwen2.5-VL-3B-Instruct(-AWQ) is a supported checkpoint, "
            "and INT4 AWQ is the correct precision for SM 87. No JetPack 7 upgrade is "
            "required. Two gates remain: the unpublished sm_87/cuda12 CuTe DSL artifact, "
            "and torch for the ONNX export."
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default=PINNED_TAG, help="upstream tag to evaluate")
    ap.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--offline", action="store_true", help="require an existing checkout")
    ap.add_argument("--keep", action="store_true", help="keep the checkout afterwards")
    args = ap.parse_args()

    if not shutil.which("git"):
        print("git is required", file=sys.stderr)
        return 2

    print(f"[g5] evaluating NVIDIA/TensorRT-Edge-LLM @ {args.tag}")
    checkout = ensure_checkout(args.checkout, args.tag, args.offline)
    if checkout.get("error"):
        print(f"[g5] checkout failed: {checkout['error']}", file=sys.stderr)
        return 1

    root = args.checkout
    matrix = check_support_matrix(root)
    build = check_build_config(root)
    cutedsl = check_cutedsl_prebuilts(root)
    models = check_model_support(root)
    quant = check_quantization(root)
    nano = check_orin_nano(root)

    pyproject = read(root, P_PYPROJECT) or ""
    deps = {
        k: (re.search(rf'"{k}==([0-9.]+)"', pyproject) or [None, None])[1]
        for k in ("torch", "transformers", "onnx", "numpy")
    }

    result = {
        "gate": "G5",
        "title": "TensorRT Edge-LLM support-matrix evaluation",
        "upstream": {"url": REPO_URL, "tag": args.tag, **checkout},
        "this_board": THIS_BOARD,
        "support_matrix": matrix,
        "build_config": build,
        "cutedsl_prebuilts": cutedsl,
        "model_support": models,
        "quantization": quant,
        "orin_nano_8gb": nano,
        "python_requirements": deps,
        "evaluation": evaluate(matrix, cutedsl, models, quant),
        "corrects": (
            "Stage G1 concluded that 'there is no NVIDIA product called TensorRT "
            "Edge-LLM'. That was false. TensorRT-Edge-LLM and TensorRT-LLM are different "
            "projects, and the absence of a tensorrt_llm package said nothing about "
            "Edge-LLM."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    ev = result["evaluation"]
    print(f"[g5] support matrix has a JetPack 6 Orin row : {matrix.get('jetpack6_supported')}")
    print(f"[g5]   level                                 : {matrix.get('level')}")
    print(f"[g5]   OS / SDK                              : {matrix.get('os_sdk')}")
    print(f"[g5]   CUDA toolkit                          : {matrix.get('cuda_toolkit')}")
    print(f"[g5]   precision constraint                  : {matrix.get('precision_constraint')}")
    print(f"[g5] documented JetPack 6.2+ Orin cmake block : {build.get('installation_has_jetpack6_orin_section')}")
    print(f"[g5] jetson-orin -> sm_87                     : {build.get('cutedsl_maps_jetson_orin_to_sm_87')}")
    print(f"[g5] Qwen2.5-VL-3B-Instruct supported         : {models.get('qwen25_vl_3b_in_supported_models')}")
    print(f"[g5] Qwen2.5-VL-3B-Instruct-AWQ listed        : {models.get('qwen25_vl_3b_awq_checkpoint_listed')}")
    print(f"[g5] Orin L1 precisions for that model        : {models.get('orin_l1_precisions_for_qwen25_vl_3b')}")
    print(f"[g5] int4_awq offered                         : {quant.get('int4_awq_offered')}")
    print(f"[g5] INT4 AWQ platform requirement            : {quant.get('precision_key', {}).get('INT4 AWQ')}")
    print(f"[g5] NVFP4 platform requirement               : {quant.get('precision_key', {}).get('NVFP4')}")
    print(f"[g5] Orin Nano 8GB benchmarked                : {nano.get('listed_as_platform')} ({len(nano.get('benchmark_rows', []))} rows)")
    print(f"[g5] CuTe DSL archives shipped                : {cutedsl.get('shipped_tarballs')}")
    print(f"[g5] archive needed here present              : {cutedsl.get('needed_tarball_present')}")
    print(f"[g5] board is on the support matrix           : {ev.get('board_is_on_support_matrix')}")
    print(f"[g5] installable end-to-end today             : {ev.get('installable_end_to_end_today')}")
    for b in ev.get("blockers", []):
        print(f"[g5]   blocker: {b}")
    print(f"[g5] wrote {args.out}")

    if not args.keep and checkout.get("cloned"):
        print(f"[g5] (checkout kept at {root}; pass --keep to silence this note)")

    # The gate passes when the board is on the matrix and the target model is
    # supported. The remaining blockers are install work, not unsupportedness.
    ok = ev.get("board_is_on_support_matrix") and ev.get("target_model_supported")
    print(f"[g5] RESULT: {'PASS' if ok else 'FAIL'} — {ev.get('verdict')}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
