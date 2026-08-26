#!/usr/bin/env python3
"""Export helpers for SmolVLA → ONNX. Does **not** run until torch + lerobot exist.

This is not Gemini's one-shot ``torch.onnx.export(policy, (image 1×3×224×224,
text 1×16))``. ``lerobot/smolvla_base`` is a flow-matching policy:

* inputs: up to three CHW cameras (feature 3×256×256, then ``resize_with_pad``
  to 512×512), language tokens (max 48), padded state (32),
* inference: ``embed_prefix`` + VLM KV cache, then a **Python Euler loop** of
  ``num_steps=10`` calling ``denoise_step`` (action expert),
* output: action chunk (50 × 6), **not** left/right wheel speed.

A single ONNX graph of ``SmolVLAPolicy`` / ``sample_actions`` is the wrong
export unit. This script refuses that graph even after #30 lands.

Blocked on GitHub issue #30 (Jetson PyTorch). Do not pip-install torch from
this script — another agent owns the shared ``.venv``. Do not download
multi-GB weights unless ``--download-weights`` is passed explicitly.

Usage::

    PYTHONPATH=. .venv/bin/python scripts/bringup/export_smolvla_onnx.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "scripts" / "bringup" / "smolvla_base.config.json"
CHECKPOINT = "lerobot/smolvla_base"

WHOLE_POLICY_REFUSAL = """\
Refusing to torch.onnx.export the whole SmolVLAPolicy.

sample_actions() is not a static graph:
  1. embed_prefix(images[list], lang_tokens, state) through SmolVLM2
  2. vlm_with_expert.forward(..., use_cache=True) → past_key_values
  3. euler_integrate: Python for-loop, num_steps=10 (config.json)
  4. denoise_step each iteration; past_key_values.crop(prefix_len) between steps

That control flow will not become one TensorRT engine. Export (later, on an
x86 box or on-device with the VLM unloaded) the prefix graph and the
denoise_step graph separately, then keep the Euler loop in Python.
Gemini dummy I/O (1×3×224×224 image, 1×16 token ids, dynamic batch/seq,
wheel speeds) does not match lerobot/smolvla_base config.json.
"""


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def dummy_io(cfg: dict) -> dict:
    """Host numpy shapes for a *correct* dummy batch (B=1). Requires numpy."""
    import numpy as np

    tlen = int(cfg.get("tokenizer_max_length", 48))
    h, w = 256, 256
    return {
        "observation.images.camera1": np.zeros((1, 3, h, w), dtype=np.float32),
        "observation.images.camera2": np.zeros((1, 3, h, w), dtype=np.float32),
        "observation.images.camera3": np.zeros((1, 3, h, w), dtype=np.float32),
        "observation.state": np.zeros((1, 6), dtype=np.float32),
        "observation.language_tokens": np.zeros((1, tlen), dtype=np.int64),
        "observation.language_attention_mask": np.ones((1, tlen), dtype=np.int64),
        "action_chunk_padded": np.zeros(
            (1, int(cfg.get("chunk_size", 50)), int(cfg.get("max_action_dim", 32))),
            dtype=np.float32,
        ),
    }


def _missing(mod: str) -> str | None:
    try:
        __import__(mod)
    except ImportError as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def require_torch_lerobot() -> None:
    errors = []
    for mod in ("torch", "lerobot"):
        err = _missing(mod)
        if err:
            errors.append(f"{mod}: {err}")
    if errors:
        joined = "; ".join(errors)
        raise SystemExit(
            "export_smolvla_onnx.py refuses to run: torch and lerobot are not "
            "installed in this interpreter.\n"
            f"  {joined}\n"
            "This is blocked on GitHub issue #30 / stage-g-pytorch. Do not "
            "pip-install torch from this script (shared .venv, numpy pin for "
            "Stage F). Once Jetson PyTorch + lerobot exist, re-run here; even "
            "then this script will not export the whole policy as one ONNX "
            "graph.\n"
            f"{WHOLE_POLICY_REFUSAL}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--graph",
        choices=("policy", "prefix", "denoise"),
        default="policy",
        help="Which unit to export. 'policy' always refuses (wrong unit).",
    )
    p.add_argument(
        "--checkpoint",
        default=CHECKPOINT,
        help="HF id or local dir. Weights are not downloaded unless --download-weights.",
    )
    p.add_argument(
        "--download-weights",
        action="store_true",
        help="Opt-in: allow HuggingFace to fetch ~0.9 GB safetensors. Default off.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "bringup" / "smolvla" / "smolvla.onnx",
        help="ONNX destination (only used if a subgraph export is attempted).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config()
    print("checkpoint:", args.checkpoint)
    print("config:", CONFIG_PATH)
    print("resize_imgs_with_padding:", cfg.get("resize_imgs_with_padding"))
    print("tokenizer_max_length:", cfg.get("tokenizer_max_length"))
    print("num_steps (Euler):", cfg.get("num_steps"))
    print("chunk_size:", cfg.get("chunk_size"))
    print("cameras: camera1/2/3 @ 3x256x256 (then pad to 512)")
    print("action: (50, 6) unpadded — not wheel PWM")

    if args.graph == "policy":
        require_torch_lerobot()
        print(WHOLE_POLICY_REFUSAL, file=sys.stderr)
        return 2

    require_torch_lerobot()
    if not args.download_weights and not Path(args.checkpoint).exists():
        print(
            "Subgraph export needs local weights or --download-weights. "
            "Refusing to pull multi-GB checkpoints by default.",
            file=sys.stderr,
        )
        return 3

    print(
        f"--graph {args.graph} is the right unit, but the actual "
        "torch.onnx.export call is not implemented until #30 lands and a first "
        "eager forward exists. No trtexec will be started from this script.",
        file=sys.stderr,
    )
    return 4


if __name__ == "__main__":
    sys.exit(main())
