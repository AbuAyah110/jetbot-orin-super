"""SmolVLA TensorRT runtime stub (Stage G3). Dummy I/O only; no PWM.

This module is importable without TensorRT, PyCUDA, an engine file, or a GPU.
Construction without an engine is allowed. Calling :meth:`infer` without a
deserialized TensorRT 10 engine raises :class:`StageNotReady`.

The Gemini plan used TensorRT 8 binding APIs (``engine[binding]`` as a buffer
slot, ``get_binding_shape``, ``max_batch_size``, ``execute_async_v2``). This
board has TensorRT **10.3.0**. G1 already ran named tensors +
``execute_async_v3`` via ctypes ``libcudart``. This stub follows that path, not
PyCUDA (absent here; a PyPI ``pycuda``/``tensorrt`` wheel is the wrong ABI).

A future engine is **not** ``image + text_ids → left/right wheel speed``.
``lerobot/smolvla_base`` emits a (chunk, 6) action vector after 10 Euler
flow-matching steps. Mapping that onto JetBot ``cmd_vel`` is a later adapter
job. This module never opens I2C / PCA9685 / ``/dev/snd``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

from jetbot_agent._stage import StageNotReady

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _ROOT / "scripts" / "bringup" / "smolvla_base.config.json"

# TensorRT 8 names that must not be used against TensorRT 10.3 on this board.
TRT8_FORBIDDEN_APIS = (
    "get_binding_shape",
    "get_binding_index",
    "max_batch_size",
    "num_bindings",
    "execute_async_v2",
    "set_binding_shape",
)

# Named tensors a *future* prefix or denoise engine should use. Not Gemini's 224/16.
INPUT_CAMERA1 = "observation.images.camera1"
INPUT_STATE = "observation.state"
INPUT_LANG_TOKENS = "observation.language_tokens"
INPUT_LANG_MASK = "observation.language_attention_mask"
OUTPUT_ACTION = "action"

_NOT_READY = (
    "SmolVLA TensorRT dummy forward is not reachable yet: no ONNX/engine is "
    "published, torch+lerobot are not installed (#30), and a single-graph "
    "export of SmolVLAPolicy is the wrong unit (flow-matching loop of "
    "num_steps=10 plus a KV-cached action expert). See docs/bringup/07-smolvla-trt.md."
)


def _load_io_spec() -> dict[str, Any]:
    if _CONFIG_PATH.is_file():
        return json.loads(_CONFIG_PATH.read_text())
    return {
        "input_features": {
            INPUT_CAMERA1: {"shape": [3, 256, 256]},
            INPUT_STATE: {"shape": [6]},
        },
        "output_features": {OUTPUT_ACTION: {"shape": [6]}},
        "tokenizer_max_length": 48,
        "chunk_size": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "resize_imgs_with_padding": [512, 512],
        "num_steps": 10,
    }


def expected_io() -> dict[str, Any]:
    """Shapes taken from ``lerobot/smolvla_base`` ``config.json``, not Gemini."""
    cfg = _load_io_spec()
    return {
        "checkpoint": "lerobot/smolvla_base",
        "images": {
            "camera1": [1, 3, 256, 256],
            "camera2": [1, 3, 256, 256],
            "camera3": [1, 3, 256, 256],
            "policy_resize_with_pad": list(cfg.get("resize_imgs_with_padding", [512, 512])),
            "note": "Checkpoint feature size is 256²; LeRobot then resize_with_pad to 512². Not 224².",
        },
        "state": {
            "raw": list(cfg["input_features"]["observation.state"]["shape"]),
            "padded": [1, int(cfg.get("max_state_dim", 32))],
        },
        "language": {
            "tokens": [1, int(cfg.get("tokenizer_max_length", 48))],
            "attention_mask": [1, int(cfg.get("tokenizer_max_length", 48))],
            "note": "Not 1×16. pad_language_to=max_length, tokenizer_max_length=48.",
        },
        "action": {
            "raw_dim": list(cfg["output_features"]["action"]["shape"]),
            "padded_chunk": [1, int(cfg.get("chunk_size", 50)), int(cfg.get("max_action_dim", 32))],
            "num_flow_steps": int(cfg.get("num_steps", 10)),
            "note": "SO-100-style 6-D action chunk, not left/right wheel PWM.",
        },
        "named_tensors": {
            "inputs": [INPUT_CAMERA1, INPUT_STATE, INPUT_LANG_TOKENS, INPUT_LANG_MASK],
            "outputs": [OUTPUT_ACTION],
        },
    }


class TrtVlaMotor:
    """Named-tensor TensorRT 10 runner. Safe to construct without an engine."""

    name = "smolvla"

    def __init__(
        self,
        engine_path: Optional[str | Path] = None,
        *,
        pythonpath_hint: str = "/usr/lib/python3.10/dist-packages",
    ) -> None:
        self.engine_path = Path(engine_path) if engine_path else None
        self.pythonpath_hint = pythonpath_hint
        self._engine = None
        self._context = None
        self._trt = None

    def ready(self) -> bool:
        return bool(self.engine_path and self.engine_path.is_file())

    def dummy_observation(self) -> dict[str, Any]:
        """Host-side dummy tensors matching ``smolvla_base`` I/O. No camera, no GPU."""
        spec = expected_io()
        try:
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise StageNotReady("numpy is required to build dummy SmolVLA tensors") from exc
        cam = spec["images"]["camera1"]
        tok = spec["language"]["tokens"]
        state_pad = spec["state"]["padded"]
        return {
            INPUT_CAMERA1: np.zeros(cam, dtype=np.float32),
            INPUT_STATE: np.zeros(state_pad, dtype=np.float32),
            INPUT_LANG_TOKENS: np.zeros(tok, dtype=np.int64),
            INPUT_LANG_MASK: np.ones(tok, dtype=np.int32),
        }

    def infer(self, observation: Optional[Mapping[str, Any]] = None) -> Any:
        """Run one dummy forward. Raises until an engine exists (#18 still blocked)."""
        del observation  # unused until an engine exists
        if not self.ready():
            raise StageNotReady(_NOT_READY)
        return self._infer_trt10()

    def _infer_trt10(self) -> Any:
        """TensorRT 10 named-I/O path (execute_async_v3). Never TRT 8 bindings."""
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise StageNotReady(
                "tensorrt Python bindings are the apt package python3-libnvinfer. "
                f"Import from the .venv needs PYTHONPATH={self.pythonpath_hint}"
            ) from exc

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if engine is None:
            raise StageNotReady(f"failed to deserialize engine at {self.engine_path}")
        # Named I/O (TRT 10). Do not use engine[i] as a device pointer, and do
        # not call execute_async_v2. G1: set_tensor_address + execute_async_v3.
        _context = engine.create_execution_context()
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        raise StageNotReady(
            "Engine deserialized but SmolVLA dummy I/O is still a stub: a real "
            "forward needs ctypes libcudart buffers, set_tensor_address, and "
            f"execute_async_v3 (G1). I/O tensors: {names}. No motors. "
            f"context={type(_context).__name__}"
        )


__all__ = [
    "INPUT_CAMERA1",
    "INPUT_LANG_MASK",
    "INPUT_LANG_TOKENS",
    "INPUT_STATE",
    "OUTPUT_ACTION",
    "TRT8_FORBIDDEN_APIS",
    "TrtVlaMotor",
    "expected_io",
]
