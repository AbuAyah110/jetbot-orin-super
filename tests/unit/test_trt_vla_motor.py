"""SmolVLA TRT stub: importable without GPU/engine; dummy infer stays StageNotReady."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jetbot_agent._stage import StageNotReady
from jetbot_agent.engine.trt_vla_motor import (
    INPUT_CAMERA1,
    OUTPUT_ACTION,
    TRT8_FORBIDDEN_APIS,
    TrtVlaMotor,
    expected_io,
)

STUB_PATH = ROOT / "jetbot_agent" / "engine" / "trt_vla_motor.py"
EXPORT_PATH = ROOT / "scripts" / "bringup" / "export_smolvla_onnx.py"


def test_stub_imports_without_tensorrt_or_engine():
    motor = TrtVlaMotor()
    assert motor.name == "smolvla"
    assert motor.ready() is False
    assert motor.engine_path is None


def test_infer_without_engine_is_stage_not_ready():
    motor = TrtVlaMotor()
    with pytest.raises(StageNotReady) as excinfo:
        motor.infer()
    msg = str(excinfo.value)
    assert "#30" in msg or "torch" in msg.lower() or "engine" in msg.lower()
    assert "PWM" not in msg or "no PWM" in msg or "motors" in msg.lower() or "flow-matching" in msg


def test_missing_engine_path_is_not_ready(tmp_path):
    motor = TrtVlaMotor(tmp_path / "nope.engine")
    assert motor.ready() is False
    with pytest.raises(StageNotReady):
        motor.infer(motor.dummy_observation())


def test_expected_io_is_smolvla_base_not_gemini_224():
    spec = expected_io()
    assert spec["images"]["camera1"] == [1, 3, 256, 256]
    assert spec["images"]["policy_resize_with_pad"] == [512, 512]
    assert spec["language"]["tokens"] == [1, 48]
    assert spec["action"]["padded_chunk"] == [1, 50, 32]
    assert spec["action"]["num_flow_steps"] == 10
    assert spec["named_tensors"]["outputs"] == [OUTPUT_ACTION]
    assert INPUT_CAMERA1 in spec["named_tensors"]["inputs"]
    # Gemini placeholders must not leak in as the contract.
    assert spec["images"]["camera1"][2:] != [224, 224]
    assert spec["language"]["tokens"][1] != 16


def test_dummy_observation_shapes():
    obs = TrtVlaMotor().dummy_observation()
    assert obs[INPUT_CAMERA1].shape == (1, 3, 256, 256)
    assert obs["observation.language_tokens"].shape == (1, 48)
    assert obs["observation.state"].shape == (1, 32)


def test_stub_source_does_not_call_trt8_binding_api():
    tree = ast.parse(STUB_PATH.read_text())
    called = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
    }
    for name in TRT8_FORBIDDEN_APIS:
        assert name not in called, f"stub must not call TensorRT 8 API {name}"
    # Positive: the source documents the TRT 10 enqueue.
    text = STUB_PATH.read_text()
    assert "execute_async_v3" in text
    assert "set_tensor_address" in text
    assert "num_io_tensors" in text
    assert "pycuda" not in text.lower() or "not pycuda" in text.lower() or "PyCUDA" in text


def test_stub_does_not_import_motor_stacks():
    tree = ast.parse(STUB_PATH.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    for name in ("smbus", "smbus2", "Adafruit_PCA9685", "board", "busio"):
        assert name not in imported


def test_export_script_refuses_without_torch():
    proc = subprocess.run(
        [sys.executable, str(EXPORT_PATH), "--graph", "policy"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    blob = proc.stdout + proc.stderr
    assert "refuses to run" in blob or "Refusing to torch.onnx.export" in blob
    assert "#30" in blob or "torch" in blob.lower()
    assert "224" in blob  # documents Gemini mismatch
    assert "wheel" in blob.lower()
