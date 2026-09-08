from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "camera_viewer", ROOT / "scripts" / "bringup" / "camera_viewer.py"
)
assert SPEC is not None and SPEC.loader is not None
camera_viewer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(camera_viewer)


def synthetic_flat_field() -> np.ndarray:
    height = width = 96
    yy, xx = np.mgrid[:height, :width]
    radius = np.sqrt(
        ((xx - width / 2) / (width / 2)) ** 2
        + ((yy - height / 2) / (height / 2)) ** 2
    )
    falloff = np.clip(1.0 - 0.4 * radius, 0.45, 1.0)
    # Deliberate magenta edge cast: red falls less than green and blue.
    red = 190.0 * np.clip(1.0 - 0.15 * radius, 0.7, 1.0)
    green = 190.0 * falloff
    blue = 190.0 * falloff
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def test_flatfield_gain_reduces_corner_shading_and_colour_cast():
    raw = synthetic_flat_field()
    frames = [raw.copy() for _ in range(8)]

    gain, metadata = camera_viewer.build_flatfield_gain(frames, blur_radius=8)
    corrected = camera_viewer.apply_flatfield(raw, gain)
    metrics = camera_viewer.flatfield_metrics(raw, corrected)

    assert gain.shape == raw.shape
    assert metadata["frames"] == 8
    assert (
        metrics["corrected"]["corner_to_centre_luma"]
        > metrics["raw"]["corner_to_centre_luma"]
    )
    assert (
        metrics["corrected"]["corner_channel_spread"]
        < metrics["raw"]["corner_channel_spread"]
    )


def test_calibration_round_trips_without_pickle(tmp_path):
    raw = synthetic_flat_field()
    gain, metadata = camera_viewer.build_flatfield_gain(
        [raw.copy() for _ in range(4)], blur_radius=5
    )
    path = tmp_path / "flatfield.npz"

    camera_viewer.save_calibration(path, gain, metadata)
    loaded_gain, loaded_metadata = camera_viewer.load_calibration(path)

    assert loaded_gain is not None
    np.testing.assert_allclose(loaded_gain, gain)
    assert loaded_metadata["frames"] == 4


def test_dark_or_wrong_shaped_flat_field_is_rejected():
    dark = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too dark"):
        camera_viewer.build_flatfield_gain([dark, dark, dark], blur_radius=2)

    with pytest.raises(ValueError, match="share one"):
        camera_viewer.build_flatfield_gain(
            [np.zeros((32, 32, 3)), np.zeros((16, 16, 3)), np.zeros((32, 32, 3))]
        )


def test_apply_rejects_calibration_for_a_different_resolution():
    with pytest.raises(ValueError, match="dimensions differ"):
        camera_viewer.apply_flatfield(
            np.zeros((32, 32, 3), dtype=np.uint8),
            np.ones((16, 16, 3), dtype=np.float32),
        )
