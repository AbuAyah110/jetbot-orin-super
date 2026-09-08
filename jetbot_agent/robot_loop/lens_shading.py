"""Application-level lens shading correction for the B0392 IMX219.

Jetson Argus cannot consume Arducam's Raspberry Pi/libcamera tuning JSON.
This module applies a locally captured flat-field gain map to the JPEG that
Cosmos, colour grounding, and the camera check viewer consume.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_PATH = REPO_ROOT / "data" / "camera" / "b0392_flatfield.npz"
GAIN_MIN = 0.5
GAIN_MAX = 2.5


def decode_jpeg(jpeg: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), dtype=np.uint8)


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    output = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8), "RGB").save(
        output, format="JPEG", quality=quality
    )
    return output.getvalue()


def build_flatfield_gain(
    frames: list[np.ndarray],
    *,
    blur_radius: float = 24.0,
) -> tuple[np.ndarray, dict]:
    """Build a smooth RGB gain map from a uniformly lit neutral field."""
    if len(frames) < 3:
        raise ValueError("at least three flat-field frames are required")
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1 or next(iter(shapes))[-1] != 3:
        raise ValueError("flat-field frames must share one HxWx3 shape")

    average = np.mean(np.stack(frames).astype(np.float32), axis=0)
    smooth = np.asarray(
        Image.fromarray(np.clip(average, 0, 255).astype(np.uint8), "RGB").filter(
            ImageFilter.GaussianBlur(radius=float(blur_radius))
        ),
        dtype=np.float32,
    )
    height, width, _ = smooth.shape
    y0, y1 = int(height * 0.4), int(height * 0.6)
    x0, x1 = int(width * 0.4), int(width * 0.6)
    reference = np.median(smooth[y0:y1, x0:x1], axis=(0, 1))
    if np.any(reference < 16.0):
        raise ValueError(
            "flat field is too dark; use a bright, uniformly lit white surface"
        )

    gain = reference.reshape(1, 1, 3) / np.maximum(smooth, 1.0)
    gain = np.clip(gain, GAIN_MIN, GAIN_MAX).astype(np.float32)
    corrected = apply_flatfield(average, gain)
    metrics = flatfield_metrics(average, corrected)
    metrics.update(
        {
            "frames": len(frames),
            "width": width,
            "height": height,
            "center_rgb": [round(float(value), 2) for value in reference],
            "gain_min": round(float(gain.min()), 4),
            "gain_max": round(float(gain.max()), 4),
            "created_unix_s": time.time(),
        }
    )
    return gain, metrics


def apply_flatfield(rgb: np.ndarray, gain: np.ndarray) -> np.ndarray:
    if tuple(rgb.shape) != tuple(gain.shape):
        raise ValueError("RGB frame and flat-field gain dimensions differ")
    return np.clip(rgb.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def correct_jpeg(jpeg: bytes, gain: np.ndarray, quality: int = 90) -> bytes:
    """Decode, apply the calibrated gain, and return one corrected JPEG."""
    return encode_jpeg(apply_flatfield(decode_jpeg(jpeg), gain), quality=quality)


def flatfield_metrics(raw: np.ndarray, corrected: np.ndarray) -> dict:
    """Report edge/centre luma and colour imbalance before and after."""

    def sample(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height, width, _ = image.shape
        cy0, cy1 = int(height * 0.4), int(height * 0.6)
        cx0, cx1 = int(width * 0.4), int(width * 0.6)
        centre = np.mean(image[cy0:cy1, cx0:cx1], axis=(0, 1))
        band = max(1, int(min(height, width) * 0.12))
        corners = np.concatenate(
            (
                image[:band, :band].reshape(-1, 3),
                image[:band, -band:].reshape(-1, 3),
                image[-band:, :band].reshape(-1, 3),
                image[-band:, -band:].reshape(-1, 3),
            ),
            axis=0,
        )
        return centre, np.mean(corners, axis=0)

    result = {}
    for name, image in (("raw", raw), ("corrected", corrected)):
        centre, corner = sample(image.astype(np.float32))
        centre_luma = float(np.mean(centre))
        corner_luma = float(np.mean(corner))
        result[name] = {
            "centre_rgb": [round(float(value), 2) for value in centre],
            "corner_rgb": [round(float(value), 2) for value in corner],
            "corner_to_centre_luma": round(
                corner_luma / max(centre_luma, 1.0), 4
            ),
            "corner_channel_spread": round(float(np.max(corner) - np.min(corner)), 2),
        }
    return result


def save_calibration(path: Path, gain: np.ndarray, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, gain=gain, metadata=json.dumps(metrics))


def load_calibration(path: Path) -> tuple[np.ndarray | None, dict]:
    if not path.is_file():
        return None, {}
    with np.load(path, allow_pickle=False) as payload:
        gain = payload["gain"].astype(np.float32)
        raw_metadata = str(payload["metadata"]) if "metadata" in payload else "{}"
    try:
        metadata = json.loads(raw_metadata)
    except json.JSONDecodeError:
        metadata = {}
    return gain, metadata
