#!/usr/bin/env python3
"""View the exact 448² Argus output and calibrate B0392 lens shading.

This owns the single CSI pipeline. Stop talk-and-drive before starting it.
The flat-field correction is an application-level fallback for Jetson Argus;
Arducam's rpicam/libcamera tuning JSON is not compatible with this pipeline.
"""

from __future__ import annotations

import argparse
import io
import json
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
# PyGObject/GStreamer is installed by JetPack under the system interpreter,
# not in the voice venv. Append it after the venv so NumPy/Pillow stay local.
SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST_PACKAGES.is_dir() and str(SYSTEM_DIST_PACKAGES) not in sys.path:
    sys.path.append(str(SYSTEM_DIST_PACKAGES))

from jetbot_agent.robot_loop.csi_jpeg import CsiJpeg448  # noqa: E402


DEFAULT_CALIBRATION = ROOT / "data" / "camera" / "b0392_flatfield.npz"
CALIBRATION_FRAMES = 30
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


class CameraFeed:
    def __init__(
        self,
        camera: CsiJpeg448,
        calibration_path: Path,
        *,
        calibration_frames: int = CALIBRATION_FRAMES,
    ) -> None:
        self.camera = camera
        self.calibration_path = calibration_path
        self.calibration_frames = int(calibration_frames)
        self.condition = threading.Condition()
        self.raw_jpeg = b""
        self.corrected_jpeg = b""
        self.sequence = 0
        self.error = ""
        self.running = False
        self.thread: threading.Thread | None = None
        self.gain, self.metadata = load_calibration(calibration_path)
        self.calibrating = False
        self.calibration_collected = 0
        self._calibration_frames: list[np.ndarray] = []

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._run, name="camera-feed", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.running = False
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=3.0)
        self.camera.close()

    def start_calibration(self) -> bool:
        with self.condition:
            if self.calibrating:
                return False
            self.calibrating = True
            self.calibration_collected = 0
            self._calibration_frames = []
            self.error = ""
            return True

    def status(self) -> dict:
        with self.condition:
            return {
                "sequence": self.sequence,
                "error": self.error,
                "calibration_loaded": self.gain is not None,
                "calibration_path": str(self.calibration_path),
                "calibrating": self.calibrating,
                "calibration_collected": self.calibration_collected,
                "calibration_required": self.calibration_frames,
                "metadata": self.metadata,
            }

    def wait_frame(self, sequence: int, corrected: bool) -> tuple[int, bytes]:
        with self.condition:
            self.condition.wait_for(
                lambda: not self.running or self.sequence > sequence,
                timeout=2.0,
            )
            frame = self.corrected_jpeg if corrected else self.raw_jpeg
            return self.sequence, frame

    def _run(self) -> None:
        try:
            self.camera.open()
            while self.running:
                raw_jpeg = self.camera.capture_jpeg()
                rgb = decode_jpeg(raw_jpeg)
                with self.condition:
                    if self.calibrating:
                        self._calibration_frames.append(rgb.copy())
                        self.calibration_collected = len(self._calibration_frames)
                        if self.calibration_collected >= self.calibration_frames:
                            try:
                                gain, metadata = build_flatfield_gain(
                                    self._calibration_frames
                                )
                                save_calibration(
                                    self.calibration_path, gain, metadata
                                )
                                self.gain = gain
                                self.metadata = metadata
                            except Exception as exc:
                                self.error = "calibration failed: {0}".format(exc)
                            self.calibrating = False
                            self._calibration_frames = []
                    corrected_jpeg = raw_jpeg
                    if self.gain is not None:
                        try:
                            corrected_jpeg = encode_jpeg(
                                apply_flatfield(rgb, self.gain)
                            )
                        except Exception as exc:
                            self.error = "correction failed: {0}".format(exc)
                    self.raw_jpeg = raw_jpeg
                    self.corrected_jpeg = corrected_jpeg
                    self.sequence += 1
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = "{0}: {1}".format(type(exc).__name__, exc)
                self.running = False
                self.condition.notify_all()


PAGE = """<!doctype html>
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>JetBot B0392 Camera</title>
<style>
body{font:16px system-ui;background:#111;color:#eee;margin:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:16px}
img{width:100%;max-width:700px;background:#222}
button{font-size:16px;padding:10px 16px} pre{white-space:pre-wrap}
.warn{color:#ffd166}.ok{color:#80ed99}
</style>
<h1>JetBot B0392 — exact 448² camera output</h1>
<p class="warn">Viewer owns the one Argus camera. Talk-and-drive must remain stopped.</p>
<div class="grid">
 <section><h2>Raw Argus</h2><img src="/stream.mjpg?mode=raw"></section>
 <section><h2>Flat-field corrected</h2><img src="/stream.mjpg?mode=corrected"></section>
</div>
<h2>Lens shading calibration</h2>
<p>Fill the entire view with an evenly lit matte white/neutral surface. Avoid
shadows, texture, glare, and the lens touching the surface. Then wait for
exposure to settle and click once. This captures 30 new frames.</p>
<button onclick="calibrate()">Capture flat field</button>
<a href="/snapshot.jpg?mode=raw" style="margin-left:12px;color:#8ecae6">raw snapshot</a>
<a href="/snapshot.jpg?mode=corrected" style="margin-left:12px;color:#8ecae6">corrected snapshot</a>
<pre id="status">loading…</pre>
<script>
async function calibrate(){
 const r=await fetch('/calibrate',{method:'POST'}); document.querySelector('#status').textContent=await r.text();
}
async function poll(){
 try{const r=await fetch('/status.json'); const j=await r.json();
 document.querySelector('#status').textContent=JSON.stringify(j,null,2);}
 catch(e){document.querySelector('#status').textContent=String(e)}
 setTimeout(poll,1000)
} poll();
</script>
"""


def make_handler(feed: CameraFeed):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            print("viewer", self.address_string(), fmt % args, flush=True)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            corrected = query.get("mode", ["raw"])[0] == "corrected"
            if parsed.path == "/":
                self._send(PAGE.encode(), "text/html; charset=utf-8")
            elif parsed.path == "/status.json":
                self._send(
                    json.dumps(feed.status(), indent=2).encode(),
                    "application/json",
                )
            elif parsed.path == "/snapshot.jpg":
                _, frame = feed.wait_frame(-1, corrected)
                if frame:
                    self._send(frame, "image/jpeg")
                else:
                    self._send(
                        json.dumps(feed.status()).encode(),
                        "application/json",
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
            elif parsed.path == "/stream.mjpg":
                self._stream(corrected)
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/calibrate":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if not feed.start_calibration():
                self._send(b"calibration already running\n", "text/plain", 409)
                return
            self._send(
                b"capturing 30 new flat-field frames; keep the white field still\n",
                "text/plain",
                202,
            )

        def _send(
            self,
            body: bytes,
            content_type: str,
            status: int = 200,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _stream(self, corrected: bool) -> None:
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = -1
            try:
                while feed.running:
                    sequence, frame = feed.wait_frame(sequence, corrected)
                    if not frame:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(frame)).encode()
                        + b"\r\n\r\n"
                        + frame
                        + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError):
                pass

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sensor-id", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()

    feed = CameraFeed(
        CsiJpeg448(sensor_id=args.sensor_id, fps=args.fps),
        args.calibration,
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(feed))

    def stop(_signum=None, _frame=None) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    feed.start()
    print(
        "camera_viewer http://0.0.0.0:{0} calibration={1}".format(
            args.port, args.calibration
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        feed.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
