#!/usr/bin/env python3
"""View the exact 448² Argus output and calibrate B0392 lens shading.

This owns the single CSI pipeline. Stop talk-and-drive before starting it.
The flat-field correction is an application-level fallback for Jetson Argus;
Arducam's rpicam/libcamera tuning JSON is not compatible with this pipeline.
"""

from __future__ import annotations

import argparse
import json
import signal
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(ROOT))
# PyGObject/GStreamer is installed by JetPack under the system interpreter,
# not in the voice venv. Append it after the venv so NumPy/Pillow stay local.
SYSTEM_DIST_PACKAGES = Path("/usr/lib/python3/dist-packages")
if SYSTEM_DIST_PACKAGES.is_dir() and str(SYSTEM_DIST_PACKAGES) not in sys.path:
    sys.path.append(str(SYSTEM_DIST_PACKAGES))

from jetbot_agent.robot_loop.csi_jpeg import CsiJpeg448  # noqa: E402
from jetbot_agent.robot_loop.lens_shading import (  # noqa: E402
    DEFAULT_CALIBRATION_PATH,
    apply_flatfield,
    build_flatfield_gain,
    decode_jpeg,
    encode_jpeg,
    flatfield_metrics,
    load_calibration,
    save_calibration,
)


CALIBRATION_FRAMES = 30


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
    parser.add_argument(
        "--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH
    )
    args = parser.parse_args()

    feed = CameraFeed(
        # Viewer owns correction itself so its raw pane remains truly raw and
        # its corrected pane applies the map exactly once.
        CsiJpeg448(
            sensor_id=args.sensor_id,
            fps=args.fps,
            lens_shading=False,
        ),
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
