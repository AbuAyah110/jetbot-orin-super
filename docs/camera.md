# Camera (Milestone 2)

Mock-first perception for JetBot Orin Super. CSI GStreamer is stubbed for Jetson; laptop work uses `fake`, `file`, or `webcam`.

## Rules (from PROJECT_PLAN)

- No YOLO / neural depth in this milestone.
- Do **not** continuously send frames to Cosmos.
- Keep a small ring buffer only.
- RGB camera is for semantic vision later — not sole collision safety.

## Python API

```python
from perception import CameraService

with CameraService.from_config({'camera': {'backend': 'fake'}}) as cam:
    frame = cam.capture_frame()
    path = cam.save_frame('data/images/shot.jpg')
    motion = cam.detect_change()
    print(cam.backend_name, frame.shape, path, motion)
```

Helpers:

| Method | Purpose |
| --- | --- |
| `capture_frame()` | Grab one frame into the ring buffer |
| `get_latest_frame()` | Last buffered frame (or `None`) |
| `save_frame(path)` | Write JPEG/PNG (PPM fallback without OpenCV) |
| `detect_change()` | Cheap grayscale absdiff motion score |

## Backends

| Backend | When |
| --- | --- |
| `fake` | Unit tests / CI (synthetic moving bar) |
| `file` | Replay a still or video |
| `webcam` | Laptop USB camera |
| `gst_csi` | Jetson Orin CSI via `nvarguscamerasrc` |

Config: [`config/camera.yaml`](../config/camera.yaml).

## CLI demo (laptop)

```bash
PYTHONPATH=src python3 scripts/demo_camera.py
PYTHONPATH=src python3 scripts/demo_camera.py --backend fake --frames 20 --out data/images/demo.jpg
# with OpenCV + webcam:
PYTHONPATH=src python3 scripts/demo_camera.py --backend webcam --device 0
```

## Jetson later

1. Set `backend: gst_csi` in `config/camera.yaml`.
2. Confirm CSI ribbon / `nvarguscamerasrc`.
3. Optional: ROS 2 `/camera/image_raw` publisher (not required for this mock milestone).

## Tests

```bash
PYTHONPATH=src python3 -m pytest tests/unit/test_camera.py -q
```
