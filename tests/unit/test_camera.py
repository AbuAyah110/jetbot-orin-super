from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from perception import CameraService, FakeCamera, FrameBuffer, MotionDetector
from perception.camera import create_camera


def test_frame_buffer_ring():
    buf = FrameBuffer(maxlen=3)
    for i in range(5):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[:] = i
        buf.push(img, source='t')
    assert len(buf) == 3
    assert buf.latest() is not None
    assert buf.latest().sequence == 5
    assert [f.sequence for f in buf.recent()] == [3, 4, 5]


def test_fake_camera_reads():
    cam = FakeCamera(width=64, height=48)
    cam.open()
    a = cam.read()
    b = cam.read()
    cam.close()
    assert a.shape == (48, 64, 3)
    assert b.shape == (48, 64, 3)
    # moving bar should change pixels across frames
    assert not np.array_equal(a, b)


def test_motion_detects_change():
    det = MotionDetector(threshold=1.0, blur_ksize=0, resize=(32, 24))
    a = np.zeros((48, 64, 3), dtype=np.uint8)
    b = np.zeros((48, 64, 3), dtype=np.uint8)
    b[:] = 255
    first = det.detect(a)
    assert first.score == 0.0
    second = det.detect(b)
    assert second.changed is True
    assert second.score > 1.0


def test_camera_service_capture_save(tmp_path):
    svc = CameraService(backend=FakeCamera(width=80, height=60), buffer_size=4, auto_open=True)
    frame = svc.capture_frame()
    assert frame.image.shape == (60, 80, 3)
    out = svc.save_frame(tmp_path / 'shot.jpg')
    assert out.exists()
    assert out.stat().st_size > 0
    latest = svc.get_latest_frame()
    assert latest is not None
    assert latest.sequence == frame.sequence
    svc.close()


def test_camera_service_motion_over_frames():
    svc = CameraService(
        backend=FakeCamera(width=100, height=80, moving_bar=True),
        motion_threshold=0.5,
    )
    svc._motion = MotionDetector(threshold=0.5, blur_ksize=0, resize=(64, 48))
    changed = False
    scores = []
    for _ in range(15):
        frame = svc.capture_frame()
        result = svc.detect_change(frame.image)
        scores.append(result.score)
        if result.changed:
            changed = True
            break
    svc.close()
    assert changed, 'scores={0}'.format(scores)


def test_create_camera_factory():
    cam = create_camera('fake', width=32, height=24)
    assert cam.name == 'fake'
    with pytest.raises(ValueError):
        create_camera('nope')


def test_buffer_rejects_bad_image():
    buf = FrameBuffer()
    with pytest.raises(ValueError):
        buf.push(np.zeros((10, 10), dtype=np.uint8))
