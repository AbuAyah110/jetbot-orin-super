"""Camera / perception package (Milestone 2). Mock-first; CSI GStreamer stub for Jetson."""

from perception.camera import (
    CameraBackend,
    FakeCamera,
    FileCamera,
    WebcamCamera,
    GstCsiCamera,
    create_camera,
)
from perception.frame_buffer import FrameBuffer
from perception.motion_detector import MotionDetector
from perception.service import CameraService

__all__ = [
    'CameraBackend',
    'FakeCamera',
    'FileCamera',
    'WebcamCamera',
    'GstCsiCamera',
    'create_camera',
    'FrameBuffer',
    'MotionDetector',
    'CameraService',
]
