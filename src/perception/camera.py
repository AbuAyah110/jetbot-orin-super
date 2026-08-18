from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

import numpy as np


class CameraBackend(ABC):
    """Hardware-independent camera capture interface (BGR8 uint8)."""

    @abstractmethod
    def open(self) -> None:
        ...

    @abstractmethod
    def read(self) -> np.ndarray:
        """Return one BGR frame. Raises RuntimeError if capture fails."""

    @abstractmethod
    def close(self) -> None:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class FakeCamera(CameraBackend):
    """Synthetic frames for unit tests and laptop bring-up without a sensor."""

    def __init__(
        self,
        width: int = 320,
        height: int = 240,
        color: Tuple[int, int, int] = (40, 120, 200),
        moving_bar: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.color = color
        self.moving_bar = moving_bar
        self._frame_idx = 0
        self._opened = False

    @property
    def name(self) -> str:
        return 'fake'

    def open(self) -> None:
        self._opened = True
        self._frame_idx = 0

    def read(self) -> np.ndarray:
        if not self._opened:
            raise RuntimeError('FakeCamera is not open')
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        image[:, :] = self.color
        if self.moving_bar:
            step = max(self.width // 8, 8)
            x = (self._frame_idx * step) % max(self.width - 20, 1)
            image[:, x : x + 20] = (0, 255, 0)
        self._frame_idx += 1
        return image

    def close(self) -> None:
        self._opened = False


class FileCamera(CameraBackend):
    """Replay a still image or video file (OpenCV)."""

    def __init__(self, path: str | Path, loop: bool = True) -> None:
        self.path = Path(path)
        self.loop = loop
        self._cap = None
        self._still: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return 'file:{0}'.format(self.path.name)

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('OpenCV (cv2) is required for FileCamera') from exc
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        suffix = self.path.suffix.lower()
        if suffix in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
            image = cv2.imread(str(self.path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError('Failed to read image {0}'.format(self.path))
            self._still = image
            self._cap = None
            return
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise RuntimeError('Failed to open video {0}'.format(self.path))

    def read(self) -> np.ndarray:
        if self._still is not None:
            return self._still.copy()
        if self._cap is None:
            raise RuntimeError('FileCamera is not open')
        ok, frame = self._cap.read()
        if not ok or frame is None:
            if self.loop:
                self._cap.set(0, 0)  # CAP_PROP_POS_FRAMES
                ok, frame = self._cap.read()
            if not ok or frame is None:
                raise RuntimeError('End of video / read failed')
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._still = None


class WebcamCamera(CameraBackend):
    """Laptop / USB webcam via OpenCV VideoCapture index."""

    def __init__(self, device_index: int = 0, width: Optional[int] = None, height: Optional[int] = None) -> None:
        self.device_index = device_index
        self.width = width
        self.height = height
        self._cap = None

    @property
    def name(self) -> str:
        return 'webcam:{0}'.format(self.device_index)

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('OpenCV (cv2) is required for WebcamCamera') from exc
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            raise RuntimeError('Could not open webcam index {0}'.format(self.device_index))
        if self.width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def read(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError('WebcamCamera is not open')
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError('Webcam read failed')
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class GstCsiCamera(CameraBackend):
    """Jetson CSI via nvarguscamerasrc (GStreamer). Not used on laptop."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        capture_width: int = 1280,
        capture_height: int = 720,
        fps: int = 30,
        sensor_id: int = 0,
    ) -> None:
        self.width = width
        self.height = height
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.fps = fps
        self.sensor_id = sensor_id
        self._cap = None

    @property
    def name(self) -> str:
        return 'gst_csi:{0}'.format(self.sensor_id)

    def _gst_pipeline(self) -> str:
        return (
            'nvarguscamerasrc sensor-id={sensor} ! '
            'video/x-raw(memory:NVMM), width={cw}, height={ch}, '
            'format=NV12, framerate={fps}/1 ! '
            'nvvidconv ! video/x-raw, width={w}, height={h}, format=BGRx ! '
            'videoconvert ! video/x-raw, format=BGR ! appsink drop=1'
        ).format(
            sensor=self.sensor_id,
            cw=self.capture_width,
            ch=self.capture_height,
            fps=self.fps,
            w=self.width,
            h=self.height,
        )

    def open(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError('OpenCV with GStreamer is required for GstCsiCamera') from exc
        pipeline = self._gst_pipeline()
        self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not self._cap.isOpened():
            raise RuntimeError(
                'Failed to open CSI GStreamer pipeline. '
                'Use backend=fake/file/webcam on non-Jetson hosts. Pipeline={0}'.format(pipeline)
            )

    def read(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError('GstCsiCamera is not open')
        ok, frame = self._cap.read()
        if not ok or frame is None:
            raise RuntimeError('CSI camera read failed')
        return frame

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def create_camera(backend: str = 'fake', **kwargs: Any) -> CameraBackend:
    name = (backend or 'fake').strip().lower()
    if name in ('fake', 'mock', 'synthetic'):
        return FakeCamera(
            width=int(kwargs.get('width', 320)),
            height=int(kwargs.get('height', 240)),
            moving_bar=bool(kwargs.get('moving_bar', True)),
        )
    if name in ('file', 'image', 'video'):
        path = kwargs.get('path') or kwargs.get('file')
        if not path:
            raise ValueError('file camera requires path=')
        return FileCamera(path, loop=bool(kwargs.get('loop', True)))
    if name in ('webcam', 'usb', 'v4l2'):
        return WebcamCamera(
            device_index=int(kwargs.get('device_index', kwargs.get('index', 0))),
            width=kwargs.get('width'),
            height=kwargs.get('height'),
        )
    if name in ('gst_csi', 'csi', 'argus', 'jetson'):
        return GstCsiCamera(
            width=int(kwargs.get('width', 640)),
            height=int(kwargs.get('height', 480)),
            capture_width=int(kwargs.get('capture_width', 1280)),
            capture_height=int(kwargs.get('capture_height', 720)),
            fps=int(kwargs.get('fps', 30)),
            sensor_id=int(kwargs.get('sensor_id', 0)),
        )
    raise ValueError('Unknown camera backend: {0}'.format(backend))


def create_camera_from_config(config: Mapping[str, Any]) -> CameraBackend:
    cam = config.get('camera', config)
    return create_camera(
        backend=cam.get('backend', 'fake'),
        path=cam.get('path'),
        device_index=cam.get('device_index', 0),
        width=cam.get('width', 320),
        height=cam.get('height', 240),
        capture_width=cam.get('capture_width', 1280),
        capture_height=cam.get('capture_height', 720),
        fps=cam.get('fps', 30),
        sensor_id=cam.get('sensor_id', 0),
        moving_bar=cam.get('moving_bar', True),
        loop=cam.get('loop', True),
    )
