from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from perception.camera import CameraBackend, create_camera, create_camera_from_config
from perception.frame_buffer import Frame, FrameBuffer
from perception.motion_detector import MotionDetector, MotionResult


class CameraService:
    """High-level camera API used by later MCP / Cosmos tools.

    Does not stream to models. Ring buffer stays small.
    """

    def __init__(
        self,
        backend: Optional[CameraBackend] = None,
        buffer_size: int = 5,
        motion_threshold: float = 8.0,
        auto_open: bool = True,
    ) -> None:
        self._backend = backend or create_camera('fake')
        self._buffer = FrameBuffer(maxlen=buffer_size)
        self._motion = MotionDetector(threshold=motion_threshold)
        self._opened = False
        if auto_open:
            self.open()

    @classmethod
    def from_config(cls, config: dict, auto_open: bool = True) -> 'CameraService':
        cam = config.get('camera', config)
        backend = create_camera_from_config(config)
        return cls(
            backend=backend,
            buffer_size=int(cam.get('buffer_size', 5)),
            motion_threshold=float(cam.get('motion_threshold', 8.0)),
            auto_open=auto_open,
        )

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def open(self) -> None:
        if not self._opened:
            self._backend.open()
            self._opened = True

    def close(self) -> None:
        if self._opened:
            self._backend.close()
            self._opened = False

    def __enter__(self) -> 'CameraService':
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def capture_frame(self) -> Frame:
        self.open()
        image = self._backend.read()
        return self._buffer.push(image, source=self._backend.name)

    def get_latest_frame(self) -> Optional[Frame]:
        return self._buffer.latest()

    def save_frame(
        self,
        path: Union[str, Path],
        frame: Optional[Frame] = None,
        jpeg_quality: int = 90,
    ) -> Path:
        target = frame or self.get_latest_frame() or self.capture_frame()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        image = target.image

        try:
            import cv2
        except ImportError:
            # Minimal PPM fallback so tests work without OpenCV.
            if out.suffix.lower() not in {'.ppm', '.pgm'}:
                out = out.with_suffix('.ppm')
            self._write_ppm(out, image)
            return out

        suffix = out.suffix.lower()
        if suffix in {'.jpg', '.jpeg'}:
            ok = cv2.imwrite(
                str(out),
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
        else:
            ok = cv2.imwrite(str(out), image)
        if not ok:
            raise RuntimeError('Failed to write {0}'.format(out))
        return out

    def detect_change(self, image: Optional[np.ndarray] = None) -> MotionResult:
        if image is None:
            frame = self.get_latest_frame() or self.capture_frame()
            image = frame.image
        return self._motion.detect(image)

    def reset_motion(self) -> None:
        self._motion.reset()

    @staticmethod
    def _write_ppm(path: Path, image: np.ndarray) -> None:
        h, w, _ = image.shape
        # PPM expects RGB
        rgb = image[:, :, ::-1]
        header = 'P6\n{0} {1}\n255\n'.format(w, h).encode('ascii')
        with path.open('wb') as handle:
            handle.write(header)
            handle.write(np.ascontiguousarray(rgb).tobytes())
