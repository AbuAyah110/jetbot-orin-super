from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class MotionResult:
    changed: bool
    score: float
    threshold: float


class MotionDetector:
    """Cheap frame-change detector (grayscale absdiff + mean). No ML."""

    def __init__(
        self,
        threshold: float = 8.0,
        blur_ksize: int = 5,
        resize: Optional[Tuple[int, int]] = (160, 120),
    ) -> None:
        self.threshold = float(threshold)
        self.blur_ksize = int(blur_ksize) if blur_ksize and blur_ksize > 0 else 0
        self.resize = resize
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev_gray = None

    def _to_gray(self, image: np.ndarray) -> np.ndarray:
        # Prefer OpenCV when available; fall back to numpy for tiny test envs.
        try:
            import cv2
        except ImportError:
            if image.ndim == 3:
                # BGR → gray approximation
                b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
                gray = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
            else:
                gray = image.astype(np.uint8)
            if self.resize is not None:
                # nearest-neighbor downsample without cv2
                h, w = self.resize[1], self.resize[0]
                ys = (np.linspace(0, gray.shape[0] - 1, h)).astype(np.int32)
                xs = (np.linspace(0, gray.shape[1] - 1, w)).astype(np.int32)
                gray = gray[ys][:, xs]
            return gray

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        if self.resize is not None:
            gray = cv2.resize(gray, self.resize, interpolation=cv2.INTER_AREA)
        if self.blur_ksize >= 3:
            k = self.blur_ksize if self.blur_ksize % 2 == 1 else self.blur_ksize + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        return gray

    def score(self, image: np.ndarray) -> float:
        gray = self._to_gray(image)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        if gray.shape != self._prev_gray.shape:
            self._prev_gray = gray
            return 0.0
        diff = np.abs(gray.astype(np.int16) - self._prev_gray.astype(np.int16))
        value = float(np.mean(diff))
        self._prev_gray = gray
        return value

    def detect(self, image: np.ndarray) -> MotionResult:
        value = self.score(image)
        return MotionResult(
            changed=value >= self.threshold,
            score=value,
            threshold=self.threshold,
        )
