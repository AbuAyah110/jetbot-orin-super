from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque, Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class Frame:
    """One BGR8 image with capture metadata."""

    image: np.ndarray
    timestamp: datetime
    source: str = ''
    sequence: int = 0

    @property
    def shape(self):
        return self.image.shape


class FrameBuffer:
    """Fixed-size ring buffer of recent frames (keeps RAM bounded)."""

    def __init__(self, maxlen: int = 5) -> None:
        if maxlen < 1:
            raise ValueError('maxlen must be >= 1')
        self._frames: Deque[Frame] = deque(maxlen=maxlen)
        self._sequence = 0

    @property
    def maxlen(self) -> int:
        return self._frames.maxlen or 1

    def __len__(self) -> int:
        return len(self._frames)

    def clear(self) -> None:
        self._frames.clear()

    def push(self, image: np.ndarray, source: str = '', timestamp: Optional[datetime] = None) -> Frame:
        if image is None or not isinstance(image, np.ndarray):
            raise TypeError('image must be a numpy ndarray')
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('image must be HxWx3 BGR')
        stamp = timestamp or datetime.now(timezone.utc)
        self._sequence += 1
        frame = Frame(
            image=np.ascontiguousarray(image),
            timestamp=stamp,
            source=source,
            sequence=self._sequence,
        )
        self._frames.append(frame)
        return frame

    def latest(self) -> Optional[Frame]:
        if not self._frames:
            return None
        return self._frames[-1]

    def recent(self, n: Optional[int] = None) -> List[Frame]:
        if n is None:
            return list(self._frames)
        if n < 1:
            return []
        return list(self._frames)[-n:]

    def __iter__(self) -> Iterable[Frame]:
        return iter(self._frames)
