"""JSON action schema for the locked Cosmos robot loop.

Allowed kinds: ``stop`` | ``drive`` | ``speak`` | ``wait`` | ``weather``.
Invalid JSON or an unknown kind becomes ``stop``. Drive velocities are clamped
to ``vx`` 0.22 and ``wz`` 1.0. Duration is a request; the executor (not this
parser) must stop when it elapses. This module never opens motors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

VX_MAX = 0.22
WZ_MAX = 1.0
MAX_DURATION_S = 5.0
SPEAK_MAX_CHARS = 240
ALLOWED_KINDS = frozenset({'stop', 'drive', 'speak', 'wait', 'weather'})


@dataclass(frozen=True)
class RobotAction:
    """One bounded command. ``kind`` is always one of :data:`ALLOWED_KINDS`."""

    kind: str
    vx: float = 0.0
    wz: float = 0.0
    duration_s: float = 0.0
    text: str = ''
    query: str = ''
    raw_ok: bool = True

    def then_stop(self) -> bool:
        """Drive/wait are time-bounded; the loop must issue stop after duration."""
        return self.kind in ('drive', 'wait')


def _stop(*, raw_ok: bool = False) -> RobotAction:
    return RobotAction(kind='stop', raw_ok=raw_ok)


def _clamp(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_str(value: Any, max_chars: int) -> str:
    if value is None:
        return ''
    text = str(value).strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def parse_action(payload: Union[str, bytes, Mapping[str, Any], None]) -> RobotAction:
    """Parse model output. Any JSON failure or unknown kind returns stop."""
    if payload is None:
        return _stop()
    if isinstance(payload, Mapping):
        data: Any = dict(payload)
    else:
        if isinstance(payload, bytes):
            try:
                payload = payload.decode('utf-8')
            except UnicodeDecodeError:
                return _stop()
        text = payload.strip() if isinstance(payload, str) else ''
        if not text:
            return _stop()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return _stop()

    if not isinstance(data, dict):
        return _stop()

    kind = data.get('action', data.get('type'))
    if not isinstance(kind, str):
        return _stop()
    kind = kind.strip().lower()
    if kind not in ALLOWED_KINDS:
        return _stop()

    if kind == 'stop':
        return RobotAction(kind='stop', raw_ok=True)

    if kind == 'drive':
        vx = _clamp(_as_float(data.get('vx', data.get('linear', 0.0))), VX_MAX)
        wz = _clamp(_as_float(data.get('wz', data.get('angular', 0.0))), WZ_MAX)
        duration_s = _as_float(data.get('duration', data.get('duration_s', 0.0)))
        if duration_s < 0.0:
            duration_s = 0.0
        if duration_s > MAX_DURATION_S:
            duration_s = MAX_DURATION_S
        return RobotAction(kind='drive', vx=vx, wz=wz, duration_s=duration_s, raw_ok=True)

    if kind == 'speak':
        return RobotAction(
            kind='speak',
            text=_as_str(data.get('text', data.get('utterance', '')), SPEAK_MAX_CHARS),
            raw_ok=True,
        )

    if kind == 'wait':
        duration_s = _as_float(data.get('duration', data.get('duration_s', 0.0)))
        if duration_s < 0.0:
            duration_s = 0.0
        if duration_s > MAX_DURATION_S:
            duration_s = MAX_DURATION_S
        return RobotAction(kind='wait', duration_s=duration_s, raw_ok=True)

    return RobotAction(
        kind='weather',
        query=_as_str(data.get('query', data.get('location', '')), SPEAK_MAX_CHARS),
        raw_ok=True,
    )


def parse_action_from_model_text(text: str) -> RobotAction:
    """Same as :func:`parse_action`; kept as an explicit alias for the loop."""
    return parse_action(text)
