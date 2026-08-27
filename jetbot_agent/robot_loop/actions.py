"""JSON action gate for Cosmos robot-loop output.

Parses model text into ``{action, vx, wz, duration_s, say, goal, reason}``.
Invalid JSON, unknown kinds, and broken ``say`` values become ``stop`` with
zero velocity. ``reason`` is debug-only and is never used as TTS. This module
never opens motors.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Union

VX_MAX = 0.22
WZ_MAX = 1.0
MAX_DURATION_S = 2.0
SPEAK_MAX_CHARS = 120
GOAL_MAX_CHARS = 80
REASON_MAX_CHARS = 160
ALLOWED_KINDS = frozenset({'stop', 'drive', 'speak', 'wait', 'weather'})
ZERO_VELOCITY_KINDS = frozenset({'stop', 'wait', 'speak', 'weather'})

_FENCE = re.compile(r'```(?:json)?\s*(.*?)\s*```', re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class RobotAction:
    """One bounded command. ``kind`` is always one of :data:`ALLOWED_KINDS`."""

    kind: str
    vx: float = 0.0
    wz: float = 0.0
    duration_s: float = 0.0
    say: str = ''
    goal: str = ''
    reason: str = ''
    raw_ok: bool = True

    @property
    def action(self) -> str:
        return self.kind

    @property
    def text(self) -> str:
        """TTS payload. Empty unless ``kind`` is ``speak`` with a usable ``say``."""
        return self.say if self.kind == 'speak' else ''

    @property
    def query(self) -> str:
        """Weather / location string; sourced from ``goal`` only."""
        return self.goal if self.kind == 'weather' else ''

    def then_stop(self) -> bool:
        """Drive/wait are time-bounded; the loop must issue stop after duration."""
        return self.kind in ('drive', 'wait')

    def as_dict(self) -> dict[str, Any]:
        return {
            'action': self.kind,
            'vx': self.vx,
            'wz': self.wz,
            'duration_s': self.duration_s,
            'say': self.say if self.kind == 'speak' else '',
            'goal': self.goal,
            'reason': self.reason,
        }


def _stop(*, raw_ok: bool = False, reason: str = '', goal: str = '') -> RobotAction:
    return RobotAction(
        kind='stop',
        vx=0.0,
        wz=0.0,
        duration_s=0.0,
        say='',
        goal=goal,
        reason=reason,
        raw_ok=raw_ok,
    )


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
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _clip_str(value: Any, max_chars: int) -> str:
    if value is None:
        return ''
    if not isinstance(value, str):
        text = str(value)
    else:
        text = value
    text = text.strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _say_is_broken(say: str) -> bool:
    """True for empty TTS or the Cosmos broken-quote ``\"`` fragment."""
    if not say:
        return True
    stripped = say.strip()
    if not stripped:
        return True
    if stripped in {'"', "'", '\\', '\\"', '"\\'}:
        return True
    if stripped.count('"') % 2 == 1 and len(stripped) <= 2:
        return True
    return False


def _extract_object_span(text: str) -> Optional[str]:
    """Return the first top-level ``{...}`` span, respecting JSON strings."""
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def extract_json_object(text: str) -> dict[str, Any]:
    """Load the first JSON object from model text (markdown fences allowed)."""
    raw = text.strip()
    if not raw:
        raise ValueError('empty model output')
    fenced = _FENCE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        span = _extract_object_span(raw)
        if span is None:
            raise
        obj = json.loads(span)
    if not isinstance(obj, dict):
        raise ValueError('JSON root is not an object')
    return obj


def _normalize(data: Mapping[str, Any]) -> RobotAction:
    kind = data.get('action')
    if not isinstance(kind, str):
        return _stop(reason='unknown_action')
    kind = kind.strip().lower()
    if kind not in ALLOWED_KINDS:
        return _stop(reason='unknown_action')

    goal = _clip_str(data.get('goal'), GOAL_MAX_CHARS)
    reason = _clip_str(data.get('reason'), REASON_MAX_CHARS)
    say = _clip_str(data.get('say'), SPEAK_MAX_CHARS)

    vx = _clamp(_as_float(data.get('vx', 0.0)), VX_MAX)
    wz = _clamp(_as_float(data.get('wz', 0.0)), WZ_MAX)
    duration_s = _as_float(data.get('duration_s', data.get('duration', 0.0)))
    if duration_s < 0.0:
        duration_s = 0.0
    if duration_s > MAX_DURATION_S:
        duration_s = MAX_DURATION_S

    if kind in ZERO_VELOCITY_KINDS:
        vx = 0.0
        wz = 0.0

    if kind == 'speak' and _say_is_broken(say):
        return _stop(raw_ok=True, reason=reason or 'empty_say', goal=goal)

    if kind != 'speak' or _say_is_broken(say):
        say = ''

    if kind != 'drive':
        if kind == 'stop':
            duration_s = 0.0

    return RobotAction(
        kind=kind,
        vx=vx,
        wz=wz,
        duration_s=duration_s,
        say=say,
        goal=goal,
        reason=reason,
        raw_ok=True,
    )


def parse_action(payload: Union[str, bytes, Mapping[str, Any], None]) -> RobotAction:
    """Parse Cosmos output. JSON failure or a broken ``say`` returns stop."""
    if payload is None:
        return _stop(reason='parse_fail')
    if isinstance(payload, Mapping):
        return _normalize(payload)
    if isinstance(payload, bytes):
        try:
            payload = payload.decode('utf-8')
        except UnicodeDecodeError:
            return _stop(reason='parse_fail')
    if not isinstance(payload, str):
        return _stop(reason='parse_fail')
    text = payload.strip()
    if not text:
        return _stop(reason='parse_fail')
    try:
        data = extract_json_object(text)
    except (ValueError, json.JSONDecodeError, TypeError):
        return _stop(reason='parse_fail')
    return _normalize(data)


def parse_action_from_model_text(text: str) -> RobotAction:
    """Same as :func:`parse_action`; kept as an explicit alias for the loop."""
    return parse_action(text)


def parse_model_output(text: str) -> dict[str, Any]:
    """Schema dict for the gate: action, vx, wz, duration_s, say, goal, reason."""
    return parse_action(text).as_dict()
