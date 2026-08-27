"""Deterministic motion intents for spoken commands.

Cosmos is not trusted for velocities: it has returned 0.03 m/s for "move
forward", a 0.01 s duration, and a *positive* vx for "move backward". Motion
words therefore skip the model entirely and map to one fixed, bounded nudge
each. Open-ended speech still goes to Cosmos.

Sign convention matches ``jetbot.Robot`` and ``unicycle_wheels``
(``left = vx - wz*k``, ``right = vx + wz*k``): +vx drives the chassis forward
like ``Robot.forward()``, and +wz turns left like ``Robot.left()``, which drives
the left wheel negative and the right wheel positive.

This module never opens I2C and never speaks.
"""

from __future__ import annotations

import re
from typing import Optional

from jetbot_agent.robot_loop.actions import WZ_MAX, RobotAction

# 0.15 hummed but did not break stiction on this chassis. Live loop uses a
# conservative duty that actually rolled (not the 0.7 diagnostic pulse).
NUDGE_VX = 0.30
NUDGE_DURATION_S = 0.35
LIVE_VX_MAX = 0.35
# unicycle_wheels() scales wz by 0.4, so 0.75 spins the wheels at the same
# 0.30 magnitude the straight-line nudge uses.
NUDGE_WZ = 0.75

FORWARD = 'forward'
BACK = 'back'
LEFT = 'left'
RIGHT = 'right'
STOP = 'stop'

ACK_PHRASES = {
    FORWARD: 'Moving forward',
    BACK: 'Moving backward',
    LEFT: 'Turning left',
    RIGHT: 'Turning right',
    STOP: 'Stopping',
}

# Stop wins over everything; backward is checked before forward so "backward"
# can never fall through to the forward nudge.
_INTENT_PATTERNS = (
    (STOP, re.compile(r'\b(stop|halt|freeze|stand\s+still)\b')),
    (BACK, re.compile(r'\b(backward|backwards|back\s+up|go\s+back|reverse|back)\b')),
    (FORWARD, re.compile(r'\b(forward|forwards|ahead|straight)\b')),
    (LEFT, re.compile(r'\bleft\b')),
    (RIGHT, re.compile(r'\bright\b')),
)


def normalize_transcript(text: str) -> str:
    """Lowercase and strip punctuation. ASR returns UPPERCASE with repeats."""
    lowered = (text or '').lower()
    cleaned = re.sub(r"[^a-z0-9']+", ' ', lowered)
    return ' '.join(cleaned.split())


def match_intent(text: str) -> Optional[str]:
    """Return a motion intent for loose ASR text, or None for open-ended speech."""
    speech = normalize_transcript(text)
    if not speech:
        return None
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(speech):
            return intent
    return None


def intent_action(intent: str) -> RobotAction:
    """Fixed, clamped action for one motion intent."""
    if intent == STOP:
        return RobotAction(kind='stop', vx=0.0, wz=0.0, duration_s=0.0, reason='intent_stop')
    vx = 0.0
    wz = 0.0
    if intent == FORWARD:
        vx = NUDGE_VX
    elif intent == BACK:
        vx = -NUDGE_VX
    elif intent == LEFT:
        wz = NUDGE_WZ
    elif intent == RIGHT:
        wz = -NUDGE_WZ
    else:
        return RobotAction(kind='stop', vx=0.0, wz=0.0, duration_s=0.0, reason='intent_unknown')
    return RobotAction(
        kind='drive',
        vx=max(-LIVE_VX_MAX, min(LIVE_VX_MAX, vx)),
        wz=max(-WZ_MAX, min(WZ_MAX, wz)),
        duration_s=NUDGE_DURATION_S,
        reason='intent_{0}'.format(intent),
    )


def ack_phrase(intent: str) -> str:
    return ACK_PHRASES.get(intent, '')
