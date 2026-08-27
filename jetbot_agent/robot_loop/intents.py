"""Deterministic motion intents for spoken commands.

Cosmos is not trusted for velocities: it has returned 0.03 m/s for "move
forward", a 0.01 s duration, and a *positive* vx for "move backward". Motion
words therefore skip the model entirely and map to one fixed, bounded nudge
each, sized by the *measured* duty in ``config/robot.yaml`` (see
``drive_calibration``). Open-ended speech still goes to Cosmos.

Sign convention matches ``jetbot.Robot`` and ``unicycle_wheels``
(``left = vx - wz*k``, ``right = vx + wz*k``): +vx drives the chassis forward
like ``Robot.forward()``, and +wz turns left like ``Robot.left()``, which drives
the left wheel negative and the right wheel positive.

This module never opens I2C and never speaks.
"""

from __future__ import annotations

import re
from typing import Optional

from jetbot_agent.robot_loop.actions import RobotAction
from jetbot_agent.robot_loop.drive_calibration import (
    DURATION_HARD_MAX,
    SPEED_HARD_MAX,
    load_calibration,
)

# Measured on the chassis, never guessed: 0.15 hummed without breaking stiction;
# 0.65 is the last pulse that actually traveled. One speed for every intent.
CALIBRATION = load_calibration()
NUDGE_VX = CALIBRATION.speed
NUDGE_DURATION_S = CALIBRATION.duration_s
LIVE_VX_MAX = SPEED_HARD_MAX
LIVE_DURATION_MAX_S = DURATION_HARD_MAX
# unicycle_wheels() scales wz by 0.4. A 0.65-duty in-place turn needs
# wz = 0.65 / 0.4 = 1.625; the old WZ_MAX=1.0 would drop the wheels to 0.40
# (hum-only). Scale the live cap so left/right use the same |duty| as forward.
WZ_WHEEL_SCALE = 0.4
LIVE_WZ_MAX = LIVE_VX_MAX / WZ_WHEEL_SCALE
NUDGE_WZ = NUDGE_VX / WZ_WHEEL_SCALE

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


def intent_wheels(intent: str) -> tuple[float, float]:
    """Signed (left, right) at the measured duty. Stop is (0, 0).

    forward:  +d, +d
    back:     -d, -d
    left:     -d, +d   (in-place, same |duty| as straight)
    right:    +d, -d
    """
    duty = NUDGE_VX
    if intent == FORWARD:
        return duty, duty
    if intent == BACK:
        return -duty, -duty
    if intent == LEFT:
        return -duty, duty
    if intent == RIGHT:
        return duty, -duty
    return 0.0, 0.0


def intent_action(intent: str) -> RobotAction:
    """Fixed, clamped action for one motion intent. Same |duty| and duration."""
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
        wz=max(-LIVE_WZ_MAX, min(LIVE_WZ_MAX, wz)),
        duration_s=min(NUDGE_DURATION_S, LIVE_DURATION_MAX_S),
        reason='intent_{0}'.format(intent),
    )


def ack_phrase(intent: str) -> str:
    return ACK_PHRASES.get(intent, '')
