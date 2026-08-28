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

# These are deliberately full-command patterns. A direction word embedded in
# an object-relative request ("move toward the red object on the left") must
# reach Cosmos with the camera image instead of becoming a blind turn.
_FILLER = r'(?:please|now|robot|jetbot)'
_POLITE_PREFIX = re.compile(
    r'^(?:(?:please\s+)|(?:(?:can|could|would|will)\s+you\s+)|'
    r'(?:i\s+(?:want|need)\s+you\s+to\s+))+'
)
_POLITE_SUFFIX = re.compile(
    r'(?:\s+(?:please|for\s+me|a\s+little|a\s+bit|now))+$'
)
_INTENT_PATTERNS = (
    (
        STOP,
        re.compile(
            r'(?:stop|halt|freeze|stand\s+still)'
            r'(?:\s+(?:moving\s+)?(?:forward|forwards|backward|backwards|left|right))?'
        ),
    ),
    (BACK, re.compile(r'(?:(?:move|go|drive)\s+)?(?:backward|backwards|back|reverse|back\s+up)')),
    (FORWARD, re.compile(r'(?:(?:move|go|drive)\s+)?(?:forward|forwards|ahead|straight)')),
    (LEFT, re.compile(r'(?:(?:turn|go|move|drive)\s+)?left')),
    (RIGHT, re.compile(r'(?:(?:turn|go|move|drive)\s+)?right')),
)

# "What do you see" is answered from the current frame with speech only. It is
# matched before the motion patterns so a phrasing like "what do you see on the
# left" describes the scene instead of spinning the chassis.
_DESCRIBE_PATTERN = re.compile(
    r'\b(?:'
    r'what(?:\s+do|\s+can|\s+are)?\s+you\s+(?:see|seeing|look(?:ing)?\s+at)'
    r"|what(?:'s|s|\s+is)\s+(?:in\s+front|there|around|ahead)"
    r'|(?:tell|show)\s+me\s+what\s+you\s+(?:see|can\s+see)'
    r'|describe\s+(?:what\s+you\s+see|the\s+\w+|your\s+\w+)'
    r'|look\s+around'
    r'|do\s+you\s+see\s+anything'
    r')\b'
)

# Plan-preview questions are a parked, no-motion debugging route. Keep this
# separate from object-relative execution so "what would you do..." cannot
# accidentally start the motors.
_PLAN_PREVIEW_PATTERN = re.compile(
    r'\b(?:'
    r"what(?:'s|\s+is|s|\s+would\s+be)\s+your\s+plan"
    r'|what\s+would\s+you\s+do'
    r'|how\s+would\s+you\s+(?:move|go|drive|approach|get)'
    r'|describe\s+your\s+plan'
    r'|plan\s+(?:how|to)\s+'
    r')\b'
)

# Parked visual follow-ups that need the current frame but are not a complete
# scene description or a navigation request. Deictic questions ("what is
# that?", "what color is it?") intentionally land here so short follow-ups can
# use both the fresh image and bounded text history.
_VISUAL_QUESTION_PATTERN = re.compile(
    r'\b(?:'
    r'(?:what|which)\s+(?:color|colour|shape|kind|type)\s+(?:is|are)'
    r'|(?:what|who)\s+is\s+(?:this|that|it)'
    r'|(?:is|are)\s+(?:this|that|it|the\s+\w+\s+object)'
    r'|(?:can|could)\s+you\s+(?:identify|recognize|see|read)'
    r'|(?:tell|talk)\s+(?:(?:to\s+)?me\s+)?about\s+(?:this|that|it|the\s+\w+\s+object)'
    r'|what\s+do\s+you\s+think\s+(?:of|about)\s+(?:this|that|it|the\s+\w+\s+object)'
    r'|how\s+many\s+(?:objects|things|items)'
    r'|where\s+is\s+(?:this|that|it|the\s+\w+\s+object)'
    r'|(?:object|thing|item)\s+(?:you\s+see|in\s+front|on\s+the\s+(?:left|right))'
    r')\b'
)

_SEARCH_PATTERN = re.compile(
    r'\b(?:'
    r'(?:look|search|hunt)\s+(?:around\s+)?(?:the\s+room\s+)?for'
    r'|find|locate'
    r')\s+(?:me\s+)?(?P<target>.+)$'
)


def normalize_transcript(text: str) -> str:
    """Lowercase and strip punctuation. ASR returns UPPERCASE with repeats."""
    lowered = (text or '').lower()
    cleaned = re.sub(r"[^a-z0-9']+", ' ', lowered)
    return ' '.join(cleaned.split())


def match_intent(text: str) -> Optional[str]:
    """Return an intent only when the whole transcript is a bare direction."""
    speech = normalize_transcript(text)
    if not speech:
        return None
    speech = _POLITE_PREFIX.sub('', speech)
    speech = _POLITE_SUFFIX.sub('', speech)
    speech = re.sub(r'^(?:' + _FILLER + r'\s+)*', '', speech)
    speech = re.sub(r'(?:\s+' + _FILLER + r')*$', '', speech)
    for intent, pattern in _INTENT_PATTERNS:
        # ASR commonly repeats a complete phrase. Repeats are accepted only
        # when every repeated segment resolves to the same bare command.
        repeated = re.compile(
            r'^(?:' + pattern.pattern + r')(?:\s+(?:' + pattern.pattern + r'))*$'
        )
        if repeated.fullmatch(speech):
            return intent
    return None


def is_describe_request(text: str) -> bool:
    """True when the speaker asked what the robot currently sees.

    Answered with a fresh frame and speech only: this route never drives, so a
    stray direction word inside the question cannot reach the motors.
    """
    speech = normalize_transcript(text)
    if not speech:
        return False
    return bool(_DESCRIBE_PATTERN.search(speech))


def is_plan_preview_request(text: str) -> bool:
    """True for requests to inspect a prospective plan without executing it."""
    speech = normalize_transcript(text)
    if not speech:
        return False
    return bool(_PLAN_PREVIEW_PATTERN.search(speech))


def is_visual_question(text: str) -> bool:
    """True for parked questions that require a fresh camera frame."""
    speech = normalize_transcript(text)
    if not speech:
        return False
    return bool(_VISUAL_QUESTION_PATTERN.search(speech))


def search_target(text: str) -> str:
    """Return a requested visual-search target, or an empty string."""
    speech = normalize_transcript(text)
    match = _SEARCH_PATTERN.search(speech)
    if match is None:
        return ''
    target = match.group('target')
    target = re.sub(r'^(?:a|an|the)\s+', '', target)
    target = re.sub(r'\s+(?:please|for\s+me)$', '', target)
    return target[:48].strip()


def is_search_request(text: str) -> bool:
    """True for bounded camera search, distinct from approach movement."""
    return bool(search_target(text))


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
