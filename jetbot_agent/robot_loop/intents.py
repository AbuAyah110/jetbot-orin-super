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
    # Zipformer repeatedly heard "look around the room for" as
    # "on the room for". Keep that bounded corruption on the search route
    # instead of letting conversation claim it looked without turning.
    r'|(?:in|on)\s+(?:the\s+)?room\s+for'
    r'|find|locate'
    r')\s+(?:me\s+)?(?P<target>.+)$'
)

# A follow-on "and go to it" clause, and the room locative, are not part of the
# object's name. Applied repeatedly so "red object in the room and go to it"
# reduces to "red object".
_SEARCH_TRAILERS = (
    re.compile(
        r'\s+and\s+(?:go|drive|move|come|head|roll|walk)\s+'
        r'(?:over\s+)?(?:to|toward|towards)\s+(?:it|there)$'
    ),
    re.compile(r'\s+and\s+(?:approach|go\s+to)\s+(?:it|there)$'),
    re.compile(r'\s+(?:in|inside|around)\s+(?:the\s+)?(?:room|area|house)$'),
    re.compile(r'\s+(?:please|for\s+me|now)$'),
)

# The same clause, used as the signal to chain an approach after the search.
_SEARCH_APPROACH_PATTERN = re.compile(
    r'\band\s+(?:'
    r'(?:go|drive|move|come|head|roll|walk)\s+(?:over\s+)?(?:to|toward|towards)'
    r'|approach'
    r')\s+(?:it|there)\b'
)

# A motion verb at the head of the utterance means the speaker asked for
# movement, whatever else the sentence contains. Parked question routes must
# defer to it: "move around the object in front of you" contains the literal
# substring "object in front", which matched _VISUAL_QUESTION_PATTERN and got
# answered with speech while the wheels never turned.
_MOTION_VERB = (
    r'(?:move|go|drive|turn|navigate|steer|roll|head|reverse|back\s+up'
    r'|circle|orbit|swing|walk)'
)
# "Get behind the truck" and "park behind it" are movement too, but "get" and
# "park" are only motion verbs in that construction, so they are kept out of
# _MOTION_VERB and matched via the behind patterns instead.
_MOTION_COMMAND_EXTRA = re.compile(
    r'^(?:get|come|park)\s+(?:around\s+)?(?:behind|to\s+the\s+(?:back|rear|other'
    r'\s+side|far\s+side))\b'
)
_MOTION_COMMAND_PATTERN = re.compile(r'^' + _MOTION_VERB + r'\b')

# Detour requests ("go around the box"). These are movement, not conversation,
# and they are not an approach: the goal is to pass the object rather than
# close on it.
_AROUND_PATTERNS = (
    re.compile(_MOTION_VERB + r'\s+(?:around|round|past)\s+(?P<target>.+)$'),
)
# "Get behind it" is a half orbit, not a sidestep: the robot has to travel round
# the object and end on its far side. Kept apart from the detour patterns
# because the two maneuvers and their failure modes are different. "Circle",
# "orbit" and "all the way around" belong here rather than with the detour,
# which only ever passes an object.
_BEHIND_PATTERNS = (
    re.compile(
        r'\b(?:get|go|move|drive|come|park)\s+(?:around\s+)?behind\s+(?P<target>.+)$'
    ),
    re.compile(
        r'\b(?:get|go|move|drive)\s+(?:to|on)\s+the\s+'
        r'(?:back|rear|other\s+side|far\s+side)\s+of\s+(?P<target>.+)$'
    ),
    re.compile(r'\b(?:circle|orbit)\s+(?:around\s+)?(?P<target>.+)$'),
    re.compile(
        _MOTION_VERB + r'\s+all\s+the\s+way\s+(?:around|round)\s+(?P<target>.+)$'
    ),
)
# "the object in front of you" names the object; the locative is not part of
# the name and must not reach TTS or the detector prompt.
_TARGET_TRAILERS = (
    re.compile(r'\s+(?:that\s+is\s+|which\s+is\s+)?in\s+front\s+of\s+(?:you|me|us)$'),
    re.compile(r'\s+(?:on|to)\s+(?:your|the|my)\s+(?:left|right)$'),
    re.compile(r'\s+(?:please|for\s+me|now)$'),
)

_REMEMBER_PATTERN = re.compile(
    r"\b(?:remember(?:\s+that)?|(?:please\s+)?remember|"
    r"don't\s+forget(?:\s+that)?)\s+(?P<fact>.+)$"
)

# Demo 1: parked JPEG-only identification. Kept off the RAG visual-question
# path so memory cannot invent what is in the hand.
_SHOW_AND_TELL_PATTERN = re.compile(
    r'\b(?:'
    r'what\s+am\s+i\s+holding'
    r'|what\s+(?:is|are)\s+(?:in\s+)?my\s+hand'
    r'|what\s+is\s+this(?:\s+object|\s+thing)?'
    r'|what\s+am\s+i\s+showing\s+(?:you)?'
    r')\b'
)

# Demo 2: one Python-gated creep. Cosmos is not allowed to authorize it.
_CREEP_PATTERN = re.compile(
    r'\b(?:'
    r'creep\s+(?:forward|ahead)'
    r'|if\s+the\s+floor\s+is\s+clear'
    r'|creep\s+if\s+(?:it\'?s|the\s+floor\s+is)\s+clear'
    r')\b'
)

# Demo 3: parked think. Motion verbs still win, so "think then drive" is not
# a think turn.
_THINK_PATTERN = re.compile(
    r'\b(?:'
    r'think\s+hard'
    r'|reason\s+(?:about|whether|if)'
    r'|think\s+(?:about\s+)?whether'
    r')\b'
)

# Demo 4: named-object location from this frame. "Where is this" stays a
# deictic visual follow-up, not a backpack lookup.
_WHERE_PATTERN = re.compile(
    r'\b(?:where\s+is|do\s+you\s+see|can\s+you\s+see)\s+'
    r'(?:the\s+)?(?P<target>(?!this\b|that\b|it\b).+)$'
)

# Demo 5: text places, never stored pictures.
_THIS_VIEW_IS_PATTERN = re.compile(
    r'\bthis\s+view\s+is\s+(?:the\s+)?(?P<place>.+)$'
)
_ARE_WE_AT_PATTERN = re.compile(
    r'\b(?:'
    r'are\s+we\s+(?:at|in)\s+(?:the\s+)?(?P<place>.+)'
    r'|is\s+this\s+the\s+(?P<place_alt>.+)'
    r')$'
)

_DEICTIC_TARGET = re.compile(
    r'^(?:that|this|it|the\s+(?:object|thing|item|one))(?:\s+please)?$'
)

# "What do you see in front of you" asks for a description, but the where-route
# captured "in front of you" as the thing to locate and answered "I see the in
# front of you on my left". A bare viewpoint or direction names no object, so it
# belongs to the describe route instead. ``fronting`` is a recurring Zipformer
# corruption of "front of".
_WHERE_NON_OBJECT = re.compile(
    r'^(?:'
    r'(?:right\s+)?(?:in\s+)?front(?:ing)?(?:\s+of)?(?:\s+(?:you|me|us|it))?'
    r'|(?:straight\s+)?ahead(?:\s+of\s+(?:you|me|us))?'
    r'|(?:over\s+|out\s+)?(?:there|here)'
    r'|(?:all\s+)?around(?:\s+(?:you|me|us|here))?'
    r'|(?:on|to)\s+(?:your|my|the)\s+(?:left|right)'
    r'|in\s+(?:your|the|this)\s+(?:view|image|picture|frame|camera)'
    r'|anything|something|everything|things|stuff'
    r')(?:\s+(?:right\s+)?now)?$'
)


def normalize_transcript(text: str) -> str:
    """Lowercase and strip punctuation. ASR returns UPPERCASE with repeats."""
    lowered = (text or '').lower()
    cleaned = re.sub(r"[^a-z0-9']+", ' ', lowered)
    return ' '.join(cleaned.split())


def strip_politeness(speech: str) -> str:
    """Remove leading requests and trailing filler from a normalized string."""
    speech = _POLITE_PREFIX.sub('', speech)
    speech = _POLITE_SUFFIX.sub('', speech)
    speech = re.sub(r'^(?:' + _FILLER + r'\s+)*', '', speech)
    speech = re.sub(r'(?:\s+' + _FILLER + r')*$', '', speech)
    return speech.strip()


def is_motion_command(text: str) -> bool:
    """True when a motion verb leads the utterance.

    Used as a veto on the parked question routes so a drive request can never
    be satisfied with speech alone.
    """
    speech = strip_politeness(normalize_transcript(text))
    if not speech:
        return False
    if _MOTION_COMMAND_PATTERN.match(speech):
        return True
    return bool(_MOTION_COMMAND_EXTRA.match(speech))


def match_intent(text: str) -> Optional[str]:
    """Return an intent only when the whole transcript is a bare direction."""
    speech = normalize_transcript(text)
    if not speech:
        return None
    speech = strip_politeness(speech)
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


def is_show_and_tell(text: str) -> bool:
    """True for parked JPEG-only 'what am I holding' identification."""
    speech = strip_politeness(normalize_transcript(text))
    if not speech or is_motion_command(speech):
        return False
    return bool(_SHOW_AND_TELL_PATTERN.search(speech))


def is_creep_request(text: str) -> bool:
    """True for a single Python-gated creep, not an unbounded drive."""
    speech = strip_politeness(normalize_transcript(text))
    if not speech:
        return False
    return bool(_CREEP_PATTERN.search(speech))


def is_think_request(text: str) -> bool:
    """True for parked think-hard. A leading motion verb vetoes this."""
    speech = strip_politeness(normalize_transcript(text))
    if not speech or is_motion_command(speech):
        return False
    return bool(_THINK_PATTERN.search(speech))


def where_target(text: str) -> str:
    """Named object the speaker asked to locate in this frame, or empty."""
    speech = strip_politeness(normalize_transcript(text))
    if not speech or is_motion_command(speech):
        return ''
    match = _WHERE_PATTERN.search(speech)
    if match is None:
        return ''
    target = match.group('target')
    for trailer in _TARGET_TRAILERS:
        target = trailer.sub('', target)
    target = re.sub(r'^(?:a|an|the)\s+', '', target).strip()
    if target in {'this', 'that', 'it', 'anything', 'something', 'everything'}:
        return ''
    if _WHERE_NON_OBJECT.match(target):
        return ''
    return target[:48]


def is_where_request(text: str) -> bool:
    """True for eyes-first 'where is the X' / 'do you see the X'."""
    return bool(where_target(text))


def place_name(text: str) -> str:
    """Place name from 'this view is …', or empty."""
    speech = strip_politeness(normalize_transcript(text))
    if not speech or is_motion_command(speech):
        return ''
    match = _THIS_VIEW_IS_PATTERN.search(speech)
    if match is None:
        return ''
    name = match.group('place').strip()
    name = re.sub(r'\s+(?:please|for\s+me|now)$', '', name)
    return name[:48]


def is_place_teach(text: str) -> bool:
    return bool(place_name(text))


def place_query_name(text: str) -> str:
    """Place name from 'are we at …', or empty."""
    speech = strip_politeness(normalize_transcript(text))
    if not speech or is_motion_command(speech):
        return ''
    match = _ARE_WE_AT_PATTERN.search(speech)
    if match is None:
        return ''
    name = (match.group('place') or match.group('place_alt') or '').strip()
    name = re.sub(r'\s+(?:please|for\s+me|now|\?+)$', '', name)
    name = re.sub(r'^(?:the\s+)?', '', name)
    if name in {'this', 'that', 'it'}:
        return ''
    return name[:48]


def is_place_query(text: str) -> bool:
    return bool(place_query_name(text))


def is_deictic_target(text: str) -> bool:
    """True when the approach target is only 'that' / 'this' / 'the object'."""
    target = _target_from_toward(text)
    if not target:
        return False
    return bool(_DEICTIC_TARGET.match(target))


def _target_from_toward(text: str) -> str:
    speech = strip_politeness(normalize_transcript(text))
    match = re.search(
        r'\b(?:toward|towards|to|at)\s+(?P<target>.+)$',
        speech,
    )
    if match is None:
        return ''
    target = match.group('target')
    for trailer in _TARGET_TRAILERS:
        target = trailer.sub('', target)
    target = re.sub(r'^(?:a|an|the)\s+', '', target).strip()
    return target[:48]


def is_plan_preview_request(text: str) -> bool:
    """True for requests to inspect a prospective plan without executing it."""
    speech = normalize_transcript(text)
    if not speech:
        return False
    return bool(_PLAN_PREVIEW_PATTERN.search(speech))


def is_visual_question(text: str) -> bool:
    """True for parked questions that require a fresh camera frame.

    A motion command is never a visual question, even when it mentions an
    object and its position.
    """
    speech = normalize_transcript(text)
    if not speech or is_motion_command(speech):
        return False
    return bool(_VISUAL_QUESTION_PATTERN.search(speech))


def _match_target(text: str, patterns) -> str:
    """First target named by any pattern, stripped of locatives and articles."""
    speech = strip_politeness(normalize_transcript(text))
    if not speech:
        return ''
    for pattern in patterns:
        match = pattern.search(speech)
        if match is None:
            continue
        target = match.group('target')
        for trailer in _TARGET_TRAILERS:
            target = trailer.sub('', target)
        target = re.sub(r'^(?:a|an|the)\s+', '', target).strip()
        if target:
            return target[:48]
    return ''


def around_target(text: str) -> str:
    """Return the object a detour was requested around, or an empty string."""
    return _match_target(text, _AROUND_PATTERNS)


def is_around_request(text: str) -> bool:
    """True for a bounded detour past an object, distinct from approaching it."""
    return bool(around_target(text))


def behind_target(text: str) -> str:
    """Return the object the robot was asked to get behind, or empty."""
    return _match_target(text, _BEHIND_PATTERNS)


def is_behind_request(text: str) -> bool:
    """True for a half orbit ending on the far side of an object."""
    return bool(behind_target(text))


def search_target(text: str) -> str:
    """Return a requested visual-search target, or an empty string.

    "Find the red object in the room and go to it" names one object. Without
    stripping the locative and the follow-on clause, the whole tail became the
    target, so TTS said "I found red object in the room and go to it on my
    left" and the detector was handed that string as a colour phrase.
    """
    speech = normalize_transcript(text)
    match = _SEARCH_PATTERN.search(speech)
    if match is None:
        return ''
    target = match.group('target')
    target = re.sub(r'^(?:a|an|the)\s+', '', target)
    for _ in range(len(_SEARCH_TRAILERS)):
        trimmed = target
        for trailer in _SEARCH_TRAILERS:
            trimmed = trailer.sub('', trimmed)
        if trimmed == target:
            break
        target = trimmed
    return target[:48].strip()


def search_wants_approach(text: str) -> bool:
    """True when a search request also asked to drive to what is found.

    "Find the blue object and go to it" is two steps. Answering only the first
    and staying still is not an honest completion of the request.
    """
    speech = strip_politeness(normalize_transcript(text))
    if not speech or not search_target(speech):
        return False
    return bool(_SEARCH_APPROACH_PATTERN.search(speech))


def is_search_request(text: str) -> bool:
    """True for bounded camera search, distinct from approach movement."""
    return bool(search_target(text))


def memory_fact(text: str) -> str:
    """Extract an explicitly requested long-term fact."""
    speech = normalize_transcript(text)
    match = _REMEMBER_PATTERN.search(speech)
    if match is None:
        return ''
    return match.group('fact')[:300].strip()


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
