"""Helpers for the five staged demos.

These functions never open I2C. Occupancy is a measured pixel heuristic, not a
VLM path gate: Cosmos-Reason2-2B at 448² was shown to be a constant on floor
clearance. Think and place answers are speak-only; a model drive is discarded.
"""

from __future__ import annotations

import io
import re
from statistics import median
from typing import Optional, Protocol

from jetbot_agent.hardware.vl53l0x import interpret_range_mm
from jetbot_agent.robot_loop.actions import RobotAction, parse_action
from jetbot_agent.robot_loop.conversation import claims_motion, MOTION_CLAIM_REPLY
from jetbot_agent.robot_loop.prompts import PARKED_THINK_PROMPT_SUFFIX

# Lower band of the 448² JPEG that a short creep would occupy.
OCCUPANCY_BAND = 0.40
# Manhattan distance from the lower-band median colour. Uniform floor, even
# with this room's pink cast, clusters near the median; a near object does not.
OCCUPANCY_DELTA = 70
# Fraction of the lower band that must look unlike floor before we call it
# blocked. Tuned on synthetic empty vs blob frames in tests/unit/test_demos.py.
OCCUPANCY_BLOCKED = 0.12
OCCUPANCY_CLEAR = 0.04
# One stiction-clearing pulse, shorter than a full nudge so a demo creep is
# a few centimetres rather than a 1.2 s lunge.
CREEP_DURATION_S = 0.45
THINK_MAX_TOKENS = 384
PLACE_COMPARE_MAX_TOKENS = 96

NOT_IN_FRAME_MARKERS = (
    "don't see",
    'do not see',
    "can't see",
    'cannot see',
    'not in this frame',
    'not in the frame',
    "isn't in this frame",
    'is not in this frame',
)

OCCUPANCY_REFUSE = (
    'I cannot tell if the floor is clear without a distance sensor, so I am '
    'staying put.'
)
DEICTIC_REFUSE = "I don't have a clear target, so I won't drive toward that."
THINK_UNSAFE_FALLBACK = 'That path does not look safe, so I am staying put.'
PLACE_UNSURE = "I'm not sure this is that place from this view."


class GenerateRuntime(Protocol):
    def generate(
        self,
        *,
        system: str,
        user_text: str,
        image_jpeg: Optional[bytes],
        max_tokens: int,
    ) -> str:
        ...


def place_slug(name: str) -> str:
    slug = re.sub(r'[^a-z0-9]+', '_', (name or '').lower()).strip('_')
    return ('place:' + slug[:40]) if slug else ''


def occupancy_score(jpeg: bytes) -> dict:
    """Pixel occupancy of the lower band. Fail-closed on decode errors."""
    detail = {
        'ok': False,
        'fraction': 1.0,
        'blocked': True,
        'clear': False,
        'rejected': '',
    }
    if not jpeg:
        detail['rejected'] = 'empty_jpeg'
        return detail
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(jpeg)).convert('RGB')
    except Exception:
        detail['rejected'] = 'decode_failed'
        return detail
    width, height = image.size
    if width < 8 or height < 8:
        detail['rejected'] = 'too_small'
        return detail
    band_top = int(height * (1.0 - OCCUPANCY_BAND))
    band = image.crop((0, band_top, width, height))
    pixels = (
        band.get_flattened_data()
        if hasattr(band, 'get_flattened_data')
        else band.getdata()
    )
    samples = list(pixels)
    if not samples:
        detail['rejected'] = 'empty_band'
        return detail
    median_r = median(pixel[0] for pixel in samples)
    median_g = median(pixel[1] for pixel in samples)
    median_b = median(pixel[2] for pixel in samples)
    unlike = 0
    for red, green, blue in samples:
        delta = abs(red - median_r) + abs(green - median_g) + abs(blue - median_b)
        if delta >= OCCUPANCY_DELTA:
            unlike += 1
    total = len(samples)
    fraction = unlike / float(total)
    detail.update(
        {
            'ok': True,
            'fraction': round(fraction, 4),
            'blocked': fraction >= OCCUPANCY_BLOCKED,
            'clear': fraction <= OCCUPANCY_CLEAR,
            'pixels': total,
            'unlike': unlike,
        }
    )
    return detail


def occupancy_allows_creep(
    jpeg: bytes,
    range_mm: Optional[int] = None,
    kind: str = '',
) -> tuple[bool, dict]:
    """True only when a short pulse is allowed.

    A VL53L0X millimetre reading, when provided, is the safety authority.
    ``kind`` distinguishes an empty field from a broken sensor, which the
    millimetre value alone cannot express. The JPEG occupancy heuristic is only
    a fallback when no range is present. Uncertain scores fail closed.
    """
    if range_mm is not None:
        detail = interpret_range_mm(range_mm, kind=kind)
        detail['source'] = 'tof'
        return bool(detail.get('clear')), detail
    detail = occupancy_score(jpeg)
    detail['source'] = 'jpeg'
    return bool(detail.get('clear')), detail


def eyes_first_reply(
    *,
    target: str,
    visible: bool,
    side: str = '',
    rag: str = '',
) -> str:
    """Speak from this frame. Memory may be hearsay, never location."""
    name = (target or 'it').strip() or 'it'
    if visible:
        location = {
            'left': 'on my left',
            'center': 'in front of me',
            'right': 'on my right',
        }.get((side or '').lower(), 'in this frame')
        return 'I see the {0} {1}.'.format(name, location)[:120]
    hearsay = _hearsay_clause(rag)
    if hearsay:
        return (
            "I don't see the {0} in this frame. Memory mentions {1}."
        ).format(name, hearsay)[:120]
    return "I don't see the {0} in this frame.".format(name)[:120]


def reply_claims_object_present(say: str, target: str) -> bool:
    """True when a spoken line asserts the named object is here now."""
    text = (say or '').lower()
    name = (target or '').lower()
    if not text or not name:
        return False
    if any(marker in text for marker in NOT_IN_FRAME_MARKERS):
        return False
    if name not in text:
        return False
    return bool(
        re.search(
            r'\b(?:is|are|i see|in front|on my|by the|next to|here)\b',
            text,
        )
    )


def _hearsay_clause(rag: str) -> str:
    text = ' '.join((rag or '').split())
    if not text:
        return ''
    # Keep one short clause, not the whole retrieved paragraph.
    text = re.sub(r'^\[[^\]]+\]\s*', '', text)
    return text[:48].rstrip(' .;')


def think_action(
    runtime: GenerateRuntime,
    speech: str,
    jpeg: Optional[bytes],
) -> tuple[RobotAction, str]:
    """Parked think-hard. Speak or stop only; never drive."""
    user = (
        '{0}\nInspect only the current image if one is attached. '
        'Reply with one JSON object: '
        '{{"action":"speak","vx":0,"wz":0,"duration_s":0,'
        '"say":"answer under 120 chars","goal":"","reason":"think"}} '
        'The say field is the only thing spoken. Do not command drive.\n'
        '{1}'
    ).format(' '.join((speech or '').split())[:400], PARKED_THINK_PROMPT_SUFFIX)
    try:
        raw = runtime.generate(
            system=(
                'You are JetBot, parked. Reason privately if needed, then JSON '
                'speak only. Never claim you are moving.'
            ),
            user_text=user,
            image_jpeg=jpeg,
            max_tokens=THINK_MAX_TOKENS,
        )
        action = parse_action(raw)
    except Exception as exc:
        return (
            RobotAction(
                kind='speak',
                say=THINK_UNSAFE_FALLBACK,
                reason='think_error:{0}'.format(type(exc).__name__),
                raw_ok=False,
            ),
            getattr(runtime, 'last_text', ''),
        )
    if not action.raw_ok or action.kind not in {'speak', 'stop'}:
        return (
            RobotAction(
                kind='speak',
                say=THINK_UNSAFE_FALLBACK,
                reason='think_non_speak',
                raw_ok=True,
            ),
            raw,
        )
    if action.kind == 'stop' or not action.say:
        return (
            RobotAction(
                kind='speak',
                say=THINK_UNSAFE_FALLBACK,
                reason='think_empty',
                raw_ok=True,
            ),
            raw,
        )
    if claims_motion(action.say):
        return (
            RobotAction(
                kind='speak',
                say=MOTION_CLAIM_REPLY,
                reason='think_motion_claim',
                raw_ok=True,
            ),
            raw,
        )
    return (
        RobotAction(kind='speak', say=action.say[:120], reason='think', raw_ok=True),
        raw,
    )


def place_compare_action(
    runtime: GenerateRuntime,
    jpeg: bytes,
    place: str,
    memory_text: str,
) -> tuple[RobotAction, str]:
    """Compare this JPEG to a retrieved sentence. No stored picture."""
    prompt = (
        'Place name: {0}\n'
        'Retrieved description (text only, not a photograph):\n'
        '<place>{1}</place>\n'
        'Inspect only the attached current image. Does this view match that '
        'description? Reply with one JSON object: '
        '{{"at_place":true,"say":"short yes with one visual reason"}} or '
        '{{"at_place":false,"say":"short no with one visual reason"}} or '
        '{{"at_place":"uncertain","say":"I am not sure"}}. '
        'at_place may be true only if a detail from the description is visible. '
        'JSON only.'
    ).format(place, (memory_text or '(none)')[:400])
    try:
        raw = runtime.generate(
            system=(
                'You compare one camera frame to a text place label. Never drive. '
                'JSON only.'
            ),
            user_text=prompt,
            image_jpeg=jpeg,
            max_tokens=PLACE_COMPARE_MAX_TOKENS,
        )
        action = parse_action(raw)
    except Exception:
        return (
            RobotAction(kind='speak', say=PLACE_UNSURE, reason='place_error', raw_ok=False),
            getattr(runtime, 'last_text', ''),
        )
    at_place = False
    say = action.say if action.kind == 'speak' else ''
    try:
        from jetbot_agent.robot_loop.actions import extract_json_object

        data = extract_json_object(raw)
        flag = data.get('at_place')
        at_place = flag is True or flag == 'true'
        if isinstance(data.get('say'), str) and data['say'].strip():
            say = data['say'].strip()
    except Exception:
        at_place = False
    if not at_place:
        return (
            RobotAction(kind='speak', say=PLACE_UNSURE, reason='place_uncertain', raw_ok=True),
            raw,
        )
    if not say:
        say = PLACE_UNSURE
        at_place = False
    memory_words = set(re.findall(r'[a-z]{4,}', (memory_text or '').lower()))
    spoken_words = set(re.findall(r'[a-z]{4,}', say.lower()))
    if at_place and memory_words and not (memory_words & spoken_words):
        return (
            RobotAction(kind='speak', say=PLACE_UNSURE, reason='place_no_overlap', raw_ok=True),
            raw,
        )
    if claims_motion(say):
        say = PLACE_UNSURE
    return (
        RobotAction(kind='speak', say=say[:120], reason='place_match', raw_ok=True),
        raw,
    )
