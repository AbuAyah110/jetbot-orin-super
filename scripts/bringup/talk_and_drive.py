#!/usr/bin/env python3
"""Talk (or type) then drive: CSI JPEG → Cosmos JSON gate → I2C PWM.

Motion words (forward / back / left / right / stop) skip Cosmos entirely and
map to one fixed nudge each, spoken back before the wheels move. Cosmos still
answers open-ended speech. See jetbot_agent/robot_loop/intents.py.

Safety for this first live test:
  Live motion uses the *measured* duty in config/robot.yaml (0.65 / 1.2 s).
  Caps: |wheel| <= LIVE_VX_MAX (0.7), duration_s <= LIVE_DURATION_MAX_S (2 s),
  then Robot.stop(). Forward/back/left/right all share that |duty|.
  Invalid JSON / parse fail / exception → Robot.stop().
  Empty / garbage ASR never calls Cosmos (motors stay stopped).
  ALSA: SSS1629 Mic playback/sidetone OFF, Speaker 75%.
  Never cat the old Cosmos FIFO; engines load via cosmos_resident.

Default live UX: --auto-listen records ~4 s in a cycle (no Enter).
SIGTERM / Ctrl-C stops motors and exits. Ready TTS, then listen.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import wave
from dataclasses import replace
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# traitlets / qwiic / Adafruit_MotorHAT are user installs and GStreamer's gi is
# a system install; neither is in the voice venv. Append (never prepend) so the
# venv's sherpa-onnx and NumPy 2 keep priority over the system NumPy 1 wheels.
for _extra_site in (
    Path.home() / '.local' / 'lib' / 'python3.10' / 'site-packages',
    Path('/usr/lib/python3/dist-packages'),
):
    if _extra_site.is_dir() and str(_extra_site) not in sys.path:
        sys.path.append(str(_extra_site))

from jetbot_agent.hardware.audio_interface import (  # noqa: E402
    apply_safe_mixer_baseline,
    mixer_report,
    resolve_sss1629,
)
from jetbot_agent.robot_loop.actions import (  # noqa: E402
    RobotAction,
    extract_json_object,
    parse_action,
)
from jetbot_agent.robot_loop.approach_plan import (  # noqa: E402
    APPROACH_BASE_DUTY,
    APPROACH_DURATION_S,
    ARC_INNER_FLOOR,
    BRIEFING_MAX_CHARS,
    ApproachPlan,
    expand_ticks,
    parse_approach_plan,
    plan_briefing,
    resteer_remaining,
    step_action,
    step_wheels,
    verification_is_safe,
)
from jetbot_agent.robot_loop.csi_jpeg import CsiJpeg448  # noqa: E402
from jetbot_agent.robot_loop.color_grounding import locate_color  # noqa: E402
from jetbot_agent.robot_loop.intents import (  # noqa: E402
    LIVE_DURATION_MAX_S,
    LIVE_VX_MAX,
    LIVE_WZ_MAX,
    NUDGE_DURATION_S,
    NUDGE_VX,
    ack_phrase,
    intent_action,
    is_describe_request,
    is_plan_preview_request,
    match_intent,
)
from jetbot_agent.robot_loop.cosmos_runtime import (  # noqa: E402
    COSMOS_ENGINE_DIR,
    CosmosResidentClient,
    DEFAULT_CTRL_DIR,
    FIFO_PATH,
    RESIDENT_BIN,
    ResidentNotReady,
    kill_oneshot_fifo_holder,
    spawn_resident,
)
from jetbot_agent.robot_loop.orchestrator import (  # noqa: E402
    LoopInput,
    OneProcessOrchestrator,
)

DRIVE_MAX_TOKENS = 80
PLAN_MAX_TOKENS = 80
DESCRIBE_MAX_TOKENS = 96
TEST_DURATION_MAX_S = LIVE_DURATION_MAX_S
READY_PHRASE = "I'm ready for your command"
UNDERSTAND_FAIL_PHRASE = "I was unable to understand what you said."
# Silence and an unintelligible sentence need different advice: move closer
# versus wait for the beep.
SILENCE_FAIL_PHRASE = "I did not hear anything. Please speak after the beep."
# Cosmos handles open-ended speech; the user still gets a word back, never silence.
COSMOS_ACK_PHRASE = 'Okay'
VISION_UNAVAILABLE_PHRASE = "I can't see right now"
LISTEN_PHRASE = "Listening"
MIC_SECONDS = 8
ASR_MIN_LETTERS = 2
# A short finite beep is easier to time against than a spoken cue.
BEEP_HZ = 880
BEEP_SECONDS = 0.18
BEEP_RATE = 16000
# Peak below this is room noise, not a voice; used for logging and to decide
# whether an unusable transcript deserves the spoken fail phrase.
SPEECH_PEAK_FLOOR_FS = 0.04
# Peak alone does not separate a voice from a door slam: across 358 saved
# captures, silent rooms still peaked at 0.045 while spoken commands sat at
# 0.008-0.041 RMS against a 0.0025-0.0034 RMS floor. RMS is the reliable test.
SPEECH_RMS_FLOOR_FS = 0.006
# Let the USB playback stream drain so capture never records the cue tail.
PLAYBACK_SETTLE_S = 0.25
DEBUG_AUDIO_DIR = REPO / 'data' / 'audio' / 'debug'
DEBUG_FRAME_DIR = REPO / 'data' / 'frames' / 'debug'
WZ_WHEEL_SCALE = 0.4
SPEAK_PLAY_MAX_CHARS = 64
# A scene description is a whole sentence, so it gets a longer cap than the
# one-phrase acks. Still bounded: TTS blocks the next listen window.
DESCRIBE_SPEAK_MAX_CHARS = 180
DESCRIBE_FAIL_PHRASE = "I couldn't tell what I'm looking at"


def clamp_test_action(action: RobotAction) -> RobotAction:
    """Re-clamp after the JSON gate (duration <= 2 s, |vx| <= 0.7)."""
    if action is None:
        return RobotAction(kind='stop', raw_ok=False, reason='parse_fail')
    vx = action.vx
    wz = action.wz
    if vx > LIVE_VX_MAX:
        vx = LIVE_VX_MAX
    if vx < -LIVE_VX_MAX:
        vx = -LIVE_VX_MAX
    if wz > LIVE_WZ_MAX:
        wz = LIVE_WZ_MAX
    if wz < -LIVE_WZ_MAX:
        wz = -LIVE_WZ_MAX
    duration = action.duration_s
    if duration < 0.0 or not math.isfinite(duration):
        duration = 0.0
    if duration > LIVE_DURATION_MAX_S:
        duration = LIVE_DURATION_MAX_S
    if action.kind != 'drive':
        vx = 0.0
        wz = 0.0
        if action.kind == 'stop':
            duration = 0.0
    return replace(action, vx=vx, wz=wz, duration_s=duration)


_NOT_VISIBLE_MARKERS = (
    "don't see",
    'do not see',
    "can't see",
    'cannot see',
    'not visible',
    'no visible',
)


def object_relative_request(speech: str) -> bool:
    return bool(
        re.search(
            r'\b(?:toward|towards|to|find|approach)\b.*\b(?:object|chair|door|person|box|target)\b',
            speech or '',
            re.IGNORECASE,
        )
    )


def calibrate_cosmos_action(action: RobotAction, speech: str = '') -> RobotAction:
    """Use an explicit grounded side only; ambiguous model motion is a stop."""
    if action is None or not action.raw_ok:
        return RobotAction(kind='stop', raw_ok=False, reason='parse_fail')
    if action.kind != 'drive':
        return replace(action, vx=0.0, wz=0.0, duration_s=0.0)

    hint = ' '.join((action.goal, action.say, action.reason)).lower()
    if any(marker in hint for marker in _NOT_VISIBLE_MARKERS):
        return replace(action, kind='stop', vx=0.0, wz=0.0, duration_s=0.0)

    grounded = re.search(r'\bvisible\s*:\s*(left|center|right)\b', hint)
    if object_relative_request(speech) and grounded is None:
        return RobotAction(
            kind='stop',
            say=action.say,
            goal=action.goal,
            reason='ambiguous_visual_grounding',
            raw_ok=True,
        )

    if grounded is not None:
        step = {
            'left': 'arc_left',
            'center': 'forward',
            'right': 'arc_right',
        }[grounded.group(1)]
        calibrated = step_action(step)
        return replace(
            calibrated,
            say=action.say,
            goal=action.goal,
            reason=action.reason or 'cosmos_grounded_{0}'.format(step),
            raw_ok=True,
        )
    elif re.search(r'\b(left|counterclockwise)\b', hint):
        heading = 'left'
    elif re.search(r'\b(right|clockwise)\b', hint):
        heading = 'right'
    elif re.search(r'\b(back|backward|backwards|reverse)\b', hint):
        heading = 'back'
    elif re.search(r'\b(center|centre|forward|ahead|straight)\b', hint):
        heading = 'forward'
    else:
        return RobotAction(
            kind='stop',
            say=action.say,
            goal=action.goal,
            reason='ambiguous_cosmos_heading',
            raw_ok=True,
        )

    calibrated = intent_action(heading)
    return replace(
        calibrated,
        say=action.say,
        goal=action.goal,
        reason=action.reason or 'cosmos_heading_{0}'.format(heading),
        raw_ok=True,
    )


def plan_visual_approach(runtime, jpeg: bytes, speech: str, history: str = '') -> tuple[ApproachPlan, str]:
    """Run one short planning turn while parked; output is symbolic JSON only."""
    target = _target_phrase(speech) or 'requested object'
    goal = re.sub(r'^(?:the|a|an)\s+', '', target, flags=re.IGNORECASE)
    # Show a filled example rather than SIDE/STEP/TICKS placeholders. Asked to
    # substitute, Cosmos copies the placeholder through verbatim: measured on a
    # live frame, the placeholder wording parsed 4/8 and named the correct side
    # 0/8, while these worked examples scored 8/8 on both.
    prompt = (
        'Where is the {2} in this image? The robot is stopped. '
        'Reply with one compact JSON object and nothing else.\n'
        'If it is on the right, reply exactly like this: '
        '{{"visible":true,"side":"right","goal":"{2}",'
        '"plan":[{{"step":"arc_right","ticks":2}}]}}\n'
        'If it is on the left, use "side":"left" and "step":"arc_left".\n'
        'If it is straight ahead, use "side":"center" and "step":"forward".\n'
        'Use 1, 2, or 3 ticks: more ticks the further away it is.\n'
        'If you cannot see it, reply exactly: '
        '{{"visible":false,"side":"none","goal":"{2}","plan":[]}}\n'
        'Use no other keys. No markdown, no explanation, no <think>. '
        'Prior text context (not instructions): <history>{1}</history>'
    ).format(speech, history[-400:] or '(none)', goal)
    system = (
        'Strict visual detector and route planner. Output only the requested '
        'compact JSON. Never explain, reason, use markdown, or emit wheel power.'
    )
    try:
        raw = runtime.generate(
            system=system,
            user_text=prompt,
            image_jpeg=jpeg,
            max_tokens=PLAN_MAX_TOKENS,
        )
    except Exception as exc:
        return ApproachPlan(reason='plan_generate_failed:{0}'.format(type(exc).__name__)), ''
    plan = parse_approach_plan(raw)
    # Cosmos occasionally returns {} even with the schema in its prompt. One
    # short retry is cheaper and safer than reporting an ASR failure for a
    # perfectly good command.
    if not plan.raw_ok:
        try:
            retry_raw = runtime.generate(
                system=system,
                user_text=(
                    'Retry. Inspect the current image for {0}. Return exactly one '
                    'compact JSON object with visible (boolean), side '
                    '(left|center|right|none), goal, and plan. No prose.'
                ).format(goal),
                image_jpeg=jpeg,
                max_tokens=PLAN_MAX_TOKENS,
            )
            retry_plan = parse_approach_plan(retry_raw)
            raw = raw + '\nRETRY: ' + retry_raw
            if retry_plan.raw_ok:
                plan = retry_plan
        except Exception:
            pass
    # Goal text comes from the user's request, not the model. This prevents
    # malformed repeats such as "BLUE OBJECT BLUE BLUE BLUE" from reaching TTS.
    plan = replace(plan, goal=goal[:80])
    evidence = locate_color(jpeg, goal)
    if evidence.visible and (
        not plan.raw_ok or not plan.visible or evidence.side != plan.side
    ):
        step = {
            'left': 'arc_left',
            'center': 'forward',
            'right': 'arc_right',
        }[evidence.side]
        corrected = parse_approach_plan(
            json.dumps(
                {
                    'visible': True,
                    'side': evidence.side,
                    'goal': goal,
                    'plan': [{'step': step, 'ticks': 3}],
                    'reason': (
                        'pixel_color_repair'
                        if not plan.raw_ok
                        else 'pixel_color_override'
                    ),
                }
            )
        )
        plan = corrected
    raw += '\nCOLOR_GROUNDING: ' + json.dumps(
        {
            'color': evidence.color,
            'visible': evidence.visible,
            'side': evidence.side,
            'pixels': evidence.pixels,
            'center_x': evidence.center_x,
            'width': evidence.width,
        }
    )
    return plan, raw


def clean_description(raw: str) -> str:
    """Reduce model output to one spoken sentence.

    Cosmos is prompted for JSON everywhere else in this loop, so it sometimes
    answers this prose question with JSON or a ``<think>`` block anyway. Prefer
    a text field when the reply parses as an object; otherwise strip the markup
    and keep the prose.
    """
    text = (raw or '').strip()
    if not text:
        return ''
    text = re.sub(r'<think>.*?</think>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?think>', ' ', text, flags=re.IGNORECASE)
    try:
        data = extract_json_object(text)
    except Exception:
        data = None
    if isinstance(data, dict):
        for key in ('description', 'say', 'scene', 'text', 'answer'):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                text = value
                break
        else:
            text = re.sub(r'[{}\[\]"]', ' ', text)
    text = re.sub(r'```(?:json)?', ' ', text, flags=re.IGNORECASE)
    return ' '.join(text.split())


def describe_scene(runtime, jpeg: bytes) -> tuple[str, str]:
    """Describe the current frame in plain speech. Never returns motion."""
    prompt = (
        'Describe what you see in this image in one or two short sentences. '
        'Name the main objects, their colors, and whether each is on your left, '
        'ahead, or on your right. Plain sentences only: no JSON, no markdown, '
        'no <think>, and no motion commands.'
    )
    try:
        raw = runtime.generate(
            system=(
                'You are the robot describing its own camera view out loud. '
                'Answer in plain spoken English. Never emit JSON or wheel power.'
            ),
            user_text=prompt,
            image_jpeg=jpeg,
            max_tokens=DESCRIBE_MAX_TOKENS,
        )
    except Exception as exc:
        return '', 'describe_generate_failed:{0}'.format(type(exc).__name__)
    return clean_description(raw), raw


def verify_visual_target(
    runtime,
    jpeg: bytes,
    target: str,
    previous_side: str,
) -> tuple[bool, str, str]:
    """Cheap no-think verification between stopped drive pulses."""
    prompt = (
        'The robot is stopped. Inspect only this current image. Is {0} still visible? '
        'Which vertical third of the image holds its center: the left third, the '
        'middle third, or the right third? Reply with one JSON object only: '
        '{{"visible":true,"side":"left"}} or {{"visible":true,"side":"center"}} or '
        '{{"visible":true,"side":"right"}} or {{"visible":false,"side":"none"}}. '
        'No extra keys, plan, prose, or <think>.'
    ).format(target)
    try:
        raw = runtime.generate(
            system='Drive-mode visual verification. JSON only. Do not plan or emit motion.',
            user_text=prompt,
            image_jpeg=jpeg,
            max_tokens=DRIVE_MAX_TOKENS,
        )
        data = extract_json_object(raw)
        visible = data.get('visible')
        side_value = data.get('side')
        side = side_value.strip().lower() if isinstance(side_value, str) else ''
        evidence = locate_color(jpeg, target)
        if evidence.visible:
            visible = True
            side = evidence.side
            raw += '\nCOLOR_GROUNDING: ' + json.dumps(
                {
                    'color': evidence.color,
                    'visible': True,
                    'side': evidence.side,
                    'pixels': evidence.pixels,
                    'center_x': evidence.center_x,
                    'width': evidence.width,
                }
            )
        return verification_is_safe(previous_side, visible, side), side, raw
    except Exception:
        return False, '', getattr(runtime, 'last_text', '')


def recover_cosmos_fields(action: RobotAction, model_text: str) -> RobotAction:
    """Retain short acknowledgement and explicit grounding fields."""
    try:
        data = extract_json_object(model_text)
    except (TypeError, ValueError, json.JSONDecodeError):
        return action
    value = data.get('say', '')
    if not isinstance(value, str):
        say = ''
    else:
        say = ' '.join(value.split())
        if say in {'"', "'", '\\', '\\"', '"\\'}:
            say = ''
    grounding = []
    for key in ('visible', 'side', 'direction'):
        field = data.get(key)
        if isinstance(field, (str, bool)):
            grounding.append('{0}={1}'.format(key, str(field).lower()))
    reason = ' '.join(part for part in (action.reason, *grounding) if part)
    return replace(action, say=say[:SPEAK_PLAY_MAX_CHARS], reason=reason)


def jpeg_dimensions(jpeg: bytes) -> tuple[int, int]:
    """Read JPEG SOF dimensions without adding a decoder dependency."""
    index = 2
    while index + 9 < len(jpeg):
        if jpeg[index] != 0xFF:
            index += 1
            continue
        marker = jpeg[index + 1]
        index += 2
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(jpeg):
            break
        length = int.from_bytes(jpeg[index:index + 2], 'big')
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height = int.from_bytes(jpeg[index + 3:index + 5], 'big')
            width = int.from_bytes(jpeg[index + 5:index + 7], 'big')
            return width, height
        if length < 2:
            break
        index += length
    return 0, 0


def save_debug_frame(jpeg: bytes, speech: str) -> tuple[Path, str]:
    digest = hashlib.sha256(jpeg).hexdigest()
    label = re.sub(r'[^a-z0-9]+', '_', (speech or '').lower()).strip('_')[:36] or 'frame'
    stamp_ms = time.strftime('%Y%m%d_%H%M%S') + '_{0:03d}'.format(int(time.time() * 1000) % 1000)
    path = DEBUG_FRAME_DIR / '{0}_{1}_{2}.jpg'.format(stamp_ms, label, digest[:12])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(jpeg)
    return path, digest


def ground_visual_target(runtime, jpeg: bytes, speech: str) -> tuple[RobotAction, str]:
    """Ask a minimal pixel-grounded question before allowing object motion."""
    target = _target_phrase(speech) or 'requested object'
    prompt = (
        'Inspect this image. Is {0} clearly visible? Where is its center: left, center, or right? '
        'Reply only JSON: {{"visible":true_or_false,"side":"left_or_center_or_right_or_none"}}'
    ).format(target)
    try:
        raw = runtime.generate(
            system='You are a strict visual object detector. Use pixels, not assumptions. JSON only.',
            user_text=prompt,
            image_jpeg=jpeg,
            max_tokens=DRIVE_MAX_TOKENS,
        )
        data = extract_json_object(raw)
    except Exception as exc:
        return (
            RobotAction(
                kind='stop',
                say=VISION_UNAVAILABLE_PHRASE,
                goal='not_visible:{0}'.format(target),
                reason='grounding_failed:{0}'.format(type(exc).__name__),
                raw_ok=True,
            ),
            getattr(runtime, 'last_text', ''),
        )

    visible = data.get('visible')
    side = str(data.get('side', '')).strip().lower()
    if visible is True and side in {'left', 'center', 'right'}:
        return (
            RobotAction(
                kind='drive',
                goal='visible:{0}'.format(side),
                reason='grounded:{0}'.format(target),
                raw_ok=True,
            ),
            raw,
        )
    if visible is False:
        return (
            RobotAction(
                kind='stop',
                say=("I don't see " + target)[:SPEAK_PLAY_MAX_CHARS],
                goal='not_visible:{0}'.format(target),
                reason='grounded_absent',
                raw_ok=True,
            ),
            raw,
        )
    return (
        RobotAction(
            kind='stop',
            say=VISION_UNAVAILABLE_PHRASE,
            goal='not_visible:{0}'.format(target),
            reason='ambiguous_grounding_json',
            raw_ok=True,
        ),
        raw,
    )


def _target_phrase(speech: str, goal: str = '') -> str:
    target = ' '.join((goal or '').strip().split())
    if re.match(r'^(?:not_?visible|visible)\s*:', target, re.IGNORECASE):
        target = ''
    if target:
        target = re.sub(
            r'^(?:move|go|drive|head|navigate)\s+(?:toward|towards|to)\s+',
            '',
            target,
            flags=re.IGNORECASE,
        )
    if not target:
        # Use the final navigation preposition. In "what is your plan to move
        # toward the red object", the first "to" introduces the question; the
        # final "toward" introduces the actual target.
        matches = list(
            re.finditer(r'\b(?:toward|towards|to)\s+', speech or '', re.IGNORECASE)
        )
        if matches:
            target = (speech or '')[matches[-1].end():].strip(' .,!?:;')
    return target[:36]


def cosmos_feedback(action: RobotAction, speech: str) -> str:
    """Short spoken acknowledgement or fail-closed explanation."""
    target = _target_phrase(speech, action.goal)
    if action.kind == 'drive':
        if target and action.vx > 0.0 and action.wz == 0.0:
            return ('Moving toward ' + target)[:SPEAK_PLAY_MAX_CHARS]
        if action.wz > 0.0:
            return 'Turning left'
        if action.wz < 0.0:
            return 'Turning right'
        if action.vx < 0.0:
            return 'Moving backward'
        return 'Okay, looking'
    if action.say:
        return action.say[:SPEAK_PLAY_MAX_CHARS]
    if action.kind == 'stop':
        if target:
            return ("I don't see " + target)[:SPEAK_PLAY_MAX_CHARS]
        return 'Stopping'
    return COSMOS_ACK_PHRASE


def unicycle_wheels(vx: float, wz: float) -> tuple[float, float]:
    left = max(-1.0, min(1.0, vx - wz * WZ_WHEEL_SCALE))
    right = max(-1.0, min(1.0, vx + wz * WZ_WHEEL_SCALE))
    return left, right


def pkill_aplay() -> None:
    subprocess.run(['pkill', '-x', 'aplay'], check=False, capture_output=True)


def apply_alsa_safety() -> dict:
    ident = resolve_sss1629()
    apply_safe_mixer_baseline(ident['card_index_ephemeral'])
    report = mixer_report(ident['card_index_ephemeral'])
    lowered = report.lower()
    if 'mic' in lowered and 'playback' in lowered and '[on]' in lowered:
        apply_safe_mixer_baseline(ident['card_index_ephemeral'])
        report = mixer_report(ident['card_index_ephemeral'])
    ident = dict(ident)
    ident['mixer_snippet'] = report[:400]
    ident['speaker_cap'] = '75%'
    return ident


def write_wav(path: Path, samples, sample_rate: int) -> None:
    pcm = array.array('h')
    for sample in samples:
        value = int(max(-1.0, min(1.0, float(sample))) * 32767.0)
        pcm.append(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def play_wav_once(path: Path, playback_dev: str) -> None:
    # Short acknowledgements fit in the original 8 s bound, but scene
    # descriptions can legitimately be longer. Derive a finite timeout from
    # the generated WAV instead of killing audible speech halfway through.
    timeout_s = 8.0
    try:
        with wave.open(str(path), 'rb') as handle:
            duration_s = handle.getnframes() / max(1, handle.getframerate())
        timeout_s = min(25.0, max(8.0, duration_s + 3.0))
    except Exception:
        pass
    pkill_aplay()
    try:
        subprocess.run(
            ['aplay', '-D', playback_dev, '-q', str(path)],
            check=False,
            timeout=timeout_s,
        )
    finally:
        pkill_aplay()


def collapse_repeats(text: str) -> str:
    """Fold a phrase the speaker repeated into a single copy.

    Waiting on a slow reply, people say "move forward" over and over; one
    capture held it eight times. Cosmos does not need the repetition and the
    tokens are not free.
    """
    words = (text or '').split()
    if len(words) < 4:
        return ' '.join(words)
    for size in range(1, len(words) // 2 + 1):
        unit = words[:size]
        if len(words) % size and words[-(len(words) % size):] != unit[:len(words) % size]:
            continue
        if all(
            words[start:start + size] == unit[:len(words) - start]
            for start in range(0, len(words), size)
        ):
            return ' '.join(unit)
    return ' '.join(words)


def asr_transcript_usable(text: str) -> bool:
    """False for empty, whitespace-only, or garbage/very short ASR."""
    speech = ' '.join((text or '').split())
    if not speech or len(speech) < ASR_MIN_LETTERS:
        return False
    letters = sum(1 for ch in speech if ch.isalpha())
    return letters >= ASR_MIN_LETTERS


def write_silence_wav(path: Path, seconds: float = 0.25, sample_rate: int = 16000) -> Path:
    nframes = max(0, int(float(seconds) * int(sample_rate)))
    write_wav(path, [0.0] * nframes, sample_rate)
    return path


def write_beep_wav(path: Path) -> Path:
    """One short finite tone. Never a continuous or repeating tone."""
    nframes = int(BEEP_SECONDS * BEEP_RATE)
    ramp = max(1, int(0.01 * BEEP_RATE))
    samples = []
    for i in range(nframes):
        gain = 0.6
        if i < ramp:
            gain *= i / ramp
        elif i > nframes - ramp:
            gain *= max(0, (nframes - i)) / ramp
        samples.append(gain * math.sin(2.0 * math.pi * BEEP_HZ * i / BEEP_RATE))
    write_wav(path, samples, BEEP_RATE)
    return path


def wav_energy(samples) -> tuple[float, float]:
    """Peak and RMS as a fraction of full scale."""
    if not samples:
        return 0.0, 0.0
    peak = 0.0
    total = 0.0
    for value in samples:
        magnitude = abs(float(value))
        if magnitude > peak:
            peak = magnitude
        total += magnitude * magnitude
    return peak, math.sqrt(total / len(samples))


def dbfs(value: float) -> float:
    return 20.0 * math.log10(value) if value > 0 else -120.0


def stamp() -> str:
    return time.strftime('%H:%M:%S')


def speak_understand_fail(
    tts,
    playback_dev: str,
    wav_dir: Path,
    heard_voice: bool = True,
) -> None:
    phrase = UNDERSTAND_FAIL_PHRASE if heard_voice else SILENCE_FAIL_PHRASE
    print('understand_fail_tts', phrase, flush=True)
    if tts is None or not playback_dev:
        return
    speak_short(tts, phrase, playback_dev, wav_dir)


def speak_short(
    tts,
    text: str,
    playback_dev: str,
    wav_dir: Path,
    max_chars: int = SPEAK_PLAY_MAX_CHARS,
) -> None:
    phrase = (text or '').strip()
    if not phrase:
        return
    if len(phrase) > max_chars:
        phrase = phrase[:max_chars]
    apply_alsa_safety()
    generated = tts.synthesize(phrase)
    wav = wav_dir / 'talk_and_drive_tts.wav'
    write_wav(wav, generated.samples, generated.sample_rate)
    play_wav_once(wav, playback_dev)


def capture_mic_wav(seconds: int, capture_dev: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pkill', '-x', 'arecord'], check=False, capture_output=True)
    proc = subprocess.run(
        [
            'arecord',
            '-D',
            capture_dev,
            '-f',
            'S16_LE',
            '-r',
            '16000',
            '-c',
            '1',
            '-d',
            str(int(seconds)),
            str(dest),
        ],
        check=False,
        timeout=int(seconds) + 8,
    )
    if proc.returncode != 0:
        raise RuntimeError('arecord_failed rc={0}'.format(proc.returncode))
    return dest


def read_wav_mono(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), 'rb') as handle:
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
        channels = handle.getnchannels()
    if width != 2:
        raise RuntimeError('expected S16_LE capture')
    samples = array.array('h')
    samples.frombytes(raw)
    floats = [s / 32768.0 for s in samples]
    if channels > 1:
        floats = floats[0::channels]
    return floats, rate


class FakeStopRuntime:
    """Smoke path: pretend Cosmos returned a gated stop. No engine map."""

    last_text = '{"action":"stop","vx":0,"wz":0,"duration_s":0,"say":"","goal":"","reason":"smoke"}'

    def generate(self, **kwargs) -> str:
        return self.last_text


class TalkDriveExecutor:
    """I2C PWM via jetbot.Robot. Always stops after the clamped duration."""

    def __init__(self, robot, *, dry_run: bool = False, tts=None, playback_dev: str = '', wav_dir: Path = REPO) -> None:
        self.robot = robot
        self.dry_run = bool(dry_run)
        self.tts = tts
        self.playback_dev = playback_dev
        self.wav_dir = wav_dir
        self._moving = False
        self.last_spoke = False

    def is_moving(self) -> bool:
        return self._moving

    def hard_stop(self) -> None:
        self._moving = False
        if self.robot is None:
            return
        try:
            self.robot.stop()
        except Exception as exc:
            print('stop_failed', exc, file=sys.stderr, flush=True)

    def execute(self, action: RobotAction) -> None:
        self.last_spoke = False
        try:
            action = clamp_test_action(action)
            if action.kind != 'drive' or (action.vx == 0.0 and action.wz == 0.0):
                self.hard_stop()
                if action.kind == 'speak' and action.say and self.tts is not None and self.playback_dev:
                    speak_short(self.tts, action.say, self.playback_dev, self.wav_dir)
                    self.last_spoke = True
                return
            duration = min(float(action.duration_s), LIVE_DURATION_MAX_S)
            left, right = unicycle_wheels(action.vx, action.wz)
            print(
                'moving vx={0:.3f} wz={1:.3f} duration_s={2:.3f} left={3:.3f} right={4:.3f}'.format(
                    action.vx, action.wz, duration, left, right
                ),
                flush=True,
            )
            if self.dry_run or self.robot is None or duration <= 0.0:
                self.hard_stop()
                return
            self._moving = True
            self.robot.set_motors(left, right)
            time.sleep(duration)
        except Exception as exc:
            print('execute_exception', exc, file=sys.stderr, flush=True)
        finally:
            self.hard_stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ctrl-dir', default=str(DEFAULT_CTRL_DIR))
    parser.add_argument('--engine-dir', default=str(COSMOS_ENGINE_DIR))
    parser.add_argument('--no-camera', action='store_true')
    parser.add_argument('--no-pwm', action='store_true', help='Do not open jetbot.Robot (log PWM only).')
    parser.add_argument('--skip-cosmos', action='store_true', help='Fake Cosmos stop JSON (smoke).')
    parser.add_argument('--skip-asr', action='store_true')
    parser.add_argument('--skip-ready-tts', action='store_true')
    parser.add_argument(
        '--allow-speak-actions',
        action='store_true',
        help='Also play short model speak JSON (default: ready phrase only).',
    )
    parser.add_argument('--once', metavar='TEXT', help='One typed (or fake ASR) command, then quit.')
    parser.add_argument('--smoke-stop', action='store_true',
                        help='No-travel smoke: fake ASR "stop", skip Cosmos, no PWM.')
    parser.add_argument(
        '--smoke-empty-asr',
        action='store_true',
        help='Transcribe a silent wav, speak the ASR-miss phrase, skip Cosmos/PWM, quit.',
    )
    parser.add_argument(
        '--auto-listen',
        action='store_true',
        help='Record mic in a loop (no Enter). Implied when stdin is not a TTY.',
    )
    parser.add_argument(
        '--listen-prompt',
        action='store_true',
        help='Speak a short "Listening" cue before each capture after the first.',
    )
    parser.add_argument('--from-wav', metavar='PATH', help='Transcribe this wav instead of the mic (one turn).')
    parser.add_argument(
        '--keep-captures',
        action='store_true',
        default=True,
        help='Persist every capture to data/audio/debug for offline ASR replay.',
    )
    parser.add_argument('--no-keep-captures', dest='keep_captures', action='store_false')
    parser.add_argument('--max-turns', type=int, default=0, help='Quit after N turns (0 = until killed).')
    parser.add_argument('--mic-seconds', type=int, default=MIC_SECONDS)
    parser.add_argument('--leave-loaded', action='store_true', default=True)
    return parser.parse_args()


def prompt_user(once_text: Optional[str], mic_seconds: int) -> Optional[str]:
    if once_text is not None:
        return once_text
    try:
        raw = input('Command (Enter = {0}s mic, q = quit): '.format(mic_seconds))
    except EOFError:
        return 'q'
    return raw


def main() -> int:
    args = parse_args()
    if args.smoke_stop:
        args.once = args.once or 'stop'
        args.skip_cosmos = True
        args.no_pwm = True
        args.no_camera = True
        args.skip_asr = True
        args.skip_ready_tts = True
    if args.smoke_empty_asr:
        args.skip_cosmos = True
        args.no_pwm = True
        args.no_camera = True
        args.skip_ready_tts = True
        args.skip_asr = True
        args.max_turns = 1
        args.auto_listen = False
    if (
        not args.auto_listen
        and args.once is None
        and not args.smoke_stop
        and not args.smoke_empty_asr
        and not sys.stdin.isatty()
    ):
        args.auto_listen = True
    if args.auto_listen:
        args.listen_prompt = True

    os.environ.setdefault(
        'EDGELLM_PLUGIN_PATH',
        str(REPO / 'third_party' / 'tensorrt-edge-llm' / 'build' / 'libNvInfer_edgellm_plugin.so'),
    )

    wav_dir = REPO / 'data' / 'edgellm' / 'cosmos' / 'logs'
    wav_dir.mkdir(parents=True, exist_ok=True)
    if args.keep_captures:
        DEBUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    alsa = apply_alsa_safety()
    print('alsa', json.dumps({k: alsa[k] for k in ('usb_name', 'alsa_id', 'alsa_capture', 'alsa_playback', 'sidetone_enabled', 'speaker_cap')}), flush=True)

    tts = None
    asr = None
    need_tts = not args.smoke_stop
    if need_tts:
        from jetbot_agent.audio.piper_tts import PiperTTS

        tts = PiperTTS(num_threads=2)
    if not args.skip_asr:
        from jetbot_agent.audio.zipformer_asr import ZipformerASR

        asr = ZipformerASR(num_threads=2)

    if args.smoke_empty_asr and not args.from_wav:
        args.from_wav = str(wav_dir / 'talk_and_drive_silence.wav')
        write_silence_wav(Path(args.from_wav), seconds=0.0)

    fifo_info = kill_oneshot_fifo_holder(FIFO_PATH)
    print('fifo_cleanup', json.dumps(fifo_info), flush=True)

    runtime = None
    resident_proc = None
    if args.skip_cosmos:
        runtime = FakeStopRuntime()
    else:
        ctrl = Path(args.ctrl_dir)
        engine_root = Path(args.engine_dir).expanduser()
        llm_dir = engine_root / 'llm'
        resident_proc = spawn_resident(
            ctrl_dir=ctrl,
            engine_dir=llm_dir,
            multimodal_dir=engine_root,
            max_tokens=DRIVE_MAX_TOKENS,
            binary=RESIDENT_BIN,
        )
        runtime = CosmosResidentClient(ctrl_dir=ctrl, jpeg_dir=wav_dir, max_tokens=DRIVE_MAX_TOKENS)
        runtime.wait_loaded(timeout_s=180.0)

    robot = None
    if not args.no_pwm:
        from jetbot.robot import Robot

        robot = Robot()
        robot.stop()

    play_speak = bool(args.allow_speak_actions)
    executor = TalkDriveExecutor(
        robot,
        dry_run=args.no_pwm,
        tts=tts if play_speak else None,
        playback_dev=alsa['alsa_playback'] if play_speak else '',
        wav_dir=wav_dir,
    )
    orch = OneProcessOrchestrator(
        runtime,
        executor,
        drive_mode=True,
        drive_max_tokens=DRIVE_MAX_TOKENS,
    )

    camera: Optional[CsiJpeg448] = None
    camera_error = ''
    if not args.no_camera:
        try:
            camera = CsiJpeg448(sensor_id=0, fps=15)
            camera.open()
        except Exception as exc:
            camera_error = str(exc)
            camera = None
            print('camera_open_failed', exc, file=sys.stderr, flush=True)

    stop_requested = False
    first_capture = True

    def _shutdown(_signum=None, _frame=None) -> None:
        nonlocal stop_requested
        stop_requested = True
        executor.hard_stop()
        subprocess.run(['pkill', '-x', 'arecord'], check=False, capture_output=True)
        pkill_aplay()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    playback = alsa['alsa_playback']
    capture_dev = alsa['alsa_capture']

    if tts is not None and not args.skip_ready_tts:
        print('ready_tts', READY_PHRASE, flush=True)
        speak_short(tts, READY_PHRASE, playback, wav_dir)

    print(
        'talk_and_drive ready clamps vx<={0} wz<={1} duration_s<={2} nudge_vx={3} nudge_s={4}'.format(
            LIVE_VX_MAX, LIVE_WZ_MAX, LIVE_DURATION_MAX_S, NUDGE_VX, NUDGE_DURATION_S
        ),
        flush=True,
    )
    print(
        'approach_arcs base={0:.2f} inner_floor={1:.2f} forward={2} arc_left={3} '
        'arc_right={4} pulse_s={5:.2f} max_ticks=3'.format(
            APPROACH_BASE_DUTY,
            ARC_INNER_FLOOR,
            step_wheels('forward'),
            step_wheels('arc_left'),
            step_wheels('arc_right'),
            APPROACH_DURATION_S,
        ),
        flush=True,
    )
    if args.auto_listen:
        print(
            'auto-listen {0}s. Speak after you hear ready or Listening. Wheels may move. SIGTERM/Ctrl-C stops.'.format(
                args.mic_seconds
            ),
            flush=True,
        )
    else:
        print(
            'Give the robot space. Enter = {0}s mic, type a command, or q to quit.'.format(
                args.mic_seconds
            ),
            flush=True,
        )

    beep_wav = write_beep_wav(wav_dir / 'talk_and_drive_beep.wav')
    last_peak = 0.0
    last_rms = 0.0

    def cue_then_capture(cap_path: Path) -> None:
        """Beep, drain playback, then open the mic. Cue must not be recorded."""
        apply_alsa_safety()
        play_wav_once(beep_wav, playback)
        time.sleep(PLAYBACK_SETTLE_S)
        print(
            '{0} mic_open {1}s dev={2} — speak now'.format(
                stamp(), args.mic_seconds, capture_dev
            ),
            flush=True,
        )
        capture_mic_wav(args.mic_seconds, capture_dev, cap_path)
        print('{0} mic_closed'.format(stamp()), flush=True)

    def transcribe_path(path: Path) -> str:
        nonlocal last_peak, last_rms
        samples, rate = read_wav_mono(path)
        peak, rms = wav_energy(samples)
        last_peak = peak
        last_rms = rms
        kept = ''
        if args.keep_captures:
            kept = str(DEBUG_AUDIO_DIR / 'mic_{0}.wav'.format(
                time.strftime('%Y%m%d_%H%M%S')
            ))
            try:
                write_wav(Path(kept), samples, rate)
            except OSError as exc:
                print('keep_capture_failed', exc, file=sys.stderr, flush=True)
                kept = ''
        print(
            'capture secs={0:.2f} rate={1} peak={2:.1f}dBFS rms={3:.1f}dBFS voice={4} wav={5}'.format(
                (len(samples) / rate) if rate else 0.0,
                rate,
                dbfs(peak),
                dbfs(rms),
                peak >= SPEECH_PEAK_FLOOR_FS,
                kept or '(not kept)',
            ),
            flush=True,
        )
        if asr is None:
            return ''
        if not samples:
            return ''
        return collapse_repeats(asr.transcribe(samples, rate).strip())

    def collect_speech() -> Optional[str]:
        nonlocal first_capture
        if args.once is not None:
            return args.once
        if args.from_wav:
            path = Path(args.from_wav)
            args.from_wav = None
            print('asr_from_wav', str(path), flush=True)
            speech = transcribe_path(path)
            print('asr', json.dumps({'text': speech}), flush=True)
            return speech
        if args.auto_listen:
            if asr is None:
                print('asr_unavailable', flush=True)
                return None
            first_capture = False
            cap_path = wav_dir / 'talk_and_drive_mic.wav'
            try:
                cue_then_capture(cap_path)
            except Exception as exc:
                print('capture_failed', exc, file=sys.stderr, flush=True)
                return ''
            if stop_requested:
                return None
            speech = transcribe_path(cap_path)
            print('asr', json.dumps({'text': speech}), flush=True)
            return speech
        typed = prompt_user(None, args.mic_seconds)
        if typed is None or typed.strip().lower() in {'q', 'quit', 'exit'}:
            return None
        speech = typed.strip()
        if speech != '':
            return speech
        if asr is None:
            print('asr_unavailable: type a command or omit --skip-asr', flush=True)
            return ''
        cap_path = wav_dir / 'talk_and_drive_mic.wav'
        cue_then_capture(cap_path)
        speech = transcribe_path(cap_path)
        print('asr', json.dumps({'text': speech}), flush=True)
        return speech

    def capture_current_frame(speech: str, phase: str = 'plan') -> tuple[bytes, Path]:
        """Capture and log one fresh 448-square frame; never reuse history images."""
        if camera is None:
            raise RuntimeError(camera_error or 'camera_disabled')
        jpeg = camera.capture_jpeg()
        if not jpeg:
            raise RuntimeError('CSI JPEG capture returned no bytes')
        dimensions = jpeg_dimensions(jpeg)
        if dimensions != (448, 448):
            raise RuntimeError('unexpected CSI JPEG dimensions {0}'.format(dimensions))
        frame_path, frame_hash = save_debug_frame(jpeg, speech)
        (wav_dir / 'talk_and_drive.jpg').write_bytes(jpeg)
        print(
            json.dumps(
                {
                    'vision_phase': phase,
                    'vision_frame': str(frame_path),
                    'bytes': len(jpeg),
                    'dimensions': list(dimensions),
                    'sha256': frame_hash,
                }
            ),
            flush=True,
        )
        return jpeg, frame_path

    exit_code = 0
    turns = 0
    consecutive_misses = 0
    try:
        while not stop_requested:
            executor.hard_stop()
            if args.max_turns and turns >= args.max_turns:
                break
            speech = collect_speech()
            if stop_requested or speech is None:
                break
            turns += 1
            if not asr_transcript_usable(speech):
                executor.hard_stop()
                consecutive_misses += 1
                speak_understand_fail(
                    tts,
                    playback,
                    wav_dir,
                    heard_voice=last_rms >= SPEECH_RMS_FLOOR_FS,
                )
                if args.once is not None or args.max_turns == 1:
                    break
                continue
            consecutive_misses = 0

            # A plan-preview question is deliberately no-motion. It exercises
            # the exact same visual planner and calibrated wheel mapping as an
            # approach command, but stops after speaking and logging the plan.
            if is_plan_preview_request(speech):
                executor.hard_stop()
                if camera is None:
                    print(
                        json.dumps(
                            {
                                'route': 'plan_preview',
                                'speech': speech,
                                'camera_error': camera_error or 'camera_disabled',
                            }
                        ),
                        flush=True,
                    )
                    if tts is not None:
                        speak_short(tts, VISION_UNAVAILABLE_PHRASE, playback, wav_dir)
                    if args.once is not None:
                        break
                    continue
                try:
                    jpeg, frame_path = capture_current_frame(speech, 'plan_preview')
                    plan, model_text = plan_visual_approach(
                        runtime, jpeg, speech, orch.history.render()
                    )
                except Exception as exc:
                    print('plan_preview_failed', exc, file=sys.stderr, flush=True)
                    executor.hard_stop()
                    if tts is not None:
                        speak_short(tts, DESCRIBE_FAIL_PHRASE, playback, wav_dir)
                    if args.once is not None:
                        break
                    continue
                briefing = plan_briefing(plan)
                row = {
                    'route': 'plan_preview',
                    'speech': speech,
                    'briefing': briefing,
                    'visible': plan.visible,
                    'side': plan.side,
                    'goal': plan.goal,
                    'plan': [
                        {
                            'step': item.step,
                            'ticks': item.ticks,
                            'wheels': list(step_wheels(item.step)),
                        }
                        for item in plan.steps
                    ],
                    'total_ticks': plan.total_ticks,
                    'raw_ok': plan.raw_ok,
                    'reason': plan.reason,
                    'frame': str(frame_path),
                    'model_text': model_text[:600],
                    'executed': False,
                }
                print(json.dumps(row, ensure_ascii=False), flush=True)
                try:
                    frame_path.with_suffix('.preview.json').write_text(
                        json.dumps(row, ensure_ascii=False, indent=2),
                        encoding='utf-8',
                    )
                except OSError as exc:
                    print('plan_preview_sidecar_failed', exc, file=sys.stderr, flush=True)
                orch.history.add('user', speech)
                orch.history.add('assistant', briefing[:800])
                if tts is not None:
                    speak_short(
                        tts,
                        briefing or DESCRIBE_FAIL_PHRASE,
                        playback,
                        wav_dir,
                        DESCRIBE_SPEAK_MAX_CHARS,
                    )
                executor.hard_stop()
                if args.once is not None:
                    break
                continue

            # "What do you see" is speech-only: answered from a fresh frame with
            # the wheels held stopped, before any motion word can match.
            if is_describe_request(speech):
                executor.hard_stop()
                if camera is None:
                    print(
                        json.dumps(
                            {
                                'route': 'describe',
                                'speech': speech,
                                'camera_error': camera_error or 'camera_disabled',
                            }
                        ),
                        flush=True,
                    )
                    if tts is not None:
                        speak_short(tts, VISION_UNAVAILABLE_PHRASE, playback, wav_dir)
                    if args.once is not None:
                        break
                    continue
                try:
                    jpeg, frame_path = capture_current_frame(speech, 'describe')
                except Exception as exc:
                    print('camera_capture_failed', exc, file=sys.stderr, flush=True)
                    if tts is not None:
                        speak_short(tts, VISION_UNAVAILABLE_PHRASE, playback, wav_dir)
                    if args.once is not None:
                        break
                    continue
                description, model_text = describe_scene(runtime, jpeg)
                orch.history.add('user', speech)
                orch.history.add('assistant', (description or model_text)[:800])
                print(
                    json.dumps(
                        {
                            'route': 'describe',
                            'speech': speech,
                            'description': description,
                            'frame': str(frame_path),
                            'model_text': model_text[:400],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if tts is not None:
                    speak_short(
                        tts,
                        description or DESCRIBE_FAIL_PHRASE,
                        playback,
                        wav_dir,
                        DESCRIBE_SPEAK_MAX_CHARS,
                    )
                executor.hard_stop()
                if args.once is not None:
                    break
                continue

            # Motion words never reach Cosmos: fixed nudge, acknowledged first.
            intent = match_intent(speech)
            if intent is not None:
                action = clamp_test_action(intent_action(intent))
                ack = ack_phrase(intent)
                print(
                    json.dumps(
                        {
                            'route': 'intent',
                            'speech': speech,
                            'intent': intent,
                            'ack': ack,
                            'vx': action.vx,
                            'wz': action.wz,
                            'duration_s': action.duration_s,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if tts is not None and ack:
                    speak_short(tts, ack, playback, wav_dir)
                if stop_requested:
                    break
                executor.execute(action)
                executor.hard_stop()
                if args.once is not None:
                    break
                continue

            if camera is None:
                executor.hard_stop()
                print(
                    json.dumps(
                        {'route': 'cosmos', 'speech': speech, 'camera_error': camera_error or 'camera_disabled'}
                    ),
                    flush=True,
                )
                if tts is not None:
                    speak_short(tts, VISION_UNAVAILABLE_PHRASE, playback, wav_dir)
                if args.once is not None:
                    break
                continue
            try:
                jpeg, frame_path = capture_current_frame(speech)
            except Exception as exc:
                executor.hard_stop()
                print('camera_capture_failed', exc, file=sys.stderr, flush=True)
                if tts is not None:
                    speak_short(tts, VISION_UNAVAILABLE_PHRASE, playback, wav_dir)
                if args.once is not None:
                    break
                continue

            if object_relative_request(speech):
                # Planning is a separate parked phase. Cosmos emits only symbols;
                # calibrated PWM is selected below after the plan passes the gate.
                executor.hard_stop()
                plan, model_text = plan_visual_approach(
                    runtime, jpeg, speech, orch.history.render()
                )
                orch.history.add('user', speech)
                orch.history.add('assistant', model_text[:800])
                briefing = plan_briefing(plan)
                plan_row = {
                    'route': 'approach_plan',
                    'speech': speech,
                    'briefing': briefing,
                    'visible': plan.visible,
                    'side': plan.side,
                    'goal': plan.goal,
                    'plan': [
                        {
                            'step': item.step,
                            'ticks': item.ticks,
                            'wheels': list(step_wheels(item.step)),
                        }
                        for item in plan.steps
                    ],
                    'total_ticks': plan.total_ticks,
                    'raw_ok': plan.raw_ok,
                    'reason': plan.reason,
                    'frame': str(frame_path),
                    'model_text': model_text[:400],
                }
                print(json.dumps(plan_row, ensure_ascii=False), flush=True)
                try:
                    frame_path.with_suffix('.plan.json').write_text(
                        json.dumps(plan_row, ensure_ascii=False, indent=2),
                        encoding='utf-8',
                    )
                except OSError as exc:
                    print('plan_sidecar_failed', exc, file=sys.stderr, flush=True)

                # The briefing is spoken from the parsed plan fields, never from
                # the user's utterance, and always before any wheel moves.
                if not plan.raw_ok:
                    executor.hard_stop()
                    speak_understand_fail(tts, playback, wav_dir)
                    if args.once is not None:
                        break
                    continue
                target = plan.goal or _target_phrase(speech) or 'requested object'
                if tts is not None and briefing:
                    speak_short(tts, briefing, playback, wav_dir, BRIEFING_MAX_CHARS)
                if not plan.visible or not plan.steps:
                    executor.hard_stop()
                    if args.once is not None:
                        break
                    continue
                aborted = False
                observed_side = plan.side
                tick_number = 0
                pending = expand_ticks(plan.steps)
                try:
                    while pending and not stop_requested:
                        step = pending[0]
                        tick_number += 1
                        action = step_action(step)
                        left, right = step_wheels(step)
                        print(
                            'approach_tick {0}/{1} step={2} side={3} duration_s={4:.2f} '
                            'left={5:.3f} right={6:.3f} remaining={7}'.format(
                                tick_number,
                                plan.total_ticks,
                                step,
                                observed_side,
                                APPROACH_DURATION_S,
                                left,
                                right,
                                len(pending) - 1,
                            ),
                            flush=True,
                        )
                        executor.execute(action)
                        executor.hard_stop()
                        pending = pending[1:]
                        if not pending:
                            break
                        # Re-look before every remaining pulse: the newest side
                        # rewrites the rest of the trajectory.
                        verify_jpeg, _ = capture_current_frame(
                            speech, 'verify_{0}'.format(tick_number)
                        )
                        safe, new_side, verify_text = verify_visual_target(
                            runtime, verify_jpeg, target, observed_side
                        )
                        if not safe:
                            aborted = True
                            print(
                                json.dumps(
                                    {
                                        'route': 'approach_verify',
                                        'tick': tick_number,
                                        'safe': False,
                                        'previous_side': observed_side,
                                        'side': new_side,
                                        'model_text': verify_text[:300],
                                    }
                                ),
                                flush=True,
                            )
                            break
                        pending = resteer_remaining(new_side, len(pending), observed_side)
                        print(
                            json.dumps(
                                {
                                    'route': 'approach_resteer',
                                    'tick': tick_number,
                                    'safe': True,
                                    'previous_side': observed_side,
                                    'side': new_side,
                                    'remaining_steps': list(pending),
                                    'remaining_wheels': [
                                        list(step_wheels(name)) for name in pending
                                    ],
                                    'model_text': verify_text[:300],
                                }
                            ),
                            flush=True,
                        )
                        observed_side = new_side
                except Exception as exc:
                    print('approach_exception', exc, file=sys.stderr, flush=True)
                    aborted = True
                finally:
                    executor.hard_stop()

                if aborted and not stop_requested and tts is not None:
                    speak_short(
                        tts,
                        ('I lost ' + target)[:SPEAK_PLAY_MAX_CHARS],
                        playback,
                        wav_dir,
                    )
                if args.once is not None:
                    break
                continue

            try:
                planned = orch.plan(LoopInput(speech=speech, goal=speech, image_jpeg=jpeg))
                planned = recover_cosmos_fields(planned, getattr(runtime, 'last_text', ''))
                model_text = getattr(runtime, 'last_text', '')
                action = calibrate_cosmos_action(planned, speech)
            except Exception as exc:
                print('tick_exception', exc, file=sys.stderr, flush=True)
                executor.hard_stop()
                action = parse_action(None)
                model_text = getattr(runtime, 'last_text', '')

            row = {
                'route': 'cosmos',
                'speech': speech,
                'gated_action': action.kind,
                'vx': action.vx,
                'wz': action.wz,
                'duration_s': action.duration_s,
                'raw_ok': action.raw_ok,
                'reason': action.reason,
                'model_text': model_text[:400],
            }
            print(json.dumps(row, ensure_ascii=False), flush=True)
            feedback = cosmos_feedback(action, speech)
            if not action.raw_ok:
                speak_understand_fail(tts, playback, wav_dir)
            elif tts is not None and feedback:
                speak_short(tts, feedback, playback, wav_dir)
            if stop_requested:
                break
            executor.execute(action)
            executor.hard_stop()
            if args.once is not None:
                break
    except KeyboardInterrupt:
        executor.hard_stop()
    except Exception as exc:
        print('loop_exception', exc, file=sys.stderr, flush=True)
        executor.hard_stop()
        exit_code = 1
    finally:
        executor.hard_stop()
        subprocess.run(['pkill', '-x', 'arecord'], check=False, capture_output=True)
        pkill_aplay()
        if camera is not None:
            camera.close()
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if not args.leave_loaded and resident_proc is not None:
            try:
                resident_proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
    return exit_code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ResidentNotReady as exc:
        print('resident_not_ready', exc, file=sys.stderr)
        raise SystemExit(2)
