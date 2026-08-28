from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'bringup'))

from jetbot_agent.robot_loop.actions import RobotAction, parse_action  # noqa: E402
from jetbot_agent.robot_loop.intents import LIVE_DURATION_MAX_S, LIVE_VX_MAX, LIVE_WZ_MAX  # noqa: E402

from talk_and_drive import (  # noqa: E402
    SPEAK_PLAY_MAX_CHARS,
    TEST_DURATION_MAX_S,
    UNDERSTAND_FAIL_PHRASE,
    TalkDriveExecutor,
    _target_phrase,
    asr_transcript_usable,
    calibrate_cosmos_action,
    clamp_test_action,
    ground_visual_target,
    plan_visual_approach,
    unicycle_wheels,
    verify_visual_target,
)


@pytest.mark.parametrize(
    'speech, expected',
    [
        ('MOVE TOWARD THE RED OBJECT', 'THE RED OBJECT'),
        ('WHAT IS YOUR PLAN TO MOVE TOWARD THE RED OBJECT', 'THE RED OBJECT'),
        ('WHAT WOULD YOU DO TO GO TO THE BLUE OBJECT', 'THE BLUE OBJECT'),
    ],
)
def test_target_phrase_uses_final_navigation_preposition(speech, expected):
    assert _target_phrase(speech) == expected


def test_color_grounding_repairs_malformed_cosmos_plan():
    import io
    from PIL import Image, ImageDraw

    image = Image.new('RGB', (448, 448), (20, 20, 20))
    ImageDraw.Draw(image).rectangle((330, 100, 430, 350), fill='blue')
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=95)

    class BadRuntime:
        def generate(self, **_kwargs):
            return '{"visible":true,"side":"left","goal":"BLUE OBJECT",' \
                   '"plan":[{"step":"STEP","ticks":1}]}'

    plan, raw = plan_visual_approach(
        BadRuntime(), stream.getvalue(), 'MOVE TOWARD THE BLUE OBJECT'
    )
    assert plan.raw_ok is True
    assert plan.visible is True
    assert plan.side == 'right'
    assert plan.steps[0].step == 'arc_right'
    assert 'COLOR_GROUNDING' in raw


def test_color_grounding_corrects_relook_side():
    import io
    from PIL import Image, ImageDraw

    image = Image.new('RGB', (448, 448), (20, 20, 20))
    ImageDraw.Draw(image).rectangle((330, 100, 430, 350), fill='blue')
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=95)

    class WrongSideRuntime:
        def generate(self, **_kwargs):
            return '{"visible":true,"side":"left"}'

    safe, side, raw = verify_visual_target(
        WrongSideRuntime(), stream.getvalue(), 'BLUE OBJECT', 'right'
    )
    assert safe is True
    assert side == 'right'
    assert 'COLOR_GROUNDING' in raw


def test_test_clamp_caps_duration_to_half_second():
    cosmos = parse_action(
        json.dumps({'action': 'drive', 'vx': 9.0, 'wz': -4.0, 'duration_s': 99})
    )
    assert cosmos.vx == 0.22
    clamped = clamp_test_action(
        RobotAction(kind='drive', vx=9.0, wz=-4.0, duration_s=99)
    )
    assert clamped.kind == 'drive'
    assert clamped.vx == LIVE_VX_MAX == 0.7
    assert clamped.wz == pytest.approx(-LIVE_WZ_MAX)
    assert clamped.duration_s == LIVE_DURATION_MAX_S
    assert TEST_DURATION_MAX_S == LIVE_DURATION_MAX_S == 2.0


def test_parse_fail_is_stop():
    action = clamp_test_action(parse_action('not json'))
    assert action.kind == 'stop'
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert action.raw_ok is False


def test_cosmos_tiny_velocity_becomes_calibrated_forward_tick():
    planned = parse_action(
        '{"action":"drive","vx":0.03,"wz":0,"duration_s":0.01,'
        '"goal":"visible:center","reason":"red object is visible"}'
    )
    action = calibrate_cosmos_action(planned, 'MOVE TOWARDS THE RED OBJECT')
    assert action.kind == 'drive'
    assert action.vx == pytest.approx(0.65)
    assert action.wz == 0.0
    assert action.duration_s == pytest.approx(1.2)
    assert unicycle_wheels(action.vx, action.wz) == pytest.approx((0.65, 0.65))


def test_small_positive_wz_does_not_default_to_left():
    planned = parse_action(
        '{"action":"drive","vx":0.22,"wz":0.01,"duration_s":0.01,'
        '"goal":"red object","reason":"directed movement"}'
    )
    action = calibrate_cosmos_action(planned, 'MOVE TOWARDS THE RED OBJECT')
    assert action.kind == 'stop'
    assert action.reason == 'ambiguous_visual_grounding'
    assert action.vx == action.wz == 0.0


@pytest.mark.parametrize(
    'side, wheels',
    [
        ('left', (0.50, 0.70)),
        ('center', (0.65, 0.65)),
        ('right', (0.70, 0.50)),
    ],
)
def test_grounded_visible_side_selects_calibrated_tick(side, wheels):
    planned = parse_action(
        json.dumps({'action': 'drive', 'vx': 0, 'wz': 0, 'goal': 'visible:' + side})
    )
    action = calibrate_cosmos_action(planned, 'MOVE TOWARDS THE RED OBJECT')
    assert action.duration_s == pytest.approx(1.2)
    assert unicycle_wheels(action.vx, action.wz) == pytest.approx(wheels)


def test_invalid_cosmos_json_calibration_stays_stopped():
    action = calibrate_cosmos_action(parse_action('not json'))
    assert action.kind == 'stop'
    assert action.raw_ok is False
    assert action.vx == action.wz == action.duration_s == 0.0


def test_grounding_question_uses_pixels_and_requires_visible_side():
    class Runtime:
        last_text = ''

        def generate(self, **kwargs):
            assert kwargs['image_jpeg'] == b'jpeg'
            assert 'Use pixels' in kwargs['system']
            self.last_text = '{"visible":true,"side":"right"}'
            return self.last_text

    planned, raw = ground_visual_target(
        Runtime(), b'jpeg', 'MOVE TOWARDS THE RED OBJECT'
    )
    assert raw == '{"visible":true,"side":"right"}'
    action = calibrate_cosmos_action(planned, 'MOVE TOWARDS THE RED OBJECT')
    assert unicycle_wheels(action.vx, action.wz) == pytest.approx((0.70, 0.50))


def test_grounding_absent_is_spoken_stop():
    class Runtime:
        last_text = '{"visible":false,"side":"none"}'

        def generate(self, **kwargs):
            return self.last_text

    planned, _ = ground_visual_target(
        Runtime(), b'jpeg', 'MOVE TOWARDS THE GREEN OBJECT'
    )
    assert planned.kind == 'stop'
    assert planned.say == "I don't see THE GREEN OBJECT"
    assert planned.vx == planned.wz == 0.0


def test_executor_parse_fail_and_drive_always_stop():
    class FakeRobot:
        def __init__(self) -> None:
            self.calls = []

        def stop(self) -> None:
            self.calls.append('stop')

        def set_motors(self, left, right) -> None:
            self.calls.append(('pwm', left, right))

    robot = FakeRobot()
    exe = TalkDriveExecutor(robot, dry_run=False)
    exe.execute(parse_action('{{{{'))
    assert robot.calls[-1] == 'stop'
    exe.execute(parse_action('{"action":"drive","vx":0.2,"wz":0.0,"duration_s":0.01}'))
    assert ('pwm', 0.2, 0.2) in robot.calls or any(
        isinstance(c, tuple) and c[0] == 'pwm' for c in robot.calls
    )
    assert robot.calls[-1] == 'stop'


def test_unicycle_forward():
    left, right = unicycle_wheels(0.22, 0.0)
    assert left == right == 0.22


def test_asr_miss_rejects_empty_and_garbage():
    assert asr_transcript_usable('') is False
    assert asr_transcript_usable('   ') is False
    assert asr_transcript_usable('.') is False
    assert asr_transcript_usable('a') is False
    assert asr_transcript_usable('go') is True
    assert asr_transcript_usable('drive forward') is True
    assert len(UNDERSTAND_FAIL_PHRASE) <= SPEAK_PLAY_MAX_CHARS
