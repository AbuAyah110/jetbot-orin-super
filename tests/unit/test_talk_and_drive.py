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
    SILENCE_FAIL_PHRASE,
    SPEAK_PLAY_MAX_CHARS,
    SPEECH_RMS_FLOOR_FS,
    TEST_DURATION_MAX_S,
    UNDERSTAND_FAIL_PHRASE,
    TalkDriveExecutor,
    _target_phrase,
    approach_handoff_speech,
    object_relative_request,
    AROUND_FORWARD_DURATION_S,
    AROUND_MAX_SWING_TURNS,
    AROUND_PASS_PULSES,
    AROUND_TURN_DURATION_S,
    around_forward_action,
    around_turn_action,
    color_corridor_clear,
    describe_scene,
    detour_side_for,
    opposite_side,
    asr_transcript_usable,
    collapse_repeats,
    camera_path_clear,
    pcm16_rms,
    search_forward_action,
    search_turn_action,
    strongest_color_lock,
    speak_understand_fail,
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


@pytest.mark.parametrize(
    'target,expected_route',
    [
        ('blue object', True),
        ('red box', True),
        ('chair', True),
        # The approach planner cannot name these, so the search must not
        # promise an approach it will not perform.
        ('my keys', False),
        ('mira', False),
    ],
)
def test_search_only_promises_an_approach_it_can_actually_route(
    target, expected_route
):
    handoff = approach_handoff_speech(target)
    assert 'the my' not in handoff
    assert object_relative_request(handoff) is expected_route


def test_deictic_lock_prefers_center_object_over_red_edge_tint():
    import io
    from PIL import Image, ImageDraw

    image = Image.new('RGB', (448, 448), (30, 30, 30))
    draw = ImageDraw.Draw(image)
    draw.rectangle((150, 120, 280, 300), fill='blue')
    draw.rectangle((405, 0, 447, 447), fill=(220, 70, 80))
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=95)

    lock = strongest_color_lock(stream.getvalue())

    assert lock is not None
    assert lock.color == 'blue'
    assert lock.side == 'center'


def test_deictic_lock_rejects_small_decorative_red_patch():
    import io
    from PIL import Image, ImageDraw

    image = Image.new('RGB', (448, 448), (30, 30, 30))
    ImageDraw.Draw(image).rectangle((180, 180, 200, 205), fill='red')
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=95)

    assert strongest_color_lock(stream.getvalue()) is None


def test_scene_prompt_is_robot_relative_and_forbids_perspective_tautology():
    class Runtime:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return 'In front of me, I can see a person sitting on the floor.'

    runtime = Runtime()
    description, _ = describe_scene(runtime, b'jpeg')

    assert description.startswith('In front of me, I can see')
    prompt = runtime.calls[0]['user_text']
    assert 'visible person is someone outside JetBot' in prompt
    assert 'perspective tautology' in prompt


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


def test_repeated_command_folds_to_one_copy():
    """Waiting on a slow reply, the speaker said this eight times in one capture."""
    assert collapse_repeats(
        'MOVE FORWARD MOVE FORWARD MOVE FORWARD MOVE FORWARD MOVE FORWARD '
        'MOVE FORWARD MOVE FORWARD MOVE'
    ) == 'MOVE FORWARD'
    assert collapse_repeats('MOVE FORWARD MOVE FORWARD') == 'MOVE FORWARD'


def test_collapse_leaves_a_normal_sentence_alone():
    for phrase in (
        'MOVE TOWARD BLUE OBJECT',
        'what is your plan to move toward the blue object',
        'AND ITS BLUE OBJECTS MOVE TOWARDS BLUE OBJECT BLUE BLUE BLUE',
        'STOP',
    ):
        assert collapse_repeats(phrase) == phrase


def test_silence_and_mumbling_get_different_advice(capsys):
    speak_understand_fail(None, '', Path('.'), heard_voice=False)
    assert SILENCE_FAIL_PHRASE in capsys.readouterr().out

    speak_understand_fail(None, '', Path('.'), heard_voice=True)
    assert UNDERSTAND_FAIL_PHRASE in capsys.readouterr().out


def test_rms_floor_separates_measured_speech_from_measured_silence():
    """RMS in full scale from the saved captures: silence 0.0025-0.0034,
    spoken commands 0.008-0.041."""
    for silent in (0.0025, 0.0030, 0.0034):
        assert silent < SPEECH_RMS_FLOOR_FS
    for spoken in (0.0081, 0.0210, 0.0411):
        assert spoken >= SPEECH_RMS_FLOOR_FS


def test_pcm16_rms_matches_known_amplitude():
    import array

    frame = array.array('h', [3276, -3276] * 160).tobytes()
    assert pcm16_rms(frame) == pytest.approx(0.10, abs=0.001)
    assert pcm16_rms(b'') == 0.0


def test_camera_path_gate_fails_closed():
    class Runtime:
        def __init__(self, text):
            self.text = text

        def generate(self, **_kwargs):
            return self.text

    assert camera_path_clear(Runtime('{"clear":true}'), b'jpeg')[0] is True
    assert camera_path_clear(Runtime('{"clear":false}'), b'jpeg')[0] is False
    assert camera_path_clear(Runtime('uncertain'), b'jpeg')[0] is False
    assert camera_path_clear(Runtime('{"clear":"probably"}'), b'jpeg')[0] is False


@pytest.mark.parametrize(
    'target_side, expected',
    [('right', 'left'), ('left', 'right'), ('center', 'right')],
)
def test_detour_passes_on_the_side_away_from_the_object(target_side, expected):
    assert detour_side_for(target_side) == expected


@pytest.mark.parametrize('side, expected', [('left', 'right'), ('right', 'left')])
def test_detour_realigns_by_turning_back_the_other_way(side, expected):
    assert opposite_side(side) == expected


def _solid_jpeg(rgb, size=64, patch=None):
    """Encode a plain frame, optionally with a coloured patch, for the corridor."""
    from io import BytesIO

    from PIL import Image

    image = Image.new('RGB', (size, size), rgb)
    if patch is not None:
        box, colour = patch
        Image.Image.paste(image, Image.new('RGB', (box[2], box[3]), colour), box[:2])
    buffer = BytesIO()
    image.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()


def test_corridor_fails_closed_when_the_target_is_no_longer_visible():
    # Measured: one 0.35 s pulse swung a centred truck out of frame, the old
    # rule read that as a clear path, and the robot drove forward blind.
    jpeg = _solid_jpeg((90, 90, 90))

    clear, evidence = color_corridor_clear(jpeg, 'red object')

    assert evidence['visible'] is False
    assert clear is False


def test_corridor_rejects_an_edge_artefact_smaller_than_the_tracked_target():
    # A few hundred pixels of reddish door frame at the edge previously passed
    # as the truck and hid the lost track.
    jpeg = _solid_jpeg((40, 40, 40), patch=((46, 20, 16, 20), (255, 20, 20)))

    clear, evidence = color_corridor_clear(jpeg, 'red object', min_pixels=8000)

    # Big enough for locate_color to call it visible, far too small to be the
    # target that was being tracked.
    assert evidence['visible'] is True
    assert evidence['rejected'] == 'lost_track'
    assert clear is False


def test_corridor_accepts_the_tracked_target_once_it_is_off_to_one_side():
    jpeg = _solid_jpeg((40, 40, 40), patch=((2, 16, 18, 28), (255, 20, 20)))

    clear, evidence = color_corridor_clear(jpeg, 'red object', min_pixels=100)

    assert evidence['side'] == 'left'
    assert clear is True


def test_corridor_is_blocked_while_the_target_sits_dead_ahead():
    # A compact saturated patch in the middle third is the target in the path.
    jpeg = _solid_jpeg((40, 40, 40), patch=((26, 20, 12, 24), (255, 20, 20)))

    clear, evidence = color_corridor_clear(jpeg, 'red object')

    assert evidence['side'] == 'center'
    assert clear is False


def test_corridor_is_clear_once_the_target_has_swung_off_to_one_side():
    jpeg = _solid_jpeg((40, 40, 40), patch=((2, 20, 12, 24), (255, 20, 20)))

    clear, evidence = color_corridor_clear(jpeg, 'red object')

    assert evidence['side'] == 'left'
    assert clear is True


def test_corridor_fails_closed_when_the_target_fills_the_frame():
    # Point-blank range looks like "colour everywhere", which locate_color
    # rejects. Reading that rejection as "gone" would drive straight into it.
    jpeg = _solid_jpeg((255, 20, 20))

    clear, evidence = color_corridor_clear(jpeg, 'red object')

    assert evidence['visible'] is False
    assert evidence['rejected'] == 'covers_too_much_of_frame'
    assert clear is False


def test_corridor_fails_closed_for_a_target_it_cannot_ground():
    clear, evidence = color_corridor_clear(_solid_jpeg((90, 90, 90)), 'bottle')

    assert evidence['rejected'] == 'unsupported_target'
    assert clear is False


def test_detour_swing_and_pass_counts_stay_bounded():
    # The swing loop is camera-gated, so its only hard stop is this cap.
    assert 0 < AROUND_MAX_SWING_TURNS <= 4
    assert 0 < AROUND_PASS_PULSES <= 2
    longest_run_s = (
        (AROUND_MAX_SWING_TURNS * 2) * AROUND_TURN_DURATION_S
        + (AROUND_PASS_PULSES + 1) * AROUND_FORWARD_DURATION_S
    )
    assert longest_run_s < 6.0


@pytest.mark.parametrize('direction, sign', [('left', 1.0), ('right', -1.0)])
def test_detour_turn_uses_measured_duty_in_the_requested_direction(direction, sign):
    turn = around_turn_action(direction)

    assert turn.kind == 'drive'
    assert turn.vx == 0.0
    assert turn.wz * sign > 0.0
    assert abs(turn.wz) <= LIVE_WZ_MAX
    assert 0.0 < turn.duration_s < LIVE_DURATION_MAX_S


def test_detour_forward_leg_is_short_and_straight():
    leg = around_forward_action()

    assert leg.kind == 'drive'
    assert leg.wz == 0.0
    assert 0.0 < leg.vx <= LIVE_VX_MAX
    assert 0.0 < leg.duration_s <= 0.5


def test_search_motion_is_short_and_bounded():
    turn = search_turn_action()
    relocate = search_forward_action()

    assert turn.kind == relocate.kind == 'drive'
    assert turn.vx == 0.0 and turn.wz < 0.0
    assert relocate.vx > 0.0 and relocate.wz == 0.0
    assert 0.0 < turn.duration_s < 0.5
    assert 0.0 < relocate.duration_s <= 0.4
    assert turn.duration_s < LIVE_DURATION_MAX_S
    assert relocate.duration_s < LIVE_DURATION_MAX_S


def test_drive_toward_that_is_object_relative_and_deictic():
    from jetbot_agent.robot_loop.intents import is_deictic_target
    from talk_and_drive import object_relative_request

    assert object_relative_request('DRIVE TOWARD THAT') is True
    assert is_deictic_target('DRIVE TOWARD THAT') is True
    assert is_deictic_target('move toward the red object') is False


def test_creep_is_one_short_pulse():
    from talk_and_drive import CREEP_DURATION_S, creep_forward_action

    action = creep_forward_action()
    assert action.kind == 'drive'
    assert action.wz == 0.0
    assert action.vx > 0.0
    assert action.duration_s == pytest.approx(CREEP_DURATION_S)
    assert action.duration_s < 1.0
