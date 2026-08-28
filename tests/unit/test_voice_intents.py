from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'bringup'))

from jetbot_agent.robot_loop.intents import (  # noqa: E402
    ACK_PHRASES,
    LIVE_DURATION_MAX_S,
    LIVE_VX_MAX,
    LIVE_WZ_MAX,
    NUDGE_DURATION_S,
    NUDGE_VX,
    ack_phrase,
    around_target,
    intent_action,
    intent_wheels,
    is_around_request,
    is_describe_request,
    is_motion_command,
    is_plan_preview_request,
    is_search_request,
    is_visual_question,
    match_intent,
    memory_fact,
    search_target,
)

from talk_and_drive import (  # noqa: E402
    DESCRIBE_SPEAK_MAX_CHARS,
    SPEAK_PLAY_MAX_CHARS,
    clamp_test_action,
    clean_description,
    unicycle_wheels,
)


@pytest.mark.parametrize(
    'transcript, expected',
    [
        ('MOVE FORWARD MOVE FORWARD MOVE FORWARD', 'forward'),
        ('forward', 'forward'),
        ('GO FORWARD', 'forward'),
        ('DRIVE FORWARD PLEASE', 'forward'),
        ('MOVE BACKWARD', 'back'),
        ('GO BACK', 'back'),
        ('REVERSE', 'back'),
        ('BACK UP BACK UP', 'back'),
        ('TURN LEFT', 'left'),
        ('LEFT', 'left'),
        ('TURN RIGHT NOW', 'right'),
        ('STOP', 'stop'),
        ('HALT!', 'stop'),
    ],
)
def test_motion_words_match_loosely(transcript, expected):
    assert match_intent(transcript) == expected


@pytest.mark.parametrize(
    'transcript, expected',
    [
        ('Could you move forward please?', 'forward'),
        ('Would you go back for me?', 'back'),
        ('Can you turn left a little?', 'left'),
        ('I want you to turn right now', 'right'),
        ('Please stop for me', 'stop'),
    ],
)
def test_polite_natural_motion_commands_match(transcript, expected):
    assert match_intent(transcript) == expected


@pytest.mark.parametrize(
    'transcript',
    [
        '',
        '   ',
        'WHAT IS THE WEATHER TODAY',
        'TELL ME A JOKE',
        'WHO ARE YOU',
        'MOVE TOWARD THE RED OBJECT',
        'GO TO THE CHAIR',
        'TURN LEFT TOWARD THE DOOR',
    ],
)
def test_open_ended_speech_falls_through_to_cosmos(transcript):
    assert match_intent(transcript) is None


def test_stop_wins_over_a_direction_word():
    assert match_intent('STOP MOVING FORWARD') == 'stop'


def test_backward_never_reads_as_forward():
    assert match_intent('BACKWARD') == 'back'
    assert intent_action('back').vx < 0.0


def test_forward_is_positive_vx_and_backward_negative():
    forward = intent_action('forward')
    back = intent_action('back')
    assert forward.kind == 'drive'
    assert forward.vx == pytest.approx(NUDGE_VX)
    assert forward.wz == 0.0
    assert back.vx == pytest.approx(-NUDGE_VX)
    assert forward.duration_s == pytest.approx(NUDGE_DURATION_S)
    assert back.duration_s == pytest.approx(NUDGE_DURATION_S)


def test_turns_are_wz_only_and_not_swapped():
    left = intent_action('left')
    right = intent_action('right')
    assert left.vx == 0.0 and right.vx == 0.0
    # +wz is a left turn, matching Robot.left(): left wheel back, right wheel forward.
    assert left.wz > 0.0
    assert right.wz < 0.0
    left_wheels = unicycle_wheels(left.vx, left.wz)
    right_wheels = unicycle_wheels(right.vx, right.wz)
    assert left_wheels[0] < 0.0 < left_wheels[1]
    assert right_wheels[1] < 0.0 < right_wheels[0]


def test_every_motion_intent_uses_the_same_measured_duty():
    """Forward, back, left, and right all spin wheels at |NUDGE_VX| for the same hold."""
    assert NUDGE_VX == pytest.approx(0.65)
    assert NUDGE_DURATION_S == pytest.approx(1.2)
    assert intent_wheels('forward') == (NUDGE_VX, NUDGE_VX)
    assert intent_wheels('back') == (-NUDGE_VX, -NUDGE_VX)
    assert intent_wheels('left') == (-NUDGE_VX, NUDGE_VX)
    assert intent_wheels('right') == (NUDGE_VX, -NUDGE_VX)
    assert intent_wheels('stop') == (0.0, 0.0)
    for name in ('forward', 'back', 'left', 'right'):
        action = intent_action(name)
        assert action.duration_s == pytest.approx(NUDGE_DURATION_S)
        wheels = unicycle_wheels(action.vx, action.wz)
        assert wheels == pytest.approx(intent_wheels(name))
        assert abs(wheels[0]) == pytest.approx(NUDGE_VX)
        assert abs(wheels[1]) == pytest.approx(NUDGE_VX)


def test_wheel_pairs_match_robot_forward_and_backward():
    assert unicycle_wheels(intent_action('forward').vx, 0.0) == (NUDGE_VX, NUDGE_VX)
    assert unicycle_wheels(intent_action('back').vx, 0.0) == (-NUDGE_VX, -NUDGE_VX)


def test_stop_intent_is_zero_velocity():
    action = intent_action('stop')
    assert action.kind == 'stop'
    assert action.vx == 0.0 and action.wz == 0.0 and action.duration_s == 0.0


def test_nudges_survive_the_live_loop_clamp():
    assert NUDGE_VX <= LIVE_VX_MAX
    for intent in ('forward', 'back', 'left', 'right'):
        action = clamp_test_action(intent_action(intent))
        assert action.duration_s == pytest.approx(NUDGE_DURATION_S)
        assert action.duration_s <= LIVE_DURATION_MAX_S
        assert abs(action.vx) <= LIVE_VX_MAX
        assert abs(action.wz) <= LIVE_WZ_MAX
        left, right = unicycle_wheels(action.vx, action.wz)
        assert abs(left) == pytest.approx(NUDGE_VX)
        assert abs(right) == pytest.approx(NUDGE_VX)
    forward = clamp_test_action(intent_action('forward'))
    assert forward.vx == pytest.approx(NUDGE_VX)


@pytest.mark.parametrize(
    'transcript',
    [
        'WHAT DO YOU SEE',
        'what do you see?',
        'WHAT ARE YOU LOOKING AT',
        'TELL ME WHAT YOU SEE',
        'DESCRIBE WHAT YOU SEE',
        "WHAT'S IN FRONT OF YOU",
        'DO YOU SEE ANYTHING',
        'LOOK AROUND',
    ],
)
def test_describe_questions_are_recognized(transcript):
    assert is_describe_request(transcript) is True


@pytest.mark.parametrize(
    'transcript',
    [
        'WHAT COLOR IS THAT OBJECT',
        'what is this?',
        'can you identify the blue thing',
        'tell me about that object',
        'what do you think of it',
        'how many objects are there',
        'where is the red object',
        'is that a toy',
    ],
)
def test_visual_follow_up_questions_are_recognized(transcript):
    assert is_visual_question(transcript) is True


@pytest.mark.parametrize(
    'transcript',
    [
        'WHAT IS THE CAPITAL OF CANADA',
        'TELL ME A JOKE',
        'HOW ARE YOU',
        'MOVE FORWARD',
        'WHAT IS YOUR PLAN TO MOVE TOWARD THE BLUE OBJECT',
    ],
)
def test_non_visual_conversation_does_not_open_camera(transcript):
    assert is_visual_question(transcript) is False


@pytest.mark.parametrize(
    'transcript,target',
    [
        ('LOOK FOR THE BLUE BOX', 'blue box'),
        ('SEARCH AROUND THE ROOM FOR MY KEYS', 'my keys'),
        ('MOVE AROUND THE ROOM AND LOOK FOR A RED OBJECT', 'red object'),
        ('FIND THE GREEN TOY', 'green toy'),
        ('LOCATE THE PERSON', 'person'),
    ],
)
def test_bounded_room_search_extracts_target(transcript, target):
    assert is_search_request(transcript) is True
    assert search_target(transcript) == target


def test_approach_is_not_mistaken_for_room_search():
    assert is_search_request('MOVE TOWARD THE BLUE OBJECT') is False
    assert is_search_request('TURN LEFT') is False


@pytest.mark.parametrize(
    'transcript,fact',
    [
        ('REMEMBER THAT MY NAME IS MOHAMMAD', 'my name is mohammad'),
        ('PLEASE REMEMBER MY FAVORITE COLOR IS BLUE', 'my favorite color is blue'),
        ("DON'T FORGET THAT THE KITCHEN IS TO THE LEFT", 'the kitchen is to the left'),
    ],
)
def test_explicit_memory_commands_extract_fact(transcript, fact):
    assert memory_fact(transcript) == fact


def test_normal_conversation_is_not_saved_as_long_term_memory():
    assert memory_fact('What is my favorite color?') == ''
    assert memory_fact('Move forward') == ''


@pytest.mark.parametrize(
    'transcript',
    ['', 'MOVE FORWARD', 'TURN LEFT', 'STOP', 'MOVE TOWARD THE RED OBJECT'],
)
def test_commands_are_not_describe_requests(transcript):
    assert is_describe_request(transcript) is False


def test_describe_question_never_becomes_a_motion_intent():
    # A direction word inside the question must not reach the wheels.
    for transcript in ('WHAT DO YOU SEE ON THE LEFT', 'WHAT DO YOU SEE AHEAD'):
        assert is_describe_request(transcript) is True
        assert match_intent(transcript) is None


def test_clean_description_strips_think_blocks_and_json():
    assert clean_description('<think>hmm</think> A red ball is on my left.') == (
        'A red ball is on my left.'
    )
    assert clean_description('{"description": "A blue box ahead."}') == 'A blue box ahead.'
    assert clean_description('```json\n{"say": "Nothing but floor."}\n```') == (
        'Nothing but floor.'
    )
    assert clean_description('   ') == ''


def test_description_cap_allows_a_full_sentence():
    assert DESCRIBE_SPEAK_MAX_CHARS > SPEAK_PLAY_MAX_CHARS


@pytest.mark.parametrize(
    'transcript',
    [
        "WHAT'S YOUR PLAN TO MOVE TOWARD THE RED OBJECT",
        'WHAT IS YOUR PLAN TO GET TO THE BLUE OBJECT',
        'WHAT WOULD YOU DO TO APPROACH THE CHAIR',
        'HOW WOULD YOU MOVE TOWARD THE BOX',
        'DESCRIBE YOUR PLAN TO REACH THE DOOR',
        'PLAN HOW TO APPROACH THE PERSON',
    ],
)
def test_plan_preview_questions_are_recognized(transcript):
    assert is_plan_preview_request(transcript) is True
    assert match_intent(transcript) is None


@pytest.mark.parametrize(
    'transcript',
    ['', 'MOVE FORWARD', 'MOVE TOWARD THE RED OBJECT', 'WHAT DO YOU SEE'],
)
def test_execution_and_description_are_not_plan_previews(transcript):
    assert is_plan_preview_request(transcript) is False


@pytest.mark.parametrize(
    'transcript, target',
    [
        ('MOVE AROUND THE OBJECT IN FRONT OF YOU', 'object'),
        ('GO AROUND THE OBJECT IN FRONT OF YOU', 'object'),
        ('MOVE AROUND THE BOX', 'box'),
        ('DRIVE AROUND THE OBJECT', 'object'),
        ('GO ALL THE WAY AROUND THE RED OBJECT', 'red object'),
        ('CIRCLE THE CHAIR', 'chair'),
        ('PLEASE GO AROUND THE BOX ON YOUR LEFT', 'box'),
    ],
)
def test_around_requests_are_recognized_as_movement(transcript, target):
    assert is_around_request(transcript) is True
    assert around_target(transcript) == target
    # A detour must not be answered by a parked speech-only route.
    assert is_visual_question(transcript) is False
    assert is_describe_request(transcript) is False
    assert is_plan_preview_request(transcript) is False


@pytest.mark.parametrize(
    'transcript',
    ['', 'MOVE FORWARD', 'WHAT DO YOU SEE', 'MOVE TOWARD THE BLUE OBJECT'],
)
def test_plain_commands_are_not_detours(transcript):
    assert is_around_request(transcript) is False


@pytest.mark.parametrize(
    'transcript',
    [
        # Each names an object and its position, which used to be enough to
        # match the visual-question pattern and swallow the drive request.
        'MOVE AROUND THE OBJECT IN FRONT OF YOU',
        'GO PAST THE OBJECT ON THE LEFT',
        'DRIVE TOWARD THE OBJECT IN FRONT',
        'TURN TO THE OBJECT ON THE RIGHT',
    ],
)
def test_motion_commands_are_never_visual_questions(transcript):
    assert is_motion_command(transcript) is True
    assert is_visual_question(transcript) is False


@pytest.mark.parametrize(
    'transcript',
    [
        'WHAT COLOR IS IT',
        'WHAT IS THAT',
        'TELL ME ABOUT THE OBJECT YOU SEE',
        'WHERE IS THE RED OBJECT',
    ],
)
def test_questions_are_not_motion_commands(transcript):
    assert is_motion_command(transcript) is False
    assert is_visual_question(transcript) is True


def test_ack_phrases_are_short_enough_to_speak():
    assert ack_phrase('forward') == 'Moving forward'
    assert ack_phrase('back') == 'Moving backward'
    assert ack_phrase('left') == 'Turning left'
    assert ack_phrase('right') == 'Turning right'
    assert ack_phrase('stop') == 'Stopping'
    for phrase in ACK_PHRASES.values():
        assert 0 < len(phrase) <= SPEAK_PLAY_MAX_CHARS
