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
    INTENT_TURN_DURATION_S,
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
    is_where_request,
    search_target,
    search_wants_approach,
    where_target,
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
    assert left.duration_s == pytest.approx(INTENT_TURN_DURATION_S)
    assert right.duration_s == pytest.approx(INTENT_TURN_DURATION_S)
    # +wz is a left turn, matching Robot.left(): left wheel back, right wheel forward.
    assert left.wz > 0.0
    assert right.wz < 0.0
    left_wheels = unicycle_wheels(left.vx, left.wz)
    right_wheels = unicycle_wheels(right.vx, right.wz)
    assert left_wheels[0] < 0.0 < left_wheels[1]
    assert right_wheels[1] < 0.0 < right_wheels[0]


def test_every_motion_intent_uses_the_same_measured_duty():
    """Forward/back use the travel pulse; turns use the calibrated swing pulse."""
    assert NUDGE_VX == pytest.approx(0.65)
    assert NUDGE_DURATION_S == pytest.approx(1.2)
    assert INTENT_TURN_DURATION_S == pytest.approx(0.15)
    assert intent_wheels('forward') == (NUDGE_VX, NUDGE_VX)
    assert intent_wheels('back') == (-NUDGE_VX, -NUDGE_VX)
    assert intent_wheels('left') == (-NUDGE_VX, NUDGE_VX)
    assert intent_wheels('right') == (NUDGE_VX, -NUDGE_VX)
    assert intent_wheels('stop') == (0.0, 0.0)
    for name in ('forward', 'back'):
        action = intent_action(name)
        assert action.duration_s == pytest.approx(NUDGE_DURATION_S)
        wheels = unicycle_wheels(action.vx, action.wz)
        assert wheels == pytest.approx(intent_wheels(name))
        assert abs(wheels[0]) == pytest.approx(NUDGE_VX)
        assert abs(wheels[1]) == pytest.approx(NUDGE_VX)
    for name in ('left', 'right'):
        action = intent_action(name)
        assert action.duration_s == pytest.approx(INTENT_TURN_DURATION_S)
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
        expected_duration = (
            INTENT_TURN_DURATION_S if intent in ('left', 'right') else NUDGE_DURATION_S
        )
        assert action.duration_s == pytest.approx(expected_duration)
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
        ('ON THE ROOM FOR THE BLUE OBJECT', 'blue object'),
        ('IN THE ROOM FOR MY BALL', 'my ball'),
        ('FIND THE GREEN TOY', 'green toy'),
        ('LOCATE THE PERSON', 'person'),
    ],
)
def test_bounded_room_search_extracts_target(transcript, target):
    assert is_search_request(transcript) is True
    assert search_target(transcript) == target


@pytest.mark.parametrize(
    'transcript',
    [
        'FIND THE BLUE OBJECT AND GO TO IT',
        'FIND THE RED OBJECT IN THE ROOM AND GO TO IT',
        'LOOK AROUND THE ROOM FOR THE BLUE OBJECT AND DRIVE TO IT',
        'FIND THE BLUE OBJECT AND APPROACH IT',
        'FIND THE BLUE OBJECT AND COME OVER TO IT',
    ],
)
def test_compound_search_names_object_and_requests_approach(transcript):
    # The follow-on clause and the room locative are not part of the object
    # name, and the request is not satisfied by looking alone.
    assert search_target(transcript).endswith('object')
    assert 'go to it' not in search_target(transcript)
    assert search_wants_approach(transcript) is True


@pytest.mark.parametrize(
    'transcript',
    [
        'FIND THE BLUE OBJECT',
        'LOOK AROUND THE ROOM FOR THE BLUE OBJECT',
        'FIND THE BLUE OBJECT AND TELL ME WHERE IT IS',
    ],
)
def test_look_only_search_does_not_request_approach(transcript):
    assert is_search_request(transcript) is True
    assert search_wants_approach(transcript) is False


@pytest.mark.parametrize(
    'transcript',
    [
        'WHAT DO YOU SEE IN FRONT OF YOU',
        # Zipformer heard this live and the reply became a nonsense sentence.
        'WHAT DO YOU SEE IN FRONTING YOU',
        'WHAT DO YOU SEE AHEAD',
        'WHAT DO YOU SEE OVER THERE',
        'DO YOU SEE ANYTHING',
        'WHAT DO YOU SEE ON YOUR LEFT',
    ],
)
def test_a_viewpoint_is_described_not_located(transcript):
    # "In front of you" names no object, so the where-route must not answer
    # "I see the in front of you on my left".
    assert where_target(transcript) == ''
    assert is_where_request(transcript) is False
    assert is_describe_request(transcript) is True


@pytest.mark.parametrize(
    'transcript,target',
    [
        ('WHERE IS MY BALL', 'my ball'),
        ('DO YOU SEE THE BLUE OBJECT', 'blue object'),
        ('WHERE IS THE RED BOX', 'red box'),
    ],
)
def test_a_named_object_still_reaches_the_where_route(transcript, target):
    assert where_target(transcript) == target
    assert is_where_request(transcript) is True


def test_approach_is_not_mistaken_for_room_search():
    assert is_search_request('MOVE TOWARD THE BLUE OBJECT') is False
    assert search_wants_approach('MOVE TOWARD THE BLUE OBJECT') is False
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
        ('PLEASE GO AROUND THE BOX ON YOUR LEFT', 'box'),
    ],
)
def test_around_requests_are_recognized_as_movement(transcript, target):
    assert is_around_request(transcript) is True
    assert around_target(transcript) == target


@pytest.mark.parametrize(
    'transcript, target',
    [
        ('GO ALL THE WAY AROUND THE RED OBJECT', 'red object'),
        ('CIRCLE THE CHAIR', 'chair'),
    ],
)
def test_full_circle_phrasing_asks_for_an_orbit_not_a_pass_by(transcript, target):
    from jetbot_agent.robot_loop.intents import behind_target, is_behind_request

    # These used to route to the detour, which only ever passes an object and
    # would have reported success after a sidestep.
    assert is_behind_request(transcript) is True
    assert behind_target(transcript) == target
    assert is_around_request(transcript) is False
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


def test_behind_requests_are_recognised_and_targets_extracted():
    from jetbot_agent.robot_loop.intents import behind_target, is_behind_request

    assert is_behind_request('get behind the red object') is True
    assert behind_target('get behind the red object') == 'red object'
    assert behind_target('go behind the blue box please') == 'blue box'
    assert behind_target('get to the other side of the red truck') == 'red truck'
    assert behind_target('circle the green chair') == 'green chair'
    assert behind_target('orbit around the red object') == 'red object'
    assert behind_target('drive all the way around the blue bin') == 'blue bin'
    assert behind_target('go behind the object in front of you') == 'object'


def test_a_pass_by_is_not_mistaken_for_a_half_orbit():
    from jetbot_agent.robot_loop.intents import (
        around_target,
        is_around_request,
        is_behind_request,
    )

    # A detour and a half orbit are different maneuvers with different failure
    # modes, so the two routes must not overlap.
    assert is_behind_request('go around the red object') is False
    assert is_around_request('go around the red object') is True
    assert is_around_request('get behind the red object') is False
    # "All the way around" and "circle" mean a full orbit, not a pass-by.
    assert is_around_request('drive all the way around the blue bin') is False
    assert around_target('drive past the red object') == 'red object'


def test_behind_requests_still_count_as_motion_commands():
    from jetbot_agent.robot_loop.intents import is_motion_command, is_visual_question

    # A motion verb must veto the speak-only routes, or the robot answers a
    # drive request with a sentence.
    assert is_motion_command('get behind the red object') is True
    assert is_visual_question('get behind the red object') is False


def test_show_and_tell_and_creep_and_think_intents():
    from jetbot_agent.robot_loop.intents import (
        is_creep_request,
        is_deictic_target,
        is_show_and_tell,
        is_think_request,
        is_visual_question,
    )

    assert is_show_and_tell('WHAT AM I HOLDING') is True
    assert is_show_and_tell('what is this object') is True
    assert is_show_and_tell('move forward') is False
    assert is_creep_request('If the floor is clear, creep forward') is True
    assert is_creep_request('creep ahead') is True
    assert is_think_request('Think hard whether that path is safe') is True
    assert is_think_request('drive toward that') is False
    assert is_deictic_target('DRIVE TOWARD THAT') is True
    assert is_deictic_target('move toward the red object') is False
    # Holding uses the JPEG-only route, not the RAG visual follow-up.
    assert is_visual_question('what am I holding') is False or is_show_and_tell(
        'what am I holding'
    )


def test_where_is_and_place_intents():
    from jetbot_agent.robot_loop.intents import (
        is_place_query,
        is_place_teach,
        is_where_request,
        place_name,
        place_query_name,
        where_target,
    )

    assert is_where_request('Where is the blue backpack?') is True
    assert where_target('Where is the blue backpack?') == 'blue backpack'
    assert is_where_request('where is this') is False
    assert is_place_teach('This view is the kitchen corner') is True
    assert place_name('This view is the kitchen corner') == 'kitchen corner'
    assert is_place_query('Are we at the kitchen corner?') is True
    assert place_query_name('Are we at the kitchen corner') == 'kitchen corner'
    assert is_place_query('is this the kitchen corner') is True
    assert is_place_query('Are we at this') is False
    assert is_place_query('is this a toy') is False
