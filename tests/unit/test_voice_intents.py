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
    intent_action,
    intent_wheels,
    match_intent,
)

from talk_and_drive import (  # noqa: E402
    SPEAK_PLAY_MAX_CHARS,
    clamp_test_action,
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
    'transcript',
    [
        '',
        '   ',
        'WHAT IS THE WEATHER TODAY',
        'TELL ME A JOKE',
        'WHO ARE YOU',
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


def test_ack_phrases_are_short_enough_to_speak():
    assert ack_phrase('forward') == 'Moving forward'
    assert ack_phrase('back') == 'Moving backward'
    assert ack_phrase('left') == 'Turning left'
    assert ack_phrase('right') == 'Turning right'
    assert ack_phrase('stop') == 'Stopping'
    for phrase in ACK_PHRASES.values():
        assert 0 < len(phrase) <= SPEAK_PLAY_MAX_CHARS
