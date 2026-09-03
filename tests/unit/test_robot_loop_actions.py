from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jetbot_agent.robot_loop.actions import (
    MAX_DURATION_S,
    SPEAK_MAX_CHARS,
    VX_MAX,
    WZ_MAX,
    extract_json_object,
    parse_action,
    parse_model_output,
)

# Cosmos answered correctly, then overran the 96-token cap describing itself in
# "reason", leaving the object unterminated.
COSMOS_TRUNCATED_REASON = (
    '{"visible":true,"side":"right","goal":"blue object",'
    '"plan":[{"step":"arc_right","ticks":2}],'
    '"reason":"The blue object is on the right side and'
)

# Real Cosmos-Reason2-2B drive-mode fragment: invalid TTS quote, nonzero stop.
COSMOS_BROKEN_SAY = (
    '{"action":"stop","vx":0.01,"wz":0.01,"duration_s":0.13,'
    '"say":"\\"","goal":"avoid red object",'
    '"reason":"The red object is in the path, causing a potential collision."}'
)


def test_output_cut_off_mid_string_keeps_the_finished_fields():
    recovered = extract_json_object(COSMOS_TRUNCATED_REASON)

    assert recovered['visible'] is True
    assert recovered['side'] == 'right'
    assert recovered['plan'] == [{'step': 'arc_right', 'ticks': 2}]
    assert 'reason' not in recovered


def test_salvage_never_invents_a_value_that_was_cut_off():
    assert 'side' not in extract_json_object('{"visible":true,"side":')
    assert extract_json_object(
        '{"visible":true,"side":"left","plan":[{"step":"arc_left","tic'
    )['plan'] == [{'step': 'arc_left'}]


def test_salvage_does_not_rescue_prose_or_emptiness():
    with pytest.raises(Exception):
        extract_json_object('I think I should probably turn left now')
    with pytest.raises(Exception):
        extract_json_object('')


def test_valid_drive():
    action = parse_action(
        '{"action":"drive","vx":0.2,"wz":-0.3,"duration_s":0.5,'
        '"say":"go","goal":"door","reason":"clear"}'
    )
    assert action.kind == 'drive'
    assert action.action == 'drive'
    assert action.vx == pytest.approx(0.2)
    assert action.wz == pytest.approx(-0.3)
    assert action.duration_s == pytest.approx(0.5)
    assert action.say == ''
    assert action.text == ''
    assert action.goal == 'door'
    assert action.reason == 'clear'
    assert action.then_stop() is True


def test_drive_clamps_vx_wz_and_duration():
    action = parse_action(
        json.dumps({'action': 'drive', 'vx': 9.0, 'wz': -4.0, 'duration_s': 99})
    )
    assert action.kind == 'drive'
    assert action.vx == pytest.approx(VX_MAX)
    assert action.wz == pytest.approx(-WZ_MAX)
    assert action.duration_s == pytest.approx(MAX_DURATION_S)
    assert MAX_DURATION_S == pytest.approx(2.0)


def test_stop_with_nonzero_vx_is_zeroed():
    action = parse_action('{"action":"stop","vx":0.2,"wz":0.5,"duration_s":1,"say":"halt"}')
    assert action.kind == 'stop'
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert action.say == ''
    assert action.text == ''


def test_invalid_json_is_stop():
    action = parse_action('not json at all')
    assert action.kind == 'stop'
    assert action.raw_ok is False
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert action.say == ''
    assert action.text == ''
    assert action.reason == 'parse_fail'


def test_broken_say_quote_is_stop_and_not_spoken():
    action = parse_action(COSMOS_BROKEN_SAY)
    assert action.kind == 'stop'
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert action.say == ''
    assert action.text == ''
    assert '"' not in action.text
    assert action.reason != action.text
    dumped = parse_model_output(COSMOS_BROKEN_SAY)
    assert dumped['say'] == ''
    assert dumped['vx'] == 0.0
    assert dumped['wz'] == 0.0


def test_truncated_say_is_capped():
    long_say = 'a' * 200
    action = parse_action(
        json.dumps({'action': 'speak', 'vx': 0.1, 'wz': 0.2, 'say': long_say})
    )
    assert action.kind == 'speak'
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert len(action.say) == SPEAK_MAX_CHARS
    assert action.say == long_say[:SPEAK_MAX_CHARS]
    assert action.text == action.say
    assert SPEAK_MAX_CHARS == 120


def test_empty_or_broken_speak_becomes_stop():
    empty = parse_action('{"action":"speak","say":""}')
    assert empty.kind == 'stop'
    assert empty.vx == 0.0
    assert empty.text == ''
    quote = parse_action('{"action":"speak","say":"\\""}')
    assert quote.kind == 'stop'
    assert quote.text == ''


def test_weather_zeros_velocity():
    action = parse_action(
        '{"action":"weather","vx":0.2,"wz":0.9,"goal":"Austin","reason":"asked"}'
    )
    assert action.kind == 'weather'
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert action.goal == 'Austin'
    assert action.query == 'Austin'
    assert action.say == ''
    assert action.text == ''
    assert action.reason == 'asked'


def test_extra_keys_ignored():
    action = parse_action(
        json.dumps(
            {
                'action': 'wait',
                'vx': 0.22,
                'wz': 1.0,
                'duration_s': 1.25,
                'say': 'nope',
                'goal': 'hold',
                'reason': 'debug only',
                'pwm': 255,
                'left_motor': 0.9,
                'extra': {'nested': True},
            }
        )
    )
    assert action.kind == 'wait'
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert action.duration_s == pytest.approx(1.25)
    assert action.say == ''
    assert action.text == ''
    assert action.goal == 'hold'
    assert action.reason == 'debug only'
    dumped = action.as_dict()
    assert 'pwm' not in dumped
    assert 'extra' not in dumped
    assert set(dumped) == {
        'action',
        'vx',
        'wz',
        'duration_s',
        'say',
        'goal',
        'reason',
    }


def test_markdown_wrapper_extracts_first_object():
    text = (
        'here is the plan\n```json\n'
        '{"action":"drive","vx":9,"wz":9,"duration_s":0.4,"say":"go"}\n'
        '```\ntrailing prose'
    )
    action = parse_action(text)
    assert action.kind == 'drive'
    assert action.vx == pytest.approx(VX_MAX)
    assert action.wz == pytest.approx(WZ_MAX)
    assert action.duration_s == pytest.approx(0.4)


def test_empty_and_non_object_are_stop():
    assert parse_action('').kind == 'stop'
    assert parse_action(None).kind == 'stop'
    assert parse_action('[]').kind == 'stop'
    assert parse_action('1').kind == 'stop'


def test_unknown_kind_is_stop():
    assert parse_action('{"action": "fly"}').kind == 'stop'
    assert parse_action('{"type": "explode"}').kind == 'stop'


def test_drive_rejects_nonfinite_and_boolean_numbers():
    action = parse_action(
        {'action': 'drive', 'vx': float('nan'), 'wz': True, 'duration_s': float('inf')}
    )
    assert action.kind == 'drive'
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert action.duration_s == 0.0


def test_mapping_input_and_bytes():
    assert parse_action({'action': 'stop'}).kind == 'stop'
    assert parse_action(b'{"action": "stop"}').kind == 'stop'


def test_reason_is_never_the_tts_string():
    action = parse_action(
        '{"action":"speak","say":"hello","reason":"do not speak this"}'
    )
    assert action.text == 'hello'
    assert action.reason == 'do not speak this'
    assert action.as_dict()['say'] != action.reason
