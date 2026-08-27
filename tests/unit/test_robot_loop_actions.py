from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jetbot_agent.robot_loop.actions import (
    MAX_DURATION_S,
    VX_MAX,
    WZ_MAX,
    parse_action,
)


def test_invalid_json_is_stop():
    action = parse_action('not json at all')
    assert action.kind == 'stop'
    assert action.raw_ok is False
    assert action.vx == 0.0
    assert action.wz == 0.0


def test_empty_and_non_object_are_stop():
    assert parse_action('').kind == 'stop'
    assert parse_action(None).kind == 'stop'
    assert parse_action('[]').kind == 'stop'
    assert parse_action('1').kind == 'stop'


def test_unknown_kind_is_stop():
    assert parse_action('{"action": "fly"}').kind == 'stop'
    assert parse_action('{"type": "explode"}').kind == 'stop'


def test_stop_action():
    action = parse_action('{"action": "stop"}')
    assert action.kind == 'stop'
    assert action.raw_ok is True


def test_drive_clamps_vx_wz_and_duration():
    action = parse_action(
        json.dumps({'action': 'drive', 'vx': 9.0, 'wz': -4.0, 'duration': 99})
    )
    assert action.kind == 'drive'
    assert action.vx == pytest.approx(VX_MAX)
    assert action.wz == pytest.approx(-WZ_MAX)
    assert action.duration_s == pytest.approx(MAX_DURATION_S)
    assert action.then_stop() is True


def test_drive_aliases_linear_angular():
    action = parse_action('{"action": "DRIVE", "linear": -0.1, "angular": 0.5, "duration_s": 0.4}')
    assert action.kind == 'drive'
    assert action.vx == pytest.approx(-0.1)
    assert action.wz == pytest.approx(0.5)
    assert action.duration_s == pytest.approx(0.4)


def test_speak_and_wait_and_weather():
    speak = parse_action('{"action": "speak", "text": "hello"}')
    assert speak.kind == 'speak'
    assert speak.text == 'hello'
    wait = parse_action('{"action": "wait", "duration": 1.25}')
    assert wait.kind == 'wait'
    assert wait.duration_s == pytest.approx(1.25)
    weather = parse_action('{"action": "weather", "query": "Austin"}')
    assert weather.kind == 'weather'
    assert weather.query == 'Austin'


def test_mapping_input_and_bytes():
    assert parse_action({'action': 'stop'}).kind == 'stop'
    assert parse_action(b'{"action": "stop"}').kind == 'stop'
