from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'bringup'))

from jetbot_agent.robot_loop.actions import parse_action  # noqa: E402

from talk_and_drive import (  # noqa: E402
    SPEAK_PLAY_MAX_CHARS,
    TEST_DURATION_MAX_S,
    UNDERSTAND_FAIL_PHRASE,
    TalkDriveExecutor,
    asr_transcript_usable,
    clamp_test_action,
    unicycle_wheels,
)


def test_test_clamp_caps_duration_to_half_second():
    action = parse_action(
        json.dumps({'action': 'drive', 'vx': 9.0, 'wz': -4.0, 'duration_s': 99})
    )
    clamped = clamp_test_action(action)
    assert clamped.kind == 'drive'
    assert clamped.vx == 0.22
    assert clamped.wz == -1.0
    assert clamped.duration_s == TEST_DURATION_MAX_S
    assert TEST_DURATION_MAX_S == 0.5


def test_parse_fail_is_stop():
    action = clamp_test_action(parse_action('not json'))
    assert action.kind == 'stop'
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert action.raw_ok is False


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
