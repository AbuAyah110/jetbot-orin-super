from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jetbot_agent.robot_loop.actions import parse_action
from jetbot_agent.robot_loop.csi_jpeg import CsiJpeg448
from jetbot_agent.robot_loop.log_executor import LogOnlyExecutor
from jetbot_agent.robot_loop.orchestrator import LoopInput, OneProcessOrchestrator
from jetbot_agent.robot_loop.prompts import DRIVE_PROMPT_SUFFIX


class FakeRuntime:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.text


class RecordingExecutor(LogOnlyExecutor):
    pass


def test_pipeline_is_one_csi_448():
    pipe = CsiJpeg448().gst_pipeline()
    assert pipe.count('nvarguscamerasrc') == 1
    assert 'num-buffers=0' not in pipe
    assert 'width=(int)448' in pipe
    assert 'height=(int)448' in pipe
    assert 'nvjpegenc' in pipe
    oneshot = CsiJpeg448(num_buffers=1).gst_pipeline()
    assert 'num-buffers=1' in oneshot
    assert oneshot.count('nvarguscamerasrc') == 1


class FakeAppsink:
    """Stands in for the GStreamer appsink so warm-up is testable off-robot."""

    def __init__(self, available=1000):
        self.available = available
        self.pulls = 0

    def emit(self, _signal, _timeout_ns):
        self.pulls += 1
        if self.pulls > self.available:
            return None
        return object()


def test_warmup_discards_frames_until_exposure_settles():
    """Argus opens ~3x underexposed; early frames must not reach the VLM."""
    camera = CsiJpeg448(warmup_s=2.5)
    camera._appsink = FakeAppsink()
    ticks = iter([0.0] + [i * 0.1 for i in range(1, 100)])
    clock = lambda: next(ticks)  # noqa: E731

    dropped = camera._drain_warmup(now=clock)

    assert dropped > 0
    assert camera.warmup_frames_dropped == dropped


def test_oneshot_capture_keeps_its_only_frame():
    camera = CsiJpeg448(num_buffers=1, warmup_s=2.5)
    camera._appsink = FakeAppsink()

    assert camera._drain_warmup() == 0
    assert camera._appsink.pulls == 0


def test_warmup_stops_early_when_the_sensor_stalls():
    camera = CsiJpeg448(warmup_s=2.5)
    camera._appsink = FakeAppsink(available=3)

    assert camera._drain_warmup() == 3


def test_log_executor_never_moves():
    exe = LogOnlyExecutor()
    exe.execute(parse_action('{"action":"drive","vx":0.2,"wz":0.5}'))
    assert exe.is_moving() is False
    assert exe.last.kind == 'drive'
    assert exe.last.vx == 0.2


def test_drive_mode_no_think_and_token_clamp():
    runtime = FakeRuntime('{"action":"stop"}')
    orch = OneProcessOrchestrator(
        runtime, LogOnlyExecutor(), drive_mode=True, drive_max_tokens=80
    )
    orch.tick(LoopInput(image_jpeg=b'\xff\xd8'))
    assert runtime.calls
    assert runtime.calls[0]['max_tokens'] == 80
    assert DRIVE_PROMPT_SUFFIX in runtime.calls[0]['user_text']
    assert 'extended thinking' in runtime.calls[0]['user_text'].lower() or 'Do not use extended thinking' in runtime.calls[0]['user_text']


def test_object_relative_prompt_requires_grounded_visible_side_and_omits_history():
    runtime = FakeRuntime('{"action":"stop","goal":"not_visible:red object"}')
    orch = OneProcessOrchestrator(runtime, LogOnlyExecutor(), drive_mode=True)
    orch.history.add('assistant', 'stale direction left')
    orch.plan(LoopInput(speech='MOVE TOWARDS THE RED OBJECT', image_jpeg=b'\xff\xd8'))
    prompt = runtime.calls[0]['user_text']
    assert 'VISUAL GROUNDING TEST' in prompt
    assert 'visible:left' in prompt
    assert 'not_visible:<target>' in prompt
    assert 'stale direction left' not in prompt
    assert runtime.calls[0]['image_jpeg'] == b'\xff\xd8'


def test_invalid_json_logs_stop_and_hold_stop_before_infer():
    runtime = FakeRuntime('not json')
    exe = LogOnlyExecutor()
    orch = OneProcessOrchestrator(runtime, exe, drive_mode=True)
    action = orch.tick(LoopInput())
    assert action.kind == 'stop'
    assert action.raw_ok is False
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert exe.history[0].kind == 'stop'
    assert exe.history[-1].kind == 'stop'


def test_clamped_drive_is_logged_not_spoken():
    payload = json.dumps(
        {'action': 'drive', 'vx': 9, 'wz': -4, 'duration_s': 1, 'say': 'nope'}
    )
    runtime = FakeRuntime(payload)
    exe = LogOnlyExecutor()
    orch = OneProcessOrchestrator(runtime, exe, drive_mode=True)
    action = orch.tick(LoopInput())
    assert action.kind == 'drive'
    assert action.vx == 0.22
    assert action.wz == -1.0
    assert action.say == ''
    assert exe.is_moving() is False
