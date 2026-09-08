from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'lab'))

from chassis_wizard import (  # noqa: E402
    AbortWizard,
    DUTY_MAX,
    LabLog,
    PULSE_DURATION_MAX,
    RecordingRobot,
    Wizard,
    arc_wheels,
    clamp_pulse,
    derive_calibration,
    jpeg_mean_abs_diff,
    likely_no_translate,
    main,
    mean_abs_diff_bytes,
    parse_verdict,
)


def test_refuses_without_floor_or_wheels_up():
    assert main([]) == 2


def test_clamp_pulse_caps_duty_and_duration():
    pulse = clamp_pulse(9.0, -4.0, 99.0)
    assert pulse.left == DUTY_MAX == 0.7
    assert pulse.right == -DUTY_MAX
    assert pulse.duration_s == PULSE_DURATION_MAX == 1.5


def test_parse_verdict_keys_and_speech():
    assert parse_verdict('p') == 'pass'
    assert parse_verdict('h') == 'hummed'
    assert parse_verdict('v') == 'pivoted'
    assert parse_verdict('f') == 'fought'
    assert parse_verdict('s') == 'skip'
    assert parse_verdict('q') == 'abort'
    assert parse_verdict('n') == 'straight'
    assert parse_verdict('t') == 'too_short'
    assert parse_verdict('r') == 'about_right'
    assert parse_verdict('d') == 'too_far'
    assert parse_verdict('it only hummed') == 'hummed'
    assert parse_verdict('pivoted in place') == 'pivoted'
    assert parse_verdict('wheels fought') == 'fought'
    assert parse_verdict('curved while advancing') == 'pass'
    assert parse_verdict('went straight') == 'straight'
    assert parse_verdict('too short') == 'too_short'
    assert parse_verdict('too far') == 'too_far'
    assert parse_verdict('about right') == 'about_right'
    assert parse_verdict('abort') == 'abort'
    assert parse_verdict('MOVE TOWARD THE RED OBJECT') is None


def test_stiction_floor_is_first_roll_duty():
    suggested = derive_calibration(
        [
            {
                'kind': 'trial',
                'phase': 'stiction',
                'left': 0.20,
                'right': 0.20,
                'verdict': 'hummed',
            },
            {
                'kind': 'trial',
                'phase': 'stiction',
                'left': 0.50,
                'right': 0.50,
                'verdict': 'pass',
            },
            {
                'kind': 'trial',
                'phase': 'stiction',
                'left': 0.65,
                'right': 0.65,
                'verdict': 'pass',
            },
        ],
        run_id='lab-test',
        battery='ok',
        measured_on='2026-08-27',
    )
    assert suggested.stiction_floor == pytest.approx(0.50)
    assert suggested.speed == pytest.approx(0.50)


def test_arc_inner_from_last_advancing_pass():
    suggested = derive_calibration(
        [
            {
                'kind': 'trial',
                'phase': 'arc',
                'left': 0.50,
                'right': 0.70,
                'verdict': 'pivoted',
            },
            {
                'kind': 'trial',
                'phase': 'arc',
                'left': 0.55,
                'right': 0.70,
                'verdict': 'pass',
            },
        ],
        run_id='lab-test',
    )
    assert suggested.arc_inner_floor == pytest.approx(0.55)
    assert any('pivoted' in note for note in suggested.notes)


def test_duration_prefers_tagged_length():
    suggested = derive_calibration(
        [
            {
                'kind': 'trial',
                'phase': 'duration',
                'duration_s': 0.6,
                'left': 0.65,
                'right': 0.65,
                'verdict': 'about_right',
            },
            {
                'kind': 'trial',
                'phase': 'duration',
                'duration_s': 1.2,
                'left': 0.65,
                'right': 0.65,
                'verdict': 'too_far',
            },
        ],
        run_id='lab-test',
    )
    assert suggested.duration_s == pytest.approx(0.6)


def test_arc_wheels_keep_inner_at_or_above_floor():
    left, right = arc_wheels('arc_left', base=0.65, inner_floor=0.55)
    assert min(left, right) >= 0.55
    assert left < right
    left, right = arc_wheels('arc_right', base=0.65, inner_floor=0.55)
    assert min(left, right) >= 0.55
    assert right < left


def test_jpeg_equal_bytes_are_still():
    jpeg = b'\xff\xd8fakejpeg\xff\xd9'
    assert jpeg_mean_abs_diff(jpeg, jpeg) == 0.0
    assert likely_no_translate(0.0) is True
    assert mean_abs_diff_bytes(b'\x00\x00', b'\x0a\x00') == pytest.approx(5.0)


def _wizard(tmp_path: Path, verdicts, *, use_csi: bool = False, frames=None):
    robot = RecordingRobot()
    log = LabLog(tmp_path, 'unit-run')
    queued = list(verdicts)
    frames = list(frames or [])

    def capture() -> bytes:
        if not frames:
            return b'\xff\xd8same\xff\xd9'
        return frames.pop(0)

    wizard = Wizard(
        robot=robot,
        log=log,
        dry_run=True,
        announce=lambda text: None,
        ask_verdict=lambda _phase: queued.pop(0),
        capture_jpeg=capture if use_csi else None,
        sleep=lambda _seconds: None,
        use_csi=use_csi,
    )
    return wizard, robot, log, queued


def test_dry_run_floor_protocol_writes_jsonl_and_stops(tmp_path: Path):
    verdicts = ['pass'] * 8
    wizard, robot, log, queued = _wizard(tmp_path, verdicts)
    suggested = wizard.run_protocol(wheels_up=False, on_floor=True, battery='ok')
    assert queued == []
    assert robot.stopped >= 1
    trials = [event for event in log.events if event.get('kind') == 'trial']
    phases = [event['phase'] for event in trials]
    assert phases[0] == 'stiction'
    assert 'polarity' in phases
    assert 'arc' in phases
    assert phases[-2:] == ['duration', 'duration']
    assert all(abs(event['left']) <= 0.7 for event in trials)
    assert all(event['duration_s'] <= 1.5 for event in trials)
    assert suggested.stiction_floor == pytest.approx(0.20)
    lines = log.path.read_text(encoding='utf-8').strip().splitlines()
    assert len(lines) >= len(trials)
    assert json.loads(lines[0])['kind'] == 'session_start'


def test_arc_pivot_retries_with_raised_inner_floor(tmp_path: Path):
    wizard, robot, log, queued = _wizard(
        tmp_path,
        ['pass', 'pass', 'pass', 'pass', 'pivoted', 'pass', 'pass', 'about_right', 'about_right'],
    )
    wizard.run_protocol(wheels_up=False, on_floor=True, battery='low')
    assert queued == []
    arcs = [event for event in log.events if event.get('phase') == 'arc']
    assert arcs[0]['verdict'] == 'pivoted'
    assert arcs[1]['attempt'] == 2
    assert arcs[1]['inner_floor'] == pytest.approx(0.55)
    assert min(abs(arcs[1]['left']), abs(arcs[1]['right'])) >= 0.55
    pulses = [(p.left, p.right, p.duration_s) for p in robot.pulses]
    assert pulses
    assert all(dur <= 1.5 for _l, _r, dur in pulses)


def test_abort_stops_motors_and_halts_protocol(tmp_path: Path):
    wizard, robot, log, queued = _wizard(tmp_path, ['pass', 'abort'])
    with pytest.raises(AbortWizard):
        wizard.run_protocol(wheels_up=True, on_floor=False, battery='ok')
    assert queued == []
    assert robot.stopped >= 1
    assert wizard.aborted is True
    trials = [event for event in log.events if event.get('kind') == 'trial']
    assert trials[-1]['verdict'] == 'abort'
    assert trials[-1]['phase'] == 'twitch'


def test_csi_before_after_hashes_and_still_flag(tmp_path: Path):
    blob = b'\xff\xd8same-frame\xff\xd9'
    wizard, _robot, log, _queued = _wizard(
        tmp_path, ['pass'], use_csi=True, frames=[blob, blob]
    )
    wizard.trial(
        phase='stiction',
        announce='Both wheels.',
        left=0.5,
        right=0.5,
        duration_s=1.2,
    )
    event = log.events[-1]
    assert event['frame_before'] == event['frame_after']
    assert event['csi_mad'] == 0.0
    assert event['csi_no_translate'] is True
    saved = list((tmp_path / 'frames').glob('*.jpg'))
    assert len(saved) == 2


def test_main_dry_run_writes_report(tmp_path: Path):
    lab = tmp_path / 'lab'
    code = main(
        [
            '--on-floor',
            '--dry-run',
            '--keys-only',
            '--battery',
            'ok',
            '--run-id',
            'dry-unit',
            '--lab-dir',
            str(lab),
            '--verdicts',
            'p,p,p,p,p,p,r,r',
        ]
    )
    assert code == 0
    report = (lab / 'dry-unit' / 'REPORT.md').read_text(encoding='utf-8')
    assert 'stiction_floor' in report
    assert 'not applied' in report.lower()
    events = (lab / 'dry-unit' / 'events.jsonl').read_text(encoding='utf-8')
    assert 'stiction' in events
