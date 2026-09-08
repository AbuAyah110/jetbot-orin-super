#!/usr/bin/env python3
"""Chassis motion wizard: bounded pulses, your eyes/ears as the sensor.

Cosmos is off. Each trial is a short PWM pulse (duty <= 0.7, duration <= 1.5 s)
then Robot.stop(). You tag what happened by key or voice. The run writes
data/lab/<run_id>/events.jsonl and REPORT.md. Yaml write-back is not this
script's job (see drive_calibration); this one only logs what we measured.

    .venv/bin/python3 scripts/lab/chassis_wizard.py --wheels-up --dry-run
    .venv/bin/python3 scripts/lab/chassis_wizard.py --on-floor --keys-only

Refuses to start unless --on-floor or --wheels-up is set. Refuses if
talk_and_drive.py is already holding I2C. Do not run this with the agent loop.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import re
import select
import signal
import subprocess
import sys
import time
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

for _extra_site in (
    Path.home() / '.local' / 'lib' / 'python3.10' / 'site-packages',
    Path('/usr/lib/python3/dist-packages'),
):
    if _extra_site.is_dir() and str(_extra_site) not in sys.path:
        sys.path.append(str(_extra_site))

from jetbot_agent.robot_loop.approach_plan import (  # noqa: E402
    ARC_INNER_DELTA,
    ARC_INNER_FLOOR,
    ARC_OUTER_DELTA,
)
from jetbot_agent.robot_loop.drive_calibration import (  # noqa: E402
    DEFAULT_DURATION_S,
    DEFAULT_SPEED,
    DURATION_HARD_MAX,
    SPEED_HARD_MAX,
    load_calibration,
)

# Plan safety: pulses stay under the 2 s / 0.7 hard caps, and under 1.5 s in
# practice so a missed stop still dies quickly.
PULSE_DURATION_MAX = 1.5
DUTY_MAX = SPEED_HARD_MAX
TWITCH_DURATION_S = 0.40
TWITCH_DUTY = 0.40
STICTION_DUTIES = (0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70)
STICTION_DURATION_S = 1.2
POLARITY_DURATION_S = 0.80
ARC_DURATION_S = 1.2
ARC_FLOOR_BUMP = 0.05
# An arc is only an arc while the wheels differ. Raising the inner wheel to the
# stiction floor can push it past the base-derived outer wheel, which would
# command an equal pair -- the straight/pivot pulse the arc trial exists to
# distinguish. Keep at least this much difference by lifting the outer wheel.
ARC_MIN_DELTA = 0.10
DURATION_SHORT_S = 0.6
DURATION_LONG_S = 1.2
CSI_MAD_STILL = 8.0
ASR_LISTEN_S = 2
VERDICT_WAIT_S = 60.0
SPEAK_MAX_CHARS = 120

# Key alphabet from the lab plan, plus duration/arc tags the protocol needs.
KEY_VERDICTS = {
    'p': 'pass',
    'h': 'hummed',
    'v': 'pivoted',
    'f': 'fought',
    's': 'skip',
    'q': 'abort',
    'n': 'straight',
    't': 'too_short',
    'r': 'about_right',
    'd': 'too_far',
}

VERDICT_HELP = (
    'p pass / rolled / curved   h hummed   v pivoted   f fought   '
    'n went straight   t too short   r about right   d too far   '
    's skip   q abort'
)

LAB_DIR = REPO / 'data' / 'lab'


@dataclass
class Pulse:
    left: float
    right: float
    duration_s: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.left, self.right, self.duration_s


@dataclass
class SuggestedCalibration:
    speed: Optional[float] = None
    duration_s: Optional[float] = None
    stiction_floor: Optional[float] = None
    arc_inner_floor: Optional[float] = None
    measured_on: str = ''
    last_lab_run: str = ''
    battery: str = ''
    notes: list[str] = field(default_factory=list)


class AbortWizard(Exception):
    """Operator or signal asked the wizard to stop motors and exit."""


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='milliseconds')


def clamp_duty(value: float) -> float:
    return max(-DUTY_MAX, min(DUTY_MAX, float(value)))


def clamp_duration(value: float) -> float:
    seconds = max(0.0, float(value))
    seconds = min(seconds, PULSE_DURATION_MAX)
    return min(seconds, DURATION_HARD_MAX)


def clamp_pulse(left: float, right: float, duration_s: float) -> Pulse:
    """Force every commanded pulse inside the non-negotiable caps."""
    return Pulse(
        left=clamp_duty(left),
        right=clamp_duty(right),
        duration_s=clamp_duration(duration_s),
    )


def arc_wheels(
    step: str,
    *,
    base: float,
    inner_floor: float,
    inner_delta: float = ARC_INNER_DELTA,
    outer_delta: float = ARC_OUTER_DELTA,
) -> tuple[float, float]:
    """Differential pair matching approach_plan.step_wheels, with a live floor."""
    duty = clamp_duty(abs(base))
    floor = clamp_duty(abs(inner_floor))
    inner = max(floor, duty - inner_delta)
    outer = min(DUTY_MAX, max(duty + outer_delta, inner + ARC_MIN_DELTA))
    if outer - inner < ARC_MIN_DELTA:
        # Only reachable when the floor itself is near DUTY_MAX. Curvature is
        # sacrificed rather than the floor: a stalled inner wheel pivots.
        inner = min(inner, outer)
    if step == 'forward':
        return duty, duty
    if step == 'arc_left':
        return inner, outer
    if step == 'arc_right':
        return outer, inner
    return 0.0, 0.0


def parse_verdict(raw: str) -> Optional[str]:
    """Map a keypress or ASR phrase onto the lab verdict alphabet."""
    text = ' '.join((raw or '').strip().lower().split())
    if not text:
        return None
    if len(text) == 1 and text in KEY_VERDICTS:
        return KEY_VERDICTS[text]
    if re.search(r'\b(abort|quit|stop wizard|emergency)\b', text):
        return 'abort'
    if re.search(r'\btoo\s+far\b|\bfarther\b|\btoo\s+long\b', text):
        return 'too_far'
    if re.search(r'\btoo\s+short\b|\bnot\s+enough\b|\bshort\b', text):
        return 'too_short'
    if re.search(r'\babout\s+right\b|\bjust\s+right\b|\bgood\s+distance\b', text):
        return 'about_right'
    if re.search(r'\bwent\s+straight\b|\bno\s+curve\b|\bstraight\b', text):
        return 'straight'
    if re.search(r'\bpivot|\bspin|\bspun|\bin\s+place\b', text):
        return 'pivoted'
    if re.search(r'\bfought\b|\bfighting\b|\bfight\b|\btwist(?:ed|ing)?\b', text):
        return 'fought'
    if re.search(r'\bhumm|\bhum\b|\bdidn.?t\s+move\b|\bno\s+travel\b|\bstill\b', text):
        return 'hummed'
    if re.search(r'\bskip(?:ped)?\b', text):
        return 'skip'
    if re.search(
        r'\bpass(?:ed)?\b|\brolled\b|\btravel(?:ed|led)?\b|\bcurved\b|'
        r'\badvanced\b|\byes\b|\bgood\b',
        text,
    ):
        return 'pass'
    return None


def frame_hash(jpeg: bytes) -> str:
    return hashlib.sha256(jpeg).hexdigest()[:16]


def decode_gray(jpeg: bytes) -> Optional[tuple[tuple[int, int], bytes]]:
    try:
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(jpeg)).convert('L')
        return image.size, image.tobytes()
    except Exception:
        pass
    try:
        import cv2
        import numpy as np

        array_in = np.frombuffer(jpeg, dtype=np.uint8)
        image = cv2.imdecode(array_in, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return (int(image.shape[1]), int(image.shape[0])), image.tobytes()
    except Exception:
        return None


def mean_abs_diff_bytes(before: bytes, after: bytes) -> Optional[float]:
    if not before or not after or len(before) != len(after):
        return None
    total = 0
    for left, right in zip(before, after):
        total += abs(left - right)
    return total / float(len(before))


def jpeg_mean_abs_diff(before: bytes, after: bytes) -> Optional[float]:
    if not before or not after:
        return None
    if before == after:
        return 0.0
    decoded_a = decode_gray(before)
    decoded_b = decode_gray(after)
    if decoded_a is None or decoded_b is None:
        return None
    if decoded_a[0] != decoded_b[0]:
        return None
    return mean_abs_diff_bytes(decoded_a[1], decoded_b[1])


def likely_no_translate(mad: Optional[float], threshold: float = CSI_MAD_STILL) -> bool:
    return mad is not None and mad < threshold


def talk_and_drive_holding_i2c() -> str:
    try:
        proc = subprocess.run(
            ['pgrep', '-af', 'talk_and_drive.py'],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ''
    for line in (proc.stdout or '').splitlines():
        blob = line.strip()
        if not blob:
            continue
        if 'pgrep' in blob:
            continue
        if 'talk_and_drive.py' in blob:
            return blob
    return ''


class RecordingRobot:
    """Stand-in used by --dry-run and unit tests. Never opens I2C."""

    def __init__(self) -> None:
        self.pulses: list[Pulse] = []
        self.stopped = 0
        self._active: Optional[Pulse] = None

    def set_motors(self, left: float, right: float) -> None:
        self._active = Pulse(left=float(left), right=float(right), duration_s=0.0)

    def stop(self) -> None:
        self.stopped += 1
        if self._active is not None:
            self.pulses.append(self._active)
            self._active = None


class LabLog:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.path = run_dir / 'events.jsonl'
        self.events: list[dict] = []
        run_dir.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict) -> dict:
        payload = dict(event)
        payload.setdefault('ts', iso_now())
        payload.setdefault('run_id', self.run_id)
        self.events.append(payload)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(payload, sort_keys=True) + '\n')
        return payload


def derive_calibration(
    events: Iterable[dict],
    *,
    run_id: str,
    battery: str = '',
    measured_on: str = '',
) -> SuggestedCalibration:
    """Turn tagged trials into the numbers yaml should eventually store."""
    suggested = SuggestedCalibration(
        last_lab_run=run_id,
        battery=battery,
        measured_on=measured_on or datetime.now().date().isoformat(),
    )
    stiction_pass: Optional[float] = None
    for event in events:
        if event.get('phase') != 'stiction':
            continue
        if event.get('verdict') == 'pass':
            stiction_pass = abs(float(event['left']))
            break
        if event.get('verdict') == 'hummed':
            suggested.notes.append(
                'stiction hummed at duty {0:.2f}'.format(float(event['left']))
            )
    suggested.stiction_floor = stiction_pass
    suggested.speed = stiction_pass

    last_advancing_inner: Optional[float] = None
    for event in events:
        if event.get('phase') != 'arc':
            continue
        inner = min(abs(float(event['left'])), abs(float(event['right'])))
        verdict = event.get('verdict')
        if verdict in ('pass',):
            last_advancing_inner = inner
        elif verdict == 'pivoted':
            suggested.notes.append(
                'arc pivoted at inner {0:.2f}; wizard raised the floor'.format(inner)
            )
        elif verdict == 'straight':
            suggested.notes.append(
                'arc went straight at inner {0:.2f}'.format(inner)
            )
    suggested.arc_inner_floor = last_advancing_inner

    short_tag = ''
    long_tag = ''
    for event in events:
        if event.get('phase') != 'duration':
            continue
        duration = float(event.get('duration_s') or 0.0)
        tag = event.get('verdict') or ''
        if abs(duration - DURATION_SHORT_S) < 0.05:
            short_tag = tag
        elif abs(duration - DURATION_LONG_S) < 0.05:
            long_tag = tag
    if long_tag == 'about_right' or long_tag == 'pass':
        suggested.duration_s = DURATION_LONG_S
    elif short_tag in ('about_right', 'pass') and long_tag == 'too_far':
        suggested.duration_s = DURATION_SHORT_S
    elif short_tag == 'too_short' and long_tag in ('about_right', 'pass', ''):
        suggested.duration_s = DURATION_LONG_S
    elif long_tag == 'too_short':
        suggested.notes.append('1.2 s still felt short; do not raise past 2.0 s here')
        suggested.duration_s = min(DURATION_LONG_S, DURATION_HARD_MAX)
    elif short_tag or long_tag:
        suggested.duration_s = DEFAULT_DURATION_S

    polarity_fight = any(
        event.get('phase') == 'polarity' and event.get('verdict') == 'fought'
        for event in events
    )
    if polarity_fight:
        suggested.notes.append('polarity fought: do not write yaml until wiring is fixed')

    return suggested


def render_report(
    *,
    run_id: str,
    mode: str,
    battery: str,
    events: list[dict],
    suggested: SuggestedCalibration,
    aborted: bool,
) -> str:
    lines = [
        '# Chassis wizard {0}'.format(run_id),
        '',
        '- mode: `{0}`'.format(mode),
        '- battery: `{0}`'.format(battery or 'unspecified'),
        '- measured_on: `{0}`'.format(suggested.measured_on),
        '- aborted: `{0}`'.format(str(aborted).lower()),
        '- events: {0}'.format(len(events)),
        '',
        '## Verdict log',
        '',
    ]
    for event in events:
        if event.get('kind') != 'trial':
            continue
        lines.append(
            '- `{id}` {phase} left={left:.2f} right={right:.2f} {dur:.2f}s → **{verdict}**{csi}'.format(
                id=event.get('trial_id', ''),
                phase=event.get('phase', ''),
                left=float(event.get('left') or 0.0),
                right=float(event.get('right') or 0.0),
                dur=float(event.get('duration_s') or 0.0),
                verdict=event.get('verdict', ''),
                csi=(
                    ' (csi mad={0:.2f}{1})'.format(
                        float(event['csi_mad']),
                        ' no-translate' if event.get('csi_no_translate') else '',
                    )
                    if event.get('csi_mad') is not None
                    else ''
                ),
            )
        )
    lines.extend(
        [
            '',
            '## Suggested `drive_calibration` (not applied)',
            '',
            'Write-back is explicit and lives in a later step. Do not paste this',
            'into yaml after a fight or a hummed-only run.',
            '',
            '```yaml',
            'drive_calibration:',
            '  speed: {0}'.format(
                '{0:.2f}'.format(suggested.speed) if suggested.speed is not None else 'null'
            ),
            '  duration_s: {0}'.format(
                '{0:.1f}'.format(suggested.duration_s)
                if suggested.duration_s is not None
                else 'null'
            ),
            '  stiction_floor: {0}'.format(
                '{0:.2f}'.format(suggested.stiction_floor)
                if suggested.stiction_floor is not None
                else 'null'
            ),
            '  arc_inner_floor: {0}'.format(
                '{0:.2f}'.format(suggested.arc_inner_floor)
                if suggested.arc_inner_floor is not None
                else 'null'
            ),
            "  measured_on: '{0}'".format(suggested.measured_on),
            '  last_lab_run: {0}'.format(run_id),
            '```',
            '',
        ]
    )
    if suggested.notes:
        lines.append('## Notes')
        lines.append('')
        for note in suggested.notes:
            lines.append('- {0}'.format(note))
        lines.append('')
    return '\n'.join(lines) + '\n'


def write_wav(path: Path, samples, sample_rate: int) -> None:
    pcm = array.array('h')
    for sample in samples:
        value = int(max(-1.0, min(1.0, float(sample))) * 32767.0)
        pcm.append(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def read_wav_mono(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), 'rb') as handle:
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
        channels = handle.getnchannels()
    if width != 2:
        raise RuntimeError('expected S16_LE capture')
    samples = array.array('h')
    samples.frombytes(raw)
    floats = [sample / 32768.0 for sample in samples]
    if channels > 1:
        floats = floats[0::channels]
    return floats, rate


class Wizard:
    def __init__(
        self,
        *,
        robot,
        log: LabLog,
        dry_run: bool,
        announce: Callable[[str], None],
        ask_verdict: Callable[[str], str],
        capture_jpeg: Optional[Callable[[], Optional[bytes]]] = None,
        sleep: Callable[[float], None] = time.sleep,
        use_csi: bool = False,
    ) -> None:
        self.robot = robot
        self.log = log
        self.dry_run = dry_run
        self.announce = announce
        self.ask_verdict = ask_verdict
        self.capture_jpeg = capture_jpeg
        self.sleep = sleep
        self.use_csi = use_csi
        self.aborted = False
        self._trial_n = 0

    def stop_motors(self) -> None:
        try:
            self.robot.stop()
        except Exception as exc:
            print('stop_failed', exc, file=sys.stderr, flush=True)

    def pulse(self, left: float, right: float, duration_s: float) -> Pulse:
        commanded = clamp_pulse(left, right, duration_s)
        if self.aborted:
            self.stop_motors()
            raise AbortWizard('aborted')
        if commanded.duration_s <= 0.0:
            self.stop_motors()
            return commanded
        try:
            self.robot.set_motors(commanded.left, commanded.right)
            active = getattr(self.robot, '_active', None)
            if active is not None:
                active.duration_s = commanded.duration_s
            if not self.dry_run:
                deadline = time.monotonic() + commanded.duration_s
                while time.monotonic() < deadline:
                    if self.aborted:
                        break
                    remaining = deadline - time.monotonic()
                    self.sleep(min(0.02, max(0.0, remaining)))
        finally:
            self.stop_motors()
        return commanded

    def witness(self) -> Optional[bytes]:
        if not self.use_csi or self.capture_jpeg is None:
            return None
        try:
            return self.capture_jpeg()
        except Exception as exc:
            print('csi_capture_failed', exc, file=sys.stderr, flush=True)
            return None

    def _save_frame(self, trial_id: str, suffix: str, jpeg: Optional[bytes]) -> None:
        if not jpeg:
            return
        frames = self.log.run_dir / 'frames'
        frames.mkdir(parents=True, exist_ok=True)
        (frames / '{0}_{1}.jpg'.format(trial_id, suffix)).write_bytes(jpeg)

    def trial(
        self,
        *,
        phase: str,
        announce: str,
        left: float,
        right: float,
        duration_s: float,
        trial_id: str = '',
        extra: Optional[dict] = None,
    ) -> dict:
        if self.aborted:
            raise AbortWizard('aborted')
        self._trial_n += 1
        trial_id = trial_id or '{0}_{1:02d}'.format(phase, self._trial_n)
        self.announce(announce)
        print(
            '{0} trial {1} {2} left={3:.2f} right={4:.2f} {5:.2f}s'.format(
                time.strftime('%H:%M:%S'), trial_id, phase, left, right, duration_s
            ),
            flush=True,
        )
        before = self.witness()
        commanded = self.pulse(left, right, duration_s)
        after = self.witness()
        self._save_frame(trial_id, 'before', before)
        self._save_frame(trial_id, 'after', after)
        if self.aborted:
            event = {
                'kind': 'trial',
                'trial_id': trial_id,
                'phase': phase,
                'left': commanded.left,
                'right': commanded.right,
                'duration_s': commanded.duration_s,
                'verdict': 'abort',
                'dry_run': self.dry_run,
            }
            if extra:
                event.update(extra)
            self.log.write(event)
            self.stop_motors()
            raise AbortWizard('aborted')
        mad = None
        if before and after:
            mad = jpeg_mean_abs_diff(before, after)
        event = {
            'kind': 'trial',
            'trial_id': trial_id,
            'phase': phase,
            'left': commanded.left,
            'right': commanded.right,
            'duration_s': commanded.duration_s,
            'dry_run': self.dry_run,
        }
        if extra:
            event.update(extra)
        if before:
            event['frame_before'] = frame_hash(before)
        if after:
            event['frame_after'] = frame_hash(after)
        if mad is not None:
            event['csi_mad'] = round(mad, 3)
            event['csi_no_translate'] = likely_no_translate(mad)
        try:
            verdict = self.ask_verdict(phase)
        except AbortWizard:
            self.aborted = True
            event['verdict'] = 'abort'
            self.log.write(event)
            self.stop_motors()
            raise
        if verdict == 'abort':
            self.aborted = True
            event['verdict'] = 'abort'
            self.log.write(event)
            self.stop_motors()
            raise AbortWizard('operator abort')
        event['verdict'] = verdict
        self.log.write(event)
        print('verdict {0} -> {1}'.format(trial_id, verdict), flush=True)
        return event

    def run_twitch(self) -> None:
        self.announce('Wheels up twitch. Left wheel, then right. Pass if both spin the same way.')
        self.trial(
            phase='twitch',
            trial_id='twitch_left',
            announce='Left wheel twitch.',
            left=TWITCH_DUTY,
            right=0.0,
            duration_s=TWITCH_DURATION_S,
        )
        self.trial(
            phase='twitch',
            trial_id='twitch_right',
            announce='Right wheel twitch.',
            left=0.0,
            right=TWITCH_DUTY,
            duration_s=TWITCH_DURATION_S,
        )

    def run_stiction(self) -> Optional[float]:
        self.announce(
            'Stiction search. Both wheels forward. Tag hummed or pass when it travels.'
        )
        floor = None
        for duty in STICTION_DUTIES:
            event = self.trial(
                phase='stiction',
                trial_id='stiction_{0:.2f}'.format(duty).replace('.', 'p'),
                announce='Both wheels at {0:.2f}.'.format(duty),
                left=duty,
                right=duty,
                duration_s=STICTION_DURATION_S,
                extra={'duty': duty},
            )
            if event['verdict'] == 'pass':
                floor = duty
                break
            if event['verdict'] == 'skip':
                continue
        return floor

    def run_polarity(self, duty: float) -> None:
        self.announce('Polarity. Tag fought if one wheel pulls backward, pass if it rolls.')
        speed = duty
        self.trial(
            phase='polarity',
            trial_id='polarity_both',
            announce='Both wheels forward.',
            left=speed,
            right=speed,
            duration_s=POLARITY_DURATION_S,
            extra={'wheel': 'both'},
        )
        self.trial(
            phase='polarity',
            trial_id='polarity_left',
            announce='Left wheel only.',
            left=speed,
            right=0.0,
            duration_s=POLARITY_DURATION_S,
            extra={'wheel': 'left'},
        )
        self.trial(
            phase='polarity',
            trial_id='polarity_right',
            announce='Right wheel only.',
            left=0.0,
            right=speed,
            duration_s=POLARITY_DURATION_S,
            extra={'wheel': 'right'},
        )

    def run_arcs(self, base: float, inner_floor: float) -> float:
        self.announce(
            'Arc versus pivot. Tag pivoted, pass if it curved while advancing, or straight.'
        )
        floor = inner_floor
        for step in ('arc_left', 'arc_right'):
            for attempt in (1, 2):
                left, right = arc_wheels(step, base=base, inner_floor=floor)
                event = self.trial(
                    phase='arc',
                    trial_id='{0}_inner{1:.2f}_try{2}'.format(step, floor, attempt).replace(
                        '.', 'p'
                    ),
                    announce='{0}, inner floor {1:.2f}.'.format(step.replace('_', ' '), floor),
                    left=left,
                    right=right,
                    duration_s=ARC_DURATION_S,
                    extra={
                        'step': step,
                        'inner_floor': floor,
                        'attempt': attempt,
                    },
                )
                if event['verdict'] == 'pivoted' and attempt == 1:
                    floor = min(DUTY_MAX, round(floor + ARC_FLOOR_BUMP, 2))
                    self.announce(
                        'Pivoted. Raising inner floor to {0:.2f} and retrying once.'.format(
                            floor
                        )
                    )
                    continue
                break
        return floor

    def run_duration(self, duty: float) -> None:
        self.announce(
            'Duration versus distance. Same duty, short then long. Tag too short, about right, or too far.'
        )
        self.trial(
            phase='duration',
            trial_id='duration_0p6',
            announce='Forward for point six seconds.',
            left=duty,
            right=duty,
            duration_s=DURATION_SHORT_S,
        )
        self.trial(
            phase='duration',
            trial_id='duration_1p2',
            announce='Forward for one point two seconds.',
            left=duty,
            right=duty,
            duration_s=DURATION_LONG_S,
        )

    def run_protocol(self, *, wheels_up: bool, on_floor: bool, battery: str) -> SuggestedCalibration:
        self.log.write(
            {
                'kind': 'session_start',
                'wheels_up': wheels_up,
                'on_floor': on_floor,
                'battery': battery,
                'calibration': asdict(load_calibration()),
            }
        )
        if wheels_up:
            self.run_twitch()
        elif on_floor:
            print('skipping wheels-up twitch (--on-floor)', flush=True)

        travel_duty = None
        if on_floor:
            travel_duty = self.run_stiction()
            duty = travel_duty if travel_duty is not None else DEFAULT_SPEED
            self.run_polarity(duty)
            inner = ARC_INNER_FLOOR
            self.run_arcs(duty, inner)
            self.run_duration(duty)
        else:
            print('skipping floor protocol (need --on-floor for stiction/arcs)', flush=True)

        suggested = derive_calibration(
            self.log.events,
            run_id=self.log.run_id,
            battery=battery,
        )
        self.log.write({'kind': 'session_end', 'suggested': asdict(suggested)})
        return suggested


def apply_alsa_safety() -> dict:
    from jetbot_agent.hardware.audio_interface import (
        apply_safe_mixer_baseline,
        mixer_report,
        resolve_sss1629,
    )

    ident = resolve_sss1629()
    apply_safe_mixer_baseline(ident['card_index_ephemeral'])
    report = mixer_report(ident['card_index_ephemeral'])
    lowered = report.lower()
    if 'mic' in lowered and 'playback' in lowered and '[on]' in lowered:
        apply_safe_mixer_baseline(ident['card_index_ephemeral'])
    ident = dict(ident)
    ident['sidetone_enabled'] = False
    return ident


def play_wav_once(path: Path, playback_dev: str) -> None:
    subprocess.run(['pkill', '-x', 'aplay'], check=False, capture_output=True)
    try:
        subprocess.run(
            ['aplay', '-D', playback_dev, '-q', str(path)],
            check=False,
            timeout=8,
        )
    finally:
        subprocess.run(['pkill', '-x', 'aplay'], check=False, capture_output=True)


def capture_mic_wav(seconds: int, capture_dev: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pkill', '-x', 'arecord'], check=False, capture_output=True)
    proc = subprocess.run(
        [
            'arecord',
            '-D',
            capture_dev,
            '-f',
            'S16_LE',
            '-r',
            '16000',
            '-c',
            '1',
            '-d',
            str(int(seconds)),
            str(dest),
        ],
        check=False,
        timeout=int(seconds) + 8,
    )
    if proc.returncode != 0:
        raise RuntimeError('arecord_failed rc={0}'.format(proc.returncode))
    return dest


def read_key(timeout_s: float) -> str:
    """One raw character, or empty on timeout. Restores terminal attributes."""
    if not sys.stdin.isatty():
        return ''
    fd = sys.stdin.fileno()
    try:
        import termios
        import tty
    except ImportError:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not ready:
            return ''
        return sys.stdin.read(1)
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
        if not ready:
            return ''
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--on-floor',
        action='store_true',
        help='Chassis is on the floor. Runs stiction, polarity, arc, duration.',
    )
    parser.add_argument(
        '--wheels-up',
        action='store_true',
        help='Wheels off the ground. Runs the left/right twitch first.',
    )
    parser.add_argument('--dry-run', action='store_true', help='No I2C. Record commanded pulses.')
    parser.add_argument('--keys-only', action='store_true', help='Disable TTS and ASR; keys only.')
    parser.add_argument('--no-tts', action='store_true', help='Print prompts; do not play speech.')
    parser.add_argument(
        '--voice',
        action='store_true',
        default=None,
        help='Force ASR even if keys are available (default: ASR on unless --keys-only).',
    )
    parser.add_argument('--no-voice', action='store_true', help='Do not capture a spoken verdict.')
    parser.add_argument('--csi', action='store_true', help='Grab CSI 448² JPEG before and after each pulse.')
    parser.add_argument(
        '--battery',
        choices=('high', 'ok', 'low'),
        default='ok',
        help='Pack sag tag stored on the report. Re-run after a charge.',
    )
    parser.add_argument('--run-id', default='')
    parser.add_argument('--lab-dir', default=str(LAB_DIR))
    parser.add_argument(
        '--verdicts',
        default='',
        help='Comma-separated scripted verdicts (keys or words). Used by tests.',
    )
    return parser.parse_args(argv)


def build_announcer(args: argparse.Namespace, wav_dir: Path):
    tts = None
    playback = ''
    if not args.keys_only and not args.no_tts and not args.dry_run:
        try:
            alsa = apply_alsa_safety()
            playback = alsa['alsa_playback']
            from jetbot_agent.audio.piper_tts import PiperTTS

            tts = PiperTTS(num_threads=2)
            print('alsa', json.dumps({'playback': playback, 'sidetone_enabled': False}), flush=True)
        except Exception as exc:
            print('tts_unavailable', exc, flush=True)
            tts = None

    def announce(text: str) -> None:
        phrase = ' '.join((text or '').split())[:SPEAK_MAX_CHARS]
        print('announce:', phrase, flush=True)
        if tts is None or not playback or not phrase:
            return
        generated = tts.synthesize(phrase)
        wav = wav_dir / 'chassis_wizard_tts.wav'
        write_wav(wav, generated.samples, generated.sample_rate)
        play_wav_once(wav, playback)

    return announce


def build_verdict_asker(args: argparse.Namespace, wav_dir: Path):
    scripted = [part.strip() for part in (args.verdicts or '').split(',') if part.strip()]
    asr = None
    capture_dev = ''
    voice = (not args.keys_only) and (not args.dry_run) and (not args.no_voice)
    if args.voice is True:
        voice = not args.dry_run
    if voice:
        try:
            alsa = apply_alsa_safety()
            capture_dev = alsa['alsa_capture']
            from jetbot_agent.audio.zipformer_asr import ZipformerASR

            asr = ZipformerASR(num_threads=2)
        except Exception as exc:
            print('asr_unavailable', exc, flush=True)
            asr = None

    def from_script() -> Optional[str]:
        if not scripted:
            return None
        raw = scripted.pop(0)
        parsed = parse_verdict(raw) or parse_verdict(raw[:1])
        if parsed is None:
            raise RuntimeError('unrecognized scripted verdict: {0!r}'.format(raw))
        return parsed

    def ask(phase: str) -> str:
        scripted_verdict = from_script()
        if scripted_verdict is not None:
            return scripted_verdict
        print('verdict? {0}'.format(VERDICT_HELP), flush=True)
        key = read_key(8.0)
        if key in ('\x03', '\x04'):
            return 'abort'
        parsed = parse_verdict(key)
        if parsed:
            return parsed
        if asr is not None and capture_dev:
            dest = wav_dir / 'chassis_wizard_verdict.wav'
            try:
                capture_mic_wav(ASR_LISTEN_S, capture_dev, dest)
                samples, rate = read_wav_mono(dest)
                transcript = asr.transcribe(samples, rate)
                print('asr', transcript, flush=True)
                parsed = parse_verdict(transcript)
                if parsed:
                    return parsed
            except Exception as exc:
                print('asr_failed', exc, flush=True)
        deadline = time.monotonic() + VERDICT_WAIT_S
        while time.monotonic() < deadline:
            key = read_key(0.25)
            if not key:
                continue
            if key in ('\x03', '\x04'):
                return 'abort'
            parsed = parse_verdict(key)
            if parsed:
                return parsed
            print('unknown key {0!r}. {1}'.format(key, VERDICT_HELP), flush=True)
        raise AbortWizard('verdict timeout')

    return ask


def open_camera(enabled: bool, dry_run: bool):
    if not enabled or dry_run:
        return None, None
    from jetbot_agent.robot_loop.csi_jpeg import CsiJpeg448

    camera = CsiJpeg448()
    camera.open()
    return camera, camera.capture_jpeg


def open_robot(dry_run: bool):
    if dry_run:
        return RecordingRobot()
    from jetbot import Robot

    robot = Robot()
    robot.stop()
    print(
        'driver backend={0} bus={1} addr=0x{2:02x}'.format(
            robot._backend,
            robot.i2c_bus,
            robot.i2c_address,
        ),
        flush=True,
    )
    return robot


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if not args.on_floor and not args.wheels_up:
        print(
            'Refusing to start without --on-floor or --wheels-up '
            '(say which world the wheels are in).',
            file=sys.stderr,
        )
        return 2
    if not args.dry_run:
        busy = talk_and_drive_holding_i2c()
        if busy:
            print(
                'Refusing to start: talk_and_drive is holding I2C: {0}'.format(busy),
                file=sys.stderr,
            )
            return 2

    run_id = args.run_id.strip() or datetime.now().strftime('chassis-%Y%m%d-%H%M%S')
    run_dir = Path(args.lab_dir) / run_id
    log = LabLog(run_dir, run_id)
    wav_dir = run_dir / 'audio'
    robot = None
    camera = None
    wizard = None

    def hard_stop(_signum=None, _frame=None) -> None:
        if wizard is not None:
            wizard.aborted = True
            wizard.stop_motors()
        elif robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        print('{0} pwm_off'.format(time.strftime('%H:%M:%S')), flush=True)

    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, hard_stop)
    signal.signal(signal.SIGTERM, hard_stop)

    try:
        robot = open_robot(args.dry_run)
        camera, capture_jpeg = open_camera(args.csi, args.dry_run)
        announce = build_announcer(args, wav_dir)
        ask = build_verdict_asker(args, wav_dir)
        mode = 'wheels-up+floor' if args.wheels_up and args.on_floor else (
            'wheels-up' if args.wheels_up else 'on-floor'
        )
        print(
            'chassis_wizard run_id={0} mode={1} battery={2} dry_run={3}'.format(
                run_id, mode, args.battery, args.dry_run
            ),
            flush=True,
        )
        print(VERDICT_HELP, flush=True)
        wizard = Wizard(
            robot=robot,
            log=log,
            dry_run=args.dry_run,
            announce=announce,
            ask_verdict=ask,
            capture_jpeg=capture_jpeg,
            sleep=(lambda _seconds: None) if args.dry_run else time.sleep,
            use_csi=bool(args.csi) and not args.dry_run,
        )
        aborted = False
        try:
            suggested = wizard.run_protocol(
                wheels_up=bool(args.wheels_up),
                on_floor=bool(args.on_floor),
                battery=args.battery,
            )
        except AbortWizard:
            aborted = True
            suggested = derive_calibration(
                log.events, run_id=run_id, battery=args.battery
            )
            log.write({'kind': 'session_end', 'aborted': True, 'suggested': asdict(suggested)})
        report = render_report(
            run_id=run_id,
            mode=mode,
            battery=args.battery,
            events=log.events,
            suggested=suggested,
            aborted=aborted,
        )
        report_path = run_dir / 'REPORT.md'
        report_path.write_text(report, encoding='utf-8')
        print('wrote', log.path, flush=True)
        print('wrote', report_path, flush=True)
        print(report, flush=True)
        return 1 if aborted else 0
    except AbortWizard:
        return 1
    except Exception as exc:
        print('wizard_exception', exc, file=sys.stderr, flush=True)
        return 1
    finally:
        if wizard is not None:
            wizard.stop_motors()
        elif robot is not None:
            try:
                robot.stop()
            except Exception as exc:
                print('stop_failed', exc, file=sys.stderr, flush=True)
        if camera is not None:
            try:
                camera.close()
            except Exception:
                pass
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)


if __name__ == '__main__':
    raise SystemExit(main())
