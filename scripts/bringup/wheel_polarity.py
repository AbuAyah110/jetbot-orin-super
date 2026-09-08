#!/usr/bin/env python3
"""Wheel polarity check: one bounded forward pulse, then Robot.stop().

Both wheels get the same signed command, so a motor wired backwards shows up as
one wheel fighting the other instead of the chassis rolling straight. Any
exception, signal, or the duration cap stops PWM before the process exits.

    # both wheels forward for 5 s (chassis will roll — clear the floor)
    .venv/bin/python3 scripts/bringup/wheel_polarity.py --seconds 5 --wheel both

    # one side at a time to name the inverted motor
    .venv/bin/python3 scripts/bringup/wheel_polarity.py --wheel left --seconds 2

The live talk-and-drive loop keeps its own 0.35 s nudge; this diagnostic's
longer window is deliberately opt-in via --seconds.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# traitlets / qwiic / Adafruit_MotorHAT are user installs outside the voice venv.
for _extra_site in (
    Path.home() / '.local' / 'lib' / 'python3.10' / 'site-packages',
    Path('/usr/lib/python3/dist-packages'),
):
    if _extra_site.is_dir() and str(_extra_site) not in sys.path:
        sys.path.append(str(_extra_site))

# 0.22 was chosen before the stiction threshold was measured: at that duty the
# motors only hum. The cap matches drive_calibration.SPEED_HARD_MAX so the
# diagnostic can reach the duty that actually rolls, and no further.
SPEED_MAX = 0.7
SECONDS_MAX = 5.0
# A stalled motor overheats and drains the pack, so ramp pulses stay short.
RAMP_SECONDS_MAX = 2.0


def wheel_pair(wheel: str, speed: float) -> tuple[float, float]:
    """Signed (left, right) for the requested side(s)."""
    if wheel == 'left':
        return speed, 0.0
    if wheel == 'right':
        return 0.0, speed
    return speed, speed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=0.35)
    parser.add_argument('--speed', type=float, default=0.15)
    parser.add_argument('--wheel', choices=('both', 'left', 'right'), default='both')
    parser.add_argument('--reverse', action='store_true', help='Negative command instead of forward.')
    parser.add_argument('--dry-run', action='store_true', help='Print the PWM pair; open no I2C.')
    parser.add_argument(
        '--left-alpha',
        type=float,
        default=None,
        help='Override jetbot.Robot left_motor_alpha to trial an inverted left motor.',
    )
    parser.add_argument(
        '--right-alpha',
        type=float,
        default=None,
        help='Override jetbot.Robot right_motor_alpha to trial an inverted right motor.',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    speed = min(abs(float(args.speed)), SPEED_MAX)
    if args.reverse:
        speed = -speed
    seconds = max(0.0, min(float(args.seconds), SECONDS_MAX))
    left, right = wheel_pair(args.wheel, speed)

    print(
        'polarity_test wheel={0} speed={1:+.3f} seconds={2:.2f} left={3:+.3f} right={4:+.3f}'.format(
            args.wheel, speed, seconds, left, right
        ),
        flush=True,
    )
    if args.dry_run:
        print('dry-run: no I2C, no PWM', flush=True)
        return 0

    from jetbot import Robot

    overrides = {}
    if args.left_alpha is not None:
        overrides['left_motor_alpha'] = float(args.left_alpha)
    if args.right_alpha is not None:
        overrides['right_motor_alpha'] = float(args.right_alpha)
    robot = Robot(**overrides)
    robot.stop()
    print(
        'driver backend={0} bus={1} addr=0x{2:02x} left_alpha={3} right_alpha={4}'.format(
            robot._backend,
            robot.i2c_bus,
            robot.i2c_address,
            robot.left_motor.alpha,
            robot.right_motor.alpha,
        ),
        flush=True,
    )

    stopped = False

    def hard_stop(_signum=None, _frame=None) -> None:
        nonlocal stopped
        stopped = True
        try:
            robot.stop()
        except Exception as exc:
            print('stop_failed', exc, file=sys.stderr, flush=True)
        print('{0} pwm_off'.format(time.strftime('%H:%M:%S')), flush=True)

    signal.signal(signal.SIGINT, hard_stop)
    signal.signal(signal.SIGTERM, hard_stop)

    try:
        print('{0} pulse_start — wheels moving'.format(time.strftime('%H:%M:%S')), flush=True)
        robot.set_motors(left, right)
        deadline = time.monotonic() + seconds
        while not stopped and time.monotonic() < deadline:
            time.sleep(0.02)
    except Exception as exc:
        print('pulse_exception', exc, file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            robot.stop()
        except Exception as exc:
            print('stop_failed', exc, file=sys.stderr, flush=True)
        print('{0} pulse_stop'.format(time.strftime('%H:%M:%S')), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
