#!/usr/bin/env python3
"""Measure how far one turn pulse moves a coloured target across the frame.

Turn degrees per pulse are not calibrated on this chassis, and the detour needs
a pulse small enough to keep its target in view. Rather than guess, drive a
series of identical short pulses and record where pixel colour grounding sees
the target after each one. Image-space shift is the quantity the detour actually
depends on, and it comes from the one sensor that works here.

Put a strongly coloured object (red, blue or green) in front of the robot, then:

    measure_turn_step.py --target 'red object' --duration 0.15 --pulses 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'bringup'))

# Same rule as talk_and_drive.py: GStreamer's gi and the motor HAT live in
# system/user site dirs. Append so the venv's NumPy 2 keeps priority.
for _extra_site in (
    Path.home() / '.local' / 'lib' / 'python3.10' / 'site-packages',
    Path('/usr/lib/python3/dist-packages'),
):
    if _extra_site.is_dir() and str(_extra_site) not in sys.path:
        sys.path.append(str(_extra_site))

from jetbot_agent.robot_loop.color_grounding import locate_color  # noqa: E402
from jetbot_agent.robot_loop.csi_jpeg import CsiJpeg448  # noqa: E402
from jetbot_agent.robot_loop.intents import NUDGE_VX  # noqa: E402

WZ_WHEEL_SCALE = 0.4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='red object')
    parser.add_argument('--duration', type=float, default=0.15)
    parser.add_argument('--pulses', type=int, default=6)
    parser.add_argument('--direction', choices=['left', 'right'], default='right')
    parser.add_argument('--no-pwm', action='store_true')
    parser.add_argument(
        '--save-frames',
        metavar='DIR',
        help='Write every frame here. Pixel counts alone hide what the camera '
        'actually saw, which is how a door frame passed as the target.',
    )
    args = parser.parse_args()

    frame_dir = Path(args.save_frames) if args.save_frames else None
    if frame_dir is not None:
        frame_dir.mkdir(parents=True, exist_ok=True)

    duty = NUDGE_VX
    # Same signed wheel pair the detour uses, so the measurement transfers.
    left = duty if args.direction == 'right' else -duty
    right = -duty if args.direction == 'right' else duty

    robot = None
    if not args.no_pwm:
        from jetbot.robot import Robot

        robot = Robot()
        robot.stop()

    camera = CsiJpeg448()
    camera.open()
    rows = []
    try:
        for pulse in range(args.pulses + 1):
            camera.settle()
            jpeg = camera.capture_jpeg()
            found = locate_color(jpeg, args.target)
            if frame_dir is not None:
                (frame_dir / 'pulse_{0:02d}.jpg'.format(pulse)).write_bytes(jpeg)
            row = {
                'pulse': pulse,
                'visible': found.visible,
                'side': found.side,
                'center_x': round(found.center_x, 1),
                'pixels': found.pixels,
                'fraction': round(found.fraction, 4),
                'rejected': found.rejected,
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if pulse == args.pulses:
                break
            if robot is not None:
                robot.set_motors(left, right)
                time.sleep(args.duration)
                robot.stop()
            time.sleep(0.2)
    finally:
        if robot is not None:
            robot.stop()
        camera.close()

    print('--- summary ---')
    seen = [row for row in rows if row['visible']]
    print('pulse duration {0:.2f}s direction {1}'.format(args.duration, args.direction))
    print('frames with target visible: {0}/{1}'.format(len(seen), len(rows)))
    for before, after in zip(rows, rows[1:]):
        if before['visible'] and after['visible']:
            print(
                'pulse {0}->{1}: center_x {2} -> {3} (shift {4:+.1f} px)'.format(
                    before['pulse'],
                    after['pulse'],
                    before['center_x'],
                    after['center_x'],
                    after['center_x'] - before['center_x'],
                )
            )
        else:
            print(
                'pulse {0}->{1}: target lost (visible {2} -> {3})'.format(
                    before['pulse'], after['pulse'], before['visible'], after['visible']
                )
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
