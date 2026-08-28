#!/usr/bin/env python3
"""Score pixel colour grounding against frames with a known answer.

Colour grounding is the only near-field perception on this robot that works, so
threshold changes must be measured rather than reasoned about. Pass frames where
the target is present and frames where it is absent, and this reports what
``locate_color`` decides for each.

    eval_color_grounding.py --target 'red object' \
        --present a.jpg b.jpg --absent c.jpg d.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jetbot_agent.robot_loop.color_grounding import locate_color  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', default='red object')
    parser.add_argument('--present', nargs='*', default=[])
    parser.add_argument('--absent', nargs='*', default=[])
    args = parser.parse_args()

    correct = 0
    total = 0
    for expected, paths in (('present', args.present), ('absent', args.absent)):
        for path in paths:
            found = locate_color(Path(path).read_bytes(), args.target)
            good = found.visible if expected == 'present' else not found.visible
            correct += 1 if good else 0
            total += 1
            print(
                json.dumps(
                    {
                        'expected': expected,
                        'ok': good,
                        'visible': found.visible,
                        'side': found.side,
                        'pixels': found.pixels,
                        'fraction': round(found.fraction, 4),
                        'span': round(found.span, 3),
                        'rejected': found.rejected,
                        'frame': Path(path).name,
                    }
                ),
                flush=True,
            )
    print('--- {0}/{1} correct ---'.format(correct, total))
    return 0 if correct == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
