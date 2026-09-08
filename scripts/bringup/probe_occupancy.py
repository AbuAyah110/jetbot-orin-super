#!/usr/bin/env python3
"""Score the Python occupancy creep gate on saved frames.

Cosmos path-clear is not a sensor. This heuristic looks at colour deviation in
the lower 40% of a 448² JPEG. Run it on known-empty and known-blocked frames
before trusting a live creep.

    probe_occupancy.py --clear empty.jpg --blocked bottle.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jetbot_agent.robot_loop.demos import occupancy_score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', nargs='*', default=[])
    parser.add_argument('--blocked', nargs='*', default=[])
    args = parser.parse_args()
    correct = 0
    total = 0
    for expected, paths in (('clear', args.clear), ('blocked', args.blocked)):
        for path in paths:
            detail = occupancy_score(Path(path).read_bytes())
            if expected == 'clear':
                good = detail.get('clear') is True
            else:
                good = detail.get('blocked') is True
            correct += int(good)
            total += 1
            print(
                json.dumps(
                    {
                        'expected': expected,
                        'ok': good,
                        'frame': Path(path).name,
                        **detail,
                    }
                ),
                flush=True,
            )
    print('--- {0}/{1} correct ---'.format(correct, total))
    if total == 0:
        return 0
    return 0 if correct == total else 1


if __name__ == '__main__':
    raise SystemExit(main())
