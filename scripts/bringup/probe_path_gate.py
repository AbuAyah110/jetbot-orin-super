#!/usr/bin/env python3
"""Compare monocular path-gate prompts against saved frames.

The detour maneuver stalled because the gate answered ``clear:false`` on frames
of visibly empty floor. Run this against known-clear and known-blocked frames
before changing the gate wording, so the prompt is chosen on measured answers
rather than on how careful it reads.

Usage:
    probe_path_gate.py --clear FRAME [FRAME ...] --blocked FRAME [FRAME ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'bringup'))

from jetbot_agent.robot_loop.cosmos_runtime import (  # noqa: E402
    COSMOS_ENGINE_DIR,
    CosmosResidentClient,
    DEFAULT_CTRL_DIR,
    spawn_resident,
)

GATE_SYSTEM = (
    'You are a conservative monocular path gate. False is always safer than '
    'guessing. JSON only.'
)

# The deployed wording. "wall" is unqualified by distance, so any indoor frame
# with a baseboard in the background can read as blocked.
CURRENT = (
    'The robot is stopped. Inspect the current image only. Is the floor '
    'immediately ahead visibly clear for ONE very short forward pulse? '
    'If an object, wall, drop, feet, clutter, blur, darkness, or uncertainty '
    'could block the chassis, clear must be false. Reply exactly one compact '
    'JSON object: {"clear":true} or {"clear":false}. No explanation.'
)

# Same caution, but scoped to the strip the chassis will actually occupy during
# one short pulse, and explicit that far-away scenery is not an obstacle.
SCOPED = (
    'The robot is stopped and will move forward about 20 centimeters. '
    'Look only at the bottom half of the image, which is the floor directly in '
    'front of the wheels. Is that strip of floor empty? '
    'Empty floor, carpet, tile, rug, or shadow means clear is true. '
    'Walls, furniture, doorways, or objects in the far background or upper half '
    'of the image are NOT obstacles and must not make clear false. '
    'Only an object, foot, cable, drop, or step within that near floor strip, '
    'or an image too dark or blurred to judge, makes clear false. '
    'Reply exactly one compact JSON object: {"clear":true} or {"clear":false}. '
    'No explanation.'
)

# Neither yes/no variant discriminates: "current" answered false on 4/4 frames
# and "scoped" answered true on 4/4, including a bottle filling the frame. The
# model is following the prompt's tone, not the pixels. Ask it to name what it
# sees instead and let code make the decision.
NAMED = (
    'Look at the lower half of this image: the floor directly in front of the '
    'robot. List every object resting on that floor. Do not list the floor '
    'itself, carpet, rugs, walls, baseboards, or anything in the background. '
    'Reply exactly one compact JSON object: {"objects":["name"]} and use '
    '{"objects":[]} when that floor area is empty. No explanation.'
)

# Same naming task, but neutral about which answer is expected, to check the
# model is not simply echoing "empty".
NAMED_BALANCED = (
    'Look at the lower half of this image: the floor directly in front of the '
    'robot. Some frames show an object there and some show only empty floor; '
    'both are equally common. List every object resting on that near floor, '
    'ignoring the floor covering itself, walls, and the background. '
    'Reply exactly one compact JSON object: {"objects":["name"]}, or '
    '{"objects":[]} if the near floor is empty. No explanation.'
)

VARIANTS = {
    'current': (CURRENT, 'clear'),
    'scoped': (SCOPED, 'clear'),
    'named': (NAMED, 'objects'),
    'named_balanced': (NAMED_BALANCED, 'objects'),
}


def ask(runtime, prompt: str, key: str, jpeg: bytes) -> tuple[object, str]:
    raw = runtime.generate(
        system=GATE_SYSTEM, user_text=prompt, image_jpeg=jpeg, max_tokens=64
    )
    try:
        start = raw.index('{')
        end = raw.index('}', start) + 1
        value = json.loads(raw[start:end]).get(key)
    except (ValueError, json.JSONDecodeError):
        return 'parse_fail', raw.strip()[:120]
    if key == 'objects':
        # The decision lives in code, not in the model.
        if not isinstance(value, list):
            return 'parse_fail', raw.strip()[:120]
        return len(value) == 0, raw.strip()[:120]
    return value, raw.strip()[:120]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear', nargs='*', default=[])
    parser.add_argument('--blocked', nargs='*', default=[])
    parser.add_argument('--repeat', type=int, default=1)
    args = parser.parse_args()

    proc = spawn_resident(
        ctrl_dir=DEFAULT_CTRL_DIR,
        engine_dir=COSMOS_ENGINE_DIR / 'llm',
        multimodal_dir=COSMOS_ENGINE_DIR,
        max_tokens=64,
        binary=None,
    )
    runtime = CosmosResidentClient(
        ctrl_dir=DEFAULT_CTRL_DIR,
        jpeg_dir=REPO / 'data' / 'audio' / 'debug',
        max_tokens=64,
    )
    runtime.wait_loaded(timeout_s=180.0)

    tally: dict[tuple[str, str], list[int]] = {}
    try:
        for expected, paths in (('clear', args.clear), ('blocked', args.blocked)):
            for path in paths:
                jpeg = Path(path).read_bytes()
                for name, (prompt, key) in VARIANTS.items():
                    for _ in range(args.repeat):
                        answer, raw = ask(runtime, prompt, key, jpeg)
                        want = answer is True if expected == 'clear' else answer is False
                        bucket = tally.setdefault((name, expected), [0, 0])
                        bucket[0] += 1
                        bucket[1] += 1 if want else 0
                        print(
                            json.dumps(
                                {
                                    'variant': name,
                                    'expected': expected,
                                    'clear': answer,
                                    'correct': want,
                                    'frame': Path(path).name,
                                    'raw': raw,
                                }
                            ),
                            flush=True,
                        )
    finally:
        if proc is not None:
            proc.terminate()

    print('--- summary ---')
    for (name, expected), (total, correct) in sorted(tally.items()):
        print('{0:<8} {1:<8} {2}/{3}'.format(name, expected, correct, total))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
