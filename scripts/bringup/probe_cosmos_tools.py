#!/usr/bin/env python3
"""Probe Cosmos tool routing for abstract spoken requests.

Motion goes to an in-memory cmd_vel sink, so this never opens I2C, PWM, or
jetbot.Robot. Use it to check whether Cosmos infers the intended outcome and
picks one authorized tool, before trusting a phrasing on the live robot.

    probe_cosmos_tools.py 'Turn so you can see the doorway'

Utterances share one session, so later ones can lean on earlier replies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from jetbot_agent.agent.cosmos_voice import (  # noqa: E402
    CosmosVoiceSession,
    build_voice_backend,
)
from jetbot_agent.robot_loop.cosmos_runtime import CosmosResidentClient  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('utterances', nargs='+')
    parser.add_argument(
        '--deny-actuation',
        action='store_true',
        help='Publish nothing at all: only nav_status stays invocable.',
    )
    parser.add_argument('--ctrl-dir', default=None)
    parser.add_argument('--max-tokens', type=int, default=80)
    parser.add_argument('--timeout-s', type=float, default=90.0)
    args = parser.parse_args()

    client_kwargs = {'max_tokens': args.max_tokens, 'timeout_s': args.timeout_s}
    if args.ctrl_dir:
        client_kwargs['ctrl_dir'] = Path(args.ctrl_dir)
    runtime = CosmosResidentClient(**client_kwargs)
    runtime.wait_loaded(timeout_s=10.0)

    backend = build_voice_backend(allow_actuation=not args.deny_actuation)
    session = CosmosVoiceSession(runtime, backend)
    failures = 0
    try:
        for utterance in args.utterances:
            turn = session.handle_transcript(utterance)
            failures += int(turn.failed)
            print(
                json.dumps(
                    {
                        'said': utterance,
                        'reply': turn.say,
                        'failed': turn.failed,
                        'reason': turn.reason,
                        'tools': [call.name for call in turn.calls],
                        'args': [dict(call.args) for call in turn.calls],
                        'tool_ok': [call.result.ok for call in turn.calls],
                        'mock_twists': list(backend.sink.twists),
                    }
                ),
                flush=True,
            )
    finally:
        session.close()
    return 0 if failures == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
