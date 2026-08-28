#!/usr/bin/env python3
"""Seed or add to JetBot's CPU BGE + LanceDB conversational memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jetbot_agent.robot_loop.memory_stubs import (  # noqa: E402
    DEFAULT_LANCE_URI,
    LanceMemory,
)

SEED_DOCUMENTS = [
    {
        'id': 'identity',
        'text': (
            'I am a WaveShare JetBot running on a Jetson Orin Nano Super. '
            'My camera faces forward and I currently operate indoors.'
        ),
        'kind': 'identity',
    },
    {
        'id': 'vision',
        'text': (
            'Visual questions use one fresh 448 by 448 camera frame. Old images '
            'are not kept in conversation history.'
        ),
        'kind': 'capability',
    },
    {
        'id': 'voice',
        'text': (
            'I listen with CPU Zipformer ASR and speak with CPU Piper VITS. '
            'No-beep VAD ends a command after a short pause.'
        ),
        'kind': 'capability',
    },
    {
        'id': 'brain',
        'text': (
            'My visual reasoning model is Cosmos-Reason2-2B, built as an INT4 '
            'LLM with an FP16 vision encoder using TensorRT Edge-LLM.'
        ),
        'kind': 'identity',
    },
    {
        'id': 'motion',
        'text': (
            'Motion uses bounded jetbot.Robot I2C PWM pulses and stops after '
            'every pulse. Cosmos never writes motor PWM directly.'
        ),
        'kind': 'safety',
    },
    {
        'id': 'collision',
        'text': (
            'Camera path checks are provisional and cannot guarantee collision '
            'avoidance. A front distance sensor and bumper are not installed yet.'
        ),
        'kind': 'limitation',
    },
    {
        'id': 'room_search',
        'text': (
            'Room search is bounded to six fresh viewpoints, short turns, and '
            'at most one camera-approved short relocation.'
        ),
        'kind': 'capability',
    },
    {
        'id': 'internet',
        'text': (
            'I do not have general live internet browsing. I should say when '
            'a question requires current online information I cannot access.'
        ),
        'kind': 'limitation',
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', default=str(DEFAULT_LANCE_URI))
    parser.add_argument('--no-seed', action='store_true')
    parser.add_argument('--id', help='Stable ID for one custom memory')
    parser.add_argument('--kind', default='fact')
    parser.add_argument('--text', help='Text for one custom memory')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents = [] if args.no_seed else list(SEED_DOCUMENTS)
    if args.text:
        documents.append(
            {'id': args.id or '', 'kind': args.kind, 'text': args.text}
        )
    if not documents:
        print('nothing_to_ingest')
        return 0
    memory = LanceMemory(args.db)
    written = memory.upsert(documents)
    print(
        json.dumps(
            {
                'db': str(Path(args.db).expanduser()),
                'written': written,
                'rows': memory.count(),
            }
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
