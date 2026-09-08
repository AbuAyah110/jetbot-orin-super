#!/usr/bin/env python3
"""Report which voice route claims a phrase, without touching hardware.

Mirrors the dispatch order in talk_and_drive.py so a phrasing can be checked
before it is spoken at the robot. This copy of the order must be updated when a
route is added or reordered, or it will confidently report a stale answer.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jetbot_agent.robot_loop.intents import (  # noqa: E402
    is_around_request,
    is_behind_request,
    is_creep_request,
    is_describe_request,
    is_place_query,
    is_place_teach,
    is_plan_preview_request,
    is_search_request,
    is_show_and_tell,
    is_think_request,
    is_visual_question,
    is_where_request,
    match_intent,
    memory_fact,
    search_target,
    search_wants_approach,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from talk_and_drive import object_relative_request  # noqa: E402


def route_for(speech: str) -> str:
    if memory_fact(speech):
        return 'remember'
    if is_place_teach(speech):
        return 'place_teach'
    if is_place_query(speech):
        return 'place_query'
    if is_think_request(speech):
        return 'think'
    if is_creep_request(speech):
        return 'creep'
    if is_where_request(speech):
        return 'where_eyes_first'
    if is_search_request(speech):
        if search_wants_approach(speech):
            return 'camera_search+approach_plan'
        return 'camera_search'
    if is_plan_preview_request(speech):
        return 'plan_preview'
    if is_show_and_tell(speech) or is_describe_request(speech):
        return 'describe'
    if is_visual_question(speech):
        return 'visual_conversation'
    if match_intent(speech) is not None:
        return 'intent'
    if is_behind_request(speech):
        return 'behind_orbit'
    if is_around_request(speech):
        return 'around_detour'
    if object_relative_request(speech):
        return 'approach_plan'
    return 'conversation'


def main(argv: list[str]) -> int:
    phrases = argv[1:]
    if not phrases:
        phrases = [line.strip() for line in sys.stdin if line.strip()]
    for phrase in phrases:
        print('{0:32s} {1}'.format(route_for(phrase), phrase))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
