#!/usr/bin/env python3
"""Orchestrator stub. Stage H / I8 — after I1–I7 gates; memory tools wait for Stage I."""

from jetbot_agent._stage import StageNotReady


def main() -> None:
    raise StageNotReady("main.py waits for Stage H / I8 (agent before memory)")


if __name__ == "__main__":
    main()
