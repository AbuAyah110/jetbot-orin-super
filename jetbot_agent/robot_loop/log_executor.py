"""Fail-closed look-then-log executor. Never opens motors or plays TTS."""

from __future__ import annotations

from typing import Optional

from jetbot_agent.robot_loop.actions import RobotAction


class LogOnlyExecutor:
    """Records gated actions. ``is_moving`` is always false; wheels stay stopped."""

    def __init__(self) -> None:
        self.last: Optional[RobotAction] = None
        self.holding_stop = True
        self.history: list[RobotAction] = []

    def is_moving(self) -> bool:
        return False

    def execute(self, action: RobotAction) -> None:
        if action is None:
            action = RobotAction(kind='stop', raw_ok=False, reason='parse_fail')
        # Motion is never applied. Speak is never played.
        self.last = action
        self.history.append(action)
        self.holding_stop = action.kind != 'drive' or (action.vx == 0.0 and action.wz == 0.0)
