"""Import-safe one-process Cosmos orchestration seam.

This is the compatible core migrated from the old ``jetbot-thin-stack``
prototype. It deliberately excludes that prototype's ROS node, HTTP VLM
client, direct wheel assignments, and motor construction. Runtime and bounded
action execution are injected only after their hardware gates pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from jetbot_agent.robot_loop.actions import RobotAction, parse_action
from jetbot_agent.robot_loop.history import ChatHistory
from jetbot_agent.robot_loop.prompts import prompt_suffix

SYSTEM_PROMPT = """You are the on-board planner for a small indoor JetBot.
Use the current 448x448 JPEG, optional speech, short text history, and optional
memory. Reply with exactly one JSON object. Allowed actions are stop, drive,
speak, wait, and weather. If blocked, uncertain, or parsing could be unsafe,
stop. Never emit PWM, wheel values, Python, markdown, or extra prose."""


class GenerateRuntime(Protocol):
    """In-process generation only; no HTTP server contract."""

    def generate(
        self,
        *,
        system: str,
        user_text: str,
        image_jpeg: Optional[bytes],
        max_tokens: int,
    ) -> str:
        ...


class ActionExecutor(Protocol):
    """Bounded executor below the model; implementation owns duration-stop."""

    def is_moving(self) -> bool:
        ...

    def execute(self, action: RobotAction) -> None:
        ...


@dataclass(frozen=True)
class LoopInput:
    speech: str = ''
    goal: str = ''
    rag: str = ''
    image_jpeg: Optional[bytes] = None


def build_user_prompt(*, request: LoopInput, history: str, moving: bool) -> str:
    """Build one bounded prompt; prior images are never included."""
    base = '\n'.join(
        (
            'Respond with JSON only.',
            'Active goal: {0}'.format(request.goal or 'none'),
            'Latest speech: {0}'.format(request.speech or 'none'),
            'Chat history:\n{0}'.format(history),
            'Memory:\n{0}'.format(request.rag or '(none)'),
        )
    )
    return base + '\n' + prompt_suffix(moving=moving)


class OneProcessOrchestrator:
    """Connect one JPEG and one runtime to a fail-closed action executor."""

    def __init__(
        self,
        runtime: GenerateRuntime,
        executor: ActionExecutor,
        *,
        history: Optional[ChatHistory] = None,
        weather: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.runtime = runtime
        self.executor = executor
        self.history = history or ChatHistory(max_turns=6)
        self.weather = weather

    def tick(self, request: LoopInput) -> RobotAction:
        moving = bool(self.executor.is_moving())
        prompt = build_user_prompt(
            request=request,
            history=self.history.render(),
            moving=moving,
        )
        try:
            output = self.runtime.generate(
                system=SYSTEM_PROMPT,
                user_text=prompt,
                image_jpeg=request.image_jpeg,
                max_tokens=80 if moving else 384,
            )
            action = parse_action(output)
        except Exception:
            action = parse_action(None)

        # Weather is data, never policy. Hold still while fetching it.
        if action.kind == 'weather':
            self.executor.execute(RobotAction(kind='stop'))
            if self.weather is not None:
                result = self.weather(action.query or request.speech)
                self.history.add('tool', result)
        else:
            self.executor.execute(action)

        self.history.add('user', request.speech)
        self.history.add(
            'assistant',
            json.dumps(
                {
                    'action': action.kind,
                    'vx': action.vx,
                    'wz': action.wz,
                    'duration_s': action.duration_s,
                    'say': action.say if action.kind == 'speak' else '',
                    'goal': action.goal,
                    'reason': action.reason,
                },
                separators=(',', ':'),
            ),
        )
        return action
