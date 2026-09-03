"""Import-safe one-process Cosmos orchestration seam.

This is the compatible core migrated from the old ``jetbot-thin-stack``
prototype. It deliberately excludes that prototype's ROS node, HTTP VLM
client, direct wheel assignments, and motor construction. Runtime and bounded
action execution are injected only after their hardware gates pass.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from jetbot_agent.robot_loop.actions import RobotAction, parse_action
from jetbot_agent.robot_loop.history import ChatHistory
from jetbot_agent.robot_loop.prompts import prompt_suffix

SYSTEM_PROMPT = """You are the on-board planner for a small indoor JetBot.
Use the current 448x448 JPEG, optional speech, short text history, and optional
memory. Reply with exactly one JSON object with keys action, vx, wz, duration_s,
say, goal, and reason. Allowed actions are stop, drive, speak, wait, and weather.
Choose a heading, but motor power and duration are calibrated downstream. If
the requested object is not visible, blocked, uncertain, or parsing could be
unsafe, stop and use a short say. Never emit PWM, Python, markdown, or prose."""


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
    object_relative = bool(
        re.search(
            r'\b(?:toward|towards|to|find|approach)\b.*\b(?:object|chair|door|person|box|target)\b',
            request.speech,
            re.IGNORECASE,
        )
    )
    if object_relative:
        return '\n'.join(
            (
                'VISUAL GROUNDING TEST. Inspect only the attached current image.',
                'Target request: {0}'.format(request.speech),
                'Reply with one complete one-line JSON object under 60 tokens.',
                'Visible target: {"action":"drive","vx":0,"wz":0,"duration_s":0,'
                '"say":"","goal":"visible:left|center|right","reason":"grounded"}',
                'Absent/uncertain: {"action":"stop","vx":0,"wz":0,"duration_s":0,'
                '"say":"I don\'t see <target>","goal":"not_visible:<target>","reason":"grounded"}',
                'Select one literal side, never the | list. Never drive without visible:<side>.',
            )
        )
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


DRIVE_MAX_TOKENS_MIN = 64
DRIVE_MAX_TOKENS_MAX = 96


class OneProcessOrchestrator:
    """Connect one JPEG and one runtime to a fail-closed action executor."""

    def __init__(
        self,
        runtime: GenerateRuntime,
        executor: ActionExecutor,
        *,
        history: Optional[ChatHistory] = None,
        weather: Optional[Callable[[str], str]] = None,
        drive_mode: bool = False,
        drive_max_tokens: int = 80,
        think_max_tokens: int = 384,
    ) -> None:
        self.runtime = runtime
        self.executor = executor
        self.history = history or ChatHistory(max_turns=6)
        self.weather = weather
        self.drive_mode = bool(drive_mode)
        tokens = int(drive_max_tokens)
        if tokens < DRIVE_MAX_TOKENS_MIN:
            tokens = DRIVE_MAX_TOKENS_MIN
        if tokens > DRIVE_MAX_TOKENS_MAX:
            tokens = DRIVE_MAX_TOKENS_MAX
        self.drive_max_tokens = tokens
        self.think_max_tokens = int(think_max_tokens)

    def plan(self, request: LoopInput) -> RobotAction:
        """Generate and gate one action without applying model motion."""
        # Hold stop for the whole generate (~1.5 s). Wheels never start here.
        self.executor.execute(RobotAction(kind='stop', vx=0.0, wz=0.0, duration_s=0.0))
        use_drive = self.drive_mode or bool(self.executor.is_moving())
        prompt = build_user_prompt(
            request=request,
            history=self.history.render(),
            moving=use_drive,
        )
        max_tokens = self.drive_max_tokens if use_drive else self.think_max_tokens
        try:
            output = self.runtime.generate(
                system=SYSTEM_PROMPT,
                user_text=prompt,
                image_jpeg=request.image_jpeg,
                max_tokens=max_tokens,
            )
            action = parse_action(output)
        except Exception:
            action = parse_action(None)

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

    def tick(self, request: LoopInput) -> RobotAction:
        """Plan, then execute one bounded action."""
        action = self.plan(request)
        # Weather is data, never policy. Hold still while fetching it.
        if action.kind == 'weather':
            self.executor.execute(RobotAction(kind='stop'))
            if self.weather is not None:
                result = self.weather(action.query or request.speech)
                self.history.add('tool', result)
        else:
            self.executor.execute(action)
        return action
