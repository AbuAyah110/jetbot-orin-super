"""Where a VLA policy's motion intents would enter — Stage H / I5 seam only.

**There is no smolvla integration here, and there must not be one yet.** The
model is a Stage G3 deliverable and Stage G3 is still open, so this module
defines the *shape* of the seam and refuses to pretend the policy exists:
:class:`UnavailableVlaPolicy` raises
:class:`~jetbot_agent._stage.StageNotReady` when asked to propose anything.

The point of the seam is where it attaches, not what fills it. A VLA is not a
tool and not a brain: it is another *producer of motion intents*, and it enters
at :class:`~jetbot_agent.navigation.motion_adapter.BoundedMotionAdapter` —
below the tool boundary, above the sink. That placement is the guarantee:

* A :class:`MotionIntent` is converted by :func:`apply_intent` into exactly the
  same ``drive`` / ``rotate`` / ``stop`` calls a tool makes, so it gets the same
  magnitude clamping, the same bounded duration, the same ``cmd_vel`` watchdog,
  and the same latched e-stop refusal. There is no faster path.
* Intents carry a ``confidence`` and :func:`apply_intent` takes a
  ``min_confidence`` floor, because an autonomous producer with no operator in
  the loop should have to clear a bar a deliberate tool call does not.
* If a future ticket wants the model in the agent's decision loop instead, the
  right shape is a normal :class:`~jetbot_agent.agent.tools.base.ActuationTool`
  with a closed schema next to ``nav_drive``. Either way the intent lands on the
  adapter, never on a controller or a motor driver.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

from jetbot_agent._stage import StageNotReady
from jetbot_agent.agent.tools.motion import MotionDenied, MotionStatus

LOGGER = logging.getLogger('jetbot_agent.navigation.vla_seam')

INTENT_KINDS = ('drive', 'rotate', 'stop')


@dataclass(frozen=True)
class MotionIntent:
    """A motion request produced by a policy rather than by a tool call.

    Frozen, and its magnitudes are advisory: the adapter clamps them. ``source``
    exists so logs and audits can tell an autonomous intent from an
    operator-driven tool call.
    """

    kind: str
    linear: float = 0.0
    angular: float = 0.0
    duration_sec: float = 0.5
    source: str = 'unknown'
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in INTENT_KINDS:
            raise ValueError(f'intent kind must be one of {INTENT_KINDS}, got {self.kind!r}')


@runtime_checkable
class VlaPolicy(Protocol):
    """What a visual-language-action policy has to offer this seam."""

    name: str

    def propose(self, observation: Mapping[str, Any]) -> Optional[MotionIntent]:
        """Return the next intent, or ``None`` to stay put."""


class UnavailableVlaPolicy:
    """Placeholder for smolvla. Importable and constructible; not callable.

    Stage G3 has not delivered the model, so anything that reaches for it gets a
    :class:`StageNotReady` instead of a fabricated answer.
    """

    name = 'smolvla'

    def __init__(self, detail: str = 'Stage G3 (smolvla) has not landed yet') -> None:
        self.detail = detail

    def propose(self, observation: Mapping[str, Any]) -> Optional[MotionIntent]:
        raise StageNotReady(
            f'{self.name} is not available: {self.detail}. The I5 seam is defined '
            '(MotionIntent -> apply_intent -> BoundedMotionAdapter) but unfilled.'
        )


def apply_intent(
    adapter: Any,
    intent: MotionIntent,
    *,
    min_confidence: float = 0.0,
    logger: Optional[logging.Logger] = None,
) -> MotionStatus:
    """Route one policy intent through the ordinary bounded motion path.

    Nothing here interprets limits, watchdogs, or the e-stop: it translates an
    intent into the adapter's vocabulary and lets the adapter refuse or clamp.
    """
    log = logger or LOGGER
    if not isinstance(intent, MotionIntent):
        raise TypeError(f'expected a MotionIntent, got {type(intent).__name__}')

    if intent.kind == 'stop':
        log.info('vla_intent kind=stop source=%s', intent.source)
        return adapter.stop()

    if intent.confidence < min_confidence:
        log.warning('vla_intent_refused source=%s confidence=%.3f floor=%.3f',
                    intent.source, intent.confidence, min_confidence)
        adapter.stop()
        raise MotionDenied(
            f'intent from {intent.source!r} scored {intent.confidence:.3f}, below the '
            f'{min_confidence:.3f} confidence floor; stopped instead'
        )

    log.info('vla_intent kind=%s source=%s linear=%.4f angular=%.4f duration=%.3f',
             intent.kind, intent.source, intent.linear, intent.angular, intent.duration_sec)
    if intent.kind == 'rotate':
        return adapter.rotate(intent.angular, intent.duration_sec)
    return adapter.drive(intent.linear, intent.angular, intent.duration_sec)


__all__ = [
    'INTENT_KINDS',
    'MotionIntent',
    'UnavailableVlaPolicy',
    'VlaPolicy',
    'apply_intent',
]
