"""Hermes agent harness — Stage H / I1.

A deterministic state machine for one agent turn. This module deliberately has
**no** tool execution, no LLM call, no camera, no motor, and no network access:
its only job is to sequence states, drain observations, ask a pluggable brain
for a decision, and hand any requested action to an injected sink.

Safety notes (see ``docs/safety.md``, ``docs/architecture.md``):

* Nothing here can reach PWM, GPIO, or I2C. The harness never imports
  ``jetbot_control``, ``jetbot_agent.hardware``, or the tool package; actions
  leave through :class:`ActionSink`, which Stage H / I2–I8 wires to the
  permissioned tool registry.
* E-stop is a privileged edge reachable from every state and latches by
  default, matching ``estop.latch: true`` in ``config/robot.yaml``.
* The core loop never sleeps for longer than :data:`MAX_BLOCK_SEC`, so an
  e-stop raised from another thread is observed within one short slice.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

LOGGER = logging.getLogger('jetbot_agent.agent.hermes')

# Longest single wait the core loop may perform. Any queue wait is clamped to
# this so a latching e-stop is picked up promptly.
MAX_BLOCK_SEC = 0.05


class State(Enum):
    """Harness states. ``SHUTDOWN`` is terminal."""

    IDLE = 'IDLE'
    PERCEIVING = 'PERCEIVING'
    DELIBERATING = 'DELIBERATING'
    ACTING = 'ACTING'
    SPEAKING = 'SPEAKING'
    ERROR = 'ERROR'
    ESTOP = 'ESTOP'
    SHUTDOWN = 'SHUTDOWN'


class HarnessError(RuntimeError):
    """Base class for harness faults."""


class IllegalTransition(HarnessError):
    """Requested state change is not in the transition table."""


class EstopLatched(HarnessError):
    """E-stop is latched; an explicit ``clear_estop()`` is required."""


class BrainContractError(HarnessError):
    """The brain returned something that is not a :class:`Decision`."""


def _build_transitions() -> Dict[State, frozenset]:
    """Legal, non-privileged transitions.

    ``ESTOP`` is intentionally absent from every value: the only way in is
    :meth:`HermesHarness.trigger_estop`, and the only way out is
    :meth:`HermesHarness.clear_estop`.
    """
    table: Dict[State, set] = {
        State.IDLE: {State.PERCEIVING, State.DELIBERATING, State.SHUTDOWN},
        State.PERCEIVING: {State.DELIBERATING, State.IDLE},
        State.DELIBERATING: {State.ACTING, State.SPEAKING, State.PERCEIVING, State.IDLE},
        State.ACTING: {State.DELIBERATING, State.SPEAKING, State.IDLE},
        State.SPEAKING: {State.DELIBERATING, State.IDLE},
        State.ERROR: {State.IDLE, State.SHUTDOWN},
        State.ESTOP: {State.SHUTDOWN},
        State.SHUTDOWN: set(),
    }
    for state, targets in table.items():
        if state not in (State.SHUTDOWN, State.ERROR, State.ESTOP):
            targets.add(State.ERROR)
    return {state: frozenset(targets) for state, targets in table.items()}


LEGAL_TRANSITIONS: Dict[State, frozenset] = _build_transitions()


def is_legal(source: State, target: State) -> bool:
    """True if ``source -> target`` is allowed without a privileged edge."""
    return target in LEGAL_TRANSITIONS[source]


class DecisionKind(Enum):
    SPEAK = 'speak'
    ACT = 'act'
    WAIT = 'wait'
    DONE = 'done'


@dataclass(frozen=True)
class Action:
    """A high-level action request. Names are tool names, never PWM values."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError('action name must be a non-empty string')
        if not isinstance(self.arguments, Mapping):
            raise ValueError('action arguments must be a mapping')


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    text: str = ''
    action: Optional[Action] = None
    rationale: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.kind, DecisionKind):
            raise ValueError('decision kind must be a DecisionKind')
        if self.kind is DecisionKind.ACT and self.action is None:
            raise ValueError('ACT decision requires an action')
        if self.kind is DecisionKind.SPEAK and not self.text:
            raise ValueError('SPEAK decision requires text')


@dataclass(frozen=True)
class Observation:
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    ts: float = 0.0


@dataclass(frozen=True)
class ActionOutcome:
    """Result of handing an action to the sink. ``executed`` is False in I1."""

    executed: bool
    detail: str = ''
    value: Any = None


@dataclass
class TurnContext:
    """Read-mostly view handed to the brain on every step."""

    turn_index: int
    step: int
    max_steps: int
    prompt: Optional[str]
    observations: Tuple[Observation, ...] = ()
    decisions: Tuple[Decision, ...] = ()
    outcomes: Tuple[ActionOutcome, ...] = ()
    estop_active: bool = False

    @property
    def steps_remaining(self) -> int:
        return max(0, self.max_steps - self.step)


@dataclass(frozen=True)
class TurnResult:
    turn_index: int
    steps: int
    stop_reason: str
    final_state: State
    spoken: Tuple[str, ...] = ()
    actions: Tuple[Action, ...] = ()
    outcomes: Tuple[ActionOutcome, ...] = ()
    error: str = ''


@dataclass(frozen=True)
class HarnessEvent:
    event: str
    ts: float
    fields: Mapping[str, Any]


@runtime_checkable
class Brain(Protocol):
    """Pluggable decision maker. A real VLM/LLM client is injected later."""

    def decide(self, context: TurnContext) -> Decision:  # pragma: no cover - protocol
        ...


class BaseBrain(ABC):
    """Optional ABC for brains that want a name and shared plumbing."""

    name: str = 'brain'

    @abstractmethod
    def decide(self, context: TurnContext) -> Decision:
        """Return the next :class:`Decision` for this turn."""


class FakeBrain(BaseBrain):
    """Deterministic scripted brain for tests and dry runs.

    Yields ``script`` decisions in order, then ``DONE`` forever (or repeats the
    script when ``loop=True``, which is how the max-steps guard is exercised).
    """

    name = 'fake'

    def __init__(self, script: Sequence[Decision], *, loop: bool = False) -> None:
        self._script = tuple(script)
        self._loop = loop
        self.calls: List[TurnContext] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def decide(self, context: TurnContext) -> Decision:
        self.calls.append(context)
        index = len(self.calls) - 1
        if not self._script:
            return Decision(DecisionKind.DONE)
        if index < len(self._script):
            return self._script[index]
        if self._loop:
            return self._script[index % len(self._script)]
        return Decision(DecisionKind.DONE)


@runtime_checkable
class ActionSink(Protocol):
    """Where actions leave the harness. I2's registry implements this."""

    def dispatch(self, action: Action) -> ActionOutcome:  # pragma: no cover - protocol
        ...


@runtime_checkable
class SpeechSink(Protocol):
    def say(self, text: str) -> None:  # pragma: no cover - protocol
        ...


class NullActionSink:
    """Records actions and executes nothing. Default for I1."""

    def __init__(self) -> None:
        self.actions: List[Action] = []

    def dispatch(self, action: Action) -> ActionOutcome:
        self.actions.append(action)
        return ActionOutcome(executed=False, detail='no tool layer wired (Stage H / I1)')


class RecordingSpeechSink:
    """Collects speech instead of touching ALSA. Voice tools are I6."""

    def __init__(self) -> None:
        self.said: List[str] = []

    def say(self, text: str) -> None:
        self.said.append(text)


@dataclass
class HarnessConfig:
    max_steps: int = 8
    estop_latch: bool = True
    idle_wait_sec: float = 0.0
    event_log_limit: int = 256

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError('max_steps must be >= 1')
        if self.idle_wait_sec < 0.0:
            raise ValueError('idle_wait_sec must be >= 0')

    @classmethod
    def from_robot_config(cls, config: Mapping[str, Any], **overrides: Any) -> 'HarnessConfig':
        """Build from a parsed ``config/robot.yaml`` mapping.

        Only ``estop.latch`` is read; motor limits stay in deterministic code
        below the tool boundary and are never harness policy.
        """
        estop = config.get('estop') or {}
        latch = bool(estop.get('latch', True))
        return cls(estop_latch=latch, **overrides)


class HermesHarness:
    """Turn-based agent state machine with a latching e-stop.

    The harness owns state and sequencing only. Perception, tools, and speech
    are injected, so tests run with no hardware and no model weights.
    """

    def __init__(
        self,
        brain: Brain,
        config: Optional[HarnessConfig] = None,
        *,
        action_sink: Optional[ActionSink] = None,
        speech_sink: Optional[SpeechSink] = None,
        clock: Callable[[], float] = time.monotonic,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not hasattr(brain, 'decide'):
            raise TypeError('brain must implement decide(context) -> Decision')
        self._brain = brain
        self._config = config or HarnessConfig()
        self._action_sink: ActionSink = action_sink or NullActionSink()
        self._speech_sink: SpeechSink = speech_sink or RecordingSpeechSink()
        self._clock = clock
        self._log = logger or LOGGER

        self._state = State.IDLE
        self._lock = threading.RLock()
        self._estop = threading.Event()
        self._estop_reason = ''
        self._queue: 'Queue[Observation]' = Queue()
        self._events: Deque[HarnessEvent] = deque(maxlen=self._config.event_log_limit)
        self._estop_hooks: List[Callable[[str], None]] = []
        self._turn_index = 0
        self._emit('harness_init', state=self._state.value, max_steps=self._config.max_steps,
                   estop_latch=self._config.estop_latch, brain=type(brain).__name__)

    # ---------------------------------------------------------------- state

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    @property
    def config(self) -> HarnessConfig:
        return self._config

    @property
    def estop_active(self) -> bool:
        return self._estop.is_set()

    @property
    def estop_reason(self) -> str:
        return self._estop_reason

    @property
    def turn_count(self) -> int:
        return self._turn_index

    @property
    def events(self) -> Tuple[HarnessEvent, ...]:
        with self._lock:
            return tuple(self._events)

    @property
    def pending_observations(self) -> int:
        return self._queue.qsize()

    def transition(self, target: State, reason: str = '') -> None:
        """Public, table-checked state change.

        ``ESTOP`` is never legal here on purpose — use :meth:`trigger_estop`.
        """
        with self._lock:
            if not is_legal(self._state, target):
                self._emit('illegal_transition', **{'from': self._state.value, 'to': target.value})
                raise IllegalTransition(f'{self._state.value} -> {target.value} is not allowed')
            self._set_state(target, reason)

    def _set_state(self, target: State, reason: str = '') -> None:
        previous = self._state
        self._state = target
        self._emit('state_transition', **{'from': previous.value, 'to': target.value, 'reason': reason})

    # ---------------------------------------------------------------- estop

    def add_estop_hook(self, hook: Callable[[str], None]) -> None:
        """Register a fail-safe callback invoked on every e-stop trigger."""
        self._estop_hooks.append(hook)

    def trigger_estop(self, reason: str = 'operator') -> None:
        """Privileged edge into ``ESTOP``, legal from every state and idempotent.

        Hook exceptions are swallowed: nothing may prevent an e-stop.
        """
        with self._lock:
            already = self._estop.is_set()
            self._estop.set()
            if not already:
                self._estop_reason = reason
            if self._state is not State.ESTOP:
                self._set_state(State.ESTOP, f'estop: {reason}')
            self._emit('estop_triggered', reason=reason, repeated=already,
                       latch=self._config.estop_latch)
        for hook in list(self._estop_hooks):
            try:
                hook(reason)
            except Exception as exc:  # noqa: BLE001 - an e-stop must never fail
                self._emit('estop_hook_failed', error=repr(exc))

    def clear_estop(self) -> None:
        """Explicit operator clear. The only exit from ``ESTOP``."""
        with self._lock:
            if not self._estop.is_set():
                return
            self._estop.clear()
            reason, self._estop_reason = self._estop_reason, ''
            if self._state is State.ESTOP:
                self._set_state(State.IDLE, 'estop cleared')
            self._emit('estop_cleared', previous_reason=reason)

    def recover(self) -> None:
        """Leave ``ERROR`` for ``IDLE`` after the caller has handled the fault."""
        with self._lock:
            if self._estop.is_set():
                raise EstopLatched('clear_estop() is required before recover()')
            if self._state is not State.ERROR:
                raise HarnessError(f'recover() is only valid in ERROR (state={self._state.value})')
            self._set_state(State.IDLE, 'recovered')

    # --------------------------------------------------------- observations

    def submit(self, observation: Observation) -> None:
        """Thread-safe enqueue. Rejected after shutdown."""
        if not isinstance(observation, Observation):
            raise TypeError('submit() expects an Observation')
        with self._lock:
            if self._state is State.SHUTDOWN:
                raise HarnessError('harness is shut down')
        self._queue.put(observation)

    def _drain(self) -> Tuple[Observation, ...]:
        """Non-blocking drain, with at most one clamped wait when configured."""
        drained: List[Observation] = []
        wait = min(self._config.idle_wait_sec, MAX_BLOCK_SEC)
        if wait > 0.0:
            try:
                drained.append(self._queue.get(timeout=wait))
            except Empty:
                pass
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except Empty:
                break
        return tuple(drained)

    # ------------------------------------------------------------- the turn

    def run_turn(
        self,
        prompt: Optional[str] = None,
        observations: Iterable[Observation] = (),
    ) -> TurnResult:
        """Run one bounded turn from ``IDLE`` and return why it stopped."""
        with self._lock:
            if self._estop.is_set():
                if self._config.estop_latch:
                    raise EstopLatched(f'e-stop latched ({self._estop_reason}); clear_estop() required')
                self._emit('estop_auto_cleared', reason=self._estop_reason)
                self._estop.clear()
                self._estop_reason = ''
                if self._state is State.ESTOP:
                    self._set_state(State.IDLE, 'non-latching estop auto-clear')
            if self._state is State.SHUTDOWN:
                raise HarnessError('harness is shut down')
            if self._state is State.ERROR:
                raise HarnessError('recover() is required before another turn')
            if self._state is not State.IDLE:
                raise HarnessError(f'run_turn() requires IDLE (state={self._state.value})')
            self._turn_index += 1
            turn_index = self._turn_index

        for observation in observations:
            self.submit(observation)
        if prompt is not None:
            self.submit(Observation('prompt', {'text': prompt}, ts=self._clock()))

        seen: List[Observation] = []
        decisions: List[Decision] = []
        outcomes: List[ActionOutcome] = []
        spoken: List[str] = []
        actions: List[Action] = []
        steps = 0
        self._emit('turn_start', turn=turn_index, prompt=prompt)

        try:
            self.transition(State.PERCEIVING, 'turn start')
            seen.extend(self._drain())
            self.transition(State.DELIBERATING, 'observations drained')
        except IllegalTransition:
            # Only reachable if another thread e-stopped us mid-setup.
            if self._estop.is_set():
                return self._finish(turn_index, steps, 'estop', spoken, actions, outcomes)
            raise

        while True:
            if self._estop.is_set():
                return self._finish(turn_index, steps, 'estop', spoken, actions, outcomes)
            if steps >= self._config.max_steps:
                self._emit('max_steps_exceeded', turn=turn_index, steps=steps,
                           max_steps=self._config.max_steps)
                with self._lock:
                    if self._state is not State.ERROR:
                        self._set_state(State.ERROR, 'max steps exceeded')
                return self._finish(turn_index, steps, 'max_steps', spoken, actions, outcomes)

            context = TurnContext(
                turn_index=turn_index,
                step=steps,
                max_steps=self._config.max_steps,
                prompt=prompt,
                observations=tuple(seen),
                decisions=tuple(decisions),
                outcomes=tuple(outcomes),
                estop_active=self._estop.is_set(),
            )

            try:
                decision = self._brain.decide(context)
                if not isinstance(decision, Decision):
                    raise BrainContractError(
                        f'brain returned {type(decision).__name__}, expected Decision'
                    )
            except Exception as exc:  # noqa: BLE001 - a bad brain must not wedge the robot
                return self._fail(turn_index, steps, exc, spoken, actions, outcomes)

            steps += 1
            decisions.append(decision)
            self._emit('decision', turn=turn_index, step=steps, kind=decision.kind.value,
                       rationale=decision.rationale)

            if self._estop.is_set():
                return self._finish(turn_index, steps, 'estop', spoken, actions, outcomes)

            try:
                if decision.kind is DecisionKind.DONE:
                    self.transition(State.IDLE, 'turn done')
                    return self._finish(turn_index, steps, 'done', spoken, actions, outcomes)

                if decision.kind is DecisionKind.WAIT:
                    self.transition(State.PERCEIVING, 'brain waiting')
                    new = self._drain()
                    seen.extend(new)
                    self._emit('wait', turn=turn_index, step=steps, new_observations=len(new))
                    self.transition(State.DELIBERATING, 'wait complete')
                    continue

                if decision.kind is DecisionKind.ACT:
                    assert decision.action is not None  # guarded by Decision.__post_init__
                    self.transition(State.ACTING, f'action {decision.action.name}')
                    outcome = self._action_sink.dispatch(decision.action)
                    if not isinstance(outcome, ActionOutcome):
                        raise HarnessError('action sink must return an ActionOutcome')
                    actions.append(decision.action)
                    outcomes.append(outcome)
                    self._emit('action_dispatched', turn=turn_index, step=steps,
                               action=decision.action.name, executed=outcome.executed,
                               detail=outcome.detail)
                    self.transition(State.DELIBERATING, 'action complete')
                    continue

                self.transition(State.SPEAKING, 'speaking')
                self._speech_sink.say(decision.text)
                spoken.append(decision.text)
                self._emit('spoke', turn=turn_index, step=steps, chars=len(decision.text))
                self.transition(State.DELIBERATING, 'speech complete')
            except Exception as exc:  # noqa: BLE001 - sink faults land in ERROR, not a crash
                return self._fail(turn_index, steps, exc, spoken, actions, outcomes)

    def _finish(
        self,
        turn_index: int,
        steps: int,
        reason: str,
        spoken: Sequence[str],
        actions: Sequence[Action],
        outcomes: Sequence[ActionOutcome],
        error: str = '',
    ) -> TurnResult:
        result = TurnResult(
            turn_index=turn_index,
            steps=steps,
            stop_reason=reason,
            final_state=self.state,
            spoken=tuple(spoken),
            actions=tuple(actions),
            outcomes=tuple(outcomes),
            error=error,
        )
        self._emit('turn_end', turn=turn_index, steps=steps, stop_reason=reason,
                   state=result.final_state.value, error=error)
        return result

    def _fail(
        self,
        turn_index: int,
        steps: int,
        exc: BaseException,
        spoken: Sequence[str],
        actions: Sequence[Action],
        outcomes: Sequence[ActionOutcome],
    ) -> TurnResult:
        if self._estop.is_set():
            # An e-stop that races a sink or transition is reported as an e-stop.
            self._emit('turn_error_during_estop', turn=turn_index, step=steps, error=repr(exc))
            return self._finish(turn_index, steps, 'estop', spoken, actions, outcomes,
                                error=repr(exc))
        self._emit('turn_error', turn=turn_index, step=steps, error=repr(exc))
        with self._lock:
            if self._state not in (State.ERROR, State.ESTOP, State.SHUTDOWN):
                self._set_state(State.ERROR, f'fault: {type(exc).__name__}')
        return self._finish(turn_index, steps, 'error', spoken, actions, outcomes, error=repr(exc))

    # ---------------------------------------------------------- lifecycle

    def shutdown(self, reason: str = 'requested') -> None:
        """Idempotent clean shutdown: drop pending observations, go terminal."""
        with self._lock:
            if self._state is State.SHUTDOWN:
                return
            dropped = 0
            while True:
                try:
                    self._queue.get_nowait()
                    dropped += 1
                except Empty:
                    break
            self._set_state(State.SHUTDOWN, f'shutdown: {reason}')
            self._emit('shutdown', reason=reason, dropped_observations=dropped,
                       estop_active=self._estop.is_set())

    def __enter__(self) -> 'HermesHarness':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.trigger_estop(f'exception: {exc_type.__name__}')
        self.shutdown('context exit')

    # ---------------------------------------------------------- logging

    def _emit(self, event: str, **fields: Any) -> None:
        record = HarnessEvent(event=event, ts=self._clock(), fields=dict(fields))
        self._events.append(record)
        self._log.info('%s %s', event, ' '.join(f'{k}={v!r}' for k, v in sorted(fields.items())))


__all__ = [
    'MAX_BLOCK_SEC',
    'Action',
    'ActionOutcome',
    'ActionSink',
    'BaseBrain',
    'Brain',
    'BrainContractError',
    'Decision',
    'DecisionKind',
    'EstopLatched',
    'FakeBrain',
    'HarnessConfig',
    'HarnessError',
    'HarnessEvent',
    'HermesHarness',
    'IllegalTransition',
    'LEGAL_TRANSITIONS',
    'NullActionSink',
    'Observation',
    'RecordingSpeechSink',
    'SpeechSink',
    'State',
    'TurnContext',
    'TurnResult',
    'is_legal',
]
