"""Deny-by-default tool registry with a per-call watchdog — Stage H / I2.

Two independent gates must both open before a tool can run:

1. **Allow-list.** ``register(tool)`` only catalogues a tool. It stays
   un-invocable until :meth:`ToolRegistry.allow` names it.
2. **Capability grant.** The tool's :class:`~.base.RiskClass` maps to a
   :class:`~.base.Capability` the operator must have granted.
   ``Capability.ACTUATE`` additionally requires ``operator_ack=True`` — an agent
   holding a registry cannot opt itself into moving the robot.

Every call runs under a watchdog whose window is
:func:`~.base.effective_timeout`, clamped to ``MAX_TOOL_TIMEOUT_SEC``. No public
method accepts a caller-supplied timeout, so a model cannot extend or disable
it. If an actuation call times out, the registry immediately issues
``motion.stop()``; the deterministic ``cmd_vel`` watchdog in ``jetbot_base``
remains the authoritative backstop.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from .base import (
    Capability,
    RISK_CAPABILITY,
    RiskClass,
    Tool,
    ToolContext,
    ToolError,
    ToolExecutionError,
    ToolPermissionError,
    ToolRegistrationError,
    ToolSafetyViolation,
    ToolTimeoutError,
    effective_timeout,
)

LOGGER = logging.getLogger('jetbot_agent.agent.tools.registry')


@dataclass(frozen=True)
class ToolResult:
    """Outcome of a dispatch. ``ok=False`` carries the refusal reason."""

    name: str
    ok: bool
    value: Any = None
    error: str = ''
    error_type: str = ''
    duration_sec: float = 0.0
    timed_out: bool = False
    risk: str = ''


@dataclass
class _Entry:
    tool: Tool
    allowed: bool = False
    calls: int = 0
    denials: int = 0
    timeouts: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """Catalogue + permission gate + watchdog for tool calls."""

    def __init__(
        self,
        context: ToolContext,
        *,
        capabilities: Iterable[Capability] = (),
        operator_ack_actuation: bool = False,
        logger: Optional[logging.Logger] = None,
        clock=time.monotonic,
    ) -> None:
        if not isinstance(context, ToolContext):
            raise ToolSafetyViolation('ToolRegistry requires a ToolContext')
        requested = frozenset(capabilities)
        bad = [cap for cap in requested if not isinstance(cap, Capability)]
        if bad:
            raise ToolRegistrationError(f'not Capability values: {bad}')
        if Capability.ACTUATE in requested and not operator_ack_actuation:
            raise ToolPermissionError(
                'Capability.ACTUATE requires operator_ack_actuation=True '
                '(wheels-up sign-off, see docs/safety.md)'
            )
        self._context = context
        self._capabilities = set(requested)
        self._entries: Dict[str, _Entry] = {}
        self._log = logger or LOGGER
        self._clock = clock
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='jetbot-tool')
        self._closed = False

    # ------------------------------------------------------------- catalogue

    @property
    def context(self) -> ToolContext:
        return self._context

    @property
    def capabilities(self) -> frozenset:
        return frozenset(self._capabilities)

    def register(self, tool: Tool, *, allow: bool = False) -> None:
        """Catalogue a tool. It remains denied until :meth:`allow` names it."""
        if not isinstance(tool, Tool):
            raise ToolRegistrationError(f'{tool!r} is not a Tool instance')
        if not tool.name:
            raise ToolRegistrationError(f'{type(tool).__name__} has no name')
        if tool.name in self._entries:
            raise ToolRegistrationError(f'tool {tool.name!r} is already registered')
        self._entries[tool.name] = _Entry(tool=tool, allowed=bool(allow))
        self._log.info('tool_registered name=%r risk=%s allowed=%s timeout=%.3f',
                       tool.name, tool.risk.value, bool(allow), effective_timeout(tool))

    def allow(self, name: str) -> None:
        self._entry(name).allowed = True
        self._log.info('tool_allowed name=%r', name)

    def deny(self, name: str) -> None:
        self._entry(name).allowed = False
        self._log.info('tool_denied name=%r', name)

    def grant(self, capability: Capability, *, operator_ack: bool = False) -> None:
        """Grant a capability. ``ACTUATE`` needs an explicit operator ack."""
        if not isinstance(capability, Capability):
            raise ToolRegistrationError(f'{capability!r} is not a Capability')
        if capability is Capability.ACTUATE and not operator_ack:
            raise ToolPermissionError(
                'Capability.ACTUATE requires operator_ack=True (see docs/safety.md)'
            )
        self._capabilities.add(capability)
        self._log.info('capability_granted capability=%s', capability.value)

    def revoke(self, capability: Capability) -> None:
        self._capabilities.discard(capability)
        self._log.info('capability_revoked capability=%s', capability.value)

    def names(self) -> Tuple[str, ...]:
        return tuple(sorted(self._entries))

    def invocable(self) -> Tuple[str, ...]:
        return tuple(sorted(
            name for name, entry in self._entries.items()
            if entry.allowed and RISK_CAPABILITY[entry.tool.risk] in self._capabilities
        ))

    def describe(self, *, invocable_only: bool = True) -> List[Dict[str, Any]]:
        """Function-calling descriptions to put in front of a model.

        Defaults to invocable tools only, so a model is not even told about
        capabilities the operator has not granted.
        """
        allowed = set(self.invocable())
        return [
            entry.tool.describe()
            for name, entry in sorted(self._entries.items())
            if not invocable_only or name in allowed
        ]

    def stats(self, name: str) -> Dict[str, int]:
        entry = self._entry(name)
        return {'calls': entry.calls, 'denials': entry.denials, 'timeouts': entry.timeouts}

    def _entry(self, name: str) -> _Entry:
        try:
            return self._entries[name]
        except KeyError:
            raise ToolPermissionError(f'tool {name!r} is not registered (deny by default)') from None

    # ------------------------------------------------------------- dispatch

    def _authorize(self, name: str) -> _Entry:
        entry = self._entry(name)
        tool = entry.tool
        if not entry.allowed:
            entry.denials += 1
            raise ToolPermissionError(f'tool {name!r} is registered but not on the allow-list')
        capability = RISK_CAPABILITY[tool.risk]
        if capability not in self._capabilities:
            entry.denials += 1
            raise ToolPermissionError(
                f'tool {name!r} needs capability {capability.value!r} '
                f'(risk={tool.risk.value}); operator has not granted it'
            )
        if tool.risk is RiskClass.ACTUATION and self._context.motion is None:
            entry.denials += 1
            raise ToolSafetyViolation(
                f'tool {name!r} is an actuation tool but no MotionInterface is wired'
            )
        return entry

    def invoke(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> Any:
        """Run a tool under the watchdog. Raises typed :class:`ToolError`s."""
        if self._closed:
            raise ToolError('registry is closed')
        entry = self._authorize(name)
        tool = entry.tool
        # Validate before spending a worker thread; execute() re-validates.
        validated = tool.validate(arguments)
        timeout = effective_timeout(tool)
        started = self._clock()
        self._log.info('tool_call name=%r risk=%s timeout=%.3f args=%r',
                       name, tool.risk.value, timeout, validated)
        future: Future = self._executor.submit(tool.execute, self._context, validated)
        try:
            value = future.result(timeout=timeout)
        except FutureTimeout:
            entry.timeouts += 1
            future.cancel()
            self._on_timeout(tool)
            elapsed = self._clock() - started
            self._log.error('tool_timeout name=%r timeout=%.3f elapsed=%.3f',
                            name, timeout, elapsed)
            raise ToolTimeoutError(
                f'tool {name!r} exceeded its {timeout:.3f}s watchdog window'
            ) from None
        except ToolError:
            entry.calls += 1
            raise
        except Exception as exc:  # noqa: BLE001 - tool bodies must not crash the agent
            entry.calls += 1
            self._log.error('tool_failed name=%r error=%r', name, exc)
            raise ToolExecutionError(f'tool {name!r} raised: {exc!r}') from exc
        entry.calls += 1
        self._log.info('tool_ok name=%r elapsed=%.3f', name, self._clock() - started)
        return value

    def dispatch(self, name: str, arguments: Optional[Mapping[str, Any]] = None) -> ToolResult:
        """Loop-friendly wrapper: never raises for tool-layer faults."""
        started = self._clock()
        risk = ''
        entry = self._entries.get(name)
        if entry is not None:
            risk = entry.tool.risk.value
        try:
            value = self.invoke(name, arguments)
        except ToolTimeoutError as exc:
            return ToolResult(name=name, ok=False, error=str(exc), error_type='ToolTimeoutError',
                              duration_sec=self._clock() - started, timed_out=True, risk=risk)
        except ToolError as exc:
            return ToolResult(name=name, ok=False, error=str(exc),
                              error_type=type(exc).__name__,
                              duration_sec=self._clock() - started, risk=risk)
        return ToolResult(name=name, ok=True, value=value,
                          duration_sec=self._clock() - started, risk=risk)

    def _on_timeout(self, tool: Tool) -> None:
        """Fail safe: a timed-out actuation call stops the base immediately."""
        if tool.risk is not RiskClass.ACTUATION:
            return
        motion = self._context.motion
        if motion is None:
            return
        try:
            motion.stop()
            self._log.error('motion_stop_on_timeout tool=%r', tool.name)
        except Exception as exc:  # noqa: BLE001 - never mask the timeout
            self._log.error('motion_stop_failed tool=%r error=%r', tool.name, exc)

    def estop(self, reason: str = 'registry') -> None:
        """Stop motion through the narrow interface. Always permitted."""
        motion = self._context.motion
        if motion is None:
            return
        try:
            motion.stop()
            self._log.error('estop_stop_issued reason=%r', reason)
        except Exception as exc:  # noqa: BLE001
            self._log.error('estop_stop_failed reason=%r error=%r', reason, exc)

    def close(self) -> None:
        """Stop motion, then release the worker pool. Idempotent."""
        if self._closed:
            return
        self._closed = True
        self.estop('registry close')
        self._executor.shutdown(wait=False)
        self._log.info('registry_closed tools=%d', len(self._entries))

    def __enter__(self) -> 'ToolRegistry':
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ['ToolRegistry', 'ToolResult']
