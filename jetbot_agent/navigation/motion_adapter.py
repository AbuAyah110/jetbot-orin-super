"""The concrete ``MotionInterface`` adapter — Stage H / I5.

This module is the *only* place where the agent's high-level motion vocabulary
becomes a velocity command. It sits deliberately **outside**
``jetbot_agent/agent/tools/``:
:func:`jetbot_agent.agent.tools.base.assert_narrow_motion` refuses any object
that exposes wheel/PWM/I2C/watchdog attributes, that has ``MotorDriver`` /
``DiffDriveController`` / ``MotorController`` in its MRO, or that originates
from ``jetbot_control.motors`` / ``jetbot_agent.hardware`` / ``smbus`` /
``Jetson.*`` — and the tool package additionally may not import its way down to
a velocity backend at all. An adapter that lived inside the tool package could
therefore never be wired to anything real.

Layering::

    navigation tool (jetbot_agent/agent/tools/navigation_tools.py)
       │  drive / rotate / stop / status / limits   <- the whole vocabulary
       ▼
    BoundedMotionAdapter                            <- THIS module
       │  clamp magnitude, bound duration, arm the cmd_vel watchdog, latch e-stop
       ▼
    CmdVelSink (mock | jetbot_base controller | ROS /cmd_vel)
       ▼
    jetbot_base    velocity limits + cmd_vel watchdog + e-stop
       ▼
    motor backend  (mock during bring-up)

What this adapter guarantees, so the tool layer does not have to:

1. **Magnitude is clamped, not silently dropped.** Every clamp is recorded on
   :attr:`BoundedMotionAdapter.clamps` and logged as ``motion_clamped``.
2. **Duration is bounded.** No call can express indefinite motion: a deadline is
   always set, always positive, and always capped by
   ``MotionLimits.max_duration_sec`` and by :data:`HARD_MAX_DURATION_SEC`.
3. **The command watchdog is honoured.** A command is only sustained while
   :meth:`BoundedMotionAdapter.poll` keeps refreshing it inside
   ``cmd_vel_timeout_sec``. Stop polling and the base stops — which is exactly
   what ``jetbot_base`` does with a stale ``/cmd_vel``.
4. **E-stop latches and refuses.** While :class:`MotionEstop` is active, drive
   and rotate raise :class:`~jetbot_agent.agent.tools.motion.MotionDenied` and
   the sink is halted. Clearing it is an operator act on the latch object; the
   adapter has no ``clear_estop`` attribute, both because that is what
   ``assert_narrow_motion`` requires and because a tool must not be able to
   un-stop the robot.
5. **Stop is never refused.** :meth:`BoundedMotionAdapter.stop` works while
   e-stopped, while clamped, and while a sink is failing.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple

from jetbot_agent.agent.tools.motion import (
    MotionDenied,
    MotionLimits,
    MotionStatus,
)

from .cmd_vel_sink import CmdVelSink, MockCmdVelSink

LOGGER = logging.getLogger('jetbot_agent.navigation.motion_adapter')

#: Shortest motion a command may request. A zero/negative duration is clamped
#: up to this rather than becoming "until something else stops me".
MIN_DURATION_SEC = 0.05

#: Ceilings the adapter applies to whatever limits it is handed, so a bad or
#: hand-edited ``config/robot.yaml`` still cannot ask for fast or long motion.
HARD_MAX_LINEAR_VELOCITY = 0.5
HARD_MAX_ANGULAR_VELOCITY = 2.0
HARD_MAX_DURATION_SEC = 5.0
HARD_MAX_CMD_VEL_TIMEOUT_SEC = 1.0


@dataclass(frozen=True)
class ClampEvent:
    """One clamped field, kept so callers can report what was reduced."""

    field: str
    requested: float
    applied: float
    limit: float


class MotionEstop:
    """Latching e-stop held below the tool boundary.

    The latch is a separate object on purpose. The tool layer only ever sees the
    adapter, and the adapter intentionally has no way to clear this, so no tool
    call and no model output can un-stop the robot; clearing is an operator act
    on the object the wiring code kept. It latches unconditionally, matching
    ``estop.latch: true`` in ``config/robot.yaml``.

    Hooks let the adapter halt its sink the instant the latch trips, and they
    run fail-safe: a hook that raises is logged and never blocks the stop.
    """

    def __init__(self, *, logger: Optional[logging.Logger] = None) -> None:
        self._active = False
        self._reason = ''
        self._hooks: List[Callable[[str], None]] = []
        self._log = logger or LOGGER

    @property
    def active(self) -> bool:
        return self._active

    @property
    def reason(self) -> str:
        return self._reason

    def add_hook(self, hook: Callable[[str], None]) -> None:
        self._hooks.append(hook)

    def trigger(self, reason: str = 'operator') -> None:
        """Latch the e-stop. Safe to call from any thread or a signal handler."""
        already = self._active
        self._active = True
        if not already:
            self._reason = reason
        self._log.error('motion_estop_triggered reason=%r repeated=%s', reason, already)
        for hook in list(self._hooks):
            try:
                hook(reason)
            except Exception as exc:  # noqa: BLE001 - an e-stop must never fail
                self._log.error('motion_estop_hook_failed error=%r', exc)

    def clear(self) -> None:
        """Explicit operator clear. The only way out of a latched e-stop."""
        if not self._active:
            return
        previous, self._reason = self._reason, ''
        self._active = False
        self._log.warning('motion_estop_cleared previous_reason=%r', previous)


def limits_from_robot_config(
    config: Mapping[str, Any],
    *,
    max_duration_sec: Optional[float] = None,
) -> MotionLimits:
    """Build :class:`MotionLimits` from a parsed ``config/robot.yaml`` mapping."""
    limits = config.get('limits') or {}
    watchdog = config.get('watchdog') or {}
    defaults = MotionLimits()
    return MotionLimits(
        max_linear_velocity=float(limits.get('max_linear_velocity',
                                             defaults.max_linear_velocity)),
        max_angular_velocity=float(limits.get('max_angular_velocity',
                                              defaults.max_angular_velocity)),
        cmd_vel_timeout_sec=float(watchdog.get('cmd_vel_timeout_sec',
                                               defaults.cmd_vel_timeout_sec)),
        max_duration_sec=float(defaults.max_duration_sec
                               if max_duration_sec is None else max_duration_sec),
    )


def load_robot_limits(path: Optional[Any] = None) -> MotionLimits:
    """Read the deterministic limits straight out of ``config/robot.yaml``.

    ``yaml`` is imported lazily so importing this module stays dependency-free.
    """
    import yaml

    if path is None:
        path = Path(__file__).resolve().parents[2] / 'config' / 'robot.yaml'
    with Path(path).open('r', encoding='utf-8') as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, Mapping):
        raise ValueError('robot config must be a mapping')
    return limits_from_robot_config(config)


def bounded_limits(limits: MotionLimits, *, logger: Optional[logging.Logger] = None) -> MotionLimits:
    """Apply the module's hard ceilings to a limits view."""
    log = logger or LOGGER
    bounded = MotionLimits(
        max_linear_velocity=_positive(limits.max_linear_velocity, HARD_MAX_LINEAR_VELOCITY),
        max_angular_velocity=_positive(limits.max_angular_velocity, HARD_MAX_ANGULAR_VELOCITY),
        cmd_vel_timeout_sec=_positive(limits.cmd_vel_timeout_sec, HARD_MAX_CMD_VEL_TIMEOUT_SEC),
        max_duration_sec=_positive(limits.max_duration_sec, HARD_MAX_DURATION_SEC),
    )
    if bounded != limits:
        log.warning('motion_limits_bounded requested=%r applied=%r', limits, bounded)
    return bounded


def _positive(value: float, ceiling: float) -> float:
    number = float(value)
    if number != number or number <= 0.0:  # NaN or non-positive
        return ceiling
    return min(number, ceiling)


class BoundedMotionAdapter:
    """High-level motion for the tool layer, bounded velocity for the base.

    Satisfies :class:`~jetbot_agent.agent.tools.motion.MotionInterface` and
    nothing wider, so :func:`assert_narrow_motion` accepts it. Note what is
    absent and must stay absent: no wheel velocities, no duty cycle, no bus, no
    driver handle, no watchdog setter, no e-stop clear.
    """

    def __init__(
        self,
        sink: Optional[CmdVelSink] = None,
        limits: Optional[MotionLimits] = None,
        *,
        estop: Optional[MotionEstop] = None,
        clock: Callable[[], float] = time.monotonic,
        logger: Optional[logging.Logger] = None,
        refresh_fraction: float = 0.5,
    ) -> None:
        self._log = logger or LOGGER
        self._limits = bounded_limits(limits or MotionLimits(), logger=self._log)
        self._sink: CmdVelSink = sink if sink is not None else MockCmdVelSink(
            cmd_vel_timeout_sec=self._limits.cmd_vel_timeout_sec
        )
        self._estop = estop if estop is not None else MotionEstop(logger=self._log)
        self._estop.add_hook(self._halt_for_estop)
        self._clock = clock
        self._refresh_interval = max(
            0.01, self._limits.cmd_vel_timeout_sec * max(0.1, min(0.9, refresh_fraction))
        )
        self._clamps: List[ClampEvent] = []
        self._linear = 0.0
        self._angular = 0.0
        self._deadline = 0.0
        self._last_publish = 0.0
        self._active = False
        self._reason = 'idle'
        self._log.info('motion_adapter_ready backend=%s limits=%r refresh=%.3f',
                       self._sink_name, self._limits, self._refresh_interval)

    # -------------------------------------------------------- MotionInterface

    def drive(self, linear: float, angular: float, duration_sec: float) -> MotionStatus:
        """Bounded body twist. Clamped, deadlined, and watchdogged below here."""
        return self._command('drive', linear, angular, duration_sec)

    def rotate(self, angular: float, duration_sec: float) -> MotionStatus:
        """Bounded in-place rotation; linear velocity is forced to zero."""
        return self._command('rotate', 0.0, angular, duration_sec)

    def stop(self) -> MotionStatus:
        """Immediate stop. Always permitted, never raises, never gated."""
        self._release('stop_requested')
        return self.status()

    def status(self) -> MotionStatus:
        """Read-only view. Reflects watchdog staleness without mutating state."""
        now = self._clock()
        stale = self._active and (now - self._last_publish) > self._limits.cmd_vel_timeout_sec
        expired = self._active and now >= self._deadline
        live = self._active and not stale and not expired
        if not self._active:
            reason = self._reason
        elif stale:
            reason = 'watchdog_expired'
        elif expired:
            reason = 'duration_elapsed'
        else:
            reason = 'moving'
        return MotionStatus(
            moving=bool(live and (self._linear or self._angular)),
            estop_active=self._estop.active,
            last_linear=self._linear if live else 0.0,
            last_angular=self._angular if live else 0.0,
            watchdog_armed=live,
            backend=self._sink_name,
            detail=f'BoundedMotionAdapter:{reason}',
        )

    def limits(self) -> MotionLimits:
        """Read-only limits. Frozen dataclass, so there is no setter to find."""
        return self._limits

    # -------------------------------------------------------------- lifecycle

    def poll(self, now: Optional[float] = None) -> MotionStatus:
        """Advance the command watchdog. Call this from the control loop.

        Sustaining motion requires refreshing the command inside
        ``cmd_vel_timeout_sec``; a late or absent poll stops the base instead of
        resuming it, which mirrors ``jetbot_base`` reacting to a stale
        ``/cmd_vel``.
        """
        stamp = self._clock() if now is None else float(now)
        if self._estop.active:
            if self._active:
                self._release('estop_active')
            return self.status()
        if not self._active:
            self._sink.tick(stamp)
            return self.status()
        if stamp >= self._deadline:
            self._release('duration_elapsed')
            return self.status()
        elapsed = stamp - self._last_publish
        if elapsed > self._limits.cmd_vel_timeout_sec:
            self._release('watchdog_expired')
            return self.status()
        if elapsed >= self._refresh_interval:
            self._publish(self._linear, self._angular, stamp)
        self._sink.tick(stamp)
        return self.status()

    @property
    def clamps(self) -> Tuple[ClampEvent, ...]:
        """Every clamp this adapter has applied, newest last."""
        return tuple(self._clamps)

    @property
    def estop_active(self) -> bool:
        return self._estop.active

    def describe(self) -> str:
        return f'BoundedMotionAdapter(sink={self._sink.describe()}, limits={self._limits!r})'

    # --------------------------------------------------------------- internals

    @property
    def _sink_name(self) -> str:
        return str(getattr(self._sink, 'name', 'unknown'))

    def _command(self, verb: str, linear: float, angular: float,
                 duration_sec: float) -> MotionStatus:
        if self._estop.active:
            self._release('estop_active')
            raise MotionDenied(
                f'e-stop latched ({self._estop.reason!r}); {verb} refused below the '
                'tool boundary until an operator clears it'
            )
        applied_linear = self._clamp('linear', linear, self._limits.max_linear_velocity)
        applied_angular = self._clamp('angular', angular, self._limits.max_angular_velocity)
        duration = self._clamp_duration(duration_sec)
        if not (applied_linear or applied_angular):
            self._log.info('motion_zero_twist verb=%s treated_as=stop', verb)
            return self.stop()

        now = self._clock()
        self._linear = applied_linear
        self._angular = applied_angular
        self._deadline = now + duration
        self._active = True
        self._reason = 'moving'
        self._publish(applied_linear, applied_angular, now)
        self._log.info('motion_command verb=%s linear=%.4f angular=%.4f duration=%.3f backend=%s',
                       verb, applied_linear, applied_angular, duration, self._sink_name)
        return self.status()

    def _clamp(self, field: str, value: float, magnitude: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = 0.0
        if math.isnan(number):
            applied = 0.0
        else:
            applied = max(-magnitude, min(magnitude, number))
        if applied != number:
            event = ClampEvent(field=field, requested=number, applied=applied, limit=magnitude)
            self._clamps.append(event)
            self._log.warning('motion_clamped field=%s requested=%r applied=%.4f limit=%.4f',
                              field, number, applied, magnitude)
        return applied

    def _clamp_duration(self, duration_sec: float) -> float:
        try:
            number = float(duration_sec)
        except (TypeError, ValueError):
            number = MIN_DURATION_SEC
        ceiling = self._limits.max_duration_sec
        if math.isnan(number):
            applied = MIN_DURATION_SEC
        else:
            applied = max(MIN_DURATION_SEC, min(ceiling, number))
        if applied != number:
            event = ClampEvent(field='duration_sec', requested=number, applied=applied,
                               limit=ceiling)
            self._clamps.append(event)
            self._log.warning('motion_clamped field=duration_sec requested=%r applied=%.4f '
                              'limit=%.4f', number, applied, ceiling)
        return applied

    def _publish(self, linear: float, angular: float, now: float) -> None:
        self._last_publish = now
        self._sink.publish(linear, angular, now)

    def _release(self, reason: str) -> None:
        """Zero the command and halt the sink. Must never raise."""
        was_active = self._active
        self._linear = 0.0
        self._angular = 0.0
        self._active = False
        self._deadline = 0.0
        self._reason = reason
        try:
            self._sink.halt(reason)
        except Exception as exc:  # noqa: BLE001 - a stop must never fail
            self._log.error('motion_halt_failed reason=%r error=%r', reason, exc)
        if was_active or reason == 'stop_requested':
            self._log.info('motion_halted reason=%s backend=%s', reason, self._sink_name)

    def _halt_for_estop(self, reason: str) -> None:
        self._release(f'estop:{reason}')


__all__ = [
    'BoundedMotionAdapter',
    'ClampEvent',
    'HARD_MAX_ANGULAR_VELOCITY',
    'HARD_MAX_CMD_VEL_TIMEOUT_SEC',
    'HARD_MAX_DURATION_SEC',
    'HARD_MAX_LINEAR_VELOCITY',
    'MIN_DURATION_SEC',
    'MotionEstop',
    'bounded_limits',
    'limits_from_robot_config',
    'load_robot_limits',
]
