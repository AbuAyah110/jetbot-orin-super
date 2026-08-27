"""Fail-closed plan gate and calibrated motion for visual approaches.

Cosmos chooses only symbolic steps. Wheel duties are fixed here from the
measured 0.65 chassis calibration. In particular, the inner wheel of an arc is
kept at or above 0.50: lower values can hum without breaking stiction and turn
an intended arc into a pivot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from jetbot_agent.robot_loop.actions import RobotAction, extract_json_object
from jetbot_agent.robot_loop.drive_calibration import (
    DURATION_HARD_MAX,
    SPEED_HARD_MAX,
    load_calibration,
)

PLAN_MAX_STEPS = 3
PLAN_MAX_TICKS = 3
ARC_INNER_FLOOR = 0.50
ARC_INNER_DELTA = 0.15
ARC_OUTER_DELTA = 0.05
ALLOWED_STEPS = frozenset({'arc_left', 'arc_right', 'forward', 'stop'})
ALLOWED_SIDES = frozenset({'left', 'center', 'right'})
BRIEFING_MAX_CHARS = 110

_SIDE_TO_STEP = {'left': 'arc_left', 'center': 'forward', 'right': 'arc_right'}
_SIDE_PHRASE = {'left': 'to my left', 'center': 'ahead', 'right': 'to my right'}

_CALIBRATION = load_calibration()
APPROACH_BASE_DUTY = min(_CALIBRATION.speed, SPEED_HARD_MAX)
APPROACH_DURATION_S = min(_CALIBRATION.duration_s, DURATION_HARD_MAX)


@dataclass(frozen=True)
class PlanStep:
    step: str
    ticks: int


@dataclass(frozen=True)
class ApproachPlan:
    visible: bool = False
    side: str = ''
    goal: str = ''
    steps: tuple[PlanStep, ...] = ()
    reason: str = ''
    raw_ok: bool = False

    @property
    def total_ticks(self) -> int:
        return sum(step.ticks for step in self.steps)


def stopped_plan(reason: str, *, goal: str = '') -> ApproachPlan:
    return ApproachPlan(goal=goal[:80], reason=reason[:160], raw_ok=False)


def normalize_step(name: Any) -> str:
    """Map the model's near-miss step wording onto one allowed symbolic step.

    Cosmos frequently answers with phrasing like ``move toward`` or ``turn
    left`` instead of the schema names. Anything that still does not resolve is
    returned empty so the caller fails closed.
    """
    if not isinstance(name, str):
        return ''
    words = re.sub(r'[^a-z]+', ' ', name.strip().lower())
    if not words:
        return ''
    if 'stop' in words or 'halt' in words or 'wait' in words:
        return 'stop'
    if 'left' in words:
        return 'arc_left'
    if 'right' in words:
        return 'arc_right'
    if re.search(r'\b(?:forward|forwards|ahead|straight|toward|towards|approach|advance)\b', words):
        return 'forward'
    return ''


def parse_approach_plan(payload: str) -> ApproachPlan:
    """Parse a bounded symbolic plan; malformed or unknown steps fail closed."""
    try:
        data = extract_json_object(payload)
    except Exception:
        return stopped_plan('plan_parse_fail')

    visible = data.get('visible')
    if not isinstance(visible, bool):
        return stopped_plan('plan_missing_visible')

    goal = data.get('goal', '')
    goal = goal.strip()[:80] if isinstance(goal, str) else ''
    reason = data.get('reason', '')
    reason = reason.strip()[:160] if isinstance(reason, str) else ''
    side_value = data.get('side', '')
    side = side_value.strip().lower() if isinstance(side_value, str) else ''

    if visible is False:
        return ApproachPlan(
            visible=False,
            goal=goal,
            reason=reason or 'target_not_visible',
            raw_ok=True,
        )
    if side not in ALLOWED_SIDES:
        return stopped_plan('plan_invalid_side', goal=goal)

    raw_steps = data.get('plan')
    if raw_steps is None:
        raw_steps = []
    if not isinstance(raw_steps, list):
        return stopped_plan('plan_missing_steps', goal=goal)

    steps: list[PlanStep] = []
    ticks_left = PLAN_MAX_TICKS
    for raw_step in raw_steps[:PLAN_MAX_STEPS]:
        if not isinstance(raw_step, dict):
            return stopped_plan('plan_invalid_step', goal=goal)
        name = normalize_step(raw_step.get('step'))
        ticks = raw_step.get('ticks')
        if name not in ALLOWED_STEPS or isinstance(ticks, bool) or not isinstance(ticks, int):
            return stopped_plan('plan_invalid_step', goal=goal)
        if ticks < 1 or ticks > PLAN_MAX_TICKS:
            return stopped_plan('plan_invalid_ticks', goal=goal)
        kept_ticks = min(ticks, ticks_left)
        if kept_ticks:
            steps.append(PlanStep(name, kept_ticks))
            ticks_left -= kept_ticks
        if ticks_left == 0:
            break

    # The grounded side, not the step list, is what the model is reliable at.
    # It frequently answers with a correct side and an empty plan, which is
    # still enough to build the same trajectory the re-steer loop would.
    if not steps:
        steps = [PlanStep(_SIDE_TO_STEP[side], PLAN_MAX_TICKS)]
        reason = ' '.join(part for part in ('plan_from_side', reason) if part)[:160]

    # Pixel grounding wins over a contradictory model step. Most importantly,
    # a centered target can never be converted into a left/right turn.
    first = steps[0]
    if first.step != 'stop':
        steps[0] = PlanStep(_SIDE_TO_STEP[side], first.ticks)

    # A single pulse rarely closes the distance, and the model tends to ask for
    # exactly one. Every tick is re-steered from a fresh look and the budget is
    # hard-capped, so spend it unless the plan deliberately ends parked.
    if steps[-1].step != 'stop':
        deficit = PLAN_MAX_TICKS - sum(step.ticks for step in steps)
        if deficit > 0:
            last = steps[-1]
            steps[-1] = PlanStep(last.step, last.ticks + deficit)

    return ApproachPlan(
        visible=True,
        side=side,
        goal=goal,
        steps=tuple(steps),
        reason=reason,
        raw_ok=True,
    )


def _with_article(goal: str) -> str:
    phrase = ' '.join((goal or '').split()) or 'object'
    if re.match(r'^(?:a|an|the)\b', phrase, re.IGNORECASE):
        return phrase
    return ('an ' if phrase[:1].lower() in 'aeiou' else 'a ') + phrase


def plan_briefing(plan: ApproachPlan) -> str:
    """One spoken sentence pair built from the plan fields, never the utterance."""
    goal = _with_article(plan.goal)
    if not plan.raw_ok:
        return "I couldn't plan that move."
    if not plan.visible:
        return "I don't see {0}.".format(goal)

    seen = plan.side if plan.side in _SIDE_PHRASE else 'center'
    sighting = 'I see {0} {1}.'.format(goal, _SIDE_PHRASE[seen])
    moves = [step.step for step in plan.steps if step.step != 'stop']
    if not moves:
        return '{0} I will hold still.'.format(sighting)

    turn = next((move for move in moves if move in ('arc_left', 'arc_right')), '')
    if turn:
        # An arc already advances, so it is described as forward plus a lean.
        intent = "I'll move forward and to the {0}.".format(
            'left' if turn == 'arc_left' else 'right'
        )
    else:
        intent = "I'll move forward."
    return '{0} {1}'.format(sighting, intent)[:BRIEFING_MAX_CHARS]


def step_wheels(step: str) -> tuple[float, float]:
    """Return calibrated differential duties for one symbolic approach step."""
    base = APPROACH_BASE_DUTY
    outer = min(SPEED_HARD_MAX, base + ARC_OUTER_DELTA)
    # Reduce curvature rather than dropping below the measured stiction floor.
    inner = max(ARC_INNER_FLOOR, base - ARC_INNER_DELTA)
    if inner > outer:
        inner = outer
    if step == 'forward':
        return base, base
    if step == 'arc_left':
        return inner, outer
    if step == 'arc_right':
        return outer, inner
    return 0.0, 0.0


def step_action(step: str) -> RobotAction:
    """Convert a symbolic step to bounded unicycle values, never model PWM."""
    left, right = step_wheels(step)
    if step == 'stop' or (left == 0.0 and right == 0.0):
        return RobotAction(kind='stop', reason='approach_stop')
    # Inverse of talk_and_drive.unicycle_wheels (wheel scale 0.4).
    vx = (left + right) / 2.0
    wz = (right - left) / 0.8
    return RobotAction(
        kind='drive',
        vx=vx,
        wz=wz,
        duration_s=APPROACH_DURATION_S,
        reason='approach_{0}'.format(step),
    )


def verification_is_safe(previous_side: str, visible: Any, side: Any) -> bool:
    """True only while the target is still grounded on a known side.

    A side flip is *not* a failure: the user may move the object, so the caller
    re-steers to the new side. Only loss or ambiguity aborts the plan.
    """
    del previous_side
    return visible is True and side in ALLOWED_SIDES


def expand_ticks(steps: tuple[PlanStep, ...]) -> tuple[str, ...]:
    """Flatten steps into one step name per pulse, capped at the tick budget."""
    ticks: list[str] = []
    for step in steps:
        if step.step == 'stop':
            break
        for _ in range(step.ticks):
            if len(ticks) >= PLAN_MAX_TICKS:
                return tuple(ticks)
            ticks.append(step.step)
    return tuple(ticks)


def resteer_remaining(side: str, remaining_ticks: int, previous_side: str = '') -> tuple[str, ...]:
    """Rewrite the unspent ticks from the newest observation of the target.

    The original parked plan is advisory only. Whatever the latest look reports
    wins, so a target that drifts to center finishes straight instead of
    continuing to arc.

    A direct left/right flip is damped to one straight tick instead of an
    immediate counter-arc: the side estimate is noisy enough to flip between two
    frames taken without moving, and a contradiction means the target is near
    the middle. The next look re-confirms and arcs then if the flip was real.
    """
    if side not in ALLOWED_SIDES or remaining_ticks <= 0:
        return ()
    budget = min(int(remaining_ticks), PLAN_MAX_TICKS)
    flipped = {previous_side, side} == {'left', 'right'}
    step = 'forward' if flipped else _SIDE_TO_STEP[side]
    return tuple([step] * budget)
