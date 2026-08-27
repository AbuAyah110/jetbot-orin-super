from __future__ import annotations

import json

import pytest

from jetbot_agent.robot_loop.approach_plan import (
    APPROACH_BASE_DUTY,
    ARC_INNER_FLOOR,
    BRIEFING_MAX_CHARS,
    expand_ticks,
    normalize_step,
    parse_approach_plan,
    plan_briefing,
    resteer_remaining,
    step_action,
    step_wheels,
    verification_is_safe,
)


def _plan(side='left', plan=None, visible=True):
    return json.dumps(
        {
            'visible': visible,
            'side': side,
            'goal': 'red object',
            'plan': plan if plan is not None else [{'step': 'arc_left', 'ticks': 1}],
            'reason': '',
        }
    )


def test_plan_parse_and_total_tick_cap():
    plan = parse_approach_plan(
        _plan(
            plan=[
                {'step': 'arc_left', 'ticks': 2},
                {'step': 'forward', 'ticks': 3},
                {'step': 'stop', 'ticks': 1},
                {'step': 'arc_right', 'ticks': 1},
            ]
        )
    )
    assert plan.raw_ok is True
    assert len(plan.steps) == 2
    assert plan.total_ticks == 3
    assert [(step.step, step.ticks) for step in plan.steps] == [
        ('arc_left', 2),
        ('forward', 1),
    ]


def test_arc_mapping_advances_and_respects_stiction_floor():
    left_arc = step_wheels('arc_left')
    right_arc = step_wheels('arc_right')
    assert left_arc == pytest.approx((0.50, 0.70))
    assert right_arc == pytest.approx((0.70, 0.50))
    assert min(left_arc) >= ARC_INNER_FLOOR
    assert min(right_arc) >= ARC_INNER_FLOOR
    assert max(left_arc + right_arc) <= 0.70
    assert all(duty > 0.0 for duty in left_arc + right_arc)


def test_center_is_forced_to_forward_and_never_turns():
    plan = parse_approach_plan(_plan(side='center', plan=[{'step': 'arc_right', 'ticks': 1}]))
    assert plan.steps[0].step == 'forward'
    assert step_wheels(plan.steps[0].step) == pytest.approx(
        (APPROACH_BASE_DUTY, APPROACH_BASE_DUTY)
    )
    action = step_action(plan.steps[0].step)
    assert action.vx == pytest.approx(0.65)
    assert action.wz == 0.0


def test_visible_false_has_no_drive_plan():
    plan = parse_approach_plan(_plan(side='none', plan=[], visible=False))
    assert plan.raw_ok is True
    assert plan.visible is False
    assert plan.steps == ()
    assert plan.total_ticks == 0


@pytest.mark.parametrize(
    'payload',
    [
        'not json',
        '{"side":"left","plan":[]}',
        _plan(plan=[{'step': 'spin', 'ticks': 1}]),
        _plan(plan=[{'step': 'arc_left', 'ticks': 0}]),
    ],
)
def test_invalid_plan_fails_closed(payload):
    plan = parse_approach_plan(payload)
    assert plan.raw_ok is False
    assert plan.steps == ()
    assert plan.total_ticks == 0


def test_single_tick_plan_is_padded_to_the_bounded_budget():
    plan = parse_approach_plan(_plan(side='left', plan=[{'step': 'arc_left', 'ticks': 1}]))
    assert plan.total_ticks == 3
    assert expand_ticks(plan.steps) == ('arc_left', 'arc_left', 'arc_left')

    parked = parse_approach_plan(
        _plan(side='center', plan=[{'step': 'forward', 'ticks': 1}, {'step': 'stop', 'ticks': 1}])
    )
    assert parked.total_ticks == 2
    assert expand_ticks(parked.steps) == ('forward',)


def test_arc_step_never_pivots_or_stalls_a_wheel():
    for step in ('arc_left', 'arc_right'):
        left, right = step_wheels(step)
        assert left > 0.0 and right > 0.0
        assert min(left, right) >= ARC_INNER_FLOOR
        assert max(left, right) > min(left, right)
        action = step_action(step)
        assert action.kind == 'drive'
        assert action.vx > 0.0
        assert action.duration_s == pytest.approx(1.2)


def test_model_step_synonyms_normalize_but_unknown_words_do_not():
    assert normalize_step('move toward') == 'forward'
    assert normalize_step('go straight ahead') == 'forward'
    assert normalize_step('turn left') == 'arc_left'
    assert normalize_step('slight right') == 'arc_right'
    assert normalize_step('halt') == 'stop'
    assert normalize_step('spin') == ''
    assert normalize_step(None) == ''
    plan = parse_approach_plan(_plan(side='center', plan=[{'step': 'move toward', 'ticks': 2}]))
    assert plan.raw_ok is True
    assert plan.steps[0].step == 'forward'


def test_resteer_rewrites_remaining_ticks_from_newest_side():
    plan = parse_approach_plan(_plan(side='left', plan=[{'step': 'arc_left', 'ticks': 3}]))
    pending = expand_ticks(plan.steps)
    assert pending == ('arc_left', 'arc_left', 'arc_left')

    # Tick 1 ran as an arc; the next look reports the target is now centered.
    pending = pending[1:]
    assert verification_is_safe('left', True, 'center') is True
    pending = resteer_remaining('center', len(pending))
    assert pending == ('forward', 'forward')
    assert step_wheels(pending[0]) == pytest.approx((0.65, 0.65))

    assert resteer_remaining('right', 1) == ('arc_right',)
    assert resteer_remaining('left', 5) == ('arc_left',) * 3


def test_noisy_left_right_flip_is_damped_to_one_straight_tick():
    # Never aborts and never counter-pivots; the next look re-confirms.
    assert resteer_remaining('right', 2, 'left') == ('forward', 'forward')
    assert resteer_remaining('left', 2, 'right') == ('forward', 'forward')
    assert resteer_remaining('right', 2, 'center') == ('arc_right',) * 2
    assert resteer_remaining('right', 2, 'right') == ('arc_right',) * 2


def test_lost_or_ambiguous_target_aborts():
    assert verification_is_safe('left', False, 'none') is False
    assert verification_is_safe('left', True, '') is False
    assert resteer_remaining('none', 2) == ()
    assert resteer_remaining('left', 0) == ()


def test_briefing_describes_sighting_and_intent_without_echoing_speech():
    red_left = parse_approach_plan(_plan(side='left', plan=[{'step': 'arc_left', 'ticks': 2}]))
    assert plan_briefing(red_left) == (
        "I see a red object to my left. I'll move forward and to the left."
    )

    blue_center = json.dumps(
        {
            'visible': True,
            'side': 'center',
            'goal': 'blue object',
            'plan': [{'step': 'forward', 'ticks': 1}],
            'reason': '',
        }
    )
    assert plan_briefing(parse_approach_plan(blue_center)) == (
        "I see a blue object ahead. I'll move forward."
    )

    absent = parse_approach_plan(_plan(side='none', plan=[], visible=False))
    assert plan_briefing(absent) == "I don't see a red object."

    assert plan_briefing(parse_approach_plan('not json')) == "I couldn't plan that move."
    for plan in (red_left, absent):
        assert len(plan_briefing(plan)) <= BRIEFING_MAX_CHARS
