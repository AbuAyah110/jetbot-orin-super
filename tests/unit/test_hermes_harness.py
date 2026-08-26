from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))

from jetbot_agent.agent.hermes_harness import (
    MAX_BLOCK_SEC,
    Action,
    ActionOutcome,
    BrainContractError,
    Decision,
    DecisionKind,
    EstopLatched,
    FakeBrain,
    HarnessConfig,
    HarnessError,
    HermesHarness,
    IllegalTransition,
    NullActionSink,
    Observation,
    RecordingSpeechSink,
    State,
    is_legal,
)


def _harness(script=(), **kwargs):
    config = kwargs.pop('config', None) or HarnessConfig(**kwargs.pop('config_kwargs', {}))
    loop = kwargs.pop('loop', False)
    brain = kwargs.pop('brain', None) or FakeBrain(script, loop=loop)
    sink = kwargs.pop('action_sink', None) or NullActionSink()
    speech = kwargs.pop('speech_sink', None) or RecordingSpeechSink()
    harness = HermesHarness(brain, config, action_sink=sink, speech_sink=speech, **kwargs)
    return harness, brain, sink, speech


def _states(harness):
    return [
        event.fields['to']
        for event in harness.events
        if event.event == 'state_transition'
    ]


# --------------------------------------------------------------- transitions


def test_scripted_turn_walks_legal_states():
    script = [
        Decision(DecisionKind.ACT, action=Action('inspect', {'target': 'door'})),
        Decision(DecisionKind.SPEAK, text='I see a door.'),
        Decision(DecisionKind.DONE),
    ]
    harness, brain, sink, speech = _harness(script)

    result = harness.run_turn('what do you see?')

    assert result.stop_reason == 'done'
    assert result.steps == 3
    assert result.final_state is State.IDLE
    assert harness.state is State.IDLE
    assert speech.said == ['I see a door.']
    assert [a.name for a in result.actions] == ['inspect']
    assert brain.call_count == 3
    visited = _states(harness)
    for expected in ('PERCEIVING', 'DELIBERATING', 'ACTING', 'SPEAKING', 'IDLE'):
        assert expected in visited


def test_i1_gate_idle_think_act_stop():
    """Issue #20 gate: scripted prompt steps through states and reaches stop."""
    script = [
        Decision(DecisionKind.ACT, action=Action('inspect')),
        Decision(DecisionKind.DONE),
    ]
    harness, _, sink, _ = _harness(script)
    assert harness.state is State.IDLE

    result = harness.run_turn('go look')

    assert result.stop_reason == 'done'
    # I1 executes nothing: the action is recorded, never run.
    assert [a.name for a in sink.actions] == ['inspect']
    assert result.outcomes[0].executed is False

    harness.shutdown()
    assert harness.state is State.SHUTDOWN


@pytest.mark.parametrize('source,target', [
    (State.IDLE, State.ACTING),
    (State.IDLE, State.SPEAKING),
    (State.PERCEIVING, State.ACTING),
    (State.SPEAKING, State.ACTING),
    (State.ERROR, State.DELIBERATING),
    (State.SHUTDOWN, State.IDLE),
    (State.SHUTDOWN, State.PERCEIVING),
    (State.ESTOP, State.IDLE),
])
def test_illegal_transitions_are_rejected(source, target):
    harness, _, _, _ = _harness()
    harness._state = source  # test-only: place the machine in a state directly

    assert not is_legal(source, target)
    with pytest.raises(IllegalTransition):
        harness.transition(target)
    assert harness.state is source


@pytest.mark.parametrize('source', list(State))
def test_estop_is_never_a_normal_transition(source):
    """ESTOP has no table edge; trigger_estop() is the only way in."""
    harness, _, _, _ = _harness()
    harness._state = source
    assert not is_legal(source, State.ESTOP)
    with pytest.raises(IllegalTransition):
        harness.transition(State.ESTOP)


def test_legal_transition_table_shape():
    assert is_legal(State.IDLE, State.PERCEIVING)
    assert is_legal(State.DELIBERATING, State.ACTING)
    assert is_legal(State.ACTING, State.DELIBERATING)
    assert is_legal(State.ERROR, State.IDLE)
    assert is_legal(State.ESTOP, State.SHUTDOWN)
    assert not is_legal(State.SHUTDOWN, State.SHUTDOWN)


# --------------------------------------------------------------------- estop


@pytest.mark.parametrize('source', list(State))
def test_estop_reachable_from_every_state(source):
    harness, _, _, _ = _harness()
    harness._state = source

    harness.trigger_estop(f'from {source.value}')

    assert harness.state is State.ESTOP
    assert harness.estop_active is True
    assert harness.estop_reason == f'from {source.value}'


def test_estop_latches_and_requires_explicit_clear():
    harness, _, _, _ = _harness([Decision(DecisionKind.DONE)])
    harness.trigger_estop('operator button')

    with pytest.raises(EstopLatched):
        harness.run_turn('move')
    assert harness.state is State.ESTOP
    assert harness.estop_active is True

    harness.clear_estop()
    assert harness.estop_active is False
    assert harness.state is State.IDLE
    assert harness.run_turn('move').stop_reason == 'done'


def test_estop_trigger_is_idempotent_and_keeps_first_reason():
    harness, _, _, _ = _harness()
    harness.trigger_estop('first')
    harness.trigger_estop('second')
    assert harness.estop_reason == 'first'
    assert harness.state is State.ESTOP
    repeated = [e for e in harness.events if e.event == 'estop_triggered' and e.fields['repeated']]
    assert len(repeated) == 1


def test_estop_mid_turn_stops_the_loop():
    harness = None

    class EstoppingBrain:
        def __init__(self):
            self.calls = 0

        def decide(self, context):
            self.calls += 1
            if self.calls == 2:
                harness.trigger_estop('mid-turn')
            return Decision(DecisionKind.SPEAK, text=f'step {self.calls}')

    brain = EstoppingBrain()
    harness = HermesHarness(brain, HarnessConfig(max_steps=10))

    result = harness.run_turn('talk')

    assert result.stop_reason == 'estop'
    assert harness.state is State.ESTOP
    assert brain.calls == 2  # loop stopped instead of taking a third step
    assert result.final_state is State.ESTOP


def test_estop_hooks_run_and_hook_failure_does_not_block_estop():
    harness, _, _, _ = _harness()
    seen = []

    harness.add_estop_hook(lambda reason: (_ for _ in ()).throw(RuntimeError('bad hook')))
    harness.add_estop_hook(seen.append)

    harness.trigger_estop('hook test')

    assert harness.state is State.ESTOP
    assert seen == ['hook test']
    assert any(e.event == 'estop_hook_failed' for e in harness.events)


def test_estop_from_another_thread_is_observed_promptly():
    harness = None

    class SlowishBrain:
        def __init__(self):
            self.calls = 0

        def decide(self, context):
            self.calls += 1
            return Decision(DecisionKind.WAIT)

    brain = SlowishBrain()
    harness = HermesHarness(brain, HarnessConfig(max_steps=200, idle_wait_sec=0.01))
    timer = threading.Timer(0.05, harness.trigger_estop, args=('other thread',))
    timer.start()
    started = time.monotonic()
    result = harness.run_turn('spin')
    elapsed = time.monotonic() - started
    timer.join()

    assert result.stop_reason == 'estop'
    assert elapsed < 2.0
    assert MAX_BLOCK_SEC <= 0.05


def test_non_latching_config_auto_clears():
    config = HarnessConfig(estop_latch=False)
    harness = HermesHarness(FakeBrain([Decision(DecisionKind.DONE)]), config)
    harness.trigger_estop('transient')

    result = harness.run_turn('go')

    assert result.stop_reason == 'done'
    assert harness.estop_active is False
    assert any(e.event == 'estop_auto_cleared' for e in harness.events)


def test_config_from_robot_yaml_mapping():
    config = HarnessConfig.from_robot_config({'estop': {'latch': True}}, max_steps=3)
    assert config.estop_latch is True
    assert config.max_steps == 3
    assert HarnessConfig.from_robot_config({}).estop_latch is True


def test_repo_robot_yaml_latches():
    yaml = pytest.importorskip('yaml')
    data = yaml.safe_load((ROOT / 'config' / 'robot.yaml').read_text(encoding='utf-8'))
    assert HarnessConfig.from_robot_config(data).estop_latch is True


# ----------------------------------------------------------------- max steps


def test_max_steps_guard_ends_turn_in_error():
    script = [Decision(DecisionKind.SPEAK, text='again')]
    harness, brain, _, _ = _harness(script, loop=True, config=HarnessConfig(max_steps=4))

    result = harness.run_turn('never finish')

    assert result.stop_reason == 'max_steps'
    assert result.steps == 4
    assert harness.state is State.ERROR
    assert brain.call_count == 4
    assert any(e.event == 'max_steps_exceeded' for e in harness.events)

    with pytest.raises(HarnessError):
        harness.run_turn('again')
    harness.recover()
    assert harness.state is State.IDLE


def test_wait_decisions_still_count_against_max_steps():
    harness, _, _, _ = _harness([Decision(DecisionKind.WAIT)], loop=True,
                                config=HarnessConfig(max_steps=3))
    result = harness.run_turn('wait forever')
    assert result.stop_reason == 'max_steps'
    assert result.steps == 3


def test_recover_requires_error_state():
    harness, _, _, _ = _harness()
    with pytest.raises(HarnessError):
        harness.recover()


def test_recover_blocked_while_estop_latched():
    harness, _, _, _ = _harness([Decision(DecisionKind.SPEAK, text='x')], loop=True,
                                config=HarnessConfig(max_steps=1))
    harness.run_turn('x')
    assert harness.state is State.ERROR
    harness.trigger_estop('while in error')
    with pytest.raises(EstopLatched):
        harness.recover()


# -------------------------------------------------------------- observations


def test_observations_are_drained_into_the_brain_context():
    seen = {}

    class PeekBrain:
        def decide(self, context):
            seen['kinds'] = [o.kind for o in context.observations]
            seen['step'] = context.step
            seen['remaining'] = context.steps_remaining
            return Decision(DecisionKind.DONE)

    harness = HermesHarness(PeekBrain(), HarnessConfig(max_steps=5))
    harness.submit(Observation('camera', {'frame_id': 1}))
    result = harness.run_turn('describe', observations=[Observation('range', {'m': 0.4})])

    assert result.stop_reason == 'done'
    assert seen['kinds'] == ['camera', 'range', 'prompt']
    assert seen['step'] == 0
    assert seen['remaining'] == 5
    assert harness.pending_observations == 0


def test_wait_picks_up_late_observations():
    harness = None

    class WaitOnceBrain:
        def __init__(self):
            self.calls = 0

        def decide(self, context):
            self.calls += 1
            if self.calls == 1:
                harness.submit(Observation('late', {'n': 1}))
                return Decision(DecisionKind.WAIT)
            assert any(o.kind == 'late' for o in context.observations)
            return Decision(DecisionKind.DONE)

    harness = HermesHarness(WaitOnceBrain(), HarnessConfig(max_steps=5))
    result = harness.run_turn('wait for it')
    assert result.stop_reason == 'done'
    assert result.steps == 2


def test_submit_rejects_non_observations():
    harness, _, _, _ = _harness()
    with pytest.raises(TypeError):
        harness.submit({'kind': 'camera'})


# -------------------------------------------------------------------- faults


def test_bad_brain_return_lands_in_error():
    class BadBrain:
        def decide(self, context):
            return 'just move forward'

    harness = HermesHarness(BadBrain())
    result = harness.run_turn('hello')

    assert result.stop_reason == 'error'
    assert harness.state is State.ERROR
    assert 'BrainContractError' in result.error
    assert BrainContractError is not None


def test_action_sink_fault_lands_in_error():
    class BoomSink:
        def dispatch(self, action):
            raise RuntimeError('sink exploded')

    harness = HermesHarness(
        FakeBrain([Decision(DecisionKind.ACT, action=Action('inspect'))]),
        action_sink=BoomSink(),
    )
    result = harness.run_turn('go')
    assert result.stop_reason == 'error'
    assert harness.state is State.ERROR


def test_action_sink_must_return_an_outcome():
    class WrongSink:
        def dispatch(self, action):
            return 'ok'

    harness = HermesHarness(
        FakeBrain([Decision(DecisionKind.ACT, action=Action('inspect'))]),
        action_sink=WrongSink(),
    )
    assert harness.run_turn('go').stop_reason == 'error'


def test_decision_contract_is_enforced():
    with pytest.raises(ValueError):
        Decision(DecisionKind.ACT)
    with pytest.raises(ValueError):
        Decision(DecisionKind.SPEAK)
    with pytest.raises(ValueError):
        Action('')
    assert Decision(DecisionKind.DONE).action is None
    assert ActionOutcome(executed=False).detail == ''


def test_harness_requires_a_brain_like_object():
    with pytest.raises(TypeError):
        HermesHarness(object())


# ----------------------------------------------------------------- lifecycle


def test_shutdown_is_terminal_and_idempotent():
    harness, _, _, _ = _harness([Decision(DecisionKind.DONE)])
    harness.submit(Observation('camera', {'frame_id': 2}))
    harness.shutdown('test')
    harness.shutdown('test again')

    assert harness.state is State.SHUTDOWN
    with pytest.raises(HarnessError):
        harness.run_turn('after shutdown')
    with pytest.raises(HarnessError):
        harness.submit(Observation('camera'))
    event = [e for e in harness.events if e.event == 'shutdown'][0]
    assert event.fields['dropped_observations'] == 1


def test_context_manager_estops_on_exception():
    harness, _, _, _ = _harness([Decision(DecisionKind.DONE)])
    with pytest.raises(ValueError):
        with harness:
            raise ValueError('boom')
    assert harness.estop_active is True
    assert harness.state is State.SHUTDOWN


def test_turn_requires_idle():
    harness, _, _, _ = _harness()
    harness._state = State.DELIBERATING
    with pytest.raises(HarnessError):
        harness.run_turn('nope')


def test_event_log_is_structured_and_bounded():
    harness, _, _, _ = _harness([Decision(DecisionKind.DONE)],
                                config=HarnessConfig(event_log_limit=5))
    harness.run_turn('hi')
    events = harness.events
    assert len(events) <= 5
    assert all(isinstance(e.fields, dict) for e in events)
    assert {'turn_end'} <= {e.event for e in events}


def test_turn_counter_increments():
    harness, _, _, _ = _harness([Decision(DecisionKind.DONE)], loop=True)
    harness.run_turn('one')
    harness.run_turn('two')
    assert harness.turn_count == 2


# ----------------------------------------------------------- I1 tool-free gate


def test_harness_import_does_not_pull_in_the_tool_layer():
    """Issue #20 gate: the harness reaches stop without importing tool modules."""
    program = (
        'import sys\n'
        'from jetbot_agent.agent.hermes_harness import '
        'HermesHarness, FakeBrain, Decision, DecisionKind, State\n'
        'h = HermesHarness(FakeBrain([Decision(DecisionKind.DONE)]))\n'
        'assert h.run_turn("hello").stop_reason == "done"\n'
        'h.shutdown()\n'
        'assert h.state is State.SHUTDOWN\n'
        'leaked = [m for m in sys.modules if "tools" in m and m.startswith("jetbot_agent")]\n'
        'forbidden = [m for m in sys.modules if m.startswith((\n'
        '    "jetbot_control", "jetbot_base", "jetbot_agent.hardware", "smbus", "Jetson"))]\n'
        'assert not leaked, leaked\n'
        'assert not forbidden, forbidden\n'
        'print("OK")\n'
    )
    proc = subprocess.run(
        [sys.executable, '-c', program],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert 'OK' in proc.stdout
