from __future__ import annotations

import json

from jetbot_agent.agent.cosmos_tool_agent import (
    AGENT_FALLBACK,
    AGENT_MAX_TOKENS,
    CosmosToolAgent,
    compact_tool_catalog,
    parse_agent_decision,
)
from jetbot_agent.agent.tools import (
    Capability,
    MockActuationTool,
    MockMotionInterface,
    MockStopTool,
    MockTool,
    ToolContext,
    ToolRegistry,
)


class ScriptedRuntime:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def _read_registry():
    registry = ToolRegistry(ToolContext(), capabilities=(Capability.READ,))
    registry.register(MockTool(), allow=True)
    return registry


def _motion_registry():
    motion = MockMotionInterface()
    registry = ToolRegistry(
        ToolContext(motion=motion),
        capabilities=(Capability.READ, Capability.ACTUATE),
        operator_ack_actuation=True,
    )
    registry.register(MockActuationTool(), allow=True)
    registry.register(MockStopTool(), allow=True)
    return registry, motion


def test_direct_natural_answer_needs_no_tool():
    registry = _read_registry()
    runtime = ScriptedRuntime(
        ['{"kind":"say","say":"Saturn has beautiful rings."}']
    )
    turn = CosmosToolAgent(runtime, registry).run(
        "Why is Saturn interesting?",
        history="user: We were discussing planets.",
    )
    assert turn.say == "Saturn has beautiful rings."
    assert turn.calls == ()
    assert turn.failed is False
    assert runtime.calls[0]["max_tokens"] == AGENT_MAX_TOKENS
    assert "Why is Saturn interesting?" in runtime.calls[0]["user_text"]
    registry.close()


def test_model_calls_registry_tool_then_speaks_from_observation():
    registry = _read_registry()
    runtime = ScriptedRuntime(
        [
            '{"kind":"tool","name":"mock_echo","args":{"message":"hello","repeat":2}}',
            '{"kind":"say","say":"The tool returned hello hello."}',
        ]
    )
    turn = CosmosToolAgent(runtime, registry).run("Echo hello twice")
    assert turn.say == "The tool returned hello hello."
    assert len(turn.calls) == 1
    assert turn.calls[0].name == "mock_echo"
    assert turn.calls[0].result.ok is True
    assert turn.calls[0].result.value["echo"] == "hello hello"
    second_prompt = runtime.calls[1]["user_text"]
    assert '"ok":true' in second_prompt
    assert "hello hello" in second_prompt
    registry.close()


def test_abstract_motion_can_select_bounded_registry_tool_in_mock_only():
    registry, motion = _motion_registry()
    runtime = ScriptedRuntime(
        [
            '{"kind":"tool","name":"mock_drive","args":'
            '{"linear":0.1,"angular":0,"duration_sec":0.5}}',
            '{"kind":"say","say":"I moved forward briefly."}',
        ]
    )
    turn = CosmosToolAgent(runtime, registry).run(
        "Move a little closer so you can get a better look."
    )
    assert turn.failed is False
    assert turn.calls[0].result.ok is True
    assert motion.commands == ("drive",)
    assert motion.calls[0][1]["duration_sec"] == 0.5
    registry.close()


def test_unknown_tool_fails_closed_and_stops_mock_motion():
    registry, motion = _motion_registry()
    runtime = ScriptedRuntime(
        ['{"kind":"tool","name":"set_pwm","args":{"left":1,"right":1}}']
    )
    turn = CosmosToolAgent(runtime, registry).run("Go faster")
    assert turn.say == AGENT_FALLBACK
    assert turn.failed is True
    assert turn.reason == "decision_error:ValueError"
    assert motion.commands == ("stop",)
    registry.close()


def test_extra_keys_and_non_object_arguments_are_rejected():
    allowed = {"mock_echo"}
    for raw in (
        '{"kind":"say","say":"hello","reason":"hidden"}',
        '{"kind":"tool","name":"mock_echo","args":{},"say":"also"}',
        '{"kind":"tool","name":"mock_echo","args":[]}',
    ):
        try:
            parse_agent_decision(raw, allowed)
        except ValueError:
            pass
        else:
            raise AssertionError(raw)


def test_markdown_wrapper_uses_first_json_object():
    decision = parse_agent_decision(
        '```json\n{"kind":"say","say":"Hello there."}\n```\nextra',
        set(),
    )
    assert decision.say == "Hello there."


def test_catalog_is_generated_only_from_invocable_tools():
    registry = ToolRegistry(ToolContext(), capabilities=(Capability.READ,))
    registry.register(MockTool(), allow=True)
    denied = MockStopTool()
    registry.register(denied, allow=False)
    catalog = json.loads(compact_tool_catalog(registry))
    assert [tool["name"] for tool in catalog] == ["mock_echo"]
    assert catalog[0]["args"]["message"]["required"] is True
    assert "pwm" not in json.dumps(catalog).lower()
    registry.close()


def test_tool_validation_error_becomes_observation_then_explanation():
    registry = _read_registry()
    runtime = ScriptedRuntime(
        [
            '{"kind":"tool","name":"mock_echo","args":{"message":"x","repeat":99}}',
            '{"kind":"say","say":"That request exceeded the tool limit."}',
        ]
    )
    turn = CosmosToolAgent(runtime, registry).run("Repeat x many times")
    assert turn.calls[0].result.ok is False
    assert turn.say == "That request exceeded the tool limit."
    assert '"ok":false' in runtime.calls[1]["user_text"]
    registry.close()


def test_repeated_tool_requests_hit_step_limit_and_estop():
    registry, motion = _motion_registry()
    raw = '{"kind":"tool","name":"mock_stop","args":{}}'
    runtime = ScriptedRuntime([raw, raw, raw])
    turn = CosmosToolAgent(runtime, registry, max_steps=2).run("Keep stopping")
    assert turn.failed is True
    assert turn.reason == "step_limit"
    assert len(turn.calls) == 2
    # Two tool stops plus the registry fail-safe stop.
    assert motion.commands == ("stop", "stop", "stop")
    registry.close()
