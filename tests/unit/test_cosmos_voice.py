from __future__ import annotations

import pytest

from jetbot_agent.agent.cosmos_voice import (
    BACKEND_MOCK,
    BACKEND_ROS2,
    CosmosVoiceSession,
    build_voice_backend,
)
from jetbot_agent.navigation import MotionBackendUnavailable


class ScriptedRuntime:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.outputs.pop(0)


def test_mock_is_default_and_actuation_is_denied():
    backend = build_voice_backend()
    assert backend.kind == BACKEND_MOCK
    assert backend.actuation_enabled is False
    assert backend.registry.invocable() == ("nav_status",)

    runtime = ScriptedRuntime(
        ['{"kind":"tool","name":"nav_drive","args":{"distance_m":0.1}}']
    )
    turn = CosmosVoiceSession(runtime, backend).handle_transcript(
        "Move closer so you can see me."
    )
    assert turn.failed is True
    assert turn.reason == "decision_error:ValueError"
    assert backend.sink.last_twist == (0.0, 0.0)
    backend.close()


def test_mock_actuation_requires_explicit_enable_and_is_bounded():
    backend = build_voice_backend(allow_actuation=True)
    assert set(backend.registry.invocable()) == {
        "nav_drive",
        "nav_rotate",
        "nav_status",
        "nav_stop",
    }
    runtime = ScriptedRuntime(
        [
            '{"kind":"tool","name":"nav_rotate","args":{"angle_deg":45}}',
            '{"kind":"say","say":"I turned left briefly."}',
        ]
    )
    session = CosmosVoiceSession(runtime, backend)
    turn = session.handle_transcript("Turn so you can get a better look.")
    assert turn.failed is False
    assert turn.calls[0].name == "nav_rotate"
    assert turn.calls[0].result.ok is True
    assert backend.sink.twists
    linear, angular = backend.sink.twists[0]
    assert linear == 0.0
    assert 0.0 < angular <= backend.motion.limits().max_angular_velocity
    session.close()


def test_voice_session_carries_spoken_history_to_next_turn():
    backend = build_voice_backend()
    runtime = ScriptedRuntime(
        [
            '{"kind":"say","say":"I am JetBot."}',
            '{"kind":"say","say":"We were talking about who I am."}',
        ]
    )
    session = CosmosVoiceSession(runtime, backend)
    first = session.handle_transcript("Who are you?")
    second = session.handle_transcript("What were we talking about?")
    assert first.say == "I am JetBot."
    assert second.say == "We were talking about who I am."
    second_prompt = runtime.calls[1]["user_text"]
    assert "user: Who are you?" in second_prompt
    assert "assistant: I am JetBot." in second_prompt
    session.close()


def test_empty_transcript_skips_cosmos_and_tools():
    backend = build_voice_backend()
    runtime = ScriptedRuntime([])
    turn = CosmosVoiceSession(runtime, backend).handle_transcript("  ")
    assert turn.failed is True
    assert turn.reason == "empty_transcript"
    assert runtime.calls == []
    backend.close()


def test_ros_backend_requires_caller_owned_node():
    with pytest.raises(MotionBackendUnavailable, match="caller-owned"):
        build_voice_backend(BACKEND_ROS2)


class Vector:
    def __init__(self):
        self.x = 0.0
        self.z = 0.0


class FakeTwist:
    def __init__(self):
        self.linear = Vector()
        self.angular = Vector()


class Publisher:
    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(message)


class RosNode:
    def __init__(self):
        self.publisher = Publisher()
        self.calls = []

    def create_publisher(self, message_type, topic, queue_depth):
        self.calls.append((message_type, topic, queue_depth))
        return self.publisher


def test_ros_backend_publishes_only_through_cmd_vel_sink():
    node = RosNode()
    backend = build_voice_backend(
        BACKEND_ROS2,
        allow_actuation=True,
        ros_node=node,
        ros_twist_type=FakeTwist,
    )
    runtime = ScriptedRuntime(
        [
            '{"kind":"tool","name":"nav_drive","args":{"distance_m":0.05}}',
            '{"kind":"say","say":"I moved forward a short distance."}',
        ]
    )
    turn = CosmosVoiceSession(runtime, backend).handle_transcript(
        "Move a little closer."
    )
    assert turn.calls[0].result.ok is True
    assert node.calls[0][1] == "/cmd_vel"
    assert node.publisher.messages
    message = node.publisher.messages[0]
    assert 0.0 < message.linear.x <= backend.motion.limits().max_linear_velocity
    assert message.angular.z == 0.0
    backend.close()
