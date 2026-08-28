from __future__ import annotations

from jetbot_agent.robot_loop.conversation import (
    CONVERSATION_FALLBACK,
    CONVERSATION_MAX_TOKENS,
    build_conversation_prompt,
    conversation_action,
)


class FakeRuntime:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []
        self.last_text = ""

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.last_text = self.text
        return self.text


def test_general_question_becomes_spoken_answer_without_motion():
    runtime = FakeRuntime(
        '{"action":"speak","vx":0.9,"wz":1,"duration_s":2,'
        '"say":"Saturn is the sixth planet from the Sun.",'
        '"goal":"","reason":"conversation","extra":"ignored"}'
    )
    action, raw = conversation_action(
        runtime,
        "Which planet is Saturn?",
        "user: What is our solar system?",
    )

    assert action.kind == "speak"
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert action.say == "Saturn is the sixth planet from the Sun."
    assert raw == runtime.text
    call = runtime.calls[0]
    assert call["image_jpeg"] is None
    assert call["max_tokens"] == CONVERSATION_MAX_TOKENS
    assert "friendly conversational robot" in call["system"]
    assert "Which planet is Saturn?" in call["user_text"]
    assert "What is our solar system?" in call["user_text"]


def test_open_ended_turn_rejects_model_drive_request():
    runtime = FakeRuntime(
        '{"action":"drive","vx":0.2,"wz":0,"duration_s":1,'
        '"say":"","goal":"","reason":"model tried motion"}'
    )
    action, _ = conversation_action(runtime, "Tell me a joke")

    assert action.kind == "stop"
    assert action.vx == 0.0
    assert action.wz == 0.0
    assert action.say == ""
    assert action.reason == "conversation_non_speak"
    assert CONVERSATION_FALLBACK


def test_invalid_conversation_json_fails_closed():
    action, raw = conversation_action(FakeRuntime("not json"), "Who are you?")
    assert raw == "not json"
    assert action.kind == "stop"
    assert action.raw_ok is False
    assert action.vx == action.wz == 0.0
    assert action.say == ""


def test_empty_say_fails_closed():
    runtime = FakeRuntime(
        '{"action":"speak","vx":0,"wz":0,"duration_s":0,'
        '"say":"","goal":"","reason":"conversation"}'
    )
    action, _ = conversation_action(runtime, "Say something")
    assert action.kind == "stop"
    assert action.vx == action.wz == 0.0


def test_runtime_error_fails_closed():
    class BrokenRuntime:
        last_text = "partial"

        def generate(self, **_kwargs):
            raise RuntimeError("offline")

    action, raw = conversation_action(BrokenRuntime(), "Hello")
    assert raw == "partial"
    assert action.kind == "stop"
    assert action.raw_ok is False
    assert action.reason == "conversation_error:RuntimeError"


def test_prompt_is_bounded_and_marks_history_as_data():
    prompt = build_conversation_prompt(" hello   there ", "x" * 4000)
    assert "<history>" in prompt
    assert "</history>" in prompt
    assert "<utterance>hello there</utterance>" in prompt
    assert len(prompt) < 2200


def test_visual_follow_up_uses_current_image_but_cannot_move():
    runtime = FakeRuntime(
        '{"action":"speak","vx":0.4,"wz":1,"duration_s":2,'
        '"say":"It looks like a blue toy box.","goal":"","reason":"visual conversation"}'
    )

    action, _ = conversation_action(
        runtime,
        'What color is it?',
        'user: What do you see?\nassistant: I see a toy box.',
        image_jpeg=b'current jpeg only',
    )

    assert action.kind == 'speak'
    assert action.vx == action.wz == 0.0
    assert action.say == 'It looks like a blue toy box.'
    call = runtime.calls[0]
    assert call['image_jpeg'] == b'current jpeg only'
    assert 'current camera frame' in call['system']
    assert 'what I can see' not in action.say


def test_visual_follow_up_admits_uncertainty():
    runtime = FakeRuntime(
        '{"action":"speak","vx":0,"wz":0,"duration_s":0,'
        '"say":"I cannot tell; that object is too blurry.","goal":"","reason":"visual conversation"}'
    )

    action, _ = conversation_action(
        runtime, 'Can you identify that?', image_jpeg=b'blurred frame'
    )

    assert action.kind == 'speak'
    assert 'cannot tell' in action.say
