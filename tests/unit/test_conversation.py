from __future__ import annotations

import pytest

from jetbot_agent.robot_loop.conversation import (
    CONVERSATION_FALLBACK,
    CONVERSATION_MAX_TOKENS,
    MOTION_CLAIM_REPLY,
    SIGHT_CLAIM_REPLY,
    build_conversation_prompt,
    claims_motion,
    claims_sight,
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
    prompt = build_conversation_prompt(
        " hello   there ",
        "x" * 4000,
        "[user_fact] My favorite color is blue." + "y" * 4000,
    )
    assert "<history>" in prompt
    assert "</history>" in prompt
    assert "<memory>" in prompt
    assert "</memory>" in prompt
    assert "My favorite color is blue." not in prompt  # newest bounded tail only
    assert "<utterance>hello there</utterance>" in prompt
    assert len(prompt) < 3100


def test_retrieved_memory_is_passed_as_quoted_context():
    runtime = FakeRuntime(
        '{"action":"speak","vx":0,"wz":0,"duration_s":0,'
        '"say":"Your favorite color is blue.","goal":"","reason":"conversation"}'
    )
    action, _ = conversation_action(
        runtime,
        'What is my favorite color?',
        rag='[user_fact] My favorite color is blue.',
    )

    assert action.say == 'Your favorite color is blue.'
    prompt = runtime.calls[0]['user_text']
    assert '<memory>[user_fact] My favorite color is blue.</memory>' in prompt


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


@pytest.mark.parametrize(
    'say',
    [
        'Sure, I am moving around the object now.',
        "I'm going around it to the left.",
        'I will drive around the box for you.',
        'I have moved around the chair.',
        'Okay, turning around it now.',
        "I've circled the object.",
        # Reported live: a parked turn answered this and nothing moved.
        'I see the blue object in front of me; I will approach it.',
        'I am approaching the blue object.',
        'Sure, I will go to it now.',
        "I'm coming over to the blue object.",
        'I will get closer to it.',
    ],
)
def test_spoken_motion_claims_are_replaced_with_an_honest_refusal(say):
    # This route cannot reach the motors, so any first-person motion claim is
    # a false report of movement the robot never performed.
    assert claims_motion(say) is True
    runtime = FakeRuntime(
        '{"action":"speak","vx":0,"wz":0,"duration_s":0,'
        '"say":"' + say + '","goal":"","reason":"conversation"}'
    )

    action, _ = conversation_action(runtime, 'move around the object')

    assert action.kind == 'speak'
    assert action.say == MOTION_CLAIM_REPLY
    assert action.reason == 'conversation_motion_claim'


@pytest.mark.parametrize(
    'say',
    [
        'Saturn is the sixth planet from the Sun.',
        'I see a blue box on my left.',
        "I can't sense contact, so I won't try that.",
        'That was a moving story about a train.',
        'I do not know the answer to that.',
    ],
)
def test_ordinary_answers_are_not_treated_as_motion_claims(say):
    assert claims_motion(say) is False


def test_a_motion_ack_after_a_first_sentence_is_still_caught():
    # Logged live: the ack sat in the second sentence, so a start-anchored
    # guard passed it and the robot reported moving while parked.
    assert claims_motion('Found blue puck. Moving toward it.') is True


@pytest.mark.parametrize(
    'say',
    [
        'Found blue puck. Moving toward it.',
        'I see the blue object in front of me.',
        'I found your keys on the table.',
        'Spotted a red box on my right.',
    ],
)
def test_a_parked_text_turn_cannot_report_a_sighting(say):
    # No image reached the model on this turn, so any sighting is invented.
    assert claims_sight(say) is True
    runtime = FakeRuntime(
        '{"action":"speak","vx":0,"wz":0,"duration_s":0,'
        '"say":"' + say + '","goal":"","reason":"conversation"}'
    )

    action, _ = conversation_action(runtime, 'find blue object and move to it')

    assert action.kind == 'speak'
    assert action.say in {MOTION_CLAIM_REPLY, SIGHT_CLAIM_REPLY}
    assert action.reason in {
        'conversation_motion_claim',
        'conversation_sight_claim',
    }


@pytest.mark.parametrize(
    'say',
    [
        'You told me the backpack is on the couch.',
        'Earlier you said your keys were on the table.',
        'Saturn is the sixth planet from the Sun.',
        'I do not know the answer to that.',
    ],
)
def test_memory_answers_are_not_sighting_claims(say):
    assert claims_sight(say) is False


def test_conversation_prompt_forbids_claiming_movement():
    runtime = FakeRuntime(
        '{"action":"speak","vx":0,"wz":0,"duration_s":0,'
        '"say":"I cannot do that move.","goal":"","reason":"conversation"}'
    )

    conversation_action(runtime, 'move around the object')

    assert 'never say that you are moving' in runtime.calls[0]['system']
