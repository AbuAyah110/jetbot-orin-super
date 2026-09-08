from __future__ import annotations

import io

from PIL import Image

from jetbot_agent.robot_loop.demos import (
    CREEP_DURATION_S,
    DEICTIC_REFUSE,
    OCCUPANCY_REFUSE,
    PLACE_UNSURE,
    eyes_first_reply,
    occupancy_allows_creep,
    occupancy_score,
    place_compare_action,
    place_slug,
    reply_claims_object_present,
    think_action,
)


def _jpeg(color, blob=None) -> bytes:
    image = Image.new('RGB', (448, 448), color)
    if blob is not None:
        x0, y0, x1, y1, fill = blob
        for x in range(x0, x1):
            for y in range(y0, y1):
                image.putpixel((x, y), fill)
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=90)
    return buf.getvalue()


def test_empty_floor_is_clear_and_a_lower_blob_is_not():
    empty = _jpeg((40, 42, 45))
    blocked = _jpeg((40, 42, 45), blob=(80, 280, 360, 440, (200, 30, 20)))
    empty_score = occupancy_score(empty)
    blocked_score = occupancy_score(blocked)
    assert empty_score['clear'] is True
    assert empty_score['blocked'] is False
    assert occupancy_allows_creep(empty)[0] is True
    assert blocked_score['blocked'] is True
    assert occupancy_allows_creep(blocked)[0] is False
    assert CREEP_DURATION_S < 1.0


def test_occupancy_fails_closed_on_garbage():
    allowed, detail = occupancy_allows_creep(b'not a jpeg')
    assert allowed is False
    assert detail['blocked'] is True


def test_eyes_first_missing_cannot_assert_the_object_is_here():
    rag = '[fact] A blue backpack is often by the couch; verify with the camera.'
    missing = eyes_first_reply(
        target='blue backpack', visible=False, rag=rag
    )
    present = eyes_first_reply(
        target='blue backpack', visible=True, side='left'
    )
    assert "don't see" in missing.lower() or 'not in this frame' in missing.lower()
    assert reply_claims_object_present(missing, 'blue backpack') is False
    assert 'couch' in missing.lower()
    assert 'left' in present.lower()
    assert reply_claims_object_present(present, 'blue backpack') is True


def test_place_slug_is_stable_and_prefixed():
    assert place_slug('the kitchen corner') == 'place:the_kitchen_corner'
    assert place_slug('') == ''


class FakeRuntime:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []
        self.last_text = ''

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.last_text = self.text
        return self.text


def test_think_route_cannot_drive():
    runtime = FakeRuntime(
        '{"action":"drive","vx":0.2,"wz":0,"duration_s":1,'
        '"say":"Going now","goal":"","reason":"nope"}'
    )
    action, _ = think_action(runtime, 'Think hard whether that path is safe.', b'x')
    assert action.kind == 'speak'
    assert action.vx == 0.0
    assert runtime.calls[0]['max_tokens'] >= 256


def test_place_compare_fails_closed_without_overlapping_reason():
    runtime = FakeRuntime(
        '{"action":"speak","say":"Yes I am there","at_place":true}'
    )
    action, _ = place_compare_action(
        runtime,
        _jpeg((10, 10, 10)),
        'kitchen corner',
        'kitchen corner with a white cabinet and a purple lamp',
    )
    assert action.kind == 'speak'
    assert action.say == PLACE_UNSURE


def test_place_compare_accepts_true_with_overlapping_reason():
    runtime = FakeRuntime(
        '{"action":"speak","say":"Yes, I see the white cabinet.","at_place":true}'
    )
    action, _ = place_compare_action(
        runtime,
        _jpeg((10, 10, 10)),
        'kitchen corner',
        'kitchen corner with a white cabinet and a purple lamp',
    )
    assert action.reason == 'place_match'
    assert 'cabinet' in action.say.lower()


def test_deictic_refuse_names_the_stop():
    assert "won't drive" in DEICTIC_REFUSE
    assert 'distance sensor' in OCCUPANCY_REFUSE
