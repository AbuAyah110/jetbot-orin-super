from __future__ import annotations

import io

from PIL import Image, ImageDraw

from jetbot_agent.robot_loop.color_grounding import locate_color, target_color


def _frame(color: str, box: tuple[int, int, int, int]) -> bytes:
    image = Image.new('RGB', (300, 120), (20, 20, 20))
    ImageDraw.Draw(image).rectangle(box, fill=color)
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=95)
    return stream.getvalue()


def test_extracts_supported_target_color():
    assert target_color('THE RED OBJECT') == 'red'
    assert target_color('blue and white box') == 'blue'
    assert target_color('chair') == ''


def test_locates_red_left_blue_center_and_green_right():
    assert locate_color(_frame('red', (10, 20, 70, 100)), 'red object').side == 'left'
    assert locate_color(_frame('blue', (120, 20, 180, 100)), 'blue object').side == 'center'
    assert locate_color(_frame('green', (230, 20, 290, 100)), 'green object').side == 'right'


def test_missing_or_unsupported_color_is_not_evidence():
    black = _frame('black', (10, 20, 70, 100))
    assert locate_color(black, 'red object').visible is False
    assert locate_color(black, 'chair').visible is False


def test_tint_spread_across_the_view_is_not_an_object():
    """A wood floor tripped the old detector across 83% of the frame width."""
    image = Image.new('RGB', (300, 120), (20, 20, 20))
    draw = ImageDraw.Draw(image)
    for x in range(10, 290, 14):
        draw.rectangle((x, 50, x + 1, 80), fill=(150, 40, 40))
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=95)

    grounding = locate_color(stream.getvalue(), 'red object')

    assert grounding.visible is False
    assert grounding.rejected == 'diffuse_not_an_object'


def test_weak_channel_dominance_is_not_an_object():
    """The dim 'blue' target measured only ~5 counts above red and green."""
    image = Image.new('RGB', (300, 120), (20, 20, 20))
    ImageDraw.Draw(image).rectangle((120, 20, 180, 100), fill=(35, 44, 49))
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=95)

    assert locate_color(stream.getvalue(), 'blue object').visible is False
