"""Independent pixel evidence for simple color-object navigation.

Cosmos remains the semantic planner. This check only offers a second opinion on
the horizontal side of a *strongly* colored target, because the same fresh
frame once produced a correct spoken description ("blue on my right") and a
contradictory route plan ("blue on my left").

It deliberately reports nothing far more often than it guesses. Measured on
this camera's dim indoor frames, a real blue target sat at RGB (35, 44, 49) --
only ~5 counts above the other channels -- while the empty floor measured ~6.
A loose threshold therefore tracks the floor, not the object. Evidence is only
returned when the colour is saturated, compact, and a small part of the frame;
otherwise the caller keeps using the model.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

SUPPORTED_COLORS = ('red', 'blue', 'green')
# Channel dominance in raw counts. Below ~20 this camera's blue-grey cast and
# the floor itself qualify, which is how background became a "target".
MIN_DOMINANCE = 20
MIN_VALUE = 60
# Dominance as a fraction of the channel value, which is what separates a
# saturated object from a tinted surface. Measured under this room's pink
# ambient lighting: wall pixels near (200, 150, 170) clear the absolute
# threshold with a dominance of 30, so 36-52% of the frame counted as "red" and
# the real red truck was thrown away as covering too much of the view. That
# wall scores 0.15 here; the truck scores about 0.8.
MIN_REL_DOMINANCE = 0.30
# Measured on this room's frames after the relative-dominance test: leftover
# specular noise peaked at 142 px, while the truck cut off at the frame edge
# still measured 2375. 250 sits clear of the noise with an order of magnitude of
# headroom on the weakest true detection.
MIN_PIXELS = 250
# A real object occupies a modest, contiguous part of the view. A mask covering
# more than this is lighting or floor, not a thing to drive at.
MAX_FRACTION = 0.20
# Fraction of frame width the middle 80% of matching pixels may span. The wood
# floor and walls trip the colour test across ~0.83 of the width; the actual
# objects measured 0.25 or less.
MAX_SPAN = 0.45
# Empty columns tolerated inside one cluster. A JPEG-compressed object has
# ragged edges and thin highlights, so a few blank columns do not mean a
# separate object; 12 of 448 is under 3% of the width.
CLUSTER_GAP_COLUMNS = 12


@dataclass(frozen=True)
class ColorGrounding:
    color: str = ''
    visible: bool = False
    side: str = ''
    pixels: int = 0
    center_x: float = -1.0
    width: int = 0
    fraction: float = 0.0
    span: float = 0.0
    rejected: str = ''

    def as_dict(self) -> dict:
        return {
            'color': self.color,
            'visible': self.visible,
            'side': self.side,
            'pixels': self.pixels,
            'center_x': self.center_x,
            'width': self.width,
            'fraction': round(self.fraction, 4),
            'span': round(self.span, 3),
            'rejected': self.rejected,
        }


def target_color(target: str) -> str:
    words = set(re.findall(r'[a-z]+', (target or '').lower()))
    return next((color for color in SUPPORTED_COLORS if color in words), '')


def _dominance(color: str, red: int, green: int, blue: int) -> int:
    if color == 'red':
        return red - max(green, blue)
    if color == 'blue':
        return blue - max(red, green)
    if color == 'green':
        return green - max(red, blue)
    return -255


def _channel(color: str, red: int, green: int, blue: int) -> int:
    return {'red': red, 'blue': blue, 'green': green}.get(color, 0)


def _largest_cluster(columns: list) -> tuple:
    """Return ``(start, end, pixels)`` of the heaviest run of matching columns.

    Runs separated by more than ``CLUSTER_GAP_COLUMNS`` blank columns are
    treated as different objects, and the heaviest one wins.
    """
    best = (0, 0, 0)
    start = None
    end = 0
    total = 0
    blanks = 0
    for x, count in enumerate(columns):
        if count:
            if start is None:
                start = x
                total = 0
            end = x
            total += count
            blanks = 0
        elif start is not None:
            blanks += 1
            if blanks > CLUSTER_GAP_COLUMNS:
                if total > best[2]:
                    best = (start, end, total)
                start = None
                total = 0
                blanks = 0
    if start is not None and total > best[2]:
        best = (start, end, total)
    return best


def locate_color(jpeg: bytes, target: str) -> ColorGrounding:
    """Return a side only when the colour evidence is unambiguous."""
    color = target_color(target)
    if not color or not jpeg:
        return ColorGrounding(color=color, rejected='unsupported_target')
    try:
        from PIL import Image

        image = Image.open(io.BytesIO(jpeg)).convert('RGB')
    except Exception:
        return ColorGrounding(color=color, rejected='decode_failed')

    width, height = image.size
    pixels = (
        image.get_flattened_data()
        if hasattr(image, 'get_flattened_data')
        else image.getdata()
    )
    columns = [0] * width
    matched = 0
    for index, (red, green, blue) in enumerate(pixels):
        value = _channel(color, red, green, blue)
        if value < MIN_VALUE:
            continue
        dominance = _dominance(color, red, green, blue)
        if dominance < MIN_DOMINANCE:
            continue
        if dominance < value * MIN_REL_DOMINANCE:
            continue
        columns[index % width] += 1
        matched += 1

    fraction = matched / float(width * height)
    if matched < MIN_PIXELS:
        return ColorGrounding(
            color=color, pixels=matched, width=width, fraction=fraction,
            rejected='too_few_pixels',
        )
    if fraction > MAX_FRACTION:
        return ColorGrounding(
            color=color, pixels=matched, width=width, fraction=fraction,
            rejected='covers_too_much_of_frame',
        )

    # Measure the heaviest cluster rather than every matching pixel. With the
    # truck centred and small red patches in two corners, a percentile span
    # stretched across 0.70 of the width while the truck itself covered 0.22, so
    # a clean centred target was rejected as diffuse. The cluster is the object
    # being tracked; stray colour elsewhere in the room no longer votes on its
    # position or its size.
    x_low, x_high, clustered = _largest_cluster(columns)
    if clustered < MIN_PIXELS:
        return ColorGrounding(
            color=color, pixels=clustered, width=width, fraction=fraction,
            rejected='too_few_pixels',
        )
    fraction = clustered / float(width * height)
    if fraction > MAX_FRACTION:
        return ColorGrounding(
            color=color, pixels=clustered, width=width, fraction=fraction,
            rejected='covers_too_much_of_frame',
        )
    span = (x_high - x_low) / float(width)
    if span > MAX_SPAN:
        return ColorGrounding(
            color=color, pixels=clustered, width=width, fraction=fraction,
            span=span, rejected='diffuse_not_an_object',
        )

    weighted = sum(
        x * columns[x] for x in range(x_low, x_high + 1)
    )
    inner = sum(columns[x_low:x_high + 1])
    center_x = weighted / float(inner) if inner else (x_low + x_high) / 2.0
    matched = clustered
    side = (
        'left' if center_x < width / 3
        else 'right' if center_x >= 2 * width / 3
        else 'center'
    )
    return ColorGrounding(
        color=color,
        visible=True,
        side=side,
        pixels=matched,
        center_x=center_x,
        width=width,
        fraction=fraction,
        span=span,
    )
