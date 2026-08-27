"""Drive vs parked-think prompt suffixes. Strings only — no model load."""

from __future__ import annotations

DRIVE_PROMPT_SUFFIX = (
    'You are driving. Reply with one JSON object only. '
    'Allowed action kinds: stop, drive, speak, wait, weather. '
    'Do not use extended thinking. Prefer a short stop or a brief drive. '
    'Clamps: vx abs <= 0.22, wz abs <= 1.0, then stop when duration elapses.'
)

PARKED_THINK_PROMPT_SUFFIX = (
    'You are parked (wheels stopped). You may reason before answering, then '
    'emit one JSON object only. Allowed action kinds: stop, drive, speak, wait, weather. '
    'Clamps: vx abs <= 0.22, wz abs <= 1.0, duration then stop. '
    'Do not command drive until you are ready to move.'
)


def prompt_suffix(*, moving: bool) -> str:
    """Return the suffix to append to the Cosmos user turn."""
    return DRIVE_PROMPT_SUFFIX if moving else PARKED_THINK_PROMPT_SUFFIX
