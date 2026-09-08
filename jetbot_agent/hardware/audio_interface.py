"""Waveshare / Solid State System USB PnP (SSS1629) ALSA identity.

Resolve the card by USB/ALSA name, never by a hardcoded card index.
Mixer baseline: capture ~80%, playback low, sidetone OFF.
Until Stage F2 AEC passes, capture and playback must not overlap.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

CARDS_PATH = Path("/proc/asound/cards")
NAME_MARKERS = (
    "solid state system",
    "usb pnp audio",
    "sss1629",
)


def list_cards() -> list[dict]:
    text = CARDS_PATH.read_text(encoding="utf-8", errors="replace")
    cards = []
    for match in re.finditer(
        r"^\s*(\d+)\s+\[([^\]]+)\]:\s+\S+\s+-\s+(.+)$\n\s+(.+)$",
        text,
        flags=re.M,
    ):
        cards.append(
            {
                "index": int(match.group(1)),
                "id": match.group(2).strip(),
                "short": match.group(3).strip(),
                "long": match.group(4).strip(),
            }
        )
    return cards


def resolve_sss1629() -> dict:
    """Pick the Waveshare/SSS USB PnP card by name. Raises if missing or ambiguous."""
    matches = []
    for card in list_cards():
        blob = f"{card['id']} {card['short']} {card['long']}".lower()
        if any(marker in blob for marker in NAME_MARKERS):
            matches.append(card)
    if not matches:
        raise RuntimeError(
            "SSS1629 / USB PnP Audio card not found in /proc/asound/cards; "
            "identify by ALSA USB name, never a numeric card index"
        )
    if len(matches) > 1:
        raise RuntimeError(f"multiple USB PnP audio cards matched: {matches}")
    card = matches[0]
    alsa_id = card["id"]
    plug = f"plughw:CARD={alsa_id},DEV=0"
    return {
        "usb_name": card["long"] or card["short"],
        "card_index_ephemeral": card["index"],
        "alsa_id": alsa_id,
        "alsa_capture": plug,
        "alsa_playback": plug,
        "sidetone_enabled": False,
    }


def _amixer(card_index: int, *args: str) -> str:
    cmd = ["amixer", "-c", str(card_index), *args]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout


def apply_safe_mixer_baseline(card_index: int) -> str:
    """Capture ~80%, speaker 75% unmuted, Mic playback/sidetone muted.

    SSS1629 Speaker 20% is −50 dB and inaudible on this mono setup.
    75% is −16 dB, the level the user confirmed they can hear.
    Never unmute Mic playback (hardware sidetone / feedback).
    """
    logs = []
    logs.append(_amixer(card_index, "sset", "Speaker", "75%", "unmute"))
    logs.append(_amixer(card_index, "sset", "Mic", "80%", "cap"))
    logs.append(_amixer(card_index, "sset", "Mic", "playback", "0%", "mute"))
    return "\n".join(logs)


def mixer_report(card_index: int) -> str:
    proc = subprocess.run(
        ["amixer", "-c", str(card_index), "scontents"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout


def persist_mixer(card_index: int, state_path: Path) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["alsactl", "--file", str(state_path), "store", str(card_index)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        state_path.write_text(mixer_report(card_index), encoding="utf-8")
        state_path.with_suffix(".scontents.txt").write_text(
            mixer_report(card_index), encoding="utf-8"
        )
