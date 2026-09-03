#!/usr/bin/env python3
"""Guide a human through voice tests and save a Markdown scorecard.

This script never drives the robot. It only presents setup instructions and
phrases to speak to the already-running talk-and-drive service.
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TestCase:
    name: str
    setup: str
    phrase: str
    passed: str
    failed: str
    motion: bool = False


TESTS = (
    TestCase(
        "General conversation",
        "Leave the robot parked. No object is required.",
        "What is the capital of France?",
        "It answers Paris naturally and does not move.",
        "It moves, emits JSON, or does not answer the question.",
    ),
    TestCase(
        "Robot-relative camera description",
        "Sit visibly in front of the camera while the robot remains parked.",
        "What is in front of you?",
        "It describes you as a visible person in front of it; no movement.",
        'It describes you as itself ("I am sitting..."), repeats the question, or moves.',
    ),
    TestCase(
        "Blue search",
        "Put one large blue object about 1 metre away, clearly to JetBot's right.",
        "Find the blue object.",
        "It turns to search, finds the blue object, reports its side, then stops.",
        "It never turns, invents another colour, keeps roaming, or drives toward it.",
        motion=True,
    ),
    TestCase(
        "Blue search and approach",
        "Reset JetBot forward. Put the large blue object about 1 metre away and to its right. Keep the path clear.",
        "Find the blue object and go to it.",
        "It searches, announces the find, re-checks the view, creeps toward it, then stops.",
        "It only says it will approach, never moves after finding it, loses the target, or keeps driving.",
        motion=True,
    ),
    TestCase(
        "Direct blue approach",
        "Place the large blue object clearly in the camera centre, about 1 metre away. Keep the path clear.",
        "Move toward the blue object.",
        "It verifies blue in the current frame, makes short approach pulses, and stops.",
        "It moves toward an absent target, moves continuously, or only promises to move.",
        motion=True,
    ),
    TestCase(
        "Clear-path ToF creep",
        "Point JetBot into at least 1 metre of clear space. Ensure nothing crosses the ToF sensor beam.",
        "If the floor is clear, creep forward.",
        "It makes at most one short forward pulse and stops.",
        "It continues moving, makes several pulses, or claims an obstacle at an implausible distance.",
        motion=True,
    ),
    TestCase(
        "Blocked-path ToF stop",
        "Place a flat obstacle directly in front of the ToF sensor, 15 to 20 cm away.",
        "If the floor is clear, creep forward.",
        "It stays still and says the distance sensor sees something close ahead.",
        "Any wheel moves, or it says the path is clear.",
        motion=True,
    ),
    TestCase(
        "Simple turn",
        "Clear space around the robot.",
        "Turn left.",
        "It acknowledges, makes one short left turn, and stops.",
        "It turns the wrong way, keeps turning, or only says it will turn.",
        motion=True,
    ),
    TestCase(
        "Unsupported target honesty",
        "Put ordinary keys in view. Leave the blue object out of view.",
        "Find my keys and go to them.",
        "It does not falsely promise or perform a reliable approach to the keys.",
        "It says it is approaching but stands still, or drives based on an ungrounded claim.",
        motion=True,
    ),
)


def service_state() -> str:
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "jetbot-talk-and-drive.service"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def ask_result() -> tuple[str, str]:
    while True:
        answer = input("Result [p=pass, f=fail, s=skip, q=save and quit]: ").strip().lower()
        result = {"p": "PASS", "f": "FAIL", "s": "SKIP", "q": "QUIT"}.get(answer)
        if result:
            break
        print("Enter p, f, s, or q.")
    if result == "QUIT":
        return result, ""
    note = input("Short note (optional): ").strip()
    return result, note


def write_report(path: Path, rows: list[tuple[TestCase, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(result == "PASS" for _, result, _ in rows)
    failed = sum(result == "FAIL" for _, result, _ in rows)
    skipped = sum(result == "SKIP" for _, result, _ in rows)
    lines = [
        "# JetBot guided voice test",
        "",
        f"- Date: {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Service at start: `{service_state()}`",
        f"- Score: **{passed} passed, {failed} failed, {skipped} skipped**",
        "",
    ]
    for index, (case, result, note) in enumerate(rows, 1):
        lines.extend(
            [
                f"## {index}. {case.name} — {result}",
                "",
                f'- Phrase: "{case.phrase}"',
                f"- Note: {note or '(none)'}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Report path (default: <repo>/data/test_reports/voice-TIMESTAMP.md)",
    )
    parser.add_argument("--list", action="store_true", help="Print tests without prompting")
    args = parser.parse_args()

    if args.list:
        for index, case in enumerate(TESTS, 1):
            print(f'{index}. {case.name}: "{case.phrase}"')
        return 0

    stamp = dt.datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    # Anchored to the repo so a run from any directory lands in one place.
    output = args.output or REPO_ROOT / "data/test_reports" / f"voice-{stamp}.md"
    print("JetBot guided voice test")
    print(f"Service: {service_state()}")
    print("This checklist never drives JetBot; only your spoken commands can.")
    print("For motion tests: use open floor, stay beside it, and keep a hand on power.")
    input("\nPress Enter when ready...")

    rows: list[tuple[TestCase, str, str]] = []
    try:
        for index, case in enumerate(TESTS, 1):
            print("\n" + "=" * 72)
            print(f"TEST {index}/{len(TESTS)}: {case.name}")
            print(f"Motion: {'YES' if case.motion else 'NO'}")
            print(f"SETUP: {case.setup}")
            print(f'SAY EXACTLY: "{case.phrase}"')
            print(f"PASS: {case.passed}")
            print(f"FAIL: {case.failed}")
            input("\nPress Enter after setup, then say the phrase...")
            result, note = ask_result()
            if result == "QUIT":
                break
            rows.append((case, result, note))
    except (EOFError, KeyboardInterrupt):
        print("\nStopping early and saving completed results.")

    write_report(output, rows)
    print(f"\nSaved report: {output}")
    return 1 if any(result == "FAIL" for _, result, _ in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
