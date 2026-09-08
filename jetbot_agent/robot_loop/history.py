"""Bounded text-only history migrated from the legacy thin-stack prototype."""

from __future__ import annotations

import json
from pathlib import Path


class ChatHistory:
    """Keep short text turns; never retain JPEGs or private think blocks."""

    def __init__(self, max_turns: int = 6) -> None:
        if max_turns < 1:
            raise ValueError('max_turns must be positive')
        self.max_turns = int(max_turns)
        self._turns: list[tuple[str, str]] = []

    def add(self, role: str, text: str) -> None:
        if isinstance(text, (bytes, bytearray)):
            raise TypeError('ChatHistory is text-only; pass only the current JPEG to generate()')
        clean = (text or '').strip()
        if not clean:
            return
        lowered = clean.lower()
        if 'data:image' in lowered or 'image_url' in lowered:
            raise ValueError('Refusing to store image payload in chat history')
        if '<think>' in clean and '</think>' in clean:
            clean = clean.split('</think>', 1)[-1].strip() or 'thought stripped'
        self._turns.append((str(role)[:32], clean[:400]))
        self._turns = self._turns[-self.max_turns :]

    def add_turn(self, user: str, assistant: str) -> None:
        """Store one complete exchange so routes cannot forget half a turn."""
        self.add('user', user)
        self.add('assistant', assistant)

    def render(self, max_chars: int = 2400) -> str:
        if not self._turns:
            return '(none)'
        while True:
            rendered = '\n'.join('{0}: {1}'.format(role, text) for role, text in self._turns)
            if len(rendered) <= max_chars or len(self._turns) == 1:
                return rendered[-max_chars:]
            self._turns.pop(0)

    def save(self, path: str | Path) -> None:
        """Persist the bounded text window atomically; images are never stored."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + '.tmp')
        temporary.write_text(
            json.dumps(
                {'version': 1, 'turns': self._turns[-self.max_turns :]},
                ensure_ascii=False,
                indent=2,
            )
            + '\n',
            encoding='utf-8',
        )
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path, max_turns: int = 10) -> 'ChatHistory':
        """Load saved text through ``add`` so all safety checks still apply."""
        history = cls(max_turns=max_turns)
        source = Path(path)
        if not source.is_file():
            return history
        try:
            payload = json.loads(source.read_text(encoding='utf-8'))
            turns = payload.get('turns', [])
            if not isinstance(turns, list):
                return history
            for item in turns:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and isinstance(item[1], str)
                ):
                    history.add(item[0], item[1])
        except (OSError, ValueError, TypeError):
            return cls(max_turns=max_turns)
        return history
