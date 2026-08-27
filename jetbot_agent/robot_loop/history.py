"""Bounded text-only history migrated from the legacy thin-stack prototype."""

from __future__ import annotations


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

    def render(self, max_chars: int = 2400) -> str:
        if not self._turns:
            return '(none)'
        while True:
            rendered = '\n'.join('{0}: {1}'.format(role, text) for role, text in self._turns)
            if len(rendered) <= max_chars or len(self._turns) == 1:
                return rendered[-max_chars:]
            self._turns.pop(0)
