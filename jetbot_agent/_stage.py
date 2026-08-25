"""Stubs import; they raise only when constructed or called."""


class StageNotReady(NotImplementedError):
    """Feature waits for a bring-up gate. Safe to import."""
