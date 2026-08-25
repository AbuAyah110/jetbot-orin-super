"""SQLite facts stub. Stage I memory (after agent)."""

from jetbot_agent._stage import StageNotReady


class FactsDb:
    def __init__(self, *args, **kwargs):
        raise StageNotReady("memory.facts_db waits for Stage I")
