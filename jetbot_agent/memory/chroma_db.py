"""ChromaDB stub. Stage I memory (after agent)."""

from jetbot_agent._stage import StageNotReady


class ChromaDb:
    def __init__(self, *args, **kwargs):
        raise StageNotReady("memory.chroma_db waits for Stage I")
