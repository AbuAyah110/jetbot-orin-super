"""Memory compactor stub. After Stage I; no LLM summarizer in the memory install pass."""

from jetbot_agent._stage import StageNotReady


class MemoryCompactor:
    def __init__(self, *args, **kwargs):
        raise StageNotReady("memory.memory_compactor waits until after Stage I")
