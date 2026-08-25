"""Hermes harness stub. Implementation is Stage H ticket I1 (before memory)."""

from jetbot_agent._stage import StageNotReady


class HermesHarness:
    def __init__(self, *args, **kwargs):
        raise StageNotReady("agent.hermes_harness waits for Stage H / I1")
