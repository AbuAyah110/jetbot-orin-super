"""Tavily search stub. Stage H / I4. Fail closed without API key."""

from jetbot_agent._stage import StageNotReady


def web_search(*args, **kwargs):
    raise StageNotReady("tools.search_tools waits for Stage H / I4")
