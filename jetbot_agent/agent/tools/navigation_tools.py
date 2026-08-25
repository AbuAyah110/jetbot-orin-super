"""Navigation tools stub. Stage H / I5. Motion only via cmd_vel + watchdog; LLM never PWM."""

from jetbot_agent._stage import StageNotReady


def navigate(*args, **kwargs):
    raise StageNotReady("tools.navigation_tools waits for Stage H / I5")
