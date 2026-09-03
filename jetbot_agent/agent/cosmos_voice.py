"""Voice-turn wiring for the Cosmos tool agent.

ASR and TTS remain adapters outside this module. A transcript enters
``CosmosVoiceSession.handle_transcript`` and a short spoken answer comes back.
Tool execution uses the existing deny-by-default registry over either an
in-memory mock sink or a caller-owned ROS 2 node.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from jetbot_agent.agent.cosmos_tool_agent import (
    AGENT_FALLBACK,
    AgentTurn,
    CosmosRuntime,
    CosmosToolAgent,
)
from jetbot_agent.agent.tools import Capability, ToolContext, ToolRegistry
from jetbot_agent.agent.tools.navigation_tools import register_navigation_tools
from jetbot_agent.navigation import (
    BoundedMotionAdapter,
    MockCmdVelSink,
    MotionBackendUnavailable,
    RosCmdVelSink,
    load_robot_limits,
)
from jetbot_agent.robot_loop.history import ChatHistory

BACKEND_MOCK = "mock"
BACKEND_ROS2 = "ros2"
SUPPORTED_BACKENDS = frozenset({BACKEND_MOCK, BACKEND_ROS2})


@dataclass
class CosmosVoiceBackend:
    """Owned agent-side resources. The caller still owns any ROS node."""

    kind: str
    registry: ToolRegistry
    motion: BoundedMotionAdapter
    sink: Any
    actuation_enabled: bool

    def close(self) -> None:
        self.registry.close()
        self.motion.stop()


def build_voice_backend(
    kind: str = BACKEND_MOCK,
    *,
    allow_actuation: bool = False,
    ros_node: Any = None,
    ros_twist_type: Any = None,
) -> CosmosVoiceBackend:
    """Build a safe registry over mock or ROS transport.

    ``allow_actuation`` is required even for mock motion. ROS additionally
    requires a node created and managed by the caller; this module never starts
    ROS, a motor process, or an I2C session.
    """
    backend = str(kind or "").strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError("backend must be mock or ros2")

    limits = load_robot_limits()
    if backend == BACKEND_MOCK:
        sink = MockCmdVelSink(cmd_vel_timeout_sec=limits.cmd_vel_timeout_sec)
    else:
        if ros_node is None:
            raise MotionBackendUnavailable(
                "ros2 backend requires a caller-owned initialized ROS node"
            )
        sink = RosCmdVelSink(ros_node, twist_type=ros_twist_type)

    motion = BoundedMotionAdapter(sink=sink, limits=limits)
    capabilities = [Capability.READ]
    if allow_actuation:
        capabilities.append(Capability.ACTUATE)
    registry = ToolRegistry(
        ToolContext(motion=motion),
        capabilities=capabilities,
        operator_ack_actuation=bool(allow_actuation),
    )
    register_navigation_tools(registry, allow=False)
    registry.allow("nav_status")
    if allow_actuation:
        for name in ("nav_drive", "nav_rotate", "nav_stop"):
            registry.allow(name)
    return CosmosVoiceBackend(
        kind=backend,
        registry=registry,
        motion=motion,
        sink=sink,
        actuation_enabled=bool(allow_actuation),
    )


class CosmosVoiceSession:
    """Stateful transcript → Cosmos tools → spoken text session."""

    def __init__(
        self,
        runtime: CosmosRuntime,
        backend: CosmosVoiceBackend,
        *,
        history: Optional[ChatHistory] = None,
    ) -> None:
        self.backend = backend
        self.history = history or ChatHistory(max_turns=10)
        self.agent = CosmosToolAgent(runtime, backend.registry)

    def handle_transcript(
        self,
        transcript: str,
        *,
        image_jpeg: Optional[bytes] = None,
    ) -> AgentTurn:
        speech = " ".join((transcript or "").split())
        if not speech:
            return AgentTurn(
                say="I didn't hear anything. Please try again.",
                failed=True,
                reason="empty_transcript",
            )
        turn = self.agent.run(
            speech,
            history=self.history.render(),
            image_jpeg=image_jpeg,
        )
        reply = turn.say or AGENT_FALLBACK
        self.history.add("user", speech)
        self.history.add("assistant", reply)
        return turn

    def close(self) -> None:
        self.backend.close()
