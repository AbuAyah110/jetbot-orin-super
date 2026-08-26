# Architecture — JetBot Orin Super AI

Architectural source of truth: [`PROJECT_PLAN.md`](../PROJECT_PLAN.md).

## Control hierarchy (non-negotiable)

```text
User / Hermes / Qwen
        │
        ▼
   high-level tools only
   (navigate, rotate, inspect, stop)
        │
        ▼
   Robot MCP  (later milestones)
        │
        ▼
   ROS 2  (/cmd_vel, Nav2, …)
        │
        ▼
   jetbot_base  (velocity limit, watchdog, e-stop)
        │
        ▼
   MotorDriver abstraction
        │
        ▼
   wheels
```

**LLMs never set PWM, never write GPIO, never disable the watchdog.**

## Milestone 0–1 (current)

| Layer | Responsibility |
| --- | --- |
| `scripts/diagnostics.sh` | Jetson/host health report |
| `src/jetbot_control` | Deterministic motor API (`MockMotorDriver`, later I2C) |
| `ros2_ws/src/jetbot_base` | `/cmd_vel` → motors, limits, watchdog, status, teleop |

No Cosmos, Qwen, Hermes, EmbeddingGemma, MCP, or Nav2 in this milestone.

## Later (summary)

| Role | Model / system |
| --- | --- |
| Executive conversation | Qwen3.5-0.8B |
| Physical / visual specialist | Cosmos Reason2-2B via llama.cpp |
| Long-term memory embeddings | **jina-clip-v2** (1024-d native, **store 256-d** Matryoshka) — see [`jina_clip_v2.md`](jina_clip_v2.md). EmbeddingGemma remains the smaller **text-only** option. **Not** SigLIP 2, **not** Nemotron-embed. |
| Agent orchestration | Hermes + MCP (Robot, Memory, …) |
| Navigation | Nav2 only after odometry + range sensing |

## Development

Laptop Cursor → Remote SSH → Jetson (`Host jetbot`). Repo, ROS, Docker, and hardware all live on the Jetson.
