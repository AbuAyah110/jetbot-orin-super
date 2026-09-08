# Roadmap

Full plan: [`PROJECT_PLAN.md`](../PROJECT_PLAN.md).

**Live Jetson install order** is [`docs/bringup/README.md`](bringup/README.md): A–G, then **agent integration (Stage H / I1–I8) before memory (Stage I)**. The M0–M14 table below is the historical `PROJECT_PLAN` sequence and is **not** the current bring-up queue (that plan put memory MCP before Hermes).

## Milestone order (do not skip)

| ID | Milestone | Status |
| --- | --- | --- |
| M0 | Remote development / diagnostics | In progress |
| M1 | ROS 2 + motor control + watchdog | In progress |
| M2 | CSI camera (mock-first on laptop) | In progress |
| M3 | Cosmos Reason2-2B (llama.cpp) | Planned |
| M4 | EmbeddingGemma + SQLite/HNSW | Planned |
| M5 | Memory MCP | Planned |
| M6 | Robot MCP | Planned |
| M7 | Hermes + Cosmos | Planned |
| M8 | Qwen3.5-0.8B executive | Planned |
| M9 | Voice (WebRTC APM / Zipformer / Piper VITS) | Compact F4/F5 replacement done; F6 duplex open |
| M10 | Sensors + Nav2 | Planned (needs encoders/IMU/range) |
| M11 | Web research | Planned |
| M12 | Proactive event loop | Planned |
| M13 | Skills / self-improvement | Planned |
| M14 | Additional MCP servers | Planned |

## Hardware guide track (parallel)

Classic JetBot assembly, BOM, STL, and Jupyter demos remain available under `docs/` and `notebooks/`. Power for Super mode is still TBD — [power.md](power.md).
