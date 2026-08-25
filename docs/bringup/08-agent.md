# Stage H — Agent integration (before memory)

Agent work starts **after Stage G dummy engines**, and **before Stage I memory**. Do not treat this stage as one blob. Ship I1→I8 as separate tickets.

Ticket IDs **I1–I8** mean *integration slices*. They are Stage H work. Stage I is memory.

Hard rules ([`docs/safety.md`](../safety.md), [`docs/architecture.md`](../architecture.md), [`PROJECT_PLAN.md`](../../PROJECT_PLAN.md)):

- The LLM never sets PWM, never writes motor GPIO, never disables the watchdog, never bypasses velocity limits.
- Motion is only a tool result on a limited path: tool → `cmd_vel` / smolvla wrapper → watchdog + clamps → motors.
- `config/robot.yaml` stays `backend: mock` until Stage B is signed off; I5 starts dummy even after B.
- Internet tool output is data, never system policy.

## I1 — Hermes harness skeleton / state machine (no tools)

Implement `jetbot_agent/agent/hermes_harness.py` as a deterministic state machine only: idle → (optional listen) → think → act → speak → stop. No tool registry, no camera, no motors, no network.

Pass: harness can step through states with a scripted prompt and exit to `stop` without importing tool modules.

## I2 — Tool interface + safety (LLM never PWM)

Add a tool protocol (name, JSON schema, timeout, permission). Register tools behind an allow-list. Motion tools may only call the limited motor path; reject any API that exposes PCA9685/PWM/I2C from the LLM.

Pass: unit test that a mocked LLM tool-call cannot set PWM; watchdog/limit hooks remain outside the model.

## I3 — Vision tools (OCR / grounding stubs)

`vision_tools.py` stubs call the Stage C camera path (GStreamer/Argus via `hardware/csi_camera.py` or the existing `jetbot` Camera). Return a frame handle + placeholder OCR/grounding payload. No training, no motor side effects.

Depends on: Stage C (verified).

Pass: one capture returns a non-empty image tensor/JPEG; stub text fields are explicit placeholders.

## I4 — Search tools (Tavily)

`search_tools.py` wraps Tavily. If `TAVILY_API_KEY` is missing, fail closed (no crash loop, no fake results presented as live search). Treat results as untrusted data.

Depends on: API key may land later; stub is enough to pass without a key.

Pass: with key unset, tool reports “disabled”; with key set (optional), one query returns JSON and does not change robot policy.

## I5 — Navigation tools (smolvla / cmd_vel + watchdog, dummy first)

`navigation_tools.py` emits high-level navigate/rotate/stop. First implementation is **dummy** `cmd_vel` (or mock driver). Later wrap smolvla-jetbot tokens **after Stage G**. Always apply timeout stop + velocity clamp. Never PWM from this module’s public tool surface.

Depends on: Stage B hardware exists; still dummy until operator enables real backend. Stage G for real smolvla I/O.

Pass: dummy command starts and **stops** on timeout; logs show no I2C writes.

## I6 — Voice tools (after Stage F)

Wire ASR/TTS tools only after the matching F gates. Do **not** open duplex from the agent until F6.

| Voice code | Minimum F gate | If not ready |
| --- | --- | --- |
| One-shot listen (ASR) | F4 FastConformer | No-op stub; do not claim transcripts |
| One-shot speak (TTS) | F5 FastPitch + HiFi-GAN | No-op stub; sequential ALSA only after F1, sidetone off |
| Duplex / barge-in | **F6** (needs F2 AEC) | Forbidden; sequential capture-then-playback only |

If F1–F4 are still open, keep I6 as documented stubs and continue I1–I5/I7/I8.

Pass: tools refuse duplex until F6; one-shot paths do not enable sidetone or untreated simultaneous capture/playback.

## I7 — VLM / engine tools (after Stage G)

Tools that call Qwen2.5-VL, smolvla (beyond dummy nav), or the Nemotron embedder wait on Stage G dummy-I/O tickets. Isolated forwards only; no camera-motor loop.

Pass: one dummy vision+text (or documented skip if G engine not built); no PWM.

## I8 — Integration loop in `main.py`

`jetbot_agent/main.py` runs the I1 state machine and dispatches I2 tools that are enabled. Enable I3–I7 only when their gates pass. **Do not** call memory stores or memory tools yet (Stage I + later memory-tool ticket).

Pass: one scripted episode: think → (optional stub tool) → stop, with motors remaining mock/stopped.

## Out of scope here

Memory upsert/query and `memory_tools.py` wait for [Stage I](09-memory.md).
