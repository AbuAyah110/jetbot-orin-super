# Task board — JetBot Orin Super staged bring-up

GitHub Project: **(URL filled after `gh project create`)**

Repo: [AbuAyah110/jetbot-orin-super](https://github.com/AbuAyah110/jetbot-orin-super)

Each row is one issue. Close the issue only when the **Verify** command passes on the Jetson.

Spec: [JETBOT_SPEC.md](JETBOT_SPEC.md) · Procedures: [docs/bringup/README.md](docs/bringup/README.md)

**Sequence:** A → B → C → D → E → F → G → **H Agent (I1–I8)** → **I Memory**. Agent integration is before memory. I1–I8 are integration tickets (Stage H), not Stage I. `gh` is not installed on this Jetson; this file is the issue source of truth until a GitHub Project exists.

## Status snapshot (2026-08-25)

Classic JetBot notebooks already ran for A/B/C. This pass only **re-verified** OS/I2C/camera with no reinstall and **no motor PWM**. Stage D audio hardware was brought up separately (sidetone off; ALSA by device name).

| Stage | Board status |
| --- | --- |
| A OS / env | Verified with notes (swap path `/ssd/32GB.swap`, swappiness 60) |
| B I2C / motors | I2C probe pass; wheels-up PWM not re-run (see notebooks) |
| C CSI camera | Argus 1-frame pass + notebook preview |
| D Audio | HW verified; voice models are Stage F |
| E–G | Open |
| H Agent I1–I8 | Open (before memory) |
| I Memory | Open (after I8) |

---

## Stage A — OS / env

| Issue | Verify |
| --- | --- |
| Confirm JetPack / L4T, NVMe as `/`, MAXN SUPER | `./scripts/diagnostics.sh` |
| Headless `multi-user.target` | `systemctl get-default` |
| 32 GB swap + `vm.swappiness=10` | `swapon --show`; `cat /proc/sys/vm/swappiness` |

Notes from 2026-08-25: L4T R36.4.4, `/` is `nvme0n1p1`, `NV Power Mode: MAXN_SUPER`, default `multi-user.target`. Swap is **32 GiB** at `/ssd/32GB.swap` (fstab), not `/swapfile`. Swappiness is **60**, not the spec target 10. Do not redo the OS install; optionally tune swappiness later.

## Stage B — Motors / I2C

| Issue | Verify |
| --- | --- |
| Enable I2C, scan buses, record address map | `./scripts/bringup/probe_i2c.sh` |
| PCA9685 talk (`0x40` or discovered `0x70`/`0x60`) | probe output lists expected addr |
| Wheels-up left/right/stop + timeout stop | `python3 scripts/bringup/test_motors.py --confirm-wheels-up` |

Notes from 2026-08-25 (**probe only, no PWM**): `/dev/i2c-1` and `/dev/i2c-7` present. Bus **1**: `UU` at `0x25` and `0x40`. Bus **7**: `0x3c` (OLED), `0x60`, `0x70` (classic HAT). Prior motion evidence: `notebooks/basic_motion/basic_motion.ipynb` (saved run: bus 7, `0x70`, `right_motor_alpha=-1`). Do not close the wheels-up PWM ticket from this probe alone.

## Stage C — CSI camera

| Issue | Verify |
| --- | --- |
| IMX219 CSI0 capture | `./scripts/bringup/test_csi_camera.sh` |

Notes from 2026-08-25: `nvargus-daemon` active; `gst-launch-1.0 nvarguscamerasrc num-buffers=1` reached EOS. Sensor modes include 3280×2464 (IMX219). Prior notebook: `notebooks/camera/csi_camera_test.ipynb` (`camera ok (224, 224, 3)`). Also exercised: `notebooks/object_following/live_demo_nanoowl_orin.ipynb`.

## Stage D — Audio

| Issue | Verify |
| --- | --- |
| Waveshare ALSA record + playback | `./scripts/bringup/test_alsa.sh` |

Hardware: Waveshare / Solid State System USB PnP Audio **SSS1629**. Identify the endpoint by ALSA **USB device name**, never a hardcoded card index (`plughw:2,0` was observed once). Keep **sidetone off**. Sequential capture-then-playback until F2 AEC. Voice stack (Stage F): FastConformer + FastPitch/HiFi-GAN + WebRTC APM (+ optional RNNoise).

## Stage E — Python skeleton

| Issue | Verify |
| --- | --- |
| venv + `jetbot_agent` imports | `./scripts/bringup/test_python_skeleton.sh` |

Pass 2026-08-25: `.venv` created with `virtualenv` (`python3-venv` apt not present). PyYAML 6.0.3. Import smoke ok. See [docs/bringup/05-python-skeleton.md](docs/bringup/05-python-skeleton.md).

## Stage F — Voice

| Exact issue title | Verify |
| --- | --- |
| F1: Identify SSS1629 ALSA device and establish safe mixer baseline | Name-resolved 16 kHz mono capture, then separate low-volume playback; sidetone off |
| F2: Validate WebRTC APM acoustic echo and noise front end | Clean/noisy/echo fixtures show speech preservation plus measured noise/echo reduction |

F1 pass 2026-08-25: `plughw:CARD=Device,DEV=0` (Solid State System USB PnP); sidetone off; capture 80%; speaker 20%. See [docs/bringup/04-audio.md](docs/bringup/04-audio.md).

F2 pass 2026-08-25: `pywebrtc-audio==0.1.0` offline fixtures; NS 8.2×; AEC 236×; no live duplex. See [docs/bringup/06-voice.md](docs/bringup/06-voice.md).
| F3: Benchmark optional RNNoise residual denoising | Reproducible APM-only vs APM+RNNoise 48 kHz/resampling A/B report |
| F4: Validate NVIDIA FastConformer ASR on Orin | One WAV → transcript; latency, real-time factor, utilization, and peak RAM/VRAM recorded |
| F5: Validate NVIDIA FastPitch and HiFi-GAN TTS on Orin | Text → mel → WAV → safe one-shot playback; latency, real-time factor, utilization, and peak RAM/VRAM recorded |
| F6: Validate AEC-protected duplex voice pipeline and feedback watchdog | Duplex turn-taking passes only after F2; watchdog and combined latency/memory evidence recorded |

I6 (agent voice tools) may one-shot after F4/F5; **duplex tools wait for F6**. If F1–F4 are open, I6 stays a stub.

### Stage F issue bodies

Use these bodies verbatim if the GitHub issues/project do not exist yet:

**F1: Identify SSS1629 ALSA device and establish safe mixer baseline**

> Identify the Waveshare/Solid State System USB PnP Audio SSS1629 by ALSA USB device name. The observed endpoint was `plughw:2,0`, but production must not hardcode card 2. Set capture near the safe 80% baseline (hardware maximum +31 dB), playback low, and sidetone OFF. Because sidetone previously caused a dangerous loud feedback loop, run sequential capture then playback only until F2 passes.
>
> Gate: capture a non-silent, unclipped 16 kHz mono WAV from the name-resolved device; stop capture; play it once at low volume; record device identity, resolved endpoint, and mixer state.

**F2: Validate WebRTC APM acoustic echo and noise front end**

> Process 16 kHz mono in 10 ms frames through WebRTC APM high-pass filtering, NS, AGC, VAD, and AEC. Feed AEC the time-aligned far-end/reference audio sent to the speaker. Test recorded clean-speech, noisy-speech, and speaker-echo fixtures; save before/after artifacts and record latency, CPU load, and peak unified memory.
>
> Gate: no frame discontinuities or clipping; intelligible speech is preserved; measured noise and playback echo are reduced; no feedback occurs; configuration and measurements are attached.

**F3: Benchmark optional RNNoise residual denoising**

> Compare APM alone with APM + RNNoise on identical fixtures. RNNoise is optional residual denoising, not AEC; document its native 48 kHz, 480-sample frame requirement and the 16↔48 kHz resampling/latency cost.
>
> Gate: attach reproducible A/B quality, latency, CPU, and peak-memory results and document adopt/reject. This optional issue does not block F4/F5.

**F4: Validate NVIDIA FastConformer ASR on Orin**

> In a supported NVIDIA runtime, run one known 16 kHz mono WAV through FastConformer independently of the agent. Do not assume TensorRT-Edge-LLM supports ASR. Consider model-specific export/TensorRT optimization only after baseline validation.
>
> Gate: attach a non-empty intelligible transcript plus exact model/precision/runtime, cold/warm latency, input duration, real-time factor, utilization, and peak unified RAM/VRAM. Assess 8 GB feasibility from measurements without inventing a threshold.

**F5: Validate NVIDIA FastPitch and HiFi-GAN TTS on Orin**

> In a supported NVIDIA runtime, run text → FastPitch mel → HiFi-GAN waveform → WAV independently of the agent. With capture stopped and sidetone off, perform one low-volume `aplay`. Do not assume TensorRT-Edge-LLM supports TTS.
>
> Gate: attach an intelligible, unclipped WAV plus exact models/precision/runtime, cold/warm latency, generated duration, synthesis real-time factor, utilization, and peak unified RAM/VRAM. Assess 8 GB feasibility from measurements.

**F6: Validate AEC-protected duplex voice pipeline and feedback watchdog**

> Only after F2 passes, connect capture → APM → VAD → FastConformer → agent boundary → FastPitch → HiFi-GAN → AEC reference tap + ALSA playback. Add immediate mute/stop behavior for clipping, runaway level, stale/missing AEC reference, backlog, and operator stop.
>
> Gate: repeated turn-taking/interruption tests complete without feedback, runaway volume, clipping, stale-reference operation, or unbounded queues. Attach end-to-end latency, watchdog evidence, and combined peak unified RAM/VRAM with measured 8 GB headroom.

## Stage G — TensorRT (isolated)

| Issue | Verify |
| --- | --- |
| TensorRT / Edge-LLM runtime present | `./scripts/diagnostics.sh` TensorRT section |
| Qwen2.5-VL-3B INT4 dummy forward | dummy I/O, no camera loop |
| smolvla-jetbot dummy I/O (no PWM) | dummy I/O |
| Nemotron embed dummy vector | dummy I/O |

I7 (VLM/engine tools) starts only after these dummy tickets.

## Stage H — Agent integration (before memory)

Implement I1–I8 as **separate issues**. Do not open Stage I memory until I8’s loop runs without memory tools. Safety: LLM never PWM; motion only via limited `cmd_vel`/smolvla + watchdog. Details: [docs/bringup/08-agent.md](docs/bringup/08-agent.md).

| Exact issue title | Verify |
| --- | --- |
| I1: Hermes harness skeleton / state machine (no tools) | Scripted idle→think→act→stop; no tool imports |
| I2: Tool interface + safety (LLM never PWM) | Mocked tool-call cannot set PWM; watchdog stays outside the model |
| I3: Vision tools (OCR/grounding stubs calling camera) | One CSI/Argus frame; placeholder OCR/grounding |
| I4: Search tools (Tavily) | Fail closed without API key; results are data not policy |
| I5: Navigation tools wrapping smolvla/cmd_vel with watchdog (dummy first) | Dummy cmd_vel starts and timeout-stops; no I2C writes |
| I6: Voice tools (after F5/F6; stub if F1–F4 open) | No duplex until F6; one-shot ASR/TTS only after F4/F5 |
| I7: VLM/engine tools (after Stage G) | Dummy VLM/engine forward; no PWM |
| I8: Integration loop in main.py | One scripted episode; memory tools not wired |

### Stage H issue bodies

**I1: Hermes harness skeleton / state machine (no tools)**

> Implement `jetbot_agent/agent/hermes_harness.py` as a state machine only (idle, listen, think, act, speak, stop). No tool registry, camera, motors, or network.
>
> Gate: scripted prompt steps through states and reaches stop without importing tool modules.

**I2: Tool interface + safety (LLM never PWM)**

> Add a tool protocol (name, schema, timeout, allow-list). Motion may only hit the limited motor path (`cmd_vel` / smolvla wrapper + watchdog + velocity limits). Reject PCA9685/PWM/I2C from the LLM. Align with `docs/safety.md` and `docs/architecture.md`.
>
> Gate: unit test that a mocked LLM tool-call cannot set PWM or disable the watchdog.

**I3: Vision tools (OCR/grounding stubs calling camera)**

> `vision_tools.py` stubs capture via the Stage C CSI/Argus path and return a frame plus placeholder OCR/grounding. No motor side effects.
>
> Gate: one capture yields a non-empty image; placeholders are explicit.

**I4: Search tools (Tavily)**

> `search_tools.py` wraps Tavily. Without `TAVILY_API_KEY`, fail closed. Search hits are untrusted data, never robot policy. Key may be added later.
>
> Gate: unset key → disabled; optional live query returns JSON only.

**I5: Navigation tools wrapping smolvla/cmd_vel with watchdog (dummy first)**

> High-level navigate/rotate/stop only. First pass: dummy `cmd_vel` / mock driver with timeout stop. Real smolvla I/O after Stage G. Never expose PWM on the tool surface. Keep `backend: mock` until Stage B wheels-up PWM is signed off.
>
> Gate: dummy motion command stops on timeout; logs show no I2C writes.

**I6: Voice tools (after F5/F6; stub if F1–F4 open)**

> One-shot ASR after F4, one-shot TTS after F5, duplex only after F6 (which requires F2 AEC). Sidetone stays off. If F1–F4 are not ready, ship a documented no-op stub and do not call ALSA from the agent.
>
> Gate: tools refuse duplex until F6; no untreated simultaneous capture/playback.

**I7: VLM/engine tools (after Stage G)**

> Wrap Qwen2.5-VL / engine dummy forwards from Stage G. Isolated I/O only; no camera-motor loop; no PWM.
>
> Gate: one dummy vision+text or documented skip if the engine is not built.

**I8: Integration loop in main.py**

> `jetbot_agent/main.py` runs the I1 harness and I2 dispatcher. Enable I3–I7 only when their gates pass. Do not wire Chroma/SQLite or memory tools (Stage I + follow-on).
>
> Gate: one scripted episode think → optional stub tool → stop; motors remain mock/stopped.

## Stage I — Memory (after agent)

| Issue | Verify |
| --- | --- |
| Chroma persist upsert/query | local persist round-trip |
| SQLite facts put/get | `data/memory/facts.db` |

Follow-on (after this stage, not I1–I8): memory tools for the Hermes allow-list.

## Columns (GitHub Project)

Backlog → Ready → In progress → Blocked (hardware) → Verify on Jetson → Done
