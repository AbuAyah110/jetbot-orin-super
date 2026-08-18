# Roadmap — Conversational JetBot (Orin Nano Super)

Phase 1 delivers a working **Orin Nano Super JetBot** (chassis, motors, camera, classic notebooks).

Phase 2 turns it into a robot you can **talk to**, with perception and memory on device.

## Phase 1 — Basic JetBot on Orin Super (current)

- [x] Repo fork with Orin I2C defaults (bus 1, addr 112)
- [x] BOM / STL / hardware docs from JetBot + Orin notes
- [x] Jetson SSD boot + MAXN SUPER links
- [ ] Final Super-capable power BOM ([power.md](power.md))
- [ ] Validate basic_motion + teleop on hardware
- [ ] Validate CSI camera notebooks

## Phase 2 — Audio I/O

- [ ] USB or I2S **microphone** (selection + mount)
- [ ] USB or I2S **speaker** / amp
- [ ] Audio bring-up notes (ALSA/Pulse, latency)
- [ ] Push-to-talk or wake-word path (TBD)

## Phase 3 — On-device AI stack

All intended to run on **Orin Nano Super** (TensorRT / optimized runtimes where possible):

| Block | Role | Status |
| --- | --- | --- |
| **ASR** | Speech → text | Planned |
| **VLM** | Vision + language understanding | Planned |
| **RAG** | Small on-device retrieval over local docs/memory | Planned |
| **TTS** | Text → speech | Planned |
| **Orchestrator** | Dialog + motor skills + safety stops | Planned |

Community precursor already in-tree: `notebooks/object_following/live_demo_nanoowl_orin.ipynb` (NanoOWL open-vocab detection + JetBot motion).

## Phase 4 — Polish

- [ ] One-command bring-up script
- [ ] Thermal / power profiles for demo vs idle
- [ ] Safety: e-stop, motor timeout, obstacle policy
- [ ] Short demo video + workshop checklist

## Design principles

1. **On-device first** — no cloud required for the core demo loop.
2. **SSD boot** — models live on NVMe.
3. **Super when demoing AI** — document power honestly; do not ship an undersized pack.
4. **Reuse JetBot skills** — motion, camera, teleop remain the substrate under the dialog stack.
