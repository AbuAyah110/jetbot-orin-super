# Task board — locked Cosmos robot plan

GitHub: [repository](https://github.com/AbuAyah110/jetbot-orin-super) ·
[project](https://github.com/users/AbuAyah110/projects/2) ·
[issues](https://github.com/AbuAyah110/jetbot-orin-super/issues) ·
[milestones](https://github.com/AbuAyah110/jetbot-orin-super/milestones)

This file is the source of truth for the WaveShare JetBot / Jetson Orin Nano
Super deployment. Close an issue only after its verify gate passes on the
Jetson. No model may write PWM directly.

## Architecture of record

One Linux process owns the robot loop:

```text
CSI/Argus → one 448×448 JPEG pipeline → Cosmos-Reason2-2B Edge-LLM
Zipformer ASR (CPU) ──────────────────→ JSON action parser
                                              │
                     stop | drive | speak | wait | weather
                                              │
                jetbot.Robot I2C + Piper TTS (CPU)
                                              │
                         BGE-small CPU → LanceDB
```

Locked limits: `abs(vx) <= 0.22`, `abs(wz) <= 1.0`; every drive has a bounded
duration followed by stop. Extended thinking is parked-only. Cosmos uses one
in-process TensorRT Edge-LLM v0.10.0 runtime: INT4 LLM + FP16 ViT,
`maxBatchSize=1`, `maxInputLen=3072`, `maxKVCacheCapacity=4096`.

Explicitly not in the robot loop: PyTorch, `transformers`, TensorRT-LLM, Hermes
64k, ROS 2, Nav2, RViz, live Jina CLIP, live SmolVLA, Qwen2.5-VL, llama.cpp, an
extra HTTP LLM server, GPU BGE, or a second CSI pipeline. Qwen2.5-VL and
llama.cpp artifacts were deleted from this device.

Legacy thin-stack tools expect `~/jetbot-thin-stack/jetbot_vlm_agent/`. That
path is retained as a compatibility symlink into the repository's ignored
archive. Its Cosmos/BGE model paths are sibling symlinks into ignored
repository-local data, so no multi-GB ONNX tree is duplicated or tracked.

## Current status — 2026-08-27

| Stage | State | Evidence / gate |
| --- | --- | --- |
| A–E hardware + Python | Pass with documented notes | [bringup index](docs/bringup/README.md) |
| F voice | Zipformer + Piper CPU integrated; no model loaded in this pass | [voice issue list](https://github.com/AbuAyah110/jetbot-orin-super/issues?q=is%3Aissue+stage-f) |
| G1 TensorRT | Pass | [#16](https://github.com/AbuAyah110/jetbot-orin-super/issues/16) |
| G-Cosmos export | Reported complete in migrated workstation notes | [#37](https://github.com/AbuAyah110/jetbot-orin-super/issues/37) |
| G-Cosmos rsync | ONNX present on device; INT4 FFN + FP16 vision, Edge-LLM 0.10.0 | `rsync -avP --checksum ...` for revalidation |
| G-Cosmos Jetson build | **Pass** — SM87 `llm.engine` 777 MiB + `visual.engine` 785 MiB | `bash ~/jetbot-thin-stack/jetbot_vlm_agent/scripts/JETSON_BUILD.sh` |
| G-Cosmos load / RAM | **Pass** — peak 5441/7620 MB, Cosmos delta **2.88 GiB**, under the 5.0 GiB abort | `tegrastats-cosmos-load.txt` |
| One-process scaffold | Parser/camera/prompts + look-then-log resident loader | `pytest tests/unit/test_look_then_log.py` |
| Robot integration | **Talk-and-drive live** — VAD listen, no beep; PWM via `jetbot.Robot` | `scripts/bringup/talk_and_drive.py` |
| Conversation | **Live** — general Q&A, fresh-frame visual follow-ups, five-exchange persistent text memory | [12-natural-conversation.md](docs/bringup/12-natural-conversation.md) |
| Five demos | **Wired** — show-and-tell, occupancy creep, deictic refuse, parked think, eyes-first where-is, text places | [13-five-demos.md](docs/bringup/13-five-demos.md) |
| Motion request routing | **Fixed** — motion verbs veto parked question routes; speak-only replies can no longer claim movement | [12-natural-conversation.md](docs/bringup/12-natural-conversation.md) |
| Go-around detour | **Partial** — colour-grounded targets only; honest refusal otherwise | `scripts/bringup/talk_and_drive.py` |
| Monocular path gate | **Does not work** — every prompt wording is a constant; superseded by ToF for creep, bumper still absent for contact | `scripts/bringup/probe_path_gate.py` |
| Collision ToF | **Live** — VL53L0X bus 1 @ `0x29` tracks distance (≈165 mm blocked, ≈550 mm clear); this board reports ST status 11, which the driver now accepts | `.venv/bin/python scripts/bringup/probe_tof.py` |
| Wake phrase | **Not started** — auto-listen still treats every utterance as a command, so background chat gets “I was unable to understand what you said.” Try a “hello jetbot” session gate before the next live voice pass | see Try next below |
| Zipformer command words | **Patched, not fixed** — live ASR hears “find” as “fine” and “blue” as “blew”; bounded text repairs restore search, but a larger CPU ASR may be needed if new phrases keep missing the router | see To fix below |
| Gamepad + episode logs | **Not started, do not implement yet** — first P0 slice of the navigation brain (continuous `v, ω` teleop + recorder). Scenario catalog stays below | see Later gamepad / [14-navigation-brain.md](docs/bringup/14-navigation-brain.md) |
| Navigation brain | **Planned, not started** — Cosmos as task executive; CNN+GRU student outputs `[v, ω]`; PID/encoders later. Live robot stays pulse-based until P0 is explicitly started | [14-navigation-brain.md](docs/bringup/14-navigation-brain.md) |
| Resume after power | User unit enabled; 20 s delay then same loop | [11-resume-after-power.md](docs/bringup/11-resume-after-power.md) |
| Memory | **Live** — 32.5 MiB CPU INT8 BGE, LanceDB float16 vectors, explicit teach + restart recall passed | [09-memory.md](docs/bringup/09-memory.md) |

Cosmos residency measured 2026-08-27: baseline **2487 / 7620 MB**, peak with the
LLM + visual engines resident **5441 / 7620 MB**, so the Cosmos delta is
**2.88 GiB** — below the 4.3–4.7 GiB planning band and below the 5.0 GiB abort
threshold, so KV stays at 4096. The system-wide peak still includes ~1.5–1.9 GiB
of Cursor remote. Swap peak 1575/32768 MB. See
[Cosmos Nano bring-up](docs/bringup/07-cosmos-nano.md).

## Try next — wake phrase before commands

Not started. The live loop (`scripts/bringup/talk_and_drive.py --auto-listen`)
keeps the mic open, transcribes every VAD utterance with Zipformer, then routes
it. Unusable ASR speaks `UNDERSTAND_FAIL_PHRASE`; leftover speech still reaches
Cosmos conversation. Room chat therefore talks back even when nobody addressed
the robot.

Wanted behaviour:

- Stay **asleep** until a wake phrase such as “hello jetbot” (plus Zipformer
  variants: *hello jet bot*, *hey jetbot*).
- While asleep: still capture and transcribe, but **stay silent** — no
  understand-fail TTS, no Cosmos, no motors.
- After wake: short ack, then the existing command router for a short window
  (about 8–15 s after the last successful command) or until “goodbye jetbot”.
- Allow wake + command in one sentence (“hello jetbot, turn left”).
- Do **not** add a streaming keyword-spotter in this pass. Offline Zipformer
  already has text before routing; a Porcupine / openWakeWord path is a later
  option if idle CPU becomes the problem.

Likely files: `jetbot_agent/robot_loop/intents.py` (matcher + ASR repairs),
`scripts/bringup/talk_and_drive.py` (session gate before
`asr_transcript_usable`), unit tests in `tests/unit/test_voice_intents.py`.
Systemd unit stays the same.

Gate: with the service running, background speech produces no TTS; “hello
jetbot” then a command is heard; after the idle window the next unmatched
utterance is silent again.

## To fix — Zipformer command words (maybe a larger ASR)

Patched, not fixed. Do not swap models until wake phrase is tried and a new
live miss is recorded.

The small int8 Zipformer (`sherpa-onnx-zipformer-small-en-2023-06-26`, CPU)
hears “find the blue object and move towards it” as `FINE BLUE OBJECT AND
MOVED TOWARDS IT` or `FINE BLEW OBJECT AND MOVED TOWARDS IT`. Those strings
did not match the search route, so parked conversation answered “Found blue
puck. Moving toward it” with no camera and no motors.

Workaround already in `normalize_transcript`: `fine` → `find` and `blew` →
`blue` only in front of an object noun, plus past-tense “and moved towards
it” on the approach clause. “I am fine” / “the wind blew” stay untouched.

Still broken in principle: any phrasing that does not look like
`fine … object` will miss the repair; “hello jetbot” will need its own
variants; stacking more regex is not a recognizer.

If live logs after the wake-phrase pass still drop command verbs, try a
**larger CPU Zipformer / sherpa-onnx English model** in the same process
(keep Piper CPU, no PyTorch, no GPU ASR). Gate: isolated RSS still fits
beside Cosmos (~2.88 GiB delta, ~1.7 GiB free with RAG); “find the blue
object and move towards it” transcribes as find/blue/move without repairs;
`I am fine` is unchanged. Do not resurrect FastConformer/NeMo for this.

## Later — gamepad teleop and episode logs

Not started. Do not implement until wake phrase is tried. This is the first
P0 slice of the [navigation brain](docs/bringup/14-navigation-brain.md):
stick → `[v, ω]`, same mux, same recorder. Logging on today’s
duration-then-stop pulses is only a stopgap; the native action is continuous
velocity once P0 `set_velocity` exists. PyTorch stays off the robot; offline
RL is a workstation pass over ignored `data/` episodes.

Wanted control: a USB gamepad whose `v` / `ω` go through the same velocity
and duration-then-stop clamps as voice. Release-to-stop / deadman beats a
held stick; stick or e-stop beats voice. Do not reuse `jetbot/local_controller.py`
as the live path (Jupyter/pygame widget); do not drive PWM from the stick.

Wanted log, one JSONL row per tick plus a JPEG path:

| Field | Meaning | Today |
| --- | --- | --- |
| `RGB_t` | CSI 448×448 JPEG at tick `t` | Live (`CsiJpeg448`) |
| `ToF_t` | VL53L0X range + `kind` | Live (bus 1 @ `0x29`) |
| `goal_distance_t` | metres to the labelled destination | **Missing** — need a goal source (colour lock, taught place, or operator mark) |
| `goal_bearing_t` | robot-relative heading to that destination, radians | **Missing** — same gap; no odometry/goal tracker yet |
| `human_v_t` | human longitudinal command | From stick, after clamps |
| `human_ω_t` | human yaw command | From stick, after clamps |

Also store `scenario`, `episode_id`, `t`, `source=gamepad`, and stop reason.
Do not invent `goal_*` from Cosmos prose.

Record these scenarios as labelled episodes (several takes each, success and
failure both kept):

- straight hallway
- wide turn
- tight turn
- left around obstacle
- right around obstacle
- S-curve
- two obstacles
- chair legs
- narrow doorway
- approach obstacle and stop
- recover from bad angle
- back away
- turn around
- approach destination

Gate when this is eventually built: releasing the stick stops the wheels; one
short drive of each listed scenario writes a readable episode with `RGB_t`,
`ToF_t`, `human_v_t`, `human_ω_t`; `goal_*` either real or explicitly absent,
never guessed.

ROS 2 is **not** part of this pass and is not a reason to start it early.
Nav2 cannot run without odometry, a rangefinder map, and a TF tree this robot
does not have. DDS next to Cosmos (~2.88 GiB) on 8 GB, plus a second writer
to the motor HAT, would make the live voice loop worse, not safer.

The only later ROS-shaped payoff is **logging transport**, not control: a
rosbag (or timestamped `/cmd_vel` + image + range topics) so workstation
offline RL can time-align ticks. If ROS is ever added, it **subscribes** to
commands the existing Python executor already accepts. It must not open I2C
or PWM. JSONL + JPEG paths remain the default log; a bag is optional once
episodes exist.

## Later — navigation brain

Planned, **not started**. Full write-up:
[docs/bringup/14-navigation-brain.md](docs/bringup/14-navigation-brain.md).

Do not implement until the operator opens P0. Live motion stays
duration-then-stop. Voice wake-phrase remains the next live try.

Invariants: Cosmos outside the servo loop; policy outputs `[v, ω]` not PWM;
one Argus tee (448² Cosmos + ~160×120 nav); stale ToF is not clear; student
is CPU ONNX beside Cosmos, not PyTorch on the Orin; teacher/critic stay on
the workstation; encoders go under PID without changing the policy interface
at first; RAG stores skills and summaries, not 10 Hz rows.

### P0 — before serious training

- [ ] Continuous `set_velocity(v, omega)`
- [ ] Differential-drive mapping
- [ ] Left/right PWM feed-forward calibration
- [ ] Acceleration / angular-rate limiting
- [ ] Continuous ToF safety controller
- [ ] Navigation camera branch (same Argus)
- [ ] Timestamped sensor hub
- [ ] Navigation episode recorder
- [ ] Continuous `v, ω` teleoperation
- [ ] AI / human / safety mux
- [ ] Canonical obs/action schema
- [ ] Isaac and real robot share that schema

### P1 — training platform

- [ ] Visual target observation
- [ ] Isaac JetBot outputs `[v, ω]`
- [ ] Privileged simulated range rays
- [ ] Empty-room PPO goal policy
- [ ] Privileged PPO obstacle teacher
- [ ] Obstacle curriculum
- [ ] Teleoperation dataset
- [ ] Robomimic export
- [ ] BC-RNN student
- [ ] Teacher demonstration dataset
- [ ] Distill teacher → RGB+ToF student
- [ ] Export student ONNX/TensorRT
- [ ] Deploy student beside Cosmos

### P1 — hardware / control

- [ ] Wheel encoders
- [ ] Left/right wheel velocity
- [ ] Independent wheel-speed PID
- [ ] Preserve `[v, ω]` interface
- [ ] Evaluate two-front-ToF
- [ ] Optional IMU

### P2 — robustness

- [ ] Human takeover logging
- [ ] DAgger corrections
- [ ] Retrain student
- [ ] PPO fine-tune student
- [ ] Privileged critic if useful
- [ ] Domain randomization
- [ ] Measure sim-to-real success rate

### P3 — continual improvement and cognition

- [ ] Real-world navigation dataset
- [ ] Episode scoring / reward reconstruction
- [ ] IQL / TD3+BC / CQL experiments
- [ ] Compare offline policy to student
- [ ] SAC only after PPO baseline is strong
- [ ] Skill library schema
- [ ] Cosmos tasks → registered skills
- [ ] Skills and compact summaries in RAG
- [ ] Keyframe embeddings for semantic visual memory
- [ ] Raw trajectories stay out of RAG
- [ ] Place/object memory once localization exists

## Ordered execution

### 1. Workstation — export Cosmos INT4 ONNX

- [ ] Pin NVIDIA TensorRT Edge-LLM **v0.10.0**.
- [ ] Quantize/export `nvidia/Cosmos-Reason2-2B` on the x86 workstation.
- [ ] Use `--externalize-weights int4_ffn --int4-gemm-plugin-version 1`.
- [ ] Reject FP8 and NVFP4 exports; Orin Nano is SM87.
- [ ] Produce `onnx/llm/model.onnx`, external data, and visual ONNX.
- [ ] Generate and verify checksums.

Tracking: [#37](https://github.com/AbuAyah110/jetbot-orin-super/issues/37).
The obsolete Qwen/llama.cpp prototype is closed as superseded:
[#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17).

### 2. Workstation → Jetson rsync

```bash
rsync -avP --checksum ~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B-ModelOpt-INT4/onnx/ impulse110@192.168.50.65:~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/
```

The compatibility destination points to repository-local ignored data:
`data/edgellm/cosmos/onnx/`. Do not copy x86 TensorRT engines.

Gate:

```bash
test -f /home/impulse110/Documents/jetbot-orin-super/data/edgellm/cosmos/onnx/llm/model.onnx
```

### 3. Jetson — build SM87 engines — **done 2026-08-27**

```bash
export TENSORRT_EDGELLM_ROOT="$HOME/TensorRT-Edge-LLM"
export COSMOS_ONNX_DIR="$HOME/jetbot-thin-stack/cosmos-onnx"
export COSMOS_ENGINE_DIR="$HOME/jetbot-thin-stack/cosmos-engines"
bash ~/jetbot-thin-stack/jetbot_vlm_agent/scripts/JETSON_BUILD.sh
```

`scripts/bringup/JETSON_BUILD.sh` is the tracked builder, mirrored into the
thin-stack `scripts/` directory. It runs Edge-LLM v0.10.0 `llm_build` with
batch 1, input 3072, KV 4096, then `visual_build` at 64/280/280 with FP16 ViT.
`--externalize-weights` and `--int4-gemm-plugin-version` are export-time flags
that this builder rejects; INT4 is already in the ONNX. No FP8, no NVFP4, no
`--memPoolSize`, no on-device re-quantization. It exits 2 without ONNX and 4 if
the ONNX is not INT4-FFN externalized.

Gate **passed**: `data/edgellm/cosmos/engines/llm/llm.engine` (777 MiB) and
`engines/visual/visual.engine` (785 MiB); LLM engine generation 103.5 s, visual
49.1 s; Cosmos load delta 2.88 GiB against the 5.0 GiB abort threshold.

Generate length is a runtime setting, not a build flag: drive 64–96 tokens at
temperature 0, parked think 256–512, never think while `vx != 0`.

### 4. One-process parked robot integration

- [x] One import-safe CSI JPEG class; one Argus/GStreamer pipeline, 448×448.
- [x] Strict JSON parser for `stop|drive|speak|wait|weather`; invalid → stop.
- [x] Velocity/duration clamps and explicit duration-then-stop contract.
- [x] Drive and parked-think prompt suffix helpers.
- [x] `CosmosRuntime` refuses in-process TRT map; look-then-log uses `cosmos_resident`.
- [x] Import-safe BGE/LanceDB stubs; no model fetch.
- [x] Wire a resident Edge-LLM generate loop (file protocol, no HTTP) after isolated inference.
- [ ] Wire Zipformer/Piper CPU without adding another process.
- [ ] Wire only the bounded action executor to `jetbot.Robot`.
- [x] Run look-then-log (4 ticks, stop held, no motors); first driving episode still open.

### 5. CPU memory

- [x] Consolidate the existing 127 MiB BGE-small ONNX candidate under ignored `data/models/` without loading it.
- [ ] Verify that BGE ONNX on CPU only after Cosmos residency passes.
- [ ] Tokenize/embed on CPU; never install `transformers` or PyTorch.
- [ ] Persist vectors in repository-local ignored LanceDB data.
- [ ] Measure combined Cosmos + voice + BGE/LanceDB RAM before enabling recall.

## Repository layout

| Path | Git status | Purpose |
| --- | --- | --- |
| `jetbot_agent/robot_loop/` | tracked | Locked loop parser, prompts, camera, runtime/memory stubs |
| `scripts/JETSON_IDLE_RAM.sh` | tracked | Idempotent headless-memory preparation |
| `scripts/bringup/JETSON_BUILD.sh` | tracked | Locked-flag SM87 engine builder (mirrored into thin-stack) |
| `scripts/bringup/llm_build_cosmos.sh` | tracked | Repo-default wrapper around `JETSON_BUILD.sh` |
| `docs/bringup/07-cosmos-nano.md` | tracked | Inventory, RAM, rsync/build evidence |
| `third_party/tensorrt-edge-llm/` | ignored | Full Edge-LLM v0.10.0 clone/build |
| `data/edgellm/cosmos/` | ignored | ONNX, external weights, SM87 engines, logs |
| `~/jetbot-thin-stack/jetbot_vlm_agent/` | external symlink | Legacy thin-stack compatibility path; production source is not here |
| `~/jetbot-thin-stack/{cosmos-onnx,cosmos-engines,bge-small-en-v1.5-onnx}` | external symlinks | Compatibility aliases into ignored repo data; no duplicate bytes |

Never commit weights, ONNX, engines, GGUF, safetensors, Edge-LLM source/build,
virtual environments, or generated LanceDB tables.
