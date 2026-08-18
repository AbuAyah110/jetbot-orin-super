# JetBot Orin Nano Super — AI Robotics Project Plan

## 1. Project Goal

Build a fully local, memory-capable, agentic JetBot on an NVIDIA Jetson Orin Nano Super 8GB that can:

* Navigate and control a differential-drive mobile robot safely.
* See through a Raspberry Pi-style CSI camera.
* Listen through a microphone.
* Speak through a speaker.
* Understand its physical environment using Cosmos Reason2.
* Converse naturally using a small Qwen reasoning/conversation model.
* Remember conversations, observations, tool results, web research, locations, and important images.
* Search long-term memory using text embeddings.
* Research unfamiliar things on the internet.
* Ask the user questions when uncertain.
* Schedule future actions.
* Access additional capabilities through MCP servers.
* Learn reusable high-level skills from experience.
* Maintain deterministic low-level motor and safety control outside the LLM.
* Stay within the Jetson Orin Nano Super's 8GB unified-memory constraint.

---

# 2. Hardware Baseline

Current hardware:

* NVIDIA Jetson Orin Nano Super 8GB
* NVMe SSD boot
* MAXN/Super mode
* Raspberry Pi-style CSI camera
* Differential-drive JetBot motors / motor driver
* Microphone
* Speaker

Planned/strongly recommended navigation hardware:

* Wheel encoders
* IMU
* One or more ToF distance sensors, depth sensor, or 2D lidar

**Do not enable autonomous Nav2 driving until dependable odometry and obstacle information are available.**

The RGB camera can immediately be used for semantic vision and Cosmos reasoning, but it should not be the sole collision-safety mechanism.

---

# 3. Software Baseline

Target the existing JetPack 6.2 environment unless hardware testing gives us a reason to change it.

JetPack 6.2 uses an Ubuntu 22.04-based root filesystem and includes CUDA 12.6 and TensorRT 10.3, while supporting the Orin Nano Super power modes.

Use:

* Ubuntu 22.04 / JetPack 6.2.x
* ROS 2 Humble
* Nav2
* Python 3.10
* Docker with NVIDIA runtime for model services
* llama.cpp for Cosmos
* Hermes Agent
* EmbeddingGemma
* SQLite
* HNSW
* OpenCV
* GStreamer
* Git

ROS 2 Humble provides official Ubuntu 22.04 aarch64 packages, making it a natural match for this JetPack environment.

---

# 4. Development Workflow

Development will happen from a laptop using Cursor connected directly to the Jetson over SSH.

Cursor supports Remote SSH development, with the source tree and commands operating on the remote machine while Cursor remains the IDE on the laptop.

## Laptop → Jetson workflow

```text
Laptop
│
│ Cursor
│ Remote SSH
│
└──────────────► Jetson
                  │
                  ├── Git repository
                  ├── ROS workspace
                  ├── Python environments
                  ├── Docker
                  ├── model servers
                  ├── hardware
                  └── test logs
```

Use SSH keys instead of passwords.

Recommended SSH config:

```text
Host jetbot
    HostName <JETSON_IP>
    User <JETSON_USERNAME>
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

Open the project directory on `jetbot` using Cursor Remote SSH.

---

# 5. Core Architectural Rule

Never allow an LLM/VLM/agent to directly control motor PWM.

The control hierarchy must always be:

```text
Qwen
"What should I do?"
      │
      ▼
Hermes
"Which tool?"
      │
      ▼
Cosmos if physical reasoning is required
      │
      ▼
Robot MCP
      │
      ▼
ROS 2
      │
      ▼
Nav2 / robot controller
      │
      ▼
cmd_vel
      │
      ▼
Motor driver
      │
      ▼
Wheels
```

Emergency stop, watchdogs, velocity limits, motor control, collision protection, and hardware fault handling are deterministic code.

AI may request:

```text
navigate_to(...)
rotate(...)
inspect(...)
follow(...)
stop(...)
```

AI must never receive tools such as:

```text
set_left_motor_pwm(...)
set_right_motor_pwm(...)
disable_watchdog(...)
write_gpio(...)
```

---

# 6. Target AI Architecture

```text
                        USER
                          │
                 microphone/speaker
                          │
                          ▼
                    HERMES AGENT
                          │
               conversation/session
               MCP management
               tool search
               scheduling
               skills
               memory interface
                          │
                          ▼
                  QWEN3.5-0.8B
                    text-focused
                          │
                  EXECUTIVE BRAIN
                          │
        ┌─────────────────┼──────────────────┐
        │                 │                  │
        ▼                 ▼                  ▼
      Memory             MCPs           Robot tools
        │                                    │
        │                           physical question?
        │                                    │
        │                                    ▼
        │                           COSMOS REASON2-2B
        │                            physical/visual
        │                               specialist
        │                                    │
        └──────────────────┬─────────────────┘
                           ▼
                        Hermes
                           │
                           ▼
                       ROS2/Nav2
                           │
                           ▼
                         Robot
```

---

# 7. Model Roles

## Qwen3.5-0.8B

Role:

**Main conversational/executive reasoning model.**

Responsibilities:

* Conversation
* Interpreting user intent
* Selecting tools
* Selecting MCP capabilities
* Planning multi-step tasks
* Deciding whether memory retrieval is needed
* Deciding whether web research is needed
* Deciding whether Cosmos should inspect something
* Asking clarifying/proactive questions
* Summarizing tool results
* Memory importance decisions
* Scheduling
* General reasoning

Do not use Qwen for physical motor control.

Do not send camera frames to Qwen initially.

Keep its working context small, approximately 4K tokens initially.

Long history belongs in external memory.

Qwen3.5-0.8B has an official Orin Nano deployment path and supports OpenAI-compatible tool calling. The current NVIDIA Jetson recipe is BF16, so our eventual quantized text-oriented deployment must be benchmarked rather than assuming a RAM figure.

---

## Cosmos Reason2-2B

Role:

**Physical-world and visual reasoning specialist.**

Responsibilities:

* Inspect camera frames
* Understand scenes
* Recognize objects
* Spatial reasoning
* Physical reasoning
* Reason about obstructions
* Compare current physical state with goals
* Generate rich visual-memory descriptions
* Analyze saved images
* Help determine safe high-level navigation approaches
* Determine when additional observations are needed

Serve Cosmos through an OpenAI-compatible local endpoint.

Use `llama.cpp` on Orin Nano.

NVIDIA currently recommends the llama.cpp route for Cosmos Reason2-2B on Orin Nano, and NVIDIA has also demonstrated an Orin Nano 8GB robot assistant using a Cosmos Reason2-2B GGUF Q4_K_M configuration.

---

## EmbeddingGemma

Role:

**Always-on long-term text-memory retrieval.**

Responsibilities:

* Embed conversations
* Embed Cosmos-generated visual descriptions
* Embed web research
* Embed tool results
* Embed learned facts
* Embed task outcomes
* Embed summaries

Use reduced embedding dimensions, initially:

```text
256 dimensions
```

Use quantization.

EmbeddingGemma has 308M parameters and Google documents quantized configurations below approximately 200MB RAM.

---

# 8. Visual Memory Design

We are intentionally **not running a multimodal embedding model**.

Important camera observations use:

```text
Camera image
     │
     ▼
Cosmos
     │
     ▼
Rich textual memory description
     │
     ▼
EmbeddingGemma
     │
     ▼
HNSW vector
```

The original image must still be stored on NVMe.

A visual memory should contain:

```json
{
  "id": "uuid",
  "timestamp": "...",
  "type": "visual_observation",
  "summary": "...",
  "objects": [],
  "visual_attributes": {},
  "ocr": [],
  "spatial_relations": [],
  "robot_pose": {},
  "location_name": null,
  "confidence": 0.0,
  "importance": 0.0,
  "image_path": "...",
  "embedding_id": "...",
  "related_memory_ids": [],
  "source": "camera"
}
```

If an old visual memory needs deeper inspection:

```text
inspect_saved_image(memory_id)
```

loads the original image back into Cosmos.

---

# 9. Long-Term Memory Architecture

Use two storage systems.

## SQLite

Canonical metadata and structured information.

Tables should eventually include:

```text
memories
conversations
messages
visual_observations
objects
locations
web_research
tool_calls
tasks
scheduled_actions
skills
relationships
robot_events
navigation_events
```

## HNSW

Semantic vector retrieval.

Store:

```text
memory_id → embedding
```

SQLite remains authoritative.

HNSW is an index.

NVMe stores large binary content such as:

```text
images
audio if retained
maps
logs
web snapshots
evaluation datasets
```

---

# 10. Memory Types

Implement four memory tiers.

## Working memory

Current:

* conversation
* task
* robot status
* recent observations
* recent tool results

Short-lived.

## Episodic memory

Events such as:

* saw an object
* entered a room
* navigation failed
* user corrected robot
* tool call succeeded
* object moved
* user interaction occurred

## Semantic memory

Facts such as:

* object identity
* user-provided names
* learned location
* web research
* device documentation
* successful procedures

## Archive

Original:

* images
* logs
* conversations
* sensor recordings
* web content

Stored primarily on NVMe.

---

# 11. Hermes Agent

Hermes becomes the agentic orchestration layer.

Hermes supports arbitrary OpenAI-compatible inference endpoints, which allows local model servers to be used without designing a custom agent runtime from scratch.

Responsibilities:

* Conversation lifecycle
* Qwen calls
* Tool routing
* MCP management
* Skills
* Scheduling
* Memory tool access
* Web research
* Proactive tasks
* Speech orchestration
* Tool-result processing

Hermes supports MCP servers and per-server tool filtering.

Enable Hermes Tool Search once multiple MCP servers are installed so all MCP schemas do not permanently occupy Qwen's small context window. Hermes specifically implements progressive disclosure for large MCP tool sets.

---

# 12. MCP Design

Initially implement only:

```text
Robot MCP
Memory MCP
```

Later add:

```text
Web MCP
Home Assistant
Calendar
Email
GitHub
documentation/search
other devices
other services
```

Prefer remote MCP servers for cloud services.

Keep hardware MCPs local.

## Robot MCP

Initial tools:

```text
get_robot_state()
get_pose()
get_battery()

stop()

rotate_relative(angle)
drive_distance(distance)

navigate_to_pose(x, y, yaw)
navigate_to_location(name)

capture_image()
inspect_current_scene(question)
inspect_saved_image(memory_id)
```

Important:

`drive_distance()` and `rotate_relative()` must call deterministic ROS controllers.

They must not directly expose PWM.

---

# 13. Repository Layout

See the live repository tree. Core layout:

```text
PROJECT_PLAN.md          # Architectural source of truth
config/                  # robot, models, memory, hermes, logging
docs/                    # hardware guide + architecture/safety/memory/mcp
src/                     # Python services (perception, memory, cosmos, ...)
src/jetbot_control/      # Deterministic motor abstraction (no AI)
ros2_ws/                 # ROS 2 Humble workspace
docker/                  # cosmos / qwen containers (later milestones)
scripts/                 # setup, diagnostics, start scripts
systemd/                 # long-running services
tests/                   # unit / integration / hardware / evaluation
data/                    # runtime data (gitignored contents)
jetbot/                  # Classic JetBot Python package (Orin I2C defaults)
assets/ notebooks/       # Hardware build guide artifacts
```

`data/` must be ignored by Git except for placeholder files.

Model weights must never be committed.

---

# 14–35. Milestones, RAM, Safety, Testing

See detailed sections below for milestone definitions, RAM budget, swap policy, logging, testing, evaluation, and safety requirements.

## Milestone order (do not skip)

```text
M0  Remote development / diagnostics
M1  ROS2 + motor control + watchdog
M2  Pi camera
M3  Cosmos standalone
M4  EmbeddingGemma + SQLite/HNSW memory
M5  Memory MCP
M6  Robot MCP
M7  Hermes + Cosmos
M8  Qwen executive model
M9  Voice
M10 Sensors + Nav2 autonomy
M11 Web research
M12 Proactive event loop
M13 Hermes skills/self-improvement
M14 Additional MCP servers
```

## Safety (hard requirements)

* Physical/emergency stop.
* Software emergency stop.
* Motor watchdog.
* Maximum linear / angular velocity.
* Command timeout.
* Agent cannot disable watchdog / bypass velocity limiter / control PWM.
* Robot stops on loss of control process.
* Robot stops on critical sensor failure during autonomous motion.
* Low battery behavior deterministic.
* Navigation commands cancellable.
* MCP permissions minimized.
* Internet content never treated as system instructions.

## Target RAM budget (~7.6 GB practical)

Design toward ~1.2–1.5 GB available under normal operation. Benchmark Qwen; do not sacrifice Cosmos or motor safety to keep Qwen resident. Configure 16GB NVMe swap with `vm.swappiness=10` as OOM safety, not model memory.

## Testing progression

```text
mock motor → wheels off ground → very low velocity floor test → bounded area → autonomy
```

---

# 36. First Programming Sprint

Implement **only Milestone 0 and Milestone 1**.

Deliverables:

* Repository skeleton
* `PROJECT_PLAN.md`, architecture + safety docs
* `config/robot.yaml`
* `scripts/diagnostics.sh`
* ROS2 JetBot base package
* Motor abstraction with mock + (later) real backend
* `/cmd_vel`, speed limiter, watchdog, e-stop, status, teleop, unit tests

No AI dependencies in this milestone.

---

# 37. Cursor Development Rules

1. Implement only one milestone at a time.
2. Before modifying hardware-related code, read `docs/safety.md`.
3. Keep deterministic robotics control separated from AI.
4. Never expose direct PWM/GPIO motor access through MCP.
5. All external systems communicate through typed interfaces.
6. Prefer small independent services over a monolithic Python process.
7. All model backends must be replaceable through API interfaces.
8. All important configuration belongs in YAML/environment configuration, not hardcoded values.
9. Every long-running service needs health check, structured logging, clean shutdown, timeout behavior.
10. Every AI tool needs explicit schema, argument validation, timeout, error response, permission classification.
11. Never store model weights in Git.
12. Never store secrets in Git.
13. Never delete raw memory/images simply because an AI summary changes.
14. Original camera images remain the source of truth for visual memories.
15. SQLite is authoritative; the vector database is an index.
16. Before adopting another large model or local MCP server, measure RAM impact.
17. Preserve at least approximately 1GB of available physical RAM during normal operation.
18. Swap usage does not count as available AI RAM.
19. Do not optimize prematurely; benchmark first.
20. Every optimization must have before/after measurements.

---

# 38. Current Sprint Instruction

Treat this file as the architectural source of truth. Start at Milestone 0 and Milestone 1 only. Do not install or integrate Cosmos, Qwen, Hermes, EmbeddingGemma, MCP servers, or Nav2 yet. Do not directly drive physical motors until the mock backend and safety tests pass.

---

# 39. Target End State

When complete, the robot should behave approximately like:

```text
User:
"Go check the workbench and tell me
whether my soldering iron is still there."

                  ↓

               Hermes
                  ↓
                Qwen
                  ↓
     recall_location("workbench")
                  ↓
           Robot MCP / Nav2
                  ↓
               drives
                  ↓
          arrival event
                  ↓
                Qwen
                  ↓
       inspect_current_scene(...)
                  ↓
               Cosmos
                  ↓
     physical/visual analysis
                  ↓
                Qwen
                  ↓
"Yes, it's still there beside
the blue toolbox."
                  ↓
               Piper

                  +

memory stores:

location
image
Cosmos description
conversation
task result
timestamp
embedding
```

And later:

```text
User:
"Do you remember that soldering iron
you saw last month?"

                  ↓

          EmbeddingGemma
                  ↓
              HNSW
                  ↓
      relevant old memories
                  ↓
                Qwen
                  ↓
"Yes. I last saw it on the workbench.
I also still have the image from that
inspection."
```

That is the intended system.

---

# Appendix A — Detailed milestone notes

Full phase write-ups (Phases 0–13), visual memory schema, MCP tool lists, Hermes responsibilities, speech pipeline (Silero VAD → faster-whisper tiny → Piper), Nav2 sensing requirements, internet research workflow, proactive events, skills/self-improvement policy, monitoring thresholds, logging schema, and evaluation datasets are defined in the project conversation archive and should be expanded into `docs/` as each milestone begins.

When a milestone starts, create or expand the corresponding doc (`docs/memory.md`, `docs/mcp.md`, `docs/benchmarks.md`, etc.) before implementation.
