# Stage H / I5 — navigation tools and the motion adapter

Design notes for the Stage H integration slice that finally connects the agent's
motion vocabulary to a velocity backend. **This file should be folded into
[`08-agent.md`](08-agent.md) alongside [`09-agent-i1-i2.md`](09-agent-i1-i2.md)
once I3–I8 land**; it is separate so I5 can be reviewed without touching the
Stage doc other agents are editing.

Scope: still pure software. No model is loaded, no camera or ALSA device is
opened, no I2C transaction happens, nothing is published to a real `/cmd_vel`,
and `config/robot.yaml` stays `backend: mock`. The default sink is in memory.

| Ticket | Issue | Code | Tests |
| --- | --- | --- | --- |
| I5 navigation tools + motion adapter | [#24](https://github.com/AbuAyah110/jetbot-orin-super/issues/24) | `jetbot_agent/agent/tools/navigation_tools.py`, `jetbot_agent/navigation/{motion_adapter,cmd_vel_sink,vla_seam}.py` | `tests/unit/test_navigation_tools.py`, `tests/unit/test_motion_adapter.py` |

## The layering

```text
brain (LLM/VLM)                          VLA policy (smolvla, Stage G3 — NOT built)
   │  Decision(ACT, name + args)            │  MotionIntent
   ▼                                        │
ToolRegistry                                │   deny-by-default allow-list,
   │                                        │   ACTUATE + operator_ack,
   │                                        │   per-call watchdog
   ▼                                        │
nav_drive / nav_rotate / nav_stop / nav_status
   │  drive / rotate / stop / status / limits   <- the entire vocabulary
   ▼                                        │
BoundedMotionAdapter  ◄─────────────────────┘   jetbot_agent/navigation/motion_adapter.py
   │  clamp magnitude · bound duration · arm the cmd_vel watchdog · latch e-stop
   ▼
CmdVelSink        MockCmdVelSink │ ControllerCmdVelSink │ RosCmdVelSink
   │  bounded (linear, angular) only
   ▼
jetbot_base       velocity limits + cmd_vel watchdog + e-stop
   │
   ▼
MotorDriver → wheels
```

Two rows matter more than the rest. The tool row can only say
`drive`/`rotate`/`stop`/`status`/`limits`, and the sink row can only carry an
already-bounded twist. Everything dangerous is squeezed into the adapter, which
is deterministic code an operator can read in one sitting.

## Why the adapter sits outside the tools package

It is not a style preference — an adapter inside `jetbot_agent/agent/tools/`
could never be wired to anything real, twice over:

1. **`assert_narrow_motion` would reject it.** `ToolContext.__post_init__` runs
   that check on whatever it is handed, and it refuses an object exposing any of
   `FORBIDDEN_MOTION_ATTRS` (`set_velocity`, `set_pwm`, `duty_cycle`,
   `write_byte`, `clear_estop`, `disable_watchdog`, `_driver`, `_bus`, …), any
   object with `MotorDriver` / `DiffDriveController` / `MotorController` in its
   MRO, and anything originating from `jetbot_control.motors`,
   `jetbot_agent.hardware`, `smbus`, or `Jetson.*`. `BoundedMotionAdapter` is
   accepted precisely because it holds none of those names.
2. **The AST guard would reject the import.** The parametrized test in
   `tests/unit/test_tool_safety.py` walks every `*.py` under
   `jetbot_agent/agent/tools/` and fails on an import of `jetbot_control`,
   `jetbot_base`, `smbus`, `busio`, `board`, `Jetson.*`, `RPi.*`, on any
   executable reference to `set_velocity` / `PCA9685` / `SMBus` / `GPIO`, and on
   device-path literals. `navigation_tools.py` is inside that glob and passes it
   — confirmed by `test_the_existing_ast_guard_covers_this_module` and a
   dedicated re-check in `test_navigation_tools_module_stays_above_the_boundary`
   (which additionally forbids importing `jetbot_agent.navigation` itself, so
   the tool file cannot reach *sideways* to the adapter either).

The adapter also does not subclass or leak the thing underneath it. It talks to
a duck-typed `CmdVelSink`, holds it as `_sink` (never `_driver`), and
deliberately does **not** proxy the controller's e-stop clear.

## The guarantees

### Magnitude: clamped, logged, never silently dropped

`BoundedMotionAdapter` clamps `linear` against `max_linear_velocity` (0.25 m/s)
and `angular` against `max_angular_velocity` (1.0 rad/s) from
`config/robot.yaml`, records a `ClampEvent` on `adapter.clamps`, and logs
`motion_clamped field=… requested=… applied=… limit=…`. A clamped command is
still issued — a request for 9 m/s becomes 0.25 m/s, not a refusal the model
never hears about — and the tool result reports the applied velocity so the
model sees the reduction. `NaN` is the one exception: it becomes zero, i.e. a
stop.

Whatever limits the adapter is handed are themselves bounded by
`HARD_MAX_LINEAR_VELOCITY` (0.5), `HARD_MAX_ANGULAR_VELOCITY` (2.0),
`HARD_MAX_CMD_VEL_TIMEOUT_SEC` (1.0), and `HARD_MAX_DURATION_SEC` (5.0), so a
mis-edited `config/robot.yaml` still cannot produce fast or long motion.

The tool surface is deliberately tighter still: `nav_drive` caps distance at
0.5 m and speed at 0.25 m/s, `nav_rotate` caps the angle at 180° and the rate at
1.0 rad/s, and both derive a duration capped at `MAX_NAV_DURATION_SEC` (3.0 s).
The most conservative layer is the one the model touches.

### Duration: bounded, and indefinite motion is not expressible

There is no "keep going" argument anywhere in the schemas, and every command
sets a deadline. `duration_sec` is clamped into
`[MIN_DURATION_SEC, max_duration_sec]` = `[0.05 s, 5.0 s]`, so zero, negative,
`NaN`, and `1e9` all become bounded requests rather than "until something else
stops me". `nav_drive` and `nav_rotate` derive their duration from the requested
distance/angle divided by the speed/rate, cap it, and report
`duration_capped: true` plus a `reachable_distance_m` that is honestly smaller
than what was asked for.

Tool calls do **not** block for the motion duration — that would trip the
registry's per-call watchdog. They issue a bounded command and return; the
deadline is enforced below.

### The command watchdog

`adapter.poll(now)` is what a control loop calls. It:

* halts if the e-stop latch is active;
* halts at the deadline (`duration_elapsed`);
* halts if the last publish is older than `cmd_vel_timeout_sec` (0.5 s) —
  a late poll **stops** rather than resuming, which is the same conclusion
  `jetbot_base` reaches about a stale `/cmd_vel`;
* otherwise refreshes the command once past half the window.

Stop polling and the robot stops. `adapter.status()` reflects that staleness
without mutating anything, so a caller that never polls still reads
`moving: false` after the window, and `MockCmdVelSink` models the same
staleness — a mock that "moved" forever would hide exactly the bug this layer
exists to prevent.

There are three independent timeout layers, and it is worth keeping them
straight:

| Layer | Window | Fires when |
| --- | --- | --- |
| Registry per-call watchdog (I2) | `effective_timeout(tool)` ≤ 5 s | a tool body overruns; for `ACTUATION` it calls `motion.stop()` |
| Adapter deadline + command watchdog (I5) | ≤ 5 s / 0.5 s | motion outlives its request, or a poll is late/absent |
| `jetbot_base` `cmd_vel` watchdog | 0.5 s | `/cmd_vel` goes stale — **the authoritative one** |

The first two are software conveniences in the agent process. The deterministic
watchdog in `jetbot_base` and the physical e-stop remain the real backstops.

### E-stop

`MotionEstop` is a separate, operator-owned latch, and separateness is the
point: the tool layer only ever sees the adapter, and the adapter has no
`clear_estop` attribute — both because `assert_narrow_motion` demands that and
because no tool call or model output should be able to un-stop a robot. It
latches unconditionally, matching `estop.latch: true` in `config/robot.yaml`.

Wiring it to the harness is one line, and the harness's `add_estop_hook`
contract already matches `trigger(reason)`:

```python
estop = MotionEstop()
adapter = BoundedMotionAdapter(MockCmdVelSink(), load_robot_limits(), estop=estop)
harness.add_estop_hook(estop.trigger)          # harness e-stop halts motion
```

On trigger the adapter's hook halts the sink immediately, `drive`/`rotate` raise
`MotionDenied` until an operator calls `estop.clear()`, and hooks are fail-safe
(a hook that raises is logged and never blocks the stop).

### Stop is never refused

`adapter.stop()` works while e-stopped, while a command is in flight, and even
when the sink itself raises. It is also reachable **without** the tool layer at
all: the registry issues `motion.stop()` on an actuation timeout and on
`close()`, and the e-stop latch trips the sink directly. So an agent that loses
its `ACTUATE` grant mid-motion cannot leave the robot moving —
`test_the_deterministic_stop_paths_do_not_go_through_the_tool_gate` pins that.

One honest nuance: `nav_stop` is classified `ACTUATION` because it does command
the base, so a *model-issued* `nav_stop` needs the same grant as `nav_drive`.
Downgrading its risk class would be a lie in the audit log, and it would buy
nothing, because every stop path that actually matters in an emergency bypasses
the tool gate entirely. `nav_status` is genuinely `READ_ONLY` and needs no
actuation grant — an agent forbidden from moving should still be able to see
whether the robot is moving.

## Tools

| Tool | Risk | Arguments | Notes |
| --- | --- | --- | --- |
| `nav_drive` | `ACTUATION` | `distance_m` ±0.5, `speed` 0.05–0.25, `turn_rate` ±1.0 | signed distance; negative reverses |
| `nav_rotate` | `ACTUATION` | `angle_deg` ±180, `turn_rate` 0.1–1.0 | positive turns left |
| `nav_stop` | `ACTUATION` | none | fail-safe direction only |
| `nav_status` | `READ_ONLY` | none | moving / e-stop / limits in force |

Distances and angles are **open-loop dead reckoning**. This robot has no
odometry yet (`/odom` is a stub), so the tools report `open_loop: true` and both
the requested and reachable magnitude rather than implying a closed-loop
guarantee they cannot keep. Closing that loop is Nav2 work, gated on odometry
and range sensing.

Wiring stays deny-by-default. `register_navigation_tools(registry)` only
catalogues; the operator must allow each name and grant `ACTUATE` with
`operator_ack=True`:

```python
registry = ToolRegistry(ToolContext(motion=adapter), capabilities=(Capability.READ,))
register_navigation_tools(registry)            # catalogued, not invocable
registry.allow('nav_status')                   # reads are cheap
for name in ('nav_drive', 'nav_rotate', 'nav_stop'):
    registry.allow(name)
registry.grant(Capability.ACTUATE, operator_ack=True)   # wheels-up sign-off
```

The tools are intentionally **not** re-exported from
`jetbot_agent/agent/tools/__init__.py`; that package's public surface belongs to
I8 wiring, and leaving it alone keeps I3/I4 out of a merge conflict. Import from
`jetbot_agent.agent.tools.navigation_tools` for now.

## Switching from mock to ROS

`BoundedMotionAdapter` takes a sink, and the sink is the only thing that changes:

| Sink | Backend | Use |
| --- | --- | --- |
| `MockCmdVelSink` | none | default; records twists and models the `cmd_vel` watchdog |
| `ControllerCmdVelSink` | injected `jetbot_base` controller | integration tests; `mock_controller_sink()` builds one over the **mock** motor backend |
| `RosCmdVelSink` | `geometry_msgs/Twist` on `/cmd_vel` | the real path, once ROS is sourced |

```python
# bring-up default — no hardware, no ROS
adapter = BoundedMotionAdapter(MockCmdVelSink(), load_robot_limits(), estop=estop)

# integration: tool -> adapter -> sink -> jetbot_base -> mock motor backend
adapter = BoundedMotionAdapter(mock_controller_sink(), load_robot_limits())

# real path: the caller owns the ROS lifecycle, the adapter never does
adapter = BoundedMotionAdapter(RosCmdVelSink(node), load_robot_limits(), estop=estop)
```

`ControllerCmdVelSink` imports nothing from `jetbot_base` — the controller is
duck typed on `command_twist` / `stop` / `tick` / `status_dict` — and
`mock_controller_sink()` constructs `MockMotorDriver` directly rather than going
through the backend factory, so no configuration value can turn it into an I2C
session. `RosCmdVelSink` imports `geometry_msgs` lazily inside its constructor,
requires a node the caller already initialised, and never calls `rclpy.init()`,
spins, or creates a node of its own. A subprocess test asserts that importing
`jetbot_agent.navigation` leaves `rclpy`, `geometry_msgs`, `jetbot_base`,
`jetbot_control`, `smbus`, and friends absent from `sys.modules`.

Going live is still gated on [`docs/safety.md`](../safety.md): mock tests green,
I2C scan healthy, **wheels off the ground**, operator on the power switch.

## The VLA seam (smolvla is NOT implemented)

Stage G3 has not delivered smolvla, so `jetbot_agent/navigation/vla_seam.py`
defines the seam and refuses to fake the model:
`UnavailableVlaPolicy.propose()` raises `StageNotReady`.

What the seam fixes is *where* such a policy attaches. A VLA is not a tool and
not a brain; it is another producer of motion intents, and it enters at the
adapter — below the tool boundary, above the sink:

```python
intent = policy.propose(observation)           # MotionIntent(kind, linear, angular, …)
apply_intent(adapter, intent, min_confidence=0.8)
```

`apply_intent` translates an intent into exactly the `drive`/`rotate`/`stop`
calls a tool makes, so a VLA-produced intent gets the same magnitude clamping,
the same bounded duration, the same command watchdog, and the same latched
e-stop refusal. **There is no faster path to the wheels.** Intents additionally
carry a `confidence` and `apply_intent` takes a floor, because an autonomous
producer with no operator in the loop should clear a bar that a deliberate tool
call does not; below the floor it stops the base and raises `MotionDenied`.

If a later ticket prefers the model inside the agent's decision loop, the right
shape is an ordinary `ActuationTool` with a closed schema next to `nav_drive`.
Either way the intent lands on the adapter, never on a controller or a driver.

## Verify

```bash
.venv/bin/python -m pytest tests/unit -q                          # whole suite
.venv/bin/python -m pytest tests/unit/test_navigation_tools.py -q  # I5 tool gate
.venv/bin/python -m pytest tests/unit/test_motion_adapter.py -q    # I5 adapter gate
.venv/bin/python -m pytest tests/unit/test_tool_safety.py -q       # I2 boundary still holds
```

Tests add the repo root, `src`, and `ros2_ws/src/jetbot_base` to `sys.path`
themselves, matching `tests/unit/test_motor_and_controller.py`, so the
`pyproject.toml` pytest config needs no change.

## Deliberately deferred

* **Real `/cmd_vel` traffic.** `RosCmdVelSink` is written and unit-tested
  against a stubbed node and message type, but it has never published to a live
  graph; `rclpy` is not importable in `.venv`. First live run needs a sourced
  ROS 2 environment, `jetbot_base` up on `backend: mock`, and wheels off the
  ground.
* **smolvla.** Stage G3. The seam is defined and tested; the policy is not.
* **Closed-loop distance and heading.** Needs odometry, then Nav2.
* **The `ActionSink` that routes harness actions into `ToolRegistry.dispatch()`**
  and re-exporting the nav tools from the tools package — Stage H / I8.
* **A background poller.** `poll()` is driven by the caller's control loop today.
  Whoever owns the agent's main loop in I8 should call it, or the base simply
  stops every 0.5 s — which is the safe failure, not the dangerous one.
