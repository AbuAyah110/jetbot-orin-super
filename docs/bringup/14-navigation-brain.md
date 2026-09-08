# Navigation brain — planned, not implemented

This is the intended upgrade from pulse-based visual command demos to a
continuously controlled, trainable navigator. **It is not live.** The running
robot remains `scripts/bringup/talk_and_drive.py`: Zipformer → Python routes →
optional Cosmos on a 448² JPEG → duration-then-stop PWM.

Do not start P0 until the operator says so. Nearer live voice work (wake
phrase, Zipformer command words) is independent and stays on
[TASKBOARD.md](../../TASKBOARD.md).

Gamepad episode logging on the board is the first slice of **P0 teleop +
recorder** here, not a competing design. ROS 2 remains optional bag transport
later; it must not open I2C.

**Guiding rule:** Cosmos decides *what* to do; the skill system picks a
capability; the navigation policy decides *how* to move right now; classical
control makes the hardware track `[v, ω]`. Cosmos stays out of the
high-frequency motor loop. The privileged PPO teacher and critic stay on the
workstation. The student deploys as CPU ONNX (or TRT) and does not install
PyTorch next to Cosmos on 8 GB.

One `nvarguscamerasrc` only. Navigation RGB is a second **branch** from the
same Argus tee (~160×120, 10–20 FPS), not a second CSI process. Cosmos keeps
448² JPEG.

---

## 1. Project goal

Finished robot should:

- Accept high-level commands through ASR / Cosmos.
- Identify or receive a navigation target.
- Continuously navigate toward the target.
- Steer around obstacles rather than stop / turn / pulse.
- Modulate speed from obstacle proximity.
- Recover from poor approaches.
- Use RGB plus ToF (later dual ToF, encoders, optional IMU).
- Support human teleoperation and takeover.
- Record navigation for BC, DAgger, PPO, and offline RL.
- Share the same observation / action interface in Isaac Lab and on the
  physical JetBot.

---

## 2. Hierarchy

```text
USER → ASR → COSMOS-REASON2-2B (semantic / task)
                 ↓
           TASK EXECUTIVE
                 ↓
          navigation goal
                 ↓
        NAVIGATION POLICY (CNN + GRU student)
                 ↓
               [v, ω]
                 ↓
            CONTROL MUX  (AI / Human / Stop)
                 ↓
         SMOOTHING + SAFETY
                 ↓
        DIFFERENTIAL DRIVE
                 ↓
     LEFT PID / RIGHT PID  (eventually)
                 ↓
              PWM → MOTORS
```

---

## 3. Invariants

- Cosmos may emit `NAVIGATE_TO_OBJECT`, `SEARCH_FOR_OBJECT`, `STOP`,
  `FOLLOW_TARGET`. It must not servo `forward / left / right` pulses for
  autonomous navigation.
- Canonical action is `set_velocity(v, omega)`, never raw PWM from the
  policy, never discrete pulses as the native nav action.
- Isaac Lab and the robot expose equivalent `NavigationObservation` /
  `NavigationAction`.
- Invalid or stale ToF is not “clear”.
- Encoders sit **under** `[v, ω]` (PID). The student need not consume
  encoder ticks at first; adding PID must not require a retrain.
- RAG stores skills and compact episode summaries, never 10 Hz samples, and
  never motor commands as retrieved prose.
- Mode A (today’s voice pulse loop) and mode B (policy) share one PWM owner.
  Do not run Cosmos generate and the student on the GPU together; in policy
  mode leave Cosmos unloaded or parked.

Canonical observation concept:

```text
rgb, tof_left_or_center, tof_right_optional,
goal_signal, previous_action, timestamp
```

Canonical action:

```text
linear_velocity, angular_velocity
```

---

## 4. P0 — Continuous motion API

Replace PWM → duration → stop as the nav primitive with continuous
`set_velocity(v, omega)` and differential-drive mapping

`v_L = v − ωL/2`, `v_R = v + ωL/2`

(`L` = effective wheel separation). Until encoders exist, map desired wheel
speed to PWM with **separate left/right feed-forward curves**.

**Gate:** continuous straight, gentle left/right arcs, changing-radius arcs,
and a smooth stop, with no stop between command updates.

---

## 5. P0 — Motion smoothing

Rate-limit linear and angular acceleration / deceleration and cap `v`, `ω`.
Log both `requested_v/ω` and `applied_v/ω`.

**Gate:** a step to max turn becomes a ramp, not an instant PWM jump.

---

## 6. P0 — Continuous ToF safety

ToF runs in the control loop, not only between pulses.

- Observation: policy sees the range.
- Hard layer: progressive slowdown → no more forward → stop.
- Distinguish valid out-of-range from I2C error, stale, or invalid. Failures
  must not authorize motion.

---

## 7. P0 — Navigation camera branch

Keep Cosmos 1280×720 → 448² JPEG. Tee the same Argus pipeline to ~160×120 RGB
at 10–20 FPS. Navigation frames **must not** wait for the 0.6 s post-motion
settle used by Cosmos/colour gates. The student learns from frames captured
**during** motion.

Physical aim, height, and exposure lock are a separate board item (**To fix —
camera aim and exposure**). Calibrate the mount before treating nav RGB as
trustworthy. Do not add a second CSI camera to work around a bad tilt.

---

## 8. P0 — Timestamped sensor hub

Monotonic timestamp on RGB, ToF(s), goal signal, previous action, later
encoders/IMU. Record requested vs applied `v/ω`. Latency must be measurable
for sim-to-real.

---

## 9. P0 — Episode recorder

Per tick / episode at least: `episode_id`, timestamp, RGB, ToF, goal,
previous action, AI proposed `v/ω`, human proposed `v/ω`, applied `v/ω`,
PWM L/R, controller source (`AI|HUMAN|SCRIPT|SAFETY|EMERGENCY`), success /
failure / intervention, task type, target description.

Scenario catalog already on the board: straight hallway, wide/tight turn,
left/right around obstacle, S-curve, two obstacles, chair legs, narrow
doorway, approach and stop, recover from bad angle, back away, turn around,
approach destination.

Export path later: Robomimic-style HDF5. Do not invent `goal_*` from Cosmos
prose.

---

## 10. P0 — Continuous teleoperation

Joystick Y → `v`, X → `ω`, same action space as the policy. Deadman /
release-to-stop. Record every teleop episode. Stick or e-stop beats AI.

---

## 11. P0 — Control mux

Priority: emergency stop → hard safety → human takeover → AI navigation →
manual default. During human intervention, still log the AI’s proposed
command (DAgger).

---

## 12. P1 — Cosmos as task executive

Replace scripted `forward / arc_left` plans with high-level tasks:
`NAVIGATE_TO_VISIBLE_TARGET`, `SEARCH_FOR_TARGET`, `APPROACH_TARGET`,
`STOP_NAVIGATION`. Cosmos initializes, reports, and reasons; the policy
servos.

---

## 13. P1 — Visual goal (no global localization yet)

`target_visible`, `target_x` in `[-1, 1]`, optional `target_y`, apparent
scale, confidence. Colour segmentation can bootstrap red/blue/green.

---

## 14–22. P1/P2 — Isaac and learning

Isaac JetBot must emit the same `[v, ω]` and deployable obs. Privileged rays
/ pose for the **teacher only**.

Order:

1. Empty-room PPO goal policy (privileged OK) to prove action/reward.
2. Privileged PPO teacher (MLP, raycasts, curriculum: empty → clutter →
   doorways).
3. Human teleop dataset on the real robot.
4. BC-RNN student (small CNN → embedding, concat ToF + target + last action
   → GRU → `[v, ω]`), Robomimic.
5. Distill teacher actions onto student obs in sim.
6. Export student ONNX/TRT; deploy **beside** (not inside) Cosmos; policy
   mode should not need Cosmos generate.
7. DAgger on human corrections.
8. PPO fine-tune the student (privileged critic allowed in sim).
9. Domain randomization only after nominal nav works.

No on-device online RL exploration as the first learning path.

---

## 23. P1 hardware — wheel encoders

Under the policy: `[v, ω]` → wheel speeds → left/right PID → PWM. Optional
later: add encoder velocity to student obs (that **does** need fine-tune).

---

## 24. P1 hardware — dual front ToF

Two VL53L0X, slightly outward (~10–20°). Shared default `0x29`: unique
addresses via XSHUT. Sequential ranging if optical crosstalk. Does not
replace RGB or lidar. One-ToF remains valid until the second unit exists.

---

## 25. Optional IMU

Yaw rate / accel after or with encoders. Not required for P0.

---

## 26–30. Embeddings, RAG, skills

- **Nav embedding:** 160×120 → small CNN → 128–256-D → GRU. This *is* the
  student vision path.
- **Semantic visual memory:** separate keyframe embeddings + captions for
  Cosmos retrieval. Never PWM.
- **Skill library in RAG:** retrieve which skill (`search_for_object`,
  `approach_visible_target`) to run; the skill points at a policy or
  Python behavior, not at free prose.
- **Episode summaries** in RAG; raw 10 Hz logs stay in the training corpus.

---

## 31. Future offline learning

Corpus: human demos, student rollouts, teacher rollouts, corrections,
real success and failure. Later IQL / TD3+BC / CQL on a workstation.

---

## 32. Runtime layout (end state)

Cosmos-Reason2-2B (parked semantic) → task executive → skill/RAG →
CNN+GRU student → `[v, ω]` → safety/smoothing → PID → motors.

Teacher and critic do not live permanently on the JetBot.

---

## 33. Checklists

Copied to [TASKBOARD.md](../../TASKBOARD.md) under **Later — navigation
brain**. Keep the board boxes in sync when work starts.

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

---

## 34. End-state gates

**Motion:** continuous motion, smooth arcs, controlled accel, reproducible
wheel speeds with PID.

**Navigation:** visible targets, left/right avoid, corridors, doorways,
clutter, recovery, later moving obstacles.

**Learning:** teleop set, BC-RNN, privileged teacher, distillation, DAgger,
PPO fine-tune, randomization, offline real-world learning.

**Cognition:** Cosmos on semantics; RAG on skills/episodes; learned control
outside the LLM; nav stays live while Cosmos is slow.
