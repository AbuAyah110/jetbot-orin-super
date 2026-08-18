# Safety — JetBot Orin Super

Read this before changing any motor, GPIO, or autonomy-related code.

## Hard requirements

1. **Physical emergency stop** available to the operator.
2. **Software emergency stop** stops motors immediately and rejects motion until cleared.
3. **Motor watchdog**: if `/cmd_vel` (or controller heartbeats) stop, motors stop.
4. **Velocity limits**: clamp linear and angular commands in deterministic code.
5. **Command timeout**: stale commands do not keep the robot moving.
6. **AI cannot**:
   - set left/right PWM directly
   - write GPIO for motors
   - disable the watchdog
   - bypass the velocity limiter
7. Robot **stops** if the control process dies.
8. During autonomy, robot **stops** on critical sensor failure (once sensors exist).
9. Low-battery behavior is **deterministic** (not LLM-decided).
10. Navigation / motion commands are **cancellable**.
11. MCP permissions stay minimal; internet content is **data**, never system policy.

## Testing progression

```text
mock motor
  → wheels raised off ground
  → very low velocity floor test
  → bounded test area
  → autonomous operation
```

**Never** first-test AI- or ROS-commanded motors with the robot free on a desk.

## Milestone 1 defaults (`config/robot.yaml`)

| Parameter | Intent |
| --- | --- |
| `max_linear_velocity` | Hard cap (m/s) |
| `max_angular_velocity` | Hard cap (rad/s) |
| `cmd_vel_timeout` | Watchdog window |
| `backend: mock` | Default until hardware validated |

## Real hardware

Do not enable `backend: jetbot_i2c` until:

1. Mock unit tests pass.
2. `scripts/diagnostics.sh` and I2C scan look healthy.
3. Wheels are off the ground for the first live test.
4. Operator is ready to cut power.
