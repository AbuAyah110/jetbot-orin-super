# Motor hardware notes (pre–real-backend)

Inspection of drivers already in this repository (Milestone 1). **No live I2C probing of a robot was performed from the laptop workspace.**

## Existing classic JetBot package

Path: `jetbot/robot.py`, `jetbot/motor.py`

| Item | Value |
| --- | --- |
| Interface | Adafruit Motor HAT / PCA9685 via `Adafruit_MotorHAT` |
| Alt board | SparkFun Serial Controlled Motor Driver (`qwiic`, addr `93`) |
| Orin default I2C bus | **7** on this JetPack 6.2 board (40-pin SDA/SCL pins 3/5); bus **1** is header pins 27/28 plus onboard chips, and carries the VL53L0X ToF |
| Orin preferred motor addr | **112 (`0x70`)**; fallback **96 (`0x60`)** |
| Wheel sign | `right_motor_alpha = -1.0` so both wheels drive forward together |
| API shape | `Robot.forward/backward/left/right/stop`, per-wheel `Motor.value` in `[-1, 1]` |
| Discovery | `qwiic.scan()` at init |

Community reference: [Orin JetBot I2C notes](https://forums.developer.nvidia.com/t/using-jetbot-with-jetson-orin-nano-dev-kit/281686/8).

## ROS / AI control path (this project)

New code must **not** call PWM from MCP or LLMs.

```text
/cmd_vel → jetbot_base (limits + watchdog) → MotorDriver.set_velocity(left, right)
```

- `MockMotorDriver` — default; unit-tested; safe on any machine.
- `JetbotI2CMotorDriver` — thin wrapper around the classic package; **disabled until explicitly enabled** in `config/robot.yaml` after on-device I2C verification.

## On-Jetson checklist before enabling real backend

```bash
./scripts/diagnostics.sh
sudo i2cdetect -y -r 7
sudo i2cdetect -y -r 1
# HAT/OLED on bus 7: expect 0x70 and/or 0x60, often 0x3c for OLED
```

Raise wheels, set `backend: jetbot_i2c`, run teleop at very low speed, confirm stop + watchdog.
