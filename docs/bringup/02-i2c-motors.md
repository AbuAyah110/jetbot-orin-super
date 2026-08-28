# Stage B — I2C / motors

Spec target: PCA9685 on `/dev/i2c-1` @ `0x40`. **This board may use bus 7 and `0x70`/`0x60`.** Probe first.

## Install

```bash
sudo apt-get update
sudo apt-get install -y i2c-tools
# enable i2c if needed (JetPack usually has /dev/i2c-*)
ls -l /dev/i2c-*
```

## Verify — probe (no motion)

```bash
./scripts/bringup/probe_i2c.sh
```

Record the address map in this file or a follow-up note under `docs/bringup/` (bus + hex addresses). Do not assume `0x40`.

## Verify — wheels up only

Raise the chassis so wheels cannot drive the robot. Then:

```bash
# Dry run (no I2C writes):
python3 scripts/bringup/test_motors.py --dry-run

# Live (low PWM, auto-stop). Confirm --bus/--addr from probe:
python3 scripts/bringup/test_motors.py --bus 7 --addr 0x70 --confirm-wheels-up
```

Pass: left then right twitch, then **stop**; timeout path also stops. Leave `config/robot.yaml` `backend: mock` until this is signed off.

## Address map recorded 2026-08-25 (probe only — no PWM)

`i2cdetect` as user `impulse110` (group `i2c`); `sudo` not required for this scan.

| Bus | Addresses | Notes |
| --- | --- | --- |
| 1 | `UU` @ `0x25`, `UU` @ `0x40` | Kernel driver claimed. Spec target `0x40` is present as `UU`, not a free `0x40`. |
| 7 | `0x3c`, `0x60`, `0x70` | Classic JetBot HAT: `0x70`/`0x60` PCA9685, `0x3c` OLED. |
| 1 (2026-08-28) | `UU` @ `0x25`, `UU` @ `0x40`, **`0x29`** | Front **VL53L0X** ToF (model id `0xEE`). Motors stay on bus 7. |

Probe ranging (no PWM):

```bash
.venv/bin/python scripts/bringup/probe_tof.py
```

Pass: five millimetre readings. `range_mm < 250` is blocked; `>= 400` is clear enough for one creep pulse. Out-of-range `8190` fails closed.

Prior wheels motion was done in Jupyter, not `test_motors.py`: `notebooks/basic_motion/basic_motion.ipynb` (saved cells through execution 19; comment: I2C bus **7**, addr **`0x70`**, `right_motor_alpha=-1`). This probe does **not** close the wheels-up PWM ticket.
