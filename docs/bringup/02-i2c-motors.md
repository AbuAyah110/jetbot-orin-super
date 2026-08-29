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

### Both buses are on the 40-pin header

Bus 1 and bus 7 are two different pin pairs on the same header, so "not on bus 7"
does not mean "not on the header". Confirmed on this board from
`/sys/class/i2c-dev` and `/proc/device-tree/bus@0`:

| Device node | Controller | 40-pin pins | What lives there |
| --- | --- | --- | --- |
| `/dev/i2c-1` | `c240000.i2c` | 27 SDA / 28 SCL | Onboard `fusb301@25` and `ina3221@40` (the two `UU`), plus the **VL53L0X @ `0x29`** |
| `/dev/i2c-7` | `c250000.i2c` | 3 SDA / 5 SCL | Motor HAT `0x70`/`0x60` and OLED `0x3c` |

The ToF is therefore wired to pins 27/28 and shares bus 1 with two onboard
chips. Do not move it to pins 3/5: that is the motor HAT's bus, and this project
keeps ranging off the bus that carries PWM traffic. `i2cget -y 7 0x29` failing is
the expected result, not a wiring fault.

Probe ranging (no PWM):

```bash
.venv/bin/python scripts/bringup/probe_tof.py
```

Pass: five millimetre readings. `range_mm < 250` is blocked; `>= 400` is clear enough for one creep pulse. Out-of-range `8190` fails closed.

Measured 2026-08-28: a hand at ~16 cm reads `167 mm blocked`, an empty room reads
`≈550 mm clear`.

### This board reports device status 11, not 0

`RESULT_RANGE_STATUS` bits 6:3 come back as **11** on this GY-530 even for good
shots, so an earlier version that accepted only status `0` rewrote every reading
as `8190` and looked exactly like a dead laser. The driver now accepts `0`, `9`,
and `11`, and still fails closed on status `5` (analog/VCSEL hardware fail) and
on the `8190`/`8191` wrap sentinel.

When ranging looks dead, check the real cause before suspecting the emitter:

```bash
.venv/bin/python scripts/bringup/probe_tof.py   # prints status and raw_mm
```

A healthy sensor shows a nonzero `signal_rate` and an `ambient_rate`, and init
completes VCSEL reference calibration in well under a second. Status `5`, or an
init timeout, is the signature of genuinely dead analog hardware.

Prior wheels motion was done in Jupyter, not `test_motors.py`: `notebooks/basic_motion/basic_motion.ipynb` (saved cells through execution 19; comment: I2C bus **7**, addr **`0x70`**, `right_motor_alpha=-1`). This probe does **not** close the wheels-up PWM ticket.
