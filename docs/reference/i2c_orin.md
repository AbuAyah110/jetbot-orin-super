# Orin I2C — motor bus and address

Community builders report that **JetBot Python code is reusable on Orin** when the expansion board uses the same motor driver chips as the original JetBot, with two differences:

1. **I2C bus:** `0` → **`1`**
2. **Motor chip address:** `96` (`0x60`) → **`112` (`0x70`)**

Source: [kimbring2 on NVIDIA Forums](https://forums.developer.nvidia.com/t/using-jetbot-with-jetson-orin-nano-dev-kit/281686/8).

## What this fork changes

In `jetbot/robot.py`:

- Default `i2c_bus = 1`
- Default `i2c_address = 112` (`0x70`)
- Still accepts `96` (`0x60`) if that device is what `i2cdetect` shows
- Still supports SparkFun SCMD (`93` / `0x5D`)

## Probe the bus

```bash
sudo i2cdetect -y -r 1
```

Example (Orin-style motor board):

```
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
00:          -- -- -- -- -- -- -- -- -- -- -- -- --
10: -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
...
70: 70 -- -- -- -- -- -- --
```

## Python overrides

```python
from jetbot import Robot

# Defaults for this fork
robot = Robot()

# Classic Adafruit Motor HAT address
robot = Robot(i2c_bus=1, i2c_address=96)
```

## OLED

PiOLED (SSD1306) is typically on the same bus. Upstream stats code already uses `i2c_bus=1`.
