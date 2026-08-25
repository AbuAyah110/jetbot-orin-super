#!/usr/bin/env python3
"""Wheels-up PCA9685 twitch test with hard stop and timeout.

Do not run with wheels on the ground. Leave config/robot.yaml backend=mock
until Stage B is signed off.
"""
from __future__ import annotations

import argparse
import sys
import time


# Adafruit Motor HAT PWM registers (PCA9685)
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06


def _set_pwm(bus, addr: int, channel: int, on: int, off: int) -> None:
    reg = LED0_ON_L + 4 * channel
    bus.write_i2c_block_data(
        addr,
        reg,
        [on & 0xFF, on >> 8, off & 0xFF, off >> 8],
    )


def _all_off(bus, addr: int) -> None:
    for ch in range(16):
        _set_pwm(bus, addr, ch, 0, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description='Stage B motor twitch (wheels up)')
    parser.add_argument('--bus', type=int, default=7, help='I2C bus number (probe first)')
    parser.add_argument('--addr', default='0x70', help='PCA9685 address (0x40 spec, 0x70 classic)')
    parser.add_argument('--duty', type=int, default=400, help='PCA9685 off-count (low; max 4095)')
    parser.add_argument('--seconds', type=float, default=0.4)
    parser.add_argument('--timeout', type=float, default=3.0)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument(
        '--confirm-wheels-up',
        action='store_true',
        help='Required for live I2C writes',
    )
    args = parser.parse_args()
    addr = int(args.addr, 0)
    duty = max(0, min(int(args.duty), 800))

    print(f'bus={args.bus} addr=0x{addr:02x} duty={duty} seconds={args.seconds}')
    if args.dry_run:
        print('dry-run: would twitch then all-off; no I2C writes')
        return 0
    if not args.confirm_wheels_up:
        print('Refusing live test without --confirm-wheels-up (wheels must be off the ground)', file=sys.stderr)
        return 2

    try:
        import smbus2
    except ImportError:
        try:
            import smbus as smbus2  # type: ignore
        except ImportError as exc:
            raise SystemExit('Need smbus2: pip install smbus2') from exc

    bus = smbus2.SMBus(args.bus)
    deadline = time.monotonic() + args.timeout
    try:
        bus.write_byte_data(addr, MODE1, 0x00)
        time.sleep(0.01)
        # Classic Waveshare/Adafruit JetBot uses PWM channels 0-3 for one H-bridge pair.
        print('left twitch')
        _set_pwm(bus, addr, 0, 0, duty)
        time.sleep(args.seconds)
        if time.monotonic() > deadline:
            raise TimeoutError('motor test timeout')
        _all_off(bus, addr)
        time.sleep(0.2)
        print('right twitch')
        _set_pwm(bus, addr, 2, 0, duty)
        time.sleep(args.seconds)
        if time.monotonic() > deadline:
            raise TimeoutError('motor test timeout')
        _all_off(bus, addr)
        print('stop ok')
        return 0
    except Exception:
        try:
            _all_off(bus, addr)
        except Exception:
            pass
        print('HARD STOP after error/timeout', file=sys.stderr)
        raise
    finally:
        try:
            _all_off(bus, addr)
        except Exception:
            pass
        bus.close()


if __name__ == '__main__':
    raise SystemExit(main())
