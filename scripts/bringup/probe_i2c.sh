#!/usr/bin/env bash
# Probe I2C buses 1 and 7 (and any /dev/i2c-*). No motor PWM.
set -euo pipefail

echo "=== /dev/i2c-* ==="
ls -l /dev/i2c-* 2>/dev/null || echo "No I2C devices"

if ! command -v i2cdetect >/dev/null 2>&1; then
  echo "Install i2c-tools: sudo apt-get install -y i2c-tools" >&2
  exit 1
fi

scan() {
  local bus="$1"
  if [[ -e "/dev/i2c-${bus}" ]]; then
    echo ""
    echo "=== i2cdetect -y -r ${bus} ==="
    echo "Spec target: bus 1 addr 0x40. Classic JetBot Orin often: bus 7 addr 0x70/0x60."
    sudo i2cdetect -y -r "${bus}" || sudo i2cdetect -y "${bus}"
  else
    echo ""
    echo "=== bus ${bus}: /dev/i2c-${bus} missing ==="
  fi
}

scan 1
scan 7

echo ""
echo "Record detected addresses in docs/bringup/ (Stage B). Do not assume 0x40."
