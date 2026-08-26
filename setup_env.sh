#!/usr/bin/env bash
# OS initialization for JetBot Orin Super bring-up (Stage A).
# Headless target, MAXN when available, 32 GB NVMe swap, swappiness=10.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

echo "=== Headless (multi-user.target) ==="
systemctl set-default multi-user.target

if systemctl list-unit-files | grep -q '^nvzramconfig.service'; then
  echo "=== Disable ZRAM (prefer NVMe swapfile) ==="
  systemctl disable nvzramconfig.service || true
fi

if command -v nvpmodel >/dev/null 2>&1; then
  echo "=== nvpmodel (query; Super mode ID varies by flash) ==="
  nvpmodel -q || true
  echo "Set MAXN SUPER with: nvpmodel -m <id from nvpmodel -q>"
fi

echo "=== Swap 32 GB, swappiness=10 ==="
SIZE_GB=32 SWAPPINESS=10 "${ROOT}/scripts/setup_swap.sh"

echo "=== setup_env.sh done ==="
echo "Verify: ${ROOT}/scripts/diagnostics.sh"
