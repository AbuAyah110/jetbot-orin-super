#!/usr/bin/env bash
# Create a 16GB swapfile on NVMe (Milestone 0). Run on Jetson with sudo.
set -euo pipefail

SWAPFILE="${SWAPFILE:-/swapfile}"
SIZE_GB="${SIZE_GB:-16}"
SWAPPINESS="${SWAPPINESS:-10}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

if [[ -f "$SWAPFILE" ]]; then
  echo "Swapfile already exists at $SWAPFILE"
else
  echo "Creating ${SIZE_GB}G swapfile at $SWAPFILE ..."
  fallocate -l "${SIZE_GB}G" "$SWAPFILE" || dd if=/dev/zero of="$SWAPFILE" bs=1G count="$SIZE_GB"
  chmod 600 "$SWAPFILE"
  mkswap "$SWAPFILE"
fi

if ! swapon --show | grep -q "$SWAPFILE"; then
  swapon "$SWAPFILE"
fi

if ! grep -q "$SWAPFILE" /etc/fstab; then
  echo "$SWAPFILE none swap sw 0 0" >> /etc/fstab
fi

sysctl -w "vm.swappiness=${SWAPPINESS}"
if ! grep -q '^vm.swappiness' /etc/sysctl.conf; then
  echo "vm.swappiness=${SWAPPINESS}" >> /etc/sysctl.conf
else
  sed -i "s/^vm.swappiness=.*/vm.swappiness=${SWAPPINESS}/" /etc/sysctl.conf
fi

echo "Swap configured:"
swapon --show
free -h
echo "swappiness=$(cat /proc/sys/vm/swappiness)"
