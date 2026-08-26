#!/usr/bin/env bash
# Create a 32GB swapfile on NVMe (JETBOT_SPEC). Run on Jetson with sudo.
set -euo pipefail

SWAPFILE="${SWAPFILE:-/swapfile}"
SIZE_GB="${SIZE_GB:-32}"
SWAPPINESS="${SWAPPINESS:-10}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root: sudo $0" >&2
  exit 1
fi

need_bytes=$((SIZE_GB * 1024 * 1024 * 1024))

# A board that already has enough swap must be left alone. This Jetson carries
# 32 GiB at /ssd/32GB.swap; creating a second file at $SWAPFILE would burn
# another 32 GiB of NVMe and add a duplicate fstab entry. Set FORCE=1 only when
# deliberately provisioning a fresh board.
# mkswap reserves a header, so an on-disk 32 GiB file reports a few KiB short of
# the nominal size. Compare in MiB with slack rather than demanding exact bytes.
need_mib=$((SIZE_GB * 1024 - 64))
have_mib=$(awk 'NR>1 {s+=$3} END {printf "%d", s/1024}' /proc/swaps)
if [[ "${FORCE:-0}" != "1" && "$have_mib" -ge "$need_mib" ]]; then
  echo "Swap already provisioned (${have_mib} MiB active, target ${SIZE_GB} GiB); leaving it alone."
  swapon --show
  echo "swappiness=$(cat /proc/sys/vm/swappiness) (spec target ${SWAPPINESS}; see docs/bringup/01-os.md)"
  exit 0
fi

recreate=false
if [[ -f "$SWAPFILE" ]]; then
  have_bytes=$(stat -c%s "$SWAPFILE" 2>/dev/null || echo 0)
  if [[ "$have_bytes" -lt "$need_bytes" ]]; then
    echo "Swapfile at $SWAPFILE is ${have_bytes} bytes; recreating at ${SIZE_GB}G"
    swapoff "$SWAPFILE" 2>/dev/null || true
    rm -f "$SWAPFILE"
    recreate=true
  else
    echo "Swapfile already exists at $SWAPFILE (${have_bytes} bytes)"
  fi
else
  recreate=true
fi

if [[ "$recreate" == true ]]; then
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
