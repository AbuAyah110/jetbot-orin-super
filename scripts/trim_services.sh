#!/usr/bin/env bash
# Disable OS services a headless JetBot does not use, to reclaim unified memory
# for models. Measured 2026-08-26: idle baseline was RAM 2243/7620 MB during the
# Stage F gates, and the memory budget is short by roughly 1.5 GB, so the idle
# baseline is worth attacking before quantizing models harder.
#
# Defaults to a dry run. Review, then re-run with --apply.
#
#   ./scripts/trim_services.sh            # show what would change
#   sudo ./scripts/trim_services.sh --apply
#
# Reverting any line is symmetric: systemctl enable --now <unit>.

set -uo pipefail

APPLY=0
TIER2=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    --include-snapd) TIER2=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Never touch these. Listed so the reason survives the next person's cleanup.
#   nvargus-daemon   CSI camera (IMX219) capture daemon
#   nvfancontrol     thermal management
#   nvphs, nvs-service, nv-tee-supplicant, nvidia-pva-allowd, nvmemwarning
#                    Jetson platform services
#   rtkit-daemon     grants realtime priority to audio threads; the voice
#                    subsystem depends on low-latency ALSA/PipeWire scheduling
#   NetworkManager, wpa_supplicant, ssh, dbus, systemd-*, cron
#                    remote access and basic system function
#   docker, containerd
#                    kept deliberately: jetson-containers/MLC is the documented
#                    fallback runtime path for Stage G

# Tier 1: no plausible use on a headless robot.
TIER1_UNITS=(
  # printing: three separate daemons on a machine with no printer
  snap.cups.cupsd.service
  snap.cups.cups-browsed.service
  cups.service
  cups.socket
  cups-browsed.service
  lpd.service
  # no cellular modem is fitted
  ModemManager.service
  # no Bluetooth peripherals in the build
  bluetooth.service
  # NFS RPC portmapper
  rpcbind.service
  rpcbind.socket
  # mDNS service discovery; the robot is reached by IP over SSH
  avahi-daemon.service
  avahi-daemon.socket
  # desktop-oriented daemons that wake the CPU on a headless box
  packagekit.service
  kerneloops.service
  upower.service
  udisks2.service
)

# Tier 2: snapd plus the desktop snaps it mounts (chromium, two GNOME 42
# runtimes, gtk-common-themes) on a box where the GUI is disabled via
# multi-user.target. Frees disk and stops snapd refresh churn, but snap removal
# is not a one-line revert, so it is opt-in via --include-snapd.
TIER2_UNITS=(
  snapd.service
  snapd.socket
  snapd.seeded.service
  snapd.apparmor.service
)
TIER2_SNAPS=(chromium gnome-42-2204 gtk-common-themes cups)

systemd_reachable() { systemctl is-system-running >/dev/null 2>&1 || \
  [ -n "$(systemctl is-active dbus.service 2>/dev/null)" ]; }

# list-unit-files misses some legacy/generated units (lpd.service is one), so
# fall back to `systemctl cat` before deciding a unit is absent.
exists() {
  systemctl list-unit-files "$1" --no-legend --no-pager 2>/dev/null | grep -q . \
    || systemctl cat "$1" >/dev/null 2>&1
}
active() { [ "$(systemctl is-active "$1" 2>/dev/null)" = active ]; }

mem_now() { awk '/MemAvailable/{printf "%d MiB available", $2/1024}' /proc/meminfo; }

echo "before: $(mem_now)"
[ "$APPLY" -eq 1 ] || echo "(dry run — nothing will change; pass --apply to act)"
systemd_reachable || cat <<'EOF'
WARNING: systemd is not reachable from here, so the state column below is
meaningless and units may be wrongly reported as absent. Run this outside any
sandbox for accurate output.
EOF
echo

units=("${TIER1_UNITS[@]}")
[ "$TIER2" -eq 1 ] && units+=("${TIER2_UNITS[@]}")

for u in "${units[@]}"; do
  if ! exists "$u"; then
    printf '  skip     %-38s (not installed)\n' "$u"
    continue
  fi
  state=$(active "$u" && echo active || echo inactive)
  if [ "$APPLY" -eq 1 ]; then
    systemctl disable --now "$u" >/dev/null 2>&1 \
      && printf '  disabled %-38s (was %s)\n' "$u" "$state" \
      || printf '  FAILED   %-38s (was %s)\n' "$u" "$state"
  else
    printf '  would disable %-33s (currently %s)\n' "$u" "$state"
  fi
done

if [ "$TIER2" -eq 1 ] && command -v snap >/dev/null 2>&1; then
  echo
  for s in "${TIER2_SNAPS[@]}"; do
    snap list "$s" >/dev/null 2>&1 || { printf '  skip     snap %-33s (not installed)\n' "$s"; continue; }
    if [ "$APPLY" -eq 1 ]; then
      snap remove --purge "$s" >/dev/null 2>&1 \
        && printf '  removed  snap %-33s\n' "$s" \
        || printf '  FAILED   snap %-33s\n' "$s"
    else
      printf '  would remove  snap %-28s\n' "$s"
    fi
  done
fi

echo
echo "after:  $(mem_now)"
if [ "$APPLY" -eq 1 ]; then
  echo
  echo "Re-measure the true headless baseline with no editor attached:"
  echo "  ssh <jetbot> 'sleep 5; tegrastats --interval 1000' | head -3"
  echo "Stage F recorded 2243/7620 MB with an editor session running, so the"
  echo "clean figure is the one the memory budget should use."
fi
