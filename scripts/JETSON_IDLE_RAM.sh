#!/usr/bin/env bash
# Reclaim idle unified memory on the Orin Nano Super before Cosmos-Reason2-2B
# loads. Locked plan: GUI off, one CSI, no extra HTTP LLM, BGE CPU later,
# KV 4096, 448² JPEG. Never motors. Never load a VLM. Never llm_build here.
#
# Defaults to a dry run. Privileged steps need sudo (no password prompt here).
#
#   ./scripts/JETSON_IDLE_RAM.sh              # measure + print plan
#   ./scripts/JETSON_IDLE_RAM.sh --apply      # user-safe steps only
#   sudo ./scripts/JETSON_IDLE_RAM.sh --apply # stop docker/snapd, swappiness 10, OLED unit
#
# Do not uninstall JetPack. Do not delete /ssd/32GB.swap. Do not kill
# nvargus-daemon unless a second camera pipeline is holding UMA.

set -uo pipefail

APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OLED_PY_REPO="$ROOT/jetbot/apps/oled_status.py"
OLED_PY_LIVE="/home/impulse110/Documents/jetbot-orin-super/jetbot/apps/oled_status.py"
if [ -f "$OLED_PY_LIVE" ]; then
  OLED_PY="$OLED_PY_LIVE"
elif [ -f "$OLED_PY_REPO" ]; then
  OLED_PY="$OLED_PY_REPO"
else
  OLED_PY="$OLED_PY_REPO"
fi
UNIT_DST=/etc/systemd/system/jetbot_oled.service
SYSCTL_FILE=/etc/sysctl.d/99-jetbot-memory.conf
IS_ROOT=0
[ "$(id -u)" -eq 0 ] && IS_ROOT=1

STOP_UNITS=(
  docker.socket
  docker.service
  nv-l4t-usb-device-mode.service
  bluetooth.service
  cups.service
  cups.socket
  cups-browsed.service
  snapd.service
  snapd.socket
)

# Never touch: nvargus-daemon, nvfancontrol, NetworkManager, ssh, JetPack.
USER_PKILL_PATTERNS=(
  'jupyter-lab'
  'jupyter-notebook'
  'ipykernel_launcher'
  'firefox'
  'chromium'
  'rviz2'
  'rviz'
)

need() {
  if [ "$APPLY" -eq 0 ]; then
    echo "  would: $*"
  else
    echo "  run: $*"
    eval "$@"
  fi
}

echo "=== Jetson idle RAM $(date -Iseconds) apply=$APPLY root=$IS_ROOT ==="
echo
echo "-- free -h --"
free -h
echo
echo "-- tegrastats (2 samples) --"
if ! timeout 3 tegrastats --interval 1000 2>/dev/null; then
  : # timeout(1) exits 124 after samples; that is success
fi
echo
echo "-- default target --"
systemctl get-default
echo
echo "-- nvpmodel --"
nvpmodel -q 2>/dev/null || echo "  (nvpmodel unavailable)"
echo
echo "-- swap (keep 32 GiB disk swap; do not add more) --"
swapon --show
echo "  vm.swappiness=$(cat /proc/sys/vm/swappiness)"
echo
echo "-- disk --"
df -h /
echo
echo "-- units of interest --"
for u in docker.socket docker.service nv-l4t-usb-device-mode.service bluetooth.service \
         cups.service snapd.service snapd.socket gdm.service; do
  printf '  %-40s %s\n' "$u" "$(systemctl is-active "$u" 2>/dev/null || echo missing)"
done
echo
echo "-- CSI / argus --"
pgrep -a nvargus || echo "  no nvargus process"
pgrep -a -f 'gst-launch|nvarguscamerasrc' || echo "  no gst/camera preview leftover"
echo
echo "-- top RSS --"
ps aux --sort=-rss | head -15
echo

echo "=== plan (locked Cosmos-Reason2-2B idle tactics) ==="
echo "  GUI: multi-user.target (already expected)"
echo "  stop/disable: docker, nv-l4t-usb-device-mode, bluetooth, cups, snapd"
echo "  leave: nvargus-daemon, 32 GiB /ssd/32GB.swap, Edge-LLM tree, Zipformer+Piper"
echo "  swappiness 10 via $SYSCTL_FILE (do not delete swap)"
echo "  OLED ExecStart by path so jetbot/__init__.py is not imported"
echo "  OLED script: $OLED_PY"
echo "  do not llm_build; do not load VLM; do not preload BGE GPU / Riva / PyTorch"
echo

if [ "$(systemctl get-default)" != "multi-user.target" ]; then
  echo "=== GUI still graphical.target ==="
  if [ "$APPLY" -eq 1 ] && [ "$IS_ROOT" -eq 1 ]; then
    systemctl set-default multi-user.target
    systemctl stop gdm.service gdm3.service 2>/dev/null || true
    systemctl isolate multi-user.target
  else
    echo "  needs sudo:"
    echo "    sudo systemctl set-default multi-user.target"
    echo "    sudo systemctl stop gdm.service"
    echo "    sudo systemctl isolate multi-user.target"
  fi
  echo
else
  echo "=== GUI already multi-user.target (no change) ==="
  echo
fi

echo "=== user leftovers (no sudo) ==="
for pat in "${USER_PKILL_PATTERNS[@]}"; do
  if pgrep -f "$pat" >/dev/null 2>&1; then
    need "pkill -f '$pat' || true"
  else
    echo "  none: $pat"
  fi
done
echo

echo "=== privileged units (sudo --apply) ==="
for u in "${STOP_UNITS[@]}"; do
  if systemctl list-unit-files "$u" >/dev/null 2>&1 && systemctl cat "$u" >/dev/null 2>&1; then
    st="$(systemctl is-active "$u" 2>/dev/null || echo inactive)"
    en="$(systemctl is-enabled "$u" 2>/dev/null || echo disabled)"
    echo "  $u  active=$st enabled=$en"
    if [ "$APPLY" -eq 1 ] && [ "$IS_ROOT" -eq 1 ]; then
      systemctl stop "$u" 2>/dev/null || true
      systemctl disable "$u" 2>/dev/null || true
      echo "    stopped+disabled $u"
    elif [ "$APPLY" -eq 1 ]; then
      echo "    skipped (need sudo): systemctl stop/disable $u"
    else
      echo "    would: systemctl stop $u && systemctl disable $u"
    fi
  else
    echo "  missing: $u"
  fi
done
echo

echo "=== swappiness (do not delete 32G swap) ==="
cur="$(cat /proc/sys/vm/swappiness)"
if [ "$cur" = "10" ] && [ -f "$SYSCTL_FILE" ]; then
  echo "  already 10 and $SYSCTL_FILE exists"
else
  if [ "$APPLY" -eq 1 ] && [ "$IS_ROOT" -eq 1 ]; then
    cat > "$SYSCTL_FILE" <<'SYSCTL'
# JetBot: keep Cosmos weights resident. 32 GiB NVMe swap is OOM safety, not a
# working set. See docs/bringup/07-cosmos-nano.md.
vm.swappiness = 10
SYSCTL
    sysctl -p "$SYSCTL_FILE"
  elif [ "$APPLY" -eq 1 ]; then
    echo "  skipped (need sudo). Current vm.swappiness=$cur"
    echo "    sudo sysctl -w vm.swappiness=10"
    echo "    sudo tee $SYSCTL_FILE <<'EOF'"
    echo "    vm.swappiness = 10"
    echo "    EOF"
    echo "    sudo sysctl -p $SYSCTL_FILE"
  else
    echo "  would persist vm.swappiness=10 (now $cur) in $SYSCTL_FILE"
    echo "  would NOT swapoff/delete /ssd/32GB.swap"
  fi
fi
echo

echo "=== OLED unit by-path (avoid python -m jetbot.apps.oled_status) ==="
if [ ! -f "$OLED_PY" ]; then
  echo "  WARNING: $OLED_PY missing; unit still rewritten to this path"
fi
if [ "$APPLY" -eq 1 ] && [ "$IS_ROOT" -eq 1 ]; then
  cat > "$UNIT_DST" <<UNIT
[Unit]
Description=JetBot PiOLED status (I2C bus 7)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=impulse110
WorkingDirectory=/home/impulse110
Environment=HOME=/home/impulse110
Environment=JETBOT_OLED_I2C_BUS=7
ExecStart=/usr/bin/python3 ${OLED_PY}
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl restart jetbot_oled.service
  echo "  installed $UNIT_DST and restarted jetbot_oled.service"
  echo "  ExecStart=/usr/bin/python3 $OLED_PY"
else
  echo "  live unit still: $(tr '\0' ' ' < /proc/$(pgrep -f 'oled_status' | head -1)/cmdline 2>/dev/null || echo '(not running)')"
  if [ "$APPLY" -eq 1 ]; then
    echo "  skipped (need sudo): install by-path unit and restart jetbot_oled"
    echo "    sudo $ROOT/scripts/JETSON_IDLE_RAM.sh --apply"
  else
    echo "  would install ExecStart=/usr/bin/python3 $OLED_PY"
    echo "  would: sudo systemctl daemon-reload && sudo systemctl restart jetbot_oled.service"
  fi
fi
echo

if [ "$APPLY" -eq 1 ] && [ "$IS_ROOT" -eq 0 ]; then
  echo "=== sudo still required for the RAM that matters (docker/snapd/sysctl/OLED) ==="
  echo "  sudo $ROOT/scripts/JETSON_IDLE_RAM.sh --apply"
  echo
fi

echo "=== after snapshot ==="
free -h
echo "done."
