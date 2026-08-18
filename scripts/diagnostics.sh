#!/usr/bin/env bash
# Milestone 0 diagnostics for Jetson Orin Nano Super (also runs partially on other hosts).
set -u

section() {
  printf '\n======== %s ========\n' "$1"
}

have() {
  command -v "$1" >/dev/null 2>&1
}

section "Host"
hostname || true
uname -a || true
date || true

section "JetPack / L4T"
if [[ -f /etc/nv_tegra_release ]]; then
  cat /etc/nv_tegra_release
else
  echo "Not a Jetson (no /etc/nv_tegra_release)"
fi
if [[ -f /etc/nv_tegra_release ]]; then
  head -n 5 /etc/nv_tegra_release 2>/dev/null || true
fi
if have dpkg; then
  dpkg -l 'nvidia-jetpack*' 2>/dev/null | sed -n '1,20p' || true
fi

section "CUDA"
if have nvcc; then
  nvcc --version || true
else
  echo "nvcc not found"
fi
if [[ -f /usr/local/cuda/version.json ]]; then
  cat /usr/local/cuda/version.json || true
elif [[ -f /usr/local/cuda/version.txt ]]; then
  cat /usr/local/cuda/version.txt || true
fi

section "TensorRT"
if have dpkg; then
  dpkg -l | grep -i tensorrt | head -n 20 || echo "No TensorRT packages listed"
fi

section "nvpmodel / clocks"
if have nvpmodel; then
  sudo -n /usr/sbin/nvpmodel -q 2>/dev/null || /usr/sbin/nvpmodel -q 2>/dev/null || echo "nvpmodel query needs privileges"
else
  echo "nvpmodel not found"
fi
if have jetson_clocks; then
  sudo -n jetson_clocks --show 2>/dev/null || echo "jetson_clocks --show unavailable without sudo"
fi

section "Memory"
free -h 2>/dev/null || vm_stat 2>/dev/null || true
if [[ -f /proc/swaps ]]; then
  cat /proc/swaps
fi
if [[ -f /proc/sys/vm/swappiness ]]; then
  echo "swappiness=$(cat /proc/sys/vm/swappiness)"
fi

section "Disk"
df -h || true
lsblk 2>/dev/null || true

section "Temperature / thermal"
if have tegrastats; then
  timeout 2 tegrastats 2>/dev/null | head -n 1 || true
fi
for zone in /sys/class/thermal/thermal_zone*/temp; do
  [[ -f "$zone" ]] || continue
  type_file="$(dirname "$zone")/type"
  name="$(cat "$type_file" 2>/dev/null || echo zone)"
  millideg="$(cat "$zone" 2>/dev/null || echo 0)"
  printf '%s: %.1f C\n' "$name" "$(echo "$millideg / 1000" | bc -l 2>/dev/null || echo "$millideg")"
done

section "CPU / GPU utilization (snapshot)"
if have top; then
  top -bn1 2>/dev/null | head -n 12 || true
fi
if [[ -f /sys/devices/gpu.0/load ]]; then
  echo "gpu.0/load=$(cat /sys/devices/gpu.0/load)"
fi

section "Docker"
if have docker; then
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}' 2>/dev/null || echo "docker ps failed"
else
  echo "docker not installed"
fi

section "ROS"
if have ros2; then
  echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
  ros2 node list 2>/dev/null || echo "No ROS daemon / no nodes"
else
  echo "ros2 CLI not found (expected until ROS 2 Humble is installed on Jetson)"
fi

section "I2C bus 1 (motors / OLED)"
if have i2cdetect; then
  if [[ -r /dev/i2c-1 ]]; then
    i2cdetect -y -r 1 2>/dev/null || sudo -n i2cdetect -y -r 1 2>/dev/null || echo "i2cdetect needs access to /dev/i2c-1"
  else
    echo "/dev/i2c-1 not present"
  fi
else
  echo "i2c-tools not installed"
fi

section "Python"
python3 --version || true
python3 -c 'import sys; print(sys.executable)' || true

section "Done"
echo "diagnostics complete"
