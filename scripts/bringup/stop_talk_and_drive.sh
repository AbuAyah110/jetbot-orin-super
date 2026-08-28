#!/usr/bin/env bash
# Park the chassis and leave the board safe to power off.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

systemctl --user stop jetbot-talk-and-drive.service 2>/dev/null || true
pkill -TERM -f 'scripts/bringup/talk_and_drive.py' 2>/dev/null || true
sleep 0.4
pkill -KILL -f 'scripts/bringup/talk_and_drive.py' 2>/dev/null || true
pkill -x arecord 2>/dev/null || true
pkill -x aplay 2>/dev/null || true

python3 - <<'PY' || true
import sys
from pathlib import Path
for extra in (
    Path.home() / '.local' / 'lib' / 'python3.10' / 'site-packages',
    Path('/usr/lib/python3/dist-packages'),
):
    if extra.is_dir() and str(extra) not in sys.path:
        sys.path.append(str(extra))
try:
    from jetbot.robot import Robot
    robot = Robot()
    robot.stop()
    print('motors_stopped')
except Exception as exc:
    print('motor_stop_skipped', type(exc).__name__, exc)
PY

sync
echo 'safe_to_shutdown'
