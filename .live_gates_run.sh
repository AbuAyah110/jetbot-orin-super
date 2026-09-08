#!/usr/bin/env bash
# Autonomous remaining five-demo live gates. Never authorize motion.
set -u
REPO="/home/impulse110/Documents/jetbot-orin-super"
cd "$REPO"
REPORT="$REPO/.live_gate_report.txt"
LOGDIR="$REPO/.live_gate_logs"
mkdir -p "$LOGDIR"
exec > >(tee -a "$REPORT") 2>&1
echo "===== START $(date -Iseconds) ====="

kill_talk() {
  pkill -TERM -f 'scripts/bringup/talk_and_drive.py' 2>/dev/null || true
  sleep 0.4
  pkill -KILL -f 'scripts/bringup/talk_and_drive.py' 2>/dev/null || true
}

echo "--- preflight processes ---"
pgrep -af talk_and_drive || echo "NO_TALK"
pgrep -af cosmos_resident || echo "NO_RESIDENT"
pgrep -af llm_inference || echo "NO_LLM"
systemctl --user is-active jetbot-talk-and-drive.service || true

echo "--- stop service for exclusive --once ---"
systemctl --user stop jetbot-talk-and-drive.service 2>/dev/null || true
kill_talk
sleep 0.5
echo "talk_and_drive after kill:"
pgrep -af 'scripts/bringup/talk_and_drive.py' || echo "NO_TALK (ok)"

PY="$REPO/.venv/bin/python"
COMMON=(--skip-asr --skip-ready-tts)

run_once() {
  local name="$1"
  shift
  echo ""
  echo "===== GATE $name $(date -Iseconds) ====="
  echo "cmd: $PY scripts/bringup/talk_and_drive.py $* "
  # exclusive: no duplicate listener
  if pgrep -f 'scripts/bringup/talk_and_drive.py' >/dev/null; then
    echo "WARN: leftover talk_and_drive; killing"
    kill_talk
    sleep 0.4
  fi
  "$PY" scripts/bringup/talk_and_drive.py "$@" >"$LOGDIR/${name}.log" 2>&1
  local rc=$?
  echo "exit=$rc"
  echo "--- log tail ---"
  tail -n 80 "$LOGDIR/${name}.log"
  echo "--- processes after $name ---"
  pgrep -af talk_and_drive || echo "NO_TALK"
  pgrep -af cosmos_resident || echo "NO_RESIDENT"
}

# (A) deictic: skip cosmos, no pwm — no motion
run_once deictic "${COMMON[@]}" --skip-cosmos --no-pwm --once 'DRIVE TOWARD THAT'

# (B) Cosmos-backed parked routes; --no-pwm extra safety; leave-loaded default True
run_once holding "${COMMON[@]}" --no-pwm --once 'WHAT AM I HOLDING'
run_once think "${COMMON[@]}" --no-pwm --once 'THINK HARD WHETHER THAT PATH IS SAFE'
run_once backpack "${COMMON[@]}" --no-pwm --once 'WHERE IS THE BLUE BACKPACK'
run_once teach_place "${COMMON[@]}" --no-pwm --once 'THIS VIEW IS THE KITCHEN CORNER'
run_once query_place "${COMMON[@]}" --no-pwm --once 'ARE WE AT THE KITCHEN CORNER'

echo ""
echo "===== remove _tmp_occ_live.py if present ====="
if [[ -f scripts/bringup/_tmp_occ_live.py ]]; then
  rm -f scripts/bringup/_tmp_occ_live.py
  echo "removed scripts/bringup/_tmp_occ_live.py"
else
  echo "not present"
fi

echo ""
echo "===== pytest tests/unit ====="
"$PY" -m pytest tests/unit -q --tb=line
echo "pytest_exit=$?"

echo ""
echo "===== git inspect ====="
git status -sb
git log -8 --oneline
echo "--- diff stat ---"
git diff --stat
git diff --cached --stat

echo "===== END $(date -Iseconds) ====="
