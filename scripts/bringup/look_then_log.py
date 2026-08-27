#!/usr/bin/env python3
"""Look-then-log loop: CSI 448² JPEG → Cosmos drive-mode JSON → log stop.

No PWM, no jetbot.Robot, no TTS playback. Invalid JSON is logged as stop.
Holds stop for the whole inference. Leaves the Cosmos resident process loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jetbot_agent.robot_loop.actions import parse_action
from jetbot_agent.robot_loop.csi_jpeg import CsiJpeg448
from jetbot_agent.robot_loop.log_executor import LogOnlyExecutor
from jetbot_agent.robot_loop.orchestrator import LoopInput, OneProcessOrchestrator
from jetbot_agent.robot_loop.cosmos_runtime import (
    COSMOS_ENGINE_DIR,
    CosmosResidentClient,
    DEFAULT_CTRL_DIR,
    FIFO_PATH,
    RESIDENT_BIN,
    ResidentNotReady,
    kill_oneshot_fifo_holder,
    spawn_resident,
)

DRIVE_MAX_TOKENS = 80
TICK_PERIOD_S = 1.0
DEFAULT_TICKS = 4
LOG_DIR = REPO / 'data' / 'edgellm' / 'cosmos' / 'logs'


def tegrastats_line(timeout_s: float = 0.45) -> str:
    try:
        proc = subprocess.run(
            ['timeout', f'{timeout_s:.2f}', 'tegrastats', '--interval', '100'],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return 'tegrastats_unavailable: {0}'.format(exc)
    lines = [ln.strip() for ln in (proc.stdout or '').splitlines() if ln.strip()]
    return lines[-1] if lines else (proc.stderr or '').strip()[:240] or 'tegrastats_empty'


def iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec='milliseconds')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticks', type=int, default=DEFAULT_TICKS)
    parser.add_argument('--ctrl-dir', default=str(DEFAULT_CTRL_DIR))
    parser.add_argument('--engine-dir', default=str(COSMOS_ENGINE_DIR))
    parser.add_argument('--leave-loaded', action='store_true', default=True)
    parser.add_argument('--no-camera', action='store_true')
    parser.add_argument('--log-jsonl', default=str(LOG_DIR / 'look_then_log.jsonl'))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ticks = max(3, min(5, int(args.ticks)))
    ctrl = Path(args.ctrl_dir)
    engine_root = Path(args.engine_dir).expanduser()
    llm_dir = engine_root / 'llm'
    log_path = Path(args.log_jsonl)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault(
        'EDGELLM_PLUGIN_PATH',
        str(REPO / 'third_party' / 'tensorrt-edge-llm' / 'build' / 'libNvInfer_edgellm_plugin.so'),
    )

    fifo_info = kill_oneshot_fifo_holder(FIFO_PATH)
    print('fifo_cleanup', json.dumps(fifo_info), flush=True)

    resident_proc = spawn_resident(
        ctrl_dir=ctrl,
        engine_dir=llm_dir,
        multimodal_dir=engine_root,
        max_tokens=DRIVE_MAX_TOKENS,
        binary=RESIDENT_BIN,
    )
    client = CosmosResidentClient(ctrl_dir=ctrl, jpeg_dir=log_path.parent, max_tokens=DRIVE_MAX_TOKENS)
    client.wait_loaded(timeout_s=180.0)
    resident_pid = None
    pid_file = ctrl / 'pid'
    if pid_file.is_file():
        try:
            resident_pid = int(pid_file.read_text().strip())
        except ValueError:
            resident_pid = resident_proc.pid if resident_proc is not None else None
    elif resident_proc is not None:
        resident_pid = resident_proc.pid

    executor = LogOnlyExecutor()
    orch = OneProcessOrchestrator(
        client,
        executor,
        drive_mode=True,
        drive_max_tokens=DRIVE_MAX_TOKENS,
    )

    camera: Optional[CsiJpeg448] = None
    if not args.no_camera:
        camera = CsiJpeg448(sensor_id=0, fps=15)
        camera.open()

    rows = []
    try:
        for tick in range(1, ticks + 1):
            t0 = time.monotonic()
            executor.execute(parse_action('{"action":"stop"}'))
            jpeg = b''
            if camera is not None:
                jpeg = camera.capture_jpeg()
                (log_path.parent / 'look_then_log_tick{0}.jpg'.format(tick)).write_bytes(jpeg)

            action = orch.tick(LoopInput(image_jpeg=jpeg, goal='look-then-log parked'))
            tegra = tegrastats_line()
            parse_failed = (not action.raw_ok) or action.reason == 'parse_fail'
            row = {
                'tick': tick,
                'timestamp': iso_now(),
                'gated_action': action.kind,
                'vx': action.vx,
                'wz': action.wz,
                'duration_s': action.duration_s,
                'parse_failed': parse_failed,
                'raw_ok': action.raw_ok,
                'reason': action.reason,
                'say_empty': action.say == '',
                'holding_stop': True,
                'infer_s': round(time.monotonic() - t0, 3),
                'tegrastats': tegra,
                'model_text': getattr(client, 'last_text', ''),
            }
            rows.append(row)
            with log_path.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
            print(json.dumps(row, ensure_ascii=False), flush=True)

            elapsed = time.monotonic() - t0
            if elapsed < TICK_PERIOD_S:
                time.sleep(TICK_PERIOD_S - elapsed)
    finally:
        if camera is not None:
            camera.close()

    summary = {
        'ticks': len(rows),
        'log': str(log_path),
        'resident_pid': resident_pid,
        'left_loaded': bool(args.leave_loaded),
        'fifo_cleanup': fifo_info,
    }
    (log_path.parent / 'look_then_log_summary.json').write_text(
        json.dumps({'summary': summary, 'rows': rows}, indent=2) + '\n',
        encoding='utf-8',
    )
    print('SUMMARY', json.dumps(summary), flush=True)
    if not args.leave_loaded and resident_pid is not None:
        try:
            os.kill(resident_pid, signal.SIGTERM)
        except OSError:
            pass
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ResidentNotReady as exc:
        print('resident_not_ready', exc, file=sys.stderr)
        raise SystemExit(2)
