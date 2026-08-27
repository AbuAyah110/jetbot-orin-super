#!/usr/bin/env python3
"""Talk (or type) then drive: CSI JPEG → Cosmos JSON gate → I2C PWM.

Safety for this first live test:
  vx abs <= 0.22, wz abs <= 1.0, duration_s <= 0.5, then Robot.stop().
  Invalid JSON / parse fail / exception → Robot.stop().
  ALSA: SSS1629 Mic playback/sidetone OFF, Speaker 75%.
  Never cat the old Cosmos FIFO; engines load via cosmos_resident.

Interactive: Enter = 3 s mic ASR; type text; q quits with stop.
When loaded, speaks a short ready phrase, then listens.
"""

from __future__ import annotations

import argparse
import array
import json
import math
import os
import signal
import subprocess
import sys
import time
import wave
from dataclasses import replace
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from jetbot_agent.hardware.audio_interface import (  # noqa: E402
    apply_safe_mixer_baseline,
    mixer_report,
    resolve_sss1629,
)
from jetbot_agent.robot_loop.actions import (  # noqa: E402
    VX_MAX,
    WZ_MAX,
    RobotAction,
    parse_action,
)
from jetbot_agent.robot_loop.csi_jpeg import CsiJpeg448  # noqa: E402
from jetbot_agent.robot_loop.cosmos_runtime import (  # noqa: E402
    COSMOS_ENGINE_DIR,
    CosmosResidentClient,
    DEFAULT_CTRL_DIR,
    FIFO_PATH,
    RESIDENT_BIN,
    ResidentNotReady,
    kill_oneshot_fifo_holder,
    spawn_resident,
)
from jetbot_agent.robot_loop.orchestrator import (  # noqa: E402
    LoopInput,
    OneProcessOrchestrator,
)

DRIVE_MAX_TOKENS = 80
TEST_DURATION_MAX_S = 0.5
READY_PHRASE = "I'm ready for your command"
MIC_SECONDS = 3
WZ_WHEEL_SCALE = 0.4
SPEAK_PLAY_MAX_CHARS = 40


def clamp_test_action(action: RobotAction) -> RobotAction:
    """Re-clamp after the JSON gate for this first test (duration <= 0.5 s)."""
    if action is None:
        return RobotAction(kind='stop', raw_ok=False, reason='parse_fail')
    vx = action.vx
    wz = action.wz
    if vx > VX_MAX:
        vx = VX_MAX
    if vx < -VX_MAX:
        vx = -VX_MAX
    if wz > WZ_MAX:
        wz = WZ_MAX
    if wz < -WZ_MAX:
        wz = -WZ_MAX
    duration = action.duration_s
    if duration < 0.0 or not math.isfinite(duration):
        duration = 0.0
    if duration > TEST_DURATION_MAX_S:
        duration = TEST_DURATION_MAX_S
    if action.kind != 'drive':
        vx = 0.0
        wz = 0.0
        if action.kind == 'stop':
            duration = 0.0
    return replace(action, vx=vx, wz=wz, duration_s=duration)


def unicycle_wheels(vx: float, wz: float) -> tuple[float, float]:
    left = max(-1.0, min(1.0, vx - wz * WZ_WHEEL_SCALE))
    right = max(-1.0, min(1.0, vx + wz * WZ_WHEEL_SCALE))
    return left, right


def pkill_aplay() -> None:
    subprocess.run(['pkill', '-x', 'aplay'], check=False, capture_output=True)


def apply_alsa_safety() -> dict:
    ident = resolve_sss1629()
    apply_safe_mixer_baseline(ident['card_index_ephemeral'])
    report = mixer_report(ident['card_index_ephemeral'])
    lowered = report.lower()
    if 'mic' in lowered and 'playback' in lowered and '[on]' in lowered:
        apply_safe_mixer_baseline(ident['card_index_ephemeral'])
        report = mixer_report(ident['card_index_ephemeral'])
    ident = dict(ident)
    ident['mixer_snippet'] = report[:400]
    ident['speaker_cap'] = '75%'
    return ident


def write_wav(path: Path, samples, sample_rate: int) -> None:
    pcm = array.array('h')
    for sample in samples:
        value = int(max(-1.0, min(1.0, float(sample))) * 32767.0)
        pcm.append(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm.tobytes())


def play_wav_once(path: Path, playback_dev: str) -> None:
    pkill_aplay()
    try:
        subprocess.run(
            ['aplay', '-D', playback_dev, '-q', str(path)],
            check=False,
            timeout=8,
        )
    finally:
        pkill_aplay()


def speak_short(tts, text: str, playback_dev: str, wav_dir: Path) -> None:
    phrase = (text or '').strip()
    if not phrase:
        return
    if len(phrase) > SPEAK_PLAY_MAX_CHARS:
        phrase = phrase[:SPEAK_PLAY_MAX_CHARS]
    apply_alsa_safety()
    generated = tts.synthesize(phrase)
    wav = wav_dir / 'talk_and_drive_tts.wav'
    write_wav(wav, generated.samples, generated.sample_rate)
    play_wav_once(wav, playback_dev)


def capture_mic_wav(seconds: int, capture_dev: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['pkill', '-x', 'arecord'], check=False, capture_output=True)
    subprocess.run(
        [
            'arecord',
            '-D',
            capture_dev,
            '-f',
            'S16_LE',
            '-r',
            '16000',
            '-c',
            '1',
            '-d',
            str(int(seconds)),
            str(dest),
        ],
        check=True,
        timeout=int(seconds) + 8,
    )
    return dest


def read_wav_mono(path: Path) -> tuple[list[float], int]:
    with wave.open(str(path), 'rb') as handle:
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
        width = handle.getsampwidth()
        channels = handle.getnchannels()
    if width != 2:
        raise RuntimeError('expected S16_LE capture')
    samples = array.array('h')
    samples.frombytes(raw)
    floats = [s / 32768.0 for s in samples]
    if channels > 1:
        floats = floats[0::channels]
    return floats, rate


class FakeStopRuntime:
    """Smoke path: pretend Cosmos returned a gated stop. No engine map."""

    last_text = '{"action":"stop","vx":0,"wz":0,"duration_s":0,"say":"","goal":"","reason":"smoke"}'

    def generate(self, **kwargs) -> str:
        return self.last_text


class TalkDriveExecutor:
    """I2C PWM via jetbot.Robot. Always stops after the clamped duration."""

    def __init__(self, robot, *, dry_run: bool = False, tts=None, playback_dev: str = '', wav_dir: Path = REPO) -> None:
        self.robot = robot
        self.dry_run = bool(dry_run)
        self.tts = tts
        self.playback_dev = playback_dev
        self.wav_dir = wav_dir
        self._moving = False

    def is_moving(self) -> bool:
        return self._moving

    def hard_stop(self) -> None:
        self._moving = False
        if self.robot is None:
            return
        try:
            self.robot.stop()
        except Exception as exc:
            print('stop_failed', exc, file=sys.stderr, flush=True)

    def execute(self, action: RobotAction) -> None:
        try:
            action = clamp_test_action(action)
            if action.kind != 'drive' or (action.vx == 0.0 and action.wz == 0.0):
                self.hard_stop()
                if action.kind == 'speak' and action.say and self.tts is not None and self.playback_dev:
                    speak_short(self.tts, action.say, self.playback_dev, self.wav_dir)
                return
            duration = min(float(action.duration_s), TEST_DURATION_MAX_S)
            left, right = unicycle_wheels(action.vx, action.wz)
            print(
                'moving vx={0:.3f} wz={1:.3f} duration_s={2:.3f} left={3:.3f} right={4:.3f}'.format(
                    action.vx, action.wz, duration, left, right
                ),
                flush=True,
            )
            if self.dry_run or self.robot is None or duration <= 0.0:
                self.hard_stop()
                return
            self._moving = True
            self.robot.set_motors(left, right)
            time.sleep(duration)
        except Exception as exc:
            print('execute_exception', exc, file=sys.stderr, flush=True)
        finally:
            self.hard_stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ctrl-dir', default=str(DEFAULT_CTRL_DIR))
    parser.add_argument('--engine-dir', default=str(COSMOS_ENGINE_DIR))
    parser.add_argument('--no-camera', action='store_true')
    parser.add_argument('--no-pwm', action='store_true', help='Do not open jetbot.Robot (log PWM only).')
    parser.add_argument('--skip-cosmos', action='store_true', help='Fake Cosmos stop JSON (smoke).')
    parser.add_argument('--skip-asr', action='store_true')
    parser.add_argument('--skip-ready-tts', action='store_true')
    parser.add_argument(
        '--allow-speak-actions',
        action='store_true',
        help='Also play short model speak JSON (default: ready phrase only).',
    )
    parser.add_argument('--once', metavar='TEXT', help='One typed (or fake ASR) command, then quit.')
    parser.add_argument('--smoke-stop', action='store_true',
                        help='No-travel smoke: fake ASR "stop", skip Cosmos, no PWM.')
    parser.add_argument('--mic-seconds', type=int, default=MIC_SECONDS)
    parser.add_argument('--leave-loaded', action='store_true', default=True)
    return parser.parse_args()


def prompt_user(once_text: Optional[str]) -> Optional[str]:
    if once_text is not None:
        return once_text
    try:
        raw = input('Command (Enter = {0}s mic, q = quit): '.format(MIC_SECONDS))
    except EOFError:
        return 'q'
    return raw


def main() -> int:
    args = parse_args()
    if args.smoke_stop:
        args.once = args.once or 'stop'
        args.skip_cosmos = True
        args.no_pwm = True
        args.no_camera = True
        args.skip_asr = True
        args.skip_ready_tts = True

    os.environ.setdefault(
        'EDGELLM_PLUGIN_PATH',
        str(REPO / 'third_party' / 'tensorrt-edge-llm' / 'build' / 'libNvInfer_edgellm_plugin.so'),
    )

    wav_dir = REPO / 'data' / 'edgellm' / 'cosmos' / 'logs'
    wav_dir.mkdir(parents=True, exist_ok=True)

    alsa = apply_alsa_safety()
    print('alsa', json.dumps({k: alsa[k] for k in ('usb_name', 'alsa_id', 'alsa_capture', 'alsa_playback', 'sidetone_enabled', 'speaker_cap')}), flush=True)

    tts = None
    asr = None
    if not args.skip_ready_tts or args.allow_speak_actions:
        from jetbot_agent.audio.piper_tts import PiperTTS

        tts = PiperTTS(num_threads=2)
    if not args.skip_asr:
        from jetbot_agent.audio.zipformer_asr import ZipformerASR

        asr = ZipformerASR(num_threads=2)

    fifo_info = kill_oneshot_fifo_holder(FIFO_PATH)
    print('fifo_cleanup', json.dumps(fifo_info), flush=True)

    runtime = None
    resident_proc = None
    if args.skip_cosmos:
        runtime = FakeStopRuntime()
    else:
        ctrl = Path(args.ctrl_dir)
        engine_root = Path(args.engine_dir).expanduser()
        llm_dir = engine_root / 'llm'
        resident_proc = spawn_resident(
            ctrl_dir=ctrl,
            engine_dir=llm_dir,
            multimodal_dir=engine_root,
            max_tokens=DRIVE_MAX_TOKENS,
            binary=RESIDENT_BIN,
        )
        runtime = CosmosResidentClient(ctrl_dir=ctrl, jpeg_dir=wav_dir, max_tokens=DRIVE_MAX_TOKENS)
        runtime.wait_loaded(timeout_s=180.0)

    robot = None
    if not args.no_pwm:
        from jetbot import Robot

        robot = Robot()
        robot.stop()

    play_speak = bool(args.allow_speak_actions)
    executor = TalkDriveExecutor(
        robot,
        dry_run=args.no_pwm,
        tts=tts if play_speak else None,
        playback_dev=alsa['alsa_playback'] if play_speak else '',
        wav_dir=wav_dir,
    )
    orch = OneProcessOrchestrator(
        runtime,
        executor,
        drive_mode=True,
        drive_max_tokens=DRIVE_MAX_TOKENS,
    )

    camera: Optional[CsiJpeg448] = None
    if not args.no_camera:
        camera = CsiJpeg448(sensor_id=0, fps=15)
        camera.open()

    def _shutdown(_signum=None, _frame=None) -> None:
        executor.hard_stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if tts is not None and not args.skip_ready_tts:
        print('ready_tts', READY_PHRASE, flush=True)
        speak_short(tts, READY_PHRASE, alsa['alsa_playback'], wav_dir)

    print(
        'talk_and_drive ready clamps vx<={0} wz<={1} duration_s<={2}'.format(
            VX_MAX, WZ_MAX, TEST_DURATION_MAX_S
        ),
        flush=True,
    )
    print(
        'Give the robot space. Enter = {0}s mic, type a command, or q to quit.'.format(
            args.mic_seconds
        ),
        flush=True,
    )

    exit_code = 0
    try:
        while True:
            executor.hard_stop()
            typed = prompt_user(args.once)
            if typed is None or typed.strip().lower() in {'q', 'quit', 'exit'}:
                break
            speech = typed.strip()
            if speech == '':
                if asr is None:
                    print('asr_unavailable: type a command or omit --skip-asr', flush=True)
                    if args.once:
                        break
                    continue
                apply_alsa_safety()
                cap_path = wav_dir / 'talk_and_drive_mic.wav'
                print('listening {0}s'.format(args.mic_seconds), flush=True)
                capture_mic_wav(args.mic_seconds, alsa['alsa_capture'], cap_path)
                samples, rate = read_wav_mono(cap_path)
                speech = asr.transcribe(samples, rate).strip()
                print('asr', json.dumps({'text': speech}), flush=True)
                if not speech:
                    executor.hard_stop()
                    if args.once:
                        break
                    continue

            jpeg = b''
            if camera is not None:
                jpeg = camera.capture_jpeg()
                (wav_dir / 'talk_and_drive.jpg').write_bytes(jpeg)

            try:
                action = orch.tick(LoopInput(speech=speech, goal=speech, image_jpeg=jpeg or None))
                action = clamp_test_action(action)
            except Exception as exc:
                print('tick_exception', exc, file=sys.stderr, flush=True)
                executor.hard_stop()
                action = parse_action(None)

            row = {
                'speech': speech,
                'gated_action': action.kind,
                'vx': action.vx,
                'wz': action.wz,
                'duration_s': action.duration_s,
                'raw_ok': action.raw_ok,
                'reason': action.reason,
                'model_text': getattr(runtime, 'last_text', '')[:400],
            }
            print(json.dumps(row, ensure_ascii=False), flush=True)
            executor.hard_stop()
            if args.once is not None:
                break
    except Exception as exc:
        print('loop_exception', exc, file=sys.stderr, flush=True)
        executor.hard_stop()
        exit_code = 1
    finally:
        executor.hard_stop()
        pkill_aplay()
        if camera is not None:
            camera.close()
        if robot is not None:
            try:
                robot.stop()
            except Exception:
                pass
        if not args.leave_loaded and resident_proc is not None:
            try:
                resident_proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
    return exit_code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ResidentNotReady as exc:
        print('resident_not_ready', exc, file=sys.stderr)
        raise SystemExit(2)
