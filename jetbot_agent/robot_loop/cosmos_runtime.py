"""Cosmos-Reason2-2B runtime: engines exist; mapping is a resident Edge-LLM process.

Constructing :class:`CosmosRuntime` still refuses to dlopen TensorRT in this
interpreter. Look-then-log talks to ``scripts/bringup/cosmos_resident`` over a
control directory (not the one-shot FIFO).
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

from jetbot_agent._stage import StageNotReady

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = REPO_ROOT / 'data' / 'edgellm' / 'cosmos'
THIN_ENGINES = Path.home() / 'jetbot-thin-stack' / 'cosmos-engines'
COSMOS_ENGINE_DIR = THIN_ENGINES if THIN_ENGINES.is_dir() else (DEFAULT_WORKSPACE / 'engines')
LLM_ENGINE = COSMOS_ENGINE_DIR / 'llm'
VISUAL_ENGINE = COSMOS_ENGINE_DIR / 'visual'
ONNX_LLM = DEFAULT_WORKSPACE / 'onnx' / 'llm' / 'model.onnx'
DEFAULT_CTRL_DIR = Path('/tmp/jetbot_cosmos_loop')
FIFO_PATH = Path('/tmp/cosmos_reason2_resident.fifo')
EDGELLM_ROOT = Path.home() / 'TensorRT-Edge-LLM'
LLM_INFERENCE = EDGELLM_ROOT / 'build' / 'examples' / 'llm' / 'llm_inference'
RESIDENT_BIN = REPO_ROOT / 'scripts' / 'bringup' / 'cosmos_resident'


def engines_present(workspace: Optional[Path] = None) -> bool:
    if workspace is not None:
        llm = Path(workspace) / 'engines' / 'llm'
        if not llm.is_dir():
            llm = Path(workspace) / 'llm'
    else:
        llm = LLM_ENGINE
    if not llm.is_dir():
        return False
    return any(llm.glob('*.engine'))


class CosmosRuntime:
    """Raises :class:`StageNotReady` unless a resident control dir is already loaded.

    Direct construction never maps TensorRT engines in-process.
    """

    def __init__(self, workspace: Optional[Path] = None, ctrl_dir: Optional[Path] = None) -> None:
        self.workspace = Path(workspace) if workspace is not None else DEFAULT_WORKSPACE
        self.ctrl_dir = Path(ctrl_dir) if ctrl_dir is not None else DEFAULT_CTRL_DIR
        if not engines_present(self.workspace) and not engines_present(COSMOS_ENGINE_DIR):
            raise StageNotReady(
                'CosmosRuntime waits for Jetson engines at {0}'.format(self.workspace / 'engines' / 'llm')
            )
        loaded = self.ctrl_dir / 'loaded'
        if not loaded.is_file():
            raise StageNotReady(
                'Cosmos engines exist but this process will not map TensorRT; '
                'start scripts/bringup/cosmos_resident and pass ctrl_dir={0}'.format(self.ctrl_dir)
            )
        self._client = CosmosResidentClient(ctrl_dir=self.ctrl_dir)

    def generate(self, *args, **kwargs):
        return self._client.generate(*args, **kwargs)


class ResidentNotReady(RuntimeError):
    """Resident binary missing, load timed out, or request failed."""


def kill_oneshot_fifo_holder(fifo: Path = FIFO_PATH) -> dict:
    """Kill any process holding the one-shot FIFO. Never read/cat the FIFO."""
    info = {'fifo': str(fifo), 'killed': [], 'unlinked': False, 'note': 'did_not_cat'}
    if not fifo.exists():
        info['missing'] = True
        return info
    pids = set()
    try:
        listed = subprocess.run(
            ['lsof', '-t', str(fifo)],
            capture_output=True,
            text=True,
            check=False,
        )
        for token in listed.stdout.split():
            if token.isdigit():
                pids.add(int(token))
    except OSError:
        pass
    try:
        fuser = subprocess.run(
            ['fuser', str(fifo)],
            capture_output=True,
            text=True,
            check=False,
        )
        for token in (fuser.stdout + ' ' + fuser.stderr).replace(':', ' ').split():
            if token.isdigit():
                pids.add(int(token))
    except OSError:
        pass
    for pid in sorted(pids):
        try:
            os.kill(pid, signal.SIGKILL)
            info['killed'].append(pid)
        except OSError as exc:
            info.setdefault('kill_errors', []).append('{0}:{1}'.format(pid, exc))
    try:
        fifo.unlink()
        info['unlinked'] = True
    except OSError as exc:
        info['unlink_error'] = str(exc)
    return info


def spawn_resident(
    *,
    ctrl_dir: Path,
    engine_dir: Path,
    multimodal_dir: Path,
    max_tokens: int = 80,
    binary: Optional[Path] = None,
) -> Optional[subprocess.Popen]:
    """Start cosmos_resident if this ctrl dir is not already loaded."""
    ctrl_dir = Path(ctrl_dir)
    ctrl_dir.mkdir(parents=True, exist_ok=True)
    loaded = ctrl_dir / 'loaded'
    pid_file = ctrl_dir / 'pid'
    if loaded.is_file() and pid_file.is_file():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return None
        except (ValueError, OSError):
            pass
    bin_path = Path(binary) if binary is not None else RESIDENT_BIN
    if not bin_path.is_file():
        raise ResidentNotReady('missing resident binary {0}'.format(bin_path))
    log_path = ctrl_dir / 'resident.log'
    log_handle = log_path.open('w', encoding='utf-8')
    env = os.environ.copy()
    plugin = REPO_ROOT / 'third_party' / 'tensorrt-edge-llm' / 'build' / 'libNvInfer_edgellm_plugin.so'
    env.setdefault('EDGELLM_PLUGIN_PATH', str(plugin))
    for stale in ('in.json', 'in.ready', 'out.json', 'out.ready', 'quit'):
        try:
            (ctrl_dir / stale).unlink()
        except OSError:
            pass
    try:
        loaded.unlink()
    except OSError:
        pass
    proc = subprocess.Popen(
        [
            str(bin_path),
            '--engineDir',
            str(engine_dir),
            '--multimodalEngineDir',
            str(multimodal_dir),
            '--ctrlDir',
            str(ctrl_dir),
            '--maxGenerateLength',
            str(int(max_tokens)),
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(REPO_ROOT),
    )
    pid_file.write_text(str(proc.pid) + '\n', encoding='utf-8')
    return proc


class CosmosResidentClient:
    """File-protocol client for cosmos_resident. One JPEG path per generate."""

    def __init__(
        self,
        *,
        ctrl_dir: Path = DEFAULT_CTRL_DIR,
        jpeg_dir: Optional[Path] = None,
        max_tokens: int = 80,
        timeout_s: float = 60.0,
    ) -> None:
        self.ctrl_dir = Path(ctrl_dir)
        self.jpeg_dir = Path(jpeg_dir) if jpeg_dir is not None else self.ctrl_dir
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 64:
            self.max_tokens = 64
        if self.max_tokens > 96:
            self.max_tokens = 96
        self.timeout_s = float(timeout_s)
        self.last_text = ''
        self._seq = 0

    def wait_loaded(self, timeout_s: float = 180.0) -> None:
        deadline = time.monotonic() + timeout_s
        loaded = self.ctrl_dir / 'loaded'
        while time.monotonic() < deadline:
            if loaded.is_file():
                return
            time.sleep(0.2)
        raise ResidentNotReady('engines did not map within {0}s at {1}'.format(timeout_s, self.ctrl_dir))

    def generate(
        self,
        *,
        system: str,
        user_text: str,
        image_jpeg: Optional[bytes],
        max_tokens: int,
    ) -> str:
        self.ctrl_dir.mkdir(parents=True, exist_ok=True)
        self.jpeg_dir.mkdir(parents=True, exist_ok=True)
        tokens = int(max_tokens)
        if tokens < 64:
            tokens = 64
        if tokens > 96:
            tokens = 96
        self._seq += 1
        jpeg_path = None
        if image_jpeg:
            jpeg_path = self.jpeg_dir / 'resident_frame_{0:04d}.jpg'.format(self._seq)
            jpeg_path.write_bytes(image_jpeg)
        user_content = []
        if jpeg_path is not None:
            user_content.append({'type': 'image', 'image': str(jpeg_path)})
        user_content.append({'type': 'text', 'text': user_text})
        payload = {
            'batch_size': 1,
            'temperature': 0.0,
            'top_p': 1.0,
            'top_k': 1,
            'max_generate_length': tokens,
            'warmup': 0,
            'enable_thinking': False,
            'requests': [
                {
                    'messages': [
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': user_content},
                    ]
                }
            ],
        }
        in_json = self.ctrl_dir / 'in.json'
        in_ready = self.ctrl_dir / 'in.ready'
        out_json = self.ctrl_dir / 'out.json'
        out_ready = self.ctrl_dir / 'out.ready'
        for path in (out_json, out_ready, in_ready):
            try:
                path.unlink()
            except OSError:
                pass
        tmp = in_json.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(payload), encoding='utf-8')
        tmp.replace(in_json)
        in_ready.write_text('go\n', encoding='utf-8')

        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if out_ready.is_file() and out_json.is_file():
                break
            time.sleep(0.05)
        else:
            raise ResidentNotReady('generate timed out after {0}s'.format(self.timeout_s))

        data = json.loads(out_json.read_text(encoding='utf-8'))
        responses = data.get('responses') or []
        text = ''
        if responses:
            text = responses[0].get('output_text') or ''
        self.last_text = text
        return text
