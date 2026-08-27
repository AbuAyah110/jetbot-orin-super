"""Cosmos-Reason2-2B Edge-LLM runtime stub. Does not load engines."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from jetbot_agent._stage import StageNotReady

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = REPO_ROOT / 'data' / 'edgellm' / 'cosmos'
LLM_ENGINE = DEFAULT_WORKSPACE / 'engines' / 'llm'
VISUAL_ENGINE = DEFAULT_WORKSPACE / 'engines' / 'visual'
ONNX_LLM = DEFAULT_WORKSPACE / 'onnx' / 'llm' / 'model.onnx'


def engines_present(workspace: Optional[Path] = None) -> bool:
    root = Path(workspace) if workspace is not None else DEFAULT_WORKSPACE
    llm = root / 'engines' / 'llm'
    if not llm.is_dir():
        return False
    return any(llm.glob('*.engine'))


class CosmosRuntime:
    """Raises :class:`StageNotReady` until SM87 engines exist *and* a loader lands.

    Constructing this class never maps TensorRT engines. Callers must not treat
    a successful import as a loaded VLM.
    """

    def __init__(self, workspace: Optional[Path] = None) -> None:
        self.workspace = Path(workspace) if workspace is not None else DEFAULT_WORKSPACE
        self._ready = engines_present(self.workspace)
        if not self._ready:
            raise StageNotReady(
                'CosmosRuntime waits for Jetson engines at {0} '
                '(rsync ONNX then scripts/bringup/llm_build_cosmos.sh)'.format(
                    self.workspace / 'engines' / 'llm'
                )
            )
        raise StageNotReady(
            'Cosmos engines exist at {0} but the in-process Edge-LLM loader '
            'is not wired yet; do not load TensorRT from this stub'.format(
                self.workspace / 'engines' / 'llm'
            )
        )

    def generate(self, *args, **kwargs):
        raise StageNotReady('CosmosRuntime.generate is not implemented')
