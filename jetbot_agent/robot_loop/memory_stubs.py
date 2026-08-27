"""BGE-small + LanceDB stubs. Import-safe without weights or optional packages."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from jetbot_agent._stage import StageNotReady

# Optional CPU deps (see jetbot_agent/requirements.txt). Never import torch.
try:
    import tokenizers as _tokenizers  # noqa: F401
except ImportError:
    _tokenizers = None

try:
    import lancedb as _lancedb  # noqa: F401
except ImportError:
    _lancedb = None


def tokenizers_available() -> bool:
    return _tokenizers is not None


def lancedb_available() -> bool:
    return _lancedb is not None


class BgeEmbedder:
    """CPU BGE-small later. No Hub fetch or ONNX load in this stub."""

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        raise StageNotReady(
            'BgeEmbedder waits for CPU validation of data/models/'
            'bge-small-en-v1.5-onnx/bge-small.onnx'
        )

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        raise StageNotReady('BgeEmbedder.encode waits for CPU weights')


class LanceMemory:
    """LanceDB handle later. Construction does not open a table."""

    def __init__(self, uri: str | None = None) -> None:
        self.uri = uri
        raise StageNotReady('LanceMemory waits for Stage I / BGE-small on CPU')

    def upsert(self, ids: Iterable[str], vectors: Sequence[Sequence[float]]) -> None:
        raise StageNotReady('LanceMemory.upsert waits for Stage I')

    def query(self, vector: Sequence[float], k: int = 4) -> list:
        raise StageNotReady('LanceMemory.query waits for Stage I')
