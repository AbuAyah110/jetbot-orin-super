"""CPU INT8 BGE-small embeddings and a small local LanceDB memory.

This module deliberately has no torch, transformers, CUDA, or TensorRT path.
The embedder is lazy so importing the robot loop does not allocate model RAM.
"""

from __future__ import annotations

import hashlib
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from jetbot_agent._stage import StageNotReady

# Optional CPU deps (see jetbot_agent/requirements.txt). Never import torch.
try:
    import tokenizers as _tokenizers  # noqa: F401
except ImportError:
    _tokenizers = None

# talk_and_drive appends JetPack's system dist-packages for ``gi`` and JetBot.
# That directory also contains pandas compiled for NumPy 1, while the verified
# voice venv uses NumPy 2. PyArrow treats pandas as optional, so hide only the
# system directory during LanceDB import and restore it immediately afterward.
_system_dist_paths = [
    path
    for path in sys.path
    if path == '/usr/lib/python3/dist-packages'
]
for _path in _system_dist_paths:
    sys.path.remove(_path)
try:
    try:
        import lancedb as _lancedb  # noqa: F401
    except ImportError:
        _lancedb = None
finally:
    sys.path.extend(
        path for path in _system_dist_paths if path not in sys.path
    )

try:
    import numpy as _np
    import onnxruntime as _ort
    import pyarrow as _pa
except ImportError:
    _np = None
    _ort = None
    _pa = None

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = REPO_ROOT / 'data' / 'models' / 'bge-small-en-v1.5-onnx'
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / 'bge-small-int8.onnx'
DEFAULT_TOKENIZER_PATH = DEFAULT_MODEL_DIR / 'tokenizer.json'
DEFAULT_LANCE_URI = REPO_ROOT / 'data' / 'memory' / 'lancedb'
EMBEDDING_DIM = 384
EMBEDDING_MAX_TOKENS = 64
DEFAULT_TOP_K = 4
RAG_MAX_CHARS = 1400
MAX_RECALL_DISTANCE = 0.85


def tokenizers_available() -> bool:
    return _tokenizers is not None


def lancedb_available() -> bool:
    return _lancedb is not None


class BgeEmbedder:
    """BGE-small ONNX on CPU only, with normalized 384-d embeddings."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        tokenizer_path: str | Path = DEFAULT_TOKENIZER_PATH,
        num_threads: int = 2,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        if _tokenizers is None or _ort is None or _np is None:
            raise StageNotReady(
                'BGE CPU runtime needs tokenizers, onnxruntime, and numpy'
            )
        missing = [
            str(path)
            for path in (self.model_path, self.tokenizer_path)
            if not path.is_file()
        ]
        if missing:
            raise StageNotReady('BGE artifacts missing: ' + ', '.join(missing))
        self._tokenizer = _tokenizers.Tokenizer.from_file(
            str(self.tokenizer_path)
        )
        self._tokenizer.enable_truncation(max_length=EMBEDDING_MAX_TOKENS)
        self._tokenizer.enable_padding(pad_id=0, pad_token='[PAD]')
        options = _ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(num_threads))
        options.inter_op_num_threads = 1
        options.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = _ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=['CPUExecutionProvider'],
        )
        if self._session.get_providers() != ['CPUExecutionProvider']:
            raise StageNotReady('BGE must run on CPUExecutionProvider only')
        outputs = self._session.get_outputs()
        if not outputs or outputs[0].shape[-1] != EMBEDDING_DIM:
            raise StageNotReady('BGE output is not 384-dimensional')

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        clean = [' '.join((text or '').split())[:2000] for text in texts]
        if not clean:
            return []
        encoded = self._tokenizer.encode_batch(clean)
        input_ids = _np.asarray([item.ids for item in encoded], dtype=_np.int64)
        attention_mask = _np.asarray(
            [item.attention_mask for item in encoded], dtype=_np.int64
        )
        vectors = self._session.run(
            None,
            {'input_ids': input_ids, 'attention_mask': attention_mask},
        )[0].astype(_np.float32, copy=False)
        norms = _np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / _np.maximum(norms, 1e-12)
        return vectors.tolist()


class LanceMemory:
    """Small repository-local LanceDB table using float16 384-d vectors."""

    def __init__(
        self,
        uri: str | Path = DEFAULT_LANCE_URI,
        embedder: Optional[BgeEmbedder] = None,
        table_name: str = 'memory',
    ) -> None:
        if _lancedb is None or _pa is None or _np is None:
            raise StageNotReady('LanceDB memory needs lancedb, pyarrow, and numpy')
        self.uri = Path(uri)
        self.uri.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or BgeEmbedder()
        self.table_name = re.sub(r'[^a-zA-Z0-9_]', '_', table_name)[:64]
        self._db = _lancedb.connect(str(self.uri))

    @property
    def schema(self):
        return _pa.schema(
            [
                _pa.field('id', _pa.string()),
                _pa.field('text', _pa.string()),
                _pa.field('kind', _pa.string()),
                _pa.field('created_at', _pa.float64()),
                _pa.field('vector', _pa.list_(_pa.float16(), EMBEDDING_DIM)),
            ]
        )

    def _table_names(self) -> list[str]:
        response = self._db.list_tables()
        return list(response.tables or [])

    def _open_table(self):
        if self.table_name not in self._table_names():
            return None
        return self._db.open_table(self.table_name)

    def upsert(self, documents: Iterable[Mapping[str, str]]) -> int:
        docs = list(documents)
        if not docs:
            return 0
        texts = [' '.join(str(doc.get('text', '')).split())[:2000] for doc in docs]
        vectors = self.embedder.encode(texts)
        now = time.time()
        rows = []
        for index, (doc, text, vector) in enumerate(zip(docs, texts, vectors)):
            if not text:
                continue
            doc_id = str(doc.get('id') or '').strip()
            if not doc_id:
                doc_id = hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]
            rows.append(
                {
                    'id': doc_id[:128],
                    'text': text,
                    'kind': str(doc.get('kind') or 'memory')[:64],
                    'created_at': float(doc.get('created_at') or now + index * 1e-6),
                    'vector': _np.asarray(vector, dtype=_np.float16).tolist(),
                }
            )
        if not rows:
            return 0
        table = self._open_table()
        if table is None:
            self._db.create_table(
                self.table_name,
                data=rows,
                schema=self.schema,
            )
            return len(rows)
        # IDs are generated locally and restricted before entering the filter.
        for row in rows:
            escaped = row['id'].replace("'", "''")
            table.delete("id = '{0}'".format(escaped))
        table.add(rows)
        return len(rows)

    def query(self, text: str, k: int = DEFAULT_TOP_K) -> list[dict]:
        table = self._open_table()
        if table is None or table.count_rows() == 0:
            return []
        vector = self.embedder.encode([text])[0]
        limit = max(1, min(int(k), 5))
        rows = table.search(vector).limit(limit).to_list()
        results = [
            {
                'id': str(row.get('id', '')),
                'text': str(row.get('text', '')),
                'kind': str(row.get('kind', 'memory')),
                'distance': float(row.get('_distance', math.inf)),
            }
            for row in rows
        ]
        return [
            row for row in results
            if math.isfinite(row['distance'])
            and row['distance'] <= MAX_RECALL_DISTANCE
        ]

    def count(self) -> int:
        table = self._open_table()
        return int(table.count_rows()) if table is not None else 0


def format_rag_context(
    hits: Sequence[Mapping[str, object]],
    max_chars: int = RAG_MAX_CHARS,
) -> str:
    """Format retrieved text as bounded, quoted context for Cosmos."""
    budget = max(0, min(int(max_chars), RAG_MAX_CHARS))
    parts = []
    used = 0
    for hit in hits[:5]:
        text = ' '.join(str(hit.get('text', '')).split())
        if not text:
            continue
        kind = re.sub(r'[^a-zA-Z0-9_-]', '', str(hit.get('kind', 'memory')))[:24]
        part = '[{0}] {1}'.format(kind or 'memory', text)
        remaining = budget - used
        if remaining <= 0:
            break
        part = part[:remaining]
        parts.append(part)
        used += len(part) + 1
    return '\n'.join(parts)
