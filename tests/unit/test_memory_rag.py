from __future__ import annotations

import math

import pytest

from jetbot_agent.robot_loop.memory_stubs import (
    DEFAULT_MODEL_PATH,
    BgeEmbedder,
    EMBEDDING_DIM,
    LanceMemory,
    RAG_MAX_CHARS,
    format_rag_context,
)


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * EMBEDDING_DIM
            vector[0 if 'camera' in text.lower() else 1] = 1.0
            vectors.append(vector)
        return vectors


def test_lancedb_upsert_is_idempotent_and_query_returns_relevant_text(tmp_path):
    memory = LanceMemory(tmp_path / 'db', embedder=FakeEmbedder())
    documents = [
        {
            'id': 'camera',
            'text': 'The JetBot camera faces forward.',
            'kind': 'hardware',
        },
        {'id': 'fruit', 'text': 'Bananas are yellow.', 'kind': 'fact'},
    ]

    assert memory.upsert(documents) == 2
    assert memory.upsert([documents[0]]) == 1
    assert memory.count() == 2
    hits = memory.query('Where does the camera point?', k=4)

    assert hits[0]['id'] == 'camera'
    assert hits[0]['text'] == documents[0]['text']
    assert math.isfinite(hits[0]['distance'])


def test_rag_context_is_bounded_and_contains_no_vectors():
    hits = [
        {'id': str(i), 'kind': 'fact', 'text': 'detail ' * 100, 'vector': [1.0]}
        for i in range(5)
    ]

    context = format_rag_context(hits)

    assert len(context) <= RAG_MAX_CHARS
    assert '[fact]' in context
    assert 'vector' not in context


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.is_file(), reason='local BGE artifact absent')
def test_real_bge_is_cpu_normalized_384_dimensions():
    embedder = BgeEmbedder()
    vectors = embedder.encode(['The robot camera faces forward.'])

    assert len(vectors) == 1
    assert len(vectors[0]) == EMBEDDING_DIM
    assert math.sqrt(sum(value * value for value in vectors[0])) == pytest.approx(
        1.0, abs=1e-4
    )
