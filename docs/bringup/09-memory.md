# Stage I — Memory installs (after agent)

Start only after **Stage H / I8** (integration loop exists without memory). No LLM summarizer / compactor in this pass.

This is **not** agent ticket I1–I8. Those are Stage H integration slices.

## ChromaDB

Local persist under `data/memory/chroma/`. Upsert one document; query it back.

**Collection defaults** (Stage I): `jinaai/jina-clip-v2`, **`embedding_dim=256`**,
`hnsw:space=cosine`, `hnsw:M=16`, `hnsw:construction_ef=100`, `hnsw:search_ef=64`.
That shrinks **index disk/RAM only**. Encoder size is unchanged. Details:
[`jina_clip_v2.md`](../jina_clip_v2.md). Do not use SigLIP 2 or Nemotron as the
collection embedding model.

## SQLite facts

Schema + one put/get (Mem0-style key-value). Path: `data/memory/facts.db` (see `jetbot_agent/config.yaml`).

Pass: both stores round-trip without the VLM.

## After this stage

Memory **tools** for the Hermes harness (wrap Chroma/SQLite as I2 tools) are a follow-on ticket, not I1–I8.
