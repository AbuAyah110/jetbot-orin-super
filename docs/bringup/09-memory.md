# Stage I — Memory installs (after agent)

Start only after **Stage H / I8** (integration loop exists without memory). No LLM summarizer / compactor in this pass.

This is **not** agent ticket I1–I8. Those are Stage H integration slices.

## ChromaDB

Local persist under `data/memory/chroma/`. Upsert one document; query it back.

## SQLite facts

Schema + one put/get (Mem0-style key-value). Path: `data/memory/facts.db` (see `jetbot_agent/config.yaml`).

Pass: both stores round-trip without the VLM.

## After this stage

Memory **tools** for the Hermes harness (wrap Chroma/SQLite as I2 tools) are a follow-on ticket, not I1–I8.
