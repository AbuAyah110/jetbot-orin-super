# Stage I — CPU BGE + LanceDB memory

The earlier ChromaDB + SQLite design is superseded. The deployed memory path
is one repository-local LanceDB table with BGE-small embeddings:

- `BAAI/bge-small-en-v1.5`, dynamically quantized ONNX weights
- ONNX Runtime `CPUExecutionProvider` only; two threads
- 384-dimensional normalized embeddings, stored as float16
- LanceDB at ignored path `data/memory/lancedb/`
- top-k 4, maximum 5
- distance cutoff 0.85
- retrieved prompt context capped at 1400 characters (about 350 tokens)
- no PyTorch, `transformers`, sentence-transformers, CUDA, or TensorRT

## Reproduce the local INT8 graph

The source FP32 graph is `data/models/bge-small-en-v1.5-onnx/bge-small.onnx`.

```bash
.venv/bin/python scripts/bringup/quantize_bge_int8.py
```

Measured on this Orin Nano Super:

| Graph | Size | Three-text inference, 2 CPU threads |
| --- | ---: | ---: |
| FP32 | 127 MiB | 37.8 ms |
| dynamic INT8 | 32.5 MiB | 15.8 ms |

INT8 versus FP32 cosine agreement was 0.9947–0.9976 over the validation
sentences. Output is `(batch, 384)` and L2 normalized.

## Seed and inspect memory

```bash
.venv/bin/python scripts/bringup/ingest_lancedb.py
```

The seed contains only JetBot identity, capabilities, and safety limitations.
It is idempotent by document ID. The live loop can add explicit user facts:

> Remember that my favorite color is blue.

Normal conversation is not automatically promoted to long-term memory.

## Prompt safety

Retrieved text is quoted as data, not instructions. It is used only in parked
general and visual conversation; deterministic motion and safety routing do
not consult RAG. Old images, vectors, hidden reasoning, and raw model JSON are
never placed in memory.

## Verification

The production voice loop was tested across two separate processes:

1. “Remember that the test phrase is cobalt.”
2. Restart.
3. “What is the test phrase?”
4. BGE retrieved one LanceDB row and Cosmos answered `cobalt`.

The synthetic row and chat history were removed after the gate.

## Live co-residency

Measured with Cosmos, camera, Zipformer, Piper, BGE, LanceDB, and VAD live:

- system RAM: 5776 / 7620 MB
- available RAM: about 1.7 GiB
- talk-and-drive cgroup: about 397 MB
- prior no-RAG talk-and-drive cgroup: about 197 MB
- practical BGE + ONNX Runtime + LanceDB/PyArrow delta: about 200 MB
- swap: effectively unused

This is higher than the earlier 50–120 MB planning estimate because PyArrow
is part of the live LanceDB process, but it remains within the board budget.
