# Local Edge-LLM runtime data

This directory is the canonical on-device home for generated, non-git payloads:

```text
data/edgellm/
  cosmos/
    onnx/llm/
    onnx/visual/
    engines/llm/
    engines/visual/
    logs/
  cutedsl-cuda12/
  cutedsl-venv/
```

Everything below `data/edgellm/` is ignored except this file. Never commit ONNX,
externalized weights, TensorRT engines, model weights, or virtual environments.

For compatibility with workstation commands, `$HOME/tensorrt-edgellm-workspace`
may be a symlink to this directory and local `Cosmos-Reason2-2B` may point to
`cosmos`. The Cosmos rsync destination remains:

```text
~/tensorrt-edgellm-workspace/Cosmos-Reason2-2B/onnx/
```

The TensorRT Edge-LLM v0.10.0 source/build checkout belongs at
`third_party/tensorrt-edge-llm/`, which is also ignored in full.

A pre-existing CPU BGE-small ONNX candidate was consolidated at
`data/models/bge-small-en-v1.5-onnx/` (127 MiB graph, tokenizer files). It is
ignored and was not loaded or installed in this pass. Legacy thin-stack source,
LanceDB data, and the stopped accidental build logs are retained under the
ignored `data/archive/` tree for provenance.
