# Stage G — TensorRT engines (isolated)

Do not connect engines to motors or the camera loop until each dummy-I/O ticket passes.

## Baseline

```bash
./scripts/diagnostics.sh   # TensorRT / CUDA sections
```

## Separate tickets (dummy I/O only)

1. TensorRT-Edge-LLM runtime present
2. Qwen2.5-VL-3B INT4 AWQ engine builds and runs one dummy vision+text forward
3. smolvla-jetbot TensorRT engine dummy motor-token I/O (no PWM)
4. llama-nemotron-embed-vl-1b-v2 embedder dummy vector out

Pass per ticket: process exits 0 with a logged output shape / token count. Fail: OOM — reduce batch, confirm 32 GB swap, MAXN SUPER.

Agent **I7** (VLM/engine tools) and real smolvla inside **I5** wait on these dummy tickets. The agent stage itself (I1–I8) is [08-agent.md](08-agent.md), **before** memory.
