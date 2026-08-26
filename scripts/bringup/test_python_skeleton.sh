#!/usr/bin/env bash
# Stage E: import smoke for jetbot_agent stubs + config.yaml.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
if [[ -x "${ROOT}/.venv/bin/python3" ]]; then
  PYTHON="${ROOT}/.venv/bin/python3"
else
  PYTHON="python3"
fi
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - <<'PY'
import yaml
from pathlib import Path
import jetbot_agent
import jetbot_agent.hardware
import jetbot_agent.hardware.motor_controller
import jetbot_agent.hardware.csi_camera
import jetbot_agent.hardware.audio_interface
import jetbot_agent.audio
import jetbot_agent.audio.audio_preprocessor
import jetbot_agent.audio.zipformer_asr
import jetbot_agent.audio.piper_tts
import jetbot_agent.engine
import jetbot_agent.engine.trt_llm_vlm
import jetbot_agent.engine.trt_vla_motor
import jetbot_agent.engine.trt_embedder
import jetbot_agent.memory
import jetbot_agent.memory.chroma_db
import jetbot_agent.memory.facts_db
import jetbot_agent.agent
import jetbot_agent.agent.hermes_harness
import jetbot_agent.agent.tools
import jetbot_agent.agent.tools.navigation_tools
import jetbot_agent.agent.tools.vision_tools
import jetbot_agent.agent.tools.search_tools

cfg = yaml.safe_load((Path("jetbot_agent") / "config.yaml").read_text())
assert "system" in cfg and "motors" in cfg
print("python", __import__("sys").executable)
print("jetbot_agent", jetbot_agent.__file__)
print("config keys", sorted(cfg.keys()))
print("python skeleton ok")
PY
