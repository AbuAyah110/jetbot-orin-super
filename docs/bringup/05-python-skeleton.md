# Stage E — Python skeleton (`jetbot_agent/`)

Production modules are stubs. This stage only proves the tree imports and a venv can be created.

## Install

```bash
cd /home/impulse110/Documents/jetbot-orin-super
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r jetbot_agent/requirements.txt
```

## Verify

```bash
./scripts/bringup/test_python_skeleton.sh
```

Pass: `jetbot_agent` packages import; `config.yaml` parses.

If `python3 -m venv` fails (`ensurepip` / `python3-venv` missing), `virtualenv .venv` is an acceptable substitute. Do not `sudo apt` just for this gate.

After E: continue F then G, then **Stage H agent (I1–I8)** before Stage I memory.

## Probe 2026-08-25

`python3-venv` apt package is not installed (sudo password required). Created `.venv` with `virtualenv` instead.

```text
.venv/bin/python3
PyYAML 6.0.3
python skeleton ok
```

Stubs import without raising; constructing motor/VLM/TTS classes still raises `StageNotReady` until those gates.
