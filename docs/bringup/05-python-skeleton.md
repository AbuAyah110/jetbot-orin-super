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

After E: continue F then G, then **Stage H agent (I1–I8)** before Stage I memory.
