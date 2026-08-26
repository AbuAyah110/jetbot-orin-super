# Remote access — driving this JetBot from a phone

How to reach this repository (and, when explicitly attended, the robot hardware)
from a phone or another machine.

Board: NVIDIA Jetson Orin Nano Super, L4T R36.4.4, aarch64, Ubuntu 22.04, headless.

## Two different things

| | Runs where | Can touch hardware? |
| --- | --- | --- |
| **Cloud agent** | Cursor's cloud VM (x86, no Jetson) | **No** |
| **My Machines worker** | This Jetson, as user `impulse110` | Yes — treat as live robot |

Read [safety.md](safety.md) before using the second one.

## Phone options

- **Cursor iOS app** — sign in with the same Cursor account, pick the agent target.
- **cursor.com/agents** in mobile Safari/Chrome — installable as a PWA; same
  agent list as the desktop Agents view.
- **Slack** — `@Cursor` in a connected workspace, useful for one-off questions
  and PR review; no interactive terminal.

All three can start either a cloud agent or an agent that runs on this Jetson,
depending on the target chosen in the "Run on" / environment picker.

## The on-device worker

Cloud VMs have no Jetson peripherals. Anything that needs real hardware — I2C
motor HAT, CSI camera, microphone, GPU — has to run through a **My Machines
worker** started on this board.

Start it from the repo checkout so the worker picks up the right git remote:

```bash
cd /home/impulse110/Documents/jetbot-orin-super
agent worker start --name jetbot-orin
```

The process must stay alive for the machine to remain selectable from the phone.
For unattended operation use the systemd `--user` template in
[`systemd/cursor-worker.service.example`](../systemd/cursor-worker.service.example),
or `nohup` for a quick session. Install instructions are in
[Keeping the worker alive](#keeping-the-worker-alive) below.

Confirm the machine is visible: open the Agents view on the phone, then the
"Run on" picker — `jetbot-orin` should appear under **My Machines**. If it does
not, run `agent worker start --debug` from the repo directory.

## Safety rule for unattended phone sessions

> **Motor and PWM commands must never be issued from an unattended phone
> session.**

The wheels-up rule from [safety.md](safety.md) and
[hardware_motors.md](hardware_motors.md) applies in full, and a phone session is
the worst possible place to violate it: you cannot see the robot, you cannot
reach the power switch, and you cannot hit a physical emergency stop.

Concretely, from a phone session an agent must **not**:

- run `scripts/bringup/test_motors.py` or any script that drives wheels;
- set `backend: jetbot_i2c` in `config/robot.yaml`;
- write PCA9685 / motor-HAT registers on I2C bus 7 (`0x60`, `0x70`);
- write motor GPIO, disable the watchdog, or bypass the velocity limiter.

Motion testing requires an operator physically present, wheels off the ground,
and a hand on the power cut — per the testing progression in
[safety.md](safety.md). Reading and editing code, running mock-backend unit
tests, and inspecting logs are all fine unattended.

Non-motor hardware reads (I2C scan, camera capture, ALSA record) are lower risk
but still produce misleading results if the robot is unpowered or unattended;
prefer to run them attended.

## What cloud agents cannot do

A Cursor cloud agent VM is a plain x86 Linux container. On this project that
means it has **no access** to:

- `/dev/i2c-*` — no motor HAT, no OLED, no `i2cdetect`
- `nvargus` / `/dev/video0` — no CSI camera, no `nvarguscamerasrc` GStreamer path
- ALSA devices — no microphone or speaker, so no voice pipeline capture
- TensorRT and the Jetson CUDA stack — no engine build, no accelerated inference

So every hardware gate in `scripts/bringup/` can only be verified **on-device**,
through the worker described above. A cloud agent that claims a hardware gate
passed is wrong by construction; see [AGENTS.md](../AGENTS.md).

## Keeping the worker alive

### Option A — `nohup` (quick, dies on reboot)

```bash
cd /home/impulse110/Documents/jetbot-orin-super
nohup agent worker start --name jetbot-orin > ~/cursor-worker.log 2>&1 &
tail -f ~/cursor-worker.log
```

Stop it with `pkill -f 'worker start'`.

### Option B — systemd `--user` unit (survives reboot, no sudo)

A `--user` unit needs no root. Copy the template, adjust `ExecStart` to the
binary path you actually have, then enable it:

```bash
mkdir -p ~/.config/systemd/user
cp systemd/cursor-worker.service.example ~/.config/systemd/user/cursor-worker.service
# edit ExecStart if your CLI is named cursor-agent rather than agent
systemctl --user daemon-reload
systemctl --user enable --now cursor-worker.service
systemctl --user status cursor-worker.service
journalctl --user -u cursor-worker.service -f
```

To keep the unit running when you are not logged in, enable lingering once:

```bash
loginctl enable-linger impulse110
```

That call may prompt for a password via polkit. Without lingering, the worker
stops when your last SSH session closes.

Log in to the CLI **before** enabling the unit — the unit reuses the credential
stored in your home directory and cannot complete a browser login itself.

## First-time setup on this board

Run these in your own terminal, in order; the first two are interactive.

```bash
gh auth login
agent login
cd /home/impulse110/Documents/jetbot-orin-super
agent worker start --name jetbot-orin
```

If `agent` is not found, the CLI may have installed as `cursor-agent`; check with
`ls ~/.local/bin | grep -i agent`. `~/.local/bin` is already on this machine's
`PATH`.
