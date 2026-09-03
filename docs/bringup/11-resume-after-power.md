# Resume after a battery power cycle

Saved 2026-08-27. Cosmos engines stay on disk; they are not rebuilt on boot.

## What was live

- Branch: `stage-g-cosmos-nano`
- Loop: no-beep VAD listen (`--auto-listen`), 8 s max utterance
- Engines: `~/jetbot-thin-stack/cosmos-engines/`
- Collision ToF: **VL53L0X on bus 1 @ 0x29** (40-pin pins 27 SDA / 28 SCL, not the
  motor HAT's pins 3/5 on bus 7) — millimetre ranging; bumper still not present

## After battery boot

Give the chassis floor space. USB audio and CSI take ~20 s.

The user systemd unit `jetbot-talk-and-drive.service` is enabled (`Linger=yes`).
It waits 20 s, then starts the same loop. You should hear **I'm ready for your
command**, then speak naturally (no beep).

If it does not start:

```bash
systemctl --user status jetbot-talk-and-drive.service
bash ~/Documents/jetbot-orin-super/scripts/bringup/resume_talk_and_drive.sh
```

To park and power off later:

```bash
bash ~/Documents/jetbot-orin-super/scripts/bringup/stop_talk_and_drive.sh
sudo shutdown -h now   # only if you already have a passwordless sudo for halt
```

Disable auto-start: `systemctl --user disable --now jetbot-talk-and-drive.service`
