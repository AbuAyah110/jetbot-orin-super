# Stage D — Waveshare USB audio (ALSA)

Observed hardware: Waveshare/Solid State System USB PnP Audio using the SSS1629 codec, with one speaker. Capture is 16 kHz mono; playback uses ALSA.

## Install

Plug in USB module. Identify cards:

```bash
arecord -l
aplay -l
arecord -L
```

The device was `plughw:2,0` during bring-up, but ALSA card indices change across boots and USB enumeration. Production configuration must resolve the endpoint by its ALSA USB device name; never hardcode card `2`. Set `JETBOT_ALSA_CAPTURE` / `JETBOT_ALSA_PLAYBACK` to the currently resolved endpoint if `default` is wrong.

## Safe mixer baseline

- Hardware capture gain tops out at +31 dB. Begin near 80%, then tune from measured recordings without clipping.
- Begin playback at low volume.
- Keep hardware sidetone/microphone monitoring **OFF**.
- Keep a physical mute/disconnect available.

**Safety:** enabling sidetone caused a dangerous loud hardware feedback loop during bring-up. Do not enable it. Until WebRTC APM acoustic echo cancellation passes Stage F2, tests must be sequential: stop capture before any playback. Noise suppression or RNNoise alone does not make simultaneous playback safe.

## Verify

```bash
./scripts/bringup/test_alsa.sh
```

Pass: a short 16 kHz mono WAV is recorded, capture is stopped, and the file is played once at low volume. Confirm the microphone is not silent, the recording is not clipped, sidetone is off, and the evidence records the USB device name and current resolved ALSA endpoint.

## Probe 2026-08-25 (F1)

Resolved by name, not card index:

| Field | Value |
| --- | --- |
| USB name | Solid State System Co.,Ltd. USB PnP Audio Device (`0c76:1203`) |
| ALSA id | `Device` (ephemeral index was 2 this boot) |
| Capture/playback | `plughw:CARD=Device,DEV=0` |
| Speaker | 20% (−50.38 dB), unmuted for one-shot play |
| Mic capture | 80% (+24.81 dB), on |
| Mic playback / sidetone | 0%, **off** |
| Mixer persist | `config/alsa-sss1629.state` (`alsactl --file … store`) |

Sequential 1 s 16 kHz mono capture then playback: `peak=0.025` `rms=0.0010`, not clipped. Restore with `alsactl --file config/alsa-sss1629.state restore <card>` after resolving the card by name.
