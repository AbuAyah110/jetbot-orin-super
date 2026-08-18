# Power — Jetson Orin Nano Super

!!! warning

    The battery option listed on the official [Orin BOM](https://jetbot.org/master/bill_of_materials_orin.html) (PD ~20W pack + USB-C to DC barrel) is **not sufficient** for sustained **Orin Nano Super / MAXN SUPER** workloads, especially once motors, Wi-Fi, camera, and later VLM/ASR/TTS run together.

    This project's power solution will be documented here once finalized. Do **not** assume the Nano-era or light Orin BOM pack will work for Super mode.

## Official Orin BOM note (upstream)

Upstream docs suggest a 10,000 mAh PD pack and recommend **7W** mode (`nvpmodel -m 1`) because **15W** mode could brown-out when motors stall or ramp hard. That is a **low-power** compromise — opposite of what we want for Super + on-device AI.

## Requirements for this project (targets)

| Need | Notes |
| --- | --- |
| Jetson input | Orin Nano Dev Kit DC jack / supported PD path — verify against your carrier |
| Headroom | Motors + CSI camera + NVMe + Wi-Fi + USB mic/speaker |
| Phase 2 | VLM + ASR + TTS + RAG under **MAXN SUPER** |
| Runtime | TBD — target useful demo runtime, not just idle boot |

## Status

- [ ] Select Super-capable battery / PSU
- [ ] Document wiring (Jetson vs motor driver rails)
- [ ] Measure brown-out under motor stall + model inference
- [ ] Update [BOM](bill_of_materials_orin.md) with final links

Until then: use a **bench supply** that meets Orin's input requirements for bring-up, or a known-good high-wattage pack you have validated — and share measurements so we can lock the BOM.
