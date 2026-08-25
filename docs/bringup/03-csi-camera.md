# Stage C — CSI camera (IMX219)

Raspberry Pi Camera v2.1 on CSI port 0 via 15-to-22 pin FPC adapter.

## Install

Hardware seated (contacts, cable orientation). No extra apt packages if JetPack Argus/GStreamer are present.

## Verify

```bash
./scripts/bringup/test_csi_camera.sh
```

Equivalent:

```bash
PYTHONPATH=src python3 scripts/demo_camera.py --backend gst_csi --frames 5 --out data/images/csi_bringup.jpg
```

Pass: JPEG written; frames have non-zero shape. Fail: check ribbon, `nvarguscamerasrc`, `dmesg | grep -i imx`.

## Probe 2026-08-25 (cheap Argus, no `test_csi_camera.sh` JPEG)

`nvargus-daemon` was active. One-buffer pipeline:

```bash
timeout 12 gst-launch-1.0 -e nvarguscamerasrc sensor-id=0 num-buffers=1 \
  ! "video/x-raw(memory:NVMM),width=640,height=480,framerate=30/1" \
  ! nvvidconv ! "video/x-raw,format=I420" ! fakesink
```

Reached EOS; Argus listed IMX219-class modes including 3280×2464. Prior live preview: `notebooks/camera/csi_camera_test.ipynb` printed `camera ok (224, 224, 3)` then `stopped`. Vision notebook with saved outputs: `notebooks/object_following/live_demo_nanoowl_orin.ipynb`.
