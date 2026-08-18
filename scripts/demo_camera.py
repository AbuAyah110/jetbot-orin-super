#!/usr/bin/env python3
"""Laptop-friendly camera demo (Milestone 2). Default backend is fake."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from perception import CameraService  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description='JetBot camera demo')
    parser.add_argument('--backend', default='fake', choices=['fake', 'file', 'webcam', 'gst_csi'])
    parser.add_argument('--path', default='', help='file backend path')
    parser.add_argument('--device', type=int, default=0, help='webcam index')
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--out', default='data/images/demo_capture.jpg')
    parser.add_argument('--motion-threshold', type=float, default=8.0)
    args = parser.parse_args()

    kwargs = {
        'camera': {
            'backend': args.backend,
            'path': args.path or None,
            'device_index': args.device,
            'motion_threshold': args.motion_threshold,
            'buffer_size': 5,
        }
    }
    changes = 0
    with CameraService.from_config(kwargs) as cam:
        print('backend:', cam.backend_name)
        for i in range(max(args.frames, 1)):
            frame = cam.capture_frame()
            motion = cam.detect_change(frame.image)
            if motion.changed:
                changes += 1
            print(
                'frame={0} seq={1} shape={2} motion={3:.2f} changed={4}'.format(
                    i,
                    frame.sequence,
                    frame.shape,
                    motion.score,
                    motion.changed,
                )
            )
        out = cam.save_frame(args.out)
        print('saved:', out)
        print('change_events:', changes)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
