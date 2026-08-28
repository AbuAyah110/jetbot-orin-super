#!/usr/bin/env python3
"""Create the CPU INT8 BGE graph from the local FP32 ONNX artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

REPO = Path(__file__).resolve().parents[2]
MODEL_DIR = REPO / 'data' / 'models' / 'bge-small-en-v1.5-onnx'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(MODEL_DIR / 'bge-small.onnx'))
    parser.add_argument('--output', default=str(MODEL_DIR / 'bge-small-int8.onnx'))
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output)
    if not source.is_file():
        parser.error('missing FP32 graph: {0}'.format(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(source),
        str(output),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(output),
        sess_options=options,
        providers=['CPUExecutionProvider'],
    )
    output_shape = session.get_outputs()[0].shape
    if session.get_providers() != ['CPUExecutionProvider']:
        raise RuntimeError('BGE INT8 did not select CPUExecutionProvider')
    if not output_shape or output_shape[-1] != 384:
        raise RuntimeError('BGE INT8 output is not 384-dimensional')
    print(
        'wrote={0} MiB={1:.1f} output={2}'.format(
            output,
            output.stat().st_size / 1024 / 1024,
            output_shape,
        )
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
