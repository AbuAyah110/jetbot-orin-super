"""One CSI JPEG pipeline at 448×448 for Cosmos ViT.

Does not start Argus until :meth:`CsiJpeg448.open`. Unit tests must only inspect
:meth:`CsiJpeg448.gst_pipeline`. Never opens a second ``nvarguscamerasrc``.
"""

from __future__ import annotations

from typing import Optional

CSI_JPEG_SIZE = 448
# Sensor mode matches Stage C (IMX219 CAM0). Resize happens in NVMM before JPEG.
_CAPTURE_WIDTH = 1280
_CAPTURE_HEIGHT = 720


class CsiJpeg448:
    """In-process GStreamer: Argus → nvvidconv 448² → nvjpegenc → appsink."""

    def __init__(self, sensor_id: int = 0, fps: int = 15) -> None:
        self.sensor_id = int(sensor_id)
        self.fps = int(fps)
        self.width = CSI_JPEG_SIZE
        self.height = CSI_JPEG_SIZE
        self._pipeline = None
        self._appsink = None

    def gst_pipeline(self) -> str:
        """Single pipeline string. One nvarguscamerasrc, one nvjpegenc."""
        return (
            'nvarguscamerasrc sensor-id={sensor} num-buffers=0 ! '
            'video/x-raw(memory:NVMM), width=(int){cw}, height=(int){ch}, '
            'format=(string)NV12, framerate=(fraction){fps}/1 ! '
            'nvvidconv ! '
            'video/x-raw(memory:NVMM), width=(int){w}, height=(int){h} ! '
            'nvjpegenc ! image/jpeg, width=(int){w}, height=(int){h} ! '
            'appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false'
        ).format(
            sensor=self.sensor_id,
            cw=_CAPTURE_WIDTH,
            ch=_CAPTURE_HEIGHT,
            fps=self.fps,
            w=CSI_JPEG_SIZE,
            h=CSI_JPEG_SIZE,
        )

    @property
    def name(self) -> str:
        return 'csi_jpeg:{0}x{0}'.format(CSI_JPEG_SIZE)

    def open(self) -> None:
        if self._pipeline is not None:
            return
        import gi

        gi.require_version('Gst', '1.0')
        from gi.repository import Gst

        Gst.init(None)
        pipeline_str = self.gst_pipeline()
        pipeline = Gst.parse_launch(pipeline_str)
        sink = pipeline.get_by_name('sink')
        if sink is None:
            raise RuntimeError('CSI JPEG pipeline missing appsink name=sink')
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError('Failed to start CSI JPEG pipeline: {0}'.format(pipeline_str))
        self._pipeline = pipeline
        self._appsink = sink

    def capture_jpeg(self, timeout_ns: int = 5_000_000_000) -> bytes:
        """Pull one JPEG. Starts the single pipeline on first call."""
        self.open()
        from gi.repository import Gst

        sample = self._appsink.emit('try-pull-sample', timeout_ns)
        if sample is None:
            raise RuntimeError('CSI JPEG capture timed out')
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError('CSI JPEG buffer map failed')
        try:
            return bytes(mapinfo.data)
        finally:
            buf.unmap(mapinfo)

    def close(self) -> None:
        if self._pipeline is None:
            return
        from gi.repository import Gst

        self._pipeline.set_state(Gst.State.NULL)
        self._pipeline = None
        self._appsink = None

    def __enter__(self) -> 'CsiJpeg448':
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


def jpeg_pipeline_string(sensor_id: int = 0, fps: int = 15) -> str:
    return CsiJpeg448(sensor_id=sensor_id, fps=fps).gst_pipeline()
