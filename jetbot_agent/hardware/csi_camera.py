"""CSI camera stub. Stage C hardware is verified; this wrapper is not the capture path yet."""

from jetbot_agent._stage import StageNotReady


class CsiCamera:
    def __init__(self, *args, **kwargs):
        raise StageNotReady("hardware.csi_camera wrapper not implemented; use Argus/GStreamer bring-up")
