from .motor import Motor
from .robot import Robot

# The voice venv carries NumPy 2 for sherpa-onnx, which the system cv2 wheel
# cannot load, and has no notebook stack. Motors must stay importable anyway.
try:
    from .camera import Camera
except Exception:
    Camera = None

try:
    from .heartbeat import Heartbeat
except Exception:
    Heartbeat = None

try:
    from .image import bgr8_to_jpeg
except Exception:
    bgr8_to_jpeg = None

try:
    from .object_detection import ObjectDetector
except Exception:
    ObjectDetector = None  # TensorRT SSD plugins; not required for basic_motion