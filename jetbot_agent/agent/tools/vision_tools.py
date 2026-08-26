"""Vision tools — Stage H / I3. Read-only camera observation, no captioning.

Every tool here is :class:`~jetbot_agent.agent.tools.base.RiskClass.READ_ONLY`:
the module has no motion vocabulary at all, and the object it is handed on
:attr:`ToolContext.perception` is checked against
:data:`~jetbot_agent.agent.tools.base.FORBIDDEN_MOTION_ATTRS` before use, so a
camera-shaped wrapper around a motor object cannot sneak in through the
perception slot.

**What is real and what is not.** Three tools work today because they only need
``src/perception`` (``vision_capture``, ``vision_describe_scene``,
``vision_detect_motion``). Two do not exist yet, because the model they need is
Stage G work and Stage G is still open: ``vision_read_text`` (OCR) and
``vision_locate_object`` (visual grounding) declare their schema and then raise
:class:`~jetbot_agent._stage.StageNotReady`. They do **not** return a plausible
answer. A fabricated caption or a made-up bounding box is worse than a refusal,
because a refusal is something the harness can branch on and a hallucination is
something it will act on.

``vision_describe_scene`` is named for what it returns: measured frame metadata
(shape, sequence, capture time, mean intensity, frame-to-frame change score).
Its ``caption`` field is always ``None`` with ``caption_available: False``, for
the same reason.

**Camera policy.** :func:`create_camera_service` defaults to the synthetic
``fake`` backend and refuses a hardware backend unless the caller passes
``allow_hardware=True``. Unit tests therefore cannot open the CSI sensor by
accident, and the real ``gst_csi`` path stays an explicit, deliberate,
one-at-a-time bring-up step (``docs/camera.md``, ``config/camera.yaml``).

See ``docs/bringup/09c-agent-i3-i4.md``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, Mapping, Optional, Tuple

from jetbot_agent._stage import StageNotReady

from .base import (
    FORBIDDEN_MOTION_ATTRS,
    RiskClass,
    Tool,
    ToolContext,
    ToolError,
    ToolSafetyViolation,
)
from .registry import ToolRegistry

LOGGER = logging.getLogger('jetbot_agent.agent.tools.vision')

#: The camera backend a test, a dry run, or an unconfigured process gets.
DEFAULT_CAMERA_BACKEND = 'fake'

#: Backends that touch no sensor. Anything else needs ``allow_hardware=True``.
SOFTWARE_CAMERA_BACKENDS = frozenset({
    'fake', 'file', 'image', 'mock', 'synthetic', 'video',
})

#: Where captures land. Gitignored (see ``.gitignore``), so frames stay local.
DEFAULT_IMAGE_DIR = Path('data/images')

#: Operator-supplied capture labels are filename components, so they are
#: restricted to a shape that cannot escape :data:`DEFAULT_IMAGE_DIR`.
LABEL_PATTERN = r'^[a-z0-9][a-z0-9_-]{0,31}$'

MAX_MOTION_SAMPLES = 5
DEFAULT_MOTION_SAMPLES = 3

#: The gate the two unimplemented tools wait on.
VLM_GATE = 'Stage G / issue #17 (Qwen2.5-VL runtime)'
VLM_ISSUE_URL = 'https://github.com/AbuAyah110/jetbot-orin-super/issues/17'

_METHODS_EVERY_VISION_TOOL_NEEDS = ('capture_frame', 'detect_change')


class VisionUnavailable(ToolError):
    """No usable camera is wired into :class:`ToolContext.perception`."""


def assert_read_only_perception(obj: Any) -> Any:
    """Refuse a perception object that is anything more than a camera.

    The mirror of :func:`~jetbot_agent.agent.tools.base.assert_narrow_motion`
    for the perception slot: presence of a wheel-level, PWM/GPIO/I2C, watchdog,
    or e-stop-clearing attribute proves the object belongs below the tool
    boundary, whatever it calls itself.
    """
    if obj is None:
        raise VisionUnavailable(
            'no camera wired into ToolContext.perception; vision tools refuse to '
            'guess. Pass CameraService (see create_camera_service) when building '
            'the context.'
        )
    for name in sorted(FORBIDDEN_MOTION_ATTRS):
        if hasattr(obj, name):
            raise ToolSafetyViolation(
                f'{type(obj).__name__} exposes {name!r}; the perception slot may '
                'only carry a read-only camera (see docs/safety.md)'
            )
    missing = [
        name for name in _METHODS_EVERY_VISION_TOOL_NEEDS
        if not callable(getattr(obj, name, None))
    ]
    if missing:
        raise VisionUnavailable(
            f'{type(obj).__name__} is not a usable camera service (missing {missing})'
        )
    return obj


def create_camera_service(
    backend: str = DEFAULT_CAMERA_BACKEND,
    *,
    allow_hardware: bool = False,
    buffer_size: int = 5,
    motion_threshold: float = 8.0,
    auto_open: bool = True,
    **camera_options: Any,
) -> Any:
    """Build a ``perception.CameraService`` for the tool layer.

    Defaults to the synthetic backend, and a hardware backend is opt-in rather
    than opt-out: the CSI sensor is a single shared resource and a unit test run
    must never be the thing that grabs it.
    """
    name = (backend or DEFAULT_CAMERA_BACKEND).strip().lower()
    if name not in SOFTWARE_CAMERA_BACKENDS and not allow_hardware:
        raise VisionUnavailable(
            f'camera backend {name!r} touches hardware; pass allow_hardware=True '
            f'deliberately. Software backends: {sorted(SOFTWARE_CAMERA_BACKENDS)}.'
        )
    # Imported here, not at module scope: the tool package must stay importable
    # without ``src`` on sys.path (see tests/unit/test_tool_safety.py).
    from perception import CameraService
    from perception.camera import create_camera

    camera = create_camera(name, **camera_options)
    return CameraService(
        backend=camera,
        buffer_size=int(buffer_size),
        motion_threshold=float(motion_threshold),
        auto_open=bool(auto_open),
    )


def _brightness(image: Any) -> Optional[float]:
    """Mean intensity of a frame, or ``None`` if numpy is unavailable."""
    try:
        import numpy
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        return None
    return round(float(numpy.asarray(image, dtype='float32').mean()), 3)


def _frame_payload(frame: Any, camera: Any) -> Dict[str, Any]:
    """Measured facts about one frame. Never the pixels themselves.

    Frames are deliberately not returned to the model: a caller that needs the
    image reads the saved file. That keeps a tool result small and keeps frame
    data out of the conversation transcript.
    """
    shape = tuple(int(dim) for dim in frame.image.shape)
    timestamp = getattr(frame, 'timestamp', None)
    return {
        'height': shape[0],
        'width': shape[1],
        'channels': shape[2] if len(shape) > 2 else 1,
        'shape': shape,
        'sequence': int(getattr(frame, 'sequence', 0)),
        'timestamp': timestamp.isoformat() if timestamp is not None else None,
        'source': str(getattr(frame, 'source', '')),
        'backend': str(getattr(camera, 'backend_name', '')),
        'mean_intensity': _brightness(frame.image),
        'image_inlined': False,
    }


def _motion_payload(result: Any) -> Dict[str, Any]:
    return {
        'changed': bool(result.changed),
        'score': round(float(result.score), 4),
        'threshold': round(float(result.threshold), 4),
    }


class VisionTool(Tool):
    """Read-only base: resolves the camera and forbids the motion slot."""

    abstract = True
    risk: ClassVar[RiskClass] = RiskClass.READ_ONLY

    #: False on a tool whose model has not landed; see :class:`GatedVisionTool`.
    stage_ready: ClassVar[bool] = True
    stage_gate: ClassVar[str] = ''

    def camera(self, context: ToolContext) -> Any:
        return assert_read_only_perception(context.perception)


class GatedVisionTool(VisionTool):
    """A declared-but-unavailable tool. The schema is real; the answer is not.

    Subclasses exist so that the interface, the parameter bounds, and the
    registry wiring can be reviewed and tested now, and so that a model asking
    for OCR gets a specific refusal naming the gate instead of an invented
    result.
    """

    abstract = True
    stage_ready: ClassVar[bool] = False
    stage_gate: ClassVar[str] = VLM_GATE

    def _run(self, context: ToolContext, **kwargs: Any) -> Any:
        # Raise before touching the camera: a refusal should have no side effects.
        raise StageNotReady(
            f'{self.name} is not implemented: it needs {self.stage_gate}, which has '
            f'not landed ({VLM_ISSUE_URL}). The schema and wiring are in place; no '
            'result is fabricated. See docs/bringup/09c-agent-i3-i4.md.'
        )


class VisionCaptureTool(VisionTool):
    """Grab one frame and optionally write it under the capture directory.

    ``READ_ONLY`` is about the robot, not the filesystem: this tool commands
    nothing and moves nothing. Its one side effect is a JPEG in
    :data:`DEFAULT_IMAGE_DIR`, with a generated filename — the optional
    ``label`` is bounded by :data:`LABEL_PATTERN` and reduced to a bare name, so
    a model cannot steer the write anywhere else.
    """

    name: ClassVar[str] = 'vision_capture'
    description: ClassVar[str] = (
        'Capture one frame from the robot camera and save it under the capture '
        'directory. Returns frame metadata and the file path, not the image '
        'itself. Read-only: nothing moves.'
    )
    timeout_sec: ClassVar[float] = 3.0
    parameters: ClassVar[Mapping[str, Any]] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'save': {
                'type': 'boolean',
                'default': True,
                'description': 'Write the frame to disk as well as reporting metadata.',
            },
            'label': {
                'type': 'string',
                'maxLength': 32,
                'pattern': LABEL_PATTERN,
                'description': (
                    'Optional short lower-case tag for the filename '
                    '(letters, digits, underscore, hyphen).'
                ),
            },
        },
        'required': [],
    }

    def __init__(self, image_dir: Optional[Path] = None) -> None:
        self._image_dir = Path(image_dir) if image_dir is not None else None

    def _target_dir(self, context: ToolContext) -> Path:
        if self._image_dir is not None:
            return self._image_dir
        configured = context.metadata.get('image_dir') if context.metadata else None
        return Path(configured) if configured else DEFAULT_IMAGE_DIR

    def _run(self, context: ToolContext, **kwargs: Any) -> Dict[str, Any]:
        camera = self.camera(context)
        save = bool(kwargs.get('save', True))
        label = kwargs.get('label') or 'capture'

        frame = camera.capture_frame()
        payload = _frame_payload(frame, camera)
        payload['saved_path'] = None
        payload['saved'] = False

        if not save:
            return payload
        if not callable(getattr(camera, 'save_frame', None)):
            raise VisionUnavailable(
                f'{type(camera).__name__} cannot save frames; call with save=false'
            )
        # Path().name strips any separator the pattern would have rejected anyway;
        # belt and braces, because this string reaches the filesystem.
        stamp = payload['timestamp'] or ''
        safe_stamp = re.sub(r'[^0-9]', '', stamp)[:14] or '0'
        filename = Path(f'{label}_{payload["sequence"]:06d}_{safe_stamp}.jpg').name
        target = self._target_dir(context) / filename
        written = Path(camera.save_frame(target))
        payload['saved_path'] = str(written)
        payload['saved'] = True
        context.logger.info('vision_capture sequence=%d path=%s',
                            payload['sequence'], written)
        return payload


class VisionDescribeSceneTool(VisionTool):
    """Report what the camera measurably shows. No captioning model is loaded.

    The honest answer to "what do you see" with no VLM available is a set of
    numbers, so that is what this returns. ``caption`` is always ``None`` and
    ``caption_available`` is always ``False`` until the gate in
    :data:`VLM_GATE` closes.
    """

    name: ClassVar[str] = 'vision_describe_scene'
    description: ClassVar[str] = (
        'Report measured metadata for the current camera frame: resolution, '
        'capture time, frame sequence, mean intensity, and frame-to-frame change '
        'score. This is not a caption and no vision-language model is involved; '
        'the caption field is always empty until the VLM runtime lands.'
    )
    timeout_sec: ClassVar[float] = 3.0
    parameters: ClassVar[Mapping[str, Any]] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'refresh': {
                'type': 'boolean',
                'default': True,
                'description': 'Capture a new frame instead of reusing the buffered one.',
            },
        },
        'required': [],
    }

    def _run(self, context: ToolContext, **kwargs: Any) -> Dict[str, Any]:
        camera = self.camera(context)
        refresh = bool(kwargs.get('refresh', True))

        frame = None
        if not refresh and callable(getattr(camera, 'get_latest_frame', None)):
            frame = camera.get_latest_frame()
        if frame is None:
            frame = camera.capture_frame()

        payload = _frame_payload(frame, camera)
        payload['motion'] = _motion_payload(camera.detect_change(frame.image))
        payload.update({
            'caption': None,
            'caption_available': False,
            'caption_gate': VLM_GATE,
            'detections': [],
            'note': (
                'Measured frame metadata only. No captioning or detection model is '
                'loaded, so no scene description is produced or implied.'
            ),
        })
        return payload


class VisionDetectMotionTool(VisionTool):
    """Sample a few frames and report the change score between them.

    The detector is a grayscale absolute-difference mean
    (``src/perception/motion_detector.py``) — cheap, deterministic, and not ML.
    The first sample only establishes a baseline, so ``comparisons`` is one less
    than ``samples`` and a single-sample call honestly reports a score of zero.
    """

    name: ClassVar[str] = 'vision_detect_motion'
    description: ClassVar[str] = (
        'Sample consecutive camera frames and report how much the image changed '
        'between them, using a cheap grayscale difference (no ML). Useful for '
        '"is something moving in front of me"; not a substitute for collision '
        'safety. Read-only.'
    )
    timeout_sec: ClassVar[float] = 4.0
    parameters: ClassVar[Mapping[str, Any]] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'samples': {
                'type': 'integer',
                'minimum': 1,
                'maximum': MAX_MOTION_SAMPLES,
                'default': DEFAULT_MOTION_SAMPLES,
                'description': 'How many consecutive frames to compare.',
            },
        },
        'required': [],
    }

    def _run(self, context: ToolContext, **kwargs: Any) -> Dict[str, Any]:
        camera = self.camera(context)
        samples = int(kwargs.get('samples', DEFAULT_MOTION_SAMPLES))
        samples = max(1, min(MAX_MOTION_SAMPLES, samples))

        scores: list = []
        changed = False
        threshold = 0.0
        frame = None
        for _ in range(samples):
            frame = camera.capture_frame()
            result = camera.detect_change(frame.image)
            scores.append(round(float(result.score), 4))
            threshold = round(float(result.threshold), 4)
            changed = changed or bool(result.changed)

        payload = _frame_payload(frame, camera)
        payload.update({
            'samples': samples,
            'comparisons': max(0, samples - 1),
            'scores': scores,
            'max_score': max(scores) if scores else 0.0,
            'threshold': threshold,
            'changed': changed,
            'method': 'grayscale_absdiff_mean',
            'note': (
                'The first sample establishes the baseline and always scores 0.0; '
                'only later samples carry information.'
            ),
        })
        return payload


class VisionReadTextTool(GatedVisionTool):
    """OCR interface. Declared now, answered when the VLM runtime lands."""

    name: ClassVar[str] = 'vision_read_text'
    description: ClassVar[str] = (
        'Read text visible in the camera frame (OCR). NOT AVAILABLE YET: the '
        'recognition model is a Stage G deliverable, so this tool refuses '
        'explicitly rather than guessing at any text.'
    )
    timeout_sec: ClassVar[float] = 5.0
    parameters: ClassVar[Mapping[str, Any]] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'region': {
                'type': 'string',
                'enum': ['full', 'center', 'top', 'bottom', 'left', 'right'],
                'default': 'full',
                'description': 'Which part of the frame to read.',
            },
            'min_confidence': {
                'type': 'number',
                'minimum': 0.0,
                'maximum': 1.0,
                'default': 0.5,
                'description': 'Drop recognised text below this confidence.',
            },
        },
        'required': [],
    }


class VisionLocateObjectTool(GatedVisionTool):
    """Visual grounding interface. Declared now, answered after the same gate."""

    name: ClassVar[str] = 'vision_locate_object'
    description: ClassVar[str] = (
        'Locate an object described in words within the camera frame and return '
        'its bounding box (visual grounding). NOT AVAILABLE YET: the grounding '
        'model is a Stage G deliverable, so this tool refuses explicitly rather '
        'than inventing a box. Never use its absence as clearance to move.'
    )
    timeout_sec: ClassVar[float] = 5.0
    parameters: ClassVar[Mapping[str, Any]] = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'query': {
                'type': 'string',
                'minLength': 2,
                'maxLength': 64,
                'description': 'Short description of the object to find.',
            },
            'max_matches': {
                'type': 'integer',
                'minimum': 1,
                'maximum': 5,
                'default': 1,
                'description': 'Most boxes to return.',
            },
        },
        'required': ['query'],
    }


def working_vision_tools(image_dir: Optional[Path] = None) -> Tuple[Tool, ...]:
    """Tools that do real work against a camera today."""
    return (
        VisionCaptureTool(image_dir=image_dir),
        VisionDescribeSceneTool(),
        VisionDetectMotionTool(),
    )


def gated_vision_tools() -> Tuple[Tool, ...]:
    """Declared tools that raise :class:`StageNotReady` until the VLM gate."""
    return (VisionReadTextTool(), VisionLocateObjectTool())


def vision_tools(
    *,
    include_gated: bool = True,
    image_dir: Optional[Path] = None,
) -> Tuple[Tool, ...]:
    """Fresh instances of the I3 tool set."""
    tools = working_vision_tools(image_dir=image_dir)
    if include_gated:
        tools = tools + gated_vision_tools()
    return tools


def register_vision_tools(
    registry: ToolRegistry,
    *,
    allow: bool = False,
    include_gated: bool = False,
    image_dir: Optional[Path] = None,
) -> Tuple[str, ...]:
    """Catalogue the vision tools on ``registry`` and return their names.

    Deny-by-default twice over. ``allow=False`` (the production default) leaves
    every tool un-invocable until wiring code names it, and ``include_gated``
    defaults to ``False`` so a model is not offered OCR and grounding verbs that
    can only refuse. Set it to ``True`` to review or test the declared surface.
    """
    names = []
    for tool in vision_tools(include_gated=include_gated, image_dir=image_dir):
        registry.register(tool, allow=allow)
        names.append(tool.name)
        if not getattr(tool, 'stage_ready', True):
            LOGGER.warning('vision_tool_unavailable name=%r gate=%r',
                           tool.name, getattr(tool, 'stage_gate', ''))
    return tuple(names)


__all__ = [
    'DEFAULT_CAMERA_BACKEND',
    'DEFAULT_IMAGE_DIR',
    'DEFAULT_MOTION_SAMPLES',
    'GatedVisionTool',
    'LABEL_PATTERN',
    'MAX_MOTION_SAMPLES',
    'SOFTWARE_CAMERA_BACKENDS',
    'VLM_GATE',
    'VLM_ISSUE_URL',
    'VisionCaptureTool',
    'VisionDescribeSceneTool',
    'VisionDetectMotionTool',
    'VisionLocateObjectTool',
    'VisionReadTextTool',
    'VisionTool',
    'VisionUnavailable',
    'assert_read_only_perception',
    'create_camera_service',
    'gated_vision_tools',
    'register_vision_tools',
    'vision_tools',
    'working_vision_tools',
]
