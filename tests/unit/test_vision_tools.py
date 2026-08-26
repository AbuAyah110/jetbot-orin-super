from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'ros2_ws' / 'src' / 'jetbot_base'))

from jetbot_agent._stage import StageNotReady
from jetbot_agent.agent.tools import (
    Capability,
    MockMotionInterface,
    RESERVED_PARAM_NAMES,
    RiskClass,
    ToolContext,
    ToolExecutionError,
    ToolPermissionError,
    ToolRegistry,
    ToolSafetyViolation,
    ToolValidationError,
)
from jetbot_agent.agent.tools.vision_tools import (
    DEFAULT_CAMERA_BACKEND,
    MAX_MOTION_SAMPLES,
    SOFTWARE_CAMERA_BACKENDS,
    VLM_GATE,
    VisionCaptureTool,
    VisionDescribeSceneTool,
    VisionDetectMotionTool,
    VisionLocateObjectTool,
    VisionReadTextTool,
    VisionUnavailable,
    assert_read_only_perception,
    create_camera_service,
    gated_vision_tools,
    register_vision_tools,
    vision_tools,
    working_vision_tools,
)
from perception import CameraService, FakeCamera

TOOLS_DIR = ROOT / 'jetbot_agent' / 'agent' / 'tools'
VISION_TOOLS_PATH = TOOLS_DIR / 'vision_tools.py'

WORKING_NAMES = ('vision_capture', 'vision_describe_scene', 'vision_detect_motion')
GATED_NAMES = ('vision_read_text', 'vision_locate_object')


class CountingCamera:
    """Camera-shaped proxy that records how often the tools touched it."""

    def __init__(self, service) -> None:
        self._service = service
        self.captures = 0
        self.change_checks = 0
        self.saves = 0

    @property
    def backend_name(self) -> str:
        return self._service.backend_name

    def capture_frame(self):
        self.captures += 1
        return self._service.capture_frame()

    def get_latest_frame(self):
        return self._service.get_latest_frame()

    def detect_change(self, image=None):
        self.change_checks += 1
        return self._service.detect_change(image)

    def save_frame(self, path):
        self.saves += 1
        return self._service.save_frame(path)


@pytest.fixture()
def camera():
    """A real CameraService over the synthetic backend. No sensor is opened."""
    service = CameraService(backend=FakeCamera(width=96, height=72), buffer_size=4)
    yield service
    service.close()


def _registry(perception, *, capabilities=(Capability.READ,), allow=True,
              include_gated=True, image_dir=None, metadata=None):
    context = ToolContext(perception=perception, metadata=metadata or {})
    registry = ToolRegistry(context, capabilities=capabilities)
    register_vision_tools(registry, allow=allow, include_gated=include_gated,
                          image_dir=image_dir)
    return registry


# ------------------------------------------------------------- the tool set


def test_the_i3_tool_set_is_declared():
    tools = {tool.name: tool for tool in vision_tools()}
    assert set(tools) == set(WORKING_NAMES) | set(GATED_NAMES)
    for tool in tools.values():
        assert tool.risk is RiskClass.READ_ONLY
        assert tool.capability is Capability.READ


def test_working_and_gated_tools_are_separable():
    assert tuple(t.name for t in working_vision_tools()) == WORKING_NAMES
    assert tuple(t.name for t in gated_vision_tools()) == GATED_NAMES
    for tool in working_vision_tools():
        assert tool.stage_ready is True
    for tool in gated_vision_tools():
        assert tool.stage_ready is False
        assert '#17' in tool.stage_gate or 'Stage G' in tool.stage_gate


def test_vision_schemas_are_closed_and_bounded():
    for tool in vision_tools():
        schema = tool.parameters
        assert schema['type'] == 'object'
        assert schema['additionalProperties'] is False
        assert set(schema['properties']).isdisjoint(RESERVED_PARAM_NAMES)
        for key, spec in schema['properties'].items():
            if spec['type'] == 'string':
                assert 'maxLength' in spec or 'enum' in spec, key
            elif spec['type'] in ('integer', 'number'):
                assert 'minimum' in spec and 'maximum' in spec, key


def test_no_vision_tool_exposes_a_device_or_watchdog_knob(camera):
    registry = _registry(camera)
    for name in WORKING_NAMES:
        for payload in ({'i2c_bus': 7}, {'timeout_sec': 999}, {'pwm': 255},
                        {'gpio': 1}, {'device': '/dev/video0'}):
            with pytest.raises(ToolValidationError):
                registry.invoke(name, payload)
    registry.close()


# ---------------------------------------------------------- deny-by-default


def test_registration_is_deny_by_default(camera):
    registry = _registry(camera, allow=False)
    assert set(registry.names()) == set(WORKING_NAMES) | set(GATED_NAMES)
    assert registry.invocable() == ()
    with pytest.raises(ToolPermissionError):
        registry.invoke('vision_capture', {'save': False})
    registry.close()


def test_read_capability_must_be_granted(camera):
    registry = _registry(camera, capabilities=())
    with pytest.raises(ToolPermissionError):
        registry.invoke('vision_describe_scene')
    registry.grant(Capability.READ)
    assert registry.invoke('vision_describe_scene')['caption'] is None
    registry.close()


def test_unavailable_tools_are_not_offered_to_the_model_by_default(camera):
    registry = _registry(camera, include_gated=False)
    assert set(registry.names()) == set(WORKING_NAMES)
    described = {entry['name'] for entry in registry.describe()}
    assert described == set(WORKING_NAMES)
    for name in GATED_NAMES:
        with pytest.raises(ToolPermissionError):
            registry.invoke(name, {'query': 'door'})
    registry.close()


def test_vision_tools_need_no_motion_interface(camera):
    """A read-only observer must work on a robot with no actuation grant."""
    context = ToolContext(perception=camera)
    assert context.motion is None
    registry = ToolRegistry(context, capabilities=(Capability.READ,))
    register_vision_tools(registry, allow=True)
    assert registry.invoke('vision_capture', {'save': False})['saved'] is False
    registry.close()


# ----------------------------------------------------- capture (fake backend)


def test_capture_works_against_the_fake_backend(camera, tmp_path):
    registry = _registry(camera, image_dir=tmp_path)
    result = registry.invoke('vision_capture', {})
    assert result['height'] == 72
    assert result['width'] == 96
    assert result['channels'] == 3
    assert result['sequence'] == 1
    assert result['timestamp']
    assert result['backend'] == 'fake'
    assert result['mean_intensity'] > 0.0
    assert result['saved'] is True

    written = Path(result['saved_path'])
    assert written.exists()
    assert written.stat().st_size > 0
    assert written.parent == tmp_path
    registry.close()


def test_capture_can_report_without_writing(camera, tmp_path):
    registry = _registry(camera, image_dir=tmp_path)
    result = registry.invoke('vision_capture', {'save': False})
    assert result['saved'] is False
    assert result['saved_path'] is None
    assert list(tmp_path.iterdir()) == []
    registry.close()


def test_capture_directory_can_come_from_the_context(camera, tmp_path):
    registry = _registry(camera, metadata={'image_dir': str(tmp_path)})
    result = registry.invoke('vision_capture', {'label': 'ctx'})
    assert Path(result['saved_path']).parent == tmp_path
    registry.close()


def test_capture_labels_cannot_steer_the_write(camera, tmp_path):
    registry = _registry(camera, image_dir=tmp_path)
    for bad in ('../escape', '/etc/passwd', 'Upper', 'has space', 'x' * 40, ''):
        with pytest.raises(ToolValidationError):
            registry.invoke('vision_capture', {'label': bad})
    assert list(tmp_path.iterdir()) == []

    result = registry.invoke('vision_capture', {'label': 'door-sign_2'})
    written = Path(result['saved_path'])
    assert written.parent == tmp_path
    assert written.name.startswith('door-sign_2_')
    registry.close()


def test_frames_are_never_inlined_into_a_tool_result(camera, tmp_path):
    """Pixels stay on disk; a tool result must be small enough for a prompt."""
    registry = _registry(camera, image_dir=tmp_path)
    for name, payload in (('vision_capture', {'save': False}),
                          ('vision_describe_scene', {}),
                          ('vision_detect_motion', {'samples': 2})):
        result = registry.invoke(name, payload)
        assert result['image_inlined'] is False
        for value in result.values():
            assert not isinstance(value, np.ndarray)
    registry.close()


# ----------------------------------------------------------- describe scene


def test_describe_scene_returns_measurements_not_a_caption(camera):
    registry = _registry(camera)
    result = registry.invoke('vision_describe_scene')
    assert result['caption'] is None
    assert result['caption_available'] is False
    assert result['detections'] == []
    assert result['caption_gate'] == VLM_GATE
    assert result['height'] == 72 and result['width'] == 96
    assert set(result['motion']) == {'changed', 'score', 'threshold'}
    registry.close()


def test_describe_scene_can_reuse_the_buffered_frame(camera):
    counting = CountingCamera(camera)
    registry = _registry(counting)
    registry.invoke('vision_capture', {'save': False})
    assert counting.captures == 1

    result = registry.invoke('vision_describe_scene', {'refresh': False})
    assert counting.captures == 1, 'refresh=False must not grab a new frame'
    assert result['sequence'] == 1

    registry.invoke('vision_describe_scene', {'refresh': True})
    assert counting.captures == 2
    registry.close()


# ------------------------------------------------------------ detect motion


def test_detect_motion_scores_change_across_frames(camera):
    registry = _registry(camera)
    result = registry.invoke('vision_detect_motion', {'samples': 4})
    assert result['samples'] == 4
    assert result['comparisons'] == 3
    assert len(result['scores']) == 4
    assert result['scores'][0] == 0.0, 'first sample only sets the baseline'
    assert result['max_score'] > 0.0, 'the synthetic moving bar should register'
    assert result['method'] == 'grayscale_absdiff_mean'
    registry.close()


def test_detect_motion_reports_a_single_sample_honestly(camera):
    registry = _registry(camera)
    result = registry.invoke('vision_detect_motion', {'samples': 1})
    assert result['comparisons'] == 0
    assert result['scores'] == [0.0]
    assert result['changed'] is False
    registry.close()


def test_detect_motion_sample_count_is_bounded(camera):
    registry = _registry(camera)
    for bad in (0, -1, MAX_MOTION_SAMPLES + 1, 999, 2.5, True):
        with pytest.raises(ToolValidationError):
            registry.invoke('vision_detect_motion', {'samples': bad})
    registry.close()


def test_unknown_and_mistyped_arguments_are_rejected(camera):
    registry = _registry(camera)
    with pytest.raises(ToolValidationError):
        registry.invoke('vision_capture', {'save': 'yes'})
    with pytest.raises(ToolValidationError):
        registry.invoke('vision_capture', {'nope': 1})
    with pytest.raises(ToolValidationError):
        registry.invoke('vision_describe_scene', {'refresh': 1})
    with pytest.raises(ToolValidationError):
        registry.invoke('vision_locate_object', {})
    with pytest.raises(ToolValidationError):
        registry.invoke('vision_read_text', {'region': 'sideways'})
    with pytest.raises(ToolValidationError):
        registry.invoke('vision_read_text', {'min_confidence': 5.0})
    registry.close()


# ------------------------------------------------- honest StageNotReady stubs


@pytest.mark.parametrize('tool', [VisionReadTextTool(), VisionLocateObjectTool()],
                         ids=lambda t: t.name)
def test_ocr_and_grounding_refuse_instead_of_fabricating(tool, camera):
    context = ToolContext(perception=camera)
    arguments = {'query': 'the door'} if tool.name == 'vision_locate_object' else {}
    with pytest.raises(StageNotReady) as excinfo:
        tool.execute(context, arguments)
    message = str(excinfo.value)
    assert '#17' in message
    assert tool.name in message


def test_gated_tools_surface_as_a_named_failure_through_the_registry(camera):
    registry = _registry(camera)
    for name, payload in (('vision_read_text', {}),
                          ('vision_locate_object', {'query': 'the door'})):
        result = registry.dispatch(name, payload)
        assert result.ok is False
        assert result.error_type == 'ToolExecutionError'
        assert 'StageNotReady' in result.error
        assert '#17' in result.error
        with pytest.raises(ToolExecutionError):
            registry.invoke(name, payload)
    registry.close()


def test_a_refusal_has_no_side_effects(camera):
    counting = CountingCamera(camera)
    registry = _registry(counting)
    registry.dispatch('vision_read_text', {'region': 'center'})
    registry.dispatch('vision_locate_object', {'query': 'a chair'})
    assert counting.captures == 0
    assert counting.change_checks == 0
    assert counting.saves == 0
    registry.close()


# ------------------------------------------------------- the perception slot


def test_no_camera_wired_is_a_clear_refusal():
    registry = _registry(None)
    with pytest.raises(VisionUnavailable) as excinfo:
        registry.invoke('vision_capture', {'save': False})
    assert 'perception' in str(excinfo.value)
    registry.close()


def test_the_perception_slot_refuses_an_object_with_a_wheel_level_api():
    class CameraShapedMotor:
        def capture_frame(self):
            return None

        def detect_change(self, image=None):
            return None

        def set_pwm(self, value):  # the tell
            return None

    with pytest.raises(ToolSafetyViolation):
        assert_read_only_perception(CameraShapedMotor())


def test_the_perception_slot_refuses_something_that_is_not_a_camera():
    with pytest.raises(VisionUnavailable):
        assert_read_only_perception(MockMotionInterface())
    with pytest.raises(VisionUnavailable):
        assert_read_only_perception(object())


def test_a_usable_camera_is_accepted(camera):
    assert assert_read_only_perception(camera) is camera


# ---------------------------------------------------------- camera backends


def test_create_camera_service_defaults_to_the_synthetic_backend():
    assert DEFAULT_CAMERA_BACKEND == 'fake'
    service = create_camera_service()
    try:
        assert service.backend_name == 'fake'
        assert service.capture_frame().image.shape[2] == 3
    finally:
        service.close()


def test_create_camera_service_refuses_hardware_without_an_explicit_opt_in():
    """A unit test run must never be what grabs the CSI sensor."""
    for backend in ('gst_csi', 'csi', 'argus', 'jetson', 'webcam', 'usb'):
        assert backend not in SOFTWARE_CAMERA_BACKENDS
        with pytest.raises(VisionUnavailable) as excinfo:
            create_camera_service(backend)
        assert 'allow_hardware' in str(excinfo.value)


def test_software_backends_are_the_only_default_ones():
    assert 'fake' in SOFTWARE_CAMERA_BACKENDS
    assert 'file' in SOFTWARE_CAMERA_BACKENDS
    assert SOFTWARE_CAMERA_BACKENDS.isdisjoint({'gst_csi', 'csi', 'webcam', 'usb'})


@pytest.mark.skip(
    reason='Opens the real CSI sensor via nvarguscamerasrc. The sensor is a single '
           'shared resource and other bring-up stages run concurrently, so this is '
           'a deliberate manual step, never part of the suite. Run it alone with '
           "create_camera_service('gst_csi', allow_hardware=True)."
)
def test_real_csi_capture_is_a_manual_step():  # pragma: no cover
    service = create_camera_service('gst_csi', allow_hardware=True)
    try:
        frame = service.capture_frame()
        assert frame.image.size > 0
    finally:
        service.close()


# ------------------------------------------------- structural guard coverage


def test_the_existing_ast_guard_covers_this_module():
    scanned = sorted(path.name for path in TOOLS_DIR.glob('*.py'))
    assert 'vision_tools.py' in scanned


def test_vision_tools_module_stays_above_the_boundary():
    forbidden_modules = ('jetbot_control', 'jetbot_base', 'jetbot_agent.hardware',
                         'smbus', 'smbus2', 'busio', 'board', 'Jetson', 'RPi',
                         'periphery')
    forbidden_identifiers = {'PCA9685', 'SMBus', 'DiffDriveController', 'MotorDriver',
                             'MockMotorDriver', 'MotorController', 'GPIO',
                             'set_velocity', 'set_pwm', 'set_duty_cycle', 'write_byte',
                             'write_byte_data', 'write_i2c_block_data',
                             'twist_to_wheel_speeds'}
    forbidden_paths = ('/dev/i2c', '/dev/mem', '/sys/class/pwm', '/dev/gpiochip')

    tree = ast.parse(VISION_TOOLS_PATH.read_text(encoding='utf-8'),
                     filename=str(VISION_TOOLS_PATH))
    imported = []
    offenders = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Name) and node.id in forbidden_identifiers:
            offenders.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in forbidden_identifiers:
            offenders.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            offenders.update(path for path in forbidden_paths if path in node.value)

    bad_imports = [name for name in imported
                   if any(name == prefix or name.startswith(prefix + '.')
                          for prefix in forbidden_modules)]
    assert bad_imports == [], bad_imports
    assert offenders == set(), sorted(offenders)


def test_the_module_imports_without_the_perception_path():
    """``perception`` is imported lazily, so the tool package stays importable."""
    import subprocess

    program = (
        'import sys\n'
        'sys.path = [p for p in sys.path if not p.endswith("/src")]\n'
        'import jetbot_agent.agent.tools.vision_tools as v\n'
        'assert v.VisionCaptureTool.name == "vision_capture"\n'
        'assert "perception" not in sys.modules\n'
        'print("OK")\n'
    )
    proc = subprocess.run([sys.executable, '-c', program], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert 'OK' in proc.stdout


def test_tool_instances_are_reusable_across_registries(camera, tmp_path):
    for _ in range(2):
        registry = _registry(camera, image_dir=tmp_path)
        assert registry.invoke('vision_capture', {'save': False})['channels'] == 3
        registry.close()


def test_vision_tool_classes_are_importable_individually():
    assert VisionCaptureTool.name == 'vision_capture'
    assert VisionDescribeSceneTool.name == 'vision_describe_scene'
    assert VisionDetectMotionTool.name == 'vision_detect_motion'
    assert VisionReadTextTool.name == 'vision_read_text'
    assert VisionLocateObjectTool.name == 'vision_locate_object'
