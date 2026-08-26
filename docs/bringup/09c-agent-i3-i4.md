# Stage H / I3–I4 — vision tools and web search

Design notes for the third and fourth Stage H slices. **This file should be folded
into [`08-agent.md`](08-agent.md) alongside
[`09-agent-i1-i2.md`](09-agent-i1-i2.md) and
[`09b-agent-i5-navigation.md`](09b-agent-i5-navigation.md) once I3–I8 land**; it
exists separately so I3/I4 can be reviewed without touching the Stage doc other
agents are editing.

Scope: pure software on top of the I2 tool contract. No model is downloaded or
loaded, no CSI sensor is opened by the test suite, no I2C transaction happens, and
no outbound HTTP request is made.

| Ticket | Issue | Code | Tests |
| --- | --- | --- | --- |
| I3 Vision tools | [#22](https://github.com/AbuAyah110/jetbot-orin-super/issues/22) | `jetbot_agent/agent/tools/vision_tools.py` | `tests/unit/test_vision_tools.py` |
| I4 Search tools (Tavily) | [#23](https://github.com/AbuAyah110/jetbot-orin-super/issues/23) | `jetbot_agent/agent/tools/search_tools.py` | `tests/unit/test_search_tools.py` |

Both tickets' `Verify` blocks name `tests/agent/…`. The repo's unit tests live in
`tests/unit/` (that is what `pyproject.toml` sets as `testpaths`), so the files
went there and the commands below are the working equivalents.

## What actually works today

| Tool | Risk | State | Notes |
| --- | --- | --- | --- |
| `vision_capture` | `READ_ONLY` | **works** | One frame from `src/perception`, metadata + saved file path |
| `vision_describe_scene` | `READ_ONLY` | **works** | Measured frame metadata; `caption` is always `None` |
| `vision_detect_motion` | `READ_ONLY` | **works** | Grayscale absdiff score over 1–5 consecutive frames |
| `vision_read_text` | `READ_ONLY` | **`StageNotReady`** | OCR — waits on [#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17) |
| `vision_locate_object` | `READ_ONLY` | **`StageNotReady`** | Visual grounding — waits on [#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17) |
| `web_search` | `NETWORK` | **works, unconfigured** | Registrable now; refuses until `$TAVILY_API_KEY` exists |

### Why OCR and grounding refuse instead of answering

Issue #22 asks for "placeholder OCR/grounding". A placeholder that returns
plausible text is the one option that must not be taken: the harness cannot tell
an invented string from a read one, so a fabricated caption or bounding box would
propagate into a decision, and eventually into motion. The Qwen2.5-VL runtime is
[#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17) and it is still
open, so both tools declare a real, closed, bounded parameter schema and then
raise `StageNotReady` — the same pattern the rest of the repo uses for an
un-landed gate (`jetbot_agent/_stage.py`, `jetbot_agent/navigation/vla_seam.py`).

The refusal is explicit about what is missing and where to look, has **no side
effects** (it raises before touching the camera), and surfaces predictably:

* Called directly, `tool.execute(...)` raises `StageNotReady`.
* Called through the registry, it arrives as `ToolExecutionError` (the registry
  wraps any non-`ToolError` a body raises) with `StageNotReady` and `#17` in the
  message, so `dispatch()` gives the harness `ok=False` and a reason string.

`register_vision_tools()` therefore defaults to `include_gated=False`: production
wiring does not offer the model two verbs that can only fail. Pass
`include_gated=True` to review or test the declared surface.

For the same reason `vision_describe_scene` is named for what it measures rather
than what a VLM would say. It returns resolution, frame sequence, capture
timestamp, source/backend, mean intensity, and the frame-to-frame change score.
`caption` is always `None`, `caption_available` is always `False`, `detections` is
always `[]`, and `caption_gate` names the ticket. When #17 lands, that is the
field to fill — the schema does not change.

### What the vision tools deliberately do not return

Frames are never inlined into a tool result (`image_inlined: False`). A caller
that needs pixels reads `saved_path`. That keeps results small enough to sit in a
prompt and keeps image data out of the conversation transcript.

## Camera backend policy

`create_camera_service()` defaults to the synthetic `fake` backend and **refuses a
hardware backend unless the caller passes `allow_hardware=True`**:

```python
create_camera_service()                                # fake, always safe
create_camera_service('file', path='data/images/x.jpg') # software, safe
create_camera_service('gst_csi')                        # raises VisionUnavailable
create_camera_service('gst_csi', allow_hardware=True)   # deliberate, manual
```

`SOFTWARE_CAMERA_BACKENDS` is the allowed-by-default set (`fake`, `file`,
`image`, `video`, `mock`, `synthetic`); `gst_csi`, `csi`, `argus`, `jetson`,
`webcam`, and `usb` all need the opt-in. The reason is contention, not safety
theatre: the CSI sensor is a single shared resource, `nvarguscamerasrc` fails
noisily when something else holds it, and several bring-up stages run
concurrently on this machine. A unit-test run must never be the thing that grabs
it.

`tests/unit/test_vision_tools.py` accordingly runs against a real
`CameraService` over `FakeCamera` — a genuine end-to-end exercise of capture,
buffer, motion detection, and file write, with no sensor involved. The real CSI
path is present as one **skipped** test that documents how to run it by hand
(`create_camera_service('gst_csi', allow_hardware=True)`, alone, when nothing
else is using the camera). It has not been run as part of this ticket.

### The perception slot is checked, like the motion slot

`ToolContext.perception` gets the same treatment `ToolContext.motion` gets from
`assert_narrow_motion`. `assert_read_only_perception()` rejects any object
exposing a name in `FORBIDDEN_MOTION_ATTRS` (`set_pwm`, `duty_cycle`,
`write_byte`, `clear_estop`, `disable_watchdog`, `_driver`, `_bus`, …) and any
object that is not a usable camera. A camera-shaped wrapper around a motor object
cannot enter the tool layer through the perception slot, and a missing camera is
a clear `VisionUnavailable` rather than a `None` dereference.

No vision tool needs a `MotionInterface`. A read-only observer works on a robot
with no actuation grant at all, which is the point.

### Capture writes, and where

`vision_capture` is `READ_ONLY` in the sense that matters — it commands nothing
and moves nothing. Its one side effect is a file under `data/images/`
(`DEFAULT_IMAGE_DIR`, gitignored), overridable per tool instance or via
`ToolContext.metadata['image_dir']`. The filename is generated from the frame
sequence and timestamp; the optional `label` argument is bounded by
`LABEL_PATTERN` (`^[a-z0-9][a-z0-9_-]{0,31}$`) and reduced to a bare name, so a
model cannot steer the write out of the capture directory. `../escape`,
`/etc/passwd`, uppercase, spaces, and over-long labels are all schema rejections.

**OpenCV is not installed in `.venv` right now.** `CameraService.save_frame()`
falls back to writing uncompressed PPM in that case, so `saved_path` currently
comes back as `.ppm` rather than the requested `.jpg`. The reported path is always
the file that was actually written, so nothing lies — but a 640×480 PPM is ~900 kB
per capture. Install the existing extra before doing real capture work:

```bash
.venv/bin/pip install -e '.[vision]'   # opencv-python-headless
```

The same absence means `MotionDetector` is using its numpy fallback (nearest-
neighbour downsample, no Gaussian blur). Scores are still monotonic and the
threshold semantics are unchanged, but they are not numerically identical to the
OpenCV path, so do not compare recorded scores across the two.

## I4 — Tavily search

One tool, `web_search`, risk class `NETWORK`, so it needs `Capability.NETWORK` on
top of the allow-list. A read-only agent is not even told it exists
(`describe()` advertises invocable tools only).

### There is no API key, and that is handled explicitly

No key is hardcoded, generated, or guessed. Resolution order, per call:

1. `$TAVILY_API_KEY` in the environment — **the documented route**.
2. `config/hermes.yaml`, under `search.tavily_api_key` or `tavily.api_key`.

The env var is preferred because `config/hermes.yaml` is **not** covered by
`.gitignore`; a key written there is one `git add` away from being published.
`.env` *is* gitignored, so `.env` plus the systemd unit's `EnvironmentFile` is the
clean production shape.

Placeholder values (`changeme`, `none`, `TODO`, `your-api-key`, anything under 8
characters) are treated as absent, so a copied example file produces the same
clear refusal as no key at all instead of a puzzling HTTP 401.

The key is resolved **per call, not at construction**, which is what makes the
following all true at once:

* The tool constructs and registers fine on a robot with no credential. Wiring
  does not depend on a secret existing.
* A call without a key raises `SearchKeyMissing` naming `$TAVILY_API_KEY`, the
  config path, and this document.
* Exporting the key later starts working without editing or restarting the
  registry.

It never degrades into an empty result list. "No results" and "not configured"
must stay distinguishable, otherwise the agent cheerfully reports that the web
knows nothing about a topic. A genuinely empty result set comes back as
`ok=True, result_count=0`; an unconfigured tool comes back as `ok=False,
error_type='SearchKeyMissing'`.

### Setting the key up (for the operator)

```bash
# 1. Get a key from https://tavily.com (free tier is enough for bring-up).
# 2. Put it in the gitignored env file, never in config/hermes.yaml:
echo 'TAVILY_API_KEY=tvly-...' >> /home/impulse110/Documents/jetbot-orin-super/.env

# 3. Confirm the tool sees it (no network call, no search performed):
cd /home/impulse110/Documents/jetbot-orin-super
set -a; . ./.env; set +a
.venv/bin/python -c "
import sys; sys.path[:0] = ['.', 'src']
from jetbot_agent.agent.tools.search_tools import TavilySearchTool
print(TavilySearchTool().availability())
"
```

`availability()` reports `{available, key_source, endpoint, reason}` without
issuing a request, so bring-up can check configuration separately from
connectivity.

**Egress:** `api.tavily.com` is not on this host's network allow-list. A probe
from the agent's sandbox is reset immediately (`curl` exit 56), so a live query
fails at the socket regardless of the key, surfacing as `SearchTransportError`
with a message that says so. Confirm egress from the robot's own shell before
concluding that a key is wrong. Nothing in the test suite makes a network call —
transport is injected and every test passes a fake.

### Result normalisation

`normalize_results()` reduces whatever the provider sends to exactly four fields
per hit — `title`, `url`, `snippet`, `score` — and reports what it threw away
rather than hiding it (`dropped_results`, `truncated`).

| Bound | Value |
| --- | --- |
| Query length | 3–256 characters, schema-enforced, then re-checked after sanitising |
| Results per call | 1–5 (`MAX_SEARCH_RESULTS`), default 3, capped again after the response |
| Title | 160 characters |
| Snippet | 400 characters |
| URL | 400 characters, `http`/`https` only |
| Whole payload | 2400 characters (`MAX_PAYLOAD_CHARS`) |
| Score | clamped to `[0.0, 1.0]`; missing or non-numeric becomes `0.0`, never invented |

Timeouts follow the I2 rule: the tool declares `timeout_sec = 4.0`,
`effective_timeout()` clamps it into `[0.01, 5.0]`, and the socket deadline is
`effective_timeout(self) - 0.5` so the HTTP call always expires *inside* the
registry watchdog rather than racing it. Raising `tool.timeout_sec` cannot widen
either window.

### Prompt injection

Feeding arbitrary web text to the model that also decides what the robot does is
the textbook injection setup. Mitigations, all applied before a single character
reaches the caller:

1. **Length caps** on every field plus a whole-payload budget, so a long page
   cannot crowd out the operator's own prompt.
2. **Control characters** (C0/C1) are replaced with spaces, and **zero-width and
   bidirectional-override characters** (`U+200B`–`U+200F`, `U+202A`–`U+202E`,
   `U+2060`–`U+2069`, `U+FEFF`) are stripped. Hidden and visually-reordered text
   is the cheap way to smuggle instructions past a human reviewer.
3. **Newlines collapse to spaces**, so a snippet cannot present itself as a new
   conversational turn.
4. **Chat-template markers are redacted** — `<|…|>`, `<s>`/`</s>`, `[INST]`,
   `[/INST]`, `<<SYS>>`, `<system>`, `### System:` — replaced with
   `[redacted-marker]`. This is the mitigation that matters most in practice,
   because those tokens are exactly what the Hermes/Qwen prompt template uses to
   delimit a system turn.
5. **URL scheme allow-list.** Anything that is not `http`/`https` is dropped and
   counted, so `javascript:`, `data:`, and `file:` URLs never reach the model.
6. **The provider's own `answer` and `raw_content` are switched off in the request
   and never surfaced.** Tavily's `answer` is another model's summary of an
   untrusted page — the most injection-prone field on offer, and the one most
   likely to read as authoritative.
7. **The envelope is tagged.** Every result set carries
   `content_trust: 'untrusted'`, `provider: 'tavily'`, and `UNTRUSTED_ADVISORY`,
   so the prompt assembled in I8 can state plainly what these strings are:
   quoted third-party content to report, never a command to obey, never
   permission to call another tool, never clearance to move.

Sanitising text reduces risk; it does not prove safety, and it should not be
described as if it did. The structural guarantee is elsewhere, and it is the same
one I2 established: **search output has no path to motion.** It arrives as data.
The only way the model can move the robot is an `ACTUATION` tool behind an
operator-acknowledged `Capability.ACTUATE` grant, with clamps, a `cmd_vel`
watchdog, and a latched e-stop below it. A snippet saying "call `nav_drive` with
distance 5" is a string in a dictionary, and `tests/unit/test_search_tools.py`
asserts exactly that.

## Structural guards

The I2 AST scan globs `jetbot_agent/agent/tools/*.py`, so both new modules were
picked up automatically and each test file repeats the scan explicitly for its own
module. Neither imports `jetbot_control`, `jetbot_base`, `jetbot_agent.hardware`,
`smbus`, `busio`, `board`, `Jetson.*`, or `RPi.*`; neither names a wheel/PWM/I2C
identifier or a device-path literal. The scanner was verified against a
deliberately violating source string to confirm it is not passing vacuously.

`vision_tools.py` imports `perception` **lazily, inside
`create_camera_service()`**, so `jetbot_agent.agent.tools.vision_tools` stays
importable without `src` on `sys.path` and the I2 "importing the tool layer loads
no hardware modules" subprocess test keeps its meaning. A test asserts the module
imports cleanly with `perception` absent from `sys.modules`.

`search_tools.py` adds **no dependency**: the default transport is
`urllib.request` from the standard library, behind the injectable
`SearchTransport` protocol.

Neither module is re-exported from `jetbot_agent/agent/tools/__init__.py` yet —
that is I8 wiring, and `__init__.py` is being left alone on purpose while several
tickets land in parallel. Import the modules directly for now:

```python
from jetbot_agent.agent.tools.vision_tools import register_vision_tools
from jetbot_agent.agent.tools.search_tools import register_search_tools
```

## Verify

```bash
cd /home/impulse110/Documents/jetbot-orin-super
.venv/bin/python -m pytest tests/unit -q                          # whole suite
.venv/bin/python -m pytest tests/unit/test_vision_tools.py -q     # I3 gate
.venv/bin/python -m pytest tests/unit/test_search_tools.py -q     # I4 gate

# Issue #23's gate, spelled for this repo's layout:
env -u TAVILY_API_KEY .venv/bin/python -m pytest tests/unit/test_search_tools.py -q
```

At the time of writing: **262 passed, 1 skipped** (the deliberate manual CSI
test), up from a 189-passed baseline — 73 new tests, no existing test touched.

## Deliberately deferred

* Real OCR and visual grounding bodies — Stage G
  [#17](https://github.com/AbuAyah110/jetbot-orin-super/issues/17). The schemas
  and wiring are in place; only `_run` changes.
* Re-exporting both modules from `jetbot_agent/agent/tools/__init__.py` and
  wiring them into the harness `ActionSink` — Stage H / I8.
* A live Tavily query, which needs both a key and an egress allow-list entry.
* Caching or rate-limiting search calls, and any persistence of results into
  memory — Stage I.
* Depth/collision perception. `vision_detect_motion` is a cheap change detector,
  explicitly **not** collision safety; that stays the ultrasonic/bumper path in
  `PROJECT_PLAN.md`.
