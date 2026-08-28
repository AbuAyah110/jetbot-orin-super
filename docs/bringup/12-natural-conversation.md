# Natural conversation and visual follow-ups

JetBot's live loop has three parked conversational paths:

1. General questions use Cosmos with text-only rolling history.
2. “What do you see?” describes one fresh 448² frame.
3. Visual follow-ups such as “what color is it?” or “tell me about that
   object” use one fresh frame plus the bounded text history.
4. “Look around the room for the blue object” runs a bounded camera search:
   up to six fresh viewpoints, short stopped turns, and at most one short
   relocation after a separate conservative path-clear image check.

All three hold the motors stopped. Conversational model output passes through
a `speak`-only gate; a generated drive action is rejected.

## Motion requests never reach a conversational route

A motion verb at the head of an utterance now vetoes every parked question
route (`is_motion_command`). Without that veto, “move around the object in
front of you” matched the visual-question pattern on the literal substring
“object in front”, so a drive command was answered from a stopped turn. Cosmos
replied “I am moving around it”, nothing moved, and the transcript showed a
successful turn. “Move around the box” matched no pattern at all and reached
general conversation with the same result.

Prompt wording alone is not enough here, so the speak-only routes also carry a
deterministic check: a first-person motion claim, or a bare acknowledgement such
as “Okay, turning around it now”, is replaced with a refusal that names what
JetBot can actually do. A route that cannot move cannot report movement.

## Going around an object

“Go around the box”, “drive past the object”, and “circle the chair” route to a
bounded detour: swing away from the target, two short forward pulses, then turn
back by the same pulse count to restore the original heading. Turn degrees per
pulse are not calibrated, so the camera decides when the swing is wide enough
rather than a computed angle.

The detour is only offered for red, blue, or green targets, because
`color_corridor_clear` (pixel arithmetic in `color_grounding.py`) is the only
near-field perception on this robot that works. Any other target gets an honest
refusal naming the missing distance sensor. The corridor check fails closed when
the colour fills the frame, which is what a target at point-blank range looks
like.

### The monocular path gate does not work

`camera_path_clear` asks Cosmos whether the floor ahead is clear. Measured with
`scripts/bringup/probe_path_gate.py` over four saved frames, two of empty floor
and two with a bottle filling the view:

| Prompt wording | empty floor called clear | blocked frame called blocked |
| --- | --- | --- |
| deployed, cautious | 0/4 | 4/4 |
| permissive rewrite | 4/4 | 0/4 |
| name objects on the floor | 0/4 | 4/4 |

Each variant is a constant that follows the prompt's tone. Asked to name what
was on the floor, the model answered “detector” and “path_gate” — words from its
own system prompt — and called both blocked frames “floor”. Cosmos-Reason2-2B at
448² does not perceive near-field floor obstacles here.

The gate is therefore kept only where a `false` answer means “skip an optional
move”, as in the camera search relocation. No maneuver should depend on it for
safety until the ToF sensor and bumper are fitted.

Frames are also settled (`CsiJpeg448.settle`) before any decision that follows a
motion pulse. A frame pulled immediately after a turn is motion-blurred and its
exposure is still adapting, which washes out the colour dominance the corridor
check measures.

## Short-term memory

The loop stores at most five user/assistant exchanges and injects at most 1200
characters into the next prompt. It stores the words JetBot actually spoke,
not model JSON, hidden reasoning, audio, or images. The text window is saved at
`data/runtime/chat_history.json`, so a service restart or battery power cycle
can continue the immediate topic.

Old camera frames are never replayed. Every visual question captures the
current view. JetBot is prompted to say when a requested detail is absent,
hidden, blurry, or uncertain.

## Examples

- “Who are you?”
- “Why is Saturn interesting?”
- “What were we talking about?”
- “What do you see?”
- “What color is that object?”
- “Can you identify the blue thing?”
- “What do you think of it?”
- “How many objects are there?”
- “Move around the room and look for the blue object.”
- “Go around the blue object.” (detour; refuses for uncoloured targets)

Long-term facts use the separate CPU BGE + LanceDB path. Say “remember that…”
to store a fact explicitly; general conversation is not silently promoted to
long-term memory. Relevant top-k results are quoted into parked conversation
only. This does not provide internet access or knowledge of events newer than
the model. Questions requiring unavailable hardware, tools, or current online
information should receive a brief limitation or clarification instead of a
fabricated answer.

The camera search and the detour are provisional look-then-move, not collision
avoidance. Neither drives continuously. Both stop when their evidence is
uncertain, and the detour tracks only its own named target: it cannot see any
other obstacle. A ToF sensor and bumper remain required for dependable
pre-impact and contact protection, and are now the blocking item for any
general “move around the room” behaviour.
