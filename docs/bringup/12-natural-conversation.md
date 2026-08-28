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

Long-term facts use the separate CPU BGE + LanceDB path. Say “remember that…”
to store a fact explicitly; general conversation is not silently promoted to
long-term memory. Relevant top-k results are quoted into parked conversation
only. This does not provide internet access or knowledge of events newer than
the model. Questions requiring unavailable hardware, tools, or current online
information should receive a brief limitation or clarification instead of a
fabricated answer.

The camera search is provisional look-then-move, not guaranteed collision
avoidance. It never drives continuously; uncertain or malformed path checks
stop. A ToF sensor and bumper remain required for dependable pre-impact and
contact protection.
