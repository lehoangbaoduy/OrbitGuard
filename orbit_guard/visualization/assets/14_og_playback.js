/* OrbitGuard replay stage - playback logic as pure functions: the clock, event jumps, keyboard mapping, scrubber maths.
   The boot module owns the state; nothing here touches the DOM. */
(function (root) {
  "use strict";
  const OG = root.OG;
  const SLIDER_PAGE_STEPS = 30, SHIFT_STEPS = 10;
  const EDITABLE_TAGS = new Set(["INPUT", "TEXTAREA", "SELECT"]);
  const NO_RUNS = () => ({ maneuver: [], shield: [], collision: [] });
  const EVENT_NAMES = { 1: "maneuver", 2: "shield", 3: "collision" };

  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

  /* One tick of the playhead. `p` is continuous (steps); 1x is one step per `stepMs`, speed scales it. */
  function advance(state, dtMs, { speed, loop, T, stepMs }) {
    if (!state.playing || dtMs <= 0) return { p: state.p, playing: state.playing };
    const next = state.p + (dtMs * speed) / stepMs;
    if (next < T) return { p: next, playing: true };
    return loop ? { p: next % T, playing: true } : { p: T, playing: false };
  }

  const stepOf = (p, T) => Math.min(T, Math.floor(p));
  const frac = (p, T) => (p >= T ? 0 : p - Math.floor(p));

  const nextKey = (keys, step) => keys.find((k) => k > step) ?? null;
  const prevKey = (keys, step) => [...keys].reverse().find((k) => k < step) ?? null;

  /* Global shortcuts; `target` describes the focused element. A focused button handles Space itself and the scrubber
     handles its own arrows, so neither is stolen here. */
  function keyAction(key, { shift = false, tag = "", editable = false, role = "" } = {}) {
    if (editable || EDITABLE_TAGS.has(tag)) return null;
    if (key === " ") return tag === "BUTTON" || role === "slider" ? null : { type: "toggle" };
    if (key === "[") return { type: "event", direction: -1 };
    if (key === "]") return { type: "event", direction: 1 };
    if ((key === "ArrowLeft" || key === "ArrowRight") && role !== "slider") return { type: "step", delta: (key === "ArrowRight" ? 1 : -1) * (shift ? SHIFT_STEPS : 1) };
    return null;
  }

  /* Keys on the scrubber itself: arrows, shift = 10, page = 30, home/end. Returns the new step or null. */
  function sliderKey(key, { shift = false }, current, T) {
    const unit = shift ? SHIFT_STEPS : 1;
    const moves = { ArrowLeft: -unit, ArrowDown: -unit, ArrowRight: unit, ArrowUp: unit, PageDown: -SLIDER_PAGE_STEPS, PageUp: SLIDER_PAGE_STEPS };
    if (key in moves) return clamp(current + moves[key], 0, T);
    if (key === "Home") return 0;
    if (key === "End") return T;
    return null;
  }

  const scrubStep = (clientX, rect, T) => Math.round(clamp((clientX - rect.left) / rect.width, 0, 1) * T);

  const pad2 = (n) => String(n).padStart(2, "0");
  function clockText(step, controlDtS) {
    const seconds = Math.round(step * controlDtS);
    return { minutes: Math.floor(seconds / 60), hms: `${pad2(Math.floor(seconds / 3600))}:${pad2(Math.floor((seconds % 3600) / 60))}:${pad2(seconds % 60)}` };
  }

  /* Satellites whose event BEGINS at `step` (its code differs from the previous step's): the one-shot animations. */
  function runStarts(ep, step) {
    const out = NO_RUNS();
    if (step < 1) return out;
    for (let i = 0; i < ep.N; i += 1) {
      const now = ep.event[step * ep.N + i], before = ep.event[(step - 1) * ep.N + i];
      if (now !== before && EVENT_NAMES[now]) out[EVENT_NAMES[now]].push(i);
    }
    return out;
  }

  /* Animations belong to the step just played; a jump over several steps starts none. */
  const runStartsBetween = (ep, from, to) => (to === from + 1 ? runStarts(ep, to) : NO_RUNS());

  OG.playback = { advance, stepOf, frac, nextKey, prevKey, keyAction, sliderKey, scrubStep, clockText, runStarts, runStartsBetween };
})(globalThis);
