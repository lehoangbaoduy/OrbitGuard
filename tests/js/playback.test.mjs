import test from "node:test";
import assert from "node:assert/strict";
import { loadOG, fixture } from "./helpers.mjs";

const OG = await loadOG();
const P = OG.playback;
const T = 30;

test("advance moves the playhead by speed x elapsed / step length and keeps playing", () => {
  const a = P.advance({ p: 2, playing: true }, 500, { speed: 1, loop: false, T, stepMs: 250 });
  assert.equal(a.p, 4);
  assert.equal(a.playing, true);
  assert.equal(P.advance({ p: 2, playing: true }, 500, { speed: 4, loop: false, T, stepMs: 250 }).p, 10);
});

test("advance stops at the last step when not looping", () => {
  const a = P.advance({ p: 29.5, playing: true }, 2000, { speed: 1, loop: false, T, stepMs: 250 });
  assert.equal(a.p, T);
  assert.equal(a.playing, false);
});

test("advance wraps to the start when looping and keeps the remainder", () => {
  const a = P.advance({ p: 29.5, playing: true }, 1000, { speed: 1, loop: true, T, stepMs: 250 });
  assert.ok(a.playing);
  assert.ok(Math.abs(a.p - 3.5) < 1e-9, `got ${a.p}`);
});

test("advance does nothing while paused and never goes backwards on a negative dt", () => {
  assert.deepEqual(P.advance({ p: 5, playing: false }, 1000, { speed: 1, loop: false, T, stepMs: 250 }), { p: 5, playing: false });
  assert.equal(P.advance({ p: 5, playing: true }, -100, { speed: 1, loop: false, T, stepMs: 250 }).p, 5);
});

test("stepOf and frac split the playhead; the last frame has no fraction", () => {
  assert.equal(P.stepOf(4.75, T), 4);
  assert.equal(P.frac(4.75, T), 0.75);
  assert.equal(P.stepOf(T + 3, T), T);
  assert.equal(P.frac(T, T), 0);
});

test("next/prev event jump between key steps and are null at the ends (buttons disable there)", () => {
  const keys = [3, 9, 20];
  assert.equal(P.nextKey(keys, 0), 3);
  assert.equal(P.nextKey(keys, 9), 20);
  assert.equal(P.nextKey(keys, 20), null);
  assert.equal(P.prevKey(keys, 20), 9);
  assert.equal(P.prevKey(keys, 3), null);
  assert.equal(P.prevKey(keys, 100), 20);
  assert.equal(P.nextKey([], 5), null);
});

test("keys: space toggles, arrows step, shift steps by ten, brackets jump events", () => {
  assert.deepEqual(P.keyAction(" ", {}), { type: "toggle" });
  assert.deepEqual(P.keyAction("ArrowRight", {}), { type: "step", delta: 1 });
  assert.deepEqual(P.keyAction("ArrowLeft", { shift: true }), { type: "step", delta: -10 });
  assert.deepEqual(P.keyAction("[", {}), { type: "event", direction: -1 });
  assert.deepEqual(P.keyAction("]", {}), { type: "event", direction: 1 });
  assert.equal(P.keyAction("x", {}), null);
});

test("keys are ignored while typing and where the focused control already handles them", () => {
  for (const tag of ["INPUT", "TEXTAREA", "SELECT"]) assert.equal(P.keyAction(" ", { tag }), null);
  assert.equal(P.keyAction("ArrowLeft", { editable: true }), null);
  assert.equal(P.keyAction(" ", { tag: "BUTTON" }), null, "a focused button handles space itself");
  assert.equal(P.keyAction("ArrowLeft", { role: "slider" }), null, "the scrubber handles its own arrows");
  assert.deepEqual(P.keyAction("]", { tag: "BUTTON" }), { type: "event", direction: 1 });
});

test("scrubber keys: arrows, shift, page, home and end", () => {
  assert.equal(P.sliderKey("ArrowRight", {}, 10, T), 11);
  assert.equal(P.sliderKey("ArrowLeft", { shift: true }, 15, T), 5);
  assert.equal(P.sliderKey("PageUp", {}, 5, T), T);
  assert.equal(P.sliderKey("PageDown", {}, 5, T), 0);
  assert.equal(P.sliderKey("Home", {}, 9, T), 0);
  assert.equal(P.sliderKey("End", {}, 9, T), T);
  assert.equal(P.sliderKey("q", {}, 9, T), null);
});

test("scrubStep maps a pointer position onto a step and clamps outside the track", () => {
  const rect = { left: 100, width: 300 };
  assert.equal(P.scrubStep(100, rect, T), 0);
  assert.equal(P.scrubStep(250, rect, T), 15);
  assert.equal(P.scrubStep(5000, rect, T), T);
  assert.equal(P.scrubStep(-5, rect, T), 0);
});

test("clockText formats minutes and hh:mm:ss from the step and the recorded step length", () => {
  const c = P.clockText(250, 60);
  assert.equal(c.minutes, 250);
  assert.equal(c.hms, "04:10:00");
  assert.equal(P.clockText(0, 60).hms, "00:00:00");
  assert.equal(P.clockText(3, 30).minutes, 1);
});

test("runStarts finds satellites whose event begins at this step, not ones that continue", () => {
  const ep = OG.decodeEpisode(fixture("shielded").payload);
  for (let t = 1; t <= ep.T; t += 1) {
    const starts = P.runStarts(ep, t);
    for (const i of starts.shield) {
      assert.equal(ep.event[t * ep.N + i], 2);
      assert.notEqual(ep.event[(t - 1) * ep.N + i], 2);
    }
  }
  assert.deepEqual(P.runStarts(ep, 0), { maneuver: [], shield: [], collision: [] });
});

test("jumping across several steps starts nothing (animations are for the step just played)", () => {
  const ep = OG.decodeEpisode(fixture("shielded").payload);
  assert.deepEqual(P.runStartsBetween(ep, 3, 12), { maneuver: [], shield: [], collision: [] });
  assert.deepEqual(P.runStartsBetween(ep, 4, 5), P.runStarts(ep, 5));
});
