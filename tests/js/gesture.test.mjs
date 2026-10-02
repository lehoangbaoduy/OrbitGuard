import test from "node:test";
import assert from "node:assert/strict";
import { loadOG } from "./helpers.mjs";

const OG = await loadOG();
const G = OG.gesture;

test("a drag is active from pointer down until pointer up", () => {
  const tracker = G.createTracker();
  assert.equal(tracker.active(0), false);
  tracker.down();
  assert.equal(tracker.active(10), true);
  assert.equal(tracker.up(), true, "up reports that a gesture just ended");
  assert.equal(tracker.active(20), false);
  assert.equal(tracker.up(), false, "a second up has nothing to end");
});

test("a wheel gesture holds the scene still for a short moment after the last tick", () => {
  const tracker = G.createTracker();
  tracker.wheel(1000);
  assert.equal(tracker.active(1200), true);
  assert.equal(tracker.active(1000 + G.WHEEL_HOLD_MS + 1), false);
  assert.equal(tracker.settled(1000 + G.WHEEL_HOLD_MS + 1), true);
});

test("settled() is true exactly once after a gesture, so the single full redraw happens once", () => {
  const tracker = G.createTracker();
  tracker.down();
  tracker.up();
  assert.equal(tracker.settled(5), true);
  assert.equal(tracker.settled(6), false);
});

function fakePlot(n, { traces = true } = {}) {
  const calls = { updates: [], redraws: 0 };
  const fullData = Array.from({ length: n }, (_, k) => ({ uid: `u${k}`, type: k === 6 ? "mesh3d" : "scatter3d", x: [0, 0], y: [0, 0], z: [0, 0], line: { color: "#000" } }));
  const sceneTraces = Object.fromEntries(fullData.map((d) => [d.uid, { update: (ft) => calls.updates.push([ft.uid, ft.x.slice()]) }]));
  const el = { _fullData: fullData, _fullLayout: { scene: { _scene: { traces: traces ? sceneTraces : undefined, glplot: { redraw: () => { calls.redraws += 1; } } } } } };
  return { el, calls, fullData };
}
const dataFor = (n, x) => Array.from({ length: n }, (_, k) => ({ type: k === 6 ? "mesh3d" : "scatter3d", x: x[k] ?? [1, 2], y: [3, 4], z: [5, 6], line: { color: "#fff" }, i: [0], j: [1], k: [2], vertexcolor: ["red"] }));

test("applyInPlace updates only the dynamic traces and redraws once without touching the camera", () => {
  const { el, calls } = fakePlot(10);
  assert.equal(G.applyInPlace(el, dataFor(10, {})), true);
  assert.deepEqual(calls.updates.map((u) => u[0]), ["u4", "u5", "u6", "u7", "u8", "u9"], "static traces 0-3 are left alone");
  assert.equal(calls.redraws, 1);
  assert.deepEqual(calls.updates[0][1], [1, 2]);
});

test("applyInPlace refuses when the trace count changed or Plotly internals are missing (caller does a full redraw)", () => {
  const a = fakePlot(10);
  assert.equal(G.applyInPlace(a.el, dataFor(9, {})), false);
  assert.equal(G.applyInPlace({}, dataFor(10, {})), false);
  const b = fakePlot(10, { traces: false });
  assert.equal(G.applyInPlace(b.el, dataFor(10, {})), false);
  assert.equal(a.calls.redraws, 0);
});

test("a mesh whose vertex count changed is skipped until the full redraw; a line trace may change length", () => {
  const { el, calls } = fakePlot(10);
  const data = dataFor(10, { 6: [1, 2, 3, 4, 5], 5: [1, 2, 3] });
  G.applyInPlace(el, data);
  assert.ok(!calls.updates.some((u) => u[0] === "u6"));
  assert.ok(calls.updates.some((u) => u[0] === "u5"));
});

test("one failing trace update does not stop the others or the redraw", () => {
  const { el, calls, fullData } = fakePlot(10);
  el._fullLayout.scene._scene.traces[fullData[5].uid].update = () => { throw new Error("gl"); };
  assert.equal(G.applyInPlace(el, dataFor(10, {})), true);
  assert.ok(calls.updates.some((u) => u[0] === "u7"));
  assert.equal(calls.redraws, 1);
});

test("strict mode (playback frames) refuses, touching nothing, when a mesh changed vertex count", () => {
  const { el, calls } = fakePlot(10);
  assert.equal(G.applyInPlace(el, dataFor(10, { 6: [1, 2, 3, 4, 5] }), { strict: true }), false);
  assert.equal(calls.updates.length, 0);
  assert.equal(calls.redraws, 0);
  assert.equal(G.applyInPlace(el, dataFor(10, { 5: [1, 2, 3] }), { strict: true }), true);
});
