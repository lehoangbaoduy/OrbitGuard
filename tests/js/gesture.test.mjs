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

function withPlates(plot, count) {
  const calls = { drawn: 0, textsWhenDrawn: [] };
  const plates = Array.from({ length: count }, (_, k) => ({ x: k, y: k, z: k, text: `old${k}`, bordercolor: "#000", _pdata: [1], _xa: {} }));
  const scene = plot.el._fullLayout.scene._scene;
  plot.el._fullLayout.scene.annotations = plates;
  scene.drawAnnotations = () => { calls.drawn += 1; calls.textsWhenDrawn.push(plates.map((p) => p.text)); };
  const redraw = scene.glplot.redraw;
  scene.glplot.redraw = () => { redraw(); scene.drawAnnotations(scene); }; // like Plotly's render(), which ends by drawing the annotations
  return { calls, plates };
}
const plateFor = (k) => ({ x: 10 + k, y: 20 + k, z: 30 + k, text: `new${k}`, bordercolor: "#fff", font: { color: "#fff" } });

test("playback frames move the label plates in place when the plate count is unchanged (no full react needed)", () => {
  const plot = fakePlot(10);
  const { calls, plates } = withPlates(plot, 2);
  assert.equal(G.applyInPlace(plot.el, dataFor(10, {}), { strict: true, annotations: [plateFor(0), plateFor(1)] }), true);
  assert.deepEqual(plates.map((p) => [p.x, p.text, p.bordercolor]), [[10, "new0", "#fff"], [11, "new1", "#fff"]]);
  assert.equal(plates[0]._pdata[0], 1, "Plotly's derived fields survive the update");
  assert.equal(calls.drawn, 1, "the plates are drawn once per frame, by the redraw itself");
  assert.deepEqual(calls.textsWhenDrawn[0], ["new0", "new1"], "the new values are in place before that draw");
});

test("without a redraw hook the plates are drawn explicitly", () => {
  const plot = fakePlot(10);
  const { calls } = withPlates(plot, 1);
  delete plot.el._fullLayout.scene._scene.glplot.redraw;
  assert.equal(G.applyInPlace(plot.el, dataFor(10, {}), { annotations: [plateFor(0)] }), true);
  assert.equal(calls.drawn, 1);
});

test("a different plate count cannot be done in place: playback frames ask for a full redraw, a gesture just leaves the plates", () => {
  const strictPlot = fakePlot(10);
  const strict = withPlates(strictPlot, 2);
  assert.equal(G.applyInPlace(strictPlot.el, dataFor(10, {}), { strict: true, annotations: [plateFor(0)] }), false);
  assert.equal(strictPlot.calls.updates.length, 0, "nothing was pushed before refusing");
  assert.equal(strict.calls.drawn, 0);
  const dragPlot = fakePlot(10);
  const drag = withPlates(dragPlot, 2);
  assert.equal(G.applyInPlace(dragPlot.el, dataFor(10, {}), { annotations: [plateFor(0)] }), true);
  assert.deepEqual(drag.plates.map((p) => p.text), ["old0", "old1"]);
});

test("without annotations the plate values are left untouched", () => {
  const plot = fakePlot(10);
  const { plates } = withPlates(plot, 1);
  assert.equal(G.applyInPlace(plot.el, dataFor(10, {})), true);
  assert.deepEqual(plates.map((p) => p.text), ["old0"]);
});

const sceneWith = (glplot, gl) => ({ _fullLayout: { scene: { _scene: { glplot, gl } } } });

test("a scene whose WebGL context was lost is detected, from Plotly's flag or from the context itself", () => {
  assert.equal(G.contextLost(sceneWith({ contextLost: true })), true);
  assert.equal(G.contextLost(sceneWith({}, { isContextLost: () => true })), true);
  assert.equal(G.contextLost(sceneWith({ contextLost: false }, { isContextLost: () => false })), false);
});

test("a missing or not-yet-drawn scene is not reported as lost", () => {
  assert.equal(G.contextLost(null), false);
  assert.equal(G.contextLost({}), false);
  assert.equal(G.contextLost({ _fullLayout: {} }), false);
  assert.equal(G.contextLost(sceneWith(undefined)), false);
});

test("a context that throws when asked is treated as lost (the scene cannot be trusted)", () => {
  assert.equal(G.contextLost(sceneWith({}, { isContextLost: () => { throw new Error("gone"); } })), true);
});

test("Plotly's own context recovery is switched off, so it cannot race the stage's rebuild", () => {
  let called = 0;
  const el = sceneWith({});
  el._fullLayout.scene._scene.recoverContext = () => { called += 1; };
  G.disableRecovery(el);
  el._fullLayout.scene._scene.recoverContext();
  assert.equal(called, 0);
  assert.doesNotThrow(() => G.disableRecovery(null));
  assert.doesNotThrow(() => G.disableRecovery({}));
});


test("hover text, click data and opacity follow the coordinates in place (they used to wait for the periodic full redraw)", () => {
  const { el, fullData } = fakePlot(10);
  const data = dataFor(10, { 8: [1, 2, 3] }).map((d, k) => (k === 8 ? { ...d, hovertext: ["a", "b", "c"], customdata: [4, 5, 7], opacity: 0.6 } : d));
  assert.equal(G.applyInPlace(el, data, { strict: true }), true);
  assert.deepEqual(fullData[8].hovertext, ["a", "b", "c"]);
  assert.deepEqual(fullData[8].customdata, [4, 5, 7], "a click on the third marker must name the third satellite now, not at the last full redraw");
  assert.equal(fullData[8].opacity, 0.6);
});

test("a trace without those fields keeps whatever it had", () => {
  const { el, fullData } = fakePlot(10);
  fullData[8].opacity = 0.9;
  assert.equal(G.applyInPlace(el, dataFor(10, {})), true);
  assert.equal(fullData[8].opacity, 0.9);
});

test("rebuilds after a lost context are limited, so a flapping GPU ends in the failure state instead of a redraw loop", () => {
  const limit = { max: 3, windowMs: 10000 };
  let history = [];
  for (const now of [0, 1000, 2000]) { const r = G.rebuildAllowed(history, now, limit); assert.equal(r.allowed, true); history = r.history; }
  assert.equal(G.rebuildAllowed(history, 3000, limit).allowed, false);
  assert.equal(G.rebuildAllowed(history, 10500, limit).allowed, true, "old losses age out of the window");
});
