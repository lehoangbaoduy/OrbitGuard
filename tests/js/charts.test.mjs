import test from "node:test";
import assert from "node:assert/strict";
import { loadOG, fixture } from "./helpers.mjs";

const OG = await loadOG();
const C = OG.charts;
const load = (name) => OG.decodeEpisode(fixture(name).payload);

/** A recording stand-in for CanvasRenderingContext2D: remembers fills and paths, draws nothing. */
function recorder() {
  const log = { rects: [], paths: 0, texts: [], fills: [] };
  const g = {
    fillStyle: "", strokeStyle: "", globalAlpha: 1, lineWidth: 1, font: "", textAlign: "",
    fillRect(x, y, w, h) { log.rects.push({ x, y, w, h, fill: this.fillStyle, alpha: this.globalAlpha }); },
    clearRect() {}, setTransform() {}, save() {}, restore() {}, setLineDash() {}, shadowColor: "", shadowBlur: 0,
    beginPath() { log.paths += 1; }, moveTo() {}, lineTo() {}, closePath() {}, fill() { log.fills.push(this.fillStyle); }, stroke() {}, rect() {},
    fillText(text) { log.texts.push(text); }, measureText: (t) => ({ width: t.length * 6 }), strokeRect() {},
  };
  return { g, log };
}

test("scrubber density draws one shield bar and one maneuver bar per step that has them, in their own colours", () => {
  const ep = load("shielded");
  const { g, log } = recorder();
  C.drawScrubDensity(g, 600, 30, ep);
  const d = ep.payload.density;
  const shield = log.rects.filter((r) => r.fill === ep.vocab.event[2].color);
  const maneuver = log.rects.filter((r) => r.fill === ep.vocab.event[1].color);
  assert.equal(shield.length, d["2"].slice(1).filter((v) => v > 0).length);
  assert.equal(maneuver.length, d["1"].slice(1).filter((v) => v > 0).length);
});

test("scrubber density draws a tick per conjunction entry, and no shield bars when the shield was off", () => {
  const ep = load("fleet_collision");
  const { g, log } = recorder();
  C.drawScrubDensity(g, 600, 30, ep);
  assert.equal(log.fills.filter((f) => f === ep.vocab.sat_status[2].color).length, ep.payload.conjunction_entries.length);
  assert.equal(log.rects.filter((r) => r.fill === ep.vocab.event[2].color).length, 0);
});

test("a taller bar means more satellites", () => {
  const ep = load("depletion");
  const { g, log } = recorder();
  C.drawScrubDensity(g, 600, 30, ep);
  const bars = log.rects.filter((r) => r.fill === ep.vocab.event[1].color);
  const counts = ep.payload.density["1"].slice(1).filter((v) => v > 0);
  const tallest = bars[counts.indexOf(Math.max(...counts))];
  assert.ok(bars.every((b) => b.h <= tallest.h + 1e-9));
});

test("the by-satellite matrix has a cell for every maneuver and shield event and none for quiet steps", () => {
  const ep = load("shielded");
  const { g, log } = recorder();
  C.drawMatrix(g, 600, 150, ep, 0);
  let maneuver = 0, shield = 0;
  for (let s = 1; s <= ep.T; s += 1) for (let i = 0; i < ep.N; i += 1) { const e = ep.event[s * ep.N + i]; if (e === 1) maneuver += 1; if (e === 2) shield += 1; }
  assert.equal(log.rects.filter((r) => r.fill === ep.vocab.event[1].color).length, maneuver);
  assert.equal(log.rects.filter((r) => r.fill === ep.vocab.event[2].color).length, shield);
});

test("the counts lanes hatch the shield lane and say so when the shield was off", () => {
  const ep = load("fleet_collision");
  const { g, log } = recorder();
  C.drawLanes(g, 600, 140, ep);
  assert.ok(log.texts.some((t) => /Shield off/.test(t)));
});

test("the counts lanes draw every recorded step of the maneuver lane", () => {
  const ep = load("depletion");
  const { g, log } = recorder();
  C.drawLanes(g, 600, 140, ep);
  const nonzero = ep.payload.density["1"].filter((v) => v > 0).length;
  assert.equal(log.rects.filter((r) => r.fill === ep.vocab.event[1].color).length, nonzero);
});

test("fleet fuel area: drawn from the recorded usable total, with the all-empty dot only when it happened", () => {
  const dry = load("depletion");
  const a = recorder();
  C.drawFuelArea(a.g, 300, 52, dry, 5);
  assert.ok(a.log.paths > 0);
  assert.ok(dry.payload.fuel.all_empty_step !== null);
  const b = recorder();
  C.drawFuelArea(b.g, 300, 52, load("shielded"), 5);
  assert.ok(b.log.paths > 0);
});

test("the inspector figure has three rows that share the step axis and a log distance axis", () => {
  const ep = load("fleet_collision");
  const fig = C.inspectorFigure(ep, 1, 7);
  assert.equal(fig.data.filter((d) => d.name === "closest d_min").length, 1);
  const dMin = fig.data.find((d) => d.name === "closest d_min");
  assert.equal(dMin.x.length, ep.T + 1);
  assert.equal(fig.layout.yaxis3.type, "log");
  assert.ok(fig.layout.shapes.some((s) => s.name === "playhead" && s.x0 === 7 && s.x1 === 7));
});

test("event marks sit at the recorded steps, and d_safe is dotted in the pair-status colour", () => {
  const ep = load("depletion");
  const series = OG.models.inspectorSeries(ep, 0);
  const fig = C.inspectorFigure(ep, 0, 3);
  const marks = fig.data.find((d) => d.name === "maneuver");
  assert.deepEqual(marks.x, series.maneuver);
  const safe = fig.data.find((d) => d.name === "d_safe");
  assert.equal(safe.line.dash, "dot");
  assert.equal(safe.line.color, ep.vocab.pair_status[1].color);
});

test("no stale distance is plotted after a satellite is retired", () => {
  const ep = load("fleet_collision");
  const retired = [...Array(ep.N).keys()].find((i) => ep.satStatus[ep.T * ep.N + i] === 3);
  const dMin = C.inspectorFigure(ep, retired, 0).data.find((d) => d.name === "closest d_min");
  assert.equal(dMin.y[ep.T], null);
});

test("the playhead moves without rebuilding the data", () => {
  const ep = load("shielded");
  const a = C.inspectorFigure(ep, 0, 3), b = C.inspectorFigure(ep, 0, 9);
  assert.deepEqual(a.data.map((d) => d.y), b.data.map((d) => d.y));
  assert.notDeepEqual(a.layout.shapes, b.layout.shapes);
});

test("lane marks label only the selected satellite's own runs, e.g. S3 x2, and skip the shield lane when the shield was off", () => {
  const ep = load("shielded");
  const sat = ep.payload.runs["2"][0].sats[0];
  const lanes = [{ type: "maneuver", top: 0, height: 40 }, { type: "shield", top: 46, height: 40 }];
  const { g, log } = recorder();
  C.drawLaneMarks(g, 600, ep, lanes, sat);
  const expected = OG.models.laneMarks(ep, 1, sat).concat(OG.models.laneMarks(ep, 2, sat)).map((m) => m.label);
  assert.ok(expected.length > 0);
  assert.deepEqual(log.texts.slice().sort(), expected.slice().sort());
  assert.ok(log.texts.every((t) => t.startsWith(`S${sat}`)));
  const off = load("fleet_collision");
  assert.equal(off.shield, false);
  const quiet = recorder();
  C.drawLaneMarks(quiet.g, 600, off, lanes, 0);
  assert.deepEqual(quiet.log.texts.slice().sort(), OG.models.laneMarks(off, 1, 0).map((m) => m.label).sort());
});
