import test from "node:test";
import assert from "node:assert/strict";
import { loadOG, fixture } from "./helpers.mjs";

const OG = await loadOG();
const M = OG.models;
const load = (name) => OG.decodeEpisode(fixture(name).payload);

// ------------------------------------------------------------------ fleet fuel
test("fuel bars follow each satellite's own tank: offline, limited and empty are told apart", () => {
  const ep = load("capability");
  const model = M.fuelBars(ep, ep.T);
  const kinds = model.bars.map((b) => b.kind);
  assert.ok(kinds.includes("offline"), "capability fixture has an offline thruster");
  for (const bar of model.bars.filter((b) => b.kind === "offline")) {
    assert.equal(bar.label, "OFF");
    assert.equal(bar.fillFrac, 0);
  }
  for (const bar of model.bars) assert.ok(bar.trackFrac > 0 && bar.trackFrac <= 1 && bar.fillFrac >= 0 && bar.fillFrac <= 1);
});

test("a limited tank has a shorter track than a full one", () => {
  const ep = load("capability");
  const budgets = ep.payload.fuel.budget_mps;
  const model = M.fuelBars(ep, 0);
  const longest = Math.max(...budgets);
  model.bars.forEach((bar, i) => assert.ok(Math.abs(bar.trackFrac - budgets[i] / longest) < 1e-9));
});

test("the fleet total counts usable tanks only and empty tanks are counted against usable ones", () => {
  const ep = load("capability");
  const model = M.fuelBars(ep, ep.T);
  const usable = model.bars.filter((b) => b.kind !== "offline");
  const sum = usable.reduce((a, b) => a + b.remaining, 0);
  assert.ok(Math.abs(model.total - sum) < 1e-3);
  assert.equal(model.usableCount, usable.length);
  assert.equal(model.text, `${model.emptyCount} of ${model.usableCount} tanks empty`);
  assert.equal(model.emptyCount, model.bars.filter((b) => b.kind === "empty").length);
});

test("a tank that has run dry reads EMPTY", () => {
  const ep = load("depletion");
  const model = M.fuelBars(ep, ep.T);
  const empty = model.bars.filter((b) => b.kind === "empty");
  assert.ok(empty.length > 0, "the always-raise policy exhausts tanks");
  assert.ok(empty.every((b) => b.label === "EMPTY" && b.remaining <= 1e-6));
});

// ------------------------------------------------------------------ inspector
test("step 0 has no decision yet: the inspector says actions exist from step 1", () => {
  const ep = load("shielded");
  const model = M.inspector(ep, 0, 0);
  assert.equal(model.noActions, true);
  assert.equal(model.decision, null);
});

test("the decision line reads proposed → approved from the catalog, with the three latest earlier decisions", () => {
  const ep = load("depletion");
  let step = -1, sat = -1;
  for (let t = ep.T; t >= 1 && step < 0; t -= 1) for (let i = 0; i < ep.N; i += 1) if (ep.event[t * ep.N + i] === 1) { step = t; sat = i; break; }
  const model = M.inspector(ep, step, sat);
  const k = (step - 1) * ep.N + sat;
  assert.equal(model.decision.proposed, ep.catalog[ep.proposed[k]].short);
  assert.equal(model.decision.approved, ep.catalog[ep.approved[k]].short);
  assert.ok(model.earlier.length <= 3);
  for (const e of model.earlier) assert.ok(e.step < step);
});

test("closest partner comes from the recorded partner arrays and is absent for a retired satellite", () => {
  const ep = load("fleet_collision");
  const survivor = [...Array(ep.N).keys()].find((i) => ep.satStatus[ep.T * ep.N + i] !== 3);
  const retired = [...Array(ep.N).keys()].find((i) => ep.satStatus[ep.T * ep.N + i] === 3);
  const alive = M.inspector(ep, ep.T, survivor);
  assert.ok(alive.closest && alive.closest.partner >= 0 && Number.isFinite(alive.closest.dMin));
  const gone = M.inspector(ep, ep.T, retired);
  assert.equal(gone.closest, null);
  assert.equal(gone.retired, true);
  assert.match(gone.closestNote, /retired/);
});

test("readouts show an em dash instead of a number that does not exist", () => {
  const ep = load("fleet_collision");
  const retired = [...Array(ep.N).keys()].find((i) => ep.satStatus[ep.T * ep.N + i] === 3);
  assert.equal(M.inspector(ep, ep.T, retired).closestText, "—");
});

test("capability is reported for the selected satellite", () => {
  const ep = load("capability");
  const offline = ep.payload.capability.offline[0];
  assert.equal(M.inspector(ep, 3, offline).capability, "offline");
  const normal = [...Array(ep.N).keys()].find((i) => !ep.payload.capability.offline.includes(i) && !ep.payload.capability.limited.includes(i));
  assert.equal(M.inspector(ep, 3, normal).capability, null);
});

test("inspector series cover every step, mark NaN as null and flag the steps with events", () => {
  const ep = load("fleet_collision");
  const retired = [...Array(ep.N).keys()].find((i) => ep.satStatus[ep.T * ep.N + i] === 3);
  const s = M.inspectorSeries(ep, retired);
  assert.equal(s.dv.length, ep.T + 1);
  assert.equal(s.dMin.length, ep.T + 1);
  assert.equal(s.dMin[ep.T], null, "no stale distance after impact");
  assert.ok(s.dMin[0] !== null);
  assert.ok(s.dv.every((v) => v === null || Number.isFinite(v)));
});

// ------------------------------------------------------------------ timeline
test("lane counts come from the recorded density; the shield lane does not exist when the shield was off", () => {
  const off = load("fleet_collision");
  assert.equal(M.countsLanes(off).shield, null);
  const on = load("shielded");
  assert.equal(M.countsLanes(on).shield.length, on.T + 1);
  assert.equal(M.countsLanes(on).maneuver.length, on.T + 1);
});

test("marks for the selected satellite merge runs that are close together", () => {
  const ep = load("depletion");
  const sat = [...Array(ep.N).keys()].find((i) => ep.payload.runs["1"].some((r) => r.sats.includes(i)));
  const marks = M.laneMarks(ep, 1, sat);
  assert.ok(marks.length >= 1);
  for (let k = 1; k < marks.length; k += 1) assert.ok(marks[k].start - marks[k - 1].end > M.MERGE_GAP_STEPS);
  assert.match(marks[0].label, new RegExp(`^S${sat}`));
});

test("collision marks are one per collided pair, labelled with both satellites", () => {
  const ep = load("fleet_collision");
  const marks = M.collisionMarks(ep);
  assert.equal(marks.length, 1);
  assert.match(marks[0].label, /^S\d×S\d$/);
  assert.equal(marks[0].start, ep.payload.runs["3"][0].start);
});

test("running counts read 'so far / episode total', with an em dash for the shield when it was off", () => {
  const off = load("fleet_collision");
  assert.equal(M.runningCounts(off, 5).shield, "— · shield off");
  const on = load("shielded");
  const c = M.runningCounts(on, on.T);
  assert.equal(c.maneuver, `${on.payload.summary.maneuvers} / ${on.payload.summary.maneuvers}`);
  assert.match(c.shield, new RegExp(`^${on.payload.summary.shield} / ${on.payload.summary.shield}`));
});

// ------------------------------------------------------------------ summary strip running values
test("running values end exactly on the episode totals and print in the same format as the totals", () => {
  const ep = load("shielded");
  const last = M.stripRunning(ep, ep.T);
  const s = ep.payload.summary;
  assert.equal(last.delta_v_mps.endsWith(s.delta_v_mps.toFixed(2)), true);
  assert.equal(last.maneuvers.endsWith(String(s.maneuvers)), true);
  assert.equal(last.collisions.endsWith(String(s.collisions)), true);
  assert.equal(last.mean_deviation.endsWith(s.mean_deviation.toFixed(3)), true);
  assert.match(last.delta_v_mps, /^T\+\d+: /);
});

test("the running mean deviation has no value before the first step, and the shield has none when it was off", () => {
  const off = load("fleet_collision");
  assert.equal(M.stripRunning(off, 0).mean_deviation, "T+0: —");
  assert.equal(M.stripRunning(off, 3).shield, "—");
});
