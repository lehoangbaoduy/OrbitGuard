import test from "node:test";
import assert from "node:assert/strict";
import { loadOG, fixture } from "./helpers.mjs";

const OG = await loadOG();
const PN = OG.panels;
const load = (name) => OG.decodeEpisode(fixture(name).payload);
const firstStep = (ep, pred) => { for (let t = 0; t <= ep.T; t += 1) if (pred(t)) return t; return -1; };
const anyFlag = (ep, t) => Array.from({ length: ep.N }, (_, i) => i).some((i) => ep.satStatus[t * ep.N + i] || ep.event[t * ep.N + i]);

test("a step with nothing flagged is an empty state, not an empty list", () => {
  const ep = load("shielded");
  const quiet = firstStep(ep, (t) => !anyFlag(ep, t));
  assert.ok(quiet >= 0, "fixture must have a quiet step");
  const model = PN.trackedRows(ep, quiet, false);
  assert.deepEqual(model.rows, []);
  assert.equal(model.empty, true);
  assert.equal(model.flagged, 0);
});

test("status rows lead, collided satellites first, each retired satellite says when it was lost", () => {
  const ep = load("fleet_collision");
  const model = PN.trackedRows(ep, ep.T, false);
  const first = model.rows[0];
  assert.equal(first.kind, "sat");
  assert.equal(first.statusName, "collided");
  assert.match(first.sub, /^retired · impact T\+\d+$/);
  const ranks = model.rows.map((r) => r.rank);
  assert.deepEqual(ranks, [...ranks].sort((a, b) => b - a));
});

test("a maneuver or shield row shows proposed → approved with the catalog's short labels", () => {
  const ep = load("depletion");
  let t = -1;
  for (let s = 1; s <= ep.T && t < 0; s += 1) for (let i = 0; i < ep.N; i += 1) if (ep.event[s * ep.N + i] === 1) t = s;
  assert.ok(t > 0);
  const row = PN.trackedRows(ep, t, true).rows.find((r) => r.eventName === "maneuver");
  const sat = row.sat;
  const want = `${ep.catalog[ep.proposed[(t - 1) * ep.N + sat]].short} → ${ep.catalog[ep.approved[(t - 1) * ep.N + sat]].short}`;
  assert.match(row.detail, new RegExp(want.replace(/[+]/g, "\\+")));
});

function withEvent(ep, step, code, sats, shield = ep.shield) {
  const copy = { ...ep, shield, event: Uint8Array.from(ep.event), satStatus: Uint8Array.from(ep.satStatus) };
  for (let i = 0; i < ep.N; i += 1) { copy.event[step * ep.N + i] = sats.includes(i) ? code : 0; copy.satStatus[step * ep.N + i] = 0; }
  return copy;
}

test("more than three satellites sharing an event collapse into one group row; expanding lists them", () => {
  const base = load("fleet_collision");
  const ep = withEvent(base, 5, 2, [0, 1, 2, 3]);
  const grouped = PN.trackedRows(ep, 5, false);
  assert.equal(grouped.rows.length, 1);
  assert.equal(grouped.rows[0].kind, "group");
  assert.equal(grouped.rows[0].count, 4);
  assert.deepEqual(grouped.rows[0].sats, [0, 1, 2, 3]);
  assert.equal(grouped.moreLabel, "Expand groups");
  const expanded = PN.trackedRows(ep, 5, true);
  assert.equal(expanded.rows.length, 4);
  assert.equal(expanded.moreLabel, "Collapse groups");
});

test("three satellites sharing an event stay individual", () => {
  const ep = withEvent(load("fleet_collision"), 5, 2, [0, 1, 2]);
  assert.equal(PN.trackedRows(ep, 5, false).rows.length, 3);
});

test("the per-satellite cells give every satellite a glyph, colour and a text title", () => {
  const ep = load("fleet_collision");
  const cells = PN.stepCells(ep, ep.T);
  assert.equal(cells.length, ep.N);
  for (const cell of cells) assert.ok(cell.glyph && cell.title.startsWith(`S${cell.sat}:`));
  const collided = cells.filter((c) => /collided/i.test(c.title));
  assert.equal(collided.length, 2);
});

test("the tally counts shield and maneuver events and flagged satellites, and says so when there are none", () => {
  const ep = withEvent(load("fleet_collision"), 5, 2, [0, 1], true);
  assert.match(PN.stepTally(ep, 5), /◇ 2/);
  const quiet = load("shielded");
  assert.equal(quiet.shield, true);
  const q = firstStep(quiet, (t) => !anyFlag(quiet, t));
  assert.ok(q >= 0, "the shielded fixture must have a quiet step");
  assert.equal(PN.stepTally(quiet, q), "no events");
});

test("with the shield off the tally says '◇ off' instead of a zero", () => {
  const ep = load("fleet_collision");
  assert.equal(ep.shield, false);
  assert.match(PN.stepTally(ep, 3), /◇ off/);
});

test("the banner names the most serious event, the satellites involved and what else happened", () => {
  const ep = withEvent(load("fleet_collision"), 5, 3, [0, 2]);
  const banner = PN.banner(ep, 5);
  assert.equal(banner.type, "collision");
  assert.equal(banner.sats, "S0 × S2");
  assert.equal(banner.glyph, "✕");
});

test("the banner collapses a long list to a count and is null on a quiet step", () => {
  const ep = withEvent(load("fleet_collision"), 5, 2, [0, 1, 2, 3]);
  assert.equal(PN.banner(ep, 5, { maxNames: 3 }).sats, "4 satellites");
  assert.equal(PN.banner(ep, 5).sats, "S0 · S1 · S2 · S3", "up to four satellites are named");
  const quiet = load("shielded");
  assert.equal(PN.banner(quiet, firstStep(quiet, (t) => !anyFlag(quiet, t))), null);
});

test("hover lines for a scrubber position list events and pair statuses, or say all is nominal", () => {
  const ep = load("fleet_collision");
  const busy = PN.hoverLines(ep, ep.T);
  assert.ok(busy.length >= 1 && busy.every((l) => l.glyph && l.text));
  const quiet = firstStep(ep, (t) => !anyFlag(ep, t) && Array.from({ length: ep.P }, (_, k) => ep.pairStatus[t * ep.P + k]).every((s) => s === 0));
  if (quiet >= 0) assert.deepEqual(PN.hoverLines(ep, quiet), [{ glyph: "·", color: null, text: "no events · all pairs nominal" }]);
});

test("risk rows are ordered by status, then risk, then closest approach; a collided pair shows no distance", () => {
  const ep = load("fleet_collision");
  const rows = PN.riskRows(ep, ep.T, 8);
  assert.ok(rows.length >= 1 && rows.length <= 8 && rows.length < ep.P, "pairs with a retired satellite dropped out");
  const key = (r) => [r.statusCode, r.risk];
  for (let k = 1; k < rows.length; k += 1) {
    const [a, b] = [key(rows[k - 1]), key(rows[k])];
    assert.ok(a[0] > b[0] || (a[0] === b[0] && a[1] >= b[1]));
  }
  const collided = rows.find((r) => r.statusName === "collision");
  assert.equal(collided.dMin, null);
  assert.match(collided.note, /^impact T\+\d+$/);
});

test("the scrubber tooltip names a collided pair instead of calling the fleet nominal", () => {
  const ep = OG.decodeEpisode(fixture("fleet_collision").payload);
  const lines = OG.panels.hoverLines(ep, ep.T).map((l) => l.text).join(" | ");
  assert.match(lines, /COLLISION/i);
  assert.doesNotMatch(lines, /all pairs nominal/);
});

test("risk rows survive NaN risk and distance without breaking the order", () => {
  const ep = OG.decodeEpisode(fixture("shielded").payload);
  ep.risk[3] = NaN; ep.dMin[3] = NaN;
  const rows = OG.panels.riskRows(ep, 3, 8);
  assert.ok(rows.length >= 1);
  assert.ok(rows.every((r) => r.risk === null || Number.isFinite(r.risk) || Number.isNaN(r.risk)));
});

test("pairs that involve a retired satellite leave the ranking, except the collided pair itself", () => {
  const ep = load("fleet_collision");
  const rows = OG.panels.riskRows(ep, ep.T, 50);
  const retired = new Set(Array.from({ length: ep.N }, (_, i) => i).filter((i) => ep.satStatus[ep.T * ep.N + i] === 3));
  assert.equal(retired.size, 2);
  for (const r of rows) assert.ok(r.statusCode === 3 || (!retired.has(r.i) && !retired.has(r.j)), `${r.pair} should have dropped out`);
  assert.ok(rows.some((r) => r.statusCode === 3), "the collided pair stays, first in the table");
  assert.equal(rows[0].statusCode, 3);
});

test("rows involving the selected satellite are flagged and a click targets the other satellite of the pair", () => {
  const ep = load("shielded");
  const rows = OG.panels.riskRows(ep, 5, 8, 1);
  assert.ok(rows.length > 0);
  for (const r of rows) {
    assert.equal(r.involvesSelected, r.i === 1 || r.j === 1);
    if (r.involvesSelected) assert.equal(r.target, r.i === 1 ? r.j : r.i);
  }
});
