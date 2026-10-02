import test from "node:test";
import assert from "node:assert/strict";
import { loadOG, fixture } from "./helpers.mjs";

const OG = await loadOG();

const item = (id, x, severity = 0, selected = false) => ({ i: id, q: [x, 0, 0], st: severity, isSel: selected, c: "#ffffff", html: `S${id}` });

test("labels closer than the threshold merge into one plate", () => {
  const groups = OG.mergeLabels([item(0, 0), item(1, 500)], 1100, 4);
  assert.equal(groups.length, 1);
  assert.deepEqual(groups[0].m.map((m) => m.i), [0, 1]);
});

test("labels farther apart than the threshold stay separate", () => {
  assert.equal(OG.mergeLabels([item(0, 0), item(1, 2000)], 1100, 4).length, 2);
});

test("a plate never holds more than the limit; the overflow opens a new plate", () => {
  const items = [0, 1, 2, 3, 4, 5].map((i) => item(i, i * 10));
  const groups = OG.mergeLabels(items, 1100, 4);
  assert.deepEqual(groups.map((g) => g.m.length), [4, 2]);
});

test("the most severe label leads its plate, the selected one next", () => {
  const groups = OG.mergeLabels([item(0, 0, 0, false), item(1, 5, 0, true), item(2, 10, 2, false)], 1100, 4);
  assert.deepEqual(groups[0].m.map((m) => m.i), [2, 1, 0]);
});

test("plates built from real satellites never sit closer than the merge threshold", () => {
  const ep = OG.decodeEpisode(fixture("fleet_collision").payload);
  const t = ep.T;
  const items = [];
  for (let i = 0; i < ep.N; i += 1) items.push({ i, q: OG.posAt(ep, t, i), st: ep.satStatus[t * ep.N + i], isSel: false, c: "#fff", html: `S${i}` });
  const groups = OG.mergeLabels(items, 1100, 4);
  for (let a = 0; a < groups.length; a += 1)
    for (let b = a + 1; b < groups.length; b += 1) {
      const d = Math.hypot(...groups[a].q.map((v, k) => v - groups[b].q[k]));
      assert.ok(d >= 1100, `plates ${a} and ${b} are ${d} km apart`);
    }
});

test("plateWidth grows with the visible text and ignores markup", () => {
  assert.ok(OG.plateWidth('<span style="color:red"><b>S1</b></span>') < OG.plateWidth("S1 S2 S3 S4 S5"));
  assert.equal(OG.plateWidth("<b>ab</b>"), OG.plateWidth("ab"));
});
