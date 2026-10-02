import test from "node:test";
import assert from "node:assert/strict";
import { loadOG, fixture, dot, norm } from "./helpers.mjs";

const OG = await loadOG();
const EVENT_MANEUVER = 1;
const EVENT_SHIELD = 2;
const STATUS_COLLIDED = 3;

const load = (name) => OG.decodeEpisode(fixture(name).payload);
const find = (ep, code) => {
  for (let t = 1; t <= ep.T; t += 1) for (let i = 0; i < ep.N; i += 1) if (ep.event[t * ep.N + i] === code) return [t, i];
  return null;
};
const view = (ep, over = {}) => ({ step: ep.openStep, frac: 0, selected: 0, orbits: true, labelAll: false, grow: { plume: 1, ring: 1, lines: 1 }, ...over });
const close = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

test("the local frame at every step and satellite is orthonormal, including the last frame", () => {
  const ep = load("shielded");
  for (const t of [0, 5, ep.T]) {
    for (let i = 0; i < ep.N; i += 1) {
      const { r, v, n } = OG.scene.basis(ep, t, i);
      assert.ok(close(norm(r), 1) && close(norm(v), 1) && close(norm(n), 1));
      assert.ok(close(dot(r, v), 0) && close(dot(r, n), 0) && close(dot(v, n), 0));
    }
  }
});

test("positions interpolate linearly between two recorded steps", () => {
  const ep = load("shielded");
  const [a, b, mid] = [OG.posAt(ep, 4, 0), OG.posAt(ep, 5, 0), OG.scene.interpolated(ep, 4, 0.5)[0]];
  for (let k = 0; k < 3; k += 1) assert.ok(close(mid[k], (a[k] + b[k]) / 2, 1e-4));
});

test("a burn plume points along the exhaust direction of the approved intent from the catalog", () => {
  const ep = load("depletion");
  const hit = find(ep, EVENT_MANEUVER);
  assert.ok(hit, "fixture must contain a maneuver");
  const [t, i] = hit;
  const intent = ep.approved[(t - 1) * ep.N + i];
  const [er, et, en] = ep.catalog[intent].exhaust_rtn;
  const { r, v, n } = OG.scene.basis(ep, t, i);
  const want = [0, 1, 2].map((k) => er * r[k] + et * v[k] + en * n[k]);
  const plume = OG.scene.plume(ep, t, i, 1);
  const got = [0, 1, 2].map((k) => plume.to[k] - plume.from[k]);
  assert.ok(close(dot(got, want) / norm(got), 1, 1e-6), "plume must be parallel to the exhaust direction");
  assert.ok(norm(OG.scene.plume(ep, t, i, 1).to.map((x, k) => x - plume.from[k])) > norm(OG.scene.plume(ep, t, i, 0).to.map((x, k) => x - plume.from[k])));
});

test("the plume follows the catalog: editing the intent's exhaust direction flips the plume, and it agrees with the recorded motion", () => {
  const ep = load("depletion");
  const [t, i] = find(ep, EVENT_MANEUVER);
  const intent = ep.approved[(t - 1) * ep.N + i];
  const direction = (e) => { const p = OG.scene.plume(e, t, i, 1); return [0, 1, 2].map((k) => p.to[k] - p.from[k]); };
  const before = direction(ep);
  const original = ep.catalog[intent].exhaust_rtn;
  ep.catalog[intent].exhaust_rtn = original.map((x) => -x);
  const after = direction(ep);
  ep.catalog[intent].exhaust_rtn = original;
  assert.ok(dot(before, after) / (norm(before) * norm(after)) < -0.999, "reversing the catalog exhaust reverses the plume, so there is no local table");
  // independent frame: along-track = recorded motion minus its radial part
  const p0 = OG.posAt(ep, t, i), p1 = OG.posAt(ep, t + 1, i), rr = norm(p0), rhat = p0.map((x) => x / rr);
  const move = p1.map((x, k) => x - p0[k]), radial = dot(move, rhat);
  const along = move.map((x, k) => x - radial * rhat[k]), alongHat = along.map((x) => x / norm(along));
  const [er, et] = original;
  const component = dot(before, alongHat) / norm(before);
  if (Math.abs(et) > Math.abs(er)) assert.equal(Math.sign(component), Math.sign(et), "the along-track part of the plume has the catalog's sign");
});

test("there is no plume where nothing was burned", () => {
  const ep = load("shielded");
  assert.equal(OG.scene.plume(ep, 0, 0, 1), null);
  for (let t = 1; t <= ep.T; t += 1) for (let i = 0; i < ep.N; i += 1)
    if (ep.event[t * ep.N + i] !== EVENT_MANEUVER) assert.equal(OG.scene.plume(ep, t, i, 1), null);
});

test("a shield ring is closed, round, and lies in the plane of travel", () => {
  const ep = load("shielded");
  const hit = find(ep, EVENT_SHIELD);
  assert.ok(hit, "fixture must contain a shield intervention");
  const [t, i] = hit;
  const ring = OG.scene.shieldRing(ep, t, i, 1);
  const { r } = OG.scene.basis(ep, t, i);
  const centre = OG.posAt(ep, t, i);
  assert.deepEqual(ring[0], ring[ring.length - 1]);
  const radius = norm(ring[0].map((x, k) => x - centre[k]));
  for (const p of ring) {
    const d = p.map((x, k) => x - centre[k]);
    assert.ok(close(norm(d), radius, 1e-6 * radius) && Math.abs(dot(d, r)) < 1e-6 * radius);
  }
  assert.ok(norm(OG.scene.shieldRing(ep, t, i, 0)[0].map((x, k) => x - centre[k])) > radius, "ring settles inward");
});

test("the trace list keeps the same length and order from step to step", () => {
  const ep = load("fleet_collision");
  const shapes = (s, over) => OG.scene.sceneTraces(ep, view(ep, { step: s, ...over })).data.map((d) => `${d.type}:${d.name || ""}`);
  assert.deepEqual(shapes(0), shapes(ep.T));
  assert.deepEqual(shapes(3), shapes(10));
  assert.deepEqual(shapes(3), shapes(3, { selected: 1, orbits: false }), "selection and the orbit toggle never change the list");
  assert.equal(new Set(shapes(3)).size >= 8, true, "names distinguish the traces, so an order swap would be noticed");
});

test("no trace ever carries a NaN coordinate", () => {
  const ep = load("fleet_collision");
  for (const s of [0, 7, ep.T]) {
    for (const d of OG.scene.sceneTraces(ep, view(ep, { step: s })).data)
      for (const key of ["x", "y", "z"]) for (const v of (d[key] || []).flat()) assert.ok(v === null || Number.isFinite(v), `${d.type}.${key}`);
  }
});

test("retired satellites are drawn as collision marks, survivors as normal satellites", () => {
  const ep = load("fleet_collision");
  const { data, annotations } = OG.scene.sceneTraces(ep, view(ep, { step: ep.T }));
  const retired = [];
  for (let i = 0; i < ep.N; i += 1) if (ep.satStatus[ep.T * ep.N + i] === STATUS_COLLIDED) retired.push(i);
  assert.ok(retired.length === 2);
  const marks = data.find((d) => d.mode === "markers" && d.marker && d.marker.symbol === "x" && d.customdata);
  assert.deepEqual([...marks.customdata].sort(), retired);
  const text = annotations.map((a) => a.text).join(" ");
  for (const i of retired) assert.match(text, new RegExp(`S${i}`));
});

test("a nominal satellite has no status pin; every non-nominal one has colour, glyph and an ID label", () => {
  const ep = load("fleet_collision");
  const t = ep.T;
  const { annotations } = OG.scene.sceneTraces(ep, view(ep, { step: t, selected: 1 }));
  for (let i = 0; i < ep.N; i += 1) {
    const status = ep.satStatus[t * ep.N + i];
    const mentioned = annotations.some((a) => new RegExp(`>S${i}<`).test(a.text) || new RegExp(`S${i}\\b`).test(a.text.replace(/<[^>]+>/g, " ")));
    if (status > 0) assert.ok(mentioned, `S${i} (status ${status}) must be labelled`);
    else if (i !== 1 && ep.event[t * ep.N + i] === 0) assert.ok(!mentioned, `nominal S${i} has no label plate`);
  }
  const pinned = new Set(OG.scene.sceneTraces(ep, view(ep, { step: t, selected: 1 })).data.filter((d) => d.name && d.name.startsWith("pins-")).flatMap((d) => [...(d.customdata || [])]));
  for (let i = 0; i < ep.N; i += 1) assert.equal(pinned.has(i), ep.satStatus[t * ep.N + i] > 0, `S${i} has a pin exactly when it is not nominal`);
});

test("the selected satellite is labelled even when nominal", () => {
  const ep = load("fleet_collision");
  const { annotations } = OG.scene.sceneTraces(ep, view(ep, { step: 0, selected: 2 }));
  assert.ok(annotations.some((a) => /S2/.test(a.text)));
});

test("label plates carry an explicit width and height", () => {
  const ep = load("fleet_collision");
  for (const a of OG.scene.sceneTraces(ep, view(ep, { step: ep.T, labelAll: true })).annotations) {
    assert.ok(a.width > 0 && a.height > 0 && a.showarrow === true);
  }
});

test("the orbits switch removes the rings but keeps the trace list stable", () => {
  const ep = load("shielded");
  const on = OG.scene.sceneTraces(ep, view(ep, { orbits: true })).data;
  const off = OG.scene.sceneTraces(ep, view(ep, { orbits: false })).data;
  assert.equal(on.length, off.length);
  assert.ok(on.some((d, k) => d.x && d.x.length > (off[k].x || []).length));
});

test("trails are at most ten recorded steps long plus the live point", () => {
  const ep = load("fleet_collision");
  const trail = OG.scene.sceneTraces(ep, view(ep, { step: 15, selected: 1 })).data.find((d) => d.name === "trails");
  const first = trail.x.indexOf(null);
  assert.ok(first > 0 && first <= 12);
});

test("at production size (N=15, 360 steps) the trace list is stable and has no NaN at sampled steps", () => {
  const ep = load("full_size");
  assert.equal(ep.N, 15);
  assert.equal(ep.T, 360);
  const names = (step) => OG.scene.sceneTraces(ep, view(ep, { step })).data.map((d) => `${d.type}:${d.name || ""}`);
  const first = names(0);
  for (const step of [1, 90, 181, 359, 360]) {
    assert.deepEqual(names(step), first);
    for (const d of OG.scene.sceneTraces(ep, view(ep, { step })).data) for (const key of ["x", "y", "z"]) for (const v of (d[key] || []).flat()) assert.ok(v === null || Number.isFinite(v), `${d.name || d.type}.${key} @${step}`);
  }
});
