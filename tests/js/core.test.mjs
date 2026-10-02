import test from "node:test";
import assert from "node:assert/strict";
import { loadOG, fixture } from "./helpers.mjs";

const OG = await loadOG();
const shielded = fixture("shielded");
const fleet = fixture("fleet_collision");

test("decodeArray returns typed arrays with the dtype and shape the payload declares", () => {
  const spec = shielded.payload.arrays.positions_km;
  const array = OG.decodeArray(spec);
  assert.ok(array instanceof Float32Array);
  assert.equal(array.length, spec.shape.reduce((a, b) => a * b, 1));
  assert.ok(OG.decodeArray(shielded.payload.arrays.sat_status) instanceof Uint8Array);
  assert.ok(OG.decodeArray(shielded.payload.arrays.approved) instanceof Int8Array);
});

test("decodeEpisode reproduces the recorded positions, approved intents and statuses exactly", () => {
  const ep = OG.decodeEpisode(shielded.payload);
  assert.deepEqual(Array.from(ep.pos), shielded.expected.positions_km);
  assert.deepEqual(Array.from(ep.approved), shielded.expected.approved);
  assert.deepEqual(Array.from(ep.satStatus), shielded.expected.sat_status);
});

test("decodeEpisode exposes the dimensions, catalog, vocabulary and constants shipped by Python", () => {
  const ep = OG.decodeEpisode(fleet.payload);
  assert.equal(ep.N, fleet.payload.dims.agents);
  assert.equal(ep.T, fleet.payload.dims.steps);
  assert.equal(ep.P, fleet.payload.dims.pairs);
  assert.equal(ep.pairs.length, 2 * ep.P);
  assert.equal(ep.catalog.length, 6);
  assert.equal(ep.vocab.sat_status[3].glyph, "✕");
  assert.equal(ep.earthRadiusKm, fleet.payload.constants.earth_radius_km);
  assert.equal(ep.shield, false);
});

test("posAt reads one satellite position at one step", () => {
  const ep = OG.decodeEpisode(shielded.payload);
  const p = OG.posAt(ep, 3, 1);
  const base = (3 * ep.N + 1) * 3;
  assert.deepEqual(p, [ep.pos[base], ep.pos[base + 1], ep.pos[base + 2]]);
});

test("axisRange covers Earth and the farthest recorded satellite", () => {
  const ep = OG.decodeEpisode(fleet.payload);
  let farthest = ep.earthRadiusKm;
  for (let k = 0; k < ep.pos.length; k += 3) farthest = Math.max(farthest, Math.hypot(ep.pos[k], ep.pos[k + 1], ep.pos[k + 2]));
  const range = OG.axisRange(ep);
  assert.ok(range > farthest && range < 1.5 * farthest);
});

test("easeOut runs from 0 to 1 and is monotone", () => {
  assert.equal(OG.easeOut(0), 0);
  assert.equal(OG.easeOut(1), 1);
  assert.ok(OG.easeOut(0.3) < OG.easeOut(0.6));
});

test("decodeEpisode rejects a payload from another schema", () => {
  assert.throws(() => OG.decodeEpisode({ ...shielded.payload, schema: "other/1" }), /schema/);
});

test("decodeEpisode refuses a payload that disagrees with itself instead of drawing it", () => {
  const clone = () => JSON.parse(JSON.stringify(shielded.payload));
  const wrongSchema = clone(); wrongSchema.schema = "something/else";
  assert.throws(() => OG.decodeEpisode(wrongSchema), /schema/);
  const wrongDtype = clone(); wrongDtype.arrays.positions_km.dtype = "float64";
  assert.throws(() => OG.decodeEpisode(wrongDtype), /dtype/);
  const wrongShape = clone(); wrongShape.arrays.sat_status.shape = [3, 3];
  assert.throws(() => OG.decodeEpisode(wrongShape), /shape says/);
  const wrongSteps = clone(); wrongSteps.dims.steps += 1;
  assert.throws(() => OG.decodeEpisode(wrongSteps), /dimensions/);
  const noSteps = clone(); noSteps.dims.steps = 0;
  assert.throws(() => OG.decodeEpisode(noSteps), /no steps|dimensions/);
  const badCatalog = clone(); badCatalog.catalog.reverse();
  assert.throws(() => OG.decodeEpisode(badCatalog), /catalog/);
  const badBase64 = clone(); badBase64.arrays.positions_km.b64 = "!!!";
  assert.throws(() => OG.decodeEpisode(badBase64));
});
