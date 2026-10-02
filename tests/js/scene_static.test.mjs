import test from "node:test";
import assert from "node:assert/strict";
import { loadOG, fixture, norm } from "./helpers.mjs";

const OG = await loadOG();
const ep = OG.decodeEpisode(fixture("fleet_collision").payload);
const R = ep.earthRadiusKm;
const SUN = [0.83, 0.45, 0.33];
const radii = (trace) => trace.x.flatMap((row, a) => row.map((_, b) => norm([trace.x[a][b], trace.y[a][b], trace.z[a][b]])));

test("the fallback Earth surface is a sphere of the Earth radius", () => {
  const earth = OG.scene.earthSurface(R, SUN);
  assert.equal(earth.type, "surface");
  for (const r of radii(earth)) assert.ok(Math.abs(r - R) < 1e-6 * R);
});

test("the atmosphere shell sits just above the surface", () => {
  const shell = OG.scene.atmosphere(R);
  for (const r of radii(shell)) assert.ok(r > R && r < 1.05 * R);
});

test("the graticule hugs the surface and separates lines with nulls", () => {
  const g = OG.scene.graticule(R);
  assert.ok(g.x.includes(null));
  for (let k = 0; k < g.x.length; k += 1) if (g.x[k] !== null) assert.ok(norm([g.x[k], g.y[k], g.z[k]]) > R && norm([g.x[k], g.y[k], g.z[k]]) < R + 20);
});

test("the textured Earth lights the day side from the day map and the night side from the night map", () => {
  const white = new Uint8ClampedArray([255, 255, 255, 255, 255, 255, 255, 255]); // 2x1 white day
  const black = new Uint8ClampedArray([0, 0, 0, 255, 0, 0, 0, 255]);
  const mesh = OG.scene.earthMesh(R, SUN, white, black, 2, 1, 8, 16);
  assert.equal(mesh.type, "mesh3d");
  assert.equal(mesh.i.length, 2 * 8 * 16);
  const lum = (c) => c.match(/\d+/g).slice(0, 3).reduce((a, b) => a + Number(b), 0);
  let lit = -1, dark = Infinity;
  mesh.vertexcolor.forEach((c, n) => {
    const d = SUN[0] * mesh.x[n] + SUN[1] * mesh.y[n] + SUN[2] * mesh.z[n];
    if (d > 0.9 * R) lit = Math.max(lit, lum(c));
    if (d < -0.9 * R) dark = Math.min(dark, lum(c));
  });
  assert.ok(lit > dark, `day ${lit} should be brighter than night ${dark}`);
});

test("orbit rings are the satellites' own recorded trajectories", () => {
  const rings = OG.scene.trajectoryRings(ep, 1);
  const pointsPerSat = ep.T + 1;
  assert.equal(rings.x.filter((v) => v !== null).length, ep.N * pointsPerSat);
  assert.deepEqual([rings.x[0], rings.y[0], rings.z[0]], OG.posAt(ep, 0, 0));
});

test("a stride thins the rings but keeps the last recorded point of every satellite", () => {
  const rings = OG.scene.trajectoryRings(ep, 7);
  const finite = rings.x.filter((v) => v !== null).length;
  assert.ok(finite < ep.N * (ep.T + 1));
  const last = OG.posAt(ep, ep.T, 0);
  const idx = rings.x.findIndex((v, k) => v === last[0] && rings.y[k] === last[1]);
  assert.ok(idx >= 0);
});

test("one satellite's trajectory can be drawn on its own", () => {
  const ring = OG.scene.satelliteRing(ep, 1);
  assert.equal(ring.x.length, ep.T + 1);
  assert.deepEqual([ring.x[2], ring.y[2], ring.z[2]], OG.posAt(ep, 2, 1));
});

test("deterministic starfield: same seed same stars, different seed different stars", () => {
  assert.equal(OG.scene.starShadows(40, 2200, 1100, 7), OG.scene.starShadows(40, 2200, 1100, 7));
  assert.notEqual(OG.scene.starShadows(40, 2200, 1100, 7), OG.scene.starShadows(40, 2200, 1100, 8));
  const shadows = OG.scene.starShadows(40, 2200, 1100, 7).split("), ");
  assert.equal(shadows.length, 80, "each star appears twice, one tile apart, so the sky wraps as the camera turns");
  const xs = shadows.map((s) => Number(s.split("px")[0]));
  for (let n = 0; n < xs.length; n += 2) assert.equal(xs[n + 1], xs[n] + 2200);
});
