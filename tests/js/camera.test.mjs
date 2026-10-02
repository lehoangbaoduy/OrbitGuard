import test from "node:test";
import assert from "node:assert/strict";
import { loadOG } from "./helpers.mjs";

const OG = await loadOG();
const camera = { eye: { x: 1.3, y: 1.05, z: 0.55 }, center: { x: 0, y: 0, z: -0.02 }, up: { x: 0, y: 0, z: 1 } };
const plotDiv = (getCamera) => ({ _fullLayout: { scene: { _scene: { getCamera } } } });

test("live() returns an independent copy of the WebGL camera", () => {
  const source = structuredClone(camera);
  const got = OG.camera.live(plotDiv(() => source));
  assert.deepEqual(got, camera);
  got.eye.x = 99;
  assert.equal(source.eye.x, 1.3);
});

test("live() is null when Plotly internals are missing or throw", () => {
  assert.equal(OG.camera.live({}), null);
  assert.equal(OG.camera.live(plotDiv(() => { throw new Error("gl lost"); })), null);
  assert.equal(OG.camera.live(plotDiv(() => ({ nothing: true }))), null);
});

test("withLive() carries the live camera into a layout so a redraw cannot reset the view", () => {
  const layout = { scene: { camera: { eye: { x: 9, y: 9, z: 9 } } } };
  const live = { ...camera, eye: { x: 2, y: 0, z: 0.1 } };
  const out = OG.camera.withLive(layout, plotDiv(() => live));
  assert.deepEqual(out.scene.camera.eye, live.eye);
  assert.equal(layout.scene.camera.eye.x, 9, "input layout is not mutated");
});

test("withLive() leaves the layout alone when there is no live camera", () => {
  const layout = { scene: { camera } };
  assert.deepEqual(OG.camera.withLive(layout, {}), layout);
});

test("starfield parallax is stable for a camera, wraps over a full turn and stays inside its vertical band", () => {
  const a = OG.camera.parallax(camera);
  assert.deepEqual(a, OG.camera.parallax(camera));
  const turn = 2 * Math.PI, { x, y, z } = camera.eye;
  const turned = { ...camera, eye: { x: x * Math.cos(turn) - y * Math.sin(turn), y: x * Math.sin(turn) + y * Math.cos(turn), z } };
  const b = OG.camera.parallax(turned);
  for (const layer of ["far", "near"]) for (const axis of ["x", "y"]) assert.ok(Math.abs(b[layer][axis] - a[layer][axis]) < 1e-6, `${layer}.${axis} wraps over a full turn`);
  for (const layer of [a.far, a.near]) {
    assert.ok(layer.y <= 0 && layer.y >= -400);
    assert.ok(layer.x <= 0 && layer.x > -2200);
  }
  const up = OG.camera.parallax({ ...camera, eye: { x: 0.01, y: 0.01, z: 5 } });
  assert.ok(up.near.y >= -400 && up.near.y <= 0);
});

test("the two starfield layers move at different rates", () => {
  const p = OG.camera.parallax({ ...camera, eye: { x: 0.2, y: 1.2, z: 0.3 } });
  assert.notEqual(p.far.x, p.near.x);
});
