import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { loadOG } from "./helpers.mjs";

const OG = await loadOG();
const M = OG.motion;
const tokens = fs.readFileSync(path.resolve(import.meta.dirname, "../../orbit_guard/visualization/assets/01_orbitguard-tokens.css"), "utf8");
const token = (name) => Number(tokens.match(new RegExp(`--${name}:\\s*(\\d+)ms`))[1]);

test("durations that the design tokens define are the ones the stage uses", () => {
  assert.equal(M.DURATION.flash, token("og-dur-flash"));
  assert.equal(M.DURATION.line, token("og-dur-count"));
  assert.equal(M.DURATION.pulse, token("og-dur-pulse"));
});

test("growth runs from 0 to 1 over its duration and is 1 for events that never started or with reduced motion", () => {
  assert.equal(M.growth(1000, 1000, 600, false), 0);
  assert.equal(M.growth(1000, 1300, 600, false), 0.5);
  assert.equal(M.growth(1000, 2000, 600, false), 1);
  assert.equal(M.growth(null, 2000, 600, false), 1);
  assert.equal(M.growth(1000, 1100, 600, true), 1);
});

test("tween closes the gap exponentially and snaps when reduced or close enough", () => {
  const one = M.tween(0, 100, 110, 110, false);
  assert.ok(Math.abs(one - 100 * (1 - Math.exp(-1))) < 1e-9);
  assert.ok(M.tween(0, 100, 5000, 110, false) > 99.9);
  assert.equal(M.tween(0, 100, 16, 110, true), 100);
  assert.equal(M.tween(99.9999, 100, 16, 110, false), 100);
  assert.equal(M.tween(50, 50, 16, 110, false), 50);
});

test("the collision flash is gone after its duration and strongest early", () => {
  assert.equal(M.flashAlpha(null, 5000, false), 0);
  assert.equal(M.flashAlpha(0, M.DURATION.flash + 1, false), 0);
  assert.ok(M.flashAlpha(0, 50, false) > M.flashAlpha(0, 1000, false));
  assert.ok(M.flashAlpha(0, 50, false) <= 1);
});

test("with reduced motion a collision step shows a static red frame instead of an animation", () => {
  assert.ok(M.flashAlpha(0, 10, true) > 0);
  assert.equal(M.flashAlpha(0, 10, true), M.flashAlpha(0, 5000, true));
});
