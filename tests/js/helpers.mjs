// Shared loader for the stage's classic-script modules: they attach pure helpers to globalThis.OG.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const ASSETS = path.resolve(import.meta.dirname, "../../orbit_guard/visualization/assets");
const MODULES = ["10_og_core.js", "11_og_camera.js", "12_og_scene_static.js", "15_og_labels.js", "13_og_scene_dynamic.js", "14_og_playback.js", "16_og_gesture.js", "17_og_motion.js", "20_og_panels.js", "21_og_models.js", "24_og_charts.js"];

export async function loadOG() {
  for (const name of MODULES) {
    const file = path.join(ASSETS, name);
    await import(pathToFileURL(file).href); // every module must exist: a missing one fails loudly
  }
  return globalThis.OG;
}

export function fixture(name) {
  const dir = process.env.OG_FIXTURES;
  if (!dir) throw new Error("OG_FIXTURES is not set; run through tests/evaluation/test_js_units.py");
  return JSON.parse(fs.readFileSync(path.join(dir, `${name}.json`), "utf8"));
}

export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const norm = (a) => Math.hypot(a[0], a[1], a[2]);
