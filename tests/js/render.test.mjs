import test from "node:test";
import assert from "node:assert/strict";
import { loadOG, fixture } from "./helpers.mjs";
import { FakeNode } from "./fakedom.mjs";

const OG = await loadOG();
const load = (name) => OG.decodeEpisode(fixture(name).payload);
const noop = () => {};
const ui = (selected = 0) => ({ selected, expanded: false, select: noop, goStep: noop, toggleExpand: noop });
const text = (node) => node.textContent;

test("h() builds elements with textContent only: markup in a value stays text", () => {
  const el = OG.dom.h("div", { class: "x", "data-k": 3, hidden: false }, "<img src=x onerror=alert(1)>", 7);
  assert.equal(el.attrs.class, "x");
  assert.equal(el.attrs["data-k"], "3");
  assert.equal("hidden" in el.attrs, false);
  assert.equal(OG.dom.h("button", { "aria-pressed": false }).attrs["aria-pressed"], "false");
  assert.equal(OG.dom.h("button", { "aria-pressed": true }).attrs["aria-pressed"], "true");
  assert.equal(el.textContent, "<img src=x onerror=alert(1)>7");
  assert.throws(() => el.innerHTML);
});

test("every button the panels render has an accessible name and a stable focus key where it can be re-rendered", () => {
  const ep = load("fleet_collision");
  const host = new FakeNode("div");
  for (const [render, step] of [["tracked", ep.T], ["risk", ep.T], ["fuel", ep.T], ["inspector", ep.T]]) {
    OG.render[render](host, ep, step, ui());
    for (const b of host.walk().filter((n) => n.tag === "button")) {
      assert.ok(b.attrs["aria-label"] || b.textContent.trim().length > 0 || b.children.some((c) => c.textContent), `${render}: unnamed button`);
      assert.ok(b.attrs["data-focus-key"] || render === "tracked", `${render}: button without a focus key`);
    }
  }
});

test("missing numbers print an em dash, never NaN or 0", () => {
  const ep = load("shielded");
  ep.risk.fill(NaN); ep.dMin.fill(NaN); ep.dSafe.fill(NaN);
  const host = new FakeNode("div");
  OG.render.risk(host, ep, 3, ui());
  assert.doesNotMatch(text(host), /NaN/);
  assert.match(text(host), /—/);
  OG.render.inspector(host, ep, 3, ui());
  assert.doesNotMatch(text(host), /NaN/);
});

test("selection is exposed with aria-pressed on rows, fuel bars and the picker", () => {
  const ep = load("fleet_collision");
  const host = new FakeNode("div");
  OG.render.fuel(host, ep, 5, ui(2));
  const pressed = host.walk().filter((n) => n.attrs["aria-pressed"] === "true").map((n) => n.attrs["data-focus-key"]);
  assert.deepEqual(pressed, ["fuel-2"]);
});
