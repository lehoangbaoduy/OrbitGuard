// A tiny stand-in for the DOM, just enough for OG.dom / OG.render: elements with attributes, children, text and style.
export class FakeNode {
  constructor(tag) { this.tag = tag; this.attrs = {}; this.children = []; this.listeners = {}; this.styleMap = {}; this.nodeValue = null; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k] ?? null; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  get style() { return { setProperty: (k, v) => { this.styleMap[k] = v; } }; }
  set textContent(v) { this.children = [new FakeText(String(v))]; }
  get textContent() { return this.children.map((c) => c.textContent).join(""); }
  walk() { return [this, ...this.children.flatMap((c) => (c.walk ? c.walk() : []))]; }
  get innerHTML() { throw new Error("innerHTML must never be used"); }
  set innerHTML(v) { throw new Error("innerHTML must never be used"); }
}
export class FakeText { constructor(t) { this.nodeValue = t; } get textContent() { return this.nodeValue; } }

export function installFakeDom() {
  globalThis.document = { createElement: (tag) => new FakeNode(tag), createTextNode: (t) => new FakeText(t) };
}
