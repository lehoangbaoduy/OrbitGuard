/* OrbitGuard replay stage - tiny DOM builder. Elements are created with createElement/textContent only, never from
   HTML strings, so nothing that came from a payload can ever be interpreted as markup. */
(function (root) {
  "use strict";
  const OG = root.OG;

  /* h("button", {class: "x", "data-sat": 3, on: {click: fn}}, child, "text", ...) -> HTMLElement.
     Attributes with null/undefined/false are skipped; `style` takes custom properties only (colours come from CSS vars). */
  function h(tag, attrs, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "on") for (const [type, handler] of Object.entries(value)) el.addEventListener(type, handler);
      else if (key === "style") for (const [name, v] of Object.entries(value)) el.style.setProperty(name, v);
      else if (key === "text") el.textContent = String(value);
      else el.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat()) {
      if (child === null || child === undefined || child === false) continue;
      el.append(typeof child === "string" || typeof child === "number" ? document.createTextNode(String(child)) : child);
    }
    return el;
  }

  /* Replace everything inside `container` with `children`. */
  function mount(container, ...children) {
    container.replaceChildren(...children.flat().filter((c) => c !== null && c !== undefined && c !== false));
  }

  OG.dom = { h, mount };
})(globalThis);
