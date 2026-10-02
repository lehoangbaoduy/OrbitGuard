/* OrbitGuard replay stage - renders the panel view-models (20_, 21_) into the DOM. Colours arrive as CSS custom
   properties (--c) that come from the payload's vocabulary; all text goes through textContent (see 22_og_dom.js). */
(function (root) {
  "use strict";
  const OG = root.OG;
  const { h, mount } = OG.dom;
  const NEUTRAL = "var(--og-text-3)";
  const RISK_ROWS = 8;
  const DASH = "—";
const fmtM = (m) => (Number.isFinite(m) ? Math.round(m).toLocaleString("en-US") : DASH);
  const fixed = (v, digits) => (Number.isFinite(v) ? v.toFixed(digits) : DASH);

  const glyph = (text, color) => h("span", { class: "og-row-glyph", style: { "--c": color || NEUTRAL } }, text);

  function satRow(row, selected, onSelect) {
    return h("button", { type: "button", class: `og-row${row.sat === selected ? " is-selected" : ""}`, "data-sat": row.sat, "data-focus-key": `sat-${row.sat}`, "aria-pressed": row.sat === selected, on: { click: () => onSelect(row.sat) } },
      glyph(row.statusGlyph, row.color),
      h("span", { class: "og-row-main" },
        h("span", { class: "og-row-id" }, `S${row.sat}`),
        h("span", { class: "og-row-status", style: { "--c": row.color } }, row.statusLabel),
        row.eventName ? h("span", { class: "og-chip", style: { "--c": row.eventColor } }, `${row.eventGlyph} ${row.eventLabel}`) : null),
      h("span", { class: "og-row-sub" }, `${row.sub}${row.detail}`));
  }

  function groupRow(row, onExpand) {
    return h("button", { type: "button", class: "og-row og-row-group", "data-focus-key": `group-${row.label}`, on: { click: onExpand } },
      glyph(row.glyph, row.color),
      h("span", { class: "og-row-main" }, h("span", { class: "og-row-id" }, `×${row.count}`), h("span", { class: "og-row-status", style: { "--c": row.color } }, row.label)),
      h("span", { class: "og-row-sub" }, `${row.sub}${row.detail}`));
  }

  function emptyCard(ep) {
    const shieldText = ep.shield ? "No shield interventions this step" : "No shield events — shield off, nothing overrides the policy";
    return h("div", { class: "og-empty-card" },
      h("span", { class: "og-empty-line" }, h("span", { class: "og-row-glyph", style: { "--c": NEUTRAL } }, "●"), `All ${ep.N} satellites nominal`),
      h("span", { class: "og-empty-line is-dim" }, h("span", { class: "og-empty-swatch" }), "No maneuvers this step"),
      h("span", { class: "og-empty-line is-dim" }, h("span", { class: "og-row-glyph", style: { "--c": NEUTRAL } }, "◇"), shieldText));
  }

  /* "Tracked this step": a cell per satellite, a tally, then the flagged rows (grouped when many share an event). */
  function tracked(container, ep, step, ui) {
    const model = OG.panels.trackedRows(ep, step, ui.expanded);
    const cells = OG.panels.stepCells(ep, step).map((c) => h("button", { type: "button", class: `og-cell-btn${c.flagged ? " is-flagged" : ""}${c.sat === ui.selected ? " is-selected" : ""}`,
      title: c.title, "aria-label": c.title, "data-focus-key": `cell-${c.sat}`, "aria-pressed": c.sat === ui.selected, style: { "--c": c.color || NEUTRAL }, on: { click: () => ui.select(c.sat) } }, c.glyph));
    const shieldNote = !ep.shield && model.flagged > 0 ? h("div", { class: "og-note" }, h("span", { class: "og-row-glyph" }, "◇"), "No shield events — shield off, nothing overrides the policy") : null;
    mount(container,
      h("div", { class: "og-count" }, model.flagged ? `${model.flagged} of ${ep.N} flagged` : ""),
      h("div", { class: "og-cells", role: "group", "aria-label": "This step, per satellite", style: { "--n": ep.N } }, cells),
      h("div", { class: "og-cells-axis" }, h("span", {}, "S0"), h("span", {}, OG.panels.stepTally(ep, step)), h("span", {}, `S${ep.N - 1}`)),
      model.empty ? emptyCard(ep) : h("div", { class: "og-rows" }, model.rows.map((r) => (r.kind === "group" ? groupRow(r, ui.toggleExpand) : satRow(r, ui.selected, ui.select)))),
      shieldNote,
      model.moreLabel ? h("button", { type: "button", class: "og-more", "data-focus-key": "more", on: { click: ui.toggleExpand } }, model.moreLabel) : null);
    return model;
  }

  /* Conjunction & risk: the top pairs by status, risk and closest approach; a collided pair has no distance. */
  function risk(container, ep, step, ui) {
    const rows = OG.panels.riskRows(ep, step, RISK_ROWS);
    mount(container, h("div", { class: "og-risk" },
      h("div", { class: "og-risk-head" }, h("span", {}, "pair"), h("span", {}, "status"), h("span", {}, "d_min (m)"), h("span", {}, "risk")),
      rows.map((r) => h("button", { type: "button", class: "og-risk-row", "data-focus-key": `risk-${r.pair}`, on: { click: () => ui.select(r.i) }, title: `${r.pair} · ${r.label}` },
        h("span", { class: "og-risk-pair" }, r.pair),
        h("span", { class: "og-chip", style: { "--c": r.color } }, `${r.glyph} ${r.label}`),
        h("span", { class: "og-risk-dist" }, r.dMin === null ? h("span", {}, "—") : h("span", {}, fmtM(r.dMin)), h("span", { class: "og-risk-sub" }, r.note || `d_safe ${fmtM(r.dSafe)}`)),
        h("span", { class: "og-risk-val" }, h("span", {}, fixed(r.risk, 3)), h("span", { class: "og-risk-bar" }, h("span", { style: { "--r": String(Number.isFinite(r.risk) ? Math.min(1, r.risk) : 0) } })))))));
  }

  /* Fleet fuel: one bar per satellite on its own budget; offline is dashed with OFF, empty reads EMPTY. */
  function fuel(container, ep, step, ui) {
    const model = OG.models.fuelBars(ep, step);
    mount(container,
      h("div", { class: "og-fuel-head" },
        h("span", { class: "og-fuel-total" }, fixed(model.total, 2)), h("span", { class: "og-fuel-unit" }, "m/s left"),
        h("span", { class: "og-fuel-text" }, model.text),
        model.stranded > 0 ? h("span", { class: "og-fuel-stranded", title: "fuel held by offline thrusters, which cannot spend it" }, `${fixed(model.stranded, 2)} m/s stranded`) : null),
      h("canvas", { id: "og-fuel-canvas", class: "og-fuel-canvas" }),
      h("div", { class: "og-fuel-bars" }, model.bars.map((b) => h("button", { type: "button", class: `og-fuel-row is-${b.kind}${b.retired ? " is-retired" : ""}${b.sat === ui.selected ? " is-selected" : ""}`, "data-focus-key": `fuel-${b.sat}`, "aria-pressed": b.sat === ui.selected, on: { click: () => ui.select(b.sat) },
        title: `S${b.sat}: ${b.kind === "offline" ? "thruster offline" : `${fixed(b.remaining, 2)} of ${fixed(b.budget, 2)} m/s`}${b.kind === "limited" ? " (limited budget)" : ""}` },
        h("span", { class: "og-fuel-id" }, `S${b.sat}`, b.kind === "offline" ? h("span", { class: "og-cap" }, "⊘") : b.kind === "limited" ? h("span", { class: "og-cap" }, "◐") : null),
        h("span", { class: "og-fuel-track", style: { "--track": String(b.trackFrac) } }, h("span", { class: "og-fuel-fill", style: { "--fill": String(b.fillFrac) } })),
        h("span", { class: "og-fuel-label" }, b.label || fixed(b.remaining, 2))))));
    return model;
  }

  /* Inspector readouts and controls (the chart is drawn separately, 24_og_charts.js). */
  function inspector(container, ep, step, ui) {
    const m = OG.models.inspector(ep, step, ui.selected);
    const step1 = (delta) => () => ui.select((ui.selected + delta + ep.N) % ep.N);
    const chips = [
      h("span", { class: "og-chip", style: { "--c": m.status.color } }, `${m.status.glyph} ${m.status.label}`),
      m.event.label ? h("span", { class: "og-chip", style: { "--c": m.event.color } }, `${m.event.glyph} ${m.event.label}`) : null,
      m.capability ? h("span", { class: "og-chip is-neutral" }, m.capability === "offline" ? "⊘ THRUSTER OFFLINE" : "◐ LIMITED Δv") : null,
    ];
    const picker = Array.from({ length: ep.N }, (_, i) => h("button", { type: "button", class: `og-pick${i === ui.selected ? " is-selected" : ""}`, on: { click: () => ui.select(i) }, "data-focus-key": `pick-${i}`, "aria-label": `Select satellite ${i}`, "aria-pressed": i === ui.selected }, `${i}`));
    const decision = m.noActions
      ? h("div", { class: "og-decision is-dim" }, `Step 0: no decision yet — actions exist for steps 1–${ep.T}`)
      : h("div", { class: "og-decision" }, h("span", { class: "og-kicker" }, "decision"), h("span", { class: "og-decision-line" }, `${m.decision.proposed} → ${m.decision.approved}`),
        m.earlier.map((e) => h("button", { type: "button", class: `og-chip is-${e.kind}`, title: "jump to that step", "data-focus-key": `earlier-${e.step}`, on: { click: () => ui.goStep(e.step) } }, e.text)));
    mount(container,
      h("div", { class: "og-insp-head" },
        h("button", { type: "button", class: "og-step-btn", "aria-label": "Previous satellite", "data-focus-key": "insp-prev", on: { click: step1(-1) } }, "‹"),
        h("span", { class: "og-insp-id" }, `S${m.sat}`),
        h("button", { type: "button", class: "og-step-btn", "aria-label": "Next satellite", "data-focus-key": "insp-next", on: { click: step1(1) } }, "›"),
        h("span", { class: "og-chips" }, chips)),
      h("div", { class: "og-readouts" },
        h("div", {}, h("span", { class: "og-kicker" }, "Δv left"), h("span", { class: "og-readout" }, m.dvText), h("span", { class: "og-unit" }, "m/s")),
        h("div", {}, h("span", { class: "og-kicker" }, "dev"), h("span", { class: "og-readout" }, m.devText)),
        h("div", { title: m.closestNote }, h("span", { class: "og-kicker" }, "closest"), h("span", { class: "og-readout" }, m.closestText), h("span", { class: "og-unit" }, m.closest ? `S${m.closest.partner}` : m.closestNote))),
      h("div", { class: "og-picker", role: "group", "aria-label": "Satellites" }, picker),
      h("div", { id: "og-inspector-chart", class: "og-insp-chart" }),
      decision);
    return m;
  }

  OG.render = { tracked, risk, fuel, inspector };
})(globalThis);
