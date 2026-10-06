/* OrbitGuard replay - controller: the playback clock, selection, panels, scrubber, timeline and keys.
   All mutable state is private to this closure (nothing application-specific is published on `window`). Dash calls
   `dash_clientside.og.boot` whenever the stored payload changes; everything else is driven by the user and by one
   requestAnimationFrame loop that runs only while something is moving. */
(function (root) {
  "use strict";
  const OG = root.OG;
  const $ = (id) => document.getElementById(id);
  const DRAW_INTERVAL_MS = 33; // the 3D scene is redrawn at most ~30 times a second
  const FALLBACK_STEP_MS = 250; // used only if the token file is missing; the real value is --og-step-ms
  const TIMELINE_HEIGHT = 130, MATRIX_HEIGHT = 150;
  const HOVER_PAD_PX = 8;
  const DOCK_MIN_PX = 780; // the stage is wide enough for the docked column at this width (mirrors 04_og_stage.css)
  const ANNOUNCE_DELAY_MS = 350, ARIA_VALUE_INTERVAL_MS = 500, PICKER_COLUMNS = 5;
  const MAX_FRAME_DT_MS = 250; // a stall (hidden tab, slow draw) slows playback instead of skipping ahead
  const S = { ep: null, p: 0, playing: false, speed: 1, loop: false, selected: 0, orbits: true, expanded: false, timeline: "counts", last: 0, lastDraw: -Infinity, raf: 0,
    anim: { plume: null, ring: null, lines: null, flash: null }, needDraw: true, panelsKey: "", stripVals: {}, stripTween: null, chart: { el: null, key: "" }, bound: false };
  let stage = null;

  const reduced = () => Boolean(root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const stepMs = () => { const v = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--og-step-ms")); return Number.isFinite(v) && v > 0 ? v : FALLBACK_STEP_MS; };
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const curStep = () => (S.ep ? OG.playback.stepOf(S.p, S.ep.T) : 0);

  function setText(id, text) { const el = $(id); if (el && el.textContent !== text) el.textContent = text; }

  /* Polite announcements for what the USER just did (stepping, jumping); playback itself stays silent. */
  let announceTimer = 0, lastAriaValue = -Infinity;
  function announce(text) {
    const live = $("og-live");
    if (!live) return;
    root.clearTimeout(announceTimer);
    announceTimer = root.setTimeout(() => { live.textContent = ""; live.textContent = text; }, ANNOUNCE_DELAY_MS);
  }
  const alertText = (text) => { const el = $("og-alert"); if (el) { el.textContent = ""; el.textContent = text; } };

  function stepSummary(step) {
    const ep = S.ep, flagged = OG.panels.stepCells(ep, step).filter((c) => c.flagged).length, b = OG.panels.banner(ep, step);
    return `Step ${step} of ${ep.T}, T+${OG.playback.clockText(step, ep.controlDtS).minutes} min, ${flagged} of ${ep.N} satellites flagged${b ? `, ${b.label}` : ""}`;
  }

  function setStatus(title, detail) {
    const loading = $("og-loading");
    if (!loading) return;
    loading.hidden = false;
    const heading = loading.querySelector(".og-loading-title");
    if (heading) heading.textContent = title;
    setText("og-loading-detail", detail || "");
  }

  /* ------------------------------------------------------------ panels */
  /* Re-rendering replaces buttons; keep keyboard focus on "the same" button (same container, same position). */
  /* A container is left alone while a pointer is pressed in it, or while it holds keyboard focus during playback: rebuilding
     it would swallow the click or make a screen reader re-read the focused control at playback rate. */
  function keepingFocus(container, render) {
    const active = document.activeElement, inside = container && active && container.contains(active);
    if (container && (S.pressedIn === container || (S.playing && inside))) return null; // refreshed on pause / after the click
    const key = inside ? active.dataset.focusKey || null : null;
    const result = render();
    if (key) {
      const again = [...container.querySelectorAll("[data-focus-key]")].find((el) => el.dataset.focusKey === key);
      if (again) again.focus({ preventScroll: true });
      else { container.tabIndex = -1; container.focus({ preventScroll: true }); } // the focused control is gone: keep focus in its panel, not on <body>
    }
    return result;
  }

  const ui = () => ({ selected: S.selected, expanded: S.expanded, select, goStep, toggleExpand: () => { S.expanded = !S.expanded; S.panelsKey = ""; renderPanels(); } });

  const refreshPanels = () => { S.panelsKey = ""; renderPanels(); };

  function renderPanels() {
    const ep = S.ep, step = curStep(), key = `${step}|${S.selected}|${S.expanded}`;
    if (!ep || key === S.panelsKey) return;
    S.panelsKey = key;
    const u = ui();
    const body = (id) => $(`${id}-body`);
    if (body("og-tracked")) keepingFocus(body("og-tracked"), () => OG.render.tracked(body("og-tracked"), ep, step, u));
    if (body("og-risk")) keepingFocus(body("og-risk"), () => OG.render.risk(body("og-risk"), ep, step, u));
    if (body("og-fuel")) {
      keepingFocus(body("og-fuel"), () => OG.render.fuel(body("og-fuel"), ep, step, u));
      drawCanvas($("og-fuel-canvas"), 52, (g, w, h) => OG.charts.drawFuelArea(g, w, h, ep, step));
    }
    if (body("og-inspector")) {
      keepingFocus(body("og-inspector"), () => OG.render.inspector(body("og-inspector"), ep, step, u));
      mountInspectorChart(step);
    }
    settleRoving();
    renderBanner(step);
    renderStrip(step);
    if (S.timelineSel !== S.selected) { S.timelineSel = S.selected; renderTimelineMarks(); } else renderTimelineNotes();
  }

  /* The inspector chart is a persistent Plotly element inside a wrapper that also holds the playhead; panel re-renders
     only re-parent the wrapper, and a step only moves the playhead (a transform, no Plotly call). */
  function mountInspectorChart(step) {
    const slot = $("og-inspector-chart");
    if (!slot || !root.Plotly) return;
    if (!S.chart.el) {
      S.chart.el = document.createElement("div");
      S.chart.head = document.createElement("div");
      S.chart.wrap = document.createElement("div");
      S.chart.wrap.append(S.chart.el, S.chart.head);
      S.chart.wrap.id = "og-inspector-chart";
      S.chart.wrap.className = "og-insp-chart";
      S.chart.el.className = "og-insp-plot"; // set once: Plotly adds its own classes to this node and its CSS depends on them
      S.chart.head.className = "og-insp-playhead";
      S.chart.wrap.setAttribute("aria-hidden", "true"); // the readouts beside it carry the same numbers as text
    }
    slot.replaceWith(S.chart.wrap);
    const key = `${S.ep.meta.scenario_id}|${S.ep.meta.policy}|${S.ep.shield}|${S.selected}`;
    if (S.chart.key !== key) {
      S.chart.key = key;
      const fig = OG.charts.inspectorFigure(S.ep, S.selected);
      root.Plotly.react(S.chart.el, fig.data, { ...fig.layout, height: S.chart.wrap.clientHeight || undefined }, { displayModeBar: false, staticPlot: true, responsive: true });
    }
    movePlayhead(step);
  }

  function movePlayhead(step) {
    const { wrap, head } = S.chart;
    if (!wrap || !head) return;
    head.style.transform = `translateX(${OG.charts.playheadOffset(step, S.ep.T, wrap.clientWidth)}px)`;
  }

  function renderBanner(step) {
    const ep = S.ep, t = OG.playback.clockText(step, ep.controlDtS), b = OG.panels.banner(ep, step);
    const clock = $("og-clock"), host = $("og-banner");
    if (clock) OG.dom.mount(clock, OG.dom.h("div", { class: "og-clock-t" }, `T+${t.minutes}`, OG.dom.h("span", { class: "og-clock-unit" }, "min")),
      OG.dom.h("div", { class: "og-clock-sub" }, `${t.hms} · step ${step} / ${ep.T}`),
      ep.shield ? null : OG.dom.h("div", { class: "og-chip og-clock-chip", style: { "--c": ep.vocab.event[2].color } }, "◇ SHIELD OFF"));
    const text = b ? `${b.type}|${b.sats}|${b.more}` : "";
    if (host && S.bannerText !== text) OG.dom.mount(host, b ? OG.dom.h("div", { class: `og-banner-card is-${b.type}`, style: { "--c": b.color } },
      OG.dom.h("span", { class: "og-banner-main" }, `${b.glyph} ${b.label} · ${b.sats}`), b.more ? OG.dom.h("span", { class: "og-banner-more" }, b.more) : null) : null);
    S.bannerText = text;
  }

  /* The legend dims the shield entry (and says so) when the episode ran without the shield. */
  function renderLegend() {
    const shieldName = S.ep.vocab.event[2].name;
    for (const chip of document.querySelectorAll(".og-legend-chip")) {
      const off = chip.dataset.legend === shieldName && !S.ep.shield;
      chip.classList.toggle("is-off", off);
      const note = chip.querySelector(".og-legend-off");
      if (note) note.textContent = off ? "(off)" : "";
    }
  }

  function renderStrip(step) {
    const running = OG.models.stripRunning(S.ep, step);
    for (const [metric, text] of Object.entries(running)) setText(`og-strip-run-${metric}`, text);
  }

  /* The strip's totals count up to their new values when a tab or the shield switches (the end value is the recorded one). */
  function countUpStrip() {
    const tweens = [];
    for (const el of document.querySelectorAll(".og-cell-val")) {
      const metric = (el.id || "").replace("og-strip-val-", ""), target = parseFloat(el.textContent), before = S.stripVals[metric];
      const decimals = (el.textContent.split(".")[1] || "").length;
      if (Number.isFinite(target)) S.stripVals[metric] = target; else delete S.stripVals[metric];
      if (Number.isFinite(target) && Number.isFinite(before) && before !== target) tweens.push({ el, current: before, target, decimals });
    }
    S.stripTween = tweens.length && !reduced() ? tweens : null;
    if (S.stripTween) kick();
  }

  function stepStripTween(dt) {
    if (!S.stripTween) return;
    S.stripTween = S.stripTween.filter((t) => t.el.isConnected);
    if (!S.stripTween.length) { S.stripTween = null; return; }
    for (const t of S.stripTween) { t.current = OG.motion.tween(t.current, t.target, dt, OG.motion.DURATION.count, false); t.el.textContent = t.current.toFixed(t.decimals); }
    if (S.stripTween.every((t) => t.current === t.target)) S.stripTween = null;
  }

  /* ------------------------------------------------------------ canvases */
  function drawCanvas(canvas, height, paint) {
    if (!canvas) return null;
    const ratio = root.devicePixelRatio || 1, w = canvas.parentElement ? canvas.parentElement.clientWidth : canvas.clientWidth;
    if (!w) return null;
    canvas.style.width = `${w}px`; canvas.style.height = `${height}px`;
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(height * ratio);
    const g = canvas.getContext("2d");
    g.setTransform(ratio, 0, 0, ratio, 0, 0);
    return paint(g, w, height) || true;
  }

  function drawScrubber() {
    const canvas = $("og-scrub-density"), host = $("og-scrub");
    if (!canvas || !host || !S.ep) return;
    const ratio = root.devicePixelRatio || 1, w = host.clientWidth, h = host.clientHeight;
    if (!w) return;
    canvas.width = Math.round(w * ratio); canvas.height = Math.round(h * ratio);
    const g = canvas.getContext("2d");
    g.setTransform(ratio, 0, 0, ratio, 0, 0);
    OG.charts.drawScrubDensity(g, w, h, S.ep);
  }

  function moveScrubber() {
    const ep = S.ep, host = $("og-scrub");
    if (!host) return;
    const w = host.clientWidth, u = S.p / ep.T, step = curStep();
    const fill = $("og-scrub-fill"), thumb = $("og-scrub-thumb");
    if (fill) fill.style.transform = `scaleX(${u})`;
    if (thumb) thumb.style.transform = `translateX(${(u * w).toFixed(1)}px)`;
    const keys = ep.payload.key_steps, off = (id, disabled) => { const el = $(id); if (el) el.setAttribute("aria-disabled", String(disabled)); };
    off("og-step-back", step <= 0); off("og-step-fwd", step >= ep.T);
    off("og-prev-event", OG.playback.prevKey(keys, step) === null); off("og-next-event", OG.playback.nextKey(keys, step) === null);
    const clock = performance.now();
    if (!S.playing || clock - lastAriaValue >= ARIA_VALUE_INTERVAL_MS) { // a screen reader must not be fed 30 updates a second
      lastAriaValue = clock;
      host.setAttribute("aria-valuenow", String(step));
      host.setAttribute("aria-valuetext", `step ${step} of ${ep.T}, T+${OG.playback.clockText(step, ep.controlDtS).minutes} min`);
    }
    const scene = $("og-scene");
    if (scene && !S.playing) scene.setAttribute("aria-label", `3D constellation view. ${stepSummary(step)}`);
    const head = $("og-timeline-head");
    if (head) head.style.transform = `translateX(${(u * head.parentElement.clientWidth).toFixed(1)}px)`;
  }

  /* ------------------------------------------------------------ timeline */
  function renderTimeline() {
    const host = $("og-timeline-body");
    if (!host || !S.ep) return;
    keepingFocus(host, () => buildTimeline(host));
  }

  function buildTimeline(host) {
    const { h, mount } = OG.dom;
    const tab = (mode, label) => h("button", { type: "button", "data-focus-key": `tl-${mode}`, class: `og-seg og-tl-tab${S.timeline === mode ? " is-active" : ""}`, "aria-pressed": S.timeline === mode, on: { click: () => { S.timeline = mode; renderTimeline(); } } }, label);
    mount(host,
      h("div", { class: "og-segmented og-tl-tabs", role: "group", "aria-label": "Timeline view" }, tab("counts", "Counts"), tab("sat", "By satellite")),
      h("div", { id: "og-timeline-wrap", class: "og-tl-wrap" }, h("canvas", { id: "og-timeline-canvas", class: "og-tl-canvas" }), h("div", { id: "og-timeline-head", class: "og-tl-head" }),
        h("div", { id: "og-timeline-tip", class: "og-scrub-hover", hidden: true, role: "tooltip" })),
      h("div", { id: "og-timeline-notes", class: "og-tl-notes" }));
    const wrap = $("og-timeline-wrap");
    wrap.addEventListener("pointerdown", timelineClick);
    wrap.addEventListener("pointermove", timelineHover);
    wrap.addEventListener("pointerleave", () => { const tip = $("og-timeline-tip"); if (tip) tip.hidden = true; });
    drawTimeline();
  }

  function drawTimeline() {
    const ep = S.ep, canvas = $("og-timeline-canvas");
    if (!ep || !canvas) return;
    const counts = S.timeline === "counts", height = counts ? TIMELINE_HEIGHT : MATRIX_HEIGHT;
    S.lanes = null;
    drawCanvas(canvas, height, (g, w, h) => {
      if (counts) { S.lanes = OG.charts.drawLanes(g, w, h, ep).lanes; OG.charts.drawLaneMarks(g, w, ep, S.lanes, S.selected); }
      else OG.charts.drawMatrix(g, w, h, ep, S.selected);
    });
    const head = $("og-timeline-head");
    if (head) head.style.height = `${height}px`;
    canvas.setAttribute("role", "img");
    S.timelineSel = S.selected;
    renderTimelineNotes();
    moveScrubber();
  }

  const renderTimelineMarks = () => { if (S.ep && $("og-timeline-canvas")) drawTimeline(); };

  function renderTimelineNotes() {
    const notes = $("og-timeline-notes");
    if (!notes) return;
    const run = OG.models.runningCounts(S.ep, curStep()), v = S.ep.vocab.event;
    const canvas = $("og-timeline-canvas");
    if (canvas) canvas.setAttribute("aria-label", `${S.timeline === "counts" ? "Event counts per step" : "Events per satellite and step"}: ${[1, 2, 3].map((c) => `${v[c].label} ${[run.maneuver, run.shield, run.collision][c - 1]}`).join(", ")}. Use previous and next event to move between key steps.`);
    OG.dom.mount(notes, [[1, run.maneuver], [2, run.shield], [3, run.collision]].map(([code, text]) => OG.dom.h("span", { class: "og-tl-note", style: { "--c": v[code].color } }, `${v[code].glyph} ${v[code].label} ${text}`)));
  }

  function eventTarget(event, element) {
    const rect = element.getBoundingClientRect();
    return { rect, x: clamp(event.clientX - rect.left, 0, rect.width), y: clamp(event.clientY - rect.top, 0, rect.height) };
  }

  function timelineClick(event) {
    const canvas = $("og-timeline-canvas");
    if (!canvas || !S.ep) return;
    const { rect, x, y } = eventTarget(event, canvas);
    if (!rect.width) return;
    const step = Math.round((x / rect.width) * S.ep.T);
    if (S.timeline === "sat") select(clamp(Math.floor((y / rect.height) * S.ep.N), 0, S.ep.N - 1));
    goStep(step);
  }

  function showTip(tip, host, x, step) {
    OG.dom.mount(tip, OG.dom.h("div", { class: "og-tip-head" }, `step ${step} · T+${OG.playback.clockText(step, S.ep.controlDtS).minutes} min`),
      OG.panels.hoverLines(S.ep, step).map((l) => OG.dom.h("div", { class: "og-tip-line", style: { "--c": l.color || "var(--og-text-3)" } }, `${l.glyph} ${l.text}`)));
    tip.hidden = false;
    tip.style.transform = `translateX(${clamp(x - tip.offsetWidth / 2, 0, Math.max(0, host.clientWidth - tip.offsetWidth - HOVER_PAD_PX)).toFixed(0)}px)`;
  }

  function timelineHover(event) {
    const canvas = $("og-timeline-canvas"), tip = $("og-timeline-tip");
    if (!canvas || !tip || !S.ep) return;
    const { rect, x } = eventTarget(event, canvas);
    showTip(tip, canvas, x, Math.round((x / rect.width) * S.ep.T));
  }

  /* ------------------------------------------------------------ actions */
  function kick() { if (!S.raf) S.raf = root.requestAnimationFrame(frame); }
  function requestDraw(force) { S.needDraw = force ? "force" : (S.needDraw || true); kick(); }

  function setPlaying(on) {
    if (!S.ep) return;
    if (on && S.p >= S.ep.T) S.p = 0;
    if (!on && S.playing) { S.p = Math.floor(S.p); S.panelsKey = ""; }  // a paused scene shows a recorded frame, never an in-between one
    S.playing = on;
    const button = $("og-play");
    if (button) {
      button.setAttribute("aria-pressed", String(on)); button.title = on ? "Pause (Space)" : "Play (Space)";
      const icon = button.querySelector(".og-ico");
      if (icon) icon.className = `og-ico ${on ? "og-ico-pause" : "og-ico-play"}`;
    }
    requestDraw(false);
  }

  function select(sat) {
    if (!S.ep || sat === S.selected || sat < 0 || sat >= S.ep.N) return;
    S.selected = sat; S.panelsKey = ""; renderPanels(); requestDraw(true);
  }

  /* Jump (or step) to `step`. Animations belong only to a single step forward; a jump starts none. */
  function goStep(step, now = performance.now()) {
    const ep = S.ep;
    if (!ep) return;
    if (!Number.isFinite(step)) return;
    const next = clamp(Math.round(step), 0, ep.T), from = curStep();
    S.p = next;
    if (next !== from) { startAnimations(OG.playback.runStartsBetween(ep, from, next), next, now, next !== from + 1); announce(stepSummary(next)); }
    requestDraw(false);
  }

  const collisionAt = (step) => { for (let i = 0; i < S.ep.N; i += 1) if (S.ep.event[step * S.ep.N + i] === 3) return true; return false; };

  /* Plume, ring and conjunction lines belong to the step just played. The flash is a screen overlay that runs to its end
     (or, under reduced motion, stays as a static red frame for as long as the step has a collision). */
  function startAnimations(starts, step, now, jumped) {
    const a = S.anim;
    a.plume = starts.maneuver.length ? now : null;
    a.ring = starts.shield.length ? now : null;
    a.lines = S.ep.payload.conjunction_entries.includes(step) ? now : null;
    if (starts.collision.length || (jumped && collisionAt(step))) a.flash = now;
    else if (reduced()) a.flash = null;
  }

  const jumpEvent = (direction) => {
    if (!S.ep) return;
    const keys = S.ep.payload.key_steps, step = curStep();
    const target = direction > 0 ? OG.playback.nextKey(keys, step) : OG.playback.prevKey(keys, step);
    if (target !== null) goStep(target);
  };

  function setSpeed(speed) {
    S.speed = speed;
    for (const button of document.querySelectorAll(".og-speed-btn")) {
      const on = Number(button.dataset.speed) === speed;
      button.classList.toggle("is-active", on); button.setAttribute("aria-pressed", String(on));
    }
  }

  function setLoop(on) { S.loop = on; const b = $("og-loop"); if (b) b.setAttribute("aria-pressed", String(on)); }

  function setOrbits(on) { S.orbits = on; const b = $("og-orbits"); if (b) b.setAttribute("aria-pressed", String(on)); requestDraw(true); }

  /* ------------------------------------------------------------ frame loop */
  function view(now, step) {
    const r = reduced(), D = OG.motion.DURATION, g = (start, d) => OG.motion.growth(start, now, d, r);
    return { step, frac: r ? 0 : OG.playback.frac(S.p, S.ep.T), selected: S.selected, orbits: S.orbits, labelAll: false,
      grow: { plume: g(S.anim.plume, D.plume), ring: g(S.anim.ring, D.ring), lines: g(S.anim.lines, D.line) } };
  }

  function animating(now) {
    const a = S.anim, D = OG.motion.DURATION;
    return (a.plume !== null && now - a.plume < D.plume) || (a.ring !== null && now - a.ring < D.ring) || (a.lines !== null && now - a.lines < D.line) || (a.flash !== null && now - a.flash < D.flash);
  }

  function paintFlash(now) {
    const host = $("og-stage");
    if (!host) return;
    let el = $("og-flash");
    if (!el) { el = document.createElement("div"); el.id = "og-flash"; el.className = "og-flash"; el.setAttribute("aria-hidden", "true"); host.append(el); }
    const alpha = OG.motion.flashAlpha(S.anim.flash, now, reduced());
    el.style.opacity = alpha.toFixed(3);
  }

  /* Draw one frame; the loading overlay goes away only once Plotly has really drawn, and any failure is shown, not hidden. */
  function drawNow(v, now, forced) {
    const ep = S.ep;
    try {
      if (!root.Plotly) throw new Error("the plotting library did not load");
      Promise.resolve(stage.draw(v, now, forced, S.playing)).then(() => {
        const loading = $("og-loading");
        if (loading && S.ep === ep) loading.hidden = true;
      }).catch((error) => { if (S.ep === ep) fail("COULD NOT DRAW THE SCENE", error); });
    } catch (error) { fail("COULD NOT DRAW THE SCENE", error); }
  }

  /* Never leave a previous episode on screen under another episode's header. */
  function clearChrome() {
    for (const id of ["og-clock", "og-banner", "og-timeline-head", "og-flash"]) { const el = $(id); if (el) { el.replaceChildren(); if (id === "og-flash") el.style.opacity = "0"; } }
    for (const el of document.querySelectorAll(".og-cell-run")) el.textContent = "";
    const host = $("og-scrub"), fill = $("og-scrub-fill"), thumb = $("og-scrub-thumb"), density = $("og-scrub-density");
    if (fill) fill.style.transform = "scaleX(0)";
    if (thumb) thumb.style.transform = "translateX(0)";
    if (density && density.getContext) density.getContext("2d").clearRect(0, 0, density.width, density.height);
    if (host) { host.removeAttribute("aria-valuenow"); host.removeAttribute("aria-valuetext"); }
    S.stripTween = null; S.panelsKey = ""; S.bannerText = null; S.last = 0;
  }

  function fail(title, error) {
    setPlaying(false);
    S.ep = null;
    clearChrome();
    if (stage) stage.clear();
    for (const id of ["og-tracked", "og-inspector", "og-risk", "og-fuel", "og-timeline"]) { const body = $(`${id}-body`); if (body) body.replaceChildren(); }
    const detail = String((error && error.message) || error || "").slice(0, 160);
    setStatus(title, detail);
    alertText(`${title}. ${detail}`);
  }

  function frame(now) {
    S.raf = 0;
    try { tick(now); } catch (error) { fail("COULD NOT UPDATE THE REPLAY", error); }
  }

  function tick(now) {
    const ep = S.ep;
    if (!ep) return;
    const dt = S.last ? Math.min(MAX_FRAME_DT_MS, now - S.last) : 0;
    S.last = now;
    const before = curStep();
    if (S.playing) {
      const next = OG.playback.advance({ p: S.p, playing: true }, dt, { speed: S.speed, loop: S.loop, T: ep.T, stepMs: stepMs() });
      S.p = next.p;
      if (!next.playing) setPlaying(false);
      const after = curStep();
      if (after !== before) {
        const starts = OG.playback.runStartsBetween(ep, before, after);
        startAnimations(starts, after, now, after !== before + 1);
        if (starts.collision.length) alertText(`Collision at step ${after}: ${starts.collision.map((i) => `S${i}`).join(" and ")}`);
      }
    }
    stepStripTween(dt);
    const step = curStep(), gestureActive = stage.gestureActive(now), settled = stage.settled(now);
    if (step !== before || S.needDraw) renderPanels();
    moveScrubber();
    paintFlash(now);
    const due = now - S.lastDraw >= DRAW_INTERVAL_MS;
    if (S.needDraw === "force" || settled || (due && (S.playing || S.needDraw || animating(now) || gestureActive))) {
      S.lastDraw = now;
      const forced = S.needDraw === "force" || settled;
      S.needDraw = false;
      drawNow(view(now, step), now, forced);
    }
    if (S.playing || animating(now) || gestureActive || S.needDraw || S.stripTween) kick();
    else S.last = 0; // idle: the next burst of motion starts with a fresh clock, not a huge dt
  }

  /* ------------------------------------------------------------ inputs */
  function bindOnce() {
    if (S.bound) return;
    S.bound = true;
    const on = (id, type, fn) => { const el = $(id); if (el) el.addEventListener(type, fn); };
    on("og-play", "click", () => setPlaying(!S.playing));
    on("og-step-back", "click", () => goStep(curStep() - 1));
    on("og-step-fwd", "click", () => goStep(curStep() + 1));
    on("og-prev-event", "click", () => jumpEvent(-1));
    on("og-next-event", "click", () => jumpEvent(1));
    on("og-loop", "click", () => setLoop(!S.loop));
    on("og-orbits", "click", () => setOrbits(!S.orbits));
    on("og-zoom-in", "click", () => { stage.zoom(1 / 1.25); requestDraw(true); });
    on("og-zoom-out", "click", () => { stage.zoom(1.25); requestDraw(true); });
    on("og-view-reset", "click", () => { stage.resetCamera(); requestDraw(true); });
    for (const button of document.querySelectorAll(".og-speed-btn")) button.addEventListener("click", () => setSpeed(Number(button.dataset.speed)));
    bindScrubber();
    bindScene();
    const releasePress = () => { if (S.pressedIn) { S.pressedIn = null; if (S.ep) refreshPanels(); } };
    document.addEventListener("pointerdown", (event) => { S.pressedIn = event.target.closest ? event.target.closest(".og-pane-body, #og-timeline-body") : null; root.setTimeout(releasePress, 1200); }, true);
    document.addEventListener("click", () => root.setTimeout(releasePress, 0));
    document.addEventListener("pointercancel", releasePress);
    for (const button of document.querySelectorAll(".og-phone-tab")) button.addEventListener("click", () => setPhoneTab(button.dataset.tab));
    if (root.matchMedia) root.matchMedia(PHONE_QUERY).addEventListener("change", placeDock);
    placeDock();
    document.addEventListener("keydown", (event) => {
      if (!S.ep || event.ctrlKey || event.metaKey || event.altKey || (event.repeat && event.key === " ")) return;
      if (event.key === "Escape") for (const id of ["og-scrub-hover", "og-timeline-tip"]) { const tip = $(id); if (tip) tip.hidden = true; }
      if (rove(event)) return;
      const t = event.target;
      const scoped = t === document.body || t === document.documentElement || ["og-stage", "og-playback", "og-timeline", "og-phone-inspect"].some((id) => $(id) && $(id).contains(t)); // single-key shortcuts only act near the player (WCAG 2.1.4)
      if (!scoped) return;
      const action = OG.playback.keyAction(event.key, { shift: event.shiftKey, tag: t.tagName || "", editable: Boolean(t.isContentEditable), role: t.getAttribute ? t.getAttribute("role") || "" : "" });
      if (!action) return;
      event.preventDefault();
      if (action.type === "toggle") setPlaying(!S.playing);
      else if (action.type === "event") jumpEvent(action.direction);
      else goStep(curStep() + action.delta);
    });
    if (root.ResizeObserver) {
      let width = 0, queued = false;
      new root.ResizeObserver((entries) => {
        const next = Math.round(entries[entries.length - 1].contentRect.width);
        if (next === width || queued) return; // height changes come from our own re-renders: ignore them
        width = next; queued = true;
        root.requestAnimationFrame(() => { queued = false; placeDock(); if (S.ep) { drawScrubber(); drawTimeline(); S.panelsKey = ""; renderPanels(); } });
      }).observe($("og-app") || document.body);
    }
    root.addEventListener("blur", () => stage && stage.releaseGesture());
    document.addEventListener("visibilitychange", () => { if (document.hidden && stage) stage.releaseGesture(); });
  }

  /* Arrow keys inside the satellite cell strip and the picker: one tab stop each, arrows move between members. */
  function rove(event) {
    const group = event.target.closest && event.target.closest(".og-cells, .og-picker");
    if (!group || !["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"].includes(event.key)) return false;
    const buttons = [...group.querySelectorAll("button")], index = buttons.indexOf(event.target), columns = group.classList.contains("og-picker") ? PICKER_COLUMNS : 1;
    const delta = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -columns, ArrowDown: columns }[event.key];
    const next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : clamp(index + delta, 0, buttons.length - 1);
    if (index < 0 || (columns === 1 && (event.key === "ArrowUp" || event.key === "ArrowDown"))) return false;
    event.preventDefault();
    for (const [n, b] of buttons.entries()) b.tabIndex = n === next ? 0 : -1;
    buttons[next].focus();
    return true;
  }

  function settleRoving() {
    for (const group of document.querySelectorAll(".og-cells, .og-picker")) {
      const buttons = [...group.querySelectorAll("button")], keep = buttons.find((b) => b.tabIndex === 0 && b === document.activeElement) || buttons.find((b) => b.getAttribute("aria-pressed") === "true") || buttons[0];
      for (const b of buttons) b.tabIndex = b === keep ? 0 : -1;
    }
  }

  function bindScene() {
    const scene = $("og-scene");
    if (!scene) return;
    scene.addEventListener("keydown", (event) => {
      if (!S.ep || !stage) return;
      const turn = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[event.key];
      if (turn) stage.rotate(turn[0], turn[1]);
      else if (event.key === "+" || event.key === "=") stage.zoom(1 / 1.25);
      else if (event.key === "-") stage.zoom(1.25);
      else return;
      event.preventDefault(); event.stopPropagation();
      requestDraw(true);
    });
  }

  function bindScrubber() {
    const host = $("og-scrub");
    if (!host) return;
    let dragging = false;
    const stepAt = (event) => OG.playback.scrubStep(event.clientX, host.getBoundingClientRect(), S.ep.T);
    host.addEventListener("pointerdown", (event) => {
      if (!S.ep || event.button > 0) return;
      dragging = true;
      try { host.setPointerCapture(event.pointerId); } catch (error) { /* a synthetic pointer cannot be captured; dragging still works while over the bar */ }
      goStep(stepAt(event));
    });
    host.addEventListener("pointermove", (event) => {
      if (!S.ep) return;
      const step = stepAt(event), tip = $("og-scrub-hover");
      if (dragging) goStep(step);
      if (tip) showTip(tip, host, event.clientX - host.getBoundingClientRect().left, step);
    });
    const stop = () => { dragging = false; };
    host.addEventListener("pointerup", stop); host.addEventListener("pointercancel", stop); host.addEventListener("lostpointercapture", stop);
    host.addEventListener("pointerleave", () => { const tip = $("og-scrub-hover"); if (tip) tip.hidden = true; });
    host.addEventListener("keydown", (event) => {
      if (!S.ep) return;
      if (event.key === " ") { event.preventDefault(); if (!event.repeat) setPlaying(!S.playing); return; }
      const next = OG.playback.sliderKey(event.key, { shift: event.shiftKey }, curStep(), S.ep.T);
      if (next === null) return;
      event.preventDefault(); goStep(next);
    });
  }

  /* ------------------------------------------------------------ phone */
  const PHONE_QUERY = "(max-width: 759px)";
  const isPhone = () => Boolean(root.matchMedia && root.matchMedia(PHONE_QUERY).matches);

  function setPhoneTab(tab) {
    const app = $("og-app");
    if (app) app.setAttribute("data-phone-tab", tab);
    for (const button of document.querySelectorAll(".og-phone-tab")) {
      const on = button.dataset.tab === tab;
      button.classList.toggle("is-active", on); button.setAttribute("aria-pressed", String(on));
    }
    S.panelsKey = ""; renderPanels(); drawTimeline();
  }

  /* On a phone the dock and the legend live in the Inspect tab instead of on top of the 3D view. */
  function placeDock() {
    const dock = $("og-dock"), legend = $("og-legend"), stage = $("og-stage"), mount = $("og-phone-inspect");
    if (!dock || !legend || !stage || !mount) return;
    const column = document.querySelector(".og-col-main"), wide = !isPhone() && column && column.clientWidth >= DOCK_MIN_PX;
    const target = wide ? stage : mount;
    if (dock.parentElement === target) return;
    target.append(legend, dock);
    S.panelsKey = ""; if (S.ep) { renderPanels(); drawTimeline(); requestDraw(true); }
  }

  /* ------------------------------------------------------------ boot */
  function adopt(ep) {
    const previous = S.ep, sameScenario = previous && previous.meta.scenario_id === ep.meta.scenario_id;
    const first = stage.setEpisode(ep);
    S.p = sameScenario ? Math.min(Math.round(S.p), ep.T) : ep.openStep;
    S.selected = previous && previous.N === ep.N ? S.selected : 0;
    S.anim = { plume: null, ring: null, lines: null, flash: null };
    if (!sameScenario) { S.playing = false; stage.resetCamera(); }
    S.ep = ep; S.panelsKey = ""; S.chart.key = ""; S.bannerText = null; S.timelineSel = null;
    if (collisionAt(Math.floor(S.p))) S.anim.flash = performance.now();
    return first;
  }

  function boot(payload) {
    if (!payload) { fail("NO TRACE", ""); return "empty"; }
    try {
      if (!stage) stage = OG.createStage({ onSelect: select, requestDraw, onNotice: alertText });
      const ep = OG.decodeEpisode(payload);
      setStatus("LOADING TRACE", `pre-recorded episode · ${ep.T + 1} frames · ${ep.P} pairs`);
      adopt(ep);
      bindOnce();
      setPlaying(S.playing);
      const host = $("og-scrub");
      if (host) { host.setAttribute("aria-valuemax", String(ep.T)); drawScrubber(); }
      renderTimeline();
      renderLegend();
      countUpStrip();
      renderPanels();
      requestDraw("force");
      return "ready";
    } catch (error) {
      fail("COULD NOT READ TRACE", error);
      return "error";
    }
  }

  root.dash_clientside = root.dash_clientside || {};
  root.dash_clientside.og = { ...(root.dash_clientside.og || {}), boot };
})(globalThis);
