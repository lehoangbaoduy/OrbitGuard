/* OrbitGuard replay stage - chart drawing: scrubber density bars, timeline lanes, the satellite-by-step matrix, the
   fleet-fuel area and the inspector figure. Canvas functions take any 2D context (tests pass a recorder); every shape is
   a rendering of a recorded series from the payload, coloured by the payload's own vocabulary. */
(function (root) {
  "use strict";
  const OG = root.OG;
  const MONO = "Martian Mono, ui-monospace, monospace";
  const TEXT_3 = "#9397ab"; // --og-text-3
  const SCRUB = { shieldBase: 13, maneuverBase: 28, maxBar: 11, minBar: 2 };
  const LANE_GAP = 6;

  const colors = (ep) => ({ maneuver: ep.vocab.event[1].color, shield: ep.vocab.event[2].color, collision: ep.vocab.event[3].color, risk: ep.vocab.sat_status[2].color });

  /* Scrubber: shield bars (upper row), maneuver bars (lower row), amber diamonds at conjunction entries. */
  function drawScrubDensity(g, w, h, ep) {
    const c = colors(ep), d = ep.payload.density, bar = Math.max(1, w / ep.T);
    g.clearRect(0, 0, w, h);
    for (let t = 1; t <= ep.T; t += 1) {
      const x = (t / ep.T) * w, s = d["2"] ? d["2"][t] : 0, m = d["1"][t];
      if (s) { const bh = SCRUB.minBar + (s / ep.N) * SCRUB.maxBar; g.fillStyle = c.shield; g.globalAlpha = 0.85; g.fillRect(x - bar / 2, SCRUB.shieldBase - bh, bar, bh); }
      if (m) { const bh = SCRUB.minBar + (m / ep.N) * SCRUB.maxBar; g.fillStyle = c.maneuver; g.globalAlpha = 0.9; g.fillRect(x - bar / 2, SCRUB.maneuverBase - bh, bar, bh); }
    }
    g.globalAlpha = 1;
    g.fillStyle = c.risk;
    for (const step of ep.payload.conjunction_entries) {
      const x = (step / ep.T) * w;
      g.beginPath(); g.moveTo(x, 0); g.lineTo(x + 3, 3); g.lineTo(x, 6); g.lineTo(x - 3, 3); g.closePath(); g.fill();
    }
  }

  function hatch(g, w, top, height) {
    g.strokeStyle = TEXT_3; g.globalAlpha = 0.25; g.lineWidth = 1;
    for (let x = -height; x < w; x += 8) { g.beginPath(); g.moveTo(x, top + height); g.lineTo(x + height, top); g.stroke(); }
    g.globalAlpha = 1;
  }

  /* Counts view: three lanes (maneuver, shield, collision); bar height = satellites with the event at that step. */
  function drawLanes(g, w, h, ep) {
    const c = colors(ep), d = ep.payload.density, laneH = (h - 2 * LANE_GAP) / 3, bar = Math.max(1, w / ep.T);
    g.clearRect(0, 0, w, h);
    const lanes = [{ type: "maneuver", series: d["1"], color: c.maneuver, top: 0 }, { type: "shield", series: d["2"], color: c.shield, top: laneH + LANE_GAP }, { type: "collision", series: d["3"], color: c.collision, top: 2 * (laneH + LANE_GAP) }];
    for (const lane of lanes) {
      if (lane.series === null) {
        hatch(g, w, lane.top, laneH);
        g.fillStyle = TEXT_3; g.font = `11px ${MONO}`; g.textAlign = "left"; g.fillText("Shield off — no overrides recorded", 8, lane.top + laneH / 2 + 4);
        continue;
      }
      for (let t = 0; t <= ep.T; t += 1) {
        const n = lane.series[t];
        if (!n) continue;
        const bh = Math.max(2, (n / ep.N) * laneH), x = (t / ep.T) * w - bar / 2;
        if (lane.type === "shield") { g.strokeStyle = lane.color; g.lineWidth = 1; g.strokeRect(x, lane.top + laneH - bh, bar, bh); g.globalAlpha = 0.9; g.fillRect(x, lane.top + laneH - bh, bar, 2); g.globalAlpha = 1; }
        else { g.fillStyle = lane.color; g.globalAlpha = 0.9; g.fillRect(x, lane.top + laneH - bh, bar, bh); g.globalAlpha = 1; }
      }
    }
    return { lanes: lanes.map((l) => ({ type: l.type, top: l.top, height: laneH })) };
  }

  /* In-lane marks for the selected satellite only: a glowing run line over its event runs plus a label ("S9 ×3"). */
  function drawLaneMarks(g, w, ep, lanes, selected) {
    const kinds = { maneuver: 1, shield: 2 }, c = colors(ep);
    for (const lane of lanes) {
      let labelEnd = -Infinity; // labels never stack: one that would overlap the previous one is dropped
      const code = kinds[lane.type];
      if (!code || (lane.type === "shield" && ep.payload.density["2"] === null)) continue;
      for (const mark of OG.models.laneMarks(ep, code, selected)) {
        const x0 = (mark.start / ep.T) * w, x1 = Math.max(x0 + 2, (mark.end / ep.T) * w);
        g.save();
        g.strokeStyle = c[lane.type]; g.lineWidth = 2; g.shadowColor = c[lane.type]; g.shadowBlur = 6;
        g.beginPath(); g.moveTo(x0, lane.top + 2); g.lineTo(x1, lane.top + 2); g.stroke();
        g.restore();
        g.fillStyle = "#cfd3e5"; g.font = `11px ${MONO}`;
        const right = x1 > w - 60, width = g.measureText ? g.measureText(mark.label).width : mark.label.length * 7, left = right ? x1 - width : x0;
        if (left < labelEnd + 4) continue;
        g.textAlign = right ? "right" : "left";
        g.fillText(mark.label, right ? x1 : x0, lane.top + 14);
        labelEnd = left + width;
      }
    }
  }

  /* By-satellite view: a row per satellite and a column per step; tall = maneuver, short = shield, red = collision. */
  function drawMatrix(g, w, h, ep, selected) {
    const c = colors(ep), rowH = h / ep.N, colW = w / ep.T;
    g.clearRect(0, 0, w, h);
    if (selected >= 0) { g.fillStyle = OG.ICE; g.globalAlpha = 0.08; g.fillRect(0, selected * rowH, w, rowH); g.globalAlpha = 1; }
    for (let s = 1; s <= ep.T; s += 1) {
      for (let i = 0; i < ep.N; i += 1) {
        const e = ep.event[s * ep.N + i];
        if (!e) continue;
        const x = ((s - 0.5) / ep.T) * w - colW / 2, y = i * rowH, cw = Math.max(1, colW);
        if (e === 1) { g.fillStyle = c.maneuver; g.fillRect(x, y + rowH * 0.1, cw, rowH * 0.8); }
        else if (e === 2) { g.fillStyle = c.shield; g.fillRect(x, y + rowH * 0.55, cw, rowH * 0.3); }
        else { g.fillStyle = c.collision; g.fillRect(x, y, cw, rowH); }
      }
    }
  }

  /* Fleet fuel over the episode as a filled area (usable tanks only), with the playhead and the all-empty step. */
  function drawFuelArea(g, w, h, ep, step) {
    const fuel = ep.payload.fuel, top = fuel.total_budget_mps || 1, y = (v) => h - 2 - (Math.max(0, v) / top) * (h - 6);
    g.clearRect(0, 0, w, h);
    g.beginPath(); g.moveTo(0, h);
    for (let t = 0; t <= ep.T; t += 1) g.lineTo((t / ep.T) * w, y(fuel.total_mps[t]));
    g.lineTo(w, h); g.closePath();
    g.fillStyle = OG.ICE; g.globalAlpha = 0.22; g.fill(); g.globalAlpha = 1;
    const px = (step / ep.T) * w;
    g.strokeStyle = OG.ICE; g.lineWidth = 1; g.beginPath(); g.moveTo(px, 0); g.lineTo(px, h); g.stroke();
    if (fuel.all_empty_step !== null) { g.fillStyle = ep.vocab.sat_status[2].color; g.beginPath(); g.moveTo((fuel.all_empty_step / ep.T) * w - 3, h - 4); g.lineTo((fuel.all_empty_step / ep.T) * w + 3, h - 4); g.lineTo((fuel.all_empty_step / ep.T) * w, h - 9); g.closePath(); g.fill(); }
  }

  const AXIS_STYLE = { gridcolor: "rgba(219,230,243,0.08)", zeroline: false, linecolor: "rgba(219,230,243,0.2)", tickfont: { family: MONO, size: 11, color: TEXT_3 } };

  const INSPECTOR_MARGIN = Object.freeze({ l: 44, r: 8, t: 8, b: 22 });

  /* Inspector figure: three rows sharing the step axis (Δv left with event marks, mission deviation, closest d_min against d_safe on a log scale). */
  function inspectorFigure(ep, sat) {
    const s = OG.models.inspectorSeries(ep, sat), c = colors(ep);
    const at = (steps, values) => steps.map((t) => values[t]);
    const line = (name, y, extra) => ({ type: "scatter", mode: "lines", name, x: s.t, y, hoverinfo: "x+y", ...extra });
    const data = [
      line("Δv left", s.dv, { line: { color: OG.ICE, width: 1.5, shape: "hv" } }),
      { type: "scatter", mode: "markers", name: "maneuver", x: s.maneuver, y: at(s.maneuver, s.dv), marker: { symbol: "square", size: 7, color: c.maneuver }, hoverinfo: "x" },
      { type: "scatter", mode: "markers", name: "shield", x: s.shield, y: at(s.shield, s.dv), marker: { symbol: "diamond-open", size: 8, color: c.shield, line: { width: 1.5, color: c.shield } }, hoverinfo: "x" },
      line("mission deviation", s.dev, { yaxis: "y2", line: { color: OG.ICE, width: 1.5 } }),
      line("closest d_min", s.dMin, { yaxis: "y3", line: { color: OG.ICE, width: 1.5 } }),
      line("d_safe", s.dSafe, { yaxis: "y3", line: { color: ep.vocab.pair_status[1].color, width: 1, dash: "dot" } }),
    ];
    const note = (y, text) => ({ xref: "paper", yref: "paper", x: 0, y, text, showarrow: false, xanchor: "left", yanchor: "bottom", font: { family: MONO, size: 11, color: TEXT_3 } });
    const layout = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", margin: { ...INSPECTOR_MARGIN }, showlegend: false, hovermode: "x",
      xaxis: { ...AXIS_STYLE, range: [0, ep.T], anchor: "y3" },
      yaxis: { ...AXIS_STYLE, domain: [0.70, 0.92], nticks: 3 }, yaxis2: { ...AXIS_STYLE, domain: [0.36, 0.58], nticks: 3 }, yaxis3: { ...AXIS_STYLE, domain: [0, 0.22], type: "log", nticks: 3 },
      annotations: [note(0.93, "Δv LEFT · m/s"), note(0.59, "MISSION DEVIATION"), note(0.23, "CLOSEST d_min · ┄ d_safe (m, log)")],
    };
    return { data, layout };
  }

  /* The playhead is a thin element laid over the chart and moved with a transform: as a Plotly shape every step cost a
     relayout (~33 ms on a real GPU, 60+ per 5 s of 4x playback). `playheadOffset` is its x offset in px from the plot
     area's left edge (the chart's own left margin is applied in CSS), clamped to the plot. */
  function playheadOffset(step, steps, widthPx) {
    const plot = widthPx - INSPECTOR_MARGIN.l - INSPECTOR_MARGIN.r;
    if (!(plot > 0) || !(steps > 0) || !Number.isFinite(step)) return 0;
    return Math.min(1, Math.max(0, step / steps)) * plot;
  }

  /* Whether the inspector chart must be (re)drawn: its key or its measured size changed (a phone tab opened, the dock
     moved), never while hidden (a hidden panel measures 0x0 and Plotly would size the chart to garbage). */
  function inspectorDrawPlan(last, key, width, height) {
    if (!(width > 0) || !(height > 0)) return "defer";
    return last && last.key === key && last.width === width && last.height === height ? "keep" : "draw";
  }

  OG.charts = { inspectorDrawPlan, drawScrubDensity, drawLanes, drawLaneMarks, drawMatrix, drawFuelArea, inspectorFigure, playheadOffset, INSPECTOR_MARGIN };
})(globalThis);
