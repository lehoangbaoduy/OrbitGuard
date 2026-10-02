/* OrbitGuard replay stage - view-models for the panels around the stage: "Tracked this step", the event banner, the
   scrubber hover and the conjunction table. Pure functions of (episode, step): they pick, sort, group and format what
   the recorded arrays already say, so the DOM code only has to print them. */
(function (root) {
  "use strict";
  const OG = root.OG;
  const STATUS_COLLIDED = 3, PAIR_COLLISION = 3, PAIR_HIGH_RISK = 2, PAIR_INSIDE = 1;
  const EVENT = { maneuver: 1, shield: 2, collision: 3 };
  const GROUP_OVER = 3; // more than this many satellites with the same event become one row
  const COLLAPSED_ROWS = 6;
  const HOVER_IDS = 5;
  const ID_LIST_MAX = 4; // the banner names up to this many satellites, then counts them

  const fmtInt = (n) => Math.round(n).toLocaleString("en-US");
  const minutesAt = (ep, step) => OG.playback.clockText(step, ep.controlDtS).minutes;
  const satName = (i) => `S${i}`;

  /* First frame at which satellite `i` is retired by a collision, or -1. */
  function retiredStep(ep, i) {
    for (let t = 0; t <= ep.T; t += 1) if (ep.satStatus[t * ep.N + i] === STATUS_COLLIDED) return t;
    return -1;
  }

  function decision(ep, step, sat) {
    if (step < 1) return "";
    const k = (step - 1) * ep.N + sat;
    return ` · ${ep.catalog[ep.proposed[k]].short} → ${ep.catalog[ep.approved[k]].short}`;
  }

  function satSub(ep, step, sat, status) {
    if (status === STATUS_COLLIDED) return `retired · impact T+${minutesAt(ep, retiredStep(ep, sat))}`;
    const a = step * ep.N + sat, partner = ep.partnerIdx[a];
    return partner >= 0 && Number.isFinite(ep.partnerDMin[a]) ? `closest ${satName(partner)} · ${fmtInt(ep.partnerDMin[a])} m` : "";
  }

  function satRow(ep, step, sat) {
    const a = step * ep.N + sat, status = ep.satStatus[a], event = ep.event[a];
    const s = ep.vocab.sat_status[status], e = ep.vocab.event[event];
    const showEvent = event && status !== STATUS_COLLIDED;
    return {
      kind: "sat", sat, rank: status * 10 + event, statusCode: status, statusName: s.name, statusGlyph: s.glyph, statusLabel: s.label, color: s.color,
      eventName: showEvent ? e.name : "", eventGlyph: showEvent ? e.glyph : "", eventLabel: showEvent ? e.label : "", eventColor: showEvent ? e.color : "",
      detail: event === EVENT.maneuver || event === EVENT.shield ? decision(ep, step, sat) : "", sub: satSub(ep, step, sat, status),
    };
  }

  function groupRow(ep, step, type, members) {
    const e = ep.vocab.event[type];
    let detail = "";
    if (type === EVENT.shield && step >= 1) {
      const outcomes = new Map();
      for (const m of members) { const label = ep.catalog[ep.approved[(step - 1) * ep.N + m.sat]].short; outcomes.set(label, (outcomes.get(label) || 0) + 1); }
      detail = ` · ${[...outcomes].map(([label, n]) => (label === ep.catalog[0].short ? `${n} blocked → ${label}` : `${n} → ${label}`)).join(" · ")}`;
    }
    return { kind: "group", eventType: type, rank: type, count: members.length, sats: members.map((m) => m.sat), glyph: e.glyph, label: e.label, color: e.color, detail, sub: members.map((m) => satName(m.sat)).join(" ") };
  }

  /* The rows of "Tracked this step". Status rows stay individual; an event type shared by more than three satellites
     collapses into one group row unless `expanded`. */
  function trackedRows(ep, step, expanded) {
    const items = [];
    for (let i = 0; i < ep.N; i += 1) if (ep.satStatus[step * ep.N + i] || ep.event[step * ep.N + i]) items.push(satRow(ep, step, i));
    items.sort((a, b) => b.rank - a.rank || a.sat - b.sat);
    const rows = items.filter((r) => r.rank >= 10), eventOnly = items.filter((r) => r.rank < 10);
    for (const type of [EVENT.collision, EVENT.shield, EVENT.maneuver]) {
      const members = eventOnly.filter((r) => r.rank === type);
      if (members.length <= GROUP_OVER || expanded) rows.push(...members);
      else rows.push(groupRow(ep, step, type, members));
    }
    const shown = rows.slice(0, expanded ? ep.N : COLLAPSED_ROWS), hidden = rows.length - shown.length;
    const showMore = expanded || hidden > 0 || eventOnly.length > GROUP_OVER;
    const moreLabel = !showMore ? "" : expanded ? "Collapse groups" : hidden > 0 ? `+${hidden} more` : "Expand groups";
    return { rows: shown, flagged: items.length, hidden, moreLabel, empty: items.length === 0 };
  }

  /* One cell per satellite: the event glyph, else the status glyph, else a dot; always with a text title. */
  function stepCells(ep, step) {
    return Array.from({ length: ep.N }, (_, i) => {
      const a = step * ep.N + i, status = ep.satStatus[a], event = ep.event[a];
      const s = ep.vocab.sat_status[status], e = ep.vocab.event[event];
      const glyph = event ? e.glyph : status ? s.glyph : "·", color = event ? e.color : status ? s.color : null;
      return { sat: i, glyph, color, flagged: Boolean(event || status), title: `${satName(i)}: ${s.label}${event ? ` · ${e.label}` : ""}` };
    });
  }

  function stepTally(ep, step) {
    let shield = 0, maneuver = 0, flagged = 0;
    for (let i = 0; i < ep.N; i += 1) {
      const a = step * ep.N + i;
      if (ep.event[a] === EVENT.shield) shield += 1;
      if (ep.event[a] === EVENT.maneuver) maneuver += 1;
      if (ep.satStatus[a]) flagged += 1;
    }
    const parts = [!ep.shield ? "◇ off" : shield ? `◇ ${shield}` : "", maneuver ? `■ ${maneuver}` : "", flagged ? `◆○ ${flagged}` : ""];
    return parts.filter(Boolean).join("  ") || "no events";
  }

  function eventsByType(ep, step) {
    const byType = { 1: [], 2: [], 3: [] };
    for (let i = 0; i < ep.N; i += 1) { const code = ep.event[step * ep.N + i]; if (code) byType[code].push(i); }
    return byType;
  }

  const listSats = (sats, joiner, maxNames) => (sats.length > maxNames ? `${sats.length} satellites` : sats.map(satName).join(joiner));

  /* The banner for the most serious event of the step (collision > shield > maneuver), or null. */
  function banner(ep, step, { maxNames = ID_LIST_MAX } = {}) {
    const byType = eventsByType(ep, step);
    const top = [EVENT.collision, EVENT.shield, EVENT.maneuver].find((t) => byType[t].length);
    if (!top) return null;
    const e = ep.vocab.event[top];
    const more = [EVENT.maneuver, EVENT.shield].filter((t) => t !== top && byType[t].length)
      .map((t) => `+ ${ep.vocab.event[t].glyph} ${byType[t].length > maxNames ? `${byType[t].length} sats` : byType[t].map(satName).join(" · ")}`).join("  ");
    return { type: e.name, glyph: e.glyph, label: e.label, color: e.color, sats: listSats(byType[top], top === EVENT.collision ? " × " : " · ", maxNames), more };
  }

  /* Tooltip lines for the scrubber hover: the events of that step and how many pairs are at risk. */
  function hoverLines(ep, step) {
    const byType = eventsByType(ep, step), lines = [];
    for (const type of [EVENT.collision, EVENT.shield, EVENT.maneuver]) {
      const sats = byType[type];
      if (!sats.length) continue;
      const e = ep.vocab.event[type], ids = sats.map(satName);
      const shown = ids.length > HOVER_IDS ? `${ids.slice(0, HOVER_IDS).join(", ")} +${ids.length - HOVER_IDS}` : ids.join(", ");
      lines.push({ glyph: e.glyph, color: e.color, text: `${e.label} ×${sats.length} · ${shown}` });
    }
    let high = 0, inside = 0, collided = 0;
    for (let k = 0; k < ep.P; k += 1) { const s = ep.pairStatus[step * ep.P + k]; if (s === PAIR_HIGH_RISK) high += 1; else if (s === PAIR_INSIDE) inside += 1; else if (s === PAIR_COLLISION) collided += 1; }
    const pv = ep.vocab.pair_status;
    if (high) lines.push({ glyph: pv[PAIR_HIGH_RISK].glyph, color: pv[PAIR_HIGH_RISK].color, text: `${high} ${pv[PAIR_HIGH_RISK].label} pair${high > 1 ? "s" : ""}` });
    if (collided) lines.push({ glyph: pv[PAIR_COLLISION].glyph, color: pv[PAIR_COLLISION].color, text: `${collided} ${pv[PAIR_COLLISION].label} pair${collided > 1 ? "s" : ""} ` });
    if (inside) lines.push({ glyph: pv[PAIR_INSIDE].glyph, color: pv[PAIR_INSIDE].color, text: `${inside} ${pv[PAIR_INSIDE].label}` });
    return lines.length ? lines : [{ glyph: "·", color: null, text: "no events · all pairs nominal" }];
  }

  function firstCollisionStep(ep, pair) {
    for (let t = 0; t <= ep.T; t += 1) if (ep.pairStatus[t * ep.P + pair] === PAIR_COLLISION) return t;
    return -1;
  }

  /* The conjunction table: pairs ordered by status, then risk, then closest approach. A collided pair has no stale distance. */
  const finiteOr = (v, fallback) => (Number.isFinite(v) ? v : fallback);

  function riskRows(ep, step, limit) {
    const rows = [];
    for (let k = 0; k < ep.P; k += 1) {
      const a = step * ep.P + k, code = ep.pairStatus[a], v = ep.vocab.pair_status[code], collided = code === PAIR_COLLISION;
      const i = ep.pairs[2 * k], j = ep.pairs[2 * k + 1];
      rows.push({ pair: `${satName(i)}–${satName(j)}`, i, j, statusCode: code, statusName: v.name, glyph: v.glyph, label: v.label, color: v.color,
        dMin: collided ? null : ep.dMin[a], dSafe: collided ? null : ep.dSafe[a], risk: collided ? null : ep.risk[a],
        note: collided ? `impact T+${minutesAt(ep, firstCollisionStep(ep, k))}` : null });
    }
    rows.sort((x, y) => y.statusCode - x.statusCode || finiteOr(y.risk, 0) - finiteOr(x.risk, 0) || finiteOr(x.dMin, Infinity) - finiteOr(y.dMin, Infinity));
    return rows.slice(0, limit);
  }

  OG.panels = { trackedRows, stepCells, stepTally, banner, hoverLines, riskRows, retiredStep };
})(globalThis);
