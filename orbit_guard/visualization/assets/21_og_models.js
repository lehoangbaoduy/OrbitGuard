/* OrbitGuard replay stage - view-models for fleet fuel, the satellite inspector, the timeline and the summary strip.
   Pure functions of (episode, step[, satellite]); every value is read or summed from the payload's recorded arrays and
   a value that does not exist prints as an em dash, never as zero. */
(function (root) {
  "use strict";
  const OG = root.OG;
  const DASH = "—";
  const STATUS_COLLIDED = 3;
  const EARLIER_DECISIONS = 3;
  const MERGE_GAP_STEPS = 24; // runs closer than this merge into one mark on the timeline
  const EVENT_KEY = { 1: "maneuver", 2: "shield", 3: "collision" };

  const finiteOrNull = (v) => (Number.isFinite(v) ? v : null);
  const fixed = (v, digits) => (Number.isFinite(v) ? v.toFixed(digits) : DASH);
  const minutesAt = (ep, step) => OG.playback.clockText(step, ep.controlDtS).minutes;

  /* ------------------------------------------------------------- fleet fuel */
  function fuelBars(ep, step) {
    const fuel = ep.payload.fuel, offline = new Set(ep.payload.capability.offline), limited = new Set(ep.payload.capability.limited);
    const longest = Math.max(...fuel.budget_mps);
    const bars = [];
    for (let i = 0; i < ep.N; i += 1) {
      const a = step * ep.N + i, remaining = ep.remaining[a], budget = fuel.budget_mps[i];
      const kind = offline.has(i) ? "offline" : ep.fuelEmpty[a] ? "empty" : limited.has(i) ? "limited" : "ok";
      bars.push({
        sat: i, kind, remaining, budget, trackFrac: longest > 0 ? budget / longest : 0, retired: ep.satStatus[a] === STATUS_COLLIDED,
        fillFrac: kind === "offline" || budget <= 0 ? 0 : Math.min(1, Math.max(0, remaining / budget)), label: kind === "offline" ? "OFF" : kind === "empty" ? "EMPTY" : "",
      });
    }
    const emptyCount = fuel.empty_count[step];
    return { bars, total: fuel.total_mps[step], budget: fuel.total_budget_mps, stranded: fuel.stranded_mps, emptyCount, usableCount: fuel.usable_count,
      allEmptyStep: fuel.all_empty_step, text: `${emptyCount} of ${fuel.usable_count} tanks empty` };
  }

  /* -------------------------------------------------------------- inspector */
  function capabilityOf(ep, sat) {
    if (ep.payload.capability.offline.includes(sat)) return "offline";
    return ep.payload.capability.limited.includes(sat) ? "limited" : null;
  }

  function earlierDecisions(ep, step, sat) {
    const out = [];
    for (let t = step - 1; t >= 1 && out.length < EARLIER_DECISIONS; t -= 1) {
      const code = ep.event[t * ep.N + sat];
      if (code !== 1 && code !== 2) continue;
      const k = (t - 1) * ep.N + sat;
      out.push({ step: t, kind: EVENT_KEY[code], text: `T+${minutesAt(ep, t)} ${ep.catalog[ep.proposed[k]].short} → ${ep.catalog[ep.approved[k]].short}` });
    }
    return out;
  }

  /* The satellite a retired satellite collided with, and the step of impact, from the recorded pair statuses. */
  function collisionOf(ep, step, sat) {
    for (let k = 0; k < ep.P; k += 1) {
      const i = ep.pairs[2 * k], j = ep.pairs[2 * k + 1];
      if ((i !== sat && j !== sat) || ep.pairStatus[step * ep.P + k] !== 3) continue;
      let first = step;
      while (first > 0 && ep.pairStatus[(first - 1) * ep.P + k] === 3) first -= 1;
      return { partner: i === sat ? j : i, step: first };
    }
    return null;
  }

  function inspector(ep, step, sat) {
    const a = step * ep.N + sat, status = ep.satStatus[a], retired = status === STATUS_COLLIDED;
    const partner = ep.partnerIdx[a], dMin = ep.partnerDMin[a], hasPartner = !retired && partner >= 0 && Number.isFinite(dMin);
    const k = (step - 1) * ep.N + sat;
    return {
      sat, retired, collision: retired ? collisionOf(ep, step, sat) : null,
      dvFrac: ep.payload.fuel.budget_mps[sat] > 0 && Number.isFinite(ep.remaining[a]) ? Math.min(1, Math.max(0, ep.remaining[a] / ep.payload.fuel.budget_mps[sat])) : null,
      status: ep.vocab.sat_status[status], event: ep.vocab.event[ep.event[a]], capability: capabilityOf(ep, sat),
      dv: finiteOrNull(ep.remaining[a]), dvText: fixed(ep.remaining[a], 2), dev: finiteOrNull(ep.deviation[a]), devText: fixed(ep.deviation[a], 3),
      closest: hasPartner ? { partner, dMin, dSafe: ep.partnerDSafe[a] } : null,
      closestText: hasPartner ? `${Math.round(dMin).toLocaleString("en-US")} m` : DASH,
      closestNote: retired ? "retired: no live partner" : hasPartner ? `S${partner}` : "no live partner",
      noActions: step < 1,
      decision: step < 1 ? null : { proposed: ep.catalog[ep.proposed[k]].short, approved: ep.catalog[ep.approved[k]].short },
      earlier: earlierDecisions(ep, step, sat),
    };
  }

  /* Series for the inspector chart: Δv left, mission deviation, closest d_min against d_safe, with the event steps. */
  function inspectorSeries(ep, sat) {
    const t = [], dv = [], dev = [], dMin = [], dSafe = [], maneuver = [], shield = [];
    for (let s = 0; s <= ep.T; s += 1) {
      const a = s * ep.N + sat;
      t.push(s); dv.push(finiteOrNull(ep.remaining[a])); dev.push(finiteOrNull(ep.deviation[a]));
      dMin.push(finiteOrNull(ep.partnerDMin[a])); dSafe.push(finiteOrNull(ep.partnerDSafe[a]));
      if (ep.event[a] === 1) maneuver.push(s);
      if (ep.event[a] === 2) shield.push(s);
    }
    return { t, dv, dev, dMin, dSafe, maneuver, shield };
  }

  /* --------------------------------------------------------------- timeline */
  function countsLanes(ep) {
    const d = ep.payload.density;
    return { maneuver: d["1"], shield: d["2"], collision: d["3"] };
  }

  /* Runs of event `type` that involve `sat`, merged when closer than MERGE_GAP_STEPS. */
  function laneMarks(ep, type, sat) {
    const out = [];
    for (const run of ep.payload.runs[String(type)] || []) {
      if (!run.sats.includes(sat)) continue;
      const last = out[out.length - 1];
      if (last && run.start - last.end <= MERGE_GAP_STEPS) { last.end = run.end; last.count += run.count; } else out.push({ start: run.start, end: run.end, count: run.count });
    }
    return out.map((m) => ({ ...m, label: `S${sat}${m.count > 1 ? ` ×${m.count}` : ""}` }));
  }

  function collisionMarks(ep) {
    return (ep.payload.runs["3"] || []).map((run) => ({ start: run.start, end: run.end, count: run.count, sats: run.sats, label: `S${run.sats[0]}×S${run.sats[1]}` }));
  }

  function runningCounts(ep, step) {
    const r = ep.payload.running, s = ep.payload.summary, peak = Math.max(0, ...(ep.payload.density["2"] || [0]));
    return {
      maneuver: `${r.maneuvers[step]} / ${s.maneuvers}`,
      shield: r.shield === null ? `${DASH} · shield off` : `${r.shield[step]} / ${s.shield} · peak ${peak}`,
      collision: `${r.collisions[step]} / ${s.collisions}`,
    };
  }

  /* ----------------------------------------------------- summary strip, live */
  const STRIP_FORMAT = { delta_v_mps: (v) => fixed(v, 2), maneuvers: (v) => fixed(v, 0), shield: (v) => fixed(v, 0), collisions: (v) => fixed(v, 0), mean_deviation: (v) => fixed(v, 3) };

  function stripRunning(ep, step) {
    const running = ep.payload.running, out = {};
    for (const [metric, format] of Object.entries(STRIP_FORMAT)) {
      const series = running[metric];
      out[metric] = series === null ? DASH : `T+${minutesAt(ep, step)}: ${format(series[step] === null ? NaN : series[step])}`;
    }
    return out;
  }

  OG.models = { capabilityOf, fuelBars, inspector, inspectorSeries, countsLanes, laneMarks, collisionMarks, runningCounts, stripRunning, MERGE_GAP_STEPS, DASH };
})(globalThis);
