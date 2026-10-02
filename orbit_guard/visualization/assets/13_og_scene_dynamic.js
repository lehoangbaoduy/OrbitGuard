/* OrbitGuard replay stage - everything that changes frame to frame: satellites, trails, conjunction lines, status pins,
   label plates, burn plumes, shield rings. Pure functions of (episode, view) returning Plotly traces. The trace list
   keeps the same length and order every call so a gesture can update traces in place (see the camera module). */
(function (root) {
  "use strict";
  const OG = root.OG;
  const MONO = "Martian Mono, ui-monospace, monospace";
  const STATUS_COLLIDED = 3;
  const EVENT_MANEUVER = 1, EVENT_SHIELD = 2, EVENT_COLLISION = 3;
  const PAIR_INSIDE = 1, PAIR_HIGH_RISK = 2; // pair statuses (derived/status.py)
  const SAT_INSIDE = 1, SAT_HIGH_RISK = 2; // satellite statuses, same codes
  // Visual scale of the satellite models (drawn far larger than any real body) and effect geometry, in km: display, not data.
  const MODEL_SCALE = 1.6;
  const PIN_HEIGHT_KM = 300, LABEL_LIFT_NOMINAL_KM = 60, MASS_ANTENNA_KM = 22;
  const PLUME_START_KM = 34, PLUME_LENGTH_KM = 300, RING_RADIUS_KM = 150, RING_SETTLE_FROM = 0.7, RING_POINTS = 24;
  const TRAIL_STEPS = 10, MERGE_KM = 1100, MAX_PER_PLATE = 4;
  const MODEL_COLORS = { bus: "#c9a14a", mast: "#e3e7ef", yoke: "#9aa0b0", wing: "#5d8ee6", hinge: "#c9d6f2" };
  const EMPTY3 = (name) => ({ type: "scatter3d", mode: "lines", x: [], y: [], z: [], hoverinfo: "skip", name });
  const AX = (range) => ({ visible: false, range: [-range, range], showspikes: false, showbackground: false });

  const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const unit = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  const retired = (ep, t, i) => ep.satStatus[t * ep.N + i] === STATUS_COLLIDED;
  const fmt = (v, digits) => (Number.isFinite(v) ? v.toFixed(digits) : "—");

  /* Local frame of a satellite: r radial, v along-track (velocity without its radial part), n = r x v. The last frame
     uses the final recorded step's motion. */
  function basis(ep, t, i) {
    const p = OG.posAt(ep, t, i), step = t >= ep.T ? ep.T - 1 : t;
    const r = unit(p);
    const move = sub(OG.posAt(ep, step + 1, i), OG.posAt(ep, step, i));
    const along = dot(move, r);
    const v = unit([move[0] - along * r[0], move[1] - along * r[1], move[2] - along * r[2]]);
    return { r, v, n: cross(r, v) };
  }

  function interpolated(ep, t, f) {
    const next = Math.min(ep.T, t + 1), out = [];
    for (let i = 0; i < ep.N; i += 1) {
      const a = OG.posAt(ep, t, i), b = OG.posAt(ep, next, i);
      out.push([a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]);
    }
    return out;
  }

  const at = (p, B, dv, dn, dr) => [0, 1, 2].map((k) => p[k] + B.v[k] * dv + B.n[k] * dn + B.r[k] * dr);

  /* Burn plume along the exhaust direction of the APPROVED intent, taken from the catalog (never a local table). */
  function plume(ep, t, i, grow, pos) {
    if (t < 1 || ep.event[t * ep.N + i] !== EVENT_MANEUVER) return null;
    const spec = ep.catalog[ep.approved[(t - 1) * ep.N + i]];
    if (!spec) return null;
    const B = basis(ep, t, i), p = pos || OG.posAt(ep, t, i), [er, et, en] = spec.exhaust_rtn;
    const length = PLUME_START_KM + PLUME_LENGTH_KM * grow;
    return { from: at(p, B, et * PLUME_START_KM, en * PLUME_START_KM, er * PLUME_START_KM), to: at(p, B, et * length, en * length, er * length) };
  }

  /* Shield ring around the satellite, in its (along-track, normal) plane, perpendicular to the radius; `settle` runs 0 (wide) to 1 (settled). */
  function shieldRing(ep, t, i, settle, pos) {
    const B = basis(ep, t, i), p = pos || OG.posAt(ep, t, i);
    const radius = RING_RADIUS_KM * (1 + RING_SETTLE_FROM * (1 - OG.easeOut(settle)));
    const ring = [];
    for (let k = 0; k <= RING_POINTS; k += 1) {
      const theta = ((k % RING_POINTS) / RING_POINTS) * Math.PI * 2;
      ring.push(at(p, B, Math.cos(theta) * radius, Math.sin(theta) * radius, 0));
    }
    return ring;
  }

  function modelMesh(ep, ctx) {
    const m = { x: [], y: [], z: [], i: [], j: [], k: [], c: [] };
    const box = (p, B, cv, cn, cr, hv, hn, hr, color) => {
      const base = m.x.length;
      for (const sa of [-1, 1]) for (const sb of [-1, 1]) for (const sd of [-1, 1]) {
        const q = at(p, B, cv + sa * hv, cn + sb * hn, cr + sd * hr);
        m.x.push(q[0]); m.y.push(q[1]); m.z.push(q[2]); m.c.push(color);
      }
      const face = (a, b, c, d) => { m.i.push(base + a, base + a); m.j.push(base + b, base + c); m.k.push(base + c, base + d); };
      face(0, 1, 3, 2); face(4, 6, 7, 5); face(0, 4, 5, 1); face(2, 3, 7, 6); face(0, 2, 6, 4); face(1, 5, 7, 3);
    };
    const K = MODEL_SCALE;
    for (let s = 0; s < ep.N; s += 1) {
      if (retired(ep, ctx.step, s)) continue;
      const status = ep.satStatus[ctx.step * ep.N + s], p = ctx.pts[s], B = ctx.bases[s];
      box(p, B, 0, 0, 0, 30 * K, 18 * K, 18 * K, status ? ep.vocab.sat_status[status].color : MODEL_COLORS.bus);
      box(p, B, 0, 0, MASS_ANTENNA_KM * K, 6 * K, 6 * K, 5 * K, MODEL_COLORS.mast);
      for (const side of [-1, 1]) {
        box(p, B, 0, side * 27 * K, 0, 4 * K, 9 * K, 3 * K, MODEL_COLORS.yoke);
        box(p, B, 0, side * 100 * K, 0, 26 * K, 68 * K, 2.5 * K, MODEL_COLORS.wing);
        box(p, B, 0, side * 100 * K, 0, 27 * K, 2 * K, 3 * K, MODEL_COLORS.hinge);
      }
    }
    return { type: "mesh3d", name: "satellites", x: m.x, y: m.y, z: m.z, i: m.i, j: m.j, k: m.k, vertexcolor: m.c, flatshading: true, hoverinfo: "skip",
      lighting: { ambient: 0.5, diffuse: 0.9, specular: 0.55, roughness: 0.35, fresnel: 0.15 }, lightposition: { x: 83000, y: 45000, z: 33000 } };
  }

  function hoverText(ep, ctx, i) {
    const t = ctx.step, status = ep.vocab.sat_status[ep.satStatus[t * ep.N + i]], event = ep.vocab.event[ep.event[t * ep.N + i]];
    const head = `S${i} · ${status.label}${event.label ? ` · ${event.label}` : ""}`;
    return `${head}<br>Δv left ${fmt(ep.remaining[t * ep.N + i], 2)} m/s<br>mission dev ${fmt(ep.deviation[t * ep.N + i], 3)}`;
  }

  function trailsTrace(ep, ctx) {
    const x = [], y = [], z = [], color = [];
    for (let i = 0; i < ep.N; i += 1) {
      const status = ep.satStatus[ctx.step * ep.N + i];
      if (retired(ep, ctx.step, i) || (!status && !ep.event[ctx.step * ep.N + i] && i !== ctx.view.selected)) continue;
      const c = status ? ep.vocab.sat_status[status].color : OG.ICE;
      for (let t = Math.max(0, ctx.step - TRAIL_STEPS); t <= ctx.step; t += 1) {
        const p = OG.posAt(ep, t, i); x.push(p[0]); y.push(p[1]); z.push(p[2]); color.push(c);
      }
      x.push(ctx.pts[i][0]); y.push(ctx.pts[i][1]); z.push(ctx.pts[i][2]); color.push(c);
      x.push(null); y.push(null); z.push(null); color.push(c);
    }
    return { type: "scatter3d", mode: "lines", name: "trails", x, y, z, line: { color, width: 3 }, opacity: 0.55, hoverinfo: "skip" };
  }

  function conjunctionLines(ep, ctx, status) {
    const x = [], y = [], z = [], style = ep.vocab.pair_status[status];
    for (let k = 0; k < ep.P; k += 1) {
      if (ep.pairStatus[ctx.step * ep.P + k] !== status) continue;
      const a = ctx.pts[ep.pairs[2 * k]], b = ctx.pts[ep.pairs[2 * k + 1]];
      x.push(a[0], b[0], null); y.push(a[1], b[1], null); z.push(a[2], b[2], null);
    }
    return { type: "scatter3d", mode: "lines", name: `conjunction-${style.name}`, x, y, z, line: { color: style.color, width: style.width, dash: style.dash },
      opacity: 0.25 + 0.75 * OG.easeOut(ctx.view.grow.lines), hoverinfo: "skip" };
  }

  function navLights(ep, ctx) {
    const idx = [];
    for (let i = 0; i < ep.N; i += 1) if (!retired(ep, ctx.step, i)) idx.push(i);
    return { type: "scatter3d", mode: "markers", name: "satellite-picks", customdata: idx, hovertext: idx.map((i) => hoverText(ep, ctx, i)), hoverinfo: "text",
      marker: { size: 4, color: "#f6ead0", line: { width: 0 } }, x: idx.map((i) => ctx.pts[i][0]), y: idx.map((i) => ctx.pts[i][1]), z: idx.map((i) => ctx.pts[i][2]) };
  }

  function pinTraces(ep, ctx) {
    const stems = { x: [], y: [], z: [] }, color = [];
    for (let i = 0; i < ep.N; i += 1) {
      const status = ep.satStatus[ctx.step * ep.N + i];
      if (!status) continue;
      const a = at(ctx.pts[i], ctx.bases[i], 0, 0, MASS_ANTENNA_KM), b = at(ctx.pts[i], ctx.bases[i], 0, 0, PIN_HEIGHT_KM), c = ep.vocab.sat_status[status].color;
      stems.x.push(a[0], b[0], null); stems.y.push(a[1], b[1], null); stems.z.push(a[2], b[2], null); color.push(c, c, c);
    }
    const lines = { type: "scatter3d", mode: "lines", name: "pin-stems", ...stems, line: { color: color.length ? color : OG.ICE, width: 1.5 }, opacity: 0.8, hoverinfo: "skip" };
    const marks = [1, 2, 3].map((status) => {
      const idx = [];
      for (let i = 0; i < ep.N; i += 1) if (ep.satStatus[ctx.step * ep.N + i] === status) idx.push(i);
      const style = ep.vocab.sat_status[status], q = idx.map((i) => at(ctx.pts[i], ctx.bases[i], 0, 0, PIN_HEIGHT_KM));
      return { type: "scatter3d", mode: "markers", name: `pins-${style.name}`, x: q.map((p) => p[0]), y: q.map((p) => p[1]), z: q.map((p) => p[2]), customdata: idx,
        hovertext: idx.map((i) => hoverText(ep, ctx, i)), hoverinfo: "text",
        marker: { symbol: style.marker, size: status === SAT_HIGH_RISK ? 10 : 9, color: style.color, line: { color: style.color, width: status === SAT_INSIDE ? 2 : 0 } } };
    });
    return [lines, ...marks];
  }

  function labelItems(ep, ctx) {
    const items = [], { selected, labelAll } = ctx.view;
    for (let i = 0; i < ep.N; i += 1) {
      const status = ep.satStatus[ctx.step * ep.N + i], isSel = i === selected;
      if (!(status || isSel || labelAll)) continue;
      const style = ep.vocab.sat_status[status], event = ep.vocab.event[ep.event[ctx.step * ep.N + i]];
      const mark = event.label && ep.event[ctx.step * ep.N + i] !== EVENT_COLLISION ? ` <span style="color:${event.color}">${event.glyph}</span>` : "";
      const c = status ? style.color : isSel ? OG.ICE : ep.vocab.sat_status[0].color;
      items.push({ i, st: status, isSel, c, q: at(ctx.pts[i], ctx.bases[i], 0, 0, status ? PIN_HEIGHT_KM : LABEL_LIFT_NOMINAL_KM),
        html: `<span style="color:${isSel ? OG.ICE : c}"><b>S${i}</b>${status ? ` ${style.glyph}` : ""}</span>${mark}` });
    }
    return items;
  }

  function effectTraces(ep, ctx) {
    const px = [], py = [], pz = [], rx = [], ry = [], rz = [], halo = [];
    for (let i = 0; i < ep.N; i += 1) {
      if (retired(ep, ctx.step, i) && ep.event[ctx.step * ep.N + i] !== EVENT_COLLISION) continue;
      const burn = plume(ep, ctx.step, i, ctx.view.grow.plume, ctx.pts[i]);
      if (burn) { px.push(burn.from[0], burn.to[0], null); py.push(burn.from[1], burn.to[1], null); pz.push(burn.from[2], burn.to[2], null); }
      if (ep.event[ctx.step * ep.N + i] === EVENT_SHIELD) {
        for (const q of shieldRing(ep, ctx.step, i, ctx.view.grow.ring, ctx.pts[i])) { rx.push(q[0]); ry.push(q[1]); rz.push(q[2]); }
        rx.push(null); ry.push(null); rz.push(null);
      }
      if (ep.event[ctx.step * ep.N + i] === EVENT_COLLISION) halo.push(i);
    }
    const ev = ep.vocab.event;
    return [
      { type: "scatter3d", mode: "lines", name: "plumes", x: px, y: py, z: pz, line: { color: ev[EVENT_MANEUVER].color, width: 7 }, opacity: 0.9, hoverinfo: "skip" },
      { type: "scatter3d", mode: "lines", name: "shield-rings", x: rx, y: ry, z: rz, line: { color: ev[EVENT_SHIELD].color, width: 2.5 }, opacity: 0.85, hoverinfo: "skip" },
      { type: "scatter3d", mode: "markers", name: "collision-halo", hoverinfo: "skip", x: halo.map((i) => ctx.pts[i][0]), y: halo.map((i) => ctx.pts[i][1]), z: halo.map((i) => ctx.pts[i][2]),
        marker: { symbol: ev[EVENT_COLLISION].marker, size: 24, color: ev[EVENT_COLLISION].color, line: { color: ev[EVENT_COLLISION].color, width: 2 } } },
    ];
  }

  /* All traces and label plates for one view: {step, frac, selected, orbits, labelAll, grow:{plume, ring, lines},
     static?, selectedRing?}. */
  function sceneTraces(ep, view) {
    const st = view.static || OG.scene.staticTraces(ep);
    const step = Math.min(ep.T, Math.max(0, view.step)), sel = Math.min(ep.N - 1, Math.max(0, view.selected));
    const pts = interpolated(ep, step, view.frac || 0), bases = pts.map((_, i) => basis(ep, step, i));
    const ctx = { step, view, pts, bases };
    const selectedRing = retired(ep, step, sel) ? EMPTY3("selected-orbit") : view.selectedRing || OG.scene.satelliteRing(ep, sel);
    const selMark = { type: "scatter3d", mode: "markers", name: "selection", x: [pts[sel][0]], y: [pts[sel][1]], z: [pts[sel][2]], hoverinfo: "skip",
      marker: { symbol: "circle-open", size: 26, color: OG.ICE, line: { color: OG.ICE, width: 1.5 } } };
    const data = [st.earth, st.atmosphere, st.graticule, view.orbits ? st.rings : EMPTY3("orbits"), selectedRing, trailsTrace(ep, ctx),
      conjunctionLines(ep, ctx, PAIR_HIGH_RISK), conjunctionLines(ep, ctx, PAIR_INSIDE), modelMesh(ep, ctx), navLights(ep, ctx),
      ...pinTraces(ep, ctx), ...effectTraces(ep, ctx), selMark];
    const plates = OG.mergeLabels(labelItems(ep, ctx), MERGE_KM, MAX_PER_PLATE);
    return { data, annotations: plates.map((g, n) => OG.annotationFor(g, n, MONO)) };
  }

  const DEFAULT_CAMERA = { eye: { x: 1.3, y: 1.05, z: 0.55 }, center: { x: 0, y: 0, z: -0.02 }, up: { x: 0, y: 0, z: 1 } };

  /* Plotly layout for the stage; the live camera (if any) is carried in so a redraw never resets the view. */
  function layout(ep, opts) {
    const range = OG.axisRange(ep), camera = opts.camera || DEFAULT_CAMERA;
    return {
      paper_bgcolor: "rgba(0,0,0,0)", margin: { l: 0, r: 0, t: 0, b: 0 }, showlegend: false, uirevision: `og${opts.revision || 0}`, hovermode: "closest",
      hoverlabel: { bgcolor: "#232532", bordercolor: "#595d6c", font: { family: MONO, size: 11, color: "#e9e9ed" } },
      scene: { domain: { x: [0, opts.domainRight || 1], y: [0, opts.domainTop || 0.9] }, bgcolor: "rgba(0,0,0,0)", aspectmode: "cube", dragmode: "turntable",
        xaxis: AX(range), yaxis: AX(range), zaxis: AX(range), camera, annotations: opts.annotations || [] },
    };
  }

  Object.assign(OG.scene, { basis, interpolated, plume, shieldRing, sceneTraces, layout, DEFAULT_CAMERA });
})(globalThis);
