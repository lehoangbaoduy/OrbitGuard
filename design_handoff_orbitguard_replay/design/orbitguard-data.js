/* ============================================================================================
   SYNTHETIC DEMO DATA — FOR THE DESIGN PROTOTYPE ONLY. DO NOT PORT, COPY OR SHIP.
   Every episode, number and trajectory below is fabricated to demonstrate the UI.
   The production app must read recorded EpisodeTrace files (see PROMPT_FOR_CLAUDE_CODE.md).
   ============================================================================================ */
/* OrbitGuard replay — synthetic pre-recorded episodes, BEFORE vs AFTER training (shield ON in both).
   Data source is abstract: the UI asks a ReplaySource for "the frame at step t". Today a source wraps a
   saved trace (EpisodeTrace-shaped arrays); a live feed can implement the same interface later.

   interface ReplaySource {
     meta: { scenario, tab, model, condition, seed, numSteps, numAgents }
     frameAt(t) -> { step, time_s, satellites:[{index, position_km, alive, status, event, remaining_dv_mps,
                      mission_deviation, proposed, approved}], pairs:[{i, j, d_min_m, d_safe_m, risk, status}] }
     summary            -> episode totals { total_dv, maneuvers, shield, collisions, mean_dev }
     density[type][t]   -> satellites with that event at step t (1 maneuver, 2 shield, 3 collision)
   }                                                                                                    */
(function () {
  'use strict';
  const RE = 6378.137, MU = 398600.4418, T = 360, N = 15, DT = 60, D2R = Math.PI / 180, HIGH = 0.5, BASE_DSAFE = 1141;
  const ACTIONS = [
    { id: 0, label: 'NO MANEUVER', short: 'COAST', dv: 0 },
    { id: 1, label: 'RAISE ORBIT', short: 'RAISE', dv: 0.10 },
    { id: 2, label: 'LOWER ORBIT', short: 'LOWER', dv: 0.10 },
    { id: 3, label: 'PHASE FORWARD', short: 'PH FWD', dv: 0.05 },
    { id: 4, label: 'PHASE BACKWARD', short: 'PH BACK', dv: 0.05 },
    { id: 5, label: 'RADIAL ADJUST', short: 'RADIAL', dv: 0.10 },
  ];
  const STATUS = ['nominal', 'inside_d_safe', 'high_risk', 'collided'], EVENT = ['none', 'maneuver', 'shield_intervention', 'collision'];
  const PAIRS = []; for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) PAIRS.push([i, j]);
  const P = PAIRS.length;
  const pk = (i, j) => { if (i > j) { const x = i; i = j; j = x; } return i * N - (i * (i + 1)) / 2 + (j - i - 1); };
  function rng(seed) { let a = seed >>> 0; return function () { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]); return [a[0] / l, a[1] / l, a[2] / l]; };
  function plane(inc, raan) { const i = inc * D2R, O = raan * D2R, p = [Math.cos(O), Math.sin(O), 0], h = [Math.sin(i) * Math.sin(O), -Math.sin(i) * Math.cos(O), Math.cos(i)]; return { p, q: cross(h, p), h }; }
  function interpLog(kf, t) {
    for (let n = 0; n < kf.length - 1; n++) { const [s0, d0] = kf[n], [s1, d1] = kf[n + 1];
      if (t >= s0 && t <= s1) { const u = (t - s0) / (s1 - s0), a = Math.log(Math.max(d0, 1)), b = Math.log(Math.max(d1, 1)); return Math.exp(a + (b - a) * u); } }
    return null;
  }
  const shuffle = (R, a) => { for (let k = a.length - 1; k > 0; k--) { const j = Math.floor(R() * (k + 1)); const x = a[k]; a[k] = a[j]; a[j] = x; } return a; };

  const SCENARIOS = [
    { id: 'normal', label: 'Normal', sub: 'seed 111080', condition: 'normal', seed: 111080, startStep: 132, focusSat: 0,
      events: [
        { a: 0, b: 8, te: 138, off: 0.25, ds: 1978, kf: [[112, 9000], [118, 2400], [124, 950], [128, 430], [132, 272], [134, 390], [136, 1250], [138, 2600], [143, 9000]] },
        { a: 9, b: 14, te: 210, off: 0.3, ds: 1850, kf: [[186, 9000], [194, 2100], [200, 720], [204, 310], [206, 540], [208, 1500], [210, 3100], [215, 9000]] }],
      targets: { before: { dv: 30.0, man: 405, sh: 3818, dev: 0.68 }, after: { dv: 3.15, man: 46, sh: 12, dev: 0.18 } },
      targetsOff: { before: { dv: 30.0, man: 405, dev: 0.70 }, after: { dv: 3.0, man: 44, dev: 0.24 } },
      ref: { none: [1, 0.0], before: [0, 30.0], after: [0, 3.0] } },
    { id: 'simul', label: '3 simultaneous conjunctions', sub: 'seed 10', condition: 'simultaneous_conjunctions', seed: 10, startStep: 250, focusSat: 9,
      events: [
        { a: 0, b: 1, te: 179, off: 0.12, ds: 1700, kf: [[152, 9000], [160, 2600], [168, 900], [173, 420], [177, 210], [179, 160], [182, 900], [186, 9000]] },
        { a: 4, b: 14, te: 182, off: 0.2, ds: 1773, kf: [[156, 9000], [166, 2700], [174, 980], [179, 376], [182, 300], [185, 1400], [190, 9000]] },
        { a: 8, b: 9, te: 186, off: 0.5, ds: 1500, kf: [[162, 9000], [172, 1900], [180, 860], [184, 640], [186, 700], [190, 1600], [195, 9000]] },
        { a: 3, b: 11, te: 252, off: 0.6, ds: 1400, kf: [[232, 9000], [240, 2000], [246, 1050], [250, 820], [252, 900], [256, 2200], [260, 9000]] }],
      targets: { before: { dv: 30.0, man: 442, sh: 3728, dev: 0.716 }, after: { dv: 10.8, man: 161, sh: 57, dev: 0.227 } },
      targetsOff: { before: { dv: 30.0, man: 442, dev: 0.716 }, after: { dv: 12.2, man: 159, dev: 0.299 } },
      offEvents: [{ a: 0, b: 1, te: 250, off: 0.9, ds: 1146, kf: [[226, 9000], [236, 2400], [244, 1200], [248, 960], [250, 931], [253, 1050], [258, 2600], [264, 9000]] }],
      ref: { none: [5, 0.0], before: [0, 30.0], after: [0, 10.9] } },
    { id: 'capability', label: 'Capability reduction', sub: 'seed 10', condition: 'capability_reduction', seed: 10, startStep: 272, focusSat: 8,
      offline: [9, 12], limited: [0, 5],
      events: [
        { a: 0, b: 2, te: 156, off: 0.5, ds: 1500, kf: [[132, 9000], [140, 2300], [146, 720], [150, 480], [153, 880], [156, 1650], [161, 9000]] },
        { a: 8, b: 9, te: 280, off: 0.04, ds: 2126, kf: [[244, 9000], [252, 2500], [258, 920], [264, 300], [268, 120], [272, 35], [276, 60], [280, 700], [284, 1650], [290, 9000]] }],
      targets: { before: { dv: 23.0, man: 355, sh: 3872, dev: 0.66 }, after: { dv: 1.6, man: 21, sh: 146, dev: 0.17 } },
      targetsOff: { before: { dv: 23.0, man: 355, dev: 0.68 }, after: { dv: 0.8, man: 12, dev: 0.21 } },
      collideOff: { a: 8, b: 9, te: 280 },
      ref: { none: [1, 0.0], before: [1, 23.0], after: [1, 0.8] } },
  ];

  function genActions(R, sc, tab, enabled, budget, shield) {
    const tg = shield === 'off' ? Object.assign({ sh: 0 }, sc.targetsOff[tab]) : sc.targets[tab], acts = [], used = new Set(), key = (i, s) => s * N + i;
    const M = tg.man, A = Math.round(tg.dv / 0.05) - M; // number of 0.10 m/s burns
    const big = [1, 2, 5], small = [3, 4], pick = (arr) => arr[Math.floor(R() * arr.length)];
    const burnEnd = new Array(N).fill(1);
    if (tab === 'before') {
      // untrained: burns its whole budget early, then keeps proposing burns the shield blocks
      const U = budget.map((b, i) => (enabled[i] ? Math.round(b / 0.05) : 0)), SU = U.reduce((x, y) => x + y, 0);
      const a = U.map((u) => Math.min(Math.floor(u / 2), Math.floor((A * u) / SU)));
      let rem = A - a.reduce((x, y) => x + y, 0);
      for (let k = 0; rem > 0 && k < 5000; k++) { const i = k % N; if (U[i] && 2 * (a[i] + 1) <= U[i]) { a[i]++; rem--; } }
      for (let i = 0; i < N; i++) {
        if (!U[i]) continue;
        const amounts = shuffle(R, [...Array(a[i]).fill(0.1), ...Array(U[i] - 2 * a[i]).fill(0.05)]);
        let s = 1 + Math.floor(R() * 3);
        for (const amt of amounts) { const act = amt > 0.07 ? pick(big) : pick(small); acts.push([i, s, act, act]); used.add(key(i, s)); s += 1 + (R() < 0.15 ? 1 + Math.floor(R() * 3) : 0); }
        burnEnd[i] = s;
      }
    } else {
      // trained: few, purposeful burns — clustered before each conjunction, plus light station-keeping
      const caps = budget.map((b, i) => (enabled[i] ? Math.round(b / 0.05) : 0));
      const slots = [];
      for (const ev of sc.events) for (const s of [ev.a, ev.b]) if (enabled[s]) for (let k = 0; k < 4; k++) slots.push([s, ev.te - 16 + Math.floor(R() * 12)]);
      while (slots.length < M) { const i = Math.floor(R() * N); if (enabled[i]) slots.push([i, 1 + Math.floor(R() * 359)]); }
      slots.length = M;
      const amounts = shuffle(R, [...Array(A).fill(0.1), ...Array(M - A).fill(0.05)]);
      for (let n = 0; n < M; n++) {
        let [i, s] = slots[n]; const u = amounts[n] > 0.07 ? 2 : 1;
        for (let tries = 0; (caps[i] < u || used.has(key(i, s))) && tries < 400; tries++) {
          if (caps[i] < u) { const ok = []; for (let x = 0; x < N; x++) if (caps[x] >= u) ok.push(x); i = ok[Math.floor(R() * ok.length)]; }
          s = Math.min(360, Math.max(1, s + Math.floor(R() * 9) - 4));
        }
        caps[i] -= u; used.add(key(i, s)); const act = u === 2 ? pick(big) : pick(small); acts.push([i, s, act, act]);
      }
    }
    if (shield === 'off') {
      // nothing overrides the policy: after its budget is gone the untrained policy keeps proposing burns that do nothing
      if (tab === 'before') for (let i = 0; i < N; i++) for (let s = burnEnd[i]; s <= 360; s++) if (!used.has(key(i, s)) && R() < 0.3) { const a = 1 + Math.floor(R() * 5); acts.push([i, s, a, a]); }
      return acts;
    }
    // shield interventions: proposed burn blocked (approved = COAST), counted per satellite-step
    const cand = [];
    for (let i = 0; i < N; i++) for (let s = enabled[i] ? burnEnd[i] : 1; s <= 360; s++) {
      if (used.has(key(i, s))) continue;
      let w = 1;
      if (tab === 'after') {
        w = 0.25;
        for (const ev of sc.events) if ((ev.a === i || ev.b === i) && Math.abs(s - ev.te) < 18) w = 8;
        if (!enabled[i]) w = Math.max(w, 2.5);
      }
      cand.push([i, s, Math.pow(R(), 1 / w)]);
    }
    cand.sort((x, y) => y[2] - x[2]);
    for (let n = 0; n < Math.min(tg.sh, cand.length); n++) { const [i, s] = cand[n]; acts.push([i, s, 1 + Math.floor(R() * 5), 0]); }
    return acts;
  }

  function build(sc, tab, shield) {
    const off = shield === 'off', evs = off ? sc.events.concat(sc.offEvents || []) : sc.events, col = off ? sc.collideOff : null;
    const R = rng(sc.seed), incs = [53, 97.6, 70, 43, 86], orb = [];
    for (let i = 0; i < N; i++) orb.push({ r: RE + 505 + i * 6.5 + R() * 1.2, inc: incs[i % 5] + (R() - 0.5) * 8, raan: R() * 360, th0: R() * Math.PI * 2 });
    for (const ev of evs) {
      const a = orb[ev.a], b = orb[ev.b];
      b.raan = (a.raan + 50 + R() * 40) % 360; b.inc = a.inc + (R() - 0.5) * 24; b.r = a.r + ev.off;
      const pa = plane(a.inc, a.raan), pb = plane(b.inc, b.raan), n = norm(cross(pa.h, pb.h));
      for (const [o, pl] of [[a, pa], [b, pb]]) { const w = Math.sqrt(MU / o.r ** 3); o.th0 = Math.atan2(dot(n, pl.q), dot(n, pl.p)) - w * ev.te * DT; }
    }
    const RA = rng(sc.seed * 7 + (tab === 'before' ? 1 : 2) + (off ? 10 : 0));
    const death = new Array(N).fill(Infinity); // shield ON: none; shield OFF: the scenario may record one
    if (col) { death[col.a] = col.te; death[col.b] = col.te; }
    const pos = new Float32Array((T + 1) * N * 3), rings = [];
    for (let i = 0; i < N; i++) {
      const o = orb[i], pl = plane(o.inc, o.raan), w = Math.sqrt(MU / o.r ** 3);
      for (let t = 0; t <= T; t++) { const th = o.th0 + w * Math.min(t, death[i]) * DT, c = Math.cos(th), s = Math.sin(th), k = (t * N + i) * 3; for (let d = 0; d < 3; d++) pos[k + d] = o.r * (c * pl.p[d] + s * pl.q[d]); }
      const ring = []; for (let n = 0; n <= 96; n++) { const th = (n / 96) * Math.PI * 2, c = Math.cos(th), s = Math.sin(th); ring.push([0, 1, 2].map((d) => o.r * (c * pl.p[d] + s * pl.q[d]))); }
      rings.push(ring);
    }
    const alive = new Uint8Array((T + 1) * N); for (let t = 0; t <= T; t++) for (let i = 0; i < N; i++) alive[t * N + i] = t < death[i] ? 1 : 0;
    const enabled = new Array(N).fill(true); (sc.offline || []).forEach((i) => (enabled[i] = false));
    const budget = new Array(N).fill(2.0); (sc.limited || []).forEach((i) => (budget[i] = 0.5));
    const proposed = new Int8Array(T * N), approved = new Int8Array(T * N), applied = new Float32Array(T * N), intervened = new Uint8Array(T * N), rem = new Float32Array((T + 1) * N);
    const act = {}; for (const [i, s, pr, ap] of genActions(RA, sc, tab, enabled, budget, shield)) act[(s - 1) * N + i] = [pr, ap];
    const left = budget.slice(), burns = Array.from({ length: N }, () => []);
    for (let t = 0; t < T; t++) for (let i = 0; i < N; i++) {
      rem[t * N + i] = left[i]; const idx = t * N + i, a = act[idx];
      if (a && alive[t * N + i]) { proposed[idx] = a[0]; approved[idx] = a[1]; intervened[idx] = a[0] !== a[1] ? 1 : 0;
        const dv = ACTIONS[a[1]].dv; if (enabled[i] && dv > 0 && left[i] >= dv - 1e-9) { applied[idx] = dv; left[i] = Math.round((left[i] - dv) * 100) / 100; burns[i].push(t + 1); } }
    }
    for (let i = 0; i < N; i++) rem[T * N + i] = left[i];

    const tg = off ? sc.targetsOff[tab] : sc.targets[tab], dev = new Float32Array((T + 1) * N);
    for (let i = 0; i < N; i++) {
      const lvl = tg.dev * (0.7 + RA() * 0.6), ph1 = RA() * 6.28, ph2 = RA() * 6.28, f1 = 0.012 + RA() * 0.01, f2 = 0.07 + RA() * 0.05; let walk = 0;
      for (let t = 0; t <= T; t++) {
        if (t >= death[i]) { dev[t * N + i] = dev[(t - 1) * N + i]; continue; }
        walk = (walk + (RA() - 0.5) * 0.03 * tg.dev * 4) * 0.97;
        const ramp = Math.min(1, 0.25 + t / 60);
        dev[t * N + i] = Math.min(1.4, Math.max(0.005, ramp * (lvl + 0.22 * lvl * Math.sin(f1 * t + ph1)) + 0.05 * lvl * Math.sin(f2 * t + ph2) + walk));
      }
    }

    const dist = new Float32Array((T + 1) * P);
    for (let t = 0; t <= T; t++) for (let k = 0; k < P; k++) { const [i, j] = PAIRS[k], a = (t * N + i) * 3, b = (t * N + j) * 3; dist[t * P + k] = Math.hypot(pos[a] - pos[b], pos[a + 1] - pos[b + 1], pos[a + 2] - pos[b + 2]) * 1000; }
    const dmin = new Float32Array((T + 1) * P), dsafe = new Float32Array((T + 1) * P), risk = new Float32Array((T + 1) * P), masked = new Uint8Array((T + 1) * P), collStep = new Int16Array(P).fill(-1);
    const evByK = {}; for (const ev of evs) evByK[pk(ev.a, ev.b)] = ev;
    if (col) collStep[pk(col.a, col.b)] = col.te;
    const scale = tab === 'after' ? (off ? 2.2 : 3.2) : 1; // trained policy resolves conjunctions earlier → shallower dips
    for (let k = 0; k < P; k++) {
      const ev = evByK[k]; let tp = 0, mv = Infinity; if (ev) for (const [s, d] of ev.kf) if (d < mv) { mv = d; tp = s; }
      for (let t = 0; t <= T; t++) {
        let m = Infinity; for (let u = t; u <= Math.min(T, t + 12); u++) m = Math.min(m, dist[u * P + k]);
        let ds = BASE_DSAFE;
        if (ev) { const v = interpLog(ev.kf, t); m = v !== null ? Math.min(Math.max(v, 9000), v * scale) * (1 + (RA() - 0.5) * 0.04) : Math.max(m, 9000 + RA() * 4000); ds = BASE_DSAFE + (ev.ds - BASE_DSAFE) * Math.exp(-(((t - tp) / 9) ** 2)); }
        if (collStep[k] >= 0) { const tc = collStep[k]; if (t <= tc) { m = Math.min(m, Math.max(0, (tc - t) * 9)); } else m = 9000 + RA() * 40000; }
        const [pi, pj] = PAIRS[k], dead = !alive[t * N + pi] || !alive[t * N + pj];
        if (dead && collStep[k] < 0) { masked[t * P + k] = 1; m = 1e7; }
        dmin[t * P + k] = Math.round(m); dsafe[t * P + k] = Math.round(ds);
        risk[t * P + k] = masked[t * P + k] || (collStep[k] >= 0 && t >= collStep[k]) ? 0 : Math.max(0, Math.min(1, 1 - m / (1.1 * ds)));
      }
    }
    const pairStatus = new Uint8Array((T + 1) * P), satStatus = new Uint8Array((T + 1) * N), satEvent = new Uint8Array((T + 1) * N);
    for (let t = 0; t <= T; t++) {
      for (let k = 0; k < P; k++) { const x = t * P + k; const s = collStep[k] >= 0 && t >= collStep[k] ? 3 : masked[x] ? 0 : risk[x] > HIGH ? 2 : dmin[x] < dsafe[x] ? 1 : 0; pairStatus[x] = s; const [i, j] = PAIRS[k]; if (s > satStatus[t * N + i]) satStatus[t * N + i] = s; if (s > satStatus[t * N + j]) satStatus[t * N + j] = s; }
      if (t >= 1) for (let i = 0; i < N; i++) { const idx = (t - 1) * N + i; satEvent[t * N + i] = col && t === col.te && (i === col.a || i === col.b) ? 3 : intervened[idx] ? 2 : applied[idx] > 0 ? 1 : 0; }
    }
    const cD = new Float32Array((T + 1) * N).fill(NaN), cDs = new Float32Array((T + 1) * N).fill(NaN), cJ = new Int8Array((T + 1) * N).fill(-1);
    for (let t = 0; t <= T; t++) for (let k = 0; k < P; k++) { const x = t * P + k; if (masked[x] || pairStatus[x] === 3) continue;
      for (const [a, b] of [[PAIRS[k][0], PAIRS[k][1]], [PAIRS[k][1], PAIRS[k][0]]]) { const y = t * N + a; if (!alive[y]) continue; if (isNaN(cD[y]) || dmin[x] < cD[y]) { cD[y] = dmin[x]; cDs[y] = dsafe[x]; cJ[y] = b; } } }

    // running metrics + per-step density
    const Mx = { collisions: [], unsafe: [], dv: [], maneuvers: [], shield: [], meanDev: [] };
    const density = { 1: new Uint8Array(T + 1), 2: new Uint8Array(T + 1), 3: new Uint8Array(T + 1) };
    const prevIn = new Uint8Array(P); let unsafe = 0, dvT = 0, man = 0, sh = 0, devSum = 0;
    for (let t = 0; t <= T; t++) {
      if (t >= 1) for (let i = 0; i < N; i++) { const idx = (t - 1) * N + i; dvT += applied[idx]; if (intervened[idx]) sh++; else if (applied[idx] > 0) man++; const e = satEvent[t * N + i]; if (e) density[e][t]++; }
      let cc = 0; for (let k = 0; k < P; k++) if (collStep[k] >= 0 && collStep[k] <= t) cc++;
      for (let k = 0; k < P; k++) { const x = t * P + k, inside = !masked[x] && pairStatus[x] !== 3 && dmin[x] < dsafe[x]; if (inside && !prevIn[k]) unsafe++; prevIn[k] = inside ? 1 : 0; }
      let s = 0, na = 0; for (let i = 0; i < N; i++) if (alive[t * N + i]) { s += dev[t * N + i]; na++; } devSum += na ? s / na : 0;
      Mx.collisions.push(cc); Mx.unsafe.push(unsafe); Mx.dv.push(dvT); Mx.maneuvers.push(man); Mx.shield.push(sh); Mx.meanDev.push(devSum / (t + 1));
    }
    const runs = { 1: [], 2: [], 3: [] };
    for (const type of [1, 2]) for (let i = 0; i < N; i++) { let cur = null;
      for (let t = 1; t <= T; t++) if (satEvent[t * N + i] === type) { if (cur && t - cur.end <= 3) { cur.end = t; cur.count++; } else { cur = { type, sats: [i], start: t, end: t, count: 1 }; runs[type].push(cur); } } }
    for (let k = 0; k < P; k++) if (collStep[k] >= 0) runs[3].push({ type: 3, sats: PAIRS[k].slice(), start: collStep[k], end: collStep[k], count: 1 });
    for (const t of [1, 2]) runs[t].sort((a, b) => a.start - b.start);
    // key events for prev/next: conjunction entries, budget depletion, and (when sparse) event-run starts
    const key = new Set();
    for (let t = 1; t <= T; t++) for (let k = 0; k < P; k++) { const a = pairStatus[(t - 1) * P + k], b = pairStatus[t * P + k]; if (b > a) key.add(t); }
    for (let i = 0; i < N; i++) for (let t = 1; t <= T; t++) if (rem[t * N + i] === 0 && rem[(t - 1) * N + i] > 0) key.add(t);
    for (const type of [1, 2]) if (runs[type].length <= 80) runs[type].forEach((r) => key.add(r.start));
    const keySteps = Array.from(key).sort((a, b) => a - b);
    const stepEvents = []; for (let t = 0; t <= T; t++) { const l = []; for (let i = 0; i < N; i++) if (satEvent[t * N + i]) l.push({ type: satEvent[t * N + i], sat: i }); stepEvents.push(l); }
    const logs = Array.from({ length: N }, () => []);
    for (let t = 1; t <= T; t++) for (let i = 0; i < N; i++) { const idx = (t - 1) * N + i; if (proposed[idx] || approved[idx] || satEvent[t * N + i] === 3) logs[i].push({ step: t, pr: proposed[idx], ap: approved[idx], dv: applied[idx], over: !!intervened[idx], type: satEvent[t * N + i] }); }
    let maxShield = 0, maxShieldAt = 0; for (let t = 0; t <= T; t++) if (density[2][t] > maxShield) { maxShield = density[2][t]; maxShieldAt = t; }

    const E = {
      id: sc.id + ':' + tab + ':' + shield, scenario: sc.id, tab, shield, ref: sc.ref, label: sc.label, condition: sc.condition, seedLabel: sc.seed, startStep: sc.startStep, focusSat: sc.focusSat,
      model: (tab === 'before' ? 'untrained policy (fresh init)' : 'trained policy') + (off ? ' · NO shield' : ' + shield'),
      T, N, P, control_dt_s: DT, pair_index: PAIRS, positions_km: pos, rings, alive, remaining_dv_mps: rem, mission_deviation: dev,
      maneuver_enabled: enabled, budget, proposed, approved, applied_dv_mps: applied, intervened, d_min_m: dmin, d_safe_m: dsafe, risk,
      masked, collStep, death, pairStatus, satStatus, satEvent, closestD: cD, closestDs: cDs, closestJ: cJ, metrics: Mx, density, maxShield, maxShieldAt,
      runs, keySteps, stepEvents, logs, offline: sc.offline || [], limited: sc.limited || [],
      summary: { total_dv: Mx.dv[T], maneuvers: Mx.maneuvers[T], shield: off ? null : Mx.shield[T], collisions: Mx.collisions[T], mean_dev: Mx.meanDev[T] },
    };
    E.meta = { scenario: sc.id, tab, shield, model: E.model, condition: sc.condition, seed: sc.seed, numSteps: T, numAgents: N };
    E.frameAt = (t) => {
      t = Math.max(0, Math.min(T, t | 0));
      return { step: t, time_s: t * DT,
        satellites: Array.from({ length: N }, (_, i) => { const a = t * N + i, x = (t - 1) * N + i;
          return { index: i, position_km: [pos[a * 3], pos[a * 3 + 1], pos[a * 3 + 2]], alive: !!alive[a], status: STATUS[satStatus[a]], event: EVENT[satEvent[a]],
            remaining_dv_mps: rem[a], mission_deviation: dev[a], proposed: t ? proposed[x] : null, approved: t ? approved[x] : null }; }),
        pairs: PAIRS.map(([i, j], k) => ({ i, j, d_min_m: dmin[t * P + k], d_safe_m: dsafe[t * P + k], risk: risk[t * P + k], status: ['nominal', 'inside_d_safe', 'high_risk', 'collision'][pairStatus[t * P + k]] })) };
    };
    return E;
  }

  const cache = {};
  function source(scenarioId, tab, shield) {
    const sc = SCENARIOS.find((s) => s.id === scenarioId) || SCENARIOS[0], tb = tab === 'before' ? 'before' : 'after', sh = shield === 'off' ? 'off' : 'on';
    const k = sc.id + ':' + tb + ':' + sh;
    if (!cache[k]) cache[k] = build(sc, tb, sh);
    return cache[k];
  }
  window.OGData = { T, N, DT, RE, P, PAIRS, ACTIONS, HIGH, scenarios: SCENARIOS.map((s) => ({ id: s.id, label: s.label, sub: s.sub, condition: s.condition })), source };
})();
