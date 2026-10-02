/* OrbitGuard replay stage - the parts of the scene that do not change while an episode plays:
   Earth (textured mesh with a plain-surface fallback), atmosphere, graticule, orbit rings, starfield. Pure builders. */
(function (root) {
  "use strict";
  const OG = root.OG;
  const SUN = [0.83, 0.45, 0.33]; // display light direction only (fixed, stated in the legend), not an ephemeris
  const NO_HIGHLIGHT = { x: { highlight: false }, y: { highlight: false }, z: { highlight: false } };
  const OCEAN = [16, 38, 78];
  const NIGHT = [18, 24, 44];
  const ATMOSPHERE_SCALE = 1.035;
  const GRATICULE_LIFT_KM = 8;

  const smoothstep = (a, b, x) => {
    const u = Math.max(0, Math.min(1, (x - a) / (b - a)));
    return u * u * (3 - 2 * u);
  };

  function sphereGrid(radius, rows, cols, valueAt) {
    const x = [], y = [], z = [], c = [];
    for (let a = 0; a < rows; a += 1) {
      const v = (Math.PI * a) / (rows - 1);
      const rx = [], ry = [], rz = [], rc = [];
      for (let b = 0; b < cols; b += 1) {
        const u = (2 * Math.PI * b) / (cols - 1);
        const n = [Math.sin(v) * Math.cos(u), Math.sin(v) * Math.sin(u), Math.cos(v)];
        rx.push(radius * n[0]); ry.push(radius * n[1]); rz.push(radius * n[2]); rc.push(valueAt(n));
      }
      x.push(rx); y.push(ry); z.push(rz); c.push(rc);
    }
    return { x, y, z, c };
  }

  function earthSurface(radius, sun) {
    const g = sphereGrid(radius, 33, 65, (n) => n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]);
    return {
      type: "surface", x: g.x, y: g.y, z: g.z, surfacecolor: g.c, cmin: -1, cmax: 1, showscale: false, hoverinfo: "skip", name: "earth",
      colorscale: [[0, "#0a0c15"], [0.44, "#0d1120"], [0.52, "#17203a"], [0.7, "#1f2c4b"], [1, "#2b3c62"]],
      lighting: { ambient: 1, diffuse: 0, specular: 0, roughness: 1, fresnel: 0 }, contours: NO_HIGHLIGHT,
    };
  }

  function atmosphere(radius) {
    const g = sphereGrid(radius * ATMOSPHERE_SCALE, 17, 33, () => 0);
    return {
      type: "surface", x: g.x, y: g.y, z: g.z, surfacecolor: g.c, showscale: false, hoverinfo: "skip", opacity: 0.09, name: "atmosphere",
      colorscale: [[0, "#9fc2ee"], [1, "#9fc2ee"]], lighting: { ambient: 0.2, diffuse: 0.1, specular: 0.9, roughness: 0.25, fresnel: 4 },
      contours: NO_HIGHLIGHT,
    };
  }

  function graticule(radius) {
    const r = radius + GRATICULE_LIFT_KM, x = [], y = [], z = [];
    const point = (lat, lon) => {
      const la = (lat * Math.PI) / 180, lo = (lon * Math.PI) / 180;
      x.push(r * Math.cos(la) * Math.cos(lo)); y.push(r * Math.cos(la) * Math.sin(lo)); z.push(r * Math.sin(la));
    };
    const gap = () => { x.push(null); y.push(null); z.push(null); };
    for (const lat of [-60, -30, 0, 30, 60]) { for (let lon = 0; lon <= 360; lon += 6) point(lat, lon); gap(); }
    for (let lon = 0; lon < 360; lon += 30) { for (let lat = -90; lat <= 90; lat += 6) point(lat, lon); gap(); }
    return { type: "scatter3d", mode: "lines", x, y, z, line: { color: OG.ICE, width: 1 }, opacity: 0.08, hoverinfo: "skip", name: "graticule" };
  }

  /* Day and night maps baked into vertex colours, blended across the terminator per vertex.
     `day` and `night` are RGBA pixel arrays of size w x h. */
  function earthMesh(radius, sun, day, night, w, h, nv = 96, nu = 192) {
    const x = [], y = [], z = [], vertexcolor = [], i = [], j = [], k = [];
    for (let a = 0; a <= nv; a += 1) {
      const v = (Math.PI * a) / nv, ty = Math.min(h - 1, Math.floor((a / nv) * h));
      for (let b = 0; b <= nu; b += 1) {
        const u = (2 * Math.PI * b) / nu;
        const n = [Math.sin(v) * Math.cos(u), Math.sin(v) * Math.sin(u), Math.cos(v)];
        x.push(radius * n[0]); y.push(radius * n[1]); z.push(radius * n[2]);
        const o = (ty * w + Math.min(w - 1, Math.floor((b / nu) * w))) * 4;
        const lit = smoothstep(-0.12, 0.25, n[0] * sun[0] + n[1] * sun[1] + n[2] * sun[2]);
        const channel = (c) => {
          const lifted = Math.max(OCEAN[c], 255 * Math.pow(day[o + c] / 255, 0.65) * 1.05);
          const dark = NIGHT[c] + night[o + c] * 0.95;
          return Math.round(Math.min(255, lifted * lit + dark * (1 - lit)));
        };
        vertexcolor.push(`rgb(${channel(0)},${channel(1)},${channel(2)})`);
      }
    }
    const row = nu + 1;
    for (let a = 0; a < nv; a += 1) for (let b = 0; b < nu; b += 1) {
      const p = a * row + b;
      i.push(p, p + 1); j.push(p + row, p + row); k.push(p + 1, p + row + 1);
    }
    return { type: "mesh3d", x, y, z, i, j, k, vertexcolor, flatshading: false, hoverinfo: "skip", name: "earth",
      lighting: { ambient: 1, diffuse: 0, specular: 0, roughness: 1, fresnel: 0 } };
  }

  /* Each satellite's recorded positions, drawn as its orbit ring (the episode spans several revolutions). */
  function trajectoryRings(ep, stride) {
    const x = [], y = [], z = [];
    for (let s = 0; s < ep.N; s += 1) {
      const steps = [];
      for (let t = 0; t < ep.T; t += stride) steps.push(t);
      steps.push(ep.T);
      for (const t of steps) { const p = OG.posAt(ep, t, s); x.push(p[0]); y.push(p[1]); z.push(p[2]); }
      x.push(null); y.push(null); z.push(null);
    }
    return { type: "scatter3d", mode: "lines", x, y, z, line: { color: OG.ICE, width: 1 }, opacity: 0.1, hoverinfo: "skip", name: "orbits" };
  }

  function satelliteRing(ep, sat) {
    const x = [], y = [], z = [];
    for (let t = 0; t <= ep.T; t += 1) { const p = OG.posAt(ep, t, sat); x.push(p[0]); y.push(p[1]); z.push(p[2]); }
    return { type: "scatter3d", mode: "lines", x, y, z, line: { color: OG.ICE, width: 2 }, opacity: 0.45, hoverinfo: "skip", name: "selected-orbit" };
  }

  function staticTraces(ep, stride = 3) {
    return { earth: earthSurface(ep.earthRadiusKm, SUN), atmosphere: atmosphere(ep.earthRadiusKm), graticule: graticule(ep.earthRadiusKm), rings: trajectoryRings(ep, stride) };
  }

  /* Decorative starfield as a box-shadow list; seeded so every load looks the same. */
  function starShadows(count, width, height, seed) {
    let state = seed >>> 0;
    const next = () => {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    const stars = [];
    for (let n = 0; n < count; n += 1) {
      const x = Math.round(next() * width), y = Math.round(next() * height), colour = `rgba(219,230,243,${(0.35 + next() * 0.6).toFixed(2)})`;
      stars.push(`${x}px ${y}px 0 ${colour}`, `${x + width}px ${y}px 0 ${colour}`); // a second copy one tile over, so the sky wraps as the camera turns
    }
    return stars.join(", ");
  }

  Object.assign(OG.scene, { SUN, earthSurface, atmosphere, graticule, earthMesh, trajectoryRings, satelliteRing, staticTraces, starShadows });
})(globalThis);
