/* OrbitGuard replay stage - core: payload decoding and shared helpers.
   Pure and stateless (app state stays closure-private in the boot module). Every number the stage draws comes from
   the payload that Python built from a recorded trace; nothing here knows an episode. */
(function (root) {
  "use strict";
  const OG = root.OG || (root.OG = {});
  OG.scene = OG.scene || {};

  const SCHEMA = "orbitguard-replay-payload/1";
  const TYPED = { float32: Float32Array, uint8: Uint8Array, int8: Int8Array, int16: Int16Array };
  const ICE = "#dbe6f3"; // --og-ice: tests keep this equal to the token file
  const AXIS_MARGIN = 1.08; // the scene box is a little larger than the farthest recorded point

  function littleEndianHost() {
    return new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
  }

  function decodeArray(spec) {
    const Type = TYPED[spec.dtype];
    if (!Type) throw new Error(`unsupported dtype ${spec.dtype}`);
    if (!littleEndianHost()) throw new Error("payload arrays are little-endian; this host is not");
    const text = atob(spec.b64);
    const bytes = new Uint8Array(text.length);
    for (let k = 0; k < text.length; k += 1) bytes[k] = text.charCodeAt(k);
    const array = new Type(bytes.buffer);
    const expected = spec.shape.reduce((a, b) => a * b, 1);
    if (array.length !== expected) throw new Error(`array has ${array.length} values, shape says ${expected}`);
    return array;
  }

  const ARRAY_FIELDS = {
    pos: "positions_km", satStatus: "sat_status", event: "events", pairStatus: "pair_status", dMin: "d_min_m",
    dSafe: "d_safe_m", risk: "risk", remaining: "remaining_dv_mps", deviation: "mission_deviation",
    proposed: "proposed", approved: "approved", appliedDv: "applied_dv_mps", intervened: "intervened",
    partnerDMin: "partner_d_min_m", partnerDSafe: "partner_d_safe_m", partnerIdx: "partner_index", fuelEmpty: "fuel_empty",
  };

  function decodeEpisode(payload) {
    if (!payload || payload.schema !== SCHEMA) throw new Error(`unexpected payload schema ${payload && payload.schema}`);
    const ep = { payload, meta: payload.meta, catalog: payload.catalog, vocab: payload.vocab, shield: payload.meta.shield };
    ep.N = payload.dims.agents;
    ep.T = payload.dims.steps;
    ep.P = payload.dims.pairs;
    ep.controlDtS = payload.dims.control_dt_s;
    ep.earthRadiusKm = payload.constants.earth_radius_km;
    ep.openStep = payload.start_step;
    ep.pairs = Int32Array.from(payload.pair_index.flat());
    for (const [field, name] of Object.entries(ARRAY_FIELDS)) ep[field] = decodeArray(payload.arrays[name]);
    if (ep.T < 1) throw new Error("the trace has no steps to replay");
    const frames = ep.T + 1, per = { pos: frames * ep.N * 3, satStatus: frames * ep.N, event: frames * ep.N, pairStatus: frames * ep.P, remaining: frames * ep.N, deviation: frames * ep.N,
      proposed: ep.T * ep.N, approved: ep.T * ep.N, partnerIdx: frames * ep.N, partnerDMin: frames * ep.N, partnerDSafe: frames * ep.N, fuelEmpty: frames * ep.N };
    for (const [field, expected] of Object.entries(per)) if (ep[field].length !== expected) throw new Error(`${ARRAY_FIELDS[field]} has ${ep[field].length} values; the payload dimensions need ${expected}`);
    if (!ep.catalog.every((entry, index) => entry.index === index)) throw new Error("the action catalog is not indexed 0..n-1");
    return ep;
  }

  const posAt = (ep, t, i) => {
    const base = (t * ep.N + i) * 3;
    return [ep.pos[base], ep.pos[base + 1], ep.pos[base + 2]];
  };

  function axisRange(ep) {
    let farthest = ep.earthRadiusKm;
    for (let k = 0; k < ep.pos.length; k += 3) farthest = Math.max(farthest, Math.hypot(ep.pos[k], ep.pos[k + 1], ep.pos[k + 2]));
    return farthest * AXIS_MARGIN;
  }

  const easeOut = (u) => 1 - Math.pow(1 - u, 3);

  Object.assign(OG, { ICE, SCHEMA, decodeArray, decodeEpisode, posAt, axisRange, easeOut });
})(globalThis);
