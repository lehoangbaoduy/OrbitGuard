/* OrbitGuard replay stage - motion helpers: the one-shot animations and metric tweens as pure functions of time.
   Under reduced motion every animation collapses to its end state and a collision step shows a static red frame. */
(function (root) {
  "use strict";
  const OG = root.OG;
  // ms. flash, line (conjunction fade) and pulse mirror --og-dur-flash / --og-dur-count / --og-dur-pulse (tested).
  const DURATION = { flash: 1400, line: 420, pulse: 900, plume: 300, ring: 600, count: 110 };
  const SNAP = 1e-3; // a tween this close to its target is finished
  const STATIC_FLASH_ALPHA = 0.35;

  const clamp01 = (u) => Math.min(1, Math.max(0, u));

  /* 0..1 progress of an animation that started at `start` (ms timestamp, or null if it never did). */
  function growth(start, now, duration, reduced) {
    if (reduced || start === null || start === undefined) return 1;
    return clamp01((now - start) / duration);
  }

  /* Exponential approach used by the metric count-up (time constant `tau`). */
  function tween(current, target, dtMs, tau, reduced) {
    if (reduced || Math.abs(target - current) < SNAP) return target;
    return current + (target - current) * (1 - Math.exp(-dtMs / tau));
  }

  /* Collision flash overlay alpha: a quick rise, then an eased decay over the flash duration. */
  function flashAlpha(start, now, reduced) {
    if (reduced) return start === null || start === undefined ? 0 : STATIC_FLASH_ALPHA;
    if (start === null || start === undefined) return 0;
    const u = clamp01((now - start) / DURATION.flash);
    return u >= 1 ? 0 : Math.pow(1 - u, 2) * clamp01((now - start) / 60 + 0.4);
  }

  OG.motion = { DURATION, growth, tween, flashAlpha };
})(globalThis);
