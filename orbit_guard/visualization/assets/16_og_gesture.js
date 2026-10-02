/* OrbitGuard replay stage - camera gestures during playback.
   Every Plotly.react runs scene.plot -> setViewport -> camera.lookAt, which would cancel a drag in progress. So while
   the user drags or wheels, the stage skips Plotly entirely: it pushes new coordinates straight into the WebGL trace
   objects and redraws, and does ONE full react (with the live camera, see 11_og_camera.js) when the gesture settles.
   This depends on Plotly internals, hence the vendored, pinned build. */
(function (root) {
  "use strict";
  const OG = root.OG;
  const WHEEL_HOLD_MS = 450; // a wheel has no "up": the scene stays still this long after the last tick
  const FIRST_DYNAMIC_TRACE = 4; // traces 0-3 (earth, atmosphere, graticule, rings) never change during playback

  function createTracker() {
    let dragging = false, wheelUntil = 0, pendingSettle = false;
    return {
      down() { dragging = true; pendingSettle = false; },
      up() { const was = dragging; dragging = false; if (was) pendingSettle = true; return was; },
      wheel(now) { wheelUntil = now + WHEEL_HOLD_MS; pendingSettle = true; },
      active(now) { return dragging || now < wheelUntil; },
      /* True exactly once after a gesture has finished: the moment for the single full redraw. */
      settled(now) {
        if (!pendingSettle || dragging || now < wheelUntil) return false;
        pendingSettle = false;
        return true;
      },
    };
  }

  /* True when `next` cannot be pushed into the live trace: it has points but Plotly built no WebGL object for it (it was
     empty at the last full draw), or it is a mesh whose vertex count changed. */
  function needsRedraw(full, next, scene) {
    if (!full || !next.x) return false;
    if (next.x.length > 0 && !scene.traces[full.uid]) return true;
    return next.type === "mesh3d" && Boolean(full.x) && next.x.length !== full.x.length;
  }

  /* Push `data` into the live WebGL traces. Returns false when that is not possible (the caller then redraws fully).
     `strict` (playback frames) also refuses when a mesh changed vertex count; during a gesture such a mesh just waits
     for the single redraw when the gesture settles. */
  function applyInPlace(el, data, { strict = false } = {}) {
    const scene = el && el._fullLayout && el._fullLayout.scene && el._fullLayout.scene._scene;
    if (!scene || !scene.traces || !el._fullData || el._fullData.length !== data.length) return false;
    if (strict && data.some((next, n) => n >= FIRST_DYNAMIC_TRACE && needsRedraw(el._fullData[n], next, scene))) return false;
    for (let n = FIRST_DYNAMIC_TRACE; n < data.length; n += 1) {
      const full = el._fullData[n], next = data[n], trace = full && scene.traces[full.uid];
      if (!trace || !next.x || !full.x) continue;
      if (next.x.length !== full.x.length && next.type !== "scatter3d") continue; // a mesh cannot change vertex count in place
      full.x = next.x; full.y = next.y; full.z = next.z; full._length = next.x.length;
      if (next.type === "mesh3d") { full.i = next.i; full.j = next.j; full.k = next.k; full.vertexcolor = next.vertexcolor; }
      if (next.text) full.text = next.text;
      if (next.line && next.line.color && full.line) full.line.color = next.line.color;
      try { trace.update(full); } catch (error) { /* one bad trace must not freeze the view */ }
    }
    try { if (scene.glplot.redraw) scene.glplot.redraw(); else scene.glplot.dirty = true; } catch (error) { /* ignore */ }
    return true;
  }

  OG.gesture = { createTracker, applyInPlace, WHEEL_HOLD_MS, FIRST_DYNAMIC_TRACE };
})(globalThis);
