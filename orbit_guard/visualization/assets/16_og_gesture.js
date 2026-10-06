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

  const PLATE_KEYS = ["x", "y", "z", "text", "width", "height", "bgcolor", "bordercolor", "borderwidth", "arrowcolor", "ax", "ay", "font"];

  /* Label plates are scene annotations. Plotly redraws them from `fullSceneLayout.annotations`, so with an unchanged
     plate count the new values can be written there and `drawAnnotations` re-run: DOM only, no scene rebuild (a full
     react costs ~150 ms on a real GPU, an in-place frame 3-7 ms). Plotly's derived fields (_pdata, _xa) are kept. */
  function platesFit(el, annotations) {
    const live = el._fullLayout.scene.annotations;
    return Array.isArray(live) && live.length === annotations.length;
  }

  function writePlates(el, annotations) {
    const live = el._fullLayout.scene.annotations;
    annotations.forEach((next, n) => PLATE_KEYS.forEach((key) => { if (key in next) live[n][key] = next[key]; }));
  }

  /* Push `data` into the live WebGL traces. Returns false when that is not possible (the caller then redraws fully).
     `strict` (playback frames) also refuses when a mesh changed vertex count or the plate count changed; during a
     gesture such a change just waits for the single redraw when the gesture settles. `annotations` (the label plates)
     are moved in place when their count is unchanged. */
  function applyInPlace(el, data, { strict = false, annotations = null } = {}) {
    const scene = el && el._fullLayout && el._fullLayout.scene && el._fullLayout.scene._scene;
    if (!scene || !scene.traces || !el._fullData || el._fullData.length !== data.length) return false;
    if (strict && data.some((next, n) => n >= FIRST_DYNAMIC_TRACE && needsRedraw(el._fullData[n], next, scene))) return false;
    const platesInPlace = Boolean(annotations) && platesFit(el, annotations);
    if (strict && annotations && !platesInPlace) return false;
    if (platesInPlace) writePlates(el, annotations); // before the redraw: Plotly's render() ends by drawing the plates
    for (let n = FIRST_DYNAMIC_TRACE; n < data.length; n += 1) {
      const full = el._fullData[n], next = data[n], trace = full && scene.traces[full.uid];
      if (!trace || !next.x || !full.x) continue;
      if (next.x.length !== full.x.length && next.type !== "scatter3d") continue; // a mesh cannot change vertex count in place
      full.x = next.x; full.y = next.y; full.z = next.z; full._length = next.x.length;
      if (next.type === "mesh3d") { full.i = next.i; full.j = next.j; full.k = next.k; full.vertexcolor = next.vertexcolor; }
      if (next.text) full.text = next.text;
      // Plotly reads these from _fullData: hover text when the trace updates, customdata on a click, opacity for the fades.
      for (const key of ["hovertext", "customdata", "opacity"]) if (next[key] !== undefined) full[key] = next[key];
      if (next.line && next.line.color && full.line) full.line.color = next.line.color;
      try { trace.update(full); } catch (error) { /* one bad trace must not freeze the view */ }
    }
    let redrawn = false;
    try { if (scene.glplot.redraw) { scene.glplot.redraw(); redrawn = true; } else scene.glplot.dirty = true; } catch (error) { /* ignore */ }
    if (platesInPlace && !redrawn) { try { if (scene.drawAnnotations) scene.drawAnnotations(scene); } catch (error) { /* a stale plate must not freeze the view */ } }
    return true;
  }

  /* True when the scene's WebGL context is gone (a GPU reset, a driver crash). Plotly never restores it, so the stage
     must throw the scene away and build a new one. A context that cannot even be asked is treated as lost. */
  function contextLost(el) {
    const scene = el && el._fullLayout && el._fullLayout.scene && el._fullLayout.scene._scene;
    if (!scene) return false;
    try {
      if (scene.glplot && scene.glplot.contextLost) return true;
      return Boolean(scene.gl && scene.gl.isContextLost && scene.gl.isContextLost());
    } catch (error) {
      return true;
    }
  }

  /* Plotly's recoverContext waits for a restore that never comes (it does not cancel the loss event) and then reads a scene
     the stage has already purged. The stage rebuilds the scene itself, so Plotly's attempt is replaced by a no-op. */
  function disableRecovery(el) {
    const scene = el && el._fullLayout && el._fullLayout.scene && el._fullLayout.scene._scene;
    if (scene && typeof scene.recoverContext === "function") scene.recoverContext = () => {};
  }

  /* A flapping GPU must not turn into an endless purge/redraw loop: allow `max` rebuilds per `windowMs`, then the caller
     fails the replay visibly. Returns {allowed, history} without mutating the input. */
  function rebuildAllowed(history, now, { max, windowMs }) {
    const recent = history.filter((t) => now - t < windowMs);
    return recent.length >= max ? { allowed: false, history: recent } : { allowed: true, history: [...recent, now] };
  }

  OG.gesture = { createTracker, applyInPlace, contextLost, disableRecovery, rebuildAllowed, WHEEL_HOLD_MS, FIRST_DYNAMIC_TRACE };
})(globalThis);
