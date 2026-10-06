/* OrbitGuard replay stage - the Plotly 3D stage: draws a view, keeps the camera, handles clicks and drag/wheel gestures.
   `createStage` returns an object whose state is private to its closure. The controller (19_) decides WHEN to draw;
   this module decides HOW: a full Plotly.react normally, or a direct WebGL update while the user is dragging/zooming so
   the gesture is never interrupted (see 16_og_gesture.js). */
(function (root) {
  "use strict";
  const OG = root.OG;
  /* Textures sit beside this script, so derive their URL from it (works under any url_base_pathname). */
  const script = document.currentScript, ASSET_BASE = script && script.src ? script.src.replace(/[^/]*$/, "") : "assets/";
  const TEXTURES = { day: `${ASSET_BASE}earth-day-1k.jpg`, night: `${ASSET_BASE}earth-night-1k.jpg` };
  const TEXTURE_SIZE = { w: 1024, h: 512 };
  const DOCK_MARGIN_PX = 24, MIN_DOMAIN = 0.4, ZOOM_MIN = 0.5, ZOOM_MAX = 3;
  // While playing, frames update the live WebGL traces and the label plates in place (3-7 ms against ~150 ms for a react on
  // a real GPU). A full Plotly.react (it also refreshes the picking data) follows after this many in-place frames or this
  // long, whichever is first, and at once when the plate count or a mesh size changes.
  const KEY_TURN_RAD = 0.2, MAX_ELEVATION = 1.45; // one arrow press turns ≈ 11°; elevation stops short of the poles
  const IN_PLACE_RUN_MAX = 240, FULL_REDRAW_MS = 8000, REBUILD_LIMIT = { max: 3, windowMs: 10000 };
  const STAR_LAYERS = { far: { id: "og-stars-far", count: 160, seed: 11 }, near: { id: "og-stars-near", count: 50, seed: 29 } };
  const plotConfig = () => ({ displayModeBar: false, responsive: true, scrollZoom: !(root.matchMedia && root.matchMedia("(max-width: 759px)").matches), doubleClick: false }); // on a phone a wheel/touch over the scene must not trap page scrolling
  const $ = (id) => document.getElementById(id);
  /* Plotly builds no WebGL object for an empty trace, so a trace that is empty now and filled later cannot be updated in
     place. Every line/marker trace therefore always carries one point, hidden inside the (opaque) Earth. */
  const HIDDEN_POINT = { x: [0], y: [0], z: [0] };
  const withHiddenPoint = (data) => data.map((d) => (d.type === "scatter3d" && d.x && d.x.length === 0 ? { ...d, ...HIDDEN_POINT } : d));

  function createStage(handlers) {
    const S = { ep: null, st: null, earthMesh: null, camera: null, revision: 0, resetView: false, bound: false, tracker: OG.gesture.createTracker(), starsDrawn: false, texturesRequested: false, selectedRing: null, lastFull: -Infinity, inPlaceRun: 0, rebuilds: [], signature: "", nativeBound: false };

    function domainRight() {
      const stage = $("og-stage"), dock = $("og-dock");
      if (!stage || !stage.clientWidth || !dock || dock.parentElement !== stage || getComputedStyle(dock).display === "none") return 1;
      return Math.max(MIN_DOMAIN, (stage.clientWidth - dock.offsetWidth - DOCK_MARGIN_PX) / stage.clientWidth);
    }

    function moveStars(camera) {
      const layers = OG.camera.parallax(camera || OG.scene.DEFAULT_CAMERA);
      for (const [name, spec] of Object.entries(STAR_LAYERS)) {
        const node = $(spec.id);
        if (node) node.style.transform = `translate(${layers[name].x.toFixed(1)}px, ${layers[name].y.toFixed(1)}px)`;
      }
    }

    function paintStars() {
      if (S.starsDrawn) return;
      for (const spec of Object.values(STAR_LAYERS)) {
        const node = $(spec.id);
        if (node) node.style.boxShadow = OG.scene.starShadows(spec.count, OG.camera.STAR_TILE.w, OG.camera.STAR_TILE.h, spec.seed);
      }
      S.starsDrawn = true;
    }

    function loadImage(src) {
      return new Promise((resolve, reject) => { const image = new Image(); image.onload = () => resolve(image); image.onerror = reject; image.src = src; });
    }

    function pixels(image) {
      const canvas = document.createElement("canvas");
      canvas.width = TEXTURE_SIZE.w; canvas.height = TEXTURE_SIZE.h;
      const context = canvas.getContext("2d", { willReadFrequently: true });
      context.drawImage(image, 0, 0, TEXTURE_SIZE.w, TEXTURE_SIZE.h);
      return context.getImageData(0, 0, TEXTURE_SIZE.w, TEXTURE_SIZE.h).data;
    }

    /* Textured Earth; if the maps fail to load the plain shaded sphere stays (EC-20). */
    function loadTextures() {
      if (S.texturesRequested) return;
      S.texturesRequested = true;
      const radius = S.ep.earthRadiusKm;
      Promise.all([loadImage(TEXTURES.day), loadImage(TEXTURES.night)]).then(([day, night]) => {
        const { w, h } = TEXTURE_SIZE;
        S.earthMesh = OG.scene.earthMesh(radius, OG.scene.SUN, pixels(day), pixels(night), w, h);
        if (S.st) S.st.earth = S.earthMesh;
        document.documentElement.setAttribute("data-og-earth", "textured");
        handlers.requestDraw(true);
      }).catch(() => document.documentElement.setAttribute("data-og-earth", "fallback"));
    }

    function bind(el) {
      if (S.bound || !el.on) return;
      S.bound = true;
      el.on("plotly_click", (event) => {
        const point = event && event.points && event.points[0];
        if (point && point.customdata !== undefined && point.customdata !== null) handlers.onSelect(Number(point.customdata));
      });
      el.on("plotly_relayout", (event) => {
        const camera = event && event["scene.camera"];
        if (camera && camera.eye) { S.camera = JSON.parse(JSON.stringify(camera)); moveStars(S.camera); }
      });
      if (S.nativeBound) return;
      S.nativeBound = true;
      el.addEventListener("webglcontextlost", () => handlers.requestDraw(true), true); // non-bubbling: capture sees it
      el.addEventListener("pointerdown", () => S.tracker.down(), true);
      el.addEventListener("touchstart", () => S.tracker.down(), { capture: true, passive: true });
      el.addEventListener("wheel", () => { S.tracker.wheel(performance.now()); handlers.requestDraw(false); }, { capture: true, passive: true });
      const release = () => { if (S.tracker.up()) handlers.requestDraw(true); };
      for (const type of ["pointerup", "pointercancel", "touchend"]) root.addEventListener(type, release, true);
      if (root.ResizeObserver && $("og-stage")) new root.ResizeObserver(() => handlers.requestDraw(true)).observe($("og-stage"));
    }

    function setEpisode(ep) {
      const first = !S.ep;
      S.ep = ep;
      S.st = OG.scene.staticTraces(ep);
      if (S.earthMesh) S.st.earth = S.earthMesh;
      S.selectedRing = null;
      paintStars();
      loadTextures();
      return first;
    }

    function selectedRing(sat) {
      if (!S.selectedRing || S.selectedRing.sat !== sat) S.selectedRing = { sat, trace: OG.scene.satelliteRing(S.ep, sat) };
      return S.selectedRing.trace;
    }

    /* Draw one view. Returns "react", "inplace" or "skipped". `force` demands a full react (after a gesture, a selection
       change, a resize); `playing` lets frames between full reacts update the live traces in place (much cheaper). */
    function draw(view, now, force, playing) {
      const el = $("og-scene");
      if (!S.ep || !el || !root.Plotly) return "skipped";
      let rebuilt = false;
      if (OG.gesture.contextLost(el)) { // the GPU reset under us: Plotly cannot recover, so start the scene afresh
        const verdict = OG.gesture.rebuildAllowed(S.rebuilds, now, REBUILD_LIMIT);
        S.rebuilds = verdict.history;
        if (!verdict.allowed) throw new Error("the WebGL context keeps being lost"); // the controller shows its failure state
        let live = null;
        try { live = OG.camera.live(el); } catch (error) { /* the dead scene may not answer: fall back to the last known camera */ }
        if (live) S.camera = live; // a loss in the middle of a drag keeps where the drag had got to
        try { root.Plotly.purge(el); } catch (error) { el.replaceChildren(); } // a dead context can make the teardown throw
        S.bound = false; S.signature = ""; S.inPlaceRun = 0;
        rebuilt = true;
        if (handlers.onNotice) handlers.onNotice("The 3D view lost its graphics context and was redrawn.");
      }
      const gesture = !rebuilt && S.tracker.active(now);
      const { data: traces, annotations } = OG.scene.sceneTraces(S.ep, { ...view, static: S.st, selectedRing: selectedRing(view.selected) });
      const data = withHiddenPoint(traces);
      const signature = `${view.selected}|${view.orbits}`;
      if (gesture && !force && OG.gesture.applyInPlace(el, data, { annotations })) return "inplace";
      const light = playing && !force && signature === S.signature && S.inPlaceRun < IN_PLACE_RUN_MAX && now - S.lastFull < FULL_REDRAW_MS;
      if (light && OG.gesture.applyInPlace(el, data, { strict: true, annotations })) { S.inPlaceRun += 1; return "inplace"; }
      S.signature = signature; S.lastFull = now; S.inPlaceRun = 0;
      let layout = OG.scene.layout(S.ep, { camera: S.resetView ? null : S.camera, annotations, domainRight: domainRight(), revision: S.revision });
      if (!S.resetView && !S.zoomed) layout = OG.camera.withLive(layout, el);
      S.resetView = false; S.zoomed = false;
      const done = root.Plotly.react(el, data, layout, plotConfig());
      OG.gesture.disableRecovery(el);
      if (done && done.then) done.then(() => OG.gesture.disableRecovery(el)); // the scene object exists only once the draw settles
      S.camera = OG.camera.live(el) || S.camera;
      moveStars(S.camera);
      bind(el);
      return done && done.then ? done.then(() => "react") : "react";
    }

    /* Remove the scene (an error or an empty selection must never leave the previous episode on screen). */
    function clear() {
      const el = $("og-scene");
      S.ep = null; S.st = null; S.selectedRing = null;
      if (el && root.Plotly && el._fullLayout) root.Plotly.purge(el);
      S.bound = false;
    }

    /* Turn the camera about the vertical axis (`yaw`) or change its elevation (`pitch`), one fixed step per key press. */
    function rotate(yaw, pitch) {
      const camera = OG.camera.live($("og-scene")) || S.camera || JSON.parse(JSON.stringify(OG.scene.DEFAULT_CAMERA));
      const { x, y, z } = camera.eye, angle = yaw * KEY_TURN_RAD, c = Math.cos(angle), s = Math.sin(angle), horizontal = Math.hypot(x, y);
      const elevation = Math.max(-MAX_ELEVATION, Math.min(MAX_ELEVATION, Math.atan2(z, horizontal) + pitch * KEY_TURN_RAD)), radius = Math.hypot(x, y, z);
      const azimuth = Math.atan2(x * s + y * c, x * c - y * s);
      S.camera = { ...camera, eye: { x: radius * Math.cos(elevation) * Math.cos(azimuth), y: radius * Math.cos(elevation) * Math.sin(azimuth), z: radius * Math.sin(elevation) } };
      S.zoomed = true;
    }

    function resetCamera() { S.camera = null; S.resetView = true; S.revision += 1; }

    /* Dolly in or out by scaling the eye distance (clamped), keeping the viewing direction. */
    function zoom(factor) {
      const el = $("og-scene"), camera = OG.camera.live(el) || S.camera || JSON.parse(JSON.stringify(OG.scene.DEFAULT_CAMERA));
      const length = Math.hypot(camera.eye.x, camera.eye.y, camera.eye.z), base = Math.hypot(OG.scene.DEFAULT_CAMERA.eye.x, OG.scene.DEFAULT_CAMERA.eye.y, OG.scene.DEFAULT_CAMERA.eye.z);
      const target = Math.min(ZOOM_MAX * base, Math.max(ZOOM_MIN * base, length * factor)), k = target / length;
      S.camera = { ...camera, eye: { x: camera.eye.x * k, y: camera.eye.y * k, z: camera.eye.z * k } };
      S.zoomed = true; // the next full draw uses this camera instead of the live one
    }

    return { setEpisode, draw, clear, rotate, releaseGesture: () => { if (S.tracker.up()) handlers.requestDraw(true); }, resetCamera, zoom, settled: (now) => S.tracker.settled(now), gestureActive: (now) => S.tracker.active(now), hasEpisode: () => S.ep !== null };
  }

  OG.createStage = createStage;
})(globalThis);
