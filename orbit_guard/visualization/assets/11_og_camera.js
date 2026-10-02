/* OrbitGuard replay stage - camera access. Plotly's gl3d resets the camera on every react/restyle, so before each
   redraw the LIVE WebGL camera is copied back into the layout. That relies on Plotly internals (scene._scene.getCamera),
   which is why the plotly.js build is vendored and pinned (see README_ASSETS.md). All of it lives in this one module. */
(function (root) {
  "use strict";
  const OG = root.OG;
  const STAR_TILE = { w: 2200, h: 1100 }; // the starfield layers are this big and wrap horizontally
  const PARALLAX_ELEVATION_PX = 220, PARALLAX_BASE_PX = -200, PARALLAX_MIN_PX = -400;

  const clone = (value) => JSON.parse(JSON.stringify(value));

  /* The camera Plotly is rendering right now, as a plain copy; null when the internals are unavailable. */
  function live(plotDiv) {
    try {
      const scene = plotDiv && plotDiv._fullLayout && plotDiv._fullLayout.scene && plotDiv._fullLayout.scene._scene;
      const camera = scene && scene.getCamera && scene.getCamera();
      return camera && camera.eye ? clone(camera) : null;
    } catch (error) {
      return null;
    }
  }

  /* A copy of `layout` whose scene camera is the live one (unchanged when there is none). */
  function withLive(layout, plotDiv) {
    const camera = live(plotDiv);
    if (!camera) return layout;
    const out = clone(layout);
    out.scene = { ...(out.scene || {}), camera };
    return out;
  }

  /* Starfield translation (px) for two depth layers, driven by the camera's azimuth and elevation. */
  function parallax(camera) {
    const { eye } = camera;
    const azimuth = Math.atan2(eye.y, eye.x), elevation = Math.atan2(eye.z, Math.hypot(eye.x, eye.y));
    const x = (k) => -((((azimuth / (2 * Math.PI)) * STAR_TILE.w * k) % STAR_TILE.w) + STAR_TILE.w) % STAR_TILE.w;
    const y = (k) => Math.max(PARALLAX_MIN_PX, Math.min(0, elevation * PARALLAX_ELEVATION_PX * k + PARALLAX_BASE_PX));
    return { far: { x: x(0.5), y: y(0.5) }, near: { x: x(1), y: y(1) } };
  }

  OG.camera = { live, withLive, parallax, STAR_TILE };
})(globalThis);
