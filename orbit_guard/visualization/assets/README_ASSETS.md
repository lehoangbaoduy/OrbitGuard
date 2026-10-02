# Replay app assets — provenance and licences

Dash serves every file in this folder to the browser, so this list is deliberate. The design prototype's
`orbitguard-data.js` (synthetic episodes) and `support.js` (prototype runtime) are **never** copied here;
`tests/evaluation/test_visualization_assets.py` enforces that.

| File | Source | Licence / note |
|---|---|---|
| `00_plotly-2.35.2.min.js` | plotly.js v2.35.2 (`cdn.plot.ly`, identical bytes on `cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2`); sha256 pinned in the test | MIT. Vendored because Dash 4.4.1 bundles plotly.js 4.1.1, and the camera-during-playback technique relies on 2.35.2 internals. |
| `00_nocturne.css` | Design handoff `_ds/nocturne-*/styles.css`, unmodified | Design-system base stylesheet from the Claude Design handoff. |
| `01_orbitguard-tokens.css` | Design handoff `orbitguard-tokens.css` | OrbitGuard tokens (colours, type, motion). Source of truth for the palette. |
| `earth-day-1k.jpg`, `earth-night-1k.jpg` (Earth day and night textures) | Design handoff; downscaled three-globe example textures (NASA Blue Marble / Black Marble derived) | NASA imagery is generally free to use with attribution, but **confirm the licence and attribution before any public release** (handoff README, "Assets"). |
