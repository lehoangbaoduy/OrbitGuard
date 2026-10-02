# PROMPT FOR CLAUDE CODE — OrbitGuard replay redesign

Paste everything below the line into Claude Code, from the root of the OrbitGuard repo, with the
`design_handoff_orbitguard_replay/` folder copied into the repo (e.g. `docs/design_handoff_orbitguard_replay/`).

---

You are implementing a redesign of OrbitGuard's episode-replay app (`orbit_guard/visualization/app.py`,
`figures.py`, `replay.py`). The design reference is in `docs/design_handoff_orbitguard_replay/`. Read `README.md` there first, then open `OrbitGuard Replay.dc.html` and `OrbitGuard Design System.dc.html` in a
browser to see the intended look and behaviour. Treat those HTML files as **design references only**: rebuild
the UI in our existing stack (Python Dash + Plotly, custom CSS in `assets/`). Do not ship or embed the HTML.

## Non-negotiable: real data only

The prototype runs on a **synthetic generator** (`reference/orbitguard-data.SYNTHETIC.js`). Every number,
trajectory, event and episode in it is made up so the design could be shown. **None of it may reach the app.**

1. Every value on screen must come from a recorded `EpisodeTrace` loaded with `load_trace()`, or be computed
   from its arrays by a pure, unit-tested function. That means positions, statuses, events, Δv, deviation, d_min,
   d_safe, risk, proposed/approved actions, summary totals, densities, run lists and key-event steps. Nothing is
   hard-coded, sampled, interpolated into existence, or "estimated for display".
2. Do not port, translate or copy `orbitguard-data.SYNTHETIC.js`. Do not create mock traces, seed-based fake
   episodes or placeholder numbers (no "12.20", "3,728", "seed 10", "T+250" literals in UI code). The design's
   example numbers are illustrations.
3. **Episodes are discovered, not invented.** The app is given a directory or manifest of `.npz` traces. Each
   trace identifies its scenario/condition, policy (untrained vs trained) and shield on/off from `trace.meta`.
   If a meta field we need is missing, extend `record_episode` / `save_trace` to store it, and fail loudly on
   load if it is absent. Do not infer it from the filename.
4. **Missing data is shown as missing.** If a scenario × policy × shield combination has no trace, its selector
   option is disabled with "no trace recorded". If a field is absent or NaN, show "—" with a short reason in the
   tooltip or caption. Never show 0 for "not applicable": when the shield is off, interventions are "—".
5. **Constants come from the system.** Read the Δv budget, d_safe defaults, HIGH_RISK_THRESHOLD (currently
   `replay.py`), step length and N from the trace, env config or existing module constants. Take the action
   labels (NO MANEUVER, RAISE ORBIT, LOWER ORBIT, PHASE FORWARD, PHASE BACKWARD, RADIAL ADJUST and their Δv) from
   the environment's action definition, not a new hand-written table. If no single source exists, add one in
   the env package and import it.
6. **Reference / comparison numbers come from evaluation output.** The "10 runs: 1/10 · no action 5/10"
   collision note must be read from an evaluation results file produced by our evaluation code. If no such
   file is configured, hide the note. Do not type the table from the design brief into code.
7. The UI must never claim training makes the fleet "safer". It shows measured values side by side; any
   comparative copy is generated from the data, such as "65× after" or "98% less than before".
8. Add a test, alongside `tests/evaluation/test_replay.py`, that fails if any module under
   `orbit_guard/visualization` contains literal episode numbers or imports a synthetic or demo data module.
   Also add tests for every derived-data function (density, runs, key steps, closest partner, running metrics,
   capability flags, fleet fuel) against a small real or recorded test trace.

## What to build (see README for exact specs)

- **ReplaySource layer (Python).** Implement `ReplaySource` (`meta`, `frame_at(t)`, `summary`,
  `density[type][t]`, `runs`, `key_steps`, `fleet_fuel(t)`, `closest_partner(t, i)`) over `EpisodeTrace`, as pure
  functions. The UI talks only to this interface, so a live feed can implement it later. Nothing runs ML in the UI.
- **Layout.**
  - **Header:** brand + SCENARIO selector.
  - **Tab bar:** BEFORE TRAINING / AFTER TRAINING tabs + SHIELD ON|OFF segmented control (three independent
    choices), then the SHIELD OFF indicator when off, then the summary strip.
  - **Main:** 3D stage with the docked "Tracked this step" + Inspector column inside it, playback bar, then the
    Counts / By-satellite timeline.
  - **Right column (sticky):** Fleet fuel + Conjunction & risk. Phone (< 760 px): sticky stage + playback, with
    Risk / Timeline / Inspect tabs.
  - Switching tab or shield keeps scenario, step, selected satellite, camera and play state.
- **3D (Plotly Scatter3d / Mesh3d).**
  - **Earth:** textured mesh (`assets/earth-day-1k.jpg`, `earth-night-1k.jpg`) with terminator, atmosphere
    shell, graticule and orbit rings.
  - **Satellites:** mesh models, deliberately drawn about 300× true size. Say so in a caption; it is a visual
    scale, not data.
  - **Labels:** status pins with `scene.annotations` label plates (explicit width/height), merged when within
    1,100 km.
  - **Events:** maneuver = burn plume along the approved intent's exhaust direction; shield = violet ring. Both
    animate once at run start.
  - **Stars:** CSS starfield with camera-driven parallax.
- **Playback.**
  - **Controls:** 1/2/4/8×, previous/next event, loop, scrubber with density bars and conjunction ticks plus
    hover preview. Keys: Space play/pause, ←/→ step, [ / ] event.
  - **Engine:** `dcc.Store` holding the serialised trace plus a `clientside_callback` rAF loop. Server
    callbacks at 4 steps/s will not feel continuous.
  - **Camera:** dragging and zooming must work during playback. Read the README section "Camera during
    playback": copy the live camera into the layout before each update; during a gesture, update trace objects
    directly. That relies on Plotly internals, so pin `plotly==2.35.2` and wrap it in one small, tested module.
- **Density handling.** Group rows in "Tracked this step" when more than 3 satellites share an event type. Use
  histogram lanes in Counts view, a 15 × 360 matrix in By-satellite view, and banner text that collapses to
  "N satellites".
- **States.** Handle loading, no trace, step 0 (no actions), all nominal, shield off, thruster offline / limited
  Δv, and retired / collided satellites (show "—" for stale d_min after impact).
- **Tokens.** Ship `assets/00_nocturne.css` and `assets/01_orbitguard-tokens.css` (copy from the handoff). Turn
  the prototype's inline styles into classes in `assets/02_og.css`. Use the hex values in `figures.py`
  SAT_STYLE / PAIR_STYLE / EVENT_STYLE from the token file. Keep colour + marker shape + text label for every
  state, and the glyph changes ◆ (high risk) and ○ (inside d_safe), because Plotly 3D has no triangle.
- **Motion.** Use the top-5 animation spec and the reduced-motion fallback in the README. Animate only
  transform and opacity.

## Working agreement

- Start by reading `replay.py`, `figures.py`, `app.py` and the handoff README. Then propose a short plan that
  covers the module layout, the trace meta fields to add, the manifest format and the test list, and wait for
  my OK.
- Work in small PRs: (1) ReplaySource + derived-data functions + tests; (2) tokens/CSS + static layout;
  (3) 3D scene; (4) playback engine + camera; (5) timeline / fuel / rail density views; (6) shield-off and
  empty states; (7) phone layout + motion + reduced motion.
- For every place where the design shows a value the trace does not contain, stop and ask. Do not fill it in.
