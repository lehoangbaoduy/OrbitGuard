# Handoff: OrbitGuard episode replay — deep-space glass HUD

## Overview
This is a redesign of OrbitGuard's Dash replay app. It is a research tool that replays recorded multi-satellite
collision-avoidance episodes: 15 LEO satellites, 360 steps of 60 s. Researchers compare an **untrained** against a **trained** policy, each with
the safety **shield ON or OFF**, across sample **scenarios** (normal, 3 simultaneous conjunctions, capability
reduction).

## About the design files
The files in `design/` are **HTML design references**: prototypes of the intended look and behaviour, not
production code. Recreate them in the existing **Python Dash + Plotly** app (`orbit_guard/visualization/`),
following its patterns. They are built as "Design Components" (`*.dc.html` + `support.js`); open
`design/OrbitGuard Replay.dc.html` from a local web server, because the browser blocks local file
loads, e.g. `python -m http.server` in `design/`.

> ⚠ **The prototype's data is synthetic.** `reference/orbitguard-data.SYNTHETIC.js` fabricates every
> episode so the UI can be demonstrated. It documents the *shape* of the data the UI needs; it must **not** be ported,
> and no number from it may appear in the app. All data must come from recorded `EpisodeTrace` files (see
> `PROMPT_FOR_CLAUDE_CODE.md` → "Non-negotiable: real data only").

## Fidelity
**High fidelity.** Colours, type, spacing, states and motion are final. `design/OrbitGuard Design System.dc.html`
is the written spec: status vocabulary with contrast ratios, tokens, component notes, the five key
animations, Plotly 3D limits with workarounds, and Dash porting notes. Read it alongside this file.

## Screen: Replay (single screen, responsive)

**Header**
- Left: the wordmark ORBITGUARD (Martian Mono 14/500, letter-spacing .34em, ice #dbe6f3 with a soft glow) and the
  kicker EPISODE REPLAY.
- SCENARIO segmented group: label + seed per option, active = ice border + glow.
- Right: the meta line (model · condition · seed · N sats · steps) in mono 11, text-3.

**Tab bar + summary strip**
- **Tabs:** BEFORE TRAINING / AFTER TRAINING (Instrument Sans 13/600, .16em). The active tab has a 2 px ice
  underline and a faint top gradient. Each tab carries a sub-line ("untrained policy · shield ON").
- **SHIELD ON | OFF** segmented control, right-aligned. These are three independent choices, giving 12
  episodes when all are recorded.
- **Shield OFF indicator** (only when off): a dashed, hatched bar reading "SHIELD OFF — nothing overrides the
  policy" plus the list of remaining events.
- **Summary strip:** 5 cells (Total Δv · Maneuvers · Shield · Collisions · Mission deviation), separated by 1 px
  hairlines.
  - Each cell shows this tab's **episode total** (mono 26/300) and paired BEFORE / AFTER bars on one shared scale
    (active tab in ice, the other dimmed).
  - Under the bars: a ratio note and the running value at the current step ("T+250: 25.10").
  - Shield OFF: the Shield cell shows "—", never 0.
  - Phone: one row of 5 compact cells, no bars.

**Main, left column: 3D stage**
- Height clamp(580px, 50vw, 740px) when the dock fits (stage ≥ 780 px wide), otherwise clamp(270px, 40vw, 520px).
- Mission clock top-left: T+ in mono 44/300. A SHIELD OFF chip appears under it when off.
- Event banner: under the clock when docked, otherwise top centre.
- Legend glass bottom-left; Zoom − / +, Orbits and View (reset) bottom-right.
- **Docked right column (330 px, inside the stage):**
  - **Tracked this step:** a 15-cell per-satellite strip plus a tally. Rows are aggregated when more than 3
    satellites share an event type ("◇ ×11 · S0 S1 … · 11 blocked → COAST"), with Expand / +N more. A designed
    empty card reads "All nominal / No maneuvers / No shield events".
  - **Inspector:**
    - ‹ S8 › stepper and chips (status, event, capability).
    - Δv / DEV / CLOSEST readouts.
    - 5×3 satellite picker.
    - A 190 px Plotly chart with 3 rows sharing x: Δv steps with ■/◇ marks, deviation, and closest d_min vs dotted
      cyan d_safe on a log scale.
    - Decision line: proposed → approved, plus up to 3 "earlier" chips.

**Playback bar**
- Controls: ⏮ previous event · ‹ step · ▶ (54 px round, glows while playing) · › step · ⏭ next event.
- Scrubber: density bars on two rows (shield, maneuver) plus amber ◆ ticks at conjunction entries; hover preview.
- 1/2/4/8× segmented speed, and Loop.

**Timeline** (Counts | By satellite)
- **Counts:** 3 lanes of 42 px. Bar height = satellites with the event per step (0–15); shield bars are hollow
  with a cap.
  - Only the selected satellite gets in-lane marks: a glowing run line + "S9 ×3" in text-2.
  - Hovering anywhere shows the step's satellite IDs.
  - Shield OFF: the lane is hatched with "Shield off — no overrides recorded".
- **By satellite:** 15 × 360 matrix; tall cell = maneuver, short cell = shield. Click a row to select that
  satellite, a column to jump to that step.

**Right column (sticky when side by side)**
- **Fleet fuel:**
  - Fleet Δv left (mono 22) and "N of M tanks empty".
  - A 52 px area chart of fleet Δv over the episode, with playhead and an "all empty" dot.
  - 15 horizontal bars of remaining Δv at the current step. Each track is that satellite's own budget:
    ◐ limited = shorter track, ⊘ offline = dashed track + OFF, empty = "EMPTY".
- **Conjunction & risk:** top 8 live pairs, ordered by status, then risk, then d_min.
  - Columns: pair · status chip · d_min (mono 14, d_safe beneath) · risk (value + 3 px bar).
  - A collided pair shows "—  impact T+n".

**Phone (< 760 px):** sticky stage + playback + Risk / Timeline / Inspect tabs. The strip stays on top, and
every hit target is at least 44 px.

## Interactions & behaviour
- **Selections:** switching tab or shield keeps scenario, step, selected satellite, camera and play state, and
  the strip counts to its new values (~420 ms). Switching scenario jumps to that scenario's chosen start step,
  which comes from config, not hard-coded.
- **Satellite selection:** 3D click, table row, picker, fuel bar, timeline row, rail row.
- **Previous / next event:** jumps to conjunction entries and Δv-budget depletions; event-run starts are
  included only when sparse (≤ 80 runs).
- **Keys:** Space, ←/→, [ / ] (ignored in inputs). The slider itself also supports PgUp/PgDn ±30, Home/End and
  Shift ±10.
- **Camera during playback:** Plotly gl3d resets the camera on every `react` / `restyle`, because
  `scene.plot → setViewport → camera.lookAt` runs each time. So: (1) before each update, copy the live
  camera (`scene._scene.getCamera()`) into `layout.scene.camera`; (2) during a drag or wheel gesture, skip
  Plotly entirely and update `scene.traces[uid].update(fullData)` + `glplot.redraw()`; (3) do one full `react`
  on release. Pin the Plotly version.
- **Motion** (tokens in `orbitguard-tokens.css`):
  1. Collision flash: 1400 ms, plus a ✕ ring at 800 ms.
  2. Shield ring settles 1.7×→1 in 600 ms, once per run start.
  3. Continuous playhead: rAF, interpolated positions.
  4. Conjunction line fade: 420 ms.
  5. Metric count-up: τ 110 ms.
  - Pane enter: 520 ms with a 70 ms stagger.
  - `prefers-reduced-motion`: all durations 0, stepping instead of interpolation, and a static red frame for a
    collision step.
- **States:** loading trace; no trace for this combination (disable the option); step 0 (no actions); all nominal;
  shield off; offline / limited thrusters; retired satellites.

## State
- Shared: `scenario`, `policy` (before|after), `shield` (on|off), `step` (int) + continuous `playhead`,
  `playing`, `speed`, `loop`, `selected_sat`, `timeline_mode`, `rail_expanded`, `camera`, `phone_tab`.
- The data source is `ReplaySource(trace)`; see the prompt for its interface. Derived data (density, runs,
  key steps, closest partner, running metrics, fleet fuel, capability flags) is computed once per trace in
  Python and serialised to `dcc.Store`.

## Design tokens
The source of truth is `design/orbitguard-tokens.css`, loaded after Nocturne `styles.css`:
- **Ground and surfaces:** ground #161826, void #0e1019, surface #232532. Glass panes use surface 60% (blur
  only on overlays inside the stage; do not use `backdrop-filter` on large or sticky panes, it causes compositing
  artefacts).
- **Accent and text:** ice accent #dbe6f3; text #e9e9ed / #cfd3e5 / #9397ab.
- **Status:** collision #ff5c7a ✕ · high risk #ffb547 ◆ · inside d_safe #4fd8e8 ○ · nominal #9397ab ● ·
  shield #bb9cf7 ◇ · maneuver #5fe08f ■. Capability glyphs: ⊘ offline, ◐ limited (neutral, dashed).
- **Type:** Martian Mono 300–500 for telemetry and IDs; Instrument Sans 400–600 for labels. Scale 44 / 34 / 26 / 22
  / 16 / 14 / 12 / 11, with 11 as the floor.
- **Radii:** stage 14 · pane 8 · chip 4. Space follows Nocturne's 0.7× scale; page gutter clamp(12px, 2vw, 28px).

## Assets
- `assets/earth-day-1k.jpg` and `assets/earth-night-1k.jpg`: downscaled from three-globe example textures
  (NASA Blue Marble / Black Marble derived). Confirm the licence and attribution before release.
- Icons: Phosphor (web font). Fonts: Google Fonts (Martian Mono, Instrument Sans); self-host them if the app
  must run offline.

## Files
- `design/OrbitGuard Replay.dc.html`: the screen prototype (template + logic class).
- `design/OrbitGuard Design System.dc.html`: spec page.
- `design/orbitguard-tokens.css`: tokens.
- `design/_ds/…`: Nocturne base stylesheet.
- `design/support.js`: prototype runtime only.
- `design/orbitguard-data.js`: **synthetic**, needed only so the prototype opens. Do not port it.
- `reference/orbitguard-data.SYNTHETIC.js`: the same file kept as an interface reference.
- `PROMPT_FOR_CLAUDE_CODE.md`: the implementation brief with the real-data rules.
