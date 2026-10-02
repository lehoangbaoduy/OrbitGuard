# OrbitGuard Replay Redesign — implementation plan & progress tracker

**Project:** port the Claude Design handoff (`design_handoff_orbitguard_replay/`) into the Dash + Plotly replay
app (`orbit_guard/visualization/`). **Process prompt:** [`docs/redesign/REDESIGN_PROMPT.md`](redesign/REDESIGN_PROMPT.md).
**Branch:** `replay_redesign` (local, from `finished_project`; nothing pushed). **Out of scope:** Phase 7, any
change to training/evaluation logic, the pending N=15 grid/sweep.

Status legend: `TODO` · `DOING` · `DONE` · `BLOCKED`. Updated after every step (see the progress log at the end).

## 0. Source of truth and how it maps onto the master template

| Master template | This project |
|---|---|
| PRD | Handoff: `README.md`, `PROMPT_FOR_CLAUDE_CODE.md`, `design/OrbitGuard Design System.dc.html` (spec), `design/OrbitGuard Replay.dc.html` (prototype) |
| PRD "Build Phases & Exit Criteria" | Handoff "Working agreement": PR1–PR7, plus PR0 (setup) added here |
| PRD "Testing & edge-case catalog" | README "States", DS "Empty & loading states", "Designing for both densities", "Shield OFF state", real-data rule 8 → catalog in §4 |
| Acceptance checklist / Definition of Done | The 8 real-data rules + colour/marker/label + no overflow + reduced motion → §5 |

## 1. What exists today, and what the handoff changes

Today: `replay.py` (trace I/O, `frame_at`), `figures.py` (Plotly figure builders + `SAT/PAIR/EVENT_STYLE`),
`app.py` (single-trace Dash app, `dcc.Graph`s, 450 ms `dcc.Interval`). Tests: `tests/evaluation/test_replay.py`,
`test_visualization_figures.py`, `test_visualization_app.py`.

The handoff replaces the layout and interaction model: scenario selector × BEFORE/AFTER tabs × SHIELD ON|OFF
(12 recorded episodes), textured 3D stage with docked rail + inspector, continuous playback engine, density-aware
views (histogram lanes, 15×360 matrix, fleet-fuel heatmap), explicit empty/shield-off states, phone layout.

## 2. Architecture decisions (D) — with evidence

| # | Decision | Why / evidence |
|---|---|---|
| D1 | **One vendored plotly.js 2.35.2 owns every chart; the layout contains no `dcc.Graph`.** Stage, inspector chart, heatmaps and bars are rendered by assets JS with that build. | Spike (2026-10-02): Dash 4.4.1 / plotly.py 7.1.0 ship plotly.js **4.1.1**; the camera technique is validated on 2.35.2 internals. With `assets/00_plotly-2.35.2.min.js` and no `dcc.Graph`, `window.Plotly.version === "2.35.2"`, Dash's bundle never loads, and `scene._scene.getCamera / .traces / glplot.redraw` all exist. Interim check (PR0): the *old* `dcc.Graph` app also adopts the vendored build (3 graphs + a WebGL canvas, 0 console errors), so there is no breakage while the old UI is replaced. We still render everything from JS (no `dcc.Graph`) because the stage needs direct access to the plotly div for camera handling. The handoff's own porting note allows client-side panels. |
| D2 | **Python computes, JS renders.** All numbers and derived series are produced by pure, unit-tested Python (`ReplaySource`) and shipped once per episode in a `dcc.Store`; JS never recomputes data, it only draws and interpolates positions. | Real-data rule 1; testability; the handoff says derived data is computed once in Python. |
| D3 | **State stays client-side** (closure-private module state, not `window.*`); one `clientside_callback` boots the engine; server callbacks only for episode selection → payload. | Handoff "Porting to Dash"; CONST-SEC-003 (no app state on `window`). |
| D4 | **Episodes are discovered from `trace.meta`** by scanning a directory (header only, no array loads). Key = `(scenario_id, policy, shield)`. | Real-data rule 3. `--trace FILE` is kept for single-trace use. |
| D5 | **Meta schema v1** (`replay_meta.py`): `schema`, `scenario_id`, `condition`, `seed`, `policy` (`untrained`/`trained`/`no_maneuver`/`rule_based`), `shield` (bool), `model`, `n_agents`, `delta_v_budget_mps`, `max_delta_v_per_action_mps`, `weights_sha256` (trained only), `init_seed` (untrained only). `load_trace` stays permissive for old traces; `validate_replay_meta` (called by discovery) raises with the missing field names. | Real-data rule 3 ("fail loudly on load"); old traces keep working with `--trace` + clear error. |
| D6 | **Action catalog has one source:** `intent_catalog()` in `orbit_guard/safety/maneuver_mapper.py` (name, short label, Δv from config `maneuver:` block, exhaust direction from the RTN table). Visualization imports it. | Real-data rule 5. `orbit_guard/safety/__init__.py` is empty and `maneuver_mapper` only imports numpy + `physics.frames`, so FR16's import closure stays clean. |
| D7 | **Default start step = the episode's peak-risk step** (derived by `derived/start_step.py`; first key step if no risk, else 0). | Handoff says "from config, not hard-coded". Derived from data = no new constants. |
| D8 | **Reference ("10 runs…") note is hidden** unless `--reference LOG.jsonl` (output of `orbit_guard.evaluation.runner`) is given; then it shows only the models present in the log. The untrained policy is not a registry model, so BEFORE never gets a reference number. | Real-data rule 6. My earlier 10-seed run was an ad-hoc script, not `runner.py`. |
| D9 | **All assets self-hosted:** plotly.js, fonts (Martian Mono, Instrument Sans), icons, Earth JPGs. No CDN at runtime. Earth textures keep an attribution/licence note in `assets/README_ASSETS.md`. | Security review rule; handoff §Assets (offline). |
| D10 | **JS in small modules** (≤ 400 lines, 800 cap), `assets/1x_og_*.js`, plain scripts loaded in numeric order, a single `dash_clientside.og` namespace. Pure helpers are tested with `node --test` (Node 22 available) from a pytest wrapper. | CONST-ARCH-003; handoff "one small, tested module" for camera internals. |
| D11 | **Legacy `figures.py` figure builders are removed** once their replacements are tested (PR7); `SAT/PAIR/EVENT_STYLE` stay and take the token hexes + glyph changes (◆, ○). | Handoff "Tokens". |
| D12 | **Handoff folder stays where it is** (repo root); `*:Zone.Identifier` is git-ignored. | Q3 default. |
| D13 | **Retirement comes from collisions, never from `alive`.** The env empties `agents` on truncation, so `alive[T]` is all False on every real trace (found by review on the 12 recordings). `retired_matrix` = satellite in a collided pair; `alive` is no longer shipped. | Review finding H1; EC-08. |
| D14 | **Fleet fuel counts usable tanks only**; offline thrusters' fuel is reported as `stranded_mps`. Shield lane is `None` (not 0) when the shield is off. One burn threshold (`BURN_EPSILON_MPS` = env constant, drift-tested). | Review findings M4–M6; real-data rule 4. |
| D15 | **BEFORE and AFTER tabs are always offered**; a policy without a trace is disabled with "no trace recorded". Meta needs `n_agents >= 2` (the env's own minimum) and must agree with the trace. | EC-03; review M9. |
| D16 | **Selection callback updates controls in place** (wildcard outputs for className/disabled/title/aria-pressed/children) instead of rebuilding them, and re-validates the client-held selection against the index. | Keyboard focus survives; the State store is client-controlled. |

## 3. Open questions (Q) — conservative defaults taken, user can override

| # | Question | Default taken |
|---|---|---|
| Q1 | Reference collision note: no evaluation-pipeline output exists for the untrained policy. | Hide unless `--reference` given (D8). Redo through `runner.py` only if you want the note. |
| Q2 | Per-scenario start step. | Derived peak-risk step (D7). |
| Q3 | Keep the handoff in place or move to `docs/`. | Keep in place. |
| Q4 | Commits. | Local commits per PR on `replay_redesign`, never pushed. |
| Q5 | The design assumes a collision in the capability shield-off episode (synthetic). The real recorded capability shield-off episode has **no collision**. | UI is data-driven; collision states are verified with a real no-maneuver collision trace (`--trace`) and test fixtures, not with the 12 episodes. |
| Q6 | After-training traces use `n15_adaptive_s0` (shield on) and `n15_noshield_s0` (shield off): two different trained models. If the grid forces a retrain, the 12 traces go stale. | Recorded in each trace's meta (`weights_sha256`); re-record with `scripts/record_replay_traces.py`. |

## 4. Edge-case catalog — every row is a required task with a test

Test locations: `py` = pytest under `tests/evaluation/` (or named), `js` = `node --test` under `tests/js/`,
`ui` = Playwright MCP evidence (screenshots under `docs/redesign/evidence/`).

| ID | Case | Source | Test | PR | Status |
|---|---|---|---|---|---|
| EC-01 | Loading trace: stage text, panes keep size (no layout jump) | DS "Empty & loading" | ui | 6 | DOING (loading region + stable pane sizes in layout, PR2; JS hide-on-ready lands PR4) |
| EC-02 | No trace / empty directory: "No episode loaded", controls disabled | DS | py + ui | 6 | DOING (server-rendered 'No episode loaded' + disabled controls tested; Playwright evidence PR6) |
| EC-03 | Scenario×policy×shield with no trace → option disabled "no trace recorded" | rule 4 | py + ui | 1/6 | DONE (py: index offers BEFORE/AFTER always, missing ones disabled 'no trace recorded' in layout + callback; ui evidence in PR6) |
| EC-04 | Step 0: inspector says actions exist for steps 1–N | DS | js + ui | 5 | TODO |
| EC-05 | All nominal: rail empty-card ("All N satellites nominal") | DS | js + ui | 5 | TODO |
| EC-06 | Shield OFF: strip "—", hatched lane, legend dim + "(off)", tally "◇ off", empty-card text | DS | py + ui | 6 | TODO |
| EC-07 | Thruster OFFLINE ⊘ / LIMITED Δv ◐ badges, fuel-track styles | README | py + ui | 1/5 | DOING (py capability flags DONE; badges PR5) |
| EC-08 | Retired/collided satellites: ✕ ghost, "—" d_min, "impact T+n", retire strip, excluded from ranking | README | py + ui | 5/6 | DOING (py: retirement derived from collisions, not `alive`; ghost/impact UI PR5/6) |
| EC-09 | NaN/absent field → "—" + reason tooltip | rule 4 | py + js | 5 | TODO |
| EC-10 | Dense BEFORE: >3 same-type grouped, "+N more", histogram lanes, 15×T matrix, banner "N satellites" | DS | py + js + ui | 5 | TODO |
| EC-11 | Sparse AFTER: IDs in lanes when ≤ 30 runs | DS | js + ui | 5 | TODO |
| EC-12 | Collision step with reduced motion → static red frame | README | ui | 7 | TODO |
| EC-13 | Switching tab/shield keeps scenario, step, selection, camera, play state | README | js + ui | 4 | TODO |
| EC-14 | Switching scenario → derived start step | D7 | py + ui | 4 | TODO |
| EC-15 | Camera drag/wheel during playback is not reset | README | js + ui | 4 | TODO |
| EC-16 | No horizontal overflow at 320/390/768/1024/1440, including **live resize** | user rule + earlier bug | ui | 2/7 | DOING (PR2 shell verified at 320/390/768/1024/1440 exact CSS px + live resize, 0 overflow; re-check each PR) |
| EC-17 | Phone < 760 px: sticky stage + Risk/Timeline/Inspect tabs, ≥ 44 px targets | README | ui | 7 | TODO |
| EC-18 | Keys: Space, ←/→, [ / ], ignored in inputs; slider PgUp/PgDn ±30, Home/End, Shift ±10 | DS | js + ui | 4 | TODO |
| EC-19 | `prefers-reduced-motion`: durations 0, stepping, no interpolation | README | js + ui | 7 | TODO |
| EC-20 | Earth textures fail → stylised surface fallback | DS | ui | 3 | TODO |
| EC-21 | Hover only repeats rail/table/inspector info; touch works | DS | ui | 5 | TODO |
| EC-22 | Selected satellite retires mid-episode | derived | js + ui | 5 | TODO |
| EC-23 | Prev/next event disabled at the ends (45%) | DS | js + ui | 4 | TODO |
| EC-24 | Reference note hidden without `--reference` | rule 6 | py + ui | 1/5 | DOING (py DONE: reference.py filters model/condition/n_agents, errors name file:line; UI PR5) |
| EC-25 | Comparative copy only from data; denominator 0 → no ratio | rule 7 | py | 1 | DONE (never '100% less' for non-zero, NaN/inf/zero denominator → None) |
| EC-26 | Other tab's trace missing → bars show "—" | rule 4 | py + ui | 1/5 | DOING (py DONE: missing counterpart → None; strip renders 'no trace recorded' without bar; JS running values PR5) |
| EC-27 | Meta missing/invalid → fail loudly naming fields | rule 3 | py | 1 | DONE |
| EC-28 | Old trace without meta via `--trace` → clear error | D5 | py | 1 | DONE |
| EC-29 | FR16: replay imports no Ray/torch | existing | py | 1/7 | DOING (py DONE: app + data layer import no ray/torch/gymnasium; final check PR7) |
| EC-30 | Literal guard: no episode numbers/synthetic imports in `.py/.js/.css`; assets folder lacks `orbitguard-data.js`, `support.js` | rule 8 | py | 0 | DONE |
| EC-31 | Status matrices equal `frame_at` at every step (consistency of vectorised vs per-frame logic) | derived | py | 1 | DONE (vectorised == frame_at at every step incl. 12 real N=15 traces, one burn threshold) |
| EC-32 | Payload size budget (≈ 1.5 MB / episode) and only the current episode in the Store | advisor | py | 1 | DONE (0.95 MB measured on a recorded N=15 T=360 episode, test runs without local files) |
| EC-33 | Vendored plotly.js checksum pinned; no `dcc.Graph` in layout | D1 | py | 0/2 | DONE (no dcc.Graph in layout, vendored build checksum pinned) |
| EC-34 | Label merging within threshold (≤ 4 per plate), plates never stack | DS | js + ui | 3 | TODO |
| EC-35 | Fleet-fuel: exhausted tanks, "all empty" step, retired tint, offline flat grey | DS | py + ui | 5 | TODO |

## 5. Exit criteria (Definition of Done)

- [ ] All 8 real-data rules hold and are test-enforced (guard, assets, meta, catalog, reference, derived tests).
- [ ] 12 real recorded episodes (3 scenarios × untrained/trained × shield on/off) discovered from meta; missing combinations disabled.
- [ ] Every status carries colour + marker + text label (ui evidence), with ◆ / ○ glyphs.
- [ ] Zero horizontal overflow at 320/390/768/1024/1440 and after live resize; ≥ 44 px targets on phone.
- [ ] Camera drag/zoom works during playback; playback is continuous (rAF) at 1/2/4/8×.
- [ ] Reduced-motion verified; no layout-triggering animation.
- [ ] Console clean in every checked state; FR16 import test green; full suite green; ruff clean.
- [ ] Code review (code-reviewer + security-reviewer for JS) done per PR; CRITICAL/HIGH findings fixed.
- [ ] Evidence screenshots committed under `docs/redesign/evidence/`; this tracker current.

## 6. Plan — PR0 … PR7 (exact files per step)

Per PR the harness-os pipeline runs: Constitution (checked once, rules noted) → Spec (`create_spec`, id
`OG-REPLAY-PRn`) → Test suite (written, observed RED) → Code (observed GREEN) → Code review → record decision.

### PR0 — Setup, spike, guardrails
Files: `.gitignore` (add `*:Zone.Identifier`), `docs/redesign/REDESIGN_PROMPT.md`, `docs/OrbitGuard_replay_redesign_progress.md`,
`orbit_guard/visualization/assets/00_plotly-2.35.2.min.js` (vendored), `assets/00_nocturne.css`,
`assets/01_orbitguard-tokens.css`, `assets/earth-day-1k.jpg`, `assets/earth-night-1k.jpg`, `assets/README_ASSETS.md`,
`tests/evaluation/test_visualization_real_data_only.py`, `tests/evaluation/test_visualization_assets.py`.
| Task | Status |
|---|---|
| Branch `replay_redesign`; `.gitignore` for Zone.Identifier | DONE |
| Spike: plotly.js 4.1.1 vs vendored 2.35.2 (decision D1) | DONE |
| Prompt + plan written; harness-os spec `OGREPLAY-PROD-000` registered (spec row 280) | DONE |
| Vendor plotly (sha256 pinned in test; identical bytes from cdn.plot.ly and cdn.jsdelivr.net), copy the 2 CSS + 2 JPG (NOT data/support JS) | DONE |
| Guard tests EC-30, EC-33 — observed RED (10 failing), then GREEN (21 passing; evaluation suite 120 passing; ruff clean). One RED was a guard false positive (`T+1` shape notes) fixed by tightening the pattern to `T+<2+ digits>` | DONE |
| Review: harness-os `request_review` (risk low) → no review gate; self-review recorded in the log | DONE |

### PR1 — Data layer (ReplaySource + derived functions + meta + discovery + recorded traces)
Files: `orbit_guard/visualization/replay_meta.py`, `replay.py` (edit), `episodes.py`, `source.py`, `payload.py`,
`reference.py`, `derived/{__init__,status,density,runs,key_steps,fuel,partners,metrics,capability,start_step}.py`,
`orbit_guard/safety/maneuver_mapper.py` (edit: `intent_catalog`), `scripts/record_replay_traces.py`,
tests: `tests/evaluation/test_replay_meta.py`, `test_episodes.py`, `test_derived_status.py`, `test_derived_density.py`,
`test_derived_runs.py`, `test_derived_key_steps.py`, `test_derived_fuel.py`, `test_derived_partners.py`,
`test_derived_metrics.py`, `test_derived_capability.py`, `test_derived_start_step.py`, `test_replay_source.py`,
`test_replay_payload.py`, `test_reference.py`, `tests/safety/test_intent_catalog.py`; FR16 test extended in `test_replay.py`.
| Task | Status |
|---|---|
| Meta schema + `validate_replay_meta` + `build_replay_meta` (EC-27, EC-28) | DONE |
| `intent_catalog()` single source (labels, Δv, exhaust axis) | DONE |
| Vectorised status/event matrices == `frame_at` (EC-31) | DONE |
| density, runs (merge ≤ 3 steps), key steps (conjunction entries, budget depletions, sparse run starts ≤ 80), fleet fuel, closest partner (excludes retired/collided), running metrics, capability flags (EC-07), start step (D7) | DONE |
| `ReplaySource` facade; payload (float32, uint8) ≤ budget (EC-32) | DONE |
| Episode discovery from meta, disabled-option info (EC-03); ratio/percent copy (EC-25); reference stats (EC-24) | DONE |
| `scripts/record_replay_traces.py` → `~/orbitguard_runs/replay_traces/` (12 traces + untrained actor + weights hashes) | DONE |

Review: python-reviewer BLOCK on 3 HIGH + 10 MEDIUM → all HIGH and every actionable MEDIUM fixed with tests first (`tests/evaluation/test_replay_hardening.py`, 36 tests). Real traces: 12 episodes recorded to `~/orbitguard_runs/replay_traces` (+ `weights/`), discovery verified, payload 0.97–0.99 MB.

### PR2 — Tokens/CSS + static layout
Files: `assets/02_og_base.css`, `03_og_layout.css`, `04_og_stage.css`, `05_og_panels.css`, `06_og_timeline.css`, `07_og_motion.css`,
`08_og_phone.css`, self-hosted `assets/fonts/*`, `assets/icons/*.svg`, `orbit_guard/visualization/layout.py`, `app.py` (rewrite),
`callbacks.py`, `figures.py` (styles → token hexes, glyphs), tests: `tests/evaluation/test_visualization_layout.py`, update `test_visualization_app.py`.
Header (brand + scenario selector), tab bar (BEFORE/AFTER + SHIELD segmented), summary-strip skeleton, main grid, loading/no-trace states.
Playwright: 320/390/768/1024/1440 + live resize (EC-16).
| Task | Status |
|---|---|
| Self-hosted fonts (Instrument Sans, Martian Mono latin subsets) + `02_og_fonts.css` | DONE |
| `selection.py` (resolve scenario/policy/shield, unavailable → ignored) + tests | DONE |
| `layout.py`, `layout_strip.py` (header, tabs, shield control, shield-off bar, summary strip, stage/rail regions, loading + empty states) — tests RED→GREEN, no `dcc.Graph`, no inline styles but the strip bar `transform` | DONE |
| `callbacks.py` (`PayloadProvider` with LRU(6) cache, `apply_click`, `parse_selection` validating client state, one selection callback updating controls IN PLACE via wildcard outputs so focus survives) + `app.py` rewrite (`create_app(index, reference_log)`, `--episodes/--trace/--reference-log`) | DONE |
| CSS `02_og_base`, `03_og_layout`, `04_og_stage`, `07_og_motion`, `08_og_phone` (container query for the dock at ≥ 780 px) | DONE (`05_og_panels`/`06_og_timeline` land with PR5, icons with PR4) |
| Playwright on the 12 real episodes: 5 widths, live resize, tab + shield clicks, console clean | DONE (evidence `docs/redesign/evidence/pr2_*.png`) |
| Review (code-reviewer on layout/callbacks/CSS) | DOING |
| `figures.py` token hexes/glyphs | MOVED to PR7 with the legacy removal (D11): nothing renders `figures.py` any more, so changing it now has no visible effect |

### PR3 — 3D scene (JS)
Files: `assets/10_og_core.js`, `12_og_scene_static.js`, `13_og_scene_dynamic.js`, `15_og_labels.js`, `tests/js/labels.test.mjs`, `tests/js/scene.test.mjs`.
Textured Earth mesh (+ fallback EC-20), atmosphere, graticule, orbit rings, satellite meshes (×300 caption), status pins, merged label plates
(EC-34), burn plumes from catalog exhaust, shield rings, legend, CSS starfield + parallax.

### PR4 — Playback engine + camera
Files: `assets/11_og_camera.js`, `14_og_playback.js`, `17_og_motion.js`, `18_og_boot.js`, `tests/js/camera.test.mjs`, `tests/js/playback.test.mjs`.
rAF loop + interpolation, controls, 1/2/4/8×, loop, prev/next event (EC-23), scrubber (density bars, conjunction ticks, hover preview), keys (EC-18),
camera module (EC-15), state kept across tab/shield switches (EC-13), scenario → start step (EC-14), the five key animations.

### PR5 — Panels, timeline, fuel, rail, inspector, strip
Files: `assets/16_og_panels.js`, `19_og_timeline.js`, `20_og_fuel.js`, `21_og_inspector.js`, `22_og_strip.js`, `tests/js/rail.test.mjs`, `tests/js/strip.test.mjs`.
Tracked-this-step (grouping, +N more), inspector (stepper, picker, 3-row chart, decision line), conjunction table, fleet-fuel heatmap + bars,
timeline Counts / By satellite, summary strip (totals, BEFORE/AFTER bars, ratio note, running value).

### PR6 — Shield-off and empty states
Files: `assets/23_og_states.js`, `05_og_panels.css` (edit), `layout.py` (edit). EC-01…EC-03, EC-06, EC-08, EC-09, EC-24, EC-26.

### PR7 — Phone layout, motion polish, reduced motion, cleanup
Files: `assets/08_og_phone.css`, `17_og_motion.js` (edit), `figures.py` (remove legacy builders), delete obsolete tests, docs.
EC-12, EC-16 (final), EC-17, EC-19, EC-29; perf (≤ 33 fps redraw), accessibility audit (a11y-architect), full review, final evidence.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Plotly gl3d internals differ across builds | D1: one vendored, checksummed build; camera logic isolated + tested; spike evidence recorded |
| Trace payload too large / slow first paint | float32/uint8 encoding, one episode in Store, EC-32 budget test |
| 15×360 matrix + 3 lanes cost on every redraw | draw static layers once; only playhead moves by transform (README) |
| 12 traces go stale after a retrain | meta carries weights hash; recording script is re-runnable |
| Playwright browser here runs at devicePixelRatio 1.1 (WSLg scaling): set the viewport to CSS width × 1.1 to test exact breakpoints | `window.innerWidth` is asserted in every check |
| Large JS surface with little automated coverage | pure helpers in `node --test`; Playwright evidence per PR |

## 8. Progress log (newest last)

| Date | Step | Result |
|---|---|---|
| 2026-10-02 | Explored handoff (README, prompt, DS spec, prototype template + 3D logic, tokens, synthetic data shape) and current `replay.py` / `figures.py` | done |
| 2026-10-02 | Spike D1 (Dash plotly.js version, internals, vendored 2.35.2) | **DONE** — vendored build wins; evidence in D1 |
| 2026-10-02 | Wrote `docs/redesign/REDESIGN_PROMPT.md` and this plan | done |
| 2026-10-02 | PR0: vendored plotly 2.35.2 + handoff CSS/JPGs, README_ASSETS.md, guard tests (RED→GREEN), harness-os spec registered | **DONE** — 21 tests green, evaluation suite 120 green, ruff clean |
| 2026-10-02 | Interim check: old dcc.Graph app still renders with the vendored plotly (3 graphs, 0 errors) | OK |
| 2026-10-02 | PR1: data layer (meta, catalog, derived series, source, payload, discovery, reference, recorder) + 12 real episodes recorded | **DONE** — reviewed (3 HIGH + 10 MEDIUM fixed test-first); suite 502 passed, ruff clean |
| 2026-10-02 | PR2: fonts, selection, layout, strip, callbacks, app rewrite, CSS; verified live on the 12 real episodes at 5 widths + live resize | DONE pending review; no overflow, console clean |
