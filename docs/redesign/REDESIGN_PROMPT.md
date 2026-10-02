# OrbitGuard Replay Redesign — project prompt

> Adapted from the user's master project template (originally written for another project). Here the
> "project" is a small one: porting the Claude Design handoff for the episode-replay app into the existing
> Python Dash + Plotly code base. This file sets the **process**; the handoff sets the **spec**.
> Plan + live tracker: [`docs/OrbitGuard_replay_redesign_progress.md`](../OrbitGuard_replay_redesign_progress.md).

I want to implement a project called **OrbitGuard Replay Redesign** — a redesign of the episode-replay web app
(`orbit_guard/visualization/`) from a Claude Design handoff: a "deep-space glass HUD" with BEFORE / AFTER
training tabs, a SHIELD ON|OFF control, a textured 3D stage, a continuous playback engine and dense-data
views. The handoff folder is `design_handoff_orbitguard_replay/` at the repo root. It is the source of truth for
every requirement below; this prompt sets the *process*, not the spec. Read, in full and before doing anything
else:

- `design_handoff_orbitguard_replay/README.md` — screen spec, interactions, state, tokens, assets.
- `design_handoff_orbitguard_replay/PROMPT_FOR_CLAUDE_CODE.md` — the implementation brief, including the eight
  **real-data-only** rules and the seven-PR working agreement.
- `design_handoff_orbitguard_replay/design/OrbitGuard Design System.dc.html` — the written spec: status
  vocabulary and contrast, tokens, the five key animations, Plotly 3D limits, Dash porting notes.
- `design_handoff_orbitguard_replay/design/OrbitGuard Replay.dc.html` — the working prototype (a **design
  reference only**; rebuild it in our stack, never ship it).
- `design_handoff_orbitguard_replay/design/orbitguard-tokens.css`, `_ds/.../styles.css` — tokens + base styles.

`design/orbitguard-data.js`, `reference/orbitguard-data.SYNTHETIC.js` and `design/support.js` are prototype
runtime / **synthetic** data: never copy, port or serve them.

## Step 1 — Implementation plan
Create a spec-kit style implementation plan and write it to `docs/OrbitGuard_replay_redesign_progress.md`.
Base its phase structure directly on the handoff's **Working agreement** (PR1–PR7 in
`PROMPT_FOR_CLAUDE_CODE.md`) plus a PR0 for setup — don't invent a different breakdown. Base its test coverage
directly on the handoff's states: README "States", the Design System's "Empty & loading states", "Designing for
both densities", "Shield OFF state", and item 8 of the real-data rules — treat every row of the plan's
edge-case catalog as a required task, not optional polish. The eight real-data rules, colour + marker + label
for every state, no horizontal overflow, and reduced-motion behaviour are what "complete" means — use them as
the plan's exit criteria. Every plan step must name the exact files it creates or edits (CONST-AI-001).

## Step 2 — Implement the plan
Work through the plan PR by PR. For each step:
1. Implement it.
2. Run a verification loop and confirm the output matches expectations.
3. If it doesn't, debug and iterate before moving on — don't proceed on a failing or unverified step.
4. Record progress in `docs/OrbitGuard_replay_redesign_progress.md`, and note any design decision made during
   implementation (especially where the handoff left an implementation detail open: manifest format, meta
   fields, start-step rule, label merging thresholds, payload encoding).

## Backend (Python data layer) work
Before implementing any data-layer logic, write the tests for it first and watch them fail. Run the suite after
every significant change, not just at the end of a PR. This applies especially to: the trace-meta schema and
"fail loudly" validation, episode discovery (never from filenames), the action catalog (single source in the
env package), every derived-data function (density, runs, key steps, closest partner, running metrics,
capability flags, fleet fuel, status matrices vs `frame_at`), payload serialisation, the reference-stats
reader, the FR16 import-closure check (no Ray/torch), and the literal-number / synthetic-import guard.

## Frontend work
Use the Playwright MCP server for every UI change. Actually look at what's rendering — don't infer correctness
from the code alone. This matters in particular for: the 3D stage (Earth texture, satellite models, labels,
plumes, rings), camera drag/zoom **during** playback, the scrubber, the dense BEFORE case vs the sparse AFTER
case, the shield-off state, and the phone layout. Check 320 / 390 / 768 / 1024 / 1440 px, a **live resize**
(the earlier overflow bug), keyboard-only operation, `prefers-reduced-motion`, and a clean console. For every
view, confirm zero synthetic/demo data remains (hard requirement, not cleanup you can defer).

## Harness-os pipeline
Use the harness-os MCP server and follow its pipeline for every unit of work:
Constitution → Specification → Test Suite → Generate Code → Code Review. Don't skip stages or generate code
before its Test Suite stage exists. Observed RED/GREEN comes from real test runs, never from a narrative claim
(CONST-AI-003).

## Code review
During the Code Review stage, review the uncommitted changes against both the plan
(`docs/OrbitGuard_replay_redesign_progress.md`) and the handoff with the discipline of an experienced QA/QC
engineer. Explicitly check correctness, performance and security before considering a step done — don't just
check that it runs. Security items specific to this project: no full application state on `window.*`
(CONST-SEC-003), no third-party script/style/font loaded at runtime without vendoring (all assets self-hosted),
no `innerHTML` with trace-derived text unless escaped, no secrets. Performance items: 3D redraw ≤ 33 fps, trace
payload size, no layout-triggering animation.

## Don't miss these — easy to under-implement in a large spec
1. **Real data only.** No literal episode numbers, seeds, "T+250", "3,728", "12.20" in UI code or assets; no
   imports of synthetic/demo modules; missing data is shown as "—" with a reason; shield-off interventions are
   "—", never 0. The guard test covers `.py`, `.js` and `.css` under `orbit_guard/visualization`.
2. **Only the two CSS files and the two Earth JPGs are copied from the handoff.** Dash auto-serves every file in
   `assets/`, so copying `orbitguard-data.js` or `support.js` would ship the synthetic data. A test asserts the
   assets folder contains neither.
3. **Episodes are discovered from `trace.meta`, not filenames;** a missing meta field fails loudly on load.
4. **One plotly.js (2.35.2, vendored, checksummed) owns every chart;** no `dcc.Graph` in the layout (Dash's own
   bundle is 4.1.1 and would conflict). Camera internals live in one small module with its own tests.
5. **Colour + marker shape + text label for every state**, including the ◆ (high risk) and ○ (inside d_safe)
   glyph changes; capability chips are neutral and dashed.
6. **Switching tab or shield keeps scenario, step, selected satellite, camera and play state.**
7. **The UI never claims training makes the fleet safer;** comparative copy is generated from data and omitted
   when its denominator is zero or a trace is missing.
8. **FR16 stays true:** the replay import closure contains no Ray, torch or trainer (existing subprocess test).
9. **Both densities:** BEFORE (every step is an event: grouping, histograms, matrix) and AFTER (sparse: IDs in lanes).
10. **Accessibility:** WCAG AA contrast from the token ratios, ≥ 44 px touch targets on phone, `role=slider` keys,
    visible focus ring, hover never the only carrier of information, reduced-motion = durations 0 + stepping.
11. **Where the design shows a value the trace does not contain: stop and ask** (or take the conservative
    option, record it as a decision, and flag it in the plan's open-questions table).

Update `docs/OrbitGuard_replay_redesign_progress.md` as you go throughout — it should always reflect current,
accurate status, not just a snapshot from when the plan was created.
