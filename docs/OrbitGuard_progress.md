# OrbitGuard — Implementation Progress & Plan

**Tracks against:** `docs/OrbitGuard_Final_PRD_v1.md` **v1.6** (confirmed via version line + §50 log, 7 entries, last entry re: real TLE snapshot format).
**Architecture:** `docs/OrbitGuard_Architecture.excalidraw` — **10 frames confirmed** (Core Pipeline · MAPPO/CTDE · RL Env Stack · Reward Function · Risk & Adaptive Margin · Safety Shield Algorithm · Experiment Design · Reference Config · Repository Module Map · 15-Week Timeline).
**Technical Guide:** `docs/OrbitGuard_Technical_Guide.md` — beginner's-guide/onboarding material (§47 onboarding note). Explanatory, not normative; does not add requirements beyond the PRD.

## 0. Tracker authority

The PRD (§44 preamble) states its own checkboxes are meant to be the single source of truth. This file, `docs/OrbitGuard_progress.md`, is the authoritative day-to-day tracker per the master implementation prompt. Rule to avoid dual-drift:

- Every milestone here carries a **stable ID** (`P{phase}-M{n}`) mapped 1:1 to a PRD §44 checkbox, so the mapping is mechanical, not interpretive.
- When a milestone here is marked done, its matching `docs/OrbitGuard_Final_PRD_v1.md` §44 checkbox should be checked in the same PR/commit. PRD checkboxes are the human-facing summary; this file carries the detail (tests, evidence, deviations).
- If the two ever disagree, this file's status + evidence log wins (it's updated more granularly), but the discrepancy must be logged under §7 (Deviations) below and reconciled same-day.

Owner tags reproduced **exactly as PRD v1.6 §44/§47 define them** (the master prompt's "Astro/RL/Eval/Joint" four-tag list is stale relative to v1.6, which split the old combined role into three — this file uses the six real tags):

`Astro` · `RL` · `Safety` · `Evaluation` · `Visualization` · `Joint`

Every table below adds an **"Agent coverage"** column: whether the implementing agent (this session) is covering a task outside its nominal PRD owner, per the master prompt's transparency requirement. Since no human team is actually staffed in this repository yet, the implementing agent will by default cover *all* owner tags — this is flagged structurally rather than noted once, because it is the common case, not the exception, until real owners are assigned.

---

## 1. Governance tooling — Harness OS (verified live)

`mcp__harness-os__*` is a real, running MCP server (docker container `harness-os:latest`, network `harness_os_default`, Postgres-backed), not a hypothetical from the master prompt. `get_constitution` was called against this project path on 2026-09-22 and returned **17 active rules**. The ones that gate implementation work directly:

| Rule | Severity | Enforcement | Implication for OrbitGuard |
|---|---|---|---|
| `CONST-CORE-001` | critical | gate | No implementation code in a `gatedGlobs` path without a **validated spec** covering it (`create_spec` → `validate_spec`), enforced at write time. |
| `CONST-CORE-002` | critical | gate | No implementation file until its test is **independently observed RED** by `gate-check --mode verify-red` (not self-reported), and not complete until GREEN is likewise observed. |
| `CONST-AI-003` | high | review | Claude must never assert "tests are RED/GREEN" as fact in a decision row — only gate-check's own observed exit code is authoritative. |
| `CONST-CORE-004` | critical | gate | `harness.config.json` / `enforce-gate.sh` / `settings.json` are checksummed at init; drift fails every gated write closed until `harness reconcile-config`. |
| `CONST-ARCH-003` | medium | review | 200–400 lines/file target, 800 hard cap — matches PRD §8 repo-structure intent and the user's own coding-style rules. |
| `CONST-CODE-001/002/003`, `CONST-SEC-001..005`, `CONST-ARCH-002` | med/high/critical | review | Standard immutability, nesting, secrets, input-validation rules — already covered by the user's global rules; no conflict. |

**Consequence for this project's workflow:** this satisfies (and hardens) the master prompt's "harness-os pipeline" requirement — Harness OS *is* configured, not a fallback case. Constitution → Specification → Test Suite → Generate Code → Code Review will use the real `create_spec`/`generate_tests`/`validate_coverage`/`record_decision` tools per implementation unit starting in Phase 1, not a manual substitute.

**Not yet verified:** whether `create_spec` can actually write in this environment (the docker mounts show `constitution/` and `specs/` mounted **read-only** from the host, but `create_spec` writes are described as going through the Postgres-backed registry, not the RO mount — these are likely different storage paths). This will be verified with the *first real spec* at the start of Phase 1 implementation (not probed here with a throwaway write, to avoid polluting the governance log before the team/user has approved starting). If it cannot write, fall back to a manual constitution/spec/test note per implementation unit and record the substitution here.

**Checked directly (2026-09-22): this repository is not yet harness-initialized.** `harness.config.json`, `enforce-gate.sh`, and `.claude/constitution/` do not exist anywhere in `/home/lehoa/projects/OrbitGuard`. Consequence: `CONST-CORE-001`/`CONST-CORE-002`'s local, fail-closed enforcement (`gatedGlobs`, `gate-check --mode verify-red`) has nothing to bind to here yet — there is no local hook that would currently block a gated write. This does **not** mean the discipline is optional: the intent behind those two critical-severity rules (spec-before-code, independently-observed RED before implementation, GREEN after) is followed manually starting Phase 1 regardless of whether local enforcement is wired up, using the real `create_spec`/`generate_tests`/`validate_coverage` MCP tools plus this project's actual test runner (`pytest`) as the RED/GREEN observation mechanism. If `harness init` is later run against this repo (a human/CLI action, not something this session does unprompted), local fail-closed enforcement becomes active and this section should be updated.

**TDD ordering implied by `CONST-CORE-002` (binding on how each milestone below is actually executed, even though the tables list "implementation" and "tests" as adjacent rows for readability):** every `P{n}-M{n}` implementation milestone expands to a five-step cycle at execution time — (1) spec drafted/registered, (2) test written, (3) test independently observed **RED** (running it and seeing it fail — not asserted from memory), (4) implementation written, (5) test independently observed **GREEN**. A milestone is never "done" because code was written and looked right; it's done when step 5 actually happened. This is the same non-negotiable ordering the user's own global TDD rules already require (`~/.claude/rules/ecc/common/testing.md`) — Harness OS's constitution reinforces it, doesn't add a new requirement.

**`CONST-AI-003` — authority for status claims in this document:** wherever this file marks a test as RED/GREEN/passing, the authority is the actual observed exit code of running the test (via `pytest`, `parallel_api_test`, etc.), never a narrative claim by the implementing agent. Non-test status claims (e.g., "file exists," "column matches") are backed by direct inspection (shown inline, e.g. §2's TLE verification), same as everywhere else in this document.

---

## 2. Repository state as of 2026-09-22 (updated: Phase 0 executed)

| Item | Status |
|---|---|
| Git repo | Scaffolded and committed (see commit referenced in §7 Phase 0 table below). `main` branch. |
| TLE snapshot | **Present, verified by direct inspection, and committed**: `data/TLE_snapshot_53deg_shell.csv`, 3,003 lines (1 header + 3,002 rows), 3,002 unique `norad_id` values, columns exactly `name,norad_id,epoch_year,epoch_day,inclination_deg,raan_deg,eccentricity,arg_periapsis_deg,mean_anomaly_deg,mean_motion_rev_day,semi_major_axis_km,altitude_km` — matches PRD §16/§39 `parsed_elements_csv` format exactly. Sample rows confirm ~53.0–53.1° inclination, ~460–464 km altitude. **Never re-pull; never write a raw-TLE parser against this file.** |
| PRD | v1.6, read in full, §50 design-decision log (7 entries) reviewed. |
| Architecture | 10 frames confirmed present and titled as expected. |
| Code (`orbit_guard/`, `configs/`, `tests/`, etc.) | **Scaffolded** — package directories + `__init__.py` per §8 module layout; no business logic yet (correctly none — Phase 1 hasn't started). |
| Python project tooling | `pyproject.toml` (hatchling build backend) + `uv` for env/dependency management (`uv sync` confirmed working, `.venv` created). Dependencies declared **incrementally by phase, not all up front** (YAGNI) — `numpy`, `scipy`, `pyyaml` + `pytest`/`pytest-cov` now; `gymnasium`/`pettingzoo`/`ray[rllib]` added at Phase 3, `dash`/`plotly` at Phase 6. |
| Provenance readiness | §40 logging requires `git_commit` per experiment. Initial commit made — see P0-M7 evidence below. |

**Phase 0 checklist — executed 2026-09-22:**

| PRD §44 Phase 0 item | Status | Evidence |
|---|---|---|
| Repo scaffolding matching §8 | **Done** | `orbit_guard/{physics,sensing,conjunction,safety,rl,scenarios,evaluation,logging,visualization}/` each with `__init__.py`; `configs/`, `configs/scenarios/`, `tests/`, `experiments/`, `artifacts/`, `scripts/` created. `.gitkeep` placeholders in empty output dirs. |
| Reference config `configs/default.yaml` | **Done** | Written verbatim from §39; parsed with `yaml.safe_load` — 14 top-level keys, matches §39 exactly, zero errors (`uv run python3 -c "yaml.safe_load(...)"`, 2026-09-22). `configs/train.yaml`/`validation.yaml`/`test.yaml` also created as honest placeholders (split identity only — no fabricated seed pools; those don't exist until Phase 4/5). |
| TLE snapshot frozen & committed | **Done** | Committed as-is, no modification; `.gitignore` excludes its Windows `:Zone.Identifier` sidecar file (not project content). |
| Team roles assigned (§47) | **N/A — human action** | No real multi-person team in this session; roles are pre-defined in §47 for a human team. Implementing agent covers all tags until told otherwise. |
| OSC compute allocation requested | **N/A — human action, cannot be done by the agent** | Flagging for the user; not blocking early solo/dev-scale work per §43 (RTX 4070 Ti is the primary dev path). |
| PRD reviewed & acknowledged by every team member | **Partially satisfied** — reviewed in full by the implementing agent (this document is that review's evidence); **human acknowledgment by the user/team is still outstanding**, same category as M4/M5 above. | Not yet closed; do not treat as done until the user confirms. |

**Exit criteria (§44) status:** repo exists ✓, config loads without error ✓ (verified above), TLE file committed ✓. "Every team member can state the current MVP scope (§3.1) from memory" is a human criterion — outstanding, same as PRD acknowledgment above. Phase 0 is **functionally complete on the engineering side**; the three human-only items (M4, M5, M6) remain open and don't block Phase 1 scaffolding-adjacent work, but should be closed by the user in parallel.

---

## 3. Open questions for Astro-physics relay (consolidated — do not trickle these out)

**Status update (2026-09-22): the user reviewed all five questions and instructed "use recommendation answer, proceed" — Phase 1 was implemented using the recommended answers below.** This is the user/team authorizing the project to proceed on these answers, not the astrophysics teammate's own sign-off — that distinction matters and is tracked, not glossed over. If the actual astrophysics teammate later reviews these and disagrees, the affected code (`physics/frames.py`'s frame convention, `physics/dynamics.py`'s constants, `scenarios/constellation.py`'s sampling) is isolated enough to revise without touching the rest of Phase 1. See §9 (Deviations) for exactly which recommended value was implemented where.

Per the master prompt: **pause and ask before assuming anything Astro-physics-related.** These are the concrete open items found while reading the PRD, batched here so they can be relayed to the astrophysics student in one pass rather than one at a time over multiple weeks:

1. **§39 numeric defaults are explicitly unvalidated** (§47.1: "my best guesses ... not validated physics") — needs sign-off or revision: position noise σ = 100 m, velocity noise σ = 0.1 m/s, `base_delta_v` = 0.1 m/s, hard-body radius = 5 m per satellite, `d_base`/`d_min`/`d_max` = 1000/200/5000 m, risk-proxy `z0`/`k` = 3.0/1.5.
2. **§9 ECI frame/epoch convention is unspecified — and this couples directly to constellation seeding.** PRD says only "must be documented in code" without naming one. The TLE-derived orbital elements in `data/TLE_snapshot_53deg_shell.csv` are TEME-referenced by origin (standard for TLE-derived mean elements). Astro needs to answer **both** parts together, not just "which frame the propagator uses": (a) EME2000/J2000 vs. a TEME-consistent internal frame, and the reference epoch convention, **and** (b) what conversion, if any, is applied when seeding satellite state from the CSV's TEME-origin elements into that chosen frame — otherwise the frame decision leaves the constellation generator (`P1-M5`) ambiguous even after §9 is nominally resolved.
3. **§23 RTN sign convention confirmation** — the mapping as written has `PHASE_FORWARD` = **−T** (retrograde) and `PHASE_BACKWARD` = **+T** (prograde), which reads inverted at first glance relative to the intuitive names. The PRD's own drift rationale (prograde burn → higher orbit → drifts *backward* relative to neighbors) explains it, but this needs explicit astro sign-off before coding `safety/maneuver_mapper.py`, not a silent "fix."
4. **§12 physics validation specifics** — which reference tool, Skyfield or poliastro (PRD lists both as options); what numerical tolerance counts as "agreed" for the propagator-vs-reference comparison; what J2 nodal-regression rate/direction is expected for the sanity check at this orbit's inclination/altitude.
5. **§32 shift-scenario #2 quantification** — "increased local density, N held fixed" is described qualitatively only. Needs a concrete physical definition (e.g., target spacing reduction factor, RAAN/phase clustering parameters) — PRD §44 Phase 5 explicitly assigns this definition to Astro.

**These block real Phase 1–2 physics coding**, not Phase 0 scaffolding — scaffolding, config file creation, and repo layout can proceed without them, but `physics/dynamics.py`, `physics/frames.py`, and `safety/maneuver_mapper.py` should not be written against a guessed answer.

---

## 4. Acceptance-test execution map (§38)

| Test | Requirement | Becomes executable at | Status | PRD-referenced exit criteria |
|---|---|---|---|---|
| **A — Benign orbit** | No-Maneuver baseline doesn't spontaneously collide, no conjunction injected | **Phase 1 exit** | **PASSES** — `tests/acceptance/test_acceptance_a_benign_orbit.py`, 2026-09-22 | §44 Phase 1 exit criteria |
| **B — Single conjunction** | Injected conjunction detected, meaningful risk signal | **Phase 2 exit** | Pending | §44 Phase 2 exit criteria |
| **C — Adaptive margin** | `d_safe` moves in expected direction within `[d_min,d_max]` as inputs change | **Phase 2 exit** | Pending | §44 Phase 2 exit criteria |
| **D — RL pipeline** | 2-satellite train→evaluate, no code changes between | **Phase 3 GATE** | Pending | §44 Phase 3 exit bullet |
| **E — Shield** | Unsafe proposal rejected/replaced when feasible alternative exists | **Phase 4** | Pending | §44 Phase 4 exit criteria |
| **F — Reproducibility** | Same config+seed → same deterministic output | **Phase 5** (co-owned RL+Evaluation) | Pending | §44 Phase 5 |
| **G — Distribution shift** | All 4 shift configs run from one experiment-config interface | **Phase 5** (Evaluation) | Pending | §44 Phase 5 |

---

## 5. §37 Testing-strategy → tracked tasks

| Test category | Tracked as | Phase |
|---|---|---|
| Physics (gravity, J2, RK4, frame transforms, conservation) | `P1-M4` | 1 |
| Geometry (distance, TCA, collision detection, boundaries) | `P2-M3`, `P2-M4` | 2 |
| Risk (monotonicity, boundedness, determinism) | `P2-M5` | 2 |
| Adaptive margin (non-decreasing w/ positive coeffs; bounded) | `P2-M6` | 2 |
| Environment (reset reproducibility, obs shape, action mapping, multi-agent sync) | `P3-M1` | 3 |
| **PettingZoo `parallel_api_test` + seed test** | **`P3-M2` — own task, not folded into environment tests** | 3 |
| Shield (capability rejection, budget respect, unsafe-detection, determinism, repair) | `P4-M2`, `P4-M3` | 4 |

---

## 6. Research-integrity checklist (tracked continuously, not phase-local)

| Requirement | Tracked in | Structural protection plan |
|---|---|---|
| Train/validation/test separation | `P0-M2` (config), `P5-M1` | Validation = 50 fixed seeds; test = 10×5=50 held-out seeds (§33), stored in a **separate seed-pool file** training/tuning code never imports. |
| Frozen TLE snapshot | Verified §2 above | Loader must reject any path other than `data/TLE_snapshot_53deg_shell.csv`; no CelesTrak network call anywhere in the codebase. |
| Deterministic scenario seeds | `P2-M7` | Scenario generator takes an explicit seed; no hidden global RNG state. |
| Matched seeds across baselines | `P5-M2` | Evaluation runner takes the seed list as a shared input, not per-baseline. |
| 3 training seeds / learned config | `P4-M1`, `P5-M2` | Enforced by evaluation-runner config schema (reject configs missing seed triples). |
| 10 evaluation seeds / model / condition | `P5-M2` | Same. |
| 50 validation seeds | `P4-M4` | Fixed file, generated once, checksummed. |
| 50 held-out final-test scenarios | `P5-M2` | Fixed file, generated once, **not readable by tuning code path** (separate module/import boundary). |
| Zero-shot distribution shift | `P5-M3` | No fine-tuning step exists in the shift-eval code path at all (not just "unused"). |
| Freeze tuned params before final test | `P4-M4` exit, `P5-M2` entry | Config-hash snapshot taken and logged before final-test sweep begins. |
| Reproducibility from config+seed+commit | `P0-M1`, `P2-M2`(logging schema) | §40 fields logged every run. |
| Logged experiment provenance | `P0-M6` (logging schema scaffold), all phases | Every phase's runs go through the same logger. |
| No single training run as primary result | `P5-M2` reporting | Statistics pipeline (`P5-M5`) requires ≥3 seeds before producing a "headline" table. |

---

## 7. Phase tables

Legend for **Agent coverage**: `own` = implementing agent covers its nominal PRD tag; `covering:<Tag>` = implementing agent is doing work nominally owned by a different tag (flagged per master-prompt instruction, not hidden).

### Phase 0 — Setup & Freeze (PRD Weeks 1) — status: **engineering items done 2026-09-22; human items outstanding**

| ID | Task | Owner | Agent coverage | Tests/Validation | Dependencies | Artifacts | PRD ref | Status |
|---|---|---|---|---|---|---|---|---|
| P0-M1 | Repo scaffolding matching §8 module layout | RL | covering:RL | Directory structure matches §8 exactly (verified by `find` listing) | none | `orbit_guard/{physics,sensing,conjunction,safety,rl,scenarios,evaluation,logging,visualization}/` (each with `__init__.py`), `configs/`, `configs/scenarios/`, `tests/`, `experiments/`, `artifacts/`, `scripts/`; `pyproject.toml`, `.gitignore`, `uv.lock` | §8, §44 | **Done** |
| P0-M2 | Reference config committed as `configs/default.yaml` | RL | covering:RL | `yaml.safe_load` succeeds, 14 top-level keys match §39 verbatim (checked 2026-09-22) | P0-M1 | `configs/default.yaml`, plus `configs/{train,validation,test}.yaml` as honest split-identity placeholders (no fabricated seed pools) | §39 | **Done** |
| P0-M3 | TLE snapshot frozen & git-committed | Astro | covering:Astro (data already verified, §2) | `git log` shows file added in initial commit, content byte-identical to pre-commit inspection | none | committed `data/TLE_snapshot_53deg_shell.csv` | §16, §44 | **Done** |
| P0-M4 | Team roles assigned (§47) | Joint | **N/A — human action** | — | — | — | §47 | **Outstanding (human)** |
| P0-M5 | OSC compute allocation requested | RL | **N/A — human action, flagged to user** | — | — | — | §48 | **Outstanding (human)** |
| P0-M6 | PRD + §50 log reviewed/acknowledged by team | Joint | own (agent review complete; human ack still pending — see §2) | This file exists and cites §50 | none | this file | §44, §50 | **Partial (agent done, human outstanding)** |
| P0-M7 *(added, not in §44 but load-bearing per §40)* | Initial git commit establishing provenance baseline | Joint | covering:Joint | `git log` non-empty, `HEAD` short hash recorded below | P0-M1..M3 | initial commit `1c3c67d` (root commit, 27 files) | §40 (implied) | **Done** |

**Exit criteria (§44):** repo exists ✓, config loads without error ✓, TLE file committed ✓, every team member can state MVP scope (§3.1) from memory — **human criterion, outstanding**. Engineering exit criteria met 2026-09-22; human criterion (M4/M5/M6/implicit-M-in-exit-criteria) remains open pending user action, per §2.

---

### Phase 1 — Physics Foundation (Weeks 2–3) — status: **core milestones done 2026-09-22**

Astro questions Q1–Q5 (docs/OrbitGuard_progress.md §3, superseded below) were resolved by adopting the user-approved recommended answers (2026-09-22), not left open — see §9 Deviations for the exact substitutions used.

| ID | Task | Owner | Agent coverage | Tests/Validation | Dependencies | Artifacts | PRD ref | Status |
|---|---|---|---|---|---|---|---|---|
| P1-M1 | Two-body + J2 acceleration model | Astro | covering:Astro | 12 tests: inverse-square law, independent reformulation of the J2 formula, pole/equatorial symmetry, fail-loud on non-finite/zero position | P0, Astro Q2 (resolved) | `orbit_guard/physics/dynamics.py`, `tests/physics/test_dynamics.py` | §10 | **Done** |
| P1-M2 | RK4 integrator, configurable `dt` | Astro | covering:Astro | Convergence vs. analytic circular orbit (errors shrink ~16x per dt-halving, 4th-order); fail-loud on non-finite input/output | P1-M1 | `orbit_guard/physics/integrator.py`, `tests/physics/test_integrator.py` | §11, §12 | **Done** |
| P1-M3 | ECI↔RTN frame transforms + COE↔Cartesian | Astro | covering:Astro | Kepler-equation solve, COE round-trip (5 mean-anomaly + 1 quadrant-edge case), RTN orthonormality/round-trip, fail-loud on near-equatorial/near-circular/near-parabolic | Astro Q2 (resolved) | `orbit_guard/physics/frames.py`, `tests/physics/test_frames.py` | §9 | **Done** |
| P1-M4 | Physics validation suite (§12), all 5 required checks + fail-loud | Astro | covering:Astro | Two-body circular sanity; energy + angular-momentum conservation (two-body-only); **J2 nodal regression measured at ≈-4.7°/day, matches analytic secular formula within 2%** (Astro Q4 rate); **Skyfield SPICE-based `keplerlib.propagate` reference comparison** (Astro Q4 tool), elliptical case, within 10m/0.01 m/s; convergence of full two-body+J2 dynamics vs. fine-dt reference | P1-M1..M3 | `tests/physics/test_validation.py` | §12 | **Done** |
| P1-M5 | Constellation generator, N=10–20 from TLE distributions | Astro | covering:Astro | Loads real 3,002-row snapshot; fitted distribution matches PRD-documented ranges (within rounding — see §9); N-range boundary rejection; deterministic under seed; differs across seeds; sampled altitude stays within shell envelope | Data (§2), P1-M1, P1-M3 | `orbit_guard/scenarios/constellation.py`, `tests/scenarios/test_constellation.py` | §16–17 | **Done** |
| P1-ACC-A *(added)* | Acceptance Test A: benign, no-conjunction, No-Maneuver episode never collides | Astro | covering:Astro | Full N=15, 6h episode horizon (`physics_dt`=10s), pairwise min-separation check against combined hard-body radius | P1-M1..M5 | `tests/acceptance/test_acceptance_a_benign_orbit.py` | §38 (Test A) | **Done — passes** |
| P1-M6 *(parallel track)* | RL: learn PettingZoo/RLlib on a toy env (not OrbitGuard yet) — no repo artifact expected | RL | — | N/A (learning task) | none | none | §47.7 | **Deferred** — no benefit to a solo-agent implementation; will build real PettingZoo env directly in Phase 3 |
| P1-M7 *(parallel track)* | Safety: shield repair-search logic against mocked risk data | Safety | — | — | mock risk schema (§24) | `safety/shield.py` (skeleton) | §47.7 | **Deferred to Phase 4** — building against mocks now would be throwaway work with no separate human owner to unblock |
| P1-M8 *(parallel track)* | Evaluation: scaffold baseline runner + seed-pool separation tooling against mocked training outputs | Evaluation | — | — | mock schema | `evaluation/` skeleton | §47.7 | **Deferred to Phase 5** |
| P1-M9 *(parallel track)* | Visualization: Dash app shell against hand-written mock episode file | Visualization | — | — | mock episode file (§40 schema) | `visualization/` skeleton | §47.7 | **Deferred to Phase 6** |

**Exit criteria (§44):** Acceptance Test **A** passes ✓; propagator matches reference tool (Skyfield) within agreed tolerance ✓ (10m / 0.01 m/s over 6000s for an elliptical two-body case — tight relative to LEO scales). **Phase 1 exit criteria met.** Full suite: **61 tests, 100% line coverage on all Phase 1 modules, `ruff check` clean.**

**Note on P1-M6–M9 (parallel tracks):** the PRD's parallel-track guidance in §47.7 assumes a 5-person human team where idle leads build against mocks while Astro works. This implementation has one agent covering all owner tags, so building throwaway mock-based scaffolding for Safety/Evaluation/Visualization now — only to replace it with real integrations in Phases 4–6 — has no parallelism benefit and would be speculative work against interfaces that may still shift. Deferred to the phase where each becomes load-bearing, flagged here rather than silently dropped.

---

### Phase 2 — Sensing, Conjunction & Risk (Weeks 4–6)

**Team-capacity decision (2026-09-27): Phase 2 is split into two parallel, independently-implementable tracks** — a real-world constraint, not a PRD-driven one. A second team member has Antigravity Pro but not Claude access, so Phase 2's Astro-tagged work is divided between this session ("Part A") and that teammate's Antigravity agent ("Part B"), working from a spec-and-tests handoff rather than shared conversational context. This is a local tracker decision, not a PRD role change — §47's owner tags are unaffected; this is sub-splitting within the "Astro" tag for practical delivery reasons, recorded here per the phase-gate-approval workflow's "make the split visible, don't silently reassign" principle.

**Split rationale:** Part A keeps the pieces that are architecturally load-bearing or research-critical and benefit from continuity with this session's established codebase context — the ground-truth/observation leakage boundary (§14, one of the project's explicit highest-risk items) and the adaptive margin (the paper's core contribution #1). Part B gets the pieces that are well-specified, self-contained pure functions/geometry problems, decidable from a frozen interface contract without deep project history — exactly the shape of task that hands off cleanly to a fresh agent in a different tool. The two tracks share almost no code-level dependency: Part B's modules take plain floats/NumPy arrays in formats Phase 1 already produces (`propagate()`'s trajectory shape); Part A's modules don't call Part B's at all until final integration.

#### Part A — this session (Claude)

| ID | Task | Owner | Tests/Validation | Dependencies | Artifacts | PRD ref | Status |
|---|---|---|---|---|---|---|---|
| P2A-M1 | Gaussian telemetry/uncertainty model | Astro (Part A) | Sampled noise matches configured σ statistically | P1 | `sensing/noise.py` | §15 | Pending |
| P2A-M2 | Ground-truth/observation/belief/prediction/eval-truth separation as distinct code paths | Astro (Part A) | Leakage test: assert policy-facing objects never reference ground-truth object | P1, P2A-M1 | `sensing/belief.py`, module boundary | §14 | Pending |
| P2A-M3 | Adaptive margin `d_safe(i,j)` | Astro (Part A) | Bounds `[d_min,d_max]`; symmetry `d_safe,ij == d_safe,ji`; non-decreasing in α/β/γ inputs; processing-order test (`d_safe` never feeds its own risk calc) | Part B's `risk.py` (interface only — a plain float risk score) | `safety/adaptive_margin.py` | §22 | Pending |
| P2A-M4 | Conjunction injection / scenario generator randomization | Astro (Part A) | Deterministic under seed; produces benign/single/multi mix per §17 | P1-M5 | `scenarios/conjunction_generator.py` | §17 | Pending |
| P2A-M5 | Acceptance Test C (adaptive margin) | Astro (Part A) | `d_safe` moves in expected direction within bounds as inputs change | P2A-M3 | test | §38 | Pending |

**Awaiting explicit go-ahead to start Part A implementation** (per the standing phase-gate-approval workflow — this session hasn't been told to begin writing Part A's production code yet, only to prepare Part B's handoff).

#### Part B — Antigravity (other teammate)

| ID | Task | Owner | Tests/Validation | Dependencies | Artifacts | PRD ref | Status |
|---|---|---|---|---|---|---|---|
| P2B-M1 | Candidate pair screening | Astro (Part B) | Exhaustive N(N-1)/2 pairs, unique, `i<j`, rejects N<2 | P1 (none beyond it) | `conjunction/screening.py` | §20 | **Spec + RED tests done** |
| P2B-M2 | TCA + minimum-separation computation | Astro (Part B) | Closed-form linear-motion test cases (exact + between-sample), boundary case, shape-mismatch guard, real-orbit smoke test | P1 | `conjunction/tca.py` | §19 | **Spec + RED tests done** |
| P2B-M3 | Collision detection at ground truth, boundary conditions | Astro (Part B) | Exact-boundary (`‖r_i−r_j‖ = R_i+R_j`, strict `<`) unit test; first-collision-time test; real-orbit integration test | P1 | `conjunction/collision.py` | §18 | **Spec + RED tests done** |
| P2B-M4 | Risk proxy `f(z)` | Astro (Part B) | Monotonicity + boundedness (closed `[0,1]`, incl. floating-point-underflow saturation) property tests; never emits/labels `P(collision)`; numerically stable at extreme inputs | none (pure function of floats) | `conjunction/risk.py` | §21 | **Spec + RED tests done** |
| P2B-M5 | Acceptance Test B (partial — TCA+risk half) | Astro (Part B) | Hand-constructed synthetic conjunction produces a meaningful risk signal | P2B-M1..M4 | test (own fixture, not P2A-M4's injector) | §38 | Pending (Antigravity) |

**Handoff artifacts (committed this session, before any go-ahead was needed since this is spec/test work, not Part A implementation):**
- `tests/conjunction/{test_screening,test_tca,test_collision,test_risk}.py` — 26 tests, confirmed RED (clean `ModuleNotFoundError`, nothing else), and separately **verified correct** by writing and then discarding a throwaway reference implementation that passed all 26 before handoff (not committed — see §9 deviations log).
- `docs/OrbitGuard_Phase2_Antigravity_Brief.md` — the full handoff document (scope, conventions, running instructions, definition of done) plus a copy-pasteable Antigravity-ready prompt.
- One real bug caught during self-verification: a naive `1/(1+exp(-x))` sigmoid overflows for realistic well-separated-satellite inputs (`d_min=5000m, sigma_rel=0.1m` → `z≈4545`) — the brief explicitly requires `scipy.special.expit` instead, and a test enforces the resulting numerical stability.

**Integration note (applies once both parts land):** P2B's `risk_proxy` takes plain floats (`d_min_m`, `sigma_rel_m`); P2A's belief module (P2A-M2) is responsible for producing `sigma_rel_m` from covariances and handing it to P2B's function — that boundary is the *only* place the two tracks actually touch, and it's a one-line call, not a shared abstraction. P2A's `scenarios/conjunction_generator.py` (P2A-M4) will also eventually call P2B's `tca.py`/`collision.py` to verify injected scenarios behave as intended — again a plain function call across the boundary, no shared state.

**Exit criteria:** Acceptance Tests **B** and **C** pass (B needs both parts integrated for the full injected-scenario version; a partial version is achievable by Part B alone against a hand-built fixture, per P2B-M5).

---

### Phase 3 — RL Environment & 2-Agent Sanity Check — **GATE** (Weeks 7–8)

| ID | Task | Owner | Tests/Validation | Dependencies | Artifacts | PRD ref |
|---|---|---|---|---|---|---|
| P3-M1 | `OrbitGuardEnv` as PettingZoo `ParallelEnv` (not AEC), Gymnasium-typed: obs/action/reward | RL | Reset reproducibility, obs-shape consistency, action-mapping correctness, multi-agent sync | Phase 2 outputs (real, not mocked) | `rl/environment.py` | §23–26, §28 |
| P3-M2 | PettingZoo `parallel_api_test` + seed/reproducibility test against `OrbitGuardEnv` | RL | Off-the-shelf conformance suite passes | P3-M1 | test log | §28, §37 |
| P3-M3 | Deterministic maneuver-intent → Δv mapper | Safety | Unit test every one of 6 actions against RTN convention (**pending Astro Q3 sign-off**) | Astro Q3 resolved | `safety/maneuver_mapper.py` | §23 |
| P3-M4 | MAPPO/CTDE training pipeline in RLlib | RL | **Verify RLlib's current recommended custom-`ParallelEnv` integration pattern against live docs at implementation time** (do not copy a cached tutorial); record which pattern used + why, here | P3-M1 | `rl/training.py` | §28 |
| P3-M5 | 2-satellite training run converges to sensible, inspectable behavior | RL | Manual inspection + reward-curve sanity | P3-M3, P3-M4 | training logs/checkpoint | §44 |
| P3-M6 | Acceptance Test **D** | RL | train→evaluate, zero code changes between | P3-M5 | eval report | §38 |

**GATE rule (verbatim from PRD):** do not scale to N=15 merely because the code technically supports more agents. If 2-agent gate is not clean by end of this phase, invoke §46 (scope cuts), stop scaling, diagnose, stabilize 2-agent first — **before requesting approval to enter Phase 4.**

---

### Phase 4 — Scale to Full MVP + Safety Shield (Weeks 9–10)

| ID | Task | Owner | Tests/Validation | Dependencies | Artifacts | PRD ref |
|---|---|---|---|---|---|---|
| P4-M1 | Scale env/training to N=10–20, N=15 primary | RL | Trains without crash across ≥3 seeds | Phase 3 GATE cleared | training runs | §44 |
| P4-M2 | Joint-action safety shield, all 7 steps | Safety | Capability validation, short-horizon sim, safety checks, accept/repair | P3-M3, P2-M6 | `safety/shield.py` | §29 |
| P4-M3 | Shield repair search: exact enum ≤4 agents, beam search width=50 otherwise | Safety | Threshold=4 boundary test; beam width exactly 50 (not approximate); deterministic tie-break test | P4-M2 | `safety/shield.py` (repair module) | §29 |
| P4-M4 | Acceptance Test **E** | Safety | Unsafe proposal rejected/repaired when feasible alt exists | P4-M2, M3 | test log | §38 |
| P4-M5 | Coarse grid search: reward weights + margin coefficients on **validation split only** | RL | Structural check: grid search code path cannot import test-seed file | Validation seed pool (§33) | tuning logs, frozen config | §25, §33 |

**Exit criteria:** N=15 shielded MAPPO trains end-to-end, ≥3 seeds, no crash; shield intervention rate logged and plausible (not ~0%, not saturating).

**Training-time shield note (hard requirement, tracked explicitly):** for Adaptive/Fixed-Margin MAPPO, shield stays active during training — realized transition = shield-approved action, reward computed against raw proposal for safety/coordination terms. Adaptive-MAPPO-no-shield is a **fully separate training run** (`P6-M1`), never an eval-time toggle. Tests must prove this distinction (`P4-M6`, added):

| P4-M6 *(added)* | Test proving shield-active-training vs. no-shield-training are structurally different code paths, not a flag flip on eval | RL | Assert no-shield config path never calls shield module during training | P4-M1, P4-M2 | test | §28 |

---

### Phase 5 — Baselines & Core Experiments (Weeks 11–13)

| ID | Task | Owner | Tests/Validation | Dependencies | Artifacts | PRD ref |
|---|---|---|---|---|---|---|
| P5-M1 | All 5 baselines runnable from config alone | Evaluation | Baseline switch via config diff only, no code edit | Phase 4 pipeline | `evaluation/baselines.py` | §31, FR14 |
| P5-M2 | Adaptive-vs-Fixed-Margin comparison, full 3-seed protocol | Evaluation (+ RL pipeline) | Paired stats across matched seeds | P5-M1 | results table | §44 |
| P5-M3 | All 4 core distribution shifts, zero-shot, from config | Evaluation (+ Astro defines shift params) | No fine-tuning code path exists | Astro Q5 resolved | `evaluation/shifts.py` | §32, FR15 |
| P5-M4 | Full seed protocol: 3 training × 10 eval × 5 conditions | Evaluation | Episode count matches exactly 150 (learned) / 50 (non-learned) per §33 | P5-M1..M3 | experiment artifacts | §33 |
| P5-M5 | Acceptance Tests **F**, **G** | RL+Evaluation (F), Evaluation (G) | Reproducibility check; single-config-interface shift run | P5-M1..M4 | test logs | §38 |
| P5-M6 | Metrics/statistics pipeline: mean/SD/CI, degradation scores | Evaluation | Requires ≥3 seeds before "headline" table generated | P5-M4 | `evaluation/statistics.py` | §34 |

**Exit criteria:** headline adaptive-vs-fixed result exists with paired stats, reproducible from fresh checkout.

---

### Phase 6 — Ablations, Visualization & Analysis (Week 14)

| ID | Task | Owner | Tests/Validation | Dependencies | Artifacts | PRD ref |
|---|---|---|---|---|---|---|
| P6-M1 | Shield-vs-no-shield ablation (separate training run) | RL (train), Evaluation (compare), Safety (interpret) | Confirms P4-M6 distinction empirically | P4-M6 | ablation report | §28, §46 |
| P6-M2 | Dash/Plotly visualization; standalone replay (FR16) | Visualization | **Playwright MCP inspection of live rendered app**, not code-only review; replay works with RL trainer stopped | Phase 5 logged episodes | `visualization/app.py` | §41 |
| P6-M3 | Failure-mode taxonomy (§36) applied to logged episodes | Evaluation (reviewed by Astro) | All 9 categories machine-readable, not collapsed to one flag | P5 artifacts | tagged episode logs | §36 |
| P6-M4 | Result tables/figures generated from logged data, not hand-typed | Evaluation | Script/notebook reads experiment DB/files directly | P5-M6 | figures/tables | §49 |
| P6-M5 *(added)* | Headless replay integration test (produce → save → load → replay, no training/inference required) | Visualization | Full round-trip test | P6-M2 | `tests/test_replay.py` | (master prompt "Headless Replay Test") |

---

### Phase 7 — Paper, Poster & Buffer (Week 15)

| ID | Task | Owner | Tests/Validation | Dependencies | Artifacts | PRD ref |
|---|---|---|---|---|---|---|
| P7-M1 | Paper draft — claims bounded by §51 | Joint | Language audit: no "operational/flight-qualified/formal-guarantee" claims | P5, P6 results | paper draft | §51 |
| P7-M2 | Poster/demo | Joint | — | P6-M2 | poster | — |
| P7-M3 | Rerun any Phase 5–6 result that looked off | Joint (module owner) | Confirmed, not silently left | — | rerun logs | §44 |
| P7-M4 | Every §49 Definition-of-Done item checked | Joint | Full checklist pass | all phases | this file, §8 below | §49 |

---

## 8. "Do Not Miss These" — high-risk checklist (master-prompt §, tracked as one anchored list)

Status column starts `pending` for all; updated in place as each becomes verifiably true (not "implemented," **verified**).

| # | Item | Status | Verified by (phase) |
|---|---|---|---|
| 1 | Ground truth never leaked into execution-time policy input | pending | P2-M2 leakage test |
| 2 | Prediction uses estimated state, not ground truth | pending | P2-M2 |
| 3 | TCA/min-separation logic identical across baselines | pending | P5-M1 |
| 4 | Risk is bounded proxy, never called P(collision) | pending | P2-M5 |
| 5 | `d_safe` pairwise and symmetric | pending | P2-M6 |
| 6 | `d_safe` calculated after risk, not before | pending | P2-M6 |
| 7 | `d_safe` bounded `[d_min,d_max]` | pending | P2-M6 |
| 8 | Adaptive-margin coeffs tuned only on train/val | pending | P4-M5 |
| 9 | RTN mapping matches frozen convention | pending | P3-M3 (post Astro Q3) |
| 10 | All agents act simultaneously | pending | P3-M1 |
| 11 | Shield evaluates joint action, not isolated | pending | P4-M2 |
| 12 | Repair search bounded and deterministic | pending | P4-M3 |
| 13 | Exact enumeration at specified threshold (≤4) | pending | P4-M3 |
| 14 | Beam width exactly 50 | pending | P4-M3 |
| 15 | `NO_MANEUVER` never automatic fallback | pending | P4-M3/M4 |
| 16 | No safety-based action masking in main experiments | pending | P3-M1, P5-M1 |
| 17 | Shielded MAPPO uses shield during training | pending | P4-M6 |
| 18 | No-shield MAPPO is separate training run | pending | P4-M6, P6-M1 |
| 19 | All 5 baselines config-driven | pending | P5-M1 |
| 20 | All 4 distribution shifts zero-shot | pending | P5-M3 |
| 21 | Density shift keeps N fixed | pending | P5-M3 (Astro Q5) |
| 22 | Test seeds never influence tuning | pending | P4-M5, §6 structural protection |
| 23 | Matched evaluation seeds reused across methods | pending | P5-M2 |
| 24 | All 3 training seeds executed per learned config | pending | P5-M4 |
| 25 | Reported results include seed counts + statistics | pending | P5-M6 |
| 26 | Experiment artifacts contain commit/config/seed provenance | pending | P0-M7, §40 schema |
| 27 | Failure-mode classifications logged | pending | P6-M3 |
| 28 | Dash replay works without live training/inference | pending | P6-M2, P6-M5 |
| 29 | UI distinguishes states via color + label/marker | pending | P6-M2 |
| 30 | Paper language avoids operational/formal-guarantee claims | pending | P7-M1 |
| 31 | Frozen TLE data never silently re-pulled | **verified — no network TLE code exists; will re-check at each phase touching `scenarios/`** | ongoing |
| 32 | Primary benchmark remains N=15 within N=10–20 | pending | P4-M1 |
| 33 | Physics/safety modules don't import RL | pending | ongoing import-boundary check every phase |
| 34 | Final tables/figures generated from logged data | pending | P6-M4 |
| 35 | TLE loaded from provided CSV directly, no raw-TLE parsing | **verified — format confirmed in §2 above** | Phase 1 loader impl. will re-confirm |
| 36 | `OrbitGuardEnv` is PettingZoo `ParallelEnv`, RLlib pattern verified live at impl. time | pending | P3-M4 |

---

## 9. Deviations / decisions log (this implementation, distinct from PRD §50)

| Date | Decision | Rationale |
|---|---|---|
| 2026-09-22 | Master prompt's four-tag owner scheme (`Astro/RL/Eval/Joint`) superseded by PRD v1.6's six-tag scheme (`Astro/RL/Safety/Evaluation/Visualization/Joint`) in this tracker | Master prompt predates PRD v1.6's role split (§47); PRD is source of truth per its own precedence rule |
| 2026-09-22 | Did not probe `create_spec` write capability yet | Would create a persistent governance-log row before implementation is approved; deferred to first real spec at Phase 1 start |
| 2026-09-22 | No repo scaffolding, config, or commits made yet despite being "Phase 0" tasks | Master prompt requires pausing for approval before moving into a phase; this document is the Step-1 planning artifact, not Phase 0 execution |
| 2026-09-22 | User approved Phase 0; executed repo scaffolding, `configs/default.yaml` (+ split placeholders), TLE commit, `pyproject.toml`/`uv` env, `.gitignore`. Initial commit `1c3c67d` (root commit, 27 files). | Direct user instruction "Proceed to phase 0." All engineering §44 Phase 0 items done same-day; human-only items (M4/M5, M6 human half) remain open and flagged, not silently closed. |
| 2026-09-22 | Chose `uv` (not pip/poetry) as the Python project/dependency manager | Already present in the environment (`~/.local/bin/uv`), no `pip` binary available; matches CONST-ARCH-002's preference for existing, battle-tested tooling over inventing a new setup |
| 2026-09-22 | `pyproject.toml` declares only `numpy`/`scipy`/`pyyaml`/`pytest`/`pytest-cov` — not the full eventual stack (`gymnasium`, `pettingzoo`, `ray[rllib]`, `dash`, `plotly`) | YAGNI (user's global coding-style rule): those aren't needed until Phase 3 (RL) and Phase 6 (visualization) respectively; adding them now would be speculative and untested against actual usage |
| 2026-09-22 | User instructed "Use recommendation answer, proceed with phase 1." Implemented all five recommended Astro answers from §3 as Phase 1 design choices, not just informational text. | Direct user authorization. Q1 (numeric defaults): kept §39 defaults unchanged in `configs/default.yaml`. Q2 (frame): `physics/frames.py` defines "OG-ECI" as whatever frame `coe_to_cartesian` produces (TEME-equivalent), with **no** epoch-based propagation or TEME/J2000 conversion anywhere — satellites are seeded fresh from sampled elements at simulator t=0. Q3 (RTN sign convention): not yet encoded (that's `safety/maneuver_mapper.py`, Phase 3/`P3-M3`) — recommendation recorded here for when that module is built. Q4 (validation tool/tolerance): used Skyfield's `keplerlib.propagate` (SPICE `prop2b.f`-based) for the two-body reference check, and the independent closed-form analytic J2 secular nodal-regression formula for the J2 check — not the same tool for both, per the recommendation's reasoning that SGP4-wrapping tools model extra physics OrbitGuard doesn't. Q5 (density-shift definition): not yet implemented (Phase 5 scope); recommendation recorded for reuse then. |
| 2026-09-22 | Registered harness-os domain spec `ORBG-DOM-001` (SatelliteState/SatelliteCapability/OrbitalElements) before writing Phase 1 code; confirmed `create_spec` **can** write in this environment (spec id 244) — resolves the write-capability question deferred from Phase 0. Logged via `record_decision` (id 1861). | CONST-CORE-001. The domain-spec schema (entities/fields) has no way to express numerical/algorithmic behavior (acceleration formulas, RK4, frame transforms) — that correctness contract is carried entirely by the test suite instead, written test-first per CONST-CORE-002. This is a genuine schema-fit limitation, not a corner cut. |
| 2026-09-22 | Followed real RED→GREEN TDD for every new module (`state.py`, `dynamics.py`, `integrator.py`, `frames.py`, `constellation.py`): wrote the test file, ran `pytest` to confirm the exact expected failure (`ModuleNotFoundError`/`ImportError`), then implemented, then reran to confirm pass. | CONST-CORE-002 discipline, followed manually since this repo isn't harness-initialized locally (§1) — `pytest`'s own exit code is the RED/GREEN authority (CONST-AI-003), not a self-asserted claim. |
| 2026-09-22 | Found and fixed two bugs **in the test suite itself**, not in the physics implementation: (1) an RK4 "one-period" convergence test compared against the wrong target because `round(period/dt)` doesn't land exactly on one period, making rounding error dominate; fixed by comparing against the exact analytic circular-orbit position at a fixed, evenly-divisible time instead. (2) the J2 nodal-regression test compared `raan_final - raan0` without unwrapping, and RAAN (returned mod 2π) wrapped through 0 during the ~47° expected drift, producing a wildly wrong 31°/day instead of -4.7°/day; fixed with a shortest-angular-difference unwrap. | Caught by running tests and reading the actual failure numerically (31.3 vs -4.7 has the wrong sign *and* magnitude, which doesn't fit "numerical noise") rather than loosening tolerances to force a pass. Recorded because both are the kind of subtle test-design bug that would otherwise silently under-verify a real physics property. |
| 2026-09-22 | `fit_shell_distribution`'s test asserts against the real CSV's exact min/max with a small tolerance, not PRD §16's literal stated bounds (which said "53.04-53.17°" / "460-465 km") | The real data's tails slightly exceed the PRD's rounded prose (observed inclination min 53.0355°, altitude max 465.30 km). The CSV is ground truth (verified in Phase 0, §2); PRD prose is a rounded human-readable summary of it. No data or code issue — a documentation-precision note worth surfacing, not silently absorbing. |
| 2026-09-22 | Added `ruff` as a dev dependency and ran `ruff check` over all Phase 1 code; fixed all 7 findings (unquoted forward-refs now that `from __future__ import annotations` is used, `Callable` import source, an unnecessary `int(round(...))`, an unused import, two blind `except Exception` assertions narrowed to `dataclasses.FrozenInstanceError`) | Matches the user's Python coding-style rule (ruff for linting) and the general code-quality checklist; all fixes were style/precision, not behavior changes — confirmed by rerunning the full suite green after each. |
| 2026-09-22 | Deferred P1-M6 through P1-M9 (the "parallel track" tasks for RL/Safety/Evaluation/Visualization leads building against mocks while Astro works) rather than building mock-based scaffolding now | These tasks exist in the PRD to keep a 5-person human team from being idle while Astro works. A single implementing agent has no idle-parallelism benefit from building throwaway mocks now, only to replace them with real integrations in Phases 3-6 — that's speculative work against interfaces likely to shift (YAGNI). Deferred explicitly to the phase each becomes load-bearing, not silently dropped. |
| 2026-09-27 | Split Phase 2 into Part A (this session) and Part B (a second teammate's Antigravity agent), per direct user instruction; this session chose the split | User: a second CSE teammate has Antigravity Pro, not Claude access, and wants to parallelize Phase 2. Split rule: Part A keeps architecturally load-bearing/research-critical work requiring continuity with this session's context (§14 leakage boundary, §22 adaptive margin — the paper's core contribution); Part B gets self-contained pure-function/geometry modules (TCA, collision, risk, screening) fully specified by a frozen interface (plain floats/arrays, no dependency on Part A's internals). See updated Phase 2 tables above. |
| 2026-09-27 | Wrote and committed Part B's full test suite (`tests/conjunction/{test_screening,test_tca,test_collision,test_risk}.py`, 26 tests) as the executable spec, confirmed clean RED (`ModuleNotFoundError` only) | Operationalizes CONST-CORE-002's test-first discipline across a tool boundary (a different agent, in a different tool, without this conversation's context) — an executable spec removes ambiguity a prose handoff would leave open. |
| 2026-09-27 | Verified all 26 Part-B tests against a throwaway reference implementation before handoff, then deleted that implementation (never committed) | Caught two real bugs before they could waste the other student's time: (1) `test_risk_proxy_is_bounded_in_zero_one` originally asserted a strict open interval `0<r<1`, which is stricter than the PRD's actual closed-interval `[0,1]` spec and would fail even a numerically-correct implementation that legitimately saturates via floating-point underflow at extreme inputs — loosened to match the PRD. (2) confirmed a naive `1/(1+exp(-x))` sigmoid raises `OverflowError` for realistic well-separated-satellite inputs (`d_min=5000m, sigma_rel=0.1m` → `z≈4545`); the brief and prompt now explicitly require `scipy.special.expit`, and a test enforces it. |
| 2026-09-27 | Did not push Phase 1 commits to `origin/main` (GitHub remote already exists from an earlier Phase-0 push; `origin/main` currently sits at `c90d2c8`, 2 commits behind local `main`) | No push was requested; flagged to the user rather than pushed unprompted, since a git push to a shared remote is a visible, hard-to-fully-reverse action requiring explicit confirmation per this session's operating rules. |

---

## 10. Next action

**Phase 0 and Phase 1 engineering items are done** (Phase 0: commits `1c3c67d`, `c90d2c8`; Phase 1: commit `0607db7`). Phase 1 exit criteria met: Acceptance Test A passes, propagator matches the Skyfield reference within 10m/0.01 m/s, full §12 validation suite passes (61 tests total, 100% coverage on all Phase 1 modules, `ruff` clean).

Three Phase-0 human-only items remain open and are not blocking: team roles (§47), OSC allocation (§48), and human acknowledgment of the PRD/§50 log.

**Astro questions Q1-Q5:** implemented per the user-approved recommendations (see §9). Q3 (RTN sign convention) and Q5 (density-shift definition) aren't consumed by any code yet — they'll matter starting Phase 3 (`safety/maneuver_mapper.py`) and Phase 5 respectively. If the actual astrophysics teammate reviews these recommendations and disagrees with any of them, flag it before those phases start.

**Phase 2 is now split into two parallel tracks (2026-09-27) — see the Phase 2 section above.** Part B's full spec-and-tests handoff is done and committed: `tests/conjunction/{test_screening,test_tca,test_collision,test_risk}.py` (26 tests, clean RED, pre-verified against a discarded reference implementation) plus `docs/OrbitGuard_Phase2_Antigravity_Brief.md` (the handoff document and Antigravity-ready prompt). That work required no phase-gate approval to produce, since it's spec/test authoring, not Part A production implementation.

**Awaiting user approval to begin Part A implementation** (`sensing/noise.py`, `sensing/belief.py`, `safety/adaptive_margin.py`, `scenarios/conjunction_generator.py`). Two loose ends worth closing in parallel, not blocking: (1) Phase 1's commits (`0607db7`, `d601069`) haven't been pushed to `origin/main` yet — the teammate starting on Part B will need at least Phase 1 pushed before they can usefully clone and run `tests/conjunction/`; (2) no new Astro-physics ambiguities were identified for Part A beyond Q5 (already flagged, not load-bearing until Phase 5).
