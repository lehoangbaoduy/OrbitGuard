# OrbitGuard — Final Product Requirements & Technical Specification

**Full name:** Orbital Reinforcement-Based Intelligent Traffic Guard
**Document status:** Implementation-ready, Stage-1 frozen
**Project type:** Course research project / research prototype
**Duration:** One semester (~15 weeks) · **Team:** 5 students
**Deliverables:** Conference-style paper, poster/demo, reproducible codebase, interactive replay UI
**Version:** 1.6 — supersedes all prior drafts (original PRD, Revised PRD v2, Technical Spec v1.0). v1.1 replaced the flat 15-week table in §44 with a checkable, phase-by-phase milestone tracker. v1.2 replaced the generic role-allocation placeholder in §47 with the team's actual assignment. v1.3 tags every individual milestone in §44 with its owner from §47. v1.4 makes the RL environment implementation stack explicit (Gymnasium → PettingZoo `ParallelEnv` → RLlib, §28). v1.5 replaces the placeholder TLE file description with the actual provided snapshot's real format and parameters (§16, §39). v1.6 restructures §47 for a five-person team — the original combined Safety/Evaluation/Visualization role split into three, and every owner tag in §44 updated to match. No scope or requirement changed in any revision.
**Date:** September 2026

This document is self-contained. An implementation team should not need to reference any earlier draft.

---

## Part I — Product & Research Framing

### 1. Executive Summary

OrbitGuard is a research simulator and decision-support prototype for adaptive multi-agent satellite collision avoidance. It combines a physics-based orbital environment, noisy state estimation, conjunction detection, a bounded collision-risk proxy, an **adaptive safety margin**, a parameter-shared **MAPPO-style multi-agent RL policy**, and a **deterministic joint-action safety shield**.

It is a controlled research environment, not an operational system. Its purpose is to test whether an uncertainty- and congestion-aware adaptive safety margin improves the safety–efficiency tradeoff of multi-agent RL for collision avoidance — particularly when deployment conditions differ from training conditions.

> **Positioning statement:** OrbitGuard is a reproducible research simulator for studying whether adaptive, uncertainty-aware safety margins improve multi-agent-RL-based satellite collision avoidance under distribution shift. It is not an operational spacecraft safety system, and it makes no flight-safety or formal-guarantee claims.

### 2. Research Question & Contributions

**Primary research question:**
> Can an uncertainty- and congestion-aware adaptive safety margin improve the safety–efficiency tradeoff of parameter-shared multi-agent RL for satellite collision avoidance under distribution shift?

**Core contributions (what the paper is built to demonstrate):**
1. **Adaptive safety margin** — `d_safe(i,j)` responds to uncertainty, local congestion, and estimated collision risk, instead of a fixed threshold.
2. **Distribution-shift robustness** — a trained policy is evaluated zero-shot on conditions that differ from training, with the central comparison being whether the adaptive-margin system degrades more gracefully than a fixed-margin counterpart.

**Secondary contribution:**
3. **Safe MARL architecture** — a learned maneuver-intent policy combined with a deterministic joint-action safety shield, letting the paper separate *what the learned policy does* from *what the safety layer prevents*.

### 3. Scope

#### 3.1 In scope — MVP

- Custom two-body + J2 orbital simulator, vectorized, RK4 integration.
- N = 10–20 satellites supported; **N = 15 is the primary benchmark.**
- Configurable noisy telemetry and covariance (Gaussian).
- Pairwise conjunction screening, numerical TCA + minimum-separation estimation.
- Bounded, monotonic collision-risk proxy (not a formal `P(collision)`).
- Pairwise adaptive `d_safe(i,j)` with explicit `[d_min, d_max]` bounds.
- Parameter-shared MAPPO-style CTDE training.
- Discrete maneuver-intent action space → deterministic RTN Δv mapper.
- Joint-action deterministic safety shield with bounded repair search.
- Five baselines (§29): No-Maneuver, Rule-Based, Fixed-Margin MAPPO, Adaptive MAPPO (proposed), Adaptive MAPPO without shield.
- Four core distribution-shift scenarios (§30).
- 3 training seeds per learned configuration; 10 evaluation seeds per trained model per condition.
- Full reproducibility: seeds, configs, git commit, data snapshot all logged.
- Plotly/Dash visualization with standalone episode replay.
- Paper-ready metrics, logs, plots, and tables generated from logged experiment data.

#### 3.2 Stretch goals

Attempt only once the full MVP is complete and the required experiments have run:

- Uncertainty-model-mismatch shift scenario (bias, heavy tails, covariance underestimation).
- Continuous Δv action space.
- True collision-probability computation (Foster/Akella–Alfriend) compared against the MVP proxy.
- An additional classical/optimization baseline (deterministic short-horizon risk minimizer).
- Scaling experiments beyond N=20 (50–100 satellites), reported as an appendix, not a core result.
- CesiumJS/Three.js visualization upgrade for the poster/demo.
- Hierarchical MARL (global manager + local clusters).
- Richer sensor/uncertainty modeling.

#### 3.3 Explicitly out of scope

- Operational collision-avoidance claims or flight qualification.
- Real spacecraft control or real-time flight integration.
- Full-fidelity atmosphere/drag/SRP/third-body/attitude modeling.
- Finite-burn propulsion dynamics (impulsive maneuvers only).
- Full-fidelity simulation at 1M-satellite scale (addressed only as a scalability discussion, per §46).
- Formal, provable safety guarantees for real spacecraft.
- Adversarial/competitive agent dynamics (all satellites are cooperative by construction).
- Real telemetry ingestion as a production service.

### 4. Safety Objective Hierarchy

OrbitGuard optimizes the following priorities, in strict order:

| Priority | Objective |
|---|---|
| 1 — Hard safety | No simulated physical collision under ground-truth state. |
| 2 — Risk constraint | Predicted conjunction risk stays below the configured risk threshold wherever physically feasible. |
| 3 — Operational separation | Predicted separation stays above the adaptive pairwise `d_safe`. |
| 4 — Mission preservation | Avoid unnecessary orbital deviation from the target mission envelope. |
| 5 — Efficiency | Minimize Δv and unnecessary maneuver frequency. |

The system must never be described, in code, docs, or the paper, as providing real-world or formal spacecraft safety guarantees.

### 5. Environment Classification

| Dimension | Classification |
|---|---|
| Observability | Partially observable |
| Agents | Multi-agent, cooperative |
| Determinism | Stochastic |
| Temporal structure | Sequential |
| Dynamism | Dynamic |
| State space | Continuous |
| MVP action space | Discrete |
| Knownness | Partially known |
| Coordination | Cooperative (no adversarial agents) |

---

## Part II — Architecture

### 6. High-Level Architecture

```
GROUND-TRUTH ORBITAL STATE
            ↓
   Sensor / Uncertainty Model
            ↓
      Agent Belief / Estimate
            ↓
       Orbit Prediction
            ↓
 TCA + Minimum Separation + Risk Proxy
            ↓
      Adaptive Pairwise d_safe
            ↓
      MAPPO / CTDE Policy
            ↓
      Maneuver Intent(s)
            ↓
   Deterministic Δv Mapper
            ↓
  Joint-Action Safety Shield
            ↓
     Approved Joint Δv Set
            ↓
      Orbital Propagator
            ↓
          (repeat)
```

RL is responsible **only** for the maneuver-intent decision. Orbital mechanics, conjunction screening, risk estimation, and the final safety filter are all deterministic and non-learned. This separation is a design principle, not an implementation detail — it's what keeps failure modes diagnosable and keeps the safety story scientifically defensible.

### 7. Runtime Model

OrbitGuard is a **Python application**, not a browser-based simulation. The Dash/Plotly interface is a separate visualization/replay layer over it.

**Mode A — Headless simulation.** Used for unit tests, training, batch experiments, evaluation, CI, and HPC/OSC execution.

**Mode B — Interactive web visualization.** Used for debugging, demos, poster presentations, and replaying saved episodes. **Must not require the RL trainer to be running** — it loads and replays logged episode artifacts.

Rationale: RL training and vectorized physics are Python-native (RLlib/PettingZoo, built on Gymnasium's space primitives — see §28); Dash gives browser-based visualization without forcing the simulator into JavaScript; headless execution stays available on the RTX 4070 Ti or OSC.

### 8. Repository Structure

```text
orbit_guard/
├── configs/
│   ├── default.yaml
│   ├── train.yaml
│   ├── validation.yaml
│   ├── test.yaml
│   └── scenarios/
├── orbit_guard/
│   ├── physics/          # constants, state, dynamics, integrator, frames
│   ├── sensing/           # noise, covariance, belief
│   ├── conjunction/       # propagation, tca, screening, risk
│   ├── safety/             # adaptive_margin, maneuver_mapper, constraints, shield
│   ├── rl/                    # observation, reward, env, policy, training
│   ├── scenarios/         # constellation, conjunction_generator, reset
│   ├── evaluation/        # runner, metrics, baselines, statistics
│   ├── logging/             # schema, writer
│   └── visualization/   # app, playback, figures
├── tests/
├── experiments/
├── artifacts/
└── scripts/
```

Exact file names may differ; responsibilities must remain separated as shown — in particular, physics/safety code must never import from `rl/`.

---

## Part III — Physics & Simulation

### 9. Coordinate System, Units, and Frames

- **Internal units are SI-consistent**: position (m), velocity (m/s), acceleration (m/s²), time (s), Δv (m/s), covariance (SI-squared units).
- **Primary numerical state**: Earth-centered inertial (ECI)-style frame. Exact frame/epoch convention must be documented in code (`physics/frames.py`) and in the paper.
- **Local maneuver frame**: RTN — `R` (radial, away from Earth center), `T` (transverse/along-track, direction of motion), `N` (normal/cross-track).
- All frame transforms require unit tests (§37).

### 10. Orbital State & Dynamics Model

Per-satellite state: `x_i = [r_x, r_y, r_z, v_x, v_y, v_z]` (Cartesian is the numerical truth; orbital elements — semi-major axis, eccentricity, inclination, RAAN, argument of periapsis, mean anomaly/phase — are derived diagnostics computed on demand, never propagated as an independent truth).

Acceleration model: `r̈ = a_2body + a_J2` — standard two-body gravity plus Earth's J2 zonal harmonic. No drag, SRP, third-body effects, attitude dynamics, or finite burns in the MVP.

**Integration:** fixed-step RK4. The propagator must support configurable `dt` for validation experiments, but all reported core experiments use the reference timescale table below.

### 11. Timescales (MVP Reference Configuration)

| Parameter | MVP default |
|---|---:|
| `physics_dt` (integration step) | 10 s |
| `control_dt` (RL decision step) | 60 s |
| `lookahead` (conjunction prediction horizon) | 1800 s (30 min) |
| `episode_horizon` | 21,600 s (6 h) |

At each control step, the environment internally runs multiple physics integration steps plus one prediction pass. All values are configuration-driven and logged with every experiment.

### 12. Physics Validation Requirements

Before RL training begins, the propagator must pass:
1. Two-body circular-orbit sanity check.
2. Energy/angular-momentum conservation in two-body-only mode.
3. J2 perturbation sanity check (e.g., correct nodal regression direction/rate).
4. Comparison against a trusted reference implementation (e.g., Skyfield or poliastro) on several test cases.
5. Convergence check as the numerical timestep is reduced.

The simulator must fail loudly (raise, not silently continue) if any state becomes non-finite or physically invalid.

### 13. Satellite Capability Model

Mostly homogeneous satellites with an explicit capability vector, so heterogeneous experiments (e.g., a stuck thruster) don't require architecture changes:

```text
SatelliteCapability:
    max_delta_v_per_action    # m/s, per maneuver type — see §23
    remaining_delta_v          # m/s, MVP default: 2.0
    maneuver_enabled            # bool
    mission_deviation_limit    # normalized units, MVP default: 1.0
```

A disabled/stuck-thruster satellite sets `maneuver_enabled = false`. The MVP uses a **remaining Δv budget**, not physical propellant modeling — this is a deliberate simplification, stated explicitly in the paper.

### 14. Ground Truth → Observation → Belief → Prediction → Evaluation Pipeline

This separation is fundamental to the POMDP formulation and must be explicit in the codebase, not implicit:

1. **Ground truth** — exact simulated physical state.
2. **Sensor observation** — noisy measurement of ground truth (§15).
3. **Agent belief / estimate** — the normalized observation actually exposed to the RL policy.
4. **Prediction** — future state propagated *using the estimated state*, not ground truth.
5. **Evaluation truth** — ground-truth trajectory, used only for collision determination and final metrics.

The policy never receives ground truth, except in a specifically-defined diagnostic oracle baseline if the team chooses to add one (not required for MVP). The evaluation system may access ground truth freely for metrics/collision determination.

### 15. Telemetry & Uncertainty Model

Configurable Gaussian state-estimation noise, independently settable per position/velocity axis.

| Parameter | MVP default | Notes |
|---|---:|---|
| Position noise σ (1σ, per axis) | 100 m | Placeholder matching realistic short-horizon tracking uncertainty |
| Velocity noise σ (1σ, per axis) | 0.1 m/s | |
| `covariance_mode` | `true_estimate` | The covariance shown to the agent is assumed correct for the configured noise model — **an explicit MVP simplification**, stated as such in the paper. |

**Stretch extension (uncertainty-model mismatch, §30):** train under this model, evaluate under bias, heavier-tailed noise, or covariance underestimation, as *separate evaluation configurations* — never by silently changing core simulator semantics.

---

## Part IV — Sensing, Risk, and the Adaptive Safety Margin

### 16. Orbital Data Strategy

**Hybrid approach:** real orbital data from a **frozen CelesTrak Starlink 53° shell snapshot** — 3,002 satellites, epoch range 2026-09-14 to 2026-09-17 (a single bulk pull; the ~3-day spread across the catalog is normal TLE staggering, not multiple snapshots) — define realistic *distributions* of orbital parameters. Stored as `data/TLE_snapshot_53deg_shell.csv`, **pre-parsed into orbital elements** (inclination, RAAN, eccentricity, argument of periapsis, mean anomaly, mean motion, semi-major axis, altitude) rather than raw two-line-element text — this is the actual format in use; do not write a raw-TLE parser against it. Observed shell parameters, to be used as the real sampling distribution rather than a placeholder: inclination 53.04°–53.17° (mean 53.16° — a single tight shell), altitude 460–465 km (mean ≈463 km), eccentricity < 0.0008 (near-circular), RAAN spanning the full 0–360°. Synthetic constellation instances of N=10–20 satellites are then sampled from these distributions, with controlled conjunction injection (§17) layered on top for experimental control.

The MVP does **not** propagate real satellite trajectories as an evaluation dataset — the paper must describe this as "real-data-derived initialization distributions," never "trained on real satellite telemetry." The snapshot date is stored with every experiment and never re-pulled mid-semester (reproducibility).

### 17. Constellation Generation & Conjunction Injection

At each episode reset, independently randomize (within the active scenario's configured bounds):

- orbital elements / phase spacing (from the frozen TLE-derived distribution),
- conjunction geometry: target TCA, minimum-separation regime, relative velocity, approach geometry, orbital-plane relationship, initial separation,
- sensor-noise realization,
- initial Δv budget,
- maneuver-capability failures (only when the active scenario calls for them).

Conjunctions are **statistically generated from sampled distributions**, never a small set of hand-coded fixed cases. The generator must support benign (no-conjunction), single-conjunction, and multi-conjunction episodes, and must expose deterministic scenario seeds.

**Training distribution mix (MVP default, tunable):** 30% benign · 50% single-conjunction · 20% multi-conjunction. The only hard requirement is that dangerous events must not occur in 100% of training episodes — benign episodes matter for the reward signal to be meaningful.

### 18. Collision Definition & Detection

Physical collision: `‖r_i − r_j‖ < R_i + R_j`, where `R_i`, `R_j` are fixed hard-body radii.

| Parameter | MVP default |
|---|---:|
| Per-satellite hard-body radius `R_i` | 5 m |
| Combined hard-body radius (typical pair) | 10 m |

Rules:
- Collision is determined from **ground truth**, never observed/estimated state.
- Checked at physics-integration resolution (every `physics_dt` = 10 s) at minimum, so no collision is missed between RL decision points. A local interpolation/refinement method may be added if needed.
- The exact numerical method must be documented in code and unit-tested (§37–38).

### 19. TCA and Minimum Separation

For candidate pair `(i,j)`:

```
t_TCA = argmin_t  ‖r_i(t) − r_j(t)‖         (over the lookahead window)
d_min = ‖r_i(t_TCA) − r_j(t_TCA)‖
```

Procedure: propagate over the lookahead window → locate the discrete minimum → optionally refine locally via interpolation → return `(t_TCA, d_min)`. **The same procedure must be used for every baseline and every model** — this is a fairness requirement, not just a numerical-accuracy one.

### 20. Candidate Pair Set

At N=10–20, exhaustive pairwise screening is acceptable: `N(N−1)/2` candidate pairs (105 pairs at N=15). No spatial indexing is required for the MVP. The paper must explicitly note this as an implementation choice that would need to change for large-N (million-satellite) deployments — this is exactly the scalability caveat referenced in §3.1 of the original concept.

### 21. Collision-Risk Proxy

The MVP computes a **bounded, monotonic proxy**, not a formal probability of collision. Never label it `P(collision)` in code, logs, or the paper — use `risk_proxy` or `R_proxy`.

```
σ_rel,ij = g(Σ_i, Σ_j)                       # combined relative uncertainty in encounter geometry
z_ij     = d_min,ij / (σ_rel,ij + ε)         # normalized separation
R_ij     = f(z_ij) ∈ [0, 1]                    # bounded risk, monotonically decreasing in z
```

**MVP default functional form** (complementary logistic sigmoid — bounded, monotonic, easy to interpret and tune):

```
f(z) = 1 / (1 + exp(-k · (z0 − z)))
```

| Parameter | MVP default | Meaning |
|---|---:|---|
| `ε` | 1.0 m | Prevents divide-by-zero at near-zero uncertainty |
| `z0` | 3.0 | Normalized-separation crossover point — risk ≈ 0.5 when miss distance ≈ 3× combined relative sigma |
| `k` | 1.5 | Sigmoid steepness |

`f` must be fixed and frozen before final experiments, and reported exactly in the paper — this is a scientific reproducibility requirement, not a style preference. Do **not** use an arbitrary weighted sum of dimensional quantities; every input must be normalized first.

### 22. Adaptive Safety Margin — Core Contribution #1

**Processing order** (prevents circularity — `d_safe` must never be an input to its own risk calculation):

```
Predict trajectories → TCA / d_min → Risk proxy → Uncertainty + local density → d_safe → Classify conjunction/safety state
```

**Formula:**

```
d_safe,ij = clip( d_base · [1 + α·Σ_norm,ij + β·ρ_ij + γ·R_ij],  d_min,  d_max )
```

**Resolved density term** (this is where the two source drafts disagreed — see design-decision log, §50): `ρ_ij` is **pairwise and symmetric**, defined as the count of *other* satellites within a reference density radius of the **midpoint** of `(i,j)`, normalized:

```
ρ_ij = min( 1,  count_within(midpoint(i,j), d_density_ref) / ρ_ref_count )
```

| Parameter | MVP default |
|---|---:|
| `d_density_ref` | 50 km |
| `ρ_ref_count` | 5 |
| `d_base` | 1000 m |
| `d_min` | 200 m |
| `d_max` | 5000 m |
| `α` (uncertainty coefficient) | 1.0 |
| `β` (density coefficient) | 0.5 |
| `γ` (risk coefficient) | 1.0 |

Properties (all mandatory, all unit-tested — see §37): pairwise internally; always bounded to `[d_min, d_max]`; deterministic; fully configuration-driven; `α, β, γ` selected on **training/validation only**, then frozen for final test. A single scalar per-satellite margin may be *displayed* in the UI as a summary (e.g., min over neighbors), but the underlying model is always pairwise.

---

## Part V — RL Problem Formulation

### 23. Action Space & Maneuver Mapping

The policy never outputs Cartesian Δv directly. It selects one of six **maneuver intents**; a deterministic mapper converts the intent into an RTN Δv vector.

| # | Intent | RTN direction | Magnitude |
|---|---|---|---:|
| 0 | `NO_MANEUVER` | — | 0 |
| 1 | `RAISE_ORBIT` | +T (prograde) | `base_delta_v` |
| 2 | `LOWER_ORBIT` | −T (retrograde) | `base_delta_v` |
| 3 | `PHASE_FORWARD` | −T (retrograde, fine adjustment) | `0.5 × base_delta_v` |
| 4 | `PHASE_BACKWARD` | +T (prograde, fine adjustment) | `0.5 × base_delta_v` |
| 5 | `RADIAL_ADJUSTMENT` | +R (radially outward) | `base_delta_v` |

| Parameter | MVP default |
|---|---:|
| `base_delta_v` | 0.1 m/s |

**Physical rationale (resolved convention — see design-decision log, §50):** a prograde (+T) burn raises semi-major axis/period, causing the satellite to drift *backward* relative to un-maneuvered neighbors over many orbits; a retrograde (−T) burn does the opposite. `RAISE_ORBIT`/`LOWER_ORBIT` are the "decisive," full-magnitude version of this effect; `PHASE_FORWARD`/`PHASE_BACKWARD` use the same axis at half magnitude as a smaller, more surgical drift adjustment that doesn't commit to as large an altitude excursion. `RADIAL_ADJUSTMENT` uses the orthogonal `R` axis instead, because a radial impulse changes separation *immediately* rather than through secular drift — useful when TCA is too close for a phasing effect to develop in time.

This mapping is fixed in one module (`safety/maneuver_mapper.py`), unit-tested, and must never be reinterpreted elsewhere in the codebase. The action names are **maneuver intents** — the paper should describe the discrete action space as approximating maneuver classes via canonical local-frame impulses, not as physically complete finite-burn maneuvers.

### 24. Observation Specification

Fixed-size, per-agent, decentralized observation:

```text
OWN:
    position_estimate
    velocity_estimate
    remaining_delta_v
    mission_deviation
    capability (max_delta_v_per_action, maneuver_enabled)

NEIGHBORS[k=4]:                        # top-k by predicted THREAT, not distance
    relative_position
    relative_velocity
    risk_proxy
    d_safe
    uncertainty_summary
    mask                                          # 0 if slot is padded/unused

GLOBAL_SUMMARY:
    threatening_neighbor_count
    max_risk
    total_risk
    minimum_predicted_separation
```

- **Neighbor ranking** is by predicted risk/threat level, not raw Euclidean distance — the nearest satellite isn't necessarily the most dangerous one dynamically.
- **Padding:** fewer than 4 relevant neighbors → zero-pad the remaining slots, `mask = 0`.
- **Anonymity:** neighbors are represented by physical/feature vectors only — no persistent satellite identity is exposed to the shared policy. This keeps the policy generalizable across constellation instances.

### 25. Reward Specification

Dense, per-control-step reward (not sparse/terminal-only — collision is a large terminal/event penalty layered on top of dense shaping):

```
R_i = w1·R_safety,i + w2·R_mission,i − w3·‖Δv_i‖ − w4·C_coordination,i
```

**Safety term** — margin-based, *not* raw-distance-based (prevents the agent wasting fuel maximizing distance for its own sake):

```
m_ij = (d_min,ij − d_safe,ij) / (d_safe,ij + ε)
```
`R_safety` is a bounded/clipped function of `m_ij` across all active neighbor pairs — bound/clip it explicitly so no single close encounter dominates the gradient.

**Mission term** — normalized weighted orbital-element deviation, deliberately **not** raw Cartesian position error (an avoidance maneuver can move a satellite away in Cartesian position while remaining a perfectly acceptable orbital adjustment):

```
D_mission = q_a·D_a + q_e·D_e + q_i·D_i + q_M·D_M
```
where `D_a, D_e, D_i, D_M` are normalized semi-major-axis, eccentricity, inclination, and along-track-phase deviations respectively.

| Parameter | MVP default |
|---|---:|
| `q_a, q_e, q_i, q_M` | 0.25 each (equal weighting starting point) |
| Semi-major-axis tolerance band | ±2 km |
| Eccentricity tolerance band | ±0.001 |
| Inclination tolerance band | ±0.05° |
| Along-track phase tolerance band | ±10 km |

**Δv term:** directly proportional to applied maneuver magnitude.

**Coordination term** — captures "I solved my problem but made the constellation-wide problem worse":

```
C_coordination = Δ(total constellation risk) = R_total,post − R_total,pre
```
Only *positive* degradation (risk got worse) is penalized — an improvement is not double-rewarded here (it's already captured by `R_safety`). Clipping/normalization must be fixed before training.

**Collision term:** large fixed negative terminal/event penalty on physical collision.

| Parameter | MVP starting default (grid-search during validation, then freeze) |
|---|---:|
| `w1` (safety) | 1.0 |
| `w2` (mission) | 0.3 |
| `w3` (Δv cost) | 0.1 |
| `w4` (coordination) | 0.5 |
| Collision terminal penalty | −100 |

### 26. Multi-Agent Decision Semantics

All agents propose actions **simultaneously** at each control step. The system then evaluates the full **joint** candidate action set through the centralized safety shield before anything is applied:

```
Agent proposals → Joint candidate action set → Short-horizon consequence simulation
  → Hard/risk/mission/capability checks → Accept or repair joint set → Applied joint Δv
```

Deterministic conflict-resolution priority: (1) resolve highest-risk conjunctions first; (2) among feasible joint configurations, select the one minimizing total post-maneuver constellation risk; (3) deterministic tie-breaking (e.g., lower total Δv, then lowest satellite index).

### 27. Action Masking Policy

**No safety-based action masking in the main proposed-policy experiments.** The actor may propose any of the six intents; the shield determines feasibility after the fact. This is intentional — it's the only way to measure unsafe-proposal frequency and shield intervention rate as meaningful diagnostics of what the *learned policy* actually does. A separately-flagged action-masked diagnostic run may be added later if useful, but it is not the primary experimental configuration.

### 28. MAPPO / CTDE Architecture

**Implementation stack (resolved — see design-decision log, §50).** Three layers, each with one job, not three competing environment abstractions:

| Layer | Job |
|---|---|
| **Gymnasium** | Standard RL space/API primitives (`Discrete(6)` for the action space, `Dict` for the observation in §24). Foundational — never used directly as OrbitGuard's environment. |
| **PettingZoo `ParallelEnv`** | The multi-agent environment contract. All 15 satellites propose actions simultaneously each control step (§26), which is exactly what the **Parallel API** is built for — not PettingZoo's turn-based **AEC API**, which assumes agents act one at a time and does not fit OrbitGuard's semantics. |
| **RLlib** | Training, rollout, and policy/critic infrastructure — consumes the `ParallelEnv`. |

`rl/environment.py` implements `OrbitGuardEnv` as a direct subclass of PettingZoo's `ParallelEnv`, typed with Gymnasium spaces — not a chain of wrappers (Gymnasium env → PettingZoo wrapper → RLlib wrapper → OrbitGuard). RLlib's exact recommended integration pattern for a custom `ParallelEnv` (e.g., a wrapper class vs. multiple inheritance with `ray.rllib.env.MultiAgentEnv`) has changed across Ray/RLlib releases and was still evolving as of this PRD's writing — the RL lead should verify the current-recommended pattern against RLlib's own documentation at Phase 3 implementation time rather than treating any specific class name as frozen here.

**Testing benefit worth using:** PettingZoo ships its own environment-conformance checks (`parallel_api_test`, a seed/reproducibility test) that directly exercise the requirements already listed in §37 — reset reproducibility, observation-shape consistency, multi-agent synchronization. Run these against `OrbitGuardEnv` as part of the Phase 3 gate, in addition to (not instead of) OrbitGuard's own environment tests.

Parameter-shared actor across all satellites; centralized critic during training.

**Actor** — input: decentralized observation (§24) for satellite `i`. Output: categorical distribution over the 6 maneuver intents. Execution uses **only** this local observation.

**Critic** — input may include centralized information: full constellation state/observation summary, pairwise risk features, global density, remaining Δv budgets, and relevant joint-action information. **No critic-only feature may leak into decentralized execution** — this is a hard architectural rule, not a preference.

**MVP hyperparameter starting point** (standard PPO/MAPPO defaults, tune during validation, freeze before final test):

```yaml
learning_rate: 3.0e-4
gamma: 0.99
gae_lambda: 0.95
clip_param: 0.2
entropy_coeff: 0.01
train_batch_size: 4000
sgd_minibatch_size: 256
num_sgd_iter: 10
actor_hidden_layers: [128, 128]
critic_hidden_layers: [256, 256]
```

**Training-time shield behavior (resolved — see design-decision log, §50):** for the "Adaptive MAPPO" and "Fixed-Margin MAPPO" configurations, the safety shield is **active during training, not just evaluation**. The agent's realized environment transition reflects the *shield-approved* action, not its raw proposal, while the agent still receives the safety/coordination reward terms computed against its own proposal where relevant (so it's penalized for proposing unsafe actions even when the shield ultimately prevents the bad outcome). The "Adaptive MAPPO without Shield" configuration is a **fully separate training run** with the shield disabled end-to-end (subject only to basic physical capability limits — Δv budget, `maneuver_enabled`) — it is not merely an evaluation-time toggle on the same trained weights.

---

## Part VI — Safety Shield

### 29. Joint-Action Safety Shield

**Inputs:** current estimated state, all agents' candidate actions, capability state, risk/conjunction state, adaptive margins, mission constraints.

**Algorithm:**
1. **Map intents → Δv** for every proposed action (deterministic, §23).
2. **Capability validation** — reject any action violating per-action Δv limit, remaining Δv budget, or `maneuver_enabled = false`.
3. **Short-horizon consequence prediction** — simulate the joint candidate maneuver set over the same lookahead window used for conjunction screening (§17: 1800 s).
4. **Safety checks** — physical collision risk, new conjunctions created, existing risk worsened, `d_safe` violations, mission-deviation-limit violations.
5. **Accept** if the full joint set is feasible.
6. **Repair if rejected** (bounded search, not exhaustive `6^N` enumeration):
   - identify only the satellites involved in unsafe or strongly-coupled conjunctions,
   - generate alternatives **only** for those affected agents; leave uninvolved agents at `NO_MANEUVER`,
   - if the number of affected agents is ≤ 4: **exact enumeration** (`6^4 = 1296` combinations — fast),
   - if more than 4 agents are affected: **beam search**, beam width 50, expanding one agent at a time, keeping the top-50 partial assignments by predicted total post-maneuver risk at each expansion,
   - select the feasible alternative minimizing total post-maneuver constellation risk, with deterministic tie-breaking (lower total Δv, then lowest satellite index),
   - `NO_MANEUVER` is selected for an agent only when it is actually the safest feasible choice for that agent — never as a default fallback.
7. **Apply** the approved joint Δv set.

| Parameter | MVP default |
|---|---:|
| `repair_mode` | `bounded_joint_search` |
| Exact-enumeration threshold | ≤ 4 affected agents |
| Beam width (when beam search triggers) | 50 |

The shield is fully deterministic given identical inputs and is a **simulated safety filter**, never described as a formal proof system or operational guarantee.

### 30. Conjunction Operational Classification

The system stores both physical geometry (`d_min`, `t_TCA`, relative velocity) and an operational classification (inside `d_safe`? above risk threshold? both?) as **separate fields** — never conflate "hard collision," "high risk," and "inside `d_safe`" into a single flag.

---

## Part VII — Experimental Design

### 31. Baselines

| # | Name | Description |
|---|---|---|
| 1 | **No-Maneuver** | Same simulator, same scenario seeds, zero actions applied. Establishes the no-intervention floor. |
| 2 | **Rule-Based** | Deterministic fixed-threshold heuristic: detect a conjunction under a fixed (non-adaptive) safety threshold → select a deterministic maneuver → apply fixed-magnitude Δv → deterministic priority for conflicts. Heuristic must be fully specified and frozen before use. |
| 3 | **Fixed-Margin MAPPO** | Identical architecture/training/shield to the proposed system, except `d_safe = constant` (no dependence on uncertainty/density/risk). This is the direct ablation isolating the value of adaptivity. |
| 4 | **Adaptive MAPPO (proposed)** | Full system as specified in this document. |
| 5 | **Adaptive MAPPO, no shield** | Same trained architecture, but the shield is fully disabled during both training and evaluation (§28), subject only to physical capability limits. High-priority ablation, not a recommended operational mode — separates learned behavior from safety-layer behavior. |

### 32. Distribution-Shift Evaluation Suite

All four are **zero-shot**: freeze the trained policy, hyperparameters, adaptive-margin coefficients, and shield thresholds; no fine-tuning; evaluate directly.

| # | Shift | Design |
|---|---|---|
| 1 | Increased telemetry noise | Position/velocity noise σ beyond the training range. |
| 2 | Increased local density | **N held fixed**; satellites concentrated into a denser local orbital configuration (isolates congestion from population-size effects — a deliberate design requirement, not incidental). |
| 3 | Simultaneous conjunctions | Multiple active conjunction events at once (directly stresses the coordination term). |
| 4 | Reduced maneuver capability | One or more satellites set to `maneuver_enabled = false` or substantially reduced Δv capability, applied identically across every baseline for fair comparison. Tests whether the system maneuvers *other* satellites to compensate. |
| Stretch | Uncertainty-model mismatch | Evaluate under biased state estimates, heavier-tailed noise, or covariance underestimation — tests whether the *adaptive, uncertainty-aware* mechanism is robust to miscalibration, not just to larger well-calibrated Gaussian noise. |

### 33. Training Protocol & Train/Validation/Test Separation

| Split | Purpose | Scenario pool |
|---|---|---|
| **Training** | Optimize policy weights. | Procedurally generated from the training distribution (§17 mix), effectively unlimited via seeded generator — no fixed count. |
| **Validation** | Select reward weights, adaptive-margin coefficients, RL hyperparameters, and checkpoints. | 50 fixed seeds, drawn from the training distribution, disjoint from training and test seeds. |
| **Final test** | Strictly held out until every model/config choice is frozen. No test-driven tuning, ever. | 10 fixed, held-out seeds per condition (normal + 4 shifts = 5 conditions) = **50 unique held-out scenario configurations.** |

**Seeds:**
- **3 independent training seeds** per learned configuration (Fixed-Margin MAPPO, Adaptive MAPPO, Adaptive MAPPO no-shield). No-Maneuver and Rule-Based are non-learned and require no training seeds.
- **10 independent evaluation seeds** per trained model per condition, drawn from the held-out final-test pool, and **matched across every baseline** (same 10 scenario seeds reused for every method) to enable paired statistical comparisons.
- Total final-test evaluation load per learned config: 3 trained models × 10 eval seeds × 5 conditions = 150 episodes. Non-learned baselines: 10 eval seeds × 5 conditions = 50 episodes each (still worth running across all 10 for consistent paired statistics, even though the policy itself is deterministic).

### 34. Metrics & Statistical Reporting

**Primary endpoint:** collision / unsafe-conjunction rate under distribution shift.

**Secondary endpoints:**
- Total applied Δv and remaining Δv.
- Mission-deviation score (§25).
- Number of maneuvers applied.
- Shield intervention rate — **treated as a diagnostic, not a direct optimization target.**
- Post-maneuver residual risk.
- Degradation relative to in-distribution performance:
  ```
  Degradation = (Metric_shift − Metric_normal) / (|Metric_normal| + ε)
  ```
  with sign convention chosen per metric (e.g., higher-is-worse for collision rate).

**Reporting requirements:** mean, standard deviation, and confidence interval (where practical) across independent seeds; always state the number of training seeds and evaluation seeds behind any reported number. Prefer **paired** comparisons (same scenario seeds across methods) for the headline baseline comparisons. Never report a single training run as the primary result.

### 35. Experiment Matrix

| Model | Normal | Noise↑ | Density↑ | Simultaneous conj. | Capability↓ |
|---|:---:|:---:|:---:|:---:|:---:|
| No-Maneuver | ✓ | ✓ | ✓ | ✓ | ✓ |
| Rule-Based | ✓ | ✓ | ✓ | ✓ | ✓ |
| Fixed-Margin MAPPO | ✓ | ✓ | ✓ | ✓ | ✓ |
| Adaptive MAPPO (proposed) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Adaptive MAPPO, no shield | ✓ | optional | optional | optional | optional |

First pass: run all 4 core baselines on all 5 conditions. Run the no-shield ablation on whichever subset of shift conditions is most informative if compute/time is constrained (do not drop it entirely — see §43 scope-cut rules).

### 36. Failure-Mode Taxonomy

Every episode's outcome should be classifiable into one or more of:

1. Physical collision.
2. Unsafe conjunction remained (violated `d_safe` without physical collision).
3. Unsafe RL proposal rejected by shield.
4. RL proposal feasible but mission-costly.
5. Maneuver budget exhausted.
6. Satellite unable to maneuver (`maneuver_enabled = false`).
7. New conjunction created by a maneuver.
8. Existing conjunction worsened by a maneuver.
9. Numerical simulation failure.

These labels drive both paper analysis and debugging, and should be logged per-episode.

---

## Part VIII — Engineering Requirements

### 37. Testing Strategy

- **Physics:** gravity calculation, J2 calculation, RK4 consistency, frame transforms, conservation/sanity checks.
- **Geometry:** distance calculation, TCA calculation, collision detection, boundary conditions.
- **Risk:** monotonicity, boundedness, deterministic output.
- **Adaptive margin:** higher uncertainty/density/risk must not *lower* the margin (given positive coefficients); output always within `[d_min, d_max]`.
- **Shield:** rejects capability violations; respects Δv budget; identifies unsafe joint actions; deterministic output; repairs an unsafe candidate when a feasible alternative exists.
- **Environment:** reset reproducibility under a fixed seed; observation shape consistency; action-mapping correctness; multi-agent synchronization — plus PettingZoo's own `parallel_api_test` and seed test, run against `OrbitGuardEnv` as an additional, off-the-shelf conformance layer (§28).

### 38. End-to-End Acceptance Tests

| Test | Requirement |
|---|---|
| **A — Benign orbit** | No conjunction injected; No-Maneuver baseline does not spontaneously collide. |
| **B — Single conjunction** | A known injected conjunction is detected and produces a meaningful risk signal. |
| **C — Adaptive margin** | Changing uncertainty/density/risk inputs changes `d_safe` in the expected direction, within bounds. |
| **D — RL pipeline** | Two satellites complete a train → evaluate loop with no code changes between phases. |
| **E — Shield** | An intentionally unsafe RL proposal is rejected or replaced when a feasible alternative exists. |
| **F — Reproducibility** | Same config + seed reproduces the same deterministic simulation/shield outputs within numerical tolerance. |
| **G — Distribution shift** | The evaluation runner executes all four core shift configurations from a single experiment-config interface. |

### 39. Reference Configuration

This is the actual starting config an implementation team should commit to the repo on day one (all "TBD" values from earlier drafts are now filled in with the MVP defaults specified throughout this document):

```yaml
simulation:
  N: 15
  N_range: [10, 20]
  physics_dt_s: 10
  control_dt_s: 60
  lookahead_s: 1800
  episode_horizon_s: 21600

dynamics:
  model: two_body_j2
  integrator: rk4

sensing:
  noise_model: gaussian
  position_sigma_m: 100
  velocity_sigma_mps: 0.1
  covariance_mode: true_estimate

collision:
  hard_body_radius_m: 5.0

risk:
  model: normalized_separation_proxy
  epsilon_m: 1.0
  z0: 3.0
  k: 1.5

adaptive_margin:
  d_base_m: 1000
  d_min_m: 200
  d_max_m: 5000
  alpha: 1.0
  beta: 0.5
  gamma: 1.0
  density_radius_km: 50
  density_ref_count: 5

rl:
  algorithm: mappo
  parameter_sharing: true
  num_training_seeds: 3
  learning_rate: 3.0e-4
  gamma: 0.99
  gae_lambda: 0.95
  clip_param: 0.2
  entropy_coeff: 0.01
  train_batch_size: 4000
  sgd_minibatch_size: 256
  num_sgd_iter: 10
  actor_hidden_layers: [128, 128]
  critic_hidden_layers: [256, 256]

observation:
  k_neighbors: 4

maneuver:
  base_delta_v_mps: 0.1
  phase_fraction: 0.5

capability:
  remaining_delta_v_mps: 2.0
  mission_deviation_limit: 1.0

reward:
  w1_safety: 1.0
  w2_mission: 0.3
  w3_delta_v: 0.1
  w4_coordination: 0.5
  collision_penalty: -100
  mission_weights: {q_a: 0.25, q_e: 0.25, q_i: 0.25, q_M: 0.25}

shield:
  repair_mode: bounded_joint_search
  exact_enumeration_max_agents: 4
  beam_width: 50

experiment:
  training_distribution_mix: {benign: 0.30, single_conjunction: 0.50, multi_conjunction: 0.20}
  validation_seeds: 50
  test_seeds_per_condition: 10
  evaluation_seeds_per_model: 10

data:
  tle_source: celestrak_starlink_shell
  tle_file: data/TLE_snapshot_53deg_shell.csv
  tle_format: parsed_elements_csv   # pre-parsed orbital elements, not raw TLE text
  tle_epoch_range: ["2026-09-14", "2026-09-17"]
  tle_satellite_count: 3002
  tle_inclination_range_deg: [53.04, 53.17]
  tle_altitude_range_km: [460, 465]
  snapshot_frozen: true
```

Every value above is a **starting default**, not final science — tune `adaptive_margin`, `reward`, and `rl` blocks on the validation split, then freeze before the final test run and report the frozen values in the paper.

### 40. Logging Specification

**Per-episode metadata:** `experiment_id, scenario_id, scenario_seed, training_seed, model_name, git_commit, config_hash, start_time, N_satellites, shift_type`.

**Per-control-step record:** `timestamp, satellite_id, observed_state, belief_features, selected_action, mapped_delta_v, shield_action, shield_intervened, shield_reason, risk_before, risk_after, d_safe_before, d_safe_after, remaining_delta_v, mission_deviation`.

**Per-episode summary:** `collision_count, unsafe_conjunction_count, total_delta_v, maneuver_count, mission_deviation_summary, shield_intervention_count, return`.

### 41. Visualization Application

Dash web app, backed by precomputed or live simulation data.

**Required views:** constellation overview · time playback · conjunction/risk panel · per-satellite inspector · maneuver/shield event timeline · key-metrics panel.

**Replay mode:** a saved episode loads and replays without running RL training or inference.

**Visual rule:** the UI must distinguish actual collision, high-risk conjunction, inside-`d_safe`, shield intervention, and applied maneuver **using both color and a label/marker** — never color alone.

### 42. Functional Requirements

- **FR1:** Support N=10–20 satellites, N=15 as the primary benchmark.
- **FR2:** Propagate two-body + J2 dynamics with configurable numerical timestep.
- **FR3:** Generate noisy telemetry and associated uncertainty estimates.
- **FR4:** Predict pairwise trajectories over a configurable lookahead horizon.
- **FR5:** Compute TCA and minimum separation, identically across all models/baselines.
- **FR6:** Compute a deterministic, bounded collision-risk proxy.
- **FR7:** Compute bounded, pairwise, symmetric adaptive `d_safe`.
- **FR8:** Expose fixed-size decentralized observations, threat-ranked and zero-padded/masked.
- **FR9:** Train a parameter-shared MAPPO-style CTDE policy.
- **FR10:** Map discrete maneuver intents to deterministic RTN Δv per §23.
- **FR11:** Evaluate joint candidate actions through a deterministic safety shield.
- **FR12:** Replace infeasible joint actions with the lowest-risk feasible alternative via bounded repair search.
- **FR13:** Log every shield intervention and outcome metric per §40.
- **FR14:** Run all 5 baselines from configuration, not code edits.
- **FR15:** Run all 4 core distribution-shift scenarios from configuration.
- **FR16:** Replay saved episodes through the visualization without retraining or live inference.

### 43. Non-Functional Requirements

- **Reproducibility:** every experiment seedable and replayable from config + seed + git commit.
- **Performance:** N=15 development workload must be practical on the RTX 4070 Ti; escalate to OSC for full training-seed × baseline × shift-scenario sweeps if needed.
- **Determinism:** physics, TCA/conjunction logic, risk calculation, adaptive margin, and shield must be deterministic given identical inputs.
- **Testability:** physics, TCA/conjunction, risk, adaptive margin, and shield each require automated unit tests (§37) before being relied on for experiments.
- **Maintainability:** simulation, RL, safety, evaluation, and visualization stay independently testable modules — no circular imports between `physics/`, `safety/`, and `rl/`.

---

## Part IX — Program Management

### 44. Phases & Milestones

Eight phases, Week 1 through Week 15. Each phase has a single goal, a checkable milestone list, and an exit criterion tied back to a specific acceptance test (§38) or functional requirement (§42) — so "done" is never a matter of opinion. Checkboxes are meant to be checked off directly in this file as work lands (e.g., in a PR that closes out a milestone), so this document stays the single source of truth for progress, not a separate tracker that drifts out of sync.

**Quick-reference summary:**

*Owner tags: **Astro** = physics & simulation lead (astrophysics major) · **RL** = RL/learning lead (you) · **Safety** = safety & shield lead · **Evaluation** = evaluation & experiments lead · **Visualization** = visualization lead (all four non-Astro, non-RL roles defined in §47 — Safety is your team's original third member; Evaluation and Visualization are the two new members).*

| Phase | Weeks | Goal | Primary owner |
|---|---|---|---|
| 0 — Setup & Freeze | 1 | Repo, config, and data snapshot ready; zero open ambiguity before coding starts. | Joint |
| 1 — Physics Foundation | 2–3 | Orbital propagator correct and validated. | Astro |
| 2 — Sensing, Conjunction & Risk | 4–6 | Everything the policy and shield will "see" is correct. | Astro |
| 3 — RL Environment (GATE) | 7–8 | Prove the full learning loop on 2 satellites before scaling. | RL |
| 4 — Scale + Safety Shield | 9–10 | N=15 training running, shield active. | RL + Safety |
| 5 — Baselines & Core Experiments | 11–13 | Generate the results the paper is built on. | Evaluation |
| 6 — Ablations & Visualization | 14 | Fill in supporting evidence, make it visible. | Evaluation + Visualization |
| 7 — Paper, Poster & Buffer | 15 | Ship the deliverables. | Joint |

---

#### Phase 0 — Setup & Freeze (Week 1)

**Goal:** No team member should be blocked on an undecided question once Phase 1 starts.

- [ ] Repo scaffolding created matching §8's module layout — **RL**
- [ ] Reference config (§39) committed as `configs/default.yaml` — **RL**
- [ ] TLE snapshot pulled from CelesTrak and frozen — `data/TLE_snapshot_53deg_shell.csv` committed, never re-pulled — **Astro**
- [ ] Team roles assigned (§47), adjusted from the suggested split to actual skills — **Joint**
- [ ] OSC compute allocation requested (don't wait until it's needed — §48) — **RL** (heaviest compute user; request under whichever account the team agrees on)
- [ ] This PRD reviewed and acknowledged by every team member, including the design-decision log (§50) — **Joint**

**Exit criteria:** repo exists, config loads without error, TLE file is committed, and every team member can state the current MVP scope (§3.1) from memory.

#### Phase 1 — Physics Foundation (Weeks 2–3)

**Goal:** Get the one component everything else depends on right, before building anything on top of it.

- [ ] Two-body + J2 acceleration model (`physics/dynamics.py`) — **Astro**
- [ ] RK4 integrator (`physics/integrator.py`), configurable timestep — **Astro**
- [ ] ECI ↔ RTN frame transforms, unit tested — **Astro**
- [ ] Physics validation suite passing (§12): energy/angular-momentum conservation, J2 nodal-regression sanity check, comparison against a reference tool (Skyfield/poliastro), convergence check as timestep shrinks — **Astro**
- [ ] Constellation generator producing N=10–20 satellites from the TLE-derived distributions (§16–17) — **Astro**

*(RL, Safety, Evaluation, and Visualization leads are not idle this phase — see §47.7 for what each builds against mocked data while physics is underway.)*

**Exit criteria:** Acceptance Test A (§38) passes — a benign, no-conjunction episode never spontaneously "collides" — and propagator output matches the reference tool within an agreed numerical tolerance.

#### Phase 2 — Sensing, Conjunction & Risk (Weeks 4–6)

**Goal:** Correct inputs for both the RL policy and the safety shield — errors here silently poison every result downstream.

- [ ] Telemetry/uncertainty model — configurable Gaussian noise (§15) — **Astro**
- [ ] Ground-truth / observation / belief / prediction / evaluation-truth separation implemented as distinct, non-conflatable code paths (§14) — **Astro**
- [ ] TCA and minimum-separation computation (§19), unit tested — **Astro**
- [ ] Collision detection against ground truth (§18), unit tested, boundary conditions covered — **Astro**
- [ ] Risk proxy `f(z)` implemented (§21) — monotonicity and boundedness unit tested — **Astro**
- [ ] Adaptive safety margin `d_safe(i,j)` implemented (§22) — bounds and symmetry unit tested — **Astro** *(the formula and its tests — coefficient tuning is a separate Phase 4 milestone owned by RL)*
- [ ] Conjunction injection / scenario generator with configurable randomization (§17) — **Astro**

**Exit criteria:** Acceptance Tests B and C (§38) pass — an injected conjunction is detected with a meaningful risk signal, and `d_safe` moves in the expected direction (and stays within `[d_min, d_max]`) as uncertainty, density, and risk inputs change.

#### Phase 3 — RL Environment & 2-Agent Sanity Check — GATE (Weeks 7–8)

**Goal:** Prove the entire learning loop works end-to-end at the smallest possible scale, before committing the remaining seven weeks to it.

- [ ] `OrbitGuardEnv` implemented as a PettingZoo **`ParallelEnv`** (not AEC), typed with Gymnasium spaces: observation (§24), action mapping (§23), reward (§25) — **RL**
- [ ] PettingZoo's `parallel_api_test` and seed/reproducibility test pass against `OrbitGuardEnv` (§28) — **RL**
- [ ] Deterministic maneuver-intent → Δv mapper, unit tested — **Safety** *(owned per §47.3; deliver early in this phase — RL's environment depends on it for the action step)*
- [ ] MAPPO/CTDE training pipeline wired up in RLlib (§28) — **RL**
- [ ] 2-satellite training run converges to sensible, inspectable behavior — **RL**
- [ ] Acceptance Test D (§38) passes — train → evaluate completes with no code changes in between — **RL**

> **GATE:** This is the highest-risk phase in the project. If the 2-agent case isn't cleanly working by the end of Week 8, invoke §46 immediately — cut every stretch goal and re-plan the remaining weeks around it. Do not proceed to Phase 4 on a shaky 2-agent result; a scaling failure at N=15 is far more expensive to debug than a 2-agent one.

#### Phase 4 — Scale to Full MVP + Safety Shield (Weeks 9–10)

**Goal:** The complete proposed system, running at the primary benchmark scale.

- [ ] Environment/training scaled to N=10–20 (primary benchmark N=15) — **RL**
- [ ] Joint-action safety shield implemented, all 7 steps (§29) — **Safety**
- [ ] Shield repair search implemented and tested — exact enumeration for ≤4 affected agents, beam search (width 50) otherwise — **Safety**
- [ ] Acceptance Test E (§38) passes — an intentionally unsafe proposal is rejected or replaced when a feasible alternative exists — **Safety**
- [ ] Coarse grid search of reward weights and margin coefficients run on the **validation** split only (§25, §33 — never touch test seeds here) — **RL**

**Exit criteria:** N=15 shielded MAPPO trains end-to-end without crashing across at least 3 training seeds; shield intervention rate is logged and its magnitude looks sane (neither ~0%, which suggests the diagnostic isn't wired up, nor so high it's clearly suppressing all learning).

#### Phase 5 — Baselines & Core Experiments (Weeks 11–13)

**Goal:** Generate the actual numbers the paper's central claim depends on.

- [ ] All 5 baselines implemented, runnable from configuration alone (§31, FR14) — **Evaluation** *(No-Maneuver and Rule-Based are Evaluation's own build; the 3 MARL-based baselines reuse RL's config-driven pipeline from Phase 3–4)*
- [ ] Adaptive-vs-Fixed-Margin comparison run across the full 3-training-seed protocol — **Evaluation** orchestrates, on top of **RL**'s pipeline
- [ ] All 4 core distribution-shift scenarios implemented and run zero-shot (§32, FR15) — **Evaluation** builds the harness; **Astro** defines the physical scenario parameters (e.g. what "increased density" or the stuck-thruster capability-down case concretely means)
- [ ] Full seed protocol executed: 3 training seeds × 10 evaluation seeds × 5 conditions (§33) — **Evaluation**
- [ ] Acceptance Tests F and G (§38) pass — same config + seed reproduces the same result; the shift-scenario suite runs from one config interface — **F: RL + Evaluation jointly · G: Evaluation**
- [ ] Metrics/statistics pipeline producing means, spread, and degradation scores (§34) — **Evaluation**

**Exit criteria:** the headline adaptive-vs-fixed-margin result exists, with paired statistics across matched scenario seeds, and is reproducible from a fresh checkout.

#### Phase 6 — Ablations, Visualization & Analysis (Week 14)

**Goal:** Fill in the supporting evidence and make the system's behavior visible to someone other than the person who built it.

- [ ] Shield-vs-no-shield ablation run (§28) — prioritized per §46 over any remaining stretch feature — **RL** trains the no-shield policy (a separate training run, per §28) · **Evaluation** runs the comparison · **Safety** supports interpreting shield-specific results
- [ ] Dash/Plotly visualization built (§41); replay mode works standalone, no live training/inference required (FR16) — **Visualization**
- [ ] Failure-mode taxonomy (§36) applied to logged episodes — **Evaluation**, reviewed by **Astro** for physical plausibility
- [ ] Result tables and figures generated directly from logged experiment data, not hand-copied numbers — **Evaluation**

#### Phase 7 — Paper, Poster & Buffer (Week 15)

**Goal:** Ship it.

- [ ] Paper draft complete — motivation, method, results, limitations, with claims staying inside the boundary set by §51 — **Joint**, each lead drafts their own section
- [ ] Poster/demo built — **Joint**
- [ ] Any result that looked off in Phase 5–6 has been rerun and confirmed, not silently left as-is — **Joint**, owned by whoever's module produced the result
- [ ] Every item in §49 (Success Criteria) checked off — **Joint**

### 45. Implementation Priority Order

If schedule pressure hits, implement in this order and do not skip ahead: (1) physics correctness, (2) TCA/conjunction/risk correctness, (3) adaptive margin, (4) environment + 2-agent RL, (5) full N=15 RL, (6) safety shield, (7) baseline runner, (8) distribution-shift experiments, (9) metrics/statistics, (10) visualization polish, (11) stretch features. **Never sacrifice physics/safety correctness to accelerate visualization.**

### 46. Scope-Cut Rules

**Cut first if behind schedule:** CesiumJS/3D upgrade · uncertainty-model mismatch shift · continuous Δv · extra classical planner baseline · large-N scaling experiments · hierarchical MARL.

**Never cut:** baseline comparison (all 5) · fixed-vs-adaptive comparison · distribution-shift evaluation (all 4 core) · shield logging · reproducibility infrastructure · physics validation · safety/shield tests.

These are load-bearing for the paper's central claim — cutting any of them undermines the research question itself, not just polish.

### 47. Team Role Allocation

Five-person team: one astrophysics major, four CSE students (one MS — the project lead, one who joined at team formation, two who joined later, likely undergraduate). This is the team's actual assignment, not a generic placeholder — module ownership, cross-role dependencies, and phase-by-phase driver responsibility are all specified below so nothing falls into a gap between roles.

**How this evolved from three roles to five:** the original team had a combined "Safety, Evaluation & Visualization" role carrying three genuinely separable threads at once. Adding two members split that role into three clean ones — each now maps to exactly one top-level repo module (`safety/`, `evaluation/`, `visualization/`), so every one of the five people owns a distinct, non-overlapping slice of the codebase. The person who originally held the combined role keeps `safety/` — the hardest and most safety-critical piece, where continuity matters most — and the two new members take `evaluation/` and `visualization/`, both of which are well-specified enough to onboard into without deep prior context on the project.

**Onboarding note for the two new members:** before Phase 1 work begins, have them read the OrbitGuard Beginner's Guide (if generated for this project) end-to-end, then this PRD's Parts I–II and their own role's section below. Their modules only start producing real work in Phase 3 onward (§47.7), so there's real ramp-up time before they're on the critical path.

#### 47.1 Physics & Simulation Lead — astrophysics major

**Owns:**
- `physics/` — two-body + J2 acceleration model, RK4 integrator, ECI ↔ RTN frame transforms, `SatelliteCapability` state (§13)
- `sensing/` — Gaussian telemetry noise model, covariance representation, belief construction (§15)
- `conjunction/` — orbit prediction over the lookahead window, TCA + minimum-separation computation (§19), collision detection at ground truth (§18), the risk-proxy function `f(z)` (§21)
- `scenarios/` — TLE-derived constellation generation and orbital-element distributions (§16), conjunction injection/randomization (§17)
- The physics validation suite (§12): energy/angular-momentum conservation, J2 nodal-regression check, comparison against a reference tool, convergence check
- **The `d_safe` formula itself** in `safety/adaptive_margin.py` — the function, its bounds, and unit tests for monotonicity/symmetry (§22)
- Acceptance Tests A, B, C (§38)

**Why this split:** every one of these modules requires understanding what the underlying physical quantities actually mean (uncertainty growth, orbital perturbation, encounter geometry) — this is the person best positioned to know whether a number is physically reasonable, not just numerically stable. This role is also the natural person to sanity-check the placeholder numeric defaults in §39 (`σ_pos=100m`, `base_delta_v=0.1 m/s`, hard-body radius, etc.) — those were my best guesses for a starting config, not validated physics.

#### 47.2 RL / Learning Lead — you (MS CSE, ML infrastructure background)

**Owns:**
- `rl/` in full — `OrbitGuardEnv` as a PettingZoo `ParallelEnv` typed with Gymnasium spaces (§28), the RLlib/MAPPO training pipeline, reward-weight tuning
- **Tuning** the adaptive-margin coefficients `α, β, γ` (§22) via the Phase 4 grid search on the **validation** split — the astro lead owns the formula's correctness; you own optimizing its parameters, since that's tied directly to the training/validation loop, not to physics
- Making the training pipeline fully config-driven so the same code trains every MARL-based baseline (Fixed-Margin MAPPO, Adaptive MAPPO, Adaptive MAPPO no-shield) from `configs/*.yaml` alone, per FR9 and FR14 — the evaluation lead *runs* these baselines, but you build the pipeline so a config swap is all that's needed
- Ensuring the shield's training-time behavior matches the design decision in §28 (shield active during training for the two shielded configs; the no-shield config is a fully separate training run, not an eval-time toggle) — this is a subtle point that's easy to implement wrong, and it's squarely an RL-pipeline concern
- Acceptance Test D (§38), plus the environment-level tests in §37 (reset reproducibility, observation-shape consistency, action-mapping correctness, multi-agent synchronization)

**Why this split:** the MAPPO/CTDE architecture, actor-critic setup, and RLlib training loop are the deepest ML-infrastructure piece in the project — the natural fit for the most ML-experienced person on the team.

**Cross-team dependency to flag early:** your environment code needs correct, tested outputs from the astro lead's `conjunction/` and `sensing/` modules before Phase 3's 2-agent gate can even be attempted meaningfully. During Phase 1–2, while physics is being built, don't wait idle — build the PettingZoo environment skeleton against **mocked** observation/risk/margin outputs (fake data matching §24's schema), so real integration in Phase 3 is a swap-in, not a build-from-scratch.

#### 47.3 Safety & Shield Lead — original third member

**Owns:**
- `safety/maneuver_mapper.py` — implementing the fixed intent → RTN Δv lookup table exactly as specified in §23 (this is mechanical once the convention is fixed, but **should be reviewed by the astro lead**, since correctness depends on RTN semantics — e.g., confirming a `+T` burn actually raises the orbit as expected in the simulator)
- `safety/shield.py` — the full joint-action shield algorithm (§29): capability validation, short-horizon consequence prediction, safety checks, and the repair search (exact enumeration ≤4 agents, beam search width 50 otherwise)
- Acceptance Test E (§38), and co-owns Acceptance Test F (reproducibility) with the RL lead, since it spans both training determinism and shield determinism
- Supporting the evaluation lead during the shield-vs-no-shield ablation (Phase 6) by interpreting shield-specific results

**Why this split:** the shield's repair search is a genuine algorithms problem (bounded search, beam search) that doesn't require ML-training expertise — good, substantial CS work that's distinct from physics and RL. It's also the most safety-critical piece in the whole system (a bug here can silently invalidate every downstream experiment), which is why it stayed with the person who already had the most context, rather than handing it to a new member.

#### 47.4 Evaluation & Experiments Lead — new member

**Owns:**
- `evaluation/` in full — the baseline runner for all 5 baselines (§31; No-Maneuver and Rule-Based are fully your own build, no RL dependency), the distribution-shift suite (§32), metrics and statistics (§34)
- The tooling that structurally protects final-test seeds from tuning code (§33, §36 of the design intent) — build this so accidental leakage is difficult, not just discouraged
- Acceptance Test G (§38), and co-owns Acceptance Test F with the RL lead
- Generating result tables/figures directly from logged experiment data (§49) — never hand-typed numbers

**Why this split:** running the experimental methodology — baselines, shift scenarios, statistics — is substantial, well-specified software and data-analysis work that doesn't require deep ML or physics background to execute correctly, which makes it a good on-ramp for someone joining without prior context on the project. It's also squarely on the path to the paper's central claim, so it benefits from a dedicated owner rather than being squeezed in alongside shield or visualization work.

**Cross-team dependency to flag early:** your baseline runner for the 3 MARL-based configurations depends on the RL lead's config-driven training pipeline (Phase 3–4), and your shift-scenario harness depends on the astro lead's definitions of what each shift condition physically means. Until those land, build against mocked training outputs and a stub scenario interface — see §47.7.

#### 47.5 Visualization Lead — new member

**Owns:**
- `visualization/` in full — the Dash/Plotly replay app (§41): constellation overview, time playback, conjunction/risk panel, per-satellite inspector, maneuver/shield event timeline, key-metrics panel
- The standalone replay requirement (FR16) — loading a saved episode with no live training or inference running
- The accessibility rule in §41: every state (collision, high risk, inside `d_safe`, shield intervention, maneuver) must be distinguishable by both color and a label/marker, never color alone

**Why this split:** visualization is a complete, self-contained deliverable with a clear specification (§41's required views) and no dependency on understanding MAPPO internals or orbital mechanics in depth — another strong on-ramp role. It's also explicitly the lowest-priority module under schedule pressure (§45, §46), which makes it a safe place for the newest/least-experienced team member to be still ramping up without blocking anyone else.

**Cross-team dependency to flag early:** the replay app needs real logged episode artifacts (§40's schema) to be meaningful. Until other modules are producing real logs, build against a hand-written mock episode file matching the schema, so the UI work isn't blocked on the rest of the system being done.

#### 47.6 Cross-cutting: logging

`logging/` (§40) is shared infrastructure — physics validation results, training metrics, shield interventions, and evaluation results all write through the same schema. Scaffold this jointly in Phase 0 (whoever sets up the repo drafts the schema), then each lead adds their own domain's fields as their module comes online. The evaluation lead is the natural schema custodian going forward, since they're the heaviest downstream consumer of logged data for the metrics/statistics pipeline — but every lead still owns writing their own fields correctly.

#### 47.7 Phase-by-phase driver

Cross-referencing the phases in §44 — who's in the driver's seat each phase, and what everyone else should be doing in parallel rather than sitting idle:

| Phase | Primary driver | Others, in parallel |
|---|---|---|
| 0 — Setup | Joint (you: repo/config scaffolding; astro lead: TLE snapshot) | Safety/Evaluation/Visualization leads: onboarding (§47 intro), stub out their own module's directory and test scaffolding. |
| 1 — Physics Foundation | **Astro lead** | You: learn PettingZoo/RLlib on a toy environment, not OrbitGuard yet. Safety lead: start the shield's repair-search logic against mocked risk data. Evaluation lead: scaffold the baseline runner and the seed-pool separation tooling against mocked training outputs. Visualization lead: build the Dash app shell against a hand-written mock episode file. |
| 2 — Sensing, Conjunction & Risk | **Astro lead** | You: build the PettingZoo env skeleton against mocked observations. Safety lead: continue shield logic against mocked risk/margin data. Evaluation lead: continue baseline-runner scaffolding. Visualization lead: build out the required views' layouts against mock data. |
| 3 — RL Environment (GATE) | **You** | Astro lead: on-call for physics bugs surfaced by real integration. Safety lead: deliver the Δv mapper early (RL's environment depends on it), keep building shield scaffolding not yet wired in. Evaluation lead: keep building against mocks, not yet wired to real training. Visualization lead: continue building against mocks. |
| 4 — Scale + Safety Shield | **You** (scaling) and **Safety lead** (shield integration), jointly | Astro lead: supports margin-coefficient tuning, validates physics holds up at N=15. Evaluation lead: start wiring the baseline runner to the now-real training pipeline. Visualization lead: start wiring to real logged episodes as they become available. |
| 5 — Baselines & Core Experiments | **Evaluation lead** | You: keep the training pipeline config-driven and bug-free across baseline runs. Safety lead: on-call for shield bugs surfaced during baseline runs. Astro lead: defines scenario parameters for the shift scenarios. Visualization lead: continues building/polishing against real logged data. |
| 6 — Ablations & Visualization | **Evaluation lead** (ablation + analysis) and **Visualization lead** (Dash app), jointly | You: run the no-shield training configuration. Safety lead: supports interpreting shield-specific ablation results. Astro lead: reviews the failure-mode taxonomy (§36) against physics plausibility. |
| 7 — Paper, Poster & Buffer | Joint — each lead writes their own section | — |

**If headcount changes again:** the cleanest merge-down is Evaluation + Visualization back into one role (both are self-contained, on-ramp-friendly threads with light cross-dependencies on each other), or Safety back into the original combined role if it must shrink to three. Keep physics and RL isolated as their own roles in any configuration — they're the two hardest dependencies in the schedule (§48).

### 48. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| MAPPO instability at N=15 | Curriculum: 2 satellites first (Week 8 gate), then scale. |
| Physics too slow to train at useful speed | Vectorize across satellites from day one; profile early; escalate to OSC before it blocks progress. |
| Shield too conservative, suppresses learning signal | Track intervention rate as a diagnostic from the start; loosen thresholds early in training if consistently high. |
| Weak/null distribution-shift signal | A clear degradation pattern — even an unflattering one — is a legitimate, reportable finding. Do not treat it as project failure. |
| Reward-weight tuning eats disproportionate time | Time-boxed coarse grid search using the reference config (§39) as the starting point; log every run. |
| Timeline slip | Strict adherence to §46 scope-cut rules; the MVP in §3.1 is the actual deliverable, not an aspiration. |
| OSC allocation delay | Apply in Week 1; RTX 4070 Ti remains the primary dev path so this can't block early weeks. |
| Over-tuning on test data | Hard train/validation/test separation (§33) enforced by tooling, not just discipline — e.g., the test-seed pool should be in a file the training/tuning code never reads. |

### 49. Success Criteria / Definition of Done

Tracked incrementally, phase by phase, in §44 — this list is the cumulative finish line, not a substitute for that tracker. MVP is complete when:
1. The simulator passes its physics validation suite (§12).
2. N=10–20 is supported with N=15 as the reliably-running primary benchmark.
3. The full proposed pipeline (adaptive margin + shield + MAPPO) trains end-to-end.
4. All 5 baselines run from configuration.
5. All 4 core distribution-shift scenarios run from configuration.
6. Results include the full seed protocol (3 training × 10 evaluation) per §33.
7. The adaptive-vs-fixed-margin comparison is reproducible from a fresh checkout.
8. The shield / no-shield ablation is available and was run (not merely implemented).
9. The Dash replay demo loads a saved episode standalone (no live training/inference required).
10. The paper draft clearly separates simulated findings from any operational claim, per §51.

**The project does not require the adaptive method to beat every baseline in every condition.** A well-characterized null result or failure mode is an acceptable, scientifically valid outcome.

### 50. Design-Decision Log (Resolutions Made in This Document)

Transparency record of every judgment call made while merging the prior drafts — override any of these if the team disagrees:

| # | Ambiguity found | Resolution | Where |
|---|---|---|---|
| 1 | Tech Spec used `ρ_ij` (pairwise); Revised PRD used `ρ_i` (single-satellite) in the same formula. | Defined as pairwise and symmetric: count of neighbors near the **midpoint** of `(i,j)`, normalized by a reference count. | §22 |
| 2 | Neither draft stated whether the shield is active during training or only at evaluation. | Active during training for the two shielded configurations; the no-shield ablation is a fully separate training run, not an eval-time toggle. | §28 |
| 3 | RTN sign convention for the 5 maneuver intents was explicitly flagged "must be agreed before coding" but left unresolved. | Concrete, physically-motivated, non-redundant mapping proposed (RAISE/LOWER = full ±T, PHASE_FWD/BWD = half ∓T, RADIAL = +R). | §23 |
| 4 | Ten numeric parameters (`d_base`, `α/β/γ`, `f(z)` form, `base_delta_v`, hard-body radii, MAPPO hyperparameters, shield beam width, etc.) were deliberately left symbolic in both prior drafts ("Section 50" freeze list). | Concrete, justified MVP starting defaults given for every one, consolidated in the reference config (§39). All remain tunable-then-frozen, not final science. | §15, §18, §21, §22, §23, §25, §28, §29, §39 |
| 5 | Train/validation/test scenario *counts* were conceptually defined but not sized. | 50 validation seeds; 10 held-out test seeds × 5 conditions = 50 final-test scenarios, matched across all baselines. | §33 |
| 6 | Neither draft specified the exact RL environment implementation stack, or whether PettingZoo's Parallel vs. AEC API applied. | Gymnasium (spaces only) → PettingZoo `ParallelEnv` (matches OrbitGuard's simultaneous joint-action semantics — not AEC) → RLlib, as a three-layer stack with one job each; `OrbitGuardEnv` subclasses `ParallelEnv` directly, no wrapper chain. RLlib's exact integration pattern for this should be re-verified against current docs at Phase 3, since it has changed across Ray releases. | §28 |
| 7 | The PRD assumed a raw TLE text file and an example ~550 km altitude; the actual provided snapshot is pre-parsed CSV at a different real altitude. | Real snapshot confirmed and adopted: `data/TLE_snapshot_53deg_shell.csv`, 3,002 satellites, epoch 2026-09-14–17, inclination 53.04–53.17°, altitude 460–465 km, pre-parsed elements format (no raw-TLE parsing needed). | §16, §39, Phase 0 |

### 51. Required & Prohibited Paper Claims

**May claim, subject to measured evidence:** whether adaptive margins improve collision/unsafe-conjunction outcomes; whether they improve robustness to the selected distribution shifts; the Δv/mission-disruption tradeoffs involved; shield intervention behavior; observed failure modes (§36).

**Must not claim:** operational readiness; flight qualification; real-world collision-avoidance guarantees; full-fidelity feasibility at 1M-satellite scale (this may be discussed only as a scalability *extrapolation*, explicitly labeled as such).

---

## Part X — Reference

### 52. Glossary

- **Δv (delta-v):** velocity change from a maneuver; standard fuel-cost proxy.
- **Conjunction:** a predicted close approach between two objects.
- **TCA:** time of closest approach.
- **Pc:** probability of collision (a formal quantity OrbitGuard's MVP proxy explicitly does *not* claim to compute).
- **TLE:** Two-Line Element set, the standard compact orbit description published for real satellites (e.g., via CelesTrak).
- **J2 perturbation:** the dominant orbital perturbation from Earth's oblateness.
- **RTN frame:** Radial / Transverse (along-track) / Normal (cross-track) local orbital frame used for maneuver directions.
- **CTDE:** Centralized Training, Decentralized Execution — agents train with shared/global information but act on local observations only.
- **(Dec-)POMDP:** (Decentralized) Partially Observable Markov Decision Process — the formal framework this problem fits (§5).
- **Distribution shift:** evaluating a trained policy under conditions that differ from its training distribution, to test generalization/robustness.
- **Shield intervention:** any case where the safety shield rejects or replaces an RL-proposed action.
