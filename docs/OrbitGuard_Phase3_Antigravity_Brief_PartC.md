# OrbitGuard Phase 3, Part C — Reward & Training Pipeline

**Written for:** the OrbitGuard teammate implementing this part of Phase 3 with their own Antigravity coding agent — not for Claude, not for the PRD's original "RL" role as a literal org chart. If you're a human reading this before pointing an agent at it, read the whole thing once; it's shorter than it looks because most of the actual specification lives in the test files, not in this prose. This is a self-contained document — you do not need Part B's brief to do this work, and Part B does not need this one.

**Status when this was written:** OrbitGuard's Phase 0 (setup), Phase 1 (physics foundation), and Phase 2 (sensing/belief, adaptive margin, conjunction injection, TCA/collision/risk) are complete, tested (165 tests, 99% coverage), and merged. Phase 3 (RL Environment & 2-Agent Sanity Check — the project's highest-risk **GATE** phase) is split three ways: **Part A** (`rl/environment.py`, `physics/mission_deviation.py`) stays with the teammate using Claude and has **not started implementation yet** — you are not blocked on it. **Part B** (`safety/maneuver_mapper.py`, `rl/observation.py`, a separate teammate's own Antigravity agent) is fully specified in its own separate brief and runs in parallel with you — you don't need to coordinate step-by-step, just avoid touching each other's files (see §3.1).

---

## 1. What OrbitGuard is (just enough to orient you)

A research simulator for multi-agent satellite collision avoidance. Satellites orbit Earth; occasionally two get dangerously close (a "conjunction"); a learned policy (MAPPO, parameter-shared across all satellites) decides whether to nudge a satellite out of the way; a safety shield (Phase 4, not yet built) will double-check that decision before it's applied. Phase 3's job is to prove the **learning loop itself** works end-to-end at the smallest possible scale (2 satellites) before committing to scaling it up. The full spec is `docs/OrbitGuard_Final_PRD_v1.md` — you do not need to read all of it. The sections that matter for your scope: §25 (reward) and §28 (RL stack/training).

**What already exists and you can import directly, without needing to understand its internals:** nothing physics-level — your two scopes (`reward.py`, `training.py`) are pure-function/RL-framework code, not physics code. You do not need `orbit_guard.physics`, `orbit_guard.sensing`, `orbit_guard.conjunction`, or `orbit_guard.scenarios` at all. Part A's future `rl/environment.py` is the only module that will wire everything (including your reward functions and training pipeline) together with the physics layer; you're building two pieces of that wiring, tested in isolation.

---

## 2. The tests are the spec — read them first, literally

`tests/rl/test_reward.py` is **already written and already committed**, and currently fails with `ModuleNotFoundError` because `rl/reward.py` doesn't exist yet. That's intentional — this project follows strict test-first development, and this test file *is* your reward-module assignment stated as executable code instead of prose. **It was independently pre-verified against a throwaway reference implementation (written, run to 100% pass, then deleted) before being handed to you** — so if something seems to fail for a plausible-and-correct implementation, treat that as worth flagging loudly, not silently routing around; this spec has already been checked to be achievable, but "checked" isn't "infallible."

**Your job is to make the reward tests pass, without editing them.** If you think a test is wrong (mis-specified, physically incorrect, testing the wrong thing): don't silently change it. Flag it back to the team with your specific objection.

Read the test file's **module docstring** first — every design decision (the asymmetric sign convention, why a term is left unnormalized) is explained there, not just asserted in the test bodies.

`rl/training.py` (§4 below) is different — there is **no frozen test file for it**, on purpose. Read §4 before assuming the same "just make tests pass" pattern applies there.

---

## 3. Your scope, part 1 — `rl/reward.py`

### 3.1 Your files

| File | Job | PRD ref | Spec |
|---|---|---|---|
| `orbit_guard/rl/reward.py` | Compute the per-agent dense reward from safety/mission/Δv/coordination terms | §25 | `tests/rl/test_reward.py` (23 tests) |
| `orbit_guard/rl/training.py` | Wire MAPPO/CTDE training in RLlib | §28 | No frozen test file (see §4) |
| `orbit_guard/rl/rllib_env.py` | Production RLlib env-wrapping helper (Dict-flatten pattern + centralized-critic support) | §28 | No frozen test file — a **third file explicitly allowed** beyond `reward.py`/`training.py`; see §4 |

**Do not touch** `orbit_guard/rl/observation.py`, `orbit_guard/safety/` (a different teammate's scope, working in parallel right now), `orbit_guard/physics/`, `orbit_guard/rl/__init__.py` (that teammate's PR also adds files under `orbit_guard/rl/`), `pyproject.toml`, or `uv.lock` (merge-conflict risk).

### 3.2 `reward.py` — compressed spec

```python
def compute_pairwise_margin(d_min_m: float, d_safe_m: float, epsilon_m: float = 1.0) -> float:
    """m_ij = (d_min_m - d_safe_m) / (d_safe_m + epsilon_m)"""

def compute_safety_term(margins: Sequence[float], clip_min: float = -1.0, clip_max: float = 1.0) -> float:
    """clip(min(margins), clip_min, clip_max); clip_max if margins is empty."""

def compute_mission_term(d_a, d_e, d_i, d_m, q_a=0.25, q_e=0.25, q_i=0.25, q_m=0.25) -> float:
    """-(q_a*d_a + q_e*d_e + q_i*d_i + q_m*d_m). Rejects any negative deviation component."""

def compute_delta_v_term(delta_v_mps: Sequence[float]) -> float:
    """||delta_v_mps|| -- the raw Euclidean norm, NOT pre-negated."""

def compute_coordination_term(total_risk_pre: float, total_risk_post: float) -> float:
    """max(0.0, total_risk_post - total_risk_pre) -- one-sided, improvements score 0."""

def compute_reward(
    safety_term, mission_term, delta_v_term, coordination_term, collision_occurred: bool,
    w1=1.0, w2=0.3, w3=0.1, w4=0.5, collision_penalty=-100.0,
) -> float:
    """w1*safety_term + w2*mission_term - w3*delta_v_term - w4*coordination_term
       [+ collision_penalty if collision_occurred]"""
```

**The sign convention is asymmetric on purpose — read this twice before implementing:**

- `compute_safety_term` and `compute_mission_term` return values that are **already reward-signed** (higher = better) and get **added** with `+w1`/`+w2` in `compute_reward`. `R_mission = -D_mission` is an explicit freeze in this spec — the PRD states `D_mission`'s formula but never states this sign; get it backwards and your mission term will reward *worse* orbital drift.
- `compute_delta_v_term` and `compute_coordination_term` return **raw, non-negative magnitudes**, not pre-negated, and get **subtracted** with `-w3`/`-w4` in `compute_reward`. Do not negate them inside their own functions — that would double the sign once `compute_reward` also subtracts.

This mirrors the PRD's own literal formula (`R_i = w1·R_safety + w2·R_mission − w3·‖Δv‖ − w4·C_coordination`) field-for-field, which is why it's the frozen convention rather than making everything uniformly signed one way.

**Two things worth flagging as you implement:**

1. **`compute_safety_term` uses `min`, not `mean`, across pairwise margins.** This is deliberate — a single dangerous encounter must dominate the safety signal, not get diluted by nine benign ones. There's a test (`test_safety_term_a_single_bad_pair_dominates_many_good_pairs`) specifically checking this.
2. **`compute_coordination_term`'s output is intentionally left unnormalized** in this spec — PRD §25 itself says "clipping/normalization must be fixed before training" without giving a formula. With N=15 there can be up to 105 active pairs, so an unscaled degradation could dominate the other (bounded) terms once real numbers are involved. This is an **open item you own** during your own reward-weight grid search later (PRD §39/§48) — don't invent a normalization inside `compute_coordination_term` itself; if you need one, add it as a documented parameter to `compute_reward` (e.g. scale `w4` or add a new argument), not silently inside the term function.

`reward.py` must never import `orbit_guard.rl.environment`, `orbit_guard.rl.training`, `gymnasium`, `pettingzoo`, `ray`, or `torch` — it must stay a pure, framework-free function module, testable without any RL framework installed. This is already enforced by a static test **inside** `tests/rl/test_reward.py` (`test_reward_module_never_imports_rl_environment_or_training_frameworks`) — don't add a duplicate of this test yourself.

**Cross-module note (informational — not something you need to coordinate directly):** a different teammate's `rl/observation.py` has an `own_mission_deviation` field that must numerically equal `-compute_mission_term(d_a, d_e, d_i, d_m)` for the same satellite at the same instant. Part A's environment is responsible for deriving both from the same four raw components with the same q-weights — this just explains why your function's sign/component choices matter beyond your own tests.

### 3.3 Running the reward spec

```bash
uv sync
uv run pytest tests/rl/test_reward.py -v
uv run pytest -q --ignore=tests/rl/test_observation.py --ignore=tests/safety/test_maneuver_mapper.py   # confirm nothing else broke
uv run ruff check orbit_guard tests
```

---

## 4. Your scope, part 2 — `rl/training.py` + `rl/rllib_env.py` (NOT a frozen pytest spec, on purpose)

Unlike `reward.py`, **there is no committed RED test file for `rl/training.py`**. This is a deliberate choice, not an oversight: PRD §28 itself says RLlib's exact recommended integration pattern for a custom multi-agent env "has changed across Ray/RLlib releases... verify the current-recommended pattern against RLlib's own documentation at Phase 3 implementation time rather than treating any specific class name as frozen." Pinning exact RLlib API calls into a committed test file would contradict that instruction and would likely be stale by the time you implement it anyway.

**What you get instead — working, pre-verified infrastructure, not just prose:**

- `tests/rl/stub_env.py` — `StubOrbitGuardEnv`, a real, PettingZoo-conformant (`parallel_api_test`-passing) 2-agent `ParallelEnv` with the exact frozen observation/action shapes the (separately-specified) `rl/observation.py` module will produce (`Dict` obs with `own`(10,)/`neighbors`(4,10)/`global_summary`(4,), `Discrete(6)` actions) — but trivial random-noise dynamics, no physics, no real reward. Use this to build and test your training wiring **before** Part A's real `rl/environment.py` exists; swapping in the real env later should be a one-line change to your `env_creator` function, not a rewrite. It also exposes `state()`/`state_space` — see the CTDE section below, this is not optional.
- `tests/rl/rllib_integration.py` — `FlattenedParallelPettingZooEnv`, a wrapper already verified against this project's actually-installed **Ray 2.58** (not the PRD's assumed 2.9). **Concrete finding, not a guess:** the current RLlib RLModule/Catalog API stack has no default encoder for a `Dict` observation space with multiple `Box` sub-spaces — a bare `PPOConfig().environment(...).build_algo()` followed by `.train()` raises `ValueError: No default encoder config for obs space=Dict(...)`. This wrapper flattens each agent's Dict observation into one Box via `gymnasium.spaces.utils.flatten` before RLlib ever sees it, which fixes it — verified with an actual `train()` → `save_to_path()` → `restore_from_path()` cycle completing successfully. **This file lives under `tests/` and is for the acceptance smoke test's own use — your production `rl/training.py` must not import it.** Port or extend this verified pattern into `orbit_guard/rl/rllib_env.py` instead, since the real centralized critic (below) will likely need to extend it anyway.
- `tests/acceptance/test_acceptance_d_training_smoke.py` — 3 tests, **already passing**, proving the *mechanical* wiring (env registration → shared-weights policy config → one training iteration → checkpoint save/restore) works end-to-end right now, against the stub. Runs in ~7 seconds. **Read its module docstring — it explicitly says parameter sharing alone is not CTDE,** see below.

**Critical: parameter sharing alone is NOT MAPPO/CTDE — read this before you start, it changes your architecture, not just a config flag.** PRD §28 requires CTDE: a parameter-shared **actor** that sees only its own agent's local observation (what `rl/observation.py` produces), PLUS a **centralized critic** that may see global information *during training only*, with a hard rule that no critic-only feature ever reaches the actor at decentralized execution time. `PPOConfig().multi_agent(policies={"shared_policy"}, ...)` on its own gives you a parameter-shared policy with RLlib's *default* (decentralized, per-agent) critic — that's IPPO, a different, simpler algorithm the PRD does not ask for. The smoke test above only proves the mechanical training loop runs; it deliberately does not build a centralized critic, and its own docstring says so.

**The frozen interface for your centralized critic's input:** `StubOrbitGuardEnv.state()` (and `.state_space`) returns the concatenation of every agent's own flattened local observation, in a fixed agent order — the simplest documented MAPPO-CTDE global-state convention. Part A's real `OrbitGuardEnv` will expose the identical `state()`/`state_space` interface, so your centralized-critic code is a swap-in once that lands, not a rewrite. You still need to:

1. Verify RLlib's **current** recommended pattern for a centralized critic under the new RLModule/Catalog API stack (a custom multi-module setup with a critic that takes the global state as an extra input, commonly via a `MultiRLModule` with inter-module communication, or a custom `Learner`/`Catalog` override — the exact current mechanism is genuinely something you need to look up, not something safe to assume from older RLlib tutorials built on the Policy-API stack). Context7 or `ray.io/en/master/rllib` docs, not cached training-data knowledge.
2. Confirm whatever you build still passes a parameter-sharing check equivalent to the smoke test's `test_shared_actor_weights_are_parameter_shared_across_agents`, AND exercises the centralized critic receiving `state()` during training — extend the acceptance test or add your own, but don't just leave the IPPO-level smoke test standing in as "CTDE done."
3. If, after investigating, you believe a full centralized critic is impractical for this phase's 2-agent gate and an IPPO-first approach is an acceptable interim step, **say so explicitly in your handoff** rather than quietly shipping IPPO under a "MAPPO/CTDE" label — this is a real scope/risk tradeoff for the team to see, not something to resolve silently either direction.

**What Acceptance Test D actually requires (PRD §38) — and what it does NOT require:** "train → evaluate completes with no code changes in between." The smoke test satisfies the *mechanical* half of this cheaply and is CI-safe. It deliberately makes **no reward-value or convergence assertion**, because a random-noise stub env has no learnable signal worth asserting on. **The real thing — a 2-satellite training run converging to sensible, inspectable behavior against the REAL `OrbitGuardEnv` (once Part A delivers it) — is a separate, human-reviewed task, not CI-gated, and not satisfied by extending this smoke test with more assertions.** Read the reward curve yourself; don't try to encode "did it learn something sensible" as a pytest assertion.

**Your actual scope for `rl/training.py` + `rl/rllib_env.py`:**

1. Before writing anything, re-verify the current RLlib docs for `PPOConfig`, `.multi_agent(...)`, the centralized-critic pattern above, and checkpoint APIs (`build_algo()`, not the deprecated `build()`) — Context7 or `ray.io/en/master/rllib` are more current than whatever's in your training data. The lookups already done (`ray.rllib.env.wrappers.pettingzoo_env.ParallelPettingZooEnv`, the RLModule Dict-obs limitation) should save you the first hour of this, not replace the rest of it.
2. Wire an `env_creator(config) -> ParallelPettingZooEnv`-shaped function in `orbit_guard/rl/rllib_env.py` that takes **any** `ParallelEnv` (stub during your own dev, the real `OrbitGuardEnv` at integration time) — don't hardcode an import of `tests.rl.stub_env` inside production code. Your production code should accept an env factory/class as a parameter or config value.
3. Build a `PPOConfig` with a **single shared-weights policy** across all agents (`policy_mapping_fn` always returning the same policy id) — necessary but not sufficient for CTDE, see above — **plus** a centralized critic wired to `state()`.
4. Write your own tests for whatever you add, following this project's AAA pattern and descriptive test names (see any existing test file for the house style) — you're not bound to zero tests just because there's no frozen file, only freed from a pre-pinned RLlib API surface, and including at least one test that your centralized critic actually receives `state()`.

---

## 5. Project conventions (not optional)

Same as every prior phase in this codebase — matching them means your code merges in without a review round-trip:

- **Type hints on every function signature**, `from __future__ import annotations` used throughout.
- **Fail loudly, never silently continue**: invalid input raises `ValueError` immediately (see negative mission-deviation components, etc. in `tests/rl/test_reward.py` for the exact cases). Don't clamp, don't warn-and-continue.
- **SI units always**: meters, seconds, m/s. Variable names carry unit suffixes (`_m`, `_s`, `_mps`) as existing code does.
- **No magic numbers**: every numeric default in this spec is already in `configs/default.yaml` (`reward.w1_safety`...`w4_coordination`, `reward.collision_penalty`) — they're given to you as function defaults so you don't need to read that file, but don't invent a different default independently.
- **Files stay under ~400 lines** (800 hard cap) — none of your modules should get anywhere near this if kept focused.
- **No new dependencies beyond what's already declared.** `pyproject.toml` already has `numpy`, `scipy`, `gymnasium`, `pettingzoo`, `ray[rllib]`, `torch` — you shouldn't need anything else. If you genuinely do, ask first; don't edit `pyproject.toml`/`uv.lock` yourself (merge-conflict risk against the parallel Part B PR).
- **Use `build_algo()`, not the deprecated `build()`**, and pass an explicit checkpoint path (e.g. a pytest `tmp_path` fixture in tests) to `save_to_path()` rather than leaving it to default into `/tmp`.

---

## 6. Running things

```bash
# from the repo root
uv sync                                                    # installs pinned dependency versions

uv run pytest tests/rl/test_reward.py -v
uv run pytest tests/acceptance/test_acceptance_d_training_smoke.py -v   # already passing against the stub -- confirms your RLlib install works

# confirm the REST of the repo is healthy before you start:
uv run pytest -q --ignore=tests/rl/test_observation.py --ignore=tests/rl/test_reward.py --ignore=tests/safety/test_maneuver_mapper.py
# -> should show 177 passing (165 Phase-0-2 baseline + 9 stub-env conformance + 3 Acceptance-D smoke)

uv run ruff check orbit_guard tests
```

A bare `uv run pytest -q` (no `--ignore`) will show 3 collection errors (`ModuleNotFoundError` for `reward.py` and a different teammate's `observation.py`/`maneuver_mapper.py`) until all of Phase 3's parts land — that's the expected starting RED state, not a broken repo.

**`uv sync` pulls `torch` with CUDA dependencies, which is multi-gigabyte on Linux.** Make sure your environment has the disk space/bandwidth for it, and budget time for the first `uv sync`.

**RLlib/Ray needs a real CPU and enough RAM to spin up a local Ray instance** — the smoke test takes ~7 seconds on this machine. Ray's multi-process scheduler has limited native-Windows support; if you're on native Windows (not WSL), confirm Ray actually starts (`uv run pytest tests/acceptance/test_acceptance_d_training_smoke.py -v`) before relying on it, and flag it to the team early if it doesn't — unlike Part B, you cannot avoid needing a working Ray install, since both your `training.py` work and the acceptance test depend on it directly.

---

## 7. Definition of done

- [ ] `uv run pytest tests/rl/test_reward.py -v` — all 23 tests pass, file not modified
- [ ] `uv run pytest tests/acceptance/test_acceptance_d_training_smoke.py -v` — still passing (or extended, not weakened, if you improve on it)
- [ ] A real centralized critic wired to `StubOrbitGuardEnv.state()`, not just a parameter-shared policy with RLlib's default decentralized critic (see §4 — this is the actual CTDE requirement, and the smoke test alone does not prove it)
- [ ] Your own tests for `rl/training.py`/`rl/rllib_env.py`, following house style (AAA pattern, descriptive names), including a test that the centralized critic actually receives global state
- [ ] `uv run pytest -q --ignore=tests/rl/test_observation.py --ignore=tests/safety/test_maneuver_mapper.py` — 177 + 23 (+ your own new tests) all green (a different teammate's two files may still be in progress in parallel — a bare `pytest -q` with no ignore will still show 2 collection errors until that work also lands, that's expected, not your bug)
- [ ] `uv run ruff check orbit_guard tests` — clean
- [ ] No file outside `orbit_guard/rl/reward.py`, `orbit_guard/rl/training.py`, `orbit_guard/rl/rllib_env.py` (plus your own new test file(s)) was created or modified
- [ ] `pyproject.toml`/`uv.lock`/`orbit_guard/rl/__init__.py`/`orbit_guard/rl/observation.py` untouched
- [ ] Your PR/handoff message states which current RLlib API pattern you verified and used for BOTH the Dict-observation flattening and the centralized critic (config class name, RLModule vs. legacy Policy API, etc.) — this becomes the permanent record in `docs/OrbitGuard_progress.md`'s P3C-M2 row, since the PRD explicitly flags this as something to re-verify and record, not assume. If you concluded a full centralized critic wasn't practical for this gate and shipped IPPO instead, say that explicitly here — don't label it MAPPO/CTDE.

---

## 8. Handing it back

Work on a branch (e.g. `phase3-reward-training`), commit only your new implementation files (the frozen test file should show as unmodified in your diff), and push / open a PR / hand the branch back however the team's already sharing this repo. Include in your PR description: confirmation that every Definition-of-Done box above is checked (with actual command output, not just "I think it's done"), and which RLlib API pattern you ended up using and why, since that's genuinely expected to need a judgment call against current docs rather than a fixed answer.

---

## Appendix: prompt for Antigravity

Copy everything between the lines below directly into Antigravity as your first message.

```
You are implementing Python modules in the OrbitGuard repository:
orbit_guard/rl/reward.py and orbit_guard/rl/training.py. This is a research
simulator for satellite collision avoidance; you only need the local
context below, not the whole project.

SCOPE -- create orbit_guard/rl/reward.py, orbit_guard/rl/training.py, and
orbit_guard/rl/rllib_env.py (a third production file for RLlib env-wrapping
helpers -- see PART 2 below for why you need it), plus your own new test
file(s) for the latter two. Do not create, edit, or delete any other file
in the repository, especially not pyproject.toml, uv.lock,
orbit_guard/rl/__init__.py (a different teammate's parallel PR also adds
files under orbit_guard/rl/), orbit_guard/rl/observation.py, or anything
under orbit_guard/safety/, orbit_guard/physics/, orbit_guard/sensing/,
orbit_guard/conjunction/, orbit_guard/scenarios/ -- those are done or owned
by a different teammate working in parallel right now.

PART 1 -- reward.py: THE SPEC IS A TEST FILE, tests/rl/test_reward.py
(23 tests), which currently fails with ModuleNotFoundError. Read its
module docstring FULLY before coding -- it explains an intentionally
asymmetric sign convention you must get exactly right. Your job is to
make every test in it pass, by implementing the module -- never by editing
the test file. Run `uv run pytest tests/rl/test_reward.py -v` to check.

Required signatures (see the test file for exact numeric behavior):

  def compute_pairwise_margin(d_min_m, d_safe_m, epsilon_m=1.0) -> float
      # (d_min_m - d_safe_m) / (d_safe_m + epsilon_m)

  def compute_safety_term(margins, clip_min=-1.0, clip_max=1.0) -> float
      # clip(min(margins), clip_min, clip_max); clip_max if margins is empty.
      # Uses MIN not mean -- one bad pairwise margin must dominate.

  def compute_mission_term(d_a, d_e, d_i, d_m, q_a=0.25, q_e=0.25, q_i=0.25, q_m=0.25) -> float
      # -(q_a*d_a + q_e*d_e + q_i*d_i + q_m*d_m). Raise ValueError if any
      # component is negative (they're deviation magnitudes, always >= 0).

  def compute_delta_v_term(delta_v_mps) -> float
      # ||delta_v_mps||, the raw Euclidean norm -- do NOT negate it here.

  def compute_coordination_term(total_risk_pre, total_risk_post) -> float
      # max(0.0, total_risk_post - total_risk_pre) -- one-sided, raw,
      # deliberately left UNNORMALIZED (a documented open item, not a bug).

  def compute_reward(safety_term, mission_term, delta_v_term, coordination_term,
      collision_occurred, w1=1.0, w2=0.3, w3=0.1, w4=0.5, collision_penalty=-100.0) -> float
      # w1*safety_term + w2*mission_term - w3*delta_v_term - w4*coordination_term,
      # + collision_penalty if collision_occurred. This is the ONLY function
      # that assembles final signs -- safety_term/mission_term arrive already
      # reward-signed (added with +), delta_v_term/coordination_term arrive
      # as raw non-negative magnitudes (subtracted with -). Do not pre-negate
      # delta_v/coordination inside their own functions.

reward.py must never import orbit_guard.rl.environment, orbit_guard.rl.training,
gymnasium, pettingzoo, ray, or torch -- it must stay a pure, framework-free
function module. This is already enforced by a static test INSIDE
tests/rl/test_reward.py (test_reward_module_never_imports_rl_environment_or_training_frameworks)
-- do not add a duplicate of this test yourself, it's already covered.

PART 2 -- training.py + rllib_env.py: THERE IS NO FROZEN TEST FILE FOR
THESE, on purpose -- the PRD explicitly says RLlib's integration pattern
has changed across releases and must be verified against CURRENT docs at
implementation time, not copied from a stale tutorial. Before writing
anything:

1. Check the currently-installed Ray/RLlib version
   (`uv run python3 -c "import ray; print(ray.__version__)"`) and look up
   ITS docs (e.g. via Context7 or ray.io/en/master/rllib), not older
   cached material.
2. Use tests/rl/stub_env.py's StubOrbitGuardEnv (a real, PettingZoo-
   conformant 2-agent ParallelEnv with the exact frozen observation/action
   shapes -- Dict obs with own(10,)/neighbors(4,10)/global_summary(4,),
   Discrete(6) actions, trivial random dynamics) to build and test your
   training wiring before the real environment exists. It also exposes
   state()/state_space -- the concatenation of every agent's own flattened
   observation, in fixed agent order -- which is the global-state input
   your CENTRALIZED CRITIC needs (see point 5, this is not optional).
3. tests/rl/rllib_integration.py's FlattenedParallelPettingZooEnv is
   ALREADY VERIFIED against this project's installed Ray 2.58: RLlib's
   current RLModule/Catalog stack has NO default encoder for a Dict
   observation space with multiple Box sub-spaces (raises
   `ValueError: No default encoder config for obs space=Dict(...)` on
   `algo.train()`). This wrapper flattens each agent's Dict obs into one
   Box first, which fixes it -- a full train -> save_to_path ->
   restore_from_path cycle already runs successfully with it (see
   tests/acceptance/test_acceptance_d_training_smoke.py, currently
   passing). BUT that file lives under tests/ -- your production
   orbit_guard/rl/training.py must NOT import it. Port/extend this exact
   verified pattern into orbit_guard/rl/rllib_env.py instead.
4. Your env_creator function must accept ANY ParallelEnv (the stub during
   your dev, the real OrbitGuardEnv once a teammate delivers it) -- your
   production code should accept an env factory/class as a parameter or
   config value, not hardcode anything from tests/.
5. CRITICAL -- read this point twice: a single shared-weights policy
   across all agents (policy_mapping_fn always returning the same policy
   id) is PARAMETER SHARING, which is necessary but NOT sufficient for
   PRD section 28's "MAPPO/CTDE" requirement. CTDE additionally needs a
   CENTRALIZED CRITIC that sees global state (StubOrbitGuardEnv.state(),
   point 2 above) during training, while the actor still only ever sees
   its own agent's local observation -- with a hard rule that no
   critic-only feature ever reaches the actor at decentralized execution
   time. A bare PPOConfig().multi_agent(policies={"shared_policy"}) with
   no further changes gives you a parameter-shared policy with RLlib's
   DEFAULT per-agent (decentralized) critic -- that is IPPO, a simpler,
   different algorithm, not what the PRD asks for. You need to look up
   RLlib's CURRENT recommended pattern for a centralized critic under the
   new RLModule/Catalog API stack (a custom multi-module setup, or a
   custom Learner/Catalog override that gives the critic the global state
   as an extra input) -- this is a real, not-yet-researched design
   decision for you to make against live docs, not something to assume
   from older Policy-API-stack tutorials. If, after investigating, you
   conclude a full centralized critic is impractical for this phase's
   2-agent gate and an IPPO-first approach is a reasonable interim step,
   say so EXPLICITLY in your handoff message rather than quietly shipping
   IPPO under a "MAPPO/CTDE" label -- that's a real scope tradeoff for the
   team to see and decide on, not yours to resolve silently either way.
6. Write your own tests for training.py/rllib_env.py in this project's
   house style (Arrange-Act-Assert, descriptive test_ names) -- including
   at least one test that your centralized critic actually receives
   state(), not just that the policy is parameter-shared (the existing
   smoke test only proves the latter). Run
   `uv run pytest tests/acceptance/test_acceptance_d_training_smoke.py -v`
   first to confirm your Ray install works at all before writing more.

IMPORTANT: Acceptance Test D (PRD section 38) means "train then evaluate
with zero code changes in between," proven mechanically against the stub --
NOT a real convergence claim. Do not add reward-value or convergence
assertions to any smoke-style test; a random-noise stub has no signal
worth asserting on. A real 2-satellite convergence run against the actual
environment is a separate, later, human-reviewed task, not something to
fake here.

HARD RULES:
- Type hints on every function; `from __future__ import annotations`.
- Raise ValueError immediately on invalid input in reward.py -- never
  clamp or silently continue.
- SI units throughout for reward.py's numeric arguments.
- No new dependencies beyond what pyproject.toml already declares
  (numpy, scipy, gymnasium, pettingzoo, ray[rllib], torch).
- Keep reward.py and rllib_env.py well under 400 lines each.
- Use `build_algo()`, not the deprecated `build()`, and pass an explicit
  checkpoint path (e.g. a pytest `tmp_path` fixture in tests) to
  `save_to_path()` rather than leaving it to default into /tmp.
- Run `uv run ruff check orbit_guard tests` before considering anything
  done -- must be clean.

DONE means: tests/rl/test_reward.py passes in full and unmodified,
tests/acceptance/test_acceptance_d_training_smoke.py still passes, you have
a real centralized critic (not just a parameter-shared policy) wired to
state() with your own test proving it, your own training.py/rllib_env.py
tests pass,
`uv run pytest -q --ignore=tests/rl/test_observation.py --ignore=tests/safety/test_maneuver_mapper.py`
(a different teammate's parallel files, which may not exist yet) doesn't
regress anything else, ruff is clean, and your handoff message states
exactly which current RLlib API pattern you verified and used for BOTH the
Dict-observation flattening and the centralized critic (config class name,
RLModule vs. legacy Policy API, etc.) -- this becomes the permanent project
record. If you shipped IPPO instead of a true centralized critic, say so
explicitly rather than labeling it MAPPO/CTDE. Report back with actual
command output, not just a claim that it works.
```
