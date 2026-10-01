# OrbitGuard Phase 3, Part B — Maneuver Mapping & Observation

**Written for:** the OrbitGuard teammate implementing this part of Phase 3 with their own Antigravity coding agent — not for Claude, not for the PRD's original "RL"/"Safety" roles as a literal org chart. If you're a human reading this before pointing an agent at it, read the whole thing once; it's shorter than it looks because most of the actual specification lives in the test files, not in this prose. This is a self-contained document — you do not need Part C's brief to do this work, and Part C does not need this one.

**Status when this was written:** OrbitGuard's Phase 0 (setup), Phase 1 (physics foundation), and Phase 2 (sensing/belief, adaptive margin, conjunction injection, TCA/collision/risk) are complete, tested (165 tests, 99% coverage), and merged. Phase 3 (RL Environment & 2-Agent Sanity Check — the project's highest-risk **GATE** phase) is split three ways: **Part A** (`rl/environment.py`, `physics/mission_deviation.py`) stays with the teammate using Claude and has **not started implementation yet** — you are not blocked on it. **Part C** (`rl/reward.py`, `rl/training.py`, a separate teammate's own Antigravity agent) is fully specified in its own separate brief and runs in parallel with you — you don't need to coordinate step-by-step, just avoid touching each other's files (see §3.1).

---

## 1. What OrbitGuard is (just enough to orient you)

A research simulator for multi-agent satellite collision avoidance. Satellites orbit Earth; occasionally two get dangerously close (a "conjunction"); a learned policy (MAPPO, parameter-shared across all satellites) decides whether to nudge a satellite out of the way; a safety shield (Phase 4, not yet built) will double-check that decision before it's applied. Phase 3's job is to prove the **learning loop itself** works end-to-end at the smallest possible scale (2 satellites) before committing to scaling it up. The full spec is `docs/OrbitGuard_Final_PRD_v1.md` — you do not need to read all of it. The sections that matter for your scope: §23 (maneuver mapping) and §24 (observation).

**What already exists and you can import directly, without needing to understand its internals:**

| Module | What it gives you |
|---|---|
| `orbit_guard.physics.constants.{MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M}` | Two physical constants, used in `rl/observation.py`'s normalization scales |
| `orbit_guard.physics.frames.{eci_vector_to_rtn, rtn_vector_to_eci, coe_to_cartesian}` | RTN↔ECI frame transforms (both directions — `maneuver_mapper.py` needs `rtn_vector_to_eci` to convert the Δv it computes back into the frame `position_m`/`velocity_mps` arrived in), orbital-elements → Cartesian state |
| `orbit_guard.physics.dynamics.state_derivative`, `orbit_guard.physics.integrator.propagate` | The ODE right-hand side and RK4 propagator, needed only for `maneuver_mapper.py`'s drift-direction test |

**You do not need `orbit_guard.sensing`, `orbit_guard.conjunction`, or `orbit_guard.scenarios`** — those are Phase 2, already done, and your two files don't import them directly. Part A's future `rl/environment.py` is the only module that will wire everything together; you're building one piece of that wiring, tested in isolation.

---

## 2. The tests are the spec — read them first, literally

Both files below are **already written and already committed**, and currently fail with `ModuleNotFoundError` because the modules you're building don't exist yet. That's intentional — this project follows strict test-first development, and these test files *are* your assignment stated as executable code instead of prose. **Each was independently pre-verified against a throwaway reference implementation (written, run to 100% pass, then deleted) before being handed to you** — so if something seems to fail for a plausible-and-correct implementation, treat that as worth flagging loudly, not silently routing around; these specs have already been checked to be achievable, but "checked" isn't "infallible."

**Your job is to make your tests pass, without editing them.** If you think a test is wrong (mis-specified, physically incorrect, testing the wrong thing): don't silently change it. Flag it back to the team with your specific objection.

Read each test file's **module docstring** first — every design decision (signs, normalization constants, tie-breaks, why a field is named what it's named) is explained there, not just asserted in the test bodies.

---

## 3. Your scope — `safety/maneuver_mapper.py` + `rl/observation.py`

### 3.1 Your two files

| File | Job | PRD ref | Spec |
|---|---|---|---|
| `orbit_guard/safety/maneuver_mapper.py` | Map one of 6 discrete maneuver intents to an RTN-frame Δv, in OG-ECI | §23 | `tests/safety/test_maneuver_mapper.py` (18 tests) |
| `orbit_guard/rl/observation.py` | Build the fixed-size, per-agent, decentralized observation dict | §24 | `tests/rl/test_observation.py` (23 tests) |

**Do not touch anything outside these two files and nothing outside `orbit_guard/safety/` and `orbit_guard/rl/` respectively.** In particular: don't touch `orbit_guard/physics/` (done, frozen), don't touch `pyproject.toml` or `uv.lock` (Part A already added the Phase 3 dependencies — gymnasium, pettingzoo, ray[rllib], torch — touching either file risks a merge conflict with the parallel Part C PR), don't touch `orbit_guard/rl/__init__.py` (Part C also adds files under `orbit_guard/rl/`), don't touch `orbit_guard/rl/reward.py`, `orbit_guard/rl/training.py`, or `orbit_guard/rl/rllib_env.py` (Part C's scope, a separate teammate working in parallel right now).

### 3.2 `maneuver_mapper.py` — compressed spec

```python
from enum import IntEnum

class ManeuverIntent(IntEnum):
    NO_MANEUVER = 0
    RAISE_ORBIT = 1
    LOWER_ORBIT = 2
    PHASE_FORWARD = 3
    PHASE_BACKWARD = 4
    RADIAL_ADJUSTMENT = 5

def map_intent_to_delta_v(
    intent: ManeuverIntent,
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    base_delta_v_mps: float = 0.1,
    phase_fraction: float = 0.5,
) -> np.ndarray:
    """Returns a 3-vector Δv in OG-ECI (same frame as position_m/velocity_mps)."""
```

Frozen RTN table (from the test file's own docstring — this IS the spec, re-derive nothing):

| Intent | RTN direction | Magnitude |
|---|---|---|
| `NO_MANEUVER` | — | 0 |
| `RAISE_ORBIT` | +T (prograde) | `base_delta_v` |
| `LOWER_ORBIT` | −T (retrograde) | `base_delta_v` |
| `PHASE_FORWARD` | −T (retrograde, fine) | `phase_fraction * base_delta_v` |
| `PHASE_BACKWARD` | +T (prograde, fine) | `phase_fraction * base_delta_v` |
| `RADIAL_ADJUSTMENT` | +R (radially outward) | `base_delta_v` |

Two things the test file checks that are easy to get subtly wrong:

1. **The energy-direction tests do NOT round-trip through `eci_vector_to_rtn`** — they use vis-viva specific orbital energy (`v²/2 - mu/r`) independently, so a bug in the RTN conversion itself can't hide behind its own inverse. A prograde (+T) burn must *increase* specific energy; retrograde must *decrease* it. If your implementation makes a test like this fail while the frozen-table test passes, you likely have the direction right in RTN-space but something about the frame conversion itself is off.
2. **`test_phase_forward_drifts_the_satellite_ahead_of_an_unmaneuvered_twin`** runs an actual 5-orbit propagation and asserts a >1000m forward drift (empirically measured at +4216.30m for the reference implementation) — this is a real physics consequence of the sign convention, not an arbitrary assertion. If your implementation gets `PHASE_FORWARD` and `PHASE_BACKWARD` backwards, this test will fail with the satellite drifting the wrong way, which is a much clearer signal than a sign flip buried in a table check.

There's also a static test (`test_safety_package_never_imports_rl_or_training_frameworks`) scanning every `.py` file under `orbit_guard/safety/` and failing if any of them import `orbit_guard.rl`, `gymnasium`, `pettingzoo`, `ray`, or `torch` — `maneuver_mapper.py` is a pure geometry function and must stay that way.

### 3.3 `observation.py` — compressed spec

```python
@dataclass(frozen=True)
class NeighborCandidate:
    relative_position_m: np.ndarray       # OWN agent's RTN frame, not raw ECI (see below)
    relative_velocity_mps: np.ndarray     # same frame
    d_min_m: float
    risk_proxy: float                      # already in [0,1]
    d_safe_m: float
    uncertainty_summary_m: float

def build_observation(
    own_position_m, own_velocity_mps,
    own_remaining_delta_v_mps, own_mission_deviation,
    own_max_delta_v_per_action_mps, own_maneuver_enabled,
    neighbor_candidates: Sequence[NeighborCandidate],
    k_neighbors: int = 4,
) -> dict[str, np.ndarray]:
    """Returns {"own": (10,), "neighbors": (k,10), "global_summary": (4,)}, all float64."""
```

Field layout (index order matters, see the test file for the exact byte-for-byte assertions):

- `own` (10,): `[pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, remaining_dv, mission_deviation, max_dv_per_action, maneuver_enabled]` — position/velocity normalized by `POSITION_SCALE_M`/`VELOCITY_SCALE_MPS`; `remaining_dv`/`max_dv_per_action` normalized by `DELTA_V_BUDGET_SCALE_MPS`/`MAX_DELTA_V_PER_ACTION_SCALE_MPS`; `mission_deviation` passed through **unscaled** (it arrives already normalized — see point 4 below); `maneuver_enabled` as 0.0/1.0.
- `neighbors` (k,10) per slot: `[relpos_x, relpos_y, relpos_z, relvel_x, relvel_y, relvel_z, risk_proxy, d_safe, uncertainty, neighbor_valid]` — `risk_proxy` passed through unscaled (already bounded [0,1]); `d_safe`/`uncertainty` normalized by `RELATIVE_POSITION_SCALE_M`; `neighbor_valid` is 1.0 for a real candidate, 0.0 for a zero-padded unused slot.
- `global_summary` (4,): `[threatening_neighbor_count, max_risk, total_risk, min_separation_norm]` — `threatening_neighbor_count` is a **raw count** (not a fraction!) of candidates with `risk_proxy > THREATENING_RISK_THRESHOLD` (0.5), computed over **all** candidates passed in, not just the top-k kept. `max_risk`/`total_risk` are likewise over all candidates. `min_separation_norm` is the smallest `d_min_m` over all candidates, normalized by `MINIMUM_SEPARATION_SCALE_M`, defaulting to 1.0 (maximally separated) when there are zero candidates.

**Frozen constants** (export these exact names from your module — anything else in this project needing the same scale imports them from you, never redefines them):

```python
POSITION_SCALE_M = R_EARTH_EQUATORIAL_M                          # 6378137.0
VELOCITY_SCALE_MPS = sqrt(MU_EARTH_M3_S2 / R_EARTH_EQUATORIAL_M)  # ~7909.79 m/s
RELATIVE_POSITION_SCALE_M = 5000.0
RELATIVE_VELOCITY_SCALE_MPS = 1000.0
DELTA_V_BUDGET_SCALE_MPS = 2.0
MAX_DELTA_V_PER_ACTION_SCALE_MPS = 0.1
MINIMUM_SEPARATION_SCALE_M = 5000.0
THREATENING_RISK_THRESHOLD = 0.5
```

**Four things that will bite you if you don't read the test docstring first:**

1. **Ranking is by `risk_proxy` descending, tie-broken by `d_min_m` ascending — never by satellite index.** `risk_proxy` correctly underflows to exactly 0.0 for well-separated pairs (confirmed in Phase 2), so most naturally-sampled candidates will tie at 0.0 — the tie-break is what determines slot order in that overwhelmingly common case, and it matters for reproducibility, not just an edge case.
2. **The field is `neighbor_valid`, not `mask`.** This is deliberate, not a naming preference — PRD §27's action masking is a *different, unused* concept in this project, and reusing "mask" here would be confusing in code review.
3. **`relative_position_m`/`relative_velocity_mps` are already in the owning agent's own RTN frame** (via `physics.frames.eci_vector_to_rtn`) by the time they reach you — Part A's environment computes this, not you. Your function does no frame transform and must not import `orbit_guard.physics.frames`.
4. **`own_mission_deviation` arrives already normalized** — it's the scalar `D_mission` from PRD §25's formula, computed upstream by Part A's environment. Pass it through; do not re-derive, re-normalize, or clip it. (This scalar is numerically related to Part C's `rl/reward.py::compute_mission_term` — a cross-module contract Part A's environment is responsible for, not something you need to coordinate on directly.)

**Dtype note:** your function returns plain `float64` numpy arrays. Part A's environment is responsible for casting to `float32` when wrapping this into the Gymnasium `Box` space RLlib expects — that's not your concern.

---

## 4. Project conventions (not optional)

Same as every prior phase in this codebase — matching them means your code merges in without a review round-trip:

- **Type hints on every function signature**, `from __future__ import annotations` used throughout.
- **Immutable data**: `NeighborCandidate` is a frozen dataclass — never mutate, construct a new one.
- **Fail loudly, never silently continue**: invalid input raises `ValueError` immediately (see `k_neighbors < 1`, invalid `ManeuverIntent`, negative `base_delta_v_mps`, etc. in the test files for the exact cases). Don't clamp, don't warn-and-continue.
- **SI units always**: meters, seconds, m/s. Variable names carry unit suffixes (`_m`, `_s`, `_mps`) as existing code does.
- **No magic numbers**: every numeric default in this spec is already in `configs/default.yaml` (`observation.k_neighbors`, `maneuver.base_delta_v_mps`, `maneuver.phase_fraction`, `capability.remaining_delta_v_mps`) — they're given to you as function defaults so you don't need to read that file, but don't invent a different default independently.
- **Files stay under ~400 lines** (800 hard cap) — neither of your two modules should get anywhere near this if kept focused.
- **No new dependencies beyond what's already declared.** `pyproject.toml` already has `numpy`, `scipy`, `gymnasium`, `pettingzoo`, `ray[rllib]`, `torch` — you shouldn't need anything else. If you genuinely do, ask first; don't edit `pyproject.toml`/`uv.lock` yourself (merge-conflict risk against the parallel Part C PR).

---

## 5. Running things

```bash
# from the repo root
uv sync                                                    # installs pinned dependency versions

uv run pytest tests/safety/test_maneuver_mapper.py tests/rl/test_observation.py -v

# confirm the REST of the repo is healthy before you start:
uv run pytest -q --ignore=tests/rl/test_observation.py --ignore=tests/rl/test_reward.py --ignore=tests/safety/test_maneuver_mapper.py
# -> should show 197 passing (165 Phase-0-2 baseline + 9 stub-env conformance + 3 centralized-critic-state + 3 Acceptance-D smoke + 17 mission-deviation), with zero collection errors

uv run ruff check orbit_guard tests
uv run pytest --cov=orbit_guard --cov-report=term-missing
```

A bare `uv run pytest -q` (no `--ignore`) will show 3 collection errors (`ModuleNotFoundError` for `maneuver_mapper`, `observation`, and a separate teammate's `reward.py`) until all of Phase 3's parts land — that's the expected starting RED state, not a broken repo.

**`uv sync` pulls `torch` with CUDA dependencies, which is multi-gigabyte on Linux even though you never use torch directly** (it's a transitive dependency of `ray[rllib]`, needed by the parallel Part C work, not by you). Make sure your environment has the disk space/bandwidth for it, and budget time for the first `uv sync`.

**Your own full-suite check depends on Ray being runnable** (`tests/acceptance/test_acceptance_d_training_smoke.py` is part of the baseline and needs a working local Ray instance). If Ray won't start in your environment (e.g. a constrained sandbox or native Windows without WSL — Ray's scheduler has limited native-Windows support), use `--ignore=tests/acceptance/test_acceptance_d_training_smoke.py` as a fallback for the "confirm the rest of the repo is healthy" check, and flag it to the team — your own two files don't need Ray at all, this only affects the full-suite sanity check.

---

## 6. Definition of done

- [ ] `uv run pytest tests/safety/test_maneuver_mapper.py tests/rl/test_observation.py -v` — all 41 tests pass, none skipped, neither file modified from what was handed to you
- [ ] `uv run pytest -q --ignore=tests/rl/test_reward.py` — 197 + 41 = 238 tests, all green, zero collection errors (Part C's `reward.py` may still be in progress in parallel — a bare `pytest -q` with no ignore will still show 1 collection error until Part C also lands, that's expected, not your bug). If Ray won't run in your environment, use `--ignore=tests/rl/test_reward.py --ignore=tests/acceptance/test_acceptance_d_training_smoke.py` instead and flag it.
- [ ] `uv run ruff check orbit_guard tests` — clean
- [ ] `uv run pytest --cov=orbit_guard --cov-report=term-missing` — your two new files at or near 100%
- [ ] No file outside `orbit_guard/safety/maneuver_mapper.py` and `orbit_guard/rl/observation.py` was created or modified
- [ ] `pyproject.toml`/`uv.lock`/`orbit_guard/rl/__init__.py` untouched

---

## 7. Handing it back

Work on a branch (e.g. `phase3-observation-safety`), commit only your two new implementation files (the test files should show as unmodified in your diff), and push / open a PR / hand the branch back however the team's already sharing this repo. Include in your PR description or handoff message: confirmation that every Definition-of-Done box above is checked, with actual command output, not just "I think it's done."

---

## Appendix: prompt for Antigravity

Copy everything between the lines below directly into Antigravity as your first message.

```
You are implementing two Python modules in the OrbitGuard repository:
orbit_guard/safety/maneuver_mapper.py and orbit_guard/rl/observation.py.
This is a research simulator for satellite collision avoidance; you only
need the local context below, not the whole project.

SCOPE -- create exactly these two files, nothing else:
  orbit_guard/safety/maneuver_mapper.py
  orbit_guard/rl/observation.py

Do not create, edit, or delete any other file in the repository, especially
not anything under tests/, not pyproject.toml, not uv.lock, and not
orbit_guard/rl/__init__.py (a teammate's parallel PR also adds files under
orbit_guard/rl/ -- touching shared files risks a merge conflict). Do not
touch orbit_guard/physics/, orbit_guard/sensing/, orbit_guard/conjunction/,
orbit_guard/scenarios/, orbit_guard/rl/reward.py, orbit_guard/rl/training.py,
or orbit_guard/rl/rllib_env.py -- those are done or owned by a different
teammate working in parallel right now.

THE SPEC IS THE TEST SUITE. Two test files already exist and currently fail
with ModuleNotFoundError:
  tests/safety/test_maneuver_mapper.py   (18 tests)
  tests/rl/test_observation.py            (23 tests)

Read each one FULLY, starting with its module docstring, before writing any
implementation code -- the docstrings explain every sign convention,
normalization constant, and naming decision, not just the test bodies. Your
job is to make every test in both files pass, by implementing the two
modules -- never by editing the tests. Run
`uv run pytest tests/safety/test_maneuver_mapper.py tests/rl/test_observation.py -v`
to see current status. Before you've created your files, a bare
`uv run pytest -q` shows 3 COLLECTION ERRORS (including one for a different
teammate's tests/rl/test_reward.py, which is not your job) -- that's
expected. To confirm the rest of the repo is healthy before you start, run:
`uv run pytest -q --ignore=tests/rl/test_observation.py --ignore=tests/rl/test_reward.py --ignore=tests/safety/test_maneuver_mapper.py`
-- it should show 197 passing with zero collection errors (this baseline count will keep growing as Part A lands more of its own work -- trust the "zero collection errors outside your own 3 known ones" check over a hardcoded number if they ever drift apart).

FUNCTION SIGNATURES REQUIRED (see the test files for exact behavior):

  class ManeuverIntent(IntEnum):
      NO_MANEUVER = 0; RAISE_ORBIT = 1; LOWER_ORBIT = 2
      PHASE_FORWARD = 3; PHASE_BACKWARD = 4; RADIAL_ADJUSTMENT = 5

  def map_intent_to_delta_v(intent, position_m, velocity_mps,
                             base_delta_v_mps=0.1, phase_fraction=0.5) -> np.ndarray
      # Returns a 3-vector delta-v in OG-ECI (same frame as the inputs).
      # RTN table: RAISE_ORBIT=+T*base_dv, LOWER_ORBIT=-T*base_dv,
      # PHASE_FORWARD=-T*phase_fraction*base_dv, PHASE_BACKWARD=+T*phase_fraction*base_dv,
      # RADIAL_ADJUSTMENT=+R*base_dv, NO_MANEUVER=zero vector.
      # Use orbit_guard.physics.frames (eci_vector_to_rtn, rtn_vector_to_eci)
      # for the RTN<->ECI conversion.
      # Raise ValueError on an invalid intent or negative base_delta_v_mps.

  @dataclass(frozen=True)
  class NeighborCandidate:
      relative_position_m: np.ndarray      # already in the OWN agent's RTN frame
      relative_velocity_mps: np.ndarray
      d_min_m: float
      risk_proxy: float                     # already in [0,1]
      d_safe_m: float
      uncertainty_summary_m: float

  def build_observation(own_position_m, own_velocity_mps,
      own_remaining_delta_v_mps, own_mission_deviation,
      own_max_delta_v_per_action_mps, own_maneuver_enabled,
      neighbor_candidates, k_neighbors=4) -> dict[str, np.ndarray]
      # Returns {"own": (10,), "neighbors": (k,10), "global_summary": (4,)},
      # all plain float64 numpy arrays (no gymnasium/pettingzoo import in
      # this file at all). Field layouts, the exact frozen normalization
      # constants (POSITION_SCALE_M, VELOCITY_SCALE_MPS,
      # RELATIVE_POSITION_SCALE_M, RELATIVE_VELOCITY_SCALE_MPS,
      # DELTA_V_BUDGET_SCALE_MPS, MAX_DELTA_V_PER_ACTION_SCALE_MPS,
      # MINIMUM_SEPARATION_SCALE_M, THREATENING_RISK_THRESHOLD=0.5), and
      # every sign/ranking/tie-break/padding rule are all in the test
      # file's module docstring -- read it before coding, it's short.
      # Key traps: rank neighbors by risk_proxy DESCENDING, tie-break by
      # d_min_m ASCENDING (never satellite index); the padding-mask field
      # is named "neighbor_valid" not "mask"; threatening_neighbor_count
      # is a RAW COUNT over ALL candidates, not a fraction; own_mission_deviation
      # passes through unscaled, do not re-normalize it.

HARD RULES:
- Type hints on every function; `from __future__ import annotations`.
- Raise ValueError immediately on invalid input -- never clamp or silently
  continue.
- SI units throughout: meters, seconds, m/s, matching existing code's
  `_m`/`_s`/`_mps` naming suffix convention.
- No new dependencies beyond what pyproject.toml already declares.
- Keep each file well under 400 lines.
- Run `uv run ruff check orbit_guard tests` before considering anything
  done -- must be clean.

DONE means: both test files pass in full,
`uv run pytest -q --ignore=tests/rl/test_reward.py` (a different teammate's
parallel work, which may not exist yet) shows 238 passing with nothing else
regressed, ruff is clean, and coverage on your two new files is at or near
100%. Report back with the actual pytest output showing all green, not
just a claim that it works.
```
