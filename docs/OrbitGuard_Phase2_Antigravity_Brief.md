# OrbitGuard Phase 2, Part B — Conjunction Geometry & Risk Proxy

**Written for:** the OrbitGuard teammate implementing this half of Phase 2 with the Antigravity coding agent — not for Claude, not for the PRD's original "Astro" role. If you're a human reading this before pointing an agent at it, read the whole thing once; it's shorter than it looks because most of the actual specification lives in the test files, not in this prose.

**Status when this was written:** OrbitGuard's Phase 0 (repo/config/data setup) and Phase 1 (physics foundation — orbital dynamics, integrator, frame transforms, constellation generator) are complete, tested (61 tests, 100% coverage), and committed. You are building on top of real, working code, not a stub.

---

## 1. What OrbitGuard is (just enough to orient you)

A research simulator for multi-agent satellite collision avoidance. Satellites orbit Earth; occasionally two get dangerously close (a "conjunction"); a learned policy (built in a later phase) decides whether to nudge a satellite out of the way; a safety shield double-checks that decision before it's applied. The full spec is `docs/OrbitGuard_Final_PRD_v1.md` — you do not need to read all of it. The sections that matter for your half are §17 (candidate screening), §18 (collision), §19 (TCA), §20 (candidate pairs), §21 (risk proxy) — all short.

**What already exists and you'll import directly, without needing to understand its internals:**

| Module | What it gives you |
|---|---|
| `orbit_guard.physics.state.SatelliteState` | Immutable 6-vector (position + velocity) representing one satellite at one instant |
| `orbit_guard.physics.dynamics.state_derivative` | The ODE right-hand side (two-body + J2 gravity) |
| `orbit_guard.physics.integrator.propagate(state_vector, dt, n_steps, derivative_fn)` | Given a starting state, returns a `(n_steps+1, 6)` NumPy array — every row is `[x, y, z, vx, vy, vz]` at one time sample, `dt` seconds apart |

That return shape — **`(n_steps+1, 6)`, columns `[x, y, z, vx, vy, vz]`, rows spaced `dt` seconds apart** — is the *only* data format you need to know about from the rest of the system. Every function you're building consumes exactly this shape. You do not need to know how positions get computed, what J2 is, or anything about orbital mechanics beyond "two arrays of `(x,y,z)` points over time."

## 2. Your scope, precisely

Four small modules, all under `orbit_guard/conjunction/`:

| File | Job | PRD ref |
|---|---|---|
| `orbit_guard/conjunction/screening.py` | List every candidate satellite pair to check | §20 |
| `orbit_guard/conjunction/tca.py` | Find time-of-closest-approach and minimum separation between two trajectories | §19 |
| `orbit_guard/conjunction/collision.py` | Ground-truth physical collision check | §18 |
| `orbit_guard/conjunction/risk.py` | Bounded risk score from a miss distance and an uncertainty | §21 |

**That's it. Do not touch anything outside `orbit_guard/conjunction/` and its matching `tests/conjunction/`.** In particular:

- Don't touch `orbit_guard/sensing/`, `orbit_guard/safety/`, or `orbit_guard/scenarios/conjunction_generator.py` (doesn't exist yet) — someone else owns those, in parallel, right now.
- Don't touch anything under `orbit_guard/physics/` — it's done, tested, and frozen for this phase.
- Don't modify `configs/default.yaml`.

## 3. The tests are the spec — read them first, literally

**`tests/conjunction/test_tca.py`, `test_collision.py`, `test_risk.py`, `test_screening.py` are already written and already committed.** They currently fail with `ModuleNotFoundError` because the four modules above don't exist yet. That's intentional — this project follows strict test-first development (write the test, watch it fail for the right reason, then implement until it passes), and these four files *are* your assignment stated as executable code instead of prose.

**Your job is to make all of them pass, without editing them.** Run:

```bash
uv run pytest tests/conjunction/ -v
```

Right now that prints 4 `ModuleNotFoundError`s. When you're done, it should print all tests passing (currently 26 test cases across the four files).

**If you think a test is wrong** (mis-specified, physically incorrect, testing the wrong thing): don't silently change it. Flag it back to the team with your specific objection. These tests were themselves verified against a working reference implementation before being handed to you, so treat a disagreement as something to raise, not something to route around — but "verified" doesn't mean infallible, so a real objection is worth raising loudly rather than swallowing.

Read each test file top to bottom before writing any code. Each one's module docstring states the exact formula or rule from the PRD. Here's a compressed version of what's in them:

### 3.1 `screening.py` — candidate pairs (§20)

```python
def generate_candidate_pairs(n: int) -> tuple[tuple[int, int], ...]:
    """All N(N-1)/2 unique (i, j) index pairs with i < j. Raises ValueError if n < 2."""
```
At N=10–20 satellites, exhaustive pairing (105 pairs at N=15) is fine — no spatial indexing needed. This is the simplest of the four; a good one to start with to get oriented.

### 3.2 `tca.py` — time of closest approach (§19)

```python
def compute_tca(trajectory_i: np.ndarray, trajectory_j: np.ndarray, dt: float) -> tuple[float, float]:
    """Returns (t_tca_seconds, d_min_meters).

    t_TCA = argmin_t ||r_i(t) - r_j(t)||   over the provided samples
    d_min = ||r_i(t_TCA) - r_j(t_TCA)||

    t_tca is measured in seconds from the start of the trajectories (row 0 == t=0).
    Raises ValueError if the two trajectories don't have matching shapes.
    """
```
A discrete argmin over the sampled distance is sufficient to pass the tests (the PRD mentions optional local interpolation refinement as a nice-to-have, not a requirement — don't build that unless the discrete version is passing and you have time left over).

### 3.3 `collision.py` — physical collision, ground truth (§18)

```python
@dataclass(frozen=True)
class CollisionResult:
    collided: bool
    collision_time_s: float | None  # None iff collided is False
    min_separation_m: float

def detect_collision(
    trajectory_i: np.ndarray, trajectory_j: np.ndarray, dt: float,
    hard_body_radius_i_m: float, hard_body_radius_j_m: float,
) -> CollisionResult:
    """Collided iff ||r_i(t) - r_j(t)|| < R_i + R_j (STRICT <) at any sampled t.
    collision_time_s is the FIRST such instant, in seconds from t=0.
    """
```
Note the strict inequality — exactly-touching does not count as a collision per the PRD. There's a test for this exact boundary; don't special-case it away, get the comparison operator right.

### 3.4 `risk.py` — bounded risk proxy (§21)

```python
def risk_proxy(d_min_m: float, sigma_rel_m: float, epsilon_m: float = 1.0, z0: float = 3.0, k: float = 1.5) -> float:
    """z = d_min_m / (sigma_rel_m + epsilon_m)
    risk = sigmoid(k * (z0 - z)),  in [0, 1]

    Bounded; monotonically decreasing in d_min; monotonically increasing in
    sigma_rel; deterministic; raises ValueError on any negative input.
    """
```

**Two things that will bite you if you don't read them now:**

1. **Naming.** This is a bounded proxy, **never** a probability of collision. Do not name anything `p_collision`, `probability_of_collision`, or similar — anywhere, including local variables and comments. There's a test that checks the module doesn't expose those names. This isn't a style nitpick — it's a hard project-wide rule (the paper this project produces must never claim to compute a formal collision probability), and this rule shows up throughout the whole codebase, not just here.
2. **Numerical stability.** A naive `1 / (1 + math.exp(-x))` **will crash with `OverflowError`** for realistic inputs — a well-separated, safe satellite pair (e.g. `d_min=5000m, sigma_rel=0.1m`) produces `z ≈ 4545`, which overflows `math.exp`. Use `scipy.special.expit(x)` instead (`scipy` is already a project dependency) — it computes the same sigmoid but is numerically safe for any input magnitude, correctly underflowing to exactly `0.0` or `1.0` at the extremes instead of crashing. This isn't a hypothetical edge case; the test suite specifically checks it (`test_risk_proxy_is_bounded_in_zero_one` sweeps `d_min` up to 50,000m).

## 4. Project conventions you're inheriting (not optional)

These apply throughout the whole codebase, not just your part — matching them means your code merges in without a review round-trip:

- **Type hints on every function signature.** (`from __future__ import annotations` is used elsewhere in the codebase; feel free to use it too.)
- **Immutable data**: `CollisionResult` above is a frozen dataclass — never mutate it, construct a new one.
- **Fail loudly, never silently continue**: invalid input (negative radius, mismatched array shapes, etc.) raises `ValueError` immediately. Don't clamp, don't warn-and-continue, don't return a sentinel.
- **SI units always**: meters, seconds, m/s — matching what `propagate()` already gives you. Name variables with the unit suffix (`_m`, `_s`, `_mps`) as the existing code does.
- **No magic numbers**: the risk formula's defaults (`epsilon_m=1.0`, `z0=3.0`, `k=1.5`) are already in `configs/default.yaml` under `risk:` — they're given to you as function defaults so you don't need to read that file, but don't invent a different default independently.
- **Files stay under ~400 lines** (800 hard cap) — none of your four modules should get anywhere near this if you keep them focused on their one job.
- **No new dependencies beyond what's already declared** (`numpy`, `scipy`) without checking first — you shouldn't need any others for this scope.

## 5. Running things

```bash
# from the repo root
uv sync                                    # installs the exact pinned dependency versions
uv run pytest tests/conjunction/ -v        # your tests
uv run pytest -q                           # the WHOLE suite — should stay green throughout
uv run ruff check orbit_guard tests        # lint — must be clean
uv run pytest --cov=orbit_guard --cov-report=term-missing   # coverage
```

If you don't have `uv`, standard `pip install -e .` from `pyproject.toml` plus `pytest`/`scipy`/`numpy` also works — `uv` is just what this project standardized on.

**Do not let the full suite (`uv run pytest -q`, currently 61 tests outside your scope) regress.** If something you did breaks a Phase 1 test, you've touched something outside your boundary — stop and check what you imported or modified.

## 6. Definition of done

- [ ] `uv run pytest tests/conjunction/ -v` — all tests pass, none skipped, none modified from what was handed to you
- [ ] `uv run pytest -q` — full suite still green (currently 61 + your new passing tests)
- [ ] `uv run ruff check orbit_guard tests` — clean
- [ ] `uv run pytest --cov=orbit_guard --cov-report=term-missing` — your four new files at or near 100% (matching the standard the rest of the codebase already holds itself to; the project-wide minimum is 80%)
- [ ] No file outside `orbit_guard/conjunction/` was created or modified
- [ ] `risk_proxy` never crashes regardless of input magnitude (re-check this one specifically — it's the easiest to get subtly wrong)

## 7. Handing it back

Work on a branch (e.g. `phase2-conjunction-risk`), commit only the four new implementation files (the test files should show as unmodified in your diff), and push / open a PR / hand the branch back however your team's already sharing this repo. Include in your PR description or handoff message: which of the two extra-credit items you attempted (interpolated TCA refinement — not required), and confirmation that all six Definition-of-Done boxes above are checked, not just "I think it's done."

---

## Appendix: prompt for Antigravity

Copy everything between the lines below directly into Antigravity as your first message. It's self-contained and repeats the essential constraints so the agent doesn't need this whole document loaded to get started correctly — though pointing it at this file too (`docs/OrbitGuard_Phase2_Antigravity_Brief.md`) as additional context is a good idea if Antigravity supports attaching repo files.

```
You are implementing four small Python modules in the OrbitGuard repository,
under orbit_guard/conjunction/. This is a research simulator for satellite
collision avoidance; you only need the local context below, not the whole
project.

SCOPE — create exactly these four files, nothing else:
  orbit_guard/conjunction/screening.py
  orbit_guard/conjunction/tca.py
  orbit_guard/conjunction/collision.py
  orbit_guard/conjunction/risk.py

Do not create, edit, or delete any other file in the repository, especially
not anything under tests/. Do not touch orbit_guard/physics/, sensing/,
safety/, or scenarios/ -- those are owned by teammates working in parallel.

THE SPEC IS THE TEST SUITE. Four test files already exist and currently fail
with ModuleNotFoundError:
  tests/conjunction/test_screening.py
  tests/conjunction/test_tca.py
  tests/conjunction/test_collision.py
  tests/conjunction/test_risk.py

Read each one fully before writing any implementation code. Your job is to
make every test in all four files pass, by implementing the four modules --
never by editing the tests. Run `uv run pytest tests/conjunction/ -v` to see
current status, and `uv run pytest -q` to confirm you haven't broken anything
else in the repo (it should show 61 passing tests before you start, and only
grow from there).

FUNCTION SIGNATURES REQUIRED (see the test files for exact behavior):

  def generate_candidate_pairs(n: int) -> tuple[tuple[int, int], ...]
      # All N(N-1)/2 pairs (i,j) with i<j; raises ValueError if n<2.

  def compute_tca(trajectory_i: np.ndarray, trajectory_j: np.ndarray, dt: float) -> tuple[float, float]
      # Returns (t_tca_seconds, d_min_meters). Both trajectories are
      # (n_steps+1, 6) arrays, columns [x,y,z,vx,vy,vz], rows dt seconds
      # apart. t_TCA = argmin_t ||r_i(t)-r_j(t)|| over the samples;
      # d_min is the separation there. Raise ValueError on shape mismatch.

  @dataclass(frozen=True)
  class CollisionResult:
      collided: bool
      collision_time_s: float | None
      min_separation_m: float

  def detect_collision(trajectory_i, trajectory_j, dt, hard_body_radius_i_m, hard_body_radius_j_m) -> CollisionResult
      # Collided iff ||r_i(t)-r_j(t)|| < R_i+R_j (STRICT less-than) at ANY
      # sample; collision_time_s is the FIRST such time. Raise ValueError on
      # shape mismatch.

  def risk_proxy(d_min_m: float, sigma_rel_m: float, epsilon_m: float = 1.0, z0: float = 3.0, k: float = 1.5) -> float
      # z = d_min_m / (sigma_rel_m + epsilon_m)
      # risk = sigmoid(k * (z0 - z)), bounded to [0,1].
      # MUST use scipy.special.expit, NOT a hand-written 1/(1+exp(-x)) --
      # the latter raises OverflowError for realistic well-separated inputs
      # (e.g. d_min=5000, sigma_rel=0.1). Raise ValueError on any negative
      # input. Never name anything p_collision or probability_of_collision
      # anywhere in this file -- this is an explicitly bounded proxy, never
      # a formal probability, and that naming rule is enforced by a test.

HARD RULES:
- Type hints on every function.
- Raise ValueError immediately on invalid input (negative radius, mismatched
  array shapes, n<2, etc.) -- never clamp or silently continue.
- SI units throughout: meters, seconds, m/s.
- No new dependencies beyond numpy and scipy (both already available).
- Keep each file well under 400 lines.
- Run `uv run ruff check orbit_guard tests` before considering anything done
  -- must be clean.

DONE means: all four test files pass, the full existing suite (uv run pytest
-q) still passes with nothing regressed, ruff is clean, and coverage on your
four new files is at or near 100%. Report back with the pytest output showing
all green, not just a claim that it works.
```
