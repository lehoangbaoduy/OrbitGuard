"""Contract tests for orbit_guard.conjunction.risk (PRD §21).

    sigma_rel = combined relative uncertainty (a plain float, meters -- how
                it's computed from covariances is NOT this module's job)
    z = d_min / (sigma_rel + epsilon)
    risk = 1 / (1 + exp(-k * (z0 - z)))      in [0, 1]

MVP defaults (configs/default.yaml `risk:` block): epsilon_m=1.0, z0=3.0, k=1.5.

Hard rule: this proxy is NEVER a probability of collision. Do not name any
variable, docstring, log field, or test `p_collision` / `probability` --
use `risk_proxy` / `R_proxy` everywhere, including in your own code.
"""

import math

import pytest
from orbit_guard.conjunction.risk import risk_proxy

_EPSILON_M = 1.0
_Z0 = 3.0
_K = 1.5


def test_risk_proxy_matches_sigmoid_formula_exactly():
    d_min, sigma_rel = 300.0, 97.0
    z = d_min / (sigma_rel + _EPSILON_M)
    expected = 1.0 / (1.0 + math.exp(-_K * (_Z0 - z)))

    assert risk_proxy(d_min, sigma_rel) == pytest.approx(expected, rel=1e-12)


def test_risk_proxy_is_bounded_in_zero_one():
    # PRD §21 specifies the closed interval R_ij in [0, 1]. At extreme z
    # (e.g. d_min=5000m, sigma_rel=0.1m -> z~4545), a numerically STABLE
    # sigmoid legitimately saturates to exactly 0.0 or 1.0 via floating-point
    # underflow -- that is correct, expected behavior, not a bound violation.
    # (A naive `1/(1+exp(-x))` will instead raise OverflowError here --
    # use scipy.special.expit or an equivalent numerically-safe form.)
    for d_min in (0.0, 1.0, 50.0, 500.0, 5000.0, 50000.0):
        for sigma_rel in (0.1, 1.0, 10.0, 100.0, 1000.0):
            r = risk_proxy(d_min, sigma_rel)
            assert 0.0 <= r <= 1.0
            assert math.isfinite(r)  # never inf/NaN/overflow, regardless of input scale


def test_risk_proxy_is_monotonically_decreasing_in_d_min():
    sigma_rel = 100.0
    d_mins = [10.0, 50.0, 100.0, 300.0, 600.0, 1200.0]
    risks = [risk_proxy(d, sigma_rel) for d in d_mins]
    assert all(risks[i] > risks[i + 1] for i in range(len(risks) - 1))


def test_risk_proxy_is_monotonically_increasing_in_sigma_rel_at_fixed_d_min():
    # More uncertainty at the same miss distance is scarier, not safer.
    d_min = 300.0
    sigmas = [10.0, 50.0, 100.0, 300.0, 600.0]
    risks = [risk_proxy(d_min, s) for s in sigmas]
    assert all(risks[i] < risks[i + 1] for i in range(len(risks) - 1))


def test_risk_proxy_equals_one_half_at_z0_crossover():
    # By construction, risk == 0.5 exactly when z == z0, i.e. when
    # d_min == z0 * (sigma_rel + epsilon).
    sigma_rel = 97.0
    d_min = _Z0 * (sigma_rel + _EPSILON_M)
    assert risk_proxy(d_min, sigma_rel) == pytest.approx(0.5, abs=1e-9)


def test_risk_proxy_is_deterministic():
    assert risk_proxy(250.0, 80.0) == risk_proxy(250.0, 80.0)


def test_risk_proxy_default_parameters_match_reference_config():
    # configs/default.yaml risk: {epsilon_m: 1.0, z0: 3.0, k: 1.5} -- calling
    # with no kwargs must reproduce those defaults exactly, so a caller who
    # doesn't pass them still gets the frozen, reported values (PRD §21).
    d_min, sigma_rel = 300.0, 97.0
    assert risk_proxy(d_min, sigma_rel) == risk_proxy(
        d_min, sigma_rel, epsilon_m=1.0, z0=3.0, k=1.5
    )


@pytest.mark.parametrize("bad_kwargs", [
    {"d_min_m": -1.0, "sigma_rel_m": 50.0},
    {"d_min_m": 50.0, "sigma_rel_m": -1.0},
    {"d_min_m": 50.0, "sigma_rel_m": 50.0, "epsilon_m": -1.0},
])
def test_risk_proxy_raises_on_negative_inputs(bad_kwargs):
    with pytest.raises(ValueError):
        risk_proxy(**bad_kwargs)


def test_risk_proxy_never_exposed_as_probability_naming():
    # Static check on this test module's own target: the public API name
    # must be risk_proxy, not anything implying a formal probability.
    import orbit_guard.conjunction.risk as risk_module

    assert hasattr(risk_module, "risk_proxy")
    assert not hasattr(risk_module, "p_collision")
    assert not hasattr(risk_module, "probability_of_collision")
