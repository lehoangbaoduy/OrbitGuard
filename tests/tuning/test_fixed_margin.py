"""The Fixed-Margin constant follows the tuned adaptive coefficients (wave-4 review, Fixed-Margin finding).

configs/default.yaml fixes `fixed_margin.d_safe_m: 1142.0` by the progress Q30 protocol: the mean adaptive d_safe
over 12 validation scenarios at N=15 with no manoeuvres (1141.9 m), under the DEFAULT coefficients. A grid winner
that changes alpha/beta/gamma would otherwise leave the Fixed-Margin arm at a margin size the adaptive arm no
longer has, so the adaptive-vs-fixed ablation (PRD §31 #3) would confound adaptivity with margin size.
`measure_fixed_margin` reproduces Q30, and freeze_best writes its result into tuned.yaml.

Validation seeds only: the protocol reads the validation pool through configs/validation.yaml.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from orbit_guard.config import (
    DEFAULT_CONFIG_PATH,
    apply_overlay,
    load_config,
    margin_from_config,
)
from orbit_guard.evaluation.baselines import build_env_factory
from orbit_guard.rl.evaluation import EvaluationReport
from orbit_guard.safety.adaptive_margin import FixedMargin
from orbit_guard.tuning.fixed_margin import (
    PROTOCOL_NUM_AGENTS,
    PROTOCOL_SEED_COUNT,
    mean_adaptive_margin,
    measure_fixed_margin,
    no_manoeuvre_margin_inputs,
    round_fixed_margin,
    validation_seeds,
)
from orbit_guard.tuning.grid_search import (
    GridPoint,
    GridResult,
    default_grid,
    freeze_best,
)

_NARROW = {"alpha": 0.5, "beta": 0.25, "gamma": 0.5}
_WIDE = {"alpha": 2.0, "beta": 1.0, "gamma": 2.0}


def _report() -> EvaluationReport:
    return EvaluationReport(
        episodes=20, mean_return_per_agent=0.0, collision_episodes=0, maneuver_rate=0.0,
        threat_maneuver_rate=0.0, calm_maneuver_rate=0.0, threat_step_fraction=0.0,
        mean_delta_v_used_mps=0.1, mean_budget_fraction_used=0.05, unsafe_episodes=1, mean_episode_steps=360.0,
    )


def _with_margin(margin: dict) -> dict:
    return apply_overlay(load_config(), {"adaptive_margin": margin})


@pytest.fixture(scope="module")
def default_inputs():
    """The Q30 protocol's margin inputs under the default config (one full run, shared by the tests below)."""
    return no_manoeuvre_margin_inputs(load_config())


def test_the_protocol_is_q30s_twelve_validation_scenarios_at_n15():
    seeds = validation_seeds()

    assert PROTOCOL_SEED_COUNT == 12 and PROTOCOL_NUM_AGENTS == 15
    assert len(seeds) == 12
    assert all(100_000 <= seed < 200_000 for seed in seeds)


def test_the_default_config_reproduces_the_frozen_q30_constant(default_inputs):
    raw = mean_adaptive_margin(load_config(), default_inputs)

    assert 1141.4 <= raw <= 1142.0  # Q30 recorded 1141.9 m
    assert round_fixed_margin(raw) == load_config()["fixed_margin"]["d_safe_m"] == 1142.0
    # every decision point of 12 full episodes (360 steps x 105 pairs), not just the reset assessments
    assert len(default_inputs) == 12 * 360 * 105


@pytest.mark.parametrize(("margin", "low", "high", "rounded"), [
    (_NARROW, 1070.7, 1071.4, 1071.0),
    (_WIDE, 1282.8, 1284.4, 1284.0),
])
def test_narrow_and_wide_coefficients_move_the_matched_constant(default_inputs, margin, low, high, rounded):
    raw = mean_adaptive_margin(_with_margin(margin), default_inputs)

    assert low <= raw <= high
    assert round_fixed_margin(raw) == rounded


def test_the_margin_inputs_do_not_depend_on_the_coefficients_they_were_collected_under():
    seeds = validation_seeds()[:1]

    under_default = no_manoeuvre_margin_inputs(load_config(), seeds=seeds, max_decisions=15, workers=1)
    under_wide = no_manoeuvre_margin_inputs(_with_margin(_WIDE), seeds=seeds, max_decisions=15, workers=1)

    assert under_default.shape == (15 * 105, 3)
    assert np.array_equal(under_default, under_wide)


def test_measuring_in_parallel_gives_the_same_inputs_as_in_process():
    seeds = validation_seeds()[:2]

    serial = no_manoeuvre_margin_inputs(load_config(), seeds=seeds, max_decisions=3, workers=1)
    parallel = no_manoeuvre_margin_inputs(load_config(), seeds=seeds, max_decisions=3, workers=2)

    assert np.array_equal(serial, parallel)


def test_rounding_is_to_the_nearest_whole_metre_half_up():
    assert round_fixed_margin(1141.915) == 1142.0
    assert round_fixed_margin(1070.958) == 1071.0
    assert round_fixed_margin(1283.5) == 1284.0
    assert round_fixed_margin(1282.49) == 1282.0
    assert isinstance(round_fixed_margin(1000.2), float)


def test_the_grid_trains_only_the_adaptive_arm():
    # factory_for_point builds margin_from_config(config, "adaptive"): no grid point trains the fixed arm, so the
    # constant only has to follow the winner at freeze time.
    from orbit_guard.safety.adaptive_margin import AdaptiveMargin
    from orbit_guard.tuning.grid_search import factory_for_point

    for point in default_grid():
        assert isinstance(factory_for_point(point, 3, shielded=False)({})._margin_fn, AdaptiveMargin)


def test_freeze_best_writes_the_measured_constant_for_a_margin_winner(tmp_path):
    target = tmp_path / "tuned.yaml"

    freeze_best([GridResult(GridPoint("margin_wide", margin=_WIDE), _report(), 0.1, "extended")], target)

    frozen = yaml.safe_load(target.read_text())
    assert frozen["fixed_margin"] == {"d_safe_m": 1284.0}
    config = load_config(overlay_path=target)  # config_hash covers the constant
    assert config["fixed_margin"]["d_safe_m"] == 1284.0
    assert margin_from_config(config, "fixed") == FixedMargin(1284.0)
    env = build_env_factory("fixed_margin_mappo", 3, overlay_path=target)({})
    assert env._margin_fn == FixedMargin(1284.0)
    assert env._margin_fn(141.4, 0.9, 0.9) == 1284.0


@pytest.mark.parametrize("point", [
    GridPoint("default"),
    GridPoint("w4_1.5", reward_weights={"w4": 1.5}),
    GridPoint("explicit_defaults", margin={"alpha": 1.0, "beta": 0.5, "gamma": 1.0}),
])
def test_freeze_best_leaves_the_constant_alone_when_the_coefficients_are_the_defaults(tmp_path, point):
    target = tmp_path / "tuned.yaml"

    def must_not_measure(config):
        raise AssertionError("the default coefficients' constant is the frozen Q30 value")

    freeze_best([GridResult(point, _report(), 0.1, "extended")], target, measure=must_not_measure)

    assert "fixed_margin" not in yaml.safe_load(target.read_text())
    assert load_config(overlay_path=target)["fixed_margin"]["d_safe_m"] == 1142.0


def test_freeze_best_measures_under_the_winners_merged_coefficients_and_hashes_the_result(tmp_path):
    seen = []
    target = tmp_path / "tuned.yaml"

    def fake_measure(config):
        seen.append(dict(config["adaptive_margin"]))
        return 1070.958

    digest = freeze_best(
        [GridResult(GridPoint("margin_narrow_w4_1.5", reward_weights={"w4": 1.5}, margin=_NARROW), _report(), 0.1,
                    "extended")],
        target, measure=fake_measure,
    )

    assert seen == [{**load_config(DEFAULT_CONFIG_PATH)["adaptive_margin"], **_NARROW}]
    frozen = yaml.safe_load(target.read_text())
    assert frozen["fixed_margin"] == {"d_safe_m": 1071.0}
    assert frozen["config_hash"] == digest
    assert load_config(overlay_path=target)["fixed_margin"]["d_safe_m"] == 1071.0


def test_measure_fixed_margin_is_the_mean_over_the_protocol_inputs(monkeypatch):
    from orbit_guard.tuning import fixed_margin

    inputs = np.array([(141.42, 0.0, 0.0), (141.42, 0.2, 0.5)])
    monkeypatch.setattr(fixed_margin, "no_manoeuvre_margin_inputs", lambda config, **kwargs: inputs)
    config = load_config()
    adaptive = margin_from_config(config, "adaptive")

    assert measure_fixed_margin(config) == pytest.approx((adaptive(*inputs[0]) + adaptive(*inputs[1])) / 2)


def test_the_default_worker_count_is_capped_for_the_grid_clis_memory(monkeypatch):
    # Under `python -m orbit_guard.tuning.grid_search` every spawned child re-imports torch and Ray.
    from orbit_guard.tuning import fixed_margin

    sizes = []

    class _Pool:
        def __init__(self, size):
            sizes.append(size)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, function, jobs):
            return [np.zeros((0, 3)) for _ in jobs]

    class _Context:
        Pool = _Pool

    monkeypatch.setattr(fixed_margin, "get_context", lambda method: _Context)
    monkeypatch.setattr(fixed_margin.os, "cpu_count", lambda: 24)

    no_manoeuvre_margin_inputs(load_config(), seeds=range(12))

    assert sizes == [fixed_margin.DEFAULT_MAX_WORKERS] == [6]
