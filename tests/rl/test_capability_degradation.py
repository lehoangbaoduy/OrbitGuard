"""capability_reduction victim choice (PRD §32 #4): never disable both members of an injected pair.

PRD §32 #4 asks whether the system "maneuvers other satellites to compensate" when some satellites lose
maneuver capability or have reduced Δv. Disabling both members of the only injected pair makes that encounter
uncontrollable for every model, so the rule is: disabled = injected-pair primaries first, then uninvolved
satellites by index; reduced = injected-pair secondaries first, then the remaining uninvolved satellites by index.
"""

from __future__ import annotations

import pytest

from orbit_guard.evaluation.baselines import MODEL_NAMES, build_env_factory
from orbit_guard.evaluation.seed_pools import load_seed_pool
from orbit_guard.evaluation.shifts import condition_env_kwargs
from orbit_guard.rl.capability_degradation import select_victims

_N = 15


def _assert_no_pair_fully_disabled(pairs, disabled):
    for primary, secondary in pairs:
        assert not (primary in disabled and secondary in disabled), f"pair {(primary, secondary)} fully disabled"


def test_single_conjunction_disables_the_primary_and_reduces_the_secondary():
    pairs = ((13, 3),)
    disabled, reduced = select_victims(_N, pairs, disabled_count=2, reduced_count=2)

    assert disabled == (13, 0)
    assert reduced == (3, 1)
    _assert_no_pair_fully_disabled(pairs, disabled)


def test_single_conjunction_with_a_low_index_pair_skips_its_members_among_the_uninvolved():
    disabled, reduced = select_victims(_N, ((8, 0),), disabled_count=2, reduced_count=2)

    assert disabled == (8, 1)
    assert reduced == (0, 2)


def test_multi_conjunction_disables_both_primaries_and_reduces_both_secondaries():
    pairs = ((4, 9), (12, 2))
    disabled, reduced = select_victims(_N, pairs, disabled_count=2, reduced_count=2)

    assert disabled == (4, 12)
    assert reduced == (9, 2)
    _assert_no_pair_fully_disabled(pairs, disabled)


def test_triple_conjunction_degrades_the_first_two_pairs_and_leaves_the_third_untouched():
    pairs = ((4, 9), (12, 2), (7, 11))
    disabled, reduced = select_victims(_N, pairs, disabled_count=2, reduced_count=2)

    assert disabled == (4, 12)
    assert reduced == (9, 2)
    assert {7, 11}.isdisjoint(disabled + reduced), "the third pair keeps full capability"


def test_benign_episode_disables_the_first_satellites_and_reduces_the_next():
    # Regression pin: the old rule already did this when there is no injected pair.
    assert select_victims(_N, (), disabled_count=2, reduced_count=2) == ((0, 1), (2, 3))


def test_a_secondary_is_never_disabled_even_when_too_few_other_satellites_exist():
    # N=2, one pair: only the primary can be disabled; the shortfall is not filled with the secondary.
    disabled, reduced = select_victims(2, ((1, 0),), disabled_count=2, reduced_count=0)

    assert disabled == (1,)
    assert reduced == ()


def test_reduced_victims_never_overlap_the_disabled_ones():
    disabled, reduced = select_victims(4, ((2, 3),), disabled_count=3, reduced_count=3)

    assert disabled == (2, 0, 1)
    assert reduced == (3,)
    assert set(disabled).isdisjoint(reduced)


def test_zero_counts_choose_no_victims():
    assert select_victims(_N, ((1, 2),), disabled_count=0, reduced_count=0) == ((), ())


def _victims(env):
    disabled = tuple(i for i, cap in enumerate(env._capabilities) if not cap.maneuver_enabled)
    return disabled, env.budget_provenance().degraded


def _pairs(env):
    return [(c.primary_index, c.secondary_index) for c in env._last_scenario.injected_conjunctions]


@pytest.fixture(scope="module")
def validation_seeds():
    return load_seed_pool("configs/validation_seeds.json", "validation")


def test_over_fifty_validation_seeds_no_injected_pair_is_fully_disabled(validation_seeds):
    env = build_env_factory("no_maneuver", _N, **condition_env_kwargs("capability_reduction"))({})
    for seed in validation_seeds:
        env.reset(seed=seed)
        disabled, degraded = _victims(env)
        _assert_no_pair_fully_disabled(_pairs(env), disabled)
        assert len(disabled) == 2, seed
        assert sum(degraded) == 2, seed
        assert not any(degraded[i] for i in disabled), seed


def test_every_model_faces_the_identical_victims(validation_seeds):
    envs = {
        model: build_env_factory(model, _N, **condition_env_kwargs("capability_reduction"))({})
        for model in MODEL_NAMES
    }
    episode_types = set()
    for seed in validation_seeds[:6]:
        victims = set()
        for env in envs.values():
            env.reset(seed=seed)
            victims.add((tuple(_pairs(env)), _victims(env)))
        assert len(victims) == 1, f"seed {seed}: victims differ across models"
        episode_types.add(envs["no_maneuver"]._last_scenario.episode_type)
    assert {"single_conjunction", "multi_conjunction", "benign"} <= episode_types, "precondition: all three types"
