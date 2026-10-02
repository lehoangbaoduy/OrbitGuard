"""Tests for orbit_guard.evaluation.shifts: the four zero-shot shift conditions (PRD §32)."""

from __future__ import annotations

import pytest

from orbit_guard.evaluation.shifts import CONDITION_NAMES, condition_env_kwargs
from orbit_guard.rl.environment import CapabilityDegradation, OrbitGuardEnv
from orbit_guard.sensing.noise import NoiseModel

_TRAINING_NOISE = NoiseModel(position_sigma_m=100.0, velocity_sigma_mps=0.1)
_FAST = {"episode_horizon_s": 120.0, "control_dt_s": 60.0, "physics_dt_s": 10.0, "lookahead_s": 60.0}


def test_the_five_conditions_of_the_experiment_matrix_exist_in_order():
    assert CONDITION_NAMES == (
        "normal", "noise_increase", "density_increase", "simultaneous_conjunctions", "capability_reduction"
    )


def test_normal_condition_changes_nothing():
    assert condition_env_kwargs("normal") == {}


def test_noise_shift_scales_both_sensing_sigmas_beyond_the_training_value():
    noise = condition_env_kwargs("noise_increase")["noise_model"]

    assert noise.position_sigma_m == pytest.approx(3.0 * _TRAINING_NOISE.position_sigma_m)
    assert noise.velocity_sigma_mps == pytest.approx(3.0 * _TRAINING_NOISE.velocity_sigma_mps)


def test_density_shift_keeps_n_fixed_and_only_sets_the_phase_window():
    kwargs = condition_env_kwargs("density_increase")

    assert set(kwargs) == {"phase_spread_rad"}
    assert kwargs["phase_spread_rad"] > 0


def test_simultaneous_shift_exceeds_the_training_conjunction_count():
    mix = condition_env_kwargs("simultaneous_conjunctions")["distribution_mix"]

    assert mix == {"triple_conjunction": 1.0}


def test_capability_shift_disables_and_reduces_satellites():
    degradation = condition_env_kwargs("capability_reduction")["capability_degradation"]

    assert isinstance(degradation, CapabilityDegradation)
    assert degradation.disabled_count >= 1
    assert degradation.reduced_count >= 1


def test_unknown_condition_is_rejected():
    with pytest.raises(KeyError):
        condition_env_kwargs("solar_storm")


@pytest.mark.parametrize("name", CONDITION_NAMES)
def test_every_condition_builds_and_resets_a_working_environment(name):
    env = OrbitGuardEnv(num_agents=12, **_FAST, **condition_env_kwargs(name))

    observations, _ = env.reset(seed=1)
    _, rewards, _, _, _ = env.step({agent: 0 for agent in env.agents})

    assert len(observations) == 12
    assert len(rewards) == 12
