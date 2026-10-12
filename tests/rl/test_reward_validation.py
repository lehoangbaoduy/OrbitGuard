"""Input-validation paths of orbit_guard.rl.reward beyond the frozen spec."""

from __future__ import annotations

import pytest

from orbit_guard.rl.reward import (
    compute_mission_term,
    compute_pairwise_margin,
    compute_safety_term,
)


def test_pairwise_margin_rejects_a_negative_safe_distance():
    with pytest.raises(ValueError):
        compute_pairwise_margin(d_min_m=100.0, d_safe_m=-1.0)


def test_pairwise_margin_rejects_a_non_positive_epsilon():
    with pytest.raises(ValueError):
        compute_pairwise_margin(d_min_m=100.0, d_safe_m=1000.0, epsilon_m=0.0)


def test_safety_term_rejects_inverted_clip_bounds():
    with pytest.raises(ValueError):
        compute_safety_term([0.1], clip_min=1.0, clip_max=-1.0)


def test_mission_term_rejects_a_negative_weight():
    with pytest.raises(ValueError):
        compute_mission_term(0.1, 0.1, 0.1, 0.1, q_a=-0.25)
