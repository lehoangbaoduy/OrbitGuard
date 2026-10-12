"""Input-validation paths of orbit_guard.safety.maneuver_mapper beyond the frozen spec."""

from __future__ import annotations

import numpy as np
import pytest

from orbit_guard.safety.maneuver_mapper import ManeuverIntent, map_intent_to_delta_v


def test_rejects_a_negative_phase_fraction():
    position_m, velocity_mps = np.array([6.8e6, 0.0, 0.0]), np.array([0.0, 7.6e3, 0.0])

    with pytest.raises(ValueError):
        map_intent_to_delta_v(ManeuverIntent.PHASE_FORWARD, position_m, velocity_mps, phase_fraction=-0.5)
