"""The replay's "high risk" threshold must stay the policy's threatening-risk threshold (PRD §41 status vocabulary).

The replay path imports NumPy only (no Ray, no torch), so it cannot import `rl` to share the constant. This test is the
drift guard instead: it imports `rl.observation` here, in the test process only, and compares the two values.
"""

from __future__ import annotations

from orbit_guard.rl.observation import THREATENING_RISK_THRESHOLD
from orbit_guard.visualization.replay import HIGH_RISK_THRESHOLD


def test_replay_high_risk_threshold_equals_the_observation_threatening_risk_threshold():
    assert HIGH_RISK_THRESHOLD == THREATENING_RISK_THRESHOLD
