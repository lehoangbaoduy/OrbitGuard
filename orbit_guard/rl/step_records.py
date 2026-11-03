"""Per-control-step records of the PRD §40 logging schema (CORE-4c).

One record per satellite that was ACTIVE when the step's actions were chosen, with exactly the §40 keys:

  timestamp          simulation seconds at the START of the step (the decision time)
  satellite_id       the agent id ("sat_<i>")
  observed_state     the decision-time belief state vector [x, y, z, vx, vy, vz] (m, m/s, ECI)
  belief_features    the flattened decision-time observation the policy saw (same order as the policy's input)
  selected_action    the raw proposed ManeuverIntent
  mapped_delta_v     the raw request mapped to an ECI delta-v (m/s), before any capability cap (Q21)
  shield_action      the approved ManeuverIntent (= selected_action without a shield)
  shield_intervened  whether the shield replaced this satellite's proposal
  shield_reason      sorted ViolationKind values involving this satellite in the shield's evaluation of the raw
                     proposal; [] when the proposal was accepted or there is no shield
  risk_before/after  max risk proxy over the pairs containing the satellite, in the decision-time / realized
                     post-step belief assessment
  d_safe_before/after  the d_safe (m) of that max-risk pair in each assessment, under the run's own margin
  remaining_delta_v  the satellite's remaining budget after the step (m/s)
  mission_deviation  the satellite's weighted mission deviation after the step (PRD §25)

Every value is a plain Python type, so a record goes to JSON as is.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from orbit_guard.conjunction.pair_assessment import PairAssessment

STEP_RECORD_KEYS = (
    "timestamp", "satellite_id", "observed_state", "belief_features", "selected_action", "mapped_delta_v",
    "shield_action", "shield_intervened", "shield_reason", "risk_before", "risk_after", "d_safe_before",
    "d_safe_after", "remaining_delta_v", "mission_deviation",
)


def max_risk_pair(assessments: Sequence[PairAssessment], index: int) -> PairAssessment:
    """The highest-risk assessment among the pairs containing satellite `index`; the first in pair order on ties."""
    involving = [a for a in assessments if index in (a.i, a.j)]
    if not involving:
        raise ValueError(f"no assessed pair contains satellite {index}")
    return max(involving, key=lambda a: a.risk_proxy)


def _floats(values) -> list[float]:
    return [float(x) for x in np.asarray(values, dtype=float).ravel()]


def build_step_record(
    *,
    timestamp_s: float,
    satellite_id: str,
    index: int,
    observed_state: np.ndarray,
    belief_features: np.ndarray,
    selected_action: int,
    mapped_delta_v_mps: np.ndarray,
    shield_action: int,
    shield_intervened: bool,
    shield_reason: Sequence[str],
    pre_assessments: Sequence[PairAssessment],
    post_assessments: Sequence[PairAssessment],
    remaining_delta_v_mps: float,
    mission_deviation: float,
) -> dict:
    """One §40 per-control-step record for satellite `index`, in plain Python types."""
    before = max_risk_pair(pre_assessments, index)
    after = max_risk_pair(post_assessments, index)
    return {
        "timestamp": float(timestamp_s),
        "satellite_id": str(satellite_id),
        "observed_state": _floats(observed_state),
        "belief_features": _floats(belief_features),
        "selected_action": int(selected_action),
        "mapped_delta_v": _floats(mapped_delta_v_mps),
        "shield_action": int(shield_action),
        "shield_intervened": bool(shield_intervened),
        "shield_reason": sorted({str(reason) for reason in shield_reason}),
        "risk_before": float(before.risk_proxy),
        "risk_after": float(after.risk_proxy),
        "d_safe_before": float(before.d_safe_m),
        "d_safe_after": float(after.d_safe_m),
        "remaining_delta_v": float(remaining_delta_v_mps),
        "mission_deviation": float(mission_deviation),
    }
