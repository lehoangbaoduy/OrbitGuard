"""The browser payload (decision D2): everything the JS stage draws, computed in Python, JSON-safe and compact.

Numeric arrays travel as base64 little-endian typed buffers (`encode_array`); NaN ("no value") stays inside those
binary buffers, while every plain list uses None for "not applicable" so the UI can print "—" instead of 0.
"""

from __future__ import annotations

import base64
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from orbit_guard.physics.constants import R_EARTH_EQUATORIAL_M
from orbit_guard.visualization.derived.comparison import comparison_note
from orbit_guard.visualization.derived.key_steps import conjunction_entries
from orbit_guard.visualization.derived.status import (
    EVENT_COLLISION,
    EVENT_MANEUVER,
    EVENT_SHIELD,
)
from orbit_guard.visualization.replay import HIGH_RISK_THRESHOLD
from orbit_guard.visualization.source import ReplaySource
from orbit_guard.visualization.summary import SummaryTotals
from orbit_guard.visualization.vocabulary import vocabulary_payload

PAYLOAD_SCHEMA = "orbitguard-replay-payload/1"
METRICS = ("delta_v_mps", "maneuvers", "shield", "collisions", "mean_deviation")
NOTHING_POLICY = "no_maneuver"  # the do-nothing baseline: its strip row is shown at every shield setting


def encode_array(array: np.ndarray, dtype: str) -> dict[str, Any]:
    typed = np.ascontiguousarray(array, dtype=np.dtype(dtype).newbyteorder("<"))
    return {"b64": base64.b64encode(typed.tobytes()).decode("ascii"), "dtype": dtype, "shape": list(typed.shape)}


def decode_array(spec: Mapping[str, Any]) -> np.ndarray:
    data = np.frombuffer(base64.b64decode(spec["b64"]), dtype=np.dtype(spec["dtype"]).newbyteorder("<"))
    return data.reshape(spec["shape"])


def _json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.floating, float)):
        return None if math.isnan(value) or math.isinf(value) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json_list(values: Any) -> list:
    return [_json_scalar(value) for value in np.asarray(values).tolist()]


def _metric_of(totals: SummaryTotals | None, metric: str) -> float | None:
    return None if totals is None else _json_scalar(getattr(totals, metric))


def build_strip(active_policy: str, totals_by_policy: Mapping[str, SummaryTotals | None]) -> dict[str, dict[str, Any]]:
    """Per summary cell: the BEFORE (untrained) and AFTER (trained) totals plus a data-generated comparison note, and
    the do-nothing (`no_maneuver`) total under "nothing" when the caller passes that key (the lane exists)."""
    before, after = totals_by_policy.get("untrained"), totals_by_policy.get("trained")
    active = totals_by_policy.get(active_policy)
    other = after if active_policy == "untrained" else before
    strip = {}
    for metric in METRICS:
        value, before_value, after_value, other_value = (_metric_of(totals, metric) for totals in (active, before, after, other))
        strip[metric] = {
            "before": before_value,
            "after": after_value,
            "note": comparison_note(active_policy, value, other_value),
        }
        if NOTHING_POLICY in totals_by_policy:
            strip[metric] = {**strip[metric], "nothing": _metric_of(totals_by_policy[NOTHING_POLICY], metric)}
    return strip


def partner_dtype(agents: int) -> str:
    """Narrowest signed integer that holds every partner index and the -1 "none" marker."""
    return "int8" if agents <= np.iinfo(np.int8).max else "int16"


def _runs_payload(source: ReplaySource) -> dict[str, list[dict[str, Any]] | None]:
    """Event runs per type; the shield lane is None (not applicable) when the shield was off."""
    payload: dict[str, list[dict[str, Any]] | None] = {
        str(event_type): [
            {"sats": list(run.sats), "start": run.start, "end": run.end, "count": run.count} for run in source.runs[event_type]
        ]
        for event_type in (EVENT_MANEUVER, EVENT_SHIELD, EVENT_COLLISION)
    }
    if not source.meta.shield:
        payload[str(EVENT_SHIELD)] = None
    return payload


def _density_payload(source: ReplaySource) -> dict[str, list | None]:
    payload: dict[str, list | None] = {str(k): _json_list(v) for k, v in source.density.items()}
    if not source.meta.shield:
        payload[str(EVENT_SHIELD)] = None
    return payload


def _arrays_payload(source: ReplaySource) -> dict[str, dict[str, Any]]:
    trace, partners = source.trace, source.partners
    return {
        "positions_km": encode_array(trace.positions_m / 1000.0, "float32"),
        "sat_status": encode_array(source.sat_status, "uint8"),
        "events": encode_array(source.events, "uint8"),
        "pair_status": encode_array(source.pair_status, "uint8"),
        "d_min_m": encode_array(trace.d_min_m, "float32"),
        "d_safe_m": encode_array(trace.d_safe_m, "float32"),
        "risk": encode_array(trace.risk, "float32"),
        "remaining_dv_mps": encode_array(trace.remaining_dv_mps, "float32"),
        "mission_deviation": encode_array(trace.mission_deviation, "float32"),
        "proposed": encode_array(trace.proposed, "int8"),
        "approved": encode_array(trace.approved, "int8"),
        "applied_dv_mps": encode_array(trace.applied_dv_mps, "float32"),
        "intervened": encode_array(trace.intervened, "uint8"),
        "partner_d_min_m": encode_array(partners.d_min_m, "float32"),
        "partner_d_safe_m": encode_array(partners.d_safe_m, "float32"),
        "partner_index": encode_array(partners.partner, partner_dtype(source.num_agents)),
        "fuel_empty": encode_array(source.fuel.empty, "uint8"),
    }


def _fuel_payload(source: ReplaySource) -> dict[str, Any]:
    fuel = source.fuel
    return {
        "budget_mps": _json_list(fuel.budget_mps), "total_mps": _json_list(fuel.total_mps),
        "total_budget_mps": _json_scalar(fuel.total_budget_mps), "stranded_mps": _json_scalar(fuel.stranded_mps),
        "empty_count": _json_list(fuel.empty_count), "usable_count": fuel.usable_count, "all_empty_step": fuel.all_empty_step,
    }


def _running_payload(source: ReplaySource) -> dict[str, list | None]:
    running = source.running
    return {
        "delta_v_mps": _json_list(running.delta_v_mps), "maneuvers": _json_list(running.maneuvers),
        "shield": None if running.shield is None else _json_list(running.shield),
        "collisions": _json_list(running.collisions), "mean_deviation": _json_list(running.mean_deviation),
    }


def _summary_payload(source: ReplaySource) -> dict[str, Any]:
    summary = source.summary
    return {metric: _json_scalar(getattr(summary, metric)) for metric in METRICS}


def build_payload(
    source: ReplaySource, *, strip: dict[str, Any] | None = None, reference_note: str | None = None
) -> dict[str, Any]:
    trace, meta = source.trace, source.meta
    return {
        "schema": PAYLOAD_SCHEMA,
        "meta": {
            "scenario_id": meta.scenario_id, "condition": meta.condition, "seed": meta.seed, "policy": meta.policy,
            "shield": meta.shield, "model": meta.model, "n_agents": meta.n_agents,
            "delta_v_budget_mps": _json_scalar(meta.delta_v_budget_mps),
        },
        "dims": {"steps": source.num_steps, "agents": source.num_agents, "pairs": source.num_pairs,
                 "control_dt_s": _json_scalar(trace.control_dt_s)},
        "thresholds": {"high_risk": HIGH_RISK_THRESHOLD},
        "constants": {"earth_radius_km": R_EARTH_EQUATORIAL_M / 1000.0},
        "vocab": vocabulary_payload(),
        "catalog": [
            {"index": c.index, "name": c.name, "label": c.label, "short": c.short, "delta_v_mps": c.delta_v_mps,
             "exhaust_rtn": list(c.exhaust_rtn)}
            for c in source.catalog
        ],
        "pair_index": trace.pair_index.tolist(),
        "arrays": _arrays_payload(source),
        "density": _density_payload(source),
        "runs": _runs_payload(source),
        "key_steps": list(source.key_steps),
        "conjunction_entries": conjunction_entries(source.pair_status),
        "fuel": _fuel_payload(source),
        "capability": {
            "offline": np.nonzero(source.capability.offline)[0].tolist(),
            "limited": np.nonzero(source.capability.limited)[0].tolist(),
        },
        "running": _running_payload(source),
        "summary": _summary_payload(source),
        "start_step": source.start_step,
        "strip": strip,
        "reference_note": reference_note,
    }
