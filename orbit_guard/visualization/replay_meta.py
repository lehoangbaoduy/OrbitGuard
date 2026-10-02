"""Replay-trace metadata schema (redesign real-data rule 3).

An episode identifies itself through `trace.meta`: which scenario (condition + seed), which policy (untrained vs
trained vs a baseline), whether the safety shield was on, and the constants the UI needs (delta-v budget, number
of satellites). Nothing is inferred from file names. `validate_replay_meta` fails loudly and names every missing
or invalid field, with a re-record hint for traces written before this schema existed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

REPLAY_META_SCHEMA = "orbitguard-replay-meta/1"
POLICIES = ("untrained", "trained", "no_maneuver", "rule_based")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_ECHO = 60  # untrusted meta values are quoted in errors; keep them short
MIN_AGENTS = 2  # the environment itself requires at least two satellites
_REQUIRED = (
    "schema", "scenario_id", "condition", "seed", "policy", "shield", "model",
    "n_agents", "delta_v_budget_mps", "max_delta_v_per_action_mps",
)
RERECORD_HINT = "re-record the episodes with scripts/record_replay_traces.py"


class ReplayMetaError(ValueError):
    """The trace meta is missing fields or holds invalid values."""


@dataclass(frozen=True)
class ReplayMeta:
    scenario_id: str
    condition: str
    seed: int
    policy: str
    shield: bool
    model: str
    n_agents: int
    delta_v_budget_mps: float
    max_delta_v_per_action_mps: float
    weights_sha256: str | None = None
    init_seed: int | None = None


def scenario_id(condition: str, seed: int) -> str:
    return f"{condition}:{seed}"


def build_replay_meta(
    *,
    condition: str,
    seed: int,
    policy: str,
    shield: bool,
    model: str,
    n_agents: int,
    delta_v_budget_mps: float,
    max_delta_v_per_action_mps: float,
    weights_sha256: str | None = None,
    init_seed: int | None = None,
) -> dict[str, Any]:
    """Plain-JSON meta for `record_episode(..., meta=...)`; optional provenance fields are omitted when None."""
    meta: dict[str, Any] = {
        "schema": REPLAY_META_SCHEMA,
        "scenario_id": scenario_id(condition, seed),
        "condition": condition,
        "seed": seed,
        "policy": policy,
        "shield": shield,
        "model": model,
        "n_agents": n_agents,
        "delta_v_budget_mps": float(delta_v_budget_mps),
        "max_delta_v_per_action_mps": float(max_delta_v_per_action_mps),
    }
    if weights_sha256 is not None:
        meta["weights_sha256"] = weights_sha256
    if init_seed is not None:
        meta["init_seed"] = init_seed
    return meta


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _echo(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= _MAX_ECHO else text[: _MAX_ECHO - 1] + "…"


def _problems(meta: Mapping[str, Any]) -> list[str]:
    missing = [field for field in _REQUIRED if field not in meta]
    if missing:
        return [f"missing field(s): {', '.join(missing)}"]
    problems = []
    if meta["schema"] != REPLAY_META_SCHEMA:
        problems.append(f"schema {_echo(meta['schema'])} is not {REPLAY_META_SCHEMA!r}")
    for field in ("scenario_id", "condition", "model"):
        if not _is_text(meta[field]):
            problems.append(f"{field} must be a non-empty string")
    if not _is_int(meta["seed"]):
        problems.append("seed must be an int")
    if meta["policy"] not in POLICIES:
        problems.append(f"policy {_echo(meta['policy'])} is not one of {POLICIES}")
    if not isinstance(meta["shield"], bool):
        problems.append("shield must be a bool")
    if not (_is_int(meta["n_agents"]) and meta["n_agents"] >= MIN_AGENTS):
        problems.append(f"n_agents must be an int >= {MIN_AGENTS}")
    for field in ("delta_v_budget_mps", "max_delta_v_per_action_mps"):
        if not _is_positive_number(meta[field]):
            problems.append(f"{field} must be a positive number")
    if _is_int(meta["seed"]) and meta["scenario_id"] != scenario_id(meta["condition"], meta["seed"]):
        problems.append(f"scenario_id {_echo(meta['scenario_id'])} does not match condition and seed")
    sha = meta.get("weights_sha256")
    if (meta["policy"] == "trained" or sha is not None) and not (isinstance(sha, str) and _SHA256.fullmatch(sha)):
        problems.append("weights_sha256 must be 64 lowercase hex chars (required for a trained policy)")
    if meta["policy"] == "untrained" and not _is_int(meta.get("init_seed")):
        problems.append("init_seed (int) is required for an untrained policy")
    return problems


def validate_replay_meta(meta: Mapping[str, Any]) -> ReplayMeta:
    """Return the typed meta, or raise `ReplayMetaError` naming every missing/invalid field."""
    if not isinstance(meta, Mapping):
        raise ReplayMetaError(f"invalid replay meta: expected a mapping, got {type(meta).__name__} ({RERECORD_HINT})")
    problems = _problems(meta)
    if problems:
        legacy = "schema" not in meta
        hint = f" ({RERECORD_HINT})" if legacy else ""
        raise ReplayMetaError("invalid replay meta: " + "; ".join(problems) + hint)
    return ReplayMeta(
        scenario_id=meta["scenario_id"],
        condition=meta["condition"],
        seed=meta["seed"],
        policy=meta["policy"],
        shield=meta["shield"],
        model=meta["model"],
        n_agents=meta["n_agents"],
        delta_v_budget_mps=float(meta["delta_v_budget_mps"]),
        max_delta_v_per_action_mps=float(meta["max_delta_v_per_action_mps"]),
        weights_sha256=meta.get("weights_sha256"),
        init_seed=meta.get("init_seed"),
    )
