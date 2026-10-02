"""Episode traces and standalone replay (PRD §41, FR16, P6-M5).

`record_episode` runs ONE episode of any policy in any environment exposing
`snapshot()` / `last_step` and returns an `EpisodeTrace`; `save_trace`/`load_trace`
persist it as a compressed `.npz`; `frame_at` turns step `t` into the display
vocabulary. Replay -- load, `frame_at`, the Dash app -- imports only NumPy: no Ray,
no torch, no trainer, so a saved episode replays with training and inference
stopped (tests/evaluation/test_replay.py pins that with a subprocess import check).

Status vocabulary (PRD §41's visual rule: every state gets a colour AND a label/marker, and
collision, high risk, inside-d_safe, shield intervention and applied manoeuvre stay distinct):
  pair status       collision > high_risk (risk > 0.5) > inside_d_safe (predicted d_min < d_safe) > nominal
  satellite status  collided | high_risk | inside_d_safe | nominal  (worst pair it belongs to)
  satellite event   collision | shield_intervention | maneuver | none  (what happened on the step INTO this frame)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HIGH_RISK_THRESHOLD = 0.5
_PAIR_RANK = {"nominal": 0, "inside_d_safe": 1, "high_risk": 2, "collision": 3}


@dataclass(frozen=True)
class EpisodeTrace:
    """Arrays are indexed by time t = 0..T (state-like) or step t = 0..T-1 (action-like)."""

    control_dt_s: float
    positions_m: np.ndarray  # (T+1, N, 3) true positions -- display and evaluation only
    alive: np.ndarray  # (T+1, N)
    remaining_dv_mps: np.ndarray  # (T+1, N)
    mission_deviation: np.ndarray  # (T+1, N)
    maneuver_enabled: np.ndarray  # (N,)
    pair_index: np.ndarray  # (P, 2)
    d_min_m: np.ndarray  # (T+1, P) belief-predicted
    d_safe_m: np.ndarray  # (T+1, P)
    risk: np.ndarray  # (T+1, P)
    proposed: np.ndarray  # (T, N)
    approved: np.ndarray  # (T, N)
    applied_dv_mps: np.ndarray  # (T, N)
    intervened: np.ndarray  # (T, N)
    collided_pair: np.ndarray  # (T, P)
    meta: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return int(self.proposed.shape[0])

    @property
    def num_agents(self) -> int:
        return int(self.positions_m.shape[1])

    @property
    def num_pairs(self) -> int:
        return int(self.pair_index.shape[0])


@dataclass(frozen=True)
class PairFrame:
    i: int
    j: int
    d_min_m: float
    d_safe_m: float
    risk: float
    status: str


@dataclass(frozen=True)
class SatelliteFrame:
    index: int
    position_m: tuple[float, float, float]
    alive: bool
    status: str
    event: str
    remaining_dv_mps: float
    mission_deviation: float
    proposed: int | None
    approved: int | None


@dataclass(frozen=True)
class Frame:
    step: int
    time_s: float
    satellites: tuple[SatelliteFrame, ...]
    pairs: tuple[PairFrame, ...]


def record_episode(env, policy, seed: int, meta: dict[str, Any] | None = None) -> EpisodeTrace:
    """Run one episode and capture everything the replay needs."""
    observations, _ = env.reset(seed=seed)
    snapshots = [env.snapshot()]
    steps = []
    while env.agents:
        actions = {agent: int(policy(agent, observations[agent])) for agent in env.agents}
        observations, _, _, _, _ = env.step(actions)
        steps.append(env.last_step)
        snapshots.append(env.snapshot())

    summary = dict(env.episode_summary().__dict__)
    summary["failure_labels"] = list(summary["failure_labels"])
    pair_rows = snapshots[0]["pairs"]
    num_pairs = len(pair_rows)
    pair_lookup = {(i, j): k for k, (i, j, *_rest) in enumerate(pair_rows)}
    collided = np.zeros((len(steps), num_pairs), dtype=bool)
    for t, step in enumerate(steps):
        for pair in step["collided_pairs"]:
            collided[t, pair_lookup[tuple(pair)]] = True

    return EpisodeTrace(
        control_dt_s=float(env._control_dt_s),
        positions_m=np.stack([s["positions_m"] for s in snapshots]),
        alive=np.array([s["alive"] for s in snapshots], dtype=bool),
        remaining_dv_mps=np.array([s["remaining_dv_mps"] for s in snapshots], dtype=float),
        mission_deviation=np.array([s["mission_deviation"] for s in snapshots], dtype=float),
        maneuver_enabled=np.array(snapshots[0]["maneuver_enabled"], dtype=bool),
        pair_index=np.array([[i, j] for i, j, *_ in pair_rows], dtype=int),
        d_min_m=np.array([[row[2] for row in s["pairs"]] for s in snapshots]),
        d_safe_m=np.array([[row[3] for row in s["pairs"]] for s in snapshots]),
        risk=np.array([[row[4] for row in s["pairs"]] for s in snapshots]),
        proposed=np.array([s["proposed"] for s in steps], dtype=int),
        approved=np.array([s["approved"] for s in steps], dtype=int),
        applied_dv_mps=np.array([s["applied_dv_mps"] for s in steps], dtype=float),
        intervened=np.array([s["intervened"] for s in steps], dtype=bool),
        collided_pair=collided,
        meta=dict(meta or {}, seed=seed),
        summary=summary,
    )


_ARRAY_FIELDS = (
    "positions_m", "alive", "remaining_dv_mps", "mission_deviation", "maneuver_enabled", "pair_index",
    "d_min_m", "d_safe_m", "risk", "proposed", "approved", "applied_dv_mps", "intervened", "collided_pair",
)


def save_trace(trace: EpisodeTrace, path: str | Path) -> None:
    arrays = {name: getattr(trace, name) for name in _ARRAY_FIELDS}
    header = json.dumps({"control_dt_s": trace.control_dt_s, "meta": trace.meta, "summary": trace.summary})
    np.savez_compressed(path, header=np.array(header), **arrays)


def load_trace(path: str | Path) -> EpisodeTrace:
    with np.load(path, allow_pickle=False) as data:
        header = json.loads(str(data["header"]))
        arrays = {name: data[name] for name in _ARRAY_FIELDS}
    return EpisodeTrace(control_dt_s=header["control_dt_s"], meta=header["meta"], summary=header["summary"], **arrays)


def _pair_status(trace: EpisodeTrace, t: int, k: int) -> str:
    if t >= 1 and trace.collided_pair[:t, k].any():  # a collision stays a collision for the rest of the replay
        return "collision"
    if trace.risk[t, k] > HIGH_RISK_THRESHOLD:
        return "high_risk"
    if trace.d_min_m[t, k] < trace.d_safe_m[t, k]:
        return "inside_d_safe"
    return "nominal"


def frame_at(trace: EpisodeTrace, t: int) -> Frame:
    """Display state at step `t` (0 = start, `num_steps` = final)."""
    if not 0 <= t <= trace.num_steps:
        raise IndexError(f"step {t} outside 0..{trace.num_steps}")

    pairs = []
    worst = ["nominal"] * trace.num_agents
    collided_agents: set[int] = set()  # in a collided pair at any step so far
    colliding_now: set[int] = set()  # in a pair that collided on the step INTO this frame
    for k, (i, j) in enumerate(trace.pair_index):
        status = _pair_status(trace, t, k)
        pairs.append(
            PairFrame(int(i), int(j), float(trace.d_min_m[t, k]), float(trace.d_safe_m[t, k]), float(trace.risk[t, k]), status)
        )
        for agent in (int(i), int(j)):
            if _PAIR_RANK[status] > _PAIR_RANK[worst[agent]]:
                worst[agent] = status
        if status == "collision":
            collided_agents |= {int(i), int(j)}
        if t >= 1 and trace.collided_pair[t - 1, k]:
            colliding_now |= {int(i), int(j)}

    satellites = []
    for agent in range(trace.num_agents):
        event = "none"
        proposed = approved = None
        if t >= 1:
            proposed, approved = int(trace.proposed[t - 1, agent]), int(trace.approved[t - 1, agent])
            if agent in colliding_now:
                event = "collision"
            elif trace.intervened[t - 1, agent]:
                event = "shield_intervention"
            elif trace.applied_dv_mps[t - 1, agent] > 0:
                event = "maneuver"
        status = "collided" if agent in collided_agents else worst[agent]
        satellites.append(
            SatelliteFrame(
                index=agent,
                position_m=tuple(float(x) for x in trace.positions_m[t, agent]),
                alive=bool(trace.alive[t, agent]),
                status=status,
                event=event,
                remaining_dv_mps=float(trace.remaining_dv_mps[t, agent]),
                mission_deviation=float(trace.mission_deviation[t, agent]),
                proposed=proposed,
                approved=approved,
            )
        )
    return Frame(step=t, time_s=t * trace.control_dt_s, satellites=tuple(satellites), pairs=tuple(pairs))
