"""Trajectory digests for the cross-process half of Acceptance Test F (PRD §38).

One module, imported both by the test process and by the child processes it launches
(`python -c "from tests.acceptance._repro_digest import main; main()" '<json spec>'`), so every
process hashes with exactly the same code. Not collected by pytest (no test_ prefix).

A digest covers, per control step: the scripted joint action, the true states (raw float64 bytes),
the observations (raw float32 bytes), rewards, terminations, truncations, the step description
(proposed/approved intents, applied delta-v norms, collided pairs), the PRD §40 step records and,
for the shielded env, the shield's decision (proposed, approved, intervened, accepted, feasible,
violation kinds and agents, predicted total risk and delta-v of both evaluations); then the episode
summary and the per-satellite delta-v used. Floats go through JSON (shortest round-trip repr) or raw
bytes, never through numpy's possibly truncated repr.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from typing import Any

import numpy as np

from orbit_guard.evaluation.baselines import build_env_factory
from orbit_guard.evaluation.runner import EpisodeTask, run_episode_task
from orbit_guard.evaluation.shifts import condition_env_kwargs
from orbit_guard.safety.maneuver_mapper import ManeuverIntent

TIMING_FIELDS = ("start_time", "wall_time_s")
_NUM_INTENTS = len(ManeuverIntent)
_BURN_EVERY = 3  # each satellite proposes a burn on every third step, staggered by index


def scripted_action(step: int, index: int) -> int:
    """A fixed, observation-independent schedule that mixes NO_MANEUVER with every burn intent."""
    if (step + index) % _BURN_EVERY:
        return int(ManeuverIntent.NO_MANEUVER)
    return 1 + (step + 2 * index) % (_NUM_INTENTS - 1)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, default=_to_plain).encode()


def _to_plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, frozenset | set):
        return sorted(value)
    raise TypeError(f"not digestible: {type(value).__name__}")


def _observation_bytes(observations: dict) -> bytes:
    return b"".join(
        np.ascontiguousarray(observations[agent][key]).tobytes()
        for agent in sorted(observations)
        for key in sorted(observations[agent])
    )


def _shield_fields(result) -> dict:
    def evaluation(ev) -> dict:
        return {
            "violations": [(v.kind.value, list(v.agents)) for v in ev.violations],
            "total_risk": ev.total_risk,
            "total_delta_v_mps": ev.total_delta_v_mps,
            "shortfall_m": ev.shortfall_m,
        }

    return {
        "proposed": list(result.proposed),
        "approved": list(result.approved),
        "intervened": list(result.intervened),
        "accepted": result.accepted,
        "feasible": result.feasible,
        "proposal": evaluation(result.proposal_evaluation),
        "approved_evaluation": evaluation(result.approved_evaluation),
    }


def trajectory_digest(model: str, condition: str, seed: int, n_agents: int, env_kwargs: dict) -> dict:
    """Run one scripted episode of `model`'s env under `condition` and hash everything it produced."""
    factory = build_env_factory(model, n_agents, **{**condition_env_kwargs(condition), **env_kwargs})
    env = factory({})
    digest = hashlib.sha256()
    observations, _ = env.reset(seed=seed)
    digest.update(np.asarray(env._ground_truth, dtype=np.float64).tobytes())
    digest.update(_observation_bytes(observations))
    rejections = 0
    step = 0
    while env.agents:
        actions = {agent: scripted_action(step, env._agent_index[agent]) for agent in env.agents}
        observations, rewards, terminations, truncations, _ = env.step(actions)
        shield = getattr(env, "last_shield_result", None)
        rejections += int(shield is not None and not shield.accepted)
        digest.update(_canonical([step, actions, rewards, terminations, truncations, env.last_step]))
        digest.update(_canonical(env.step_records()))
        digest.update(_canonical(_shield_fields(shield) if shield is not None else None))
        digest.update(np.asarray(env._ground_truth, dtype=np.float64).tobytes())
        digest.update(_observation_bytes(observations))
        step += 1
    summary = dataclasses.asdict(env.episode_summary())
    delta_v_used = env.delta_v_used_mps()
    digest.update(_canonical([summary, delta_v_used]))
    return {
        "digest": digest.hexdigest(),
        "steps": step,
        "total_delta_v_mps": summary["total_delta_v_mps"],
        "shield_rejections": rejections,
    }


def runner_record(task_fields: dict) -> str:
    """`run_episode_task` for the task, timing fields removed, as canonical JSON text (compared byte for byte)."""
    record = run_episode_task(EpisodeTask(**task_fields))
    return json.dumps({k: v for k, v in record.items() if k not in TIMING_FIELDS}, sort_keys=True)


def compute(spec: dict) -> dict:
    """Everything one process contributes: one digest per trajectory spec, plus the runner record if asked."""
    return {
        "trajectories": [trajectory_digest(**trajectory) for trajectory in spec["trajectories"]],
        "runner_record": runner_record(spec["runner_task"]) if spec.get("runner_task") else None,
        "str_hash": hash("orbit_guard"),  # differs between processes with different PYTHONHASHSEED values
    }


def main() -> None:
    json.dump(compute(json.loads(sys.argv[1])), sys.stdout)


if __name__ == "__main__":
    main()
