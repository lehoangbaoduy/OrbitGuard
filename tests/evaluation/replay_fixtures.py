"""Small REAL recorded episodes for the replay-redesign tests (never synthetic arrays).

Every trace is produced by running the actual environments through `record_episode`; the horizons are short
(30 steps) so the whole module costs a few seconds. Seeds were chosen offline to give the properties named below.
"""

from __future__ import annotations

from functools import cache

from orbit_guard.evaluation.baselines import rule_based_policy
from orbit_guard.rl.environment import CapabilityDegradation, OrbitGuardEnv
from orbit_guard.rl.evaluation import no_maneuver_policy
from orbit_guard.rl.shielded_environment import ShieldedOrbitGuardEnv
from orbit_guard.sensing.noise import NoiseModel
from orbit_guard.visualization.replay import EpisodeTrace, record_episode
from orbit_guard.visualization.replay_meta import build_replay_meta

ENV_KWARGS = {
    "episode_horizon_s": 1800.0,
    "control_dt_s": 60.0,
    "physics_dt_s": 10.0,
    "lookahead_s": 1800.0,
    "noise_model": NoiseModel(0.0, 0.0),
    "hard_body_radius_m": 100.0,
    # Fixed 2.0 m/s budgets (no PRD §17 per-episode draw): the expected fuel totals and capability flags below are
    # literals chosen offline against the template budget.
    "remaining_delta_v_range_mps": None,
}
COLLISION_SEED = 3  # N=2, no maneuvers: the pair collides
SHIELD_SEED = 0  # N=2, rule-based policy: many shield interventions and burns
CAPABILITY_SEED = 1


def _meta(env, *, condition: str, seed: int, policy: str, shield: bool, model: str) -> dict:
    extra = {"weights_sha256": "0" * 64} if policy == "trained" else {"init_seed": 0} if policy == "untrained" else {}
    return build_replay_meta(
        condition=condition,
        seed=seed,
        policy=policy,
        shield=shield,
        model=model,
        n_agents=len(env.possible_agents),
        delta_v_budget_mps=env.delta_v_budget_mps,
        max_delta_v_per_action_mps=0.1,
        **extra,
    )


@cache
def collision_trace() -> EpisodeTrace:
    """No maneuvers, no shield: the two satellites collide."""
    env = OrbitGuardEnv(num_agents=2, **ENV_KWARGS)
    meta = _meta(env, condition="simultaneous_conjunctions", seed=COLLISION_SEED, policy="no_maneuver", shield=False, model="no_maneuver")
    return record_episode(env, no_maneuver_policy, seed=COLLISION_SEED, meta=meta)


@cache
def shielded_trace() -> EpisodeTrace:
    """Rule-based proposals through the shield: interventions and applied burns."""
    env = ShieldedOrbitGuardEnv(num_agents=2, **ENV_KWARGS)
    meta = _meta(env, condition="normal", seed=SHIELD_SEED, policy="trained", shield=True, model="adaptive_mappo")
    return record_episode(env, rule_based_policy, seed=SHIELD_SEED, meta=meta)


@cache
def unshielded_trace() -> EpisodeTrace:
    """Same proposals applied as-is: burns but zero shield interventions."""
    env = OrbitGuardEnv(num_agents=2, **ENV_KWARGS)
    meta = _meta(env, condition="normal", seed=SHIELD_SEED, policy="trained", shield=False, model="adaptive_mappo_no_shield")
    return record_episode(env, rule_based_policy, seed=SHIELD_SEED, meta=meta)


@cache
def capability_trace() -> EpisodeTrace:
    """Four satellites; one thruster disabled and one limited to 25% of the budget."""
    degradation = CapabilityDegradation(disabled_count=1, reduced_count=1, reduced_budget_scale=0.25)
    env = ShieldedOrbitGuardEnv(num_agents=4, capability_degradation=degradation, **ENV_KWARGS)
    meta = _meta(env, condition="capability_reduction", seed=CAPABILITY_SEED, policy="untrained", shield=True, model="untrained_policy")
    return record_episode(env, rule_based_policy, seed=CAPABILITY_SEED, meta=meta)


# Injected pair (1, 2): primary 1 keeps the full tank and every tank, the full one included, runs dry. Seed 2 pinned
# the pre-fix victim order, which reduced BOTH members of its pair (2, 0); with secondary-first reduction its
# full-tank primary 2 collides with 0 at step 12, before that tank is empty.
DEPLETION_SEED = 11


def _always_raise_policy(agent: str, observation: dict) -> int:
    return 1  # RAISE_ORBIT every step: deterministic, drains any budget


@cache
def depletion_trace() -> EpisodeTrace:
    """Three satellites, two with a 0.12 m/s budget, burning every step: their tanks run dry within a few steps."""
    degradation = CapabilityDegradation(disabled_count=0, reduced_count=2, reduced_budget_scale=0.06)
    env = OrbitGuardEnv(num_agents=3, capability_degradation=degradation, **ENV_KWARGS)
    meta = _meta(env, condition="capability_reduction", seed=DEPLETION_SEED, policy="untrained", shield=False, model="untrained_policy")
    return record_episode(env, _always_raise_policy, seed=DEPLETION_SEED, meta=meta)


FLEET_COLLISION_SEED = 0  # N=4, no maneuvers: satellites 0 and 2 collide, 1 and 3 survive


@cache
def fleet_collision_trace() -> EpisodeTrace:
    """A collision inside a larger fleet, so survivors exist (the N=2 collision fixture has none)."""
    env = OrbitGuardEnv(num_agents=4, **ENV_KWARGS)
    meta = _meta(env, condition="normal", seed=FLEET_COLLISION_SEED, policy="no_maneuver", shield=False, model="no_maneuver")
    return record_episode(env, no_maneuver_policy, seed=FLEET_COLLISION_SEED, meta=meta)



@cache
def full_size_trace() -> EpisodeTrace:
    """A real production-size episode (N=15, 360 steps of 60 s) for payload-budget checks; about 15 s to record."""
    env = OrbitGuardEnv(num_agents=15, **{**ENV_KWARGS, "episode_horizon_s": 21600.0})
    meta = _meta(env, condition="normal", seed=0, policy="no_maneuver", shield=False, model="no_maneuver")
    return record_episode(env, no_maneuver_policy, seed=0, meta=meta)
