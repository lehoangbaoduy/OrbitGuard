"""OrbitGuardEnv: the real PettingZoo ParallelEnv (Phase 3 Part A, PRD §19-28).

Wires together Phase 1 physics (dynamics/integrator/state), Phase 2 sensing
and conjunction screening, Part A's own mission_deviation/small_n_scenario/
conjunction_assessment modules, and Part B/C's maneuver_mapper/observation/
reward contracts (frozen in tests/safety/test_maneuver_mapper.py,
tests/rl/test_observation.py, tests/rl/test_reward.py).

Per-control-step pipeline (docs/OrbitGuard_progress.md Q14-Q20):
  1. Form a fresh noisy belief of every satellite (ground truth -> observe ->
     form_belief, PRD §14-15).
  2. Assess all candidate pairs from belief (predict_belief over
     `lookahead_s`, screening/TCA/risk/adaptive-margin) -- this assessment
     was already computed as the "post" assessment of the PREVIOUS step (or,
     on the first step after reset, computed fresh); it is this step's
     "pre" assessment, i.e. what each agent observes and acts on.
  3. Build this step's observations from belief + "pre" assessment.
  4. Predict the outcome of the RAW joint proposal from this step's "pre"
     beliefs and baseline predictions (`pair_assessment.assess_proposal`):
     each proposed burn is mapped from the BELIEF state, sized by the
     satellite's own per-action limit, never capability-capped; NO_MANEUVER
     and non-controllable satellites reuse their baseline prediction. This is
     the ONE estimator scoring R_safety and the coordination term (PRD §25/
     §28, CORE-2) in both this environment and the shielded one, whose shield
     builds `proposal_evaluation` with the same functions.
     Then map each agent's discrete action to an ECI delta-v (Part B) for
     actuation, capped by capability (never silently substituted with
     NO_MANEUVER, PRD §13).
  5. Apply delta-v to GROUND TRUTH and propagate it at `physics_dt_s`
     resolution across the full control step (not just the endpoints --
     relative velocities ~2km/s traverse ~120km in one 60s control step).
  6. Ground-truth collision check over that fine-grained window (PRD §18).
  7. Form a fresh belief AFTER the physical step and assess all pairs again
     -- next step's "pre" assessment and observations, and this step's
     realized outcome metrics. It does NOT feed the safety/coordination
     reward: a second, independently-noised assessment would put pure sensing
     noise into the coordination term (~0.095/step under all-no-op actions at
     N=15) that the shielded arm never sees.
  8. Mission deviation (PRD §25) compares the post-step belief against a
     deterministically-evolved (zero-noise-after-reset) shadow belief, at
     the same instant -- never ground truth (Q14's leakage-boundary choice).
  9. Assemble each agent's reward (Part C): safety and coordination from
     step 4's predicted outcome (the coordination term clipped at
     `coordination_clip`, PRD §25, CORE-2), mission from step 8, delta-v on the raw
     request, the collision penalty from step 6 -- and this step's
     observations (next step's "pre" observations) together.
 10. Keep one PRD §40 per-control-step record per satellite active at the
     decision time (`step_records()`, `rl.step_records`, CORE-4c).

Ground truth is used ONLY for actuation (converting a maneuver intent to an
ECI delta-v relative to the satellite's own true state, step 4's actuation half), propagation,
and collision detection (steps 5-6) above; everything feeding observations
or rewards is belief-derived, enforced structurally by `sensing.belief`'s own
TypeError guards.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatten
from pettingzoo import ParallelEnv

from orbit_guard.config import (
    budget_range,
    env_kwargs_from_config,
    load_config,
    margin_from_config,
)
from orbit_guard.conjunction.collision import detect_collision
from orbit_guard.conjunction.pair_assessment import (
    PairAssessment,
    assess_all_pairs,
    assess_proposal,
    pair_density,
    predict_all,
)
from orbit_guard.conjunction.screening import generate_candidate_pairs
from orbit_guard.evaluation.seed_pools import TRAINING_SEED_FLOOR
from orbit_guard.physics.dynamics import state_derivative_batch
from orbit_guard.physics.integrator import propagate_batch
from orbit_guard.physics.mission_deviation import (
    compute_mission_deviation,
    compute_weighted_mission_deviation,
)
from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.rl.conjunction_assessment import build_neighbor_candidates
from orbit_guard.rl.episode_metrics import (
    EpisodeMetricsTracker,
    EpisodeSummary,
    ShieldStep,
)
from orbit_guard.rl.observation import build_observation
from orbit_guard.rl.reward import (
    compute_clipped_coordination_term,
    compute_delta_v_term,
    compute_mission_term,
    compute_pairwise_margin,
    compute_reward,
    compute_safety_term,
)
from orbit_guard.rl.step_records import build_step_record
from orbit_guard.safety.adaptive_margin import MarginFn
from orbit_guard.safety.maneuver_mapper import ManeuverIntent, map_intent_to_delta_v
from orbit_guard.scenarios.conjunction_generator import generate_conjunction_scenario
from orbit_guard.scenarios.constellation import (
    MIN_SUPPORTED_N,
    fit_shell_distribution,
    load_tle_snapshot,
)
from orbit_guard.scenarios.small_n_scenario import generate_small_n_conjunction_scenario
from orbit_guard.sensing.belief import GroundTruthState, form_belief, observe
from orbit_guard.sensing.noise import NoiseModel

_NUM_ACTIONS = len(ManeuverIntent)

# Duplicated raw shape numbers rather than imported from rl.observation -- Part
# B's frozen contract (tests/rl/test_observation.py) fixes these shapes via
# literals, not named constants, so there is nothing to import; matches
# tests/rl/stub_env.py's own documented rationale for the same duplication.
_OWN_DIM = 10
_NEIGHBOR_DIM = 10
_GLOBAL_SUMMARY_DIM = 4

# Per-episode Δv budgets (PRD §17, CORE-3) come from their OWN stream, SeedSequence([episode_seed, this id]), so
# drawing them leaves every existing stream (scenario, sensing) bit-identical and the draw count never depends on
# actions, the model or the shield. The value is arbitrary but fixed: changing it changes every episode's budgets.
_DELTA_V_BUDGET_STREAM_ID = 0xDE17A


class _Unset:
    """Marker for "argument not given", where None is itself a meaningful value."""


_UNSET = _Unset()


@dataclass(frozen=True)
class CapabilityDegradation:
    """Reduced-capability distribution shift (PRD §32 #4), deterministic given the scenario.

    Victims are chosen in a fixed order -- primaries of the injected conjunctions first, then their
    secondaries, then the remaining satellites by index -- so the satellites that most need to
    manoeuvre are the ones that cannot, and every model faces the identical handicap. The first
    `disabled_count` victims get `maneuver_enabled=False`; the next `reduced_count` keep it but
    start with `reduced_budget_scale` of the delta-v budget.
    """

    disabled_count: int = 0
    reduced_count: int = 0
    reduced_budget_scale: float = 0.25

    def __post_init__(self):
        if self.disabled_count < 0 or self.reduced_count < 0:
            raise ValueError("disabled_count and reduced_count must be >= 0")
        if not 0.0 < self.reduced_budget_scale <= 1.0:
            raise ValueError("reduced_budget_scale must be in (0, 1]")

    def apply(self, capabilities: list[SatelliteCapability], conjunctions) -> list[SatelliteCapability]:
        order: list[int] = [spec.primary_index for spec in conjunctions]
        order += [spec.secondary_index for spec in conjunctions]
        order += [i for i in range(len(capabilities)) if i not in order]
        updated = list(capabilities)
        for position, index in enumerate(order[: self.disabled_count + self.reduced_count]):
            capability = updated[index]
            if position < self.disabled_count:
                updated[index] = replace(capability, maneuver_enabled=False)
            else:
                updated[index] = capability.with_remaining_delta_v(
                    capability.remaining_delta_v_mps * self.reduced_budget_scale
                )
        return updated


class OrbitGuardEnv(ParallelEnv):
    """Multi-agent satellite collision-avoidance environment (PRD §19)."""

    metadata: ClassVar[dict] = {"name": "orbit_guard_env_v0", "is_parallelizable": True}

    def __init__(
        self,
        num_agents: int = 2,
        episode_horizon_s: float | None = None,
        physics_dt_s: float | None = None,
        control_dt_s: float | None = None,
        lookahead_s: float | None = None,
        k_neighbors: int | None = None,
        hard_body_radius_m: float | None = None,
        capability_template: SatelliteCapability | None = None,
        distribution_mix=None,
        noise_model: NoiseModel | None = None,
        tle_path: str | Path | None = None,
        reward_weights: dict | None = None,
        margin_fn: MarginFn | None = None,
        phase_spread_rad: float | None = None,
        capability_degradation: CapabilityDegradation | None = None,
        phase_fraction: float | None = None,
        risk_epsilon_m: float | None = None,
        risk_z0: float | None = None,
        risk_k: float | None = None,
        density_radius_m: float | None = None,
        density_ref_count: float | None = None,
        mission_weights: dict | None = None,
        coordination_clip: float | None = None,
        remaining_delta_v_range_mps: tuple[float, float] | None | _Unset = _UNSET,
    ):
        # Every argument not given resolves from configs/default.yaml (the single source of truth, CORE-1).
        base_config = load_config()
        defaults = env_kwargs_from_config(base_config, shielded=False)
        explicit = {
            "episode_horizon_s": episode_horizon_s,
            "physics_dt_s": physics_dt_s,
            "control_dt_s": control_dt_s,
            "lookahead_s": lookahead_s,
            "k_neighbors": k_neighbors,
            "hard_body_radius_m": hard_body_radius_m,
            "capability_template": capability_template,
            "distribution_mix": distribution_mix,
            "noise_model": noise_model,
            "tle_path": tle_path,
            "reward_weights": reward_weights,
            "phase_fraction": phase_fraction,
            "risk_epsilon_m": risk_epsilon_m,
            "risk_z0": risk_z0,
            "risk_k": risk_k,
            "density_radius_m": density_radius_m,
            "density_ref_count": density_ref_count,
            "mission_weights": mission_weights,
            "coordination_clip": coordination_clip,
        }
        resolved = {key: defaults[key] if value is None else value for key, value in explicit.items()}
        if margin_fn is None:
            margin_fn = margin_from_config(base_config, "adaptive")
        # Not given: an explicit capability_template keeps its fixed budget (bare test constructions); otherwise the
        # config's range applies. Given (None included, meaning fixed): it wins.
        if isinstance(remaining_delta_v_range_mps, _Unset):
            remaining_delta_v_range_mps = (
                None if capability_template is not None else defaults["remaining_delta_v_range_mps"]
            )
        self._budget_range_mps = budget_range(remaining_delta_v_range_mps)

        if num_agents < 2:
            raise ValueError("OrbitGuard is a multi-agent env; num_agents must be >= 2")
        if not np.isfinite(resolved["coordination_clip"]) or resolved["coordination_clip"] <= 0:
            raise ValueError(f"coordination_clip must be finite and > 0, got {resolved['coordination_clip']}")
        if resolved["control_dt_s"] <= 0 or resolved["physics_dt_s"] <= 0 or resolved["control_dt_s"] < resolved["physics_dt_s"]:
            raise ValueError("control_dt_s and physics_dt_s must be > 0 with control_dt_s >= physics_dt_s")
        episode_horizon_s = resolved["episode_horizon_s"]
        physics_dt_s = resolved["physics_dt_s"]
        control_dt_s = resolved["control_dt_s"]
        lookahead_s = resolved["lookahead_s"]
        k_neighbors = resolved["k_neighbors"]
        hard_body_radius_m = resolved["hard_body_radius_m"]
        capability_template = resolved["capability_template"]
        distribution_mix = resolved["distribution_mix"]
        noise_model = resolved["noise_model"]
        tle_path = resolved["tle_path"]
        reward_weights = resolved["reward_weights"]

        self.possible_agents = [f"sat_{i}" for i in range(num_agents)]
        self._agent_index = {agent: i for i, agent in enumerate(self.possible_agents)}
        self.agents: list[str] = list(self.possible_agents)

        self._num_agents = num_agents
        self._episode_horizon_s = episode_horizon_s
        self._physics_dt_s = physics_dt_s
        self._control_dt_s = control_dt_s
        self._lookahead_s = lookahead_s
        self._k_neighbors = k_neighbors
        self._hard_body_radius_m = hard_body_radius_m
        self._capability_template = capability_template
        self._distribution_mix = distribution_mix
        self._noise_model = noise_model
        self._reward_weights = reward_weights
        self._margin_fn = margin_fn
        self._phase_fraction = resolved["phase_fraction"]
        self._risk_params = {
            "risk_epsilon_m": resolved["risk_epsilon_m"],
            "risk_z0": resolved["risk_z0"],
            "risk_k": resolved["risk_k"],
        }
        self._density_radius_m = resolved["density_radius_m"]
        self._density_ref_count = resolved["density_ref_count"]
        self._mission_weights = resolved["mission_weights"]
        self._coordination_clip = float(resolved["coordination_clip"])
        # Outcome metrics are scored against the PRD section 22 standard of the DEFAULT config, never a tuned
        # overlay, so every run (and every tuning point) is judged by one reference (docs/OrbitGuard_progress.md).
        self._reference_margin = margin_from_config(base_config, "adaptive")
        self._reference_density = (defaults["density_radius_m"], defaults["density_ref_count"])
        self._phase_spread_rad = phase_spread_rad
        self._capability_degradation = capability_degradation
        self._last_scenario = None
        self._shell = fit_shell_distribution(load_tle_snapshot(tle_path))
        self._substeps_per_control = round(control_dt_s / physics_dt_s)
        self._horizon_steps = round(episode_horizon_s / control_dt_s)

        self._observation_space = spaces.Dict(
            {
                "own": spaces.Box(low=-np.inf, high=np.inf, shape=(_OWN_DIM,), dtype=np.float32),
                "neighbors": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(k_neighbors, _NEIGHBOR_DIM), dtype=np.float32
                ),
                "global_summary": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(_GLOBAL_SUMMARY_DIM,), dtype=np.float32
                ),
            }
        )
        self._action_space = spaces.Discrete(_NUM_ACTIONS)
        self.state_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(num_agents * (_OWN_DIM + k_neighbors * _NEIGHBOR_DIM + _GLOBAL_SUMMARY_DIM),),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self._sensing_rng = np.random.default_rng()
        self._step_count = 0
        self._ground_truth: list[np.ndarray] = []
        self._capabilities: list[SatelliteCapability] = []
        self._initial_budgets_mps: tuple[float, ...] = ()
        self._shadow_states: list[np.ndarray] = []
        self._last_observations: dict = {}
        self._pending_assessments = None
        self._pending_beliefs: list = []
        self._pending_predictions: list = []
        self._pending_mission_deviations: list[float] = [0.0] * num_agents
        self._collided_agents: set[int] = set()
        self.episode_seed: int | None = None
        self._metrics = EpisodeMetricsTracker()
        self.last_step: dict = {}
        self._step_records: tuple[dict, ...] = ()

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        if seed is None:
            # Unseeded (training) episodes draw from a range reserved away from the validation and
            # final-test pools (evaluation/seed_pools.py), so a split can never leak into training.
            seed = int(np.random.default_rng().integers(TRAINING_SEED_FLOOR, 2**31 - 1))
        self.episode_seed = seed
        self._rng = np.random.default_rng(seed)
        scenario_seed = int(self._rng.integers(0, 2**31 - 1))

        if self._num_agents < MIN_SUPPORTED_N:
            if self._phase_spread_rad is not None:
                raise ValueError("phase_spread_rad (density shift) needs num_agents >= 10")
            scenario = generate_small_n_conjunction_scenario(
                self._num_agents, scenario_seed, self._shell, self._capability_template,
                self._episode_horizon_s, self._physics_dt_s, self._distribution_mix,
            )
        else:
            scenario = generate_conjunction_scenario(
                self._num_agents, scenario_seed, self._shell, self._capability_template,
                self._episode_horizon_s, self._physics_dt_s, self._distribution_mix,
                phase_spread_rad=self._phase_spread_rad,
            )

        self._last_scenario = scenario
        self._ground_truth = [sat.state.as_vector() for sat in scenario.satellites]
        self._capabilities = self._with_drawn_budgets([sat.capability for sat in scenario.satellites], seed)
        if self._capability_degradation is not None:
            self._capabilities = self._capability_degradation.apply(self._capabilities, scenario.injected_conjunctions)
        self._initial_budgets_mps = tuple(c.remaining_delta_v_mps for c in self._capabilities)
        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._collided_agents = set()
        self._metrics.reset()
        self.last_step = {}
        self._step_records = ()

        self._sensing_rng = np.random.default_rng(int(self._rng.integers(0, 2**31 - 1)))
        self._shadow_states = [
            form_belief(observe(GroundTruthState(gt), self._sensing_rng, self._noise_model), self._noise_model).state_vector
            for gt in self._ground_truth
        ]

        beliefs = self._current_beliefs()
        self._store_pending(beliefs)
        self._refresh_mission_deviations(beliefs)
        observations = self._observations_from(beliefs, self._pending_assessments)
        self._last_observations = dict(observations)
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        pre_assessments = self._pending_assessments
        pre_beliefs = self._pending_beliefs
        # Decision-time facts for the §40 records, captured before the step moves time, agents or observations.
        decision_time_s = self._step_count * self._control_dt_s
        decision_agents = list(self.agents)
        decision_observations = dict(self._last_observations)
        approved_actions, decision = self._approve_actions(actions)
        outcome_assessments = self._outcome_assessments(actions, decision)  # before actuation moves the state
        applied_delta_v, approved_requested = self._apply_actions(approved_actions)
        requested_delta_v = approved_requested if approved_actions is actions else self._requested_delta_v(actions)
        fine_trajectories = self._propagate_ground_truth(applied_delta_v)
        newly_collided, collided_pairs, min_separations = self._check_collisions(fine_trajectories)
        self._collided_agents |= newly_collided
        self._step_count += 1
        truncated_now = self._step_count >= self._horizon_steps
        self._advance_shadow_states()

        post_beliefs = self._current_beliefs()
        post_assessments = self._store_pending(post_beliefs)
        # Before the metrics record them: the episode's mission deviation is the POST-step value, the same one
        # the reward's mission term and the next observation use (it used to lag one step behind).
        self._refresh_mission_deviations(post_beliefs)
        self.last_step = self._describe_step(actions, approved_actions, applied_delta_v, collided_pairs)
        self._step_records = tuple(
            self._step_record(
                agent, decision_time_s, decision_observations[agent], actions, approved_actions,
                requested_delta_v, pre_beliefs, pre_assessments, post_assessments, decision,
            )
            for agent in decision_agents
        )
        self._record_step_metrics(
            applied_delta_v, collided_pairs, min_separations, pre_assessments, post_assessments, decision, pre_beliefs
        )

        rewards = self._compute_rewards(
            actions, requested_delta_v, pre_assessments, outcome_assessments,
            post_beliefs, self._shadow_states, newly_collided,
        )

        observations = self._observations_from(post_beliefs, post_assessments)
        self._last_observations.update(observations)

        terminations = {agent: self._agent_index[agent] in newly_collided for agent in self.agents}
        truncations = {agent: truncated_now for agent in self.agents}
        infos = self._step_infos(decision)

        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        flattened = [flatten(self._observation_space, self._last_observations[agent]) for agent in self.possible_agents]
        return np.concatenate(flattened).astype(np.float32)

    @property
    def delta_v_budget_mps(self) -> float:
        """The NOMINAL per-satellite budget (the capability template); each satellite's own is `initial_delta_v_budgets_mps`."""
        return self._capability_template.remaining_delta_v_mps

    @property
    def max_delta_v_per_action_mps(self) -> float:
        """Delta-v of one full-size burn (phase burns use a fraction of it)."""
        return self._capability_template.max_delta_v_per_action_mps

    def initial_delta_v_budgets_mps(self) -> dict[str, float]:
        """Each satellite's budget at the start of this episode, after any capability degradation (CORE-3)."""
        return {agent: self._initial_budgets_mps[self._agent_index[agent]] for agent in self.possible_agents}

    def delta_v_used_mps(self) -> dict[str, float]:
        """Delta-v each satellite has spent so far this episode: its OWN initial budget minus what remains."""
        return {
            agent: self._initial_budgets_mps[index] - self._capabilities[index].remaining_delta_v_mps
            for agent, index in self._agent_index.items()
        }

    def step_records(self) -> list[dict]:
        """PRD §40 per-control-step records of the last step, one per satellite active at its decision time.

        Keys and units are documented in `rl.step_records`; values are plain Python types. Empty before the
        first step of an episode. Returns copies, so callers may keep or modify them.
        """
        return copy.deepcopy(list(self._step_records))

    def render(self):
        raise NotImplementedError("OrbitGuardEnv has no rendering.")

    def _step_record(
        self, agent, decision_time_s, observation, actions, approved_actions, requested_delta_v,
        pre_beliefs, pre_assessments, post_assessments, decision,
    ) -> dict:
        idx = self._agent_index[agent]
        intervened, reason = self._shield_record(decision, idx)
        return build_step_record(
            timestamp_s=decision_time_s,
            satellite_id=agent,
            index=idx,
            observed_state=pre_beliefs[idx].state_vector,
            belief_features=flatten(self._observation_space, observation),
            selected_action=int(actions.get(agent, ManeuverIntent.NO_MANEUVER)),
            mapped_delta_v_mps=requested_delta_v[idx],
            shield_action=int(approved_actions.get(agent, ManeuverIntent.NO_MANEUVER)),
            shield_intervened=intervened,
            shield_reason=reason,
            pre_assessments=pre_assessments,
            post_assessments=post_assessments,
            remaining_delta_v_mps=self._capabilities[idx].remaining_delta_v_mps,
            mission_deviation=self._pending_mission_deviations[idx],
        )

    def _shield_record(self, decision, index: int) -> tuple[bool, list[str]]:
        """Hook: (intervened, reason kinds) of the shield for satellite `index`. No shield here: (False, [])."""
        return False, []

    def _with_drawn_budgets(self, capabilities: list[SatelliteCapability], episode_seed: int) -> list[SatelliteCapability]:
        """Each satellite's initial budget ~ Uniform(range) from the dedicated budget stream; unchanged without a range."""
        if self._budget_range_mps is None:
            return capabilities
        rng = np.random.default_rng(np.random.SeedSequence([int(episode_seed), _DELTA_V_BUDGET_STREAM_ID]))
        budgets = rng.uniform(*self._budget_range_mps, size=len(capabilities))
        return [c.with_remaining_delta_v(float(b)) for c, b in zip(capabilities, budgets, strict=True)]

    def _store_pending(self, beliefs) -> list:
        """Predict and assess `beliefs`; keep all three as next step's decision-time state."""
        self._pending_beliefs = list(beliefs)
        self._pending_predictions = predict_all(beliefs, self._lookahead_s, self._physics_dt_s)
        self._pending_assessments = assess_all_pairs(
            beliefs, self._lookahead_s, self._physics_dt_s,
            density_radius_m=self._density_radius_m, density_ref_count=self._density_ref_count,
            margin_fn=self._margin_fn, predicted=self._pending_predictions, **self._risk_params,
        )
        return self._pending_assessments

    def _approve_actions(self, actions: dict):
        """Hook: turn raw proposals into the actions actually applied. Identity here (no shield)."""
        return actions, None

    def _outcome_assessments(self, actions: dict, decision) -> list[PairAssessment]:
        """Hook: the predicted outcome of the RAW joint proposal, scoring R_safety and the coordination term.

        The one estimator of both shield arms (CORE-2): decision-time beliefs and baseline predictions, burns
        from the belief state sized by each satellite's own per-action limit; satellites without an action
        (non-controllable) burn nothing. Must run before actuation, while the pending state is still "pre".
        """
        intents = [int(actions.get(agent, ManeuverIntent.NO_MANEUVER)) for agent in self.possible_agents]
        return assess_proposal(
            self._pending_beliefs, intents, [c.max_delta_v_per_action_mps for c in self._capabilities],
            phase_fraction=self._phase_fraction, lookahead_s=self._lookahead_s, physics_dt_s=self._physics_dt_s,
            margin_fn=self._margin_fn, baseline_predictions=self._pending_predictions,
            density_radius_m=self._density_radius_m, density_ref_count=self._density_ref_count, **self._risk_params,
        )

    def _step_infos(self, decision) -> dict:
        return {agent: {} for agent in self.agents}

    def _requested_delta_v(self, actions: dict) -> list[np.ndarray]:
        """Delta-v each agent's (raw, pre-cap) request maps to, from its true state -- no side effects."""
        requested = [np.zeros(3) for _ in range(self._num_agents)]
        for agent, action in actions.items():
            idx = self._agent_index[agent]
            gt = self._ground_truth[idx]
            requested[idx] = map_intent_to_delta_v(
                ManeuverIntent(action), gt[:3], gt[3:6],
                base_delta_v_mps=self._capabilities[idx].max_delta_v_per_action_mps,
                phase_fraction=self._phase_fraction,
            )
        return requested

    def _current_beliefs(self):
        beliefs = []
        for gt in self._ground_truth:
            observation = observe(GroundTruthState(gt), self._sensing_rng, self._noise_model)
            beliefs.append(form_belief(observation, self._noise_model))
        return beliefs

    def _advance_shadow_states(self) -> None:
        """Advance the no-maneuver shadow one control step (PRD §25, Q9).

        Propagating incrementally (vs. from t=0 every call) matches a
        from-scratch propagation of the same total step count (to roundoff) --
        RK4 is a pure step-wise recurrence (`physics.integrator.propagate` is
        just repeated `rk4_step` calls) -- but turns an O(step_count) per-step cost
        into O(1), avoiding a quadratic episode cost.
        """
        trajectories = propagate_batch(
            np.array(self._shadow_states), self._physics_dt_s, self._substeps_per_control, state_derivative_batch
        )
        self._shadow_states = [trajectory[-1] for trajectory in trajectories]

    def _refresh_mission_deviations(self, beliefs) -> None:
        """Weighted mission deviation (PRD §25) of every active satellite: `beliefs` against the shadow, same instant."""
        for agent in self.agents:
            idx = self._agent_index[agent]
            components = compute_mission_deviation(beliefs[idx].state_vector, self._shadow_states[idx])
            self._pending_mission_deviations[idx] = compute_weighted_mission_deviation(components, **self._mission_weights)

    def _observations_from(self, beliefs, assessments) -> dict:
        """Observations from `beliefs`; the mission deviation is the one `_refresh_mission_deviations` just stored."""
        observations = {}
        for agent in self.agents:
            idx = self._agent_index[agent]
            own_mission_deviation = self._pending_mission_deviations[idx]
            candidates = build_neighbor_candidates(idx, beliefs, assessments)
            cap = self._capabilities[idx]
            obs = build_observation(
                own_position_m=beliefs[idx].state_vector[:3],
                own_velocity_mps=beliefs[idx].state_vector[3:6],
                own_remaining_delta_v_mps=cap.remaining_delta_v_mps,
                own_mission_deviation=own_mission_deviation,
                own_max_delta_v_per_action_mps=cap.max_delta_v_per_action_mps,
                own_maneuver_enabled=cap.maneuver_enabled,
                neighbor_candidates=candidates,
                k_neighbors=self._k_neighbors,
            )
            observations[agent] = {key: np.asarray(value, dtype=np.float32) for key, value in obs.items()}
        return observations

    def _apply_actions(self, actions: dict) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Apply capability-capped burns; also return what each agent REQUESTED (pre-cap)."""
        applied_delta_v = [np.zeros(3) for _ in range(self._num_agents)]
        requested_delta_v = [np.zeros(3) for _ in range(self._num_agents)]
        for agent, action in actions.items():
            if action not in range(_NUM_ACTIONS):
                raise ValueError(f"action {action} out of range for agent {agent}")
            idx = self._agent_index[agent]
            cap = self._capabilities[idx]
            gt = self._ground_truth[idx]
            proposed = map_intent_to_delta_v(
                ManeuverIntent(action), gt[:3], gt[3:6],
                base_delta_v_mps=cap.max_delta_v_per_action_mps, phase_fraction=self._phase_fraction,
            )
            proposed_mag = float(np.linalg.norm(proposed))

            if not cap.maneuver_enabled or proposed_mag < 1e-12:
                applied_mag = 0.0
                applied = np.zeros(3)
            else:
                applied_mag = min(proposed_mag, cap.remaining_delta_v_mps)
                applied = proposed * (applied_mag / proposed_mag)

            applied_delta_v[idx] = applied
            requested_delta_v[idx] = proposed
            self._capabilities[idx] = cap.with_remaining_delta_v(cap.remaining_delta_v_mps - applied_mag)
        return applied_delta_v, requested_delta_v

    def _propagate_ground_truth(self, applied_delta_v: list[np.ndarray]) -> list[np.ndarray]:
        states_after_burn = np.array(
            [np.concatenate([gt[:3], gt[3:6] + dv]) for gt, dv in zip(self._ground_truth, applied_delta_v, strict=True)]
        )
        trajectories = propagate_batch(
            states_after_burn, self._physics_dt_s, self._substeps_per_control, state_derivative_batch
        )
        fine_trajectories = list(trajectories)
        self._ground_truth = [trajectory[-1] for trajectory in fine_trajectories]
        return fine_trajectories

    def _check_collisions(self, fine_trajectories: list[np.ndarray]):
        """Ground-truth collision check; returns (newly collided agents, colliding pairs, per-pair min separation)."""
        newly_collided: set[int] = set()
        collided_pairs: set[tuple[int, int]] = set()
        min_separations: dict[tuple[int, int], float] = {}
        for i, j in generate_candidate_pairs(self._num_agents):
            result = detect_collision(
                fine_trajectories[i], fine_trajectories[j], self._physics_dt_s,
                self._hard_body_radius_m, self._hard_body_radius_m,
            )
            min_separations[(i, j)] = result.min_separation_m
            if result.collided:
                newly_collided.add(i)
                newly_collided.add(j)
                collided_pairs.add((i, j))
        return newly_collided, collided_pairs, min_separations

    def _describe_step(self, proposed: dict, approved: dict, applied_delta_v, collided_pairs) -> dict:
        """Per-satellite record of the step just taken, for traces and replay (inactive satellites: 0/False)."""
        proposals = [int(proposed.get(agent, 0)) for agent in self.possible_agents]
        approvals = [int(approved.get(agent, 0)) for agent in self.possible_agents]
        return {
            "proposed": proposals,
            "approved": approvals,
            "intervened": [p != a for p, a in zip(proposals, approvals, strict=True)],
            "applied_dv_mps": [float(np.linalg.norm(dv)) for dv in applied_delta_v],
            "collided_pairs": sorted(collided_pairs),
        }

    def snapshot(self) -> dict:
        """State for traces/replay: true positions (display and metrics only), capability, belief-based pair assessments."""
        return {
            "positions_m": np.array([gt[:3] for gt in self._ground_truth]),
            "alive": [agent in self.agents for agent in self.possible_agents],
            "remaining_dv_mps": [c.remaining_delta_v_mps for c in self._capabilities],
            "maneuver_enabled": [c.maneuver_enabled for c in self._capabilities],
            "mission_deviation": list(self._pending_mission_deviations),
            "pairs": [(a.i, a.j, a.d_min_m, a.d_safe_m, a.risk_proxy) for a in self._pending_assessments],
        }

    def _reference_assessments(self, assessments, beliefs) -> list:
        """Re-score `assessments` under the PRD §22 adaptive margin, whatever margin the run uses.

        Outcome metrics must compare every model against ONE safety standard; a Fixed-Margin run
        scoring itself against its own constant would be marking its own homework.
        """
        positions = np.array([belief.state_vector[:3] for belief in beliefs])
        radius_m, ref_count = self._reference_density
        return [
            replace(
                a,
                d_safe_m=self._reference_margin(
                    a.sigma_rel_m, pair_density(a.i, a.j, positions, radius_m, ref_count), a.risk_proxy
                ),
            )
            for a in assessments
        ]

    def _record_step_metrics(
        self, applied_delta_v, collided_pairs, min_separations, pre_assessments, post_assessments, decision, pre_beliefs
    ) -> None:
        def as_triples(assessments) -> dict:
            return {(a.i, a.j): (a.d_min_m, a.d_safe_m, a.risk_proxy) for a in assessments}

        self._metrics.record_step(
            applied_delta_v_mps=[float(np.linalg.norm(dv)) for dv in applied_delta_v],
            collided=collided_pairs,
            min_separation_m=min_separations,
            pre=as_triples(self._reference_assessments(pre_assessments, pre_beliefs)),
            post=as_triples(self._reference_assessments(post_assessments, self._pending_beliefs)),
            mission_deviations=[self._pending_mission_deviations[self._agent_index[a]] for a in self.agents],
            shield=self._shield_step(decision),
        )

    def _shield_step(self, decision) -> ShieldStep | None:
        """Hook: shield accounting for the metrics tracker. None here (no shield)."""
        return None

    def episode_summary(self) -> EpisodeSummary:
        """Outcome metrics and failure labels for the episode so far (PRD §34, §36, §40)."""
        capabilities = self._capabilities
        return self._metrics.summary(
            remaining_delta_v_mps=[c.remaining_delta_v_mps for c in capabilities],
            smallest_burn_mps=[0.5 * c.max_delta_v_per_action_mps for c in capabilities],
            maneuver_enabled=[c.maneuver_enabled for c in capabilities],
        )

    def _compute_rewards(
        self, actions, requested_delta_v, pre_assessments, outcome_assessments, post_beliefs, shadow_states, newly_collided
    ) -> dict:
        """Safety/coordination on the predicted raw-proposal outcome; mission/collision on the realized step."""
        total_risk_pre = sum(a.risk_proxy for a in pre_assessments)
        total_risk_post = sum(a.risk_proxy for a in outcome_assessments)
        coordination_term = compute_clipped_coordination_term(total_risk_pre, total_risk_post, self._coordination_clip)

        rewards = {}
        for agent in actions:
            idx = self._agent_index[agent]
            margins = [
                compute_pairwise_margin(a.d_min_m, a.d_safe_m)
                for a in outcome_assessments
                if idx in (a.i, a.j)
            ]
            safety_term = compute_safety_term(margins)

            components = compute_mission_deviation(post_beliefs[idx].state_vector, shadow_states[idx])
            mission_term = compute_mission_term(components.d_a, components.d_e, components.d_i, components.d_m, **self._mission_weights)

            # Charged on the REQUEST, not the capped burn: once the budget is spent every
            # action is a physical no-op, and an applied-only cost would leave the policy
            # no reason to stop asking (PRD §28: agents are penalized for their proposals).
            delta_v_term = compute_delta_v_term(requested_delta_v[idx])
            collided = idx in newly_collided

            rewards[agent] = compute_reward(
                safety_term, mission_term, delta_v_term, coordination_term, collided, **self._reward_weights
            )
        return rewards
