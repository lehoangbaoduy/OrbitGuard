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
  4. Map each agent's discrete action to an ECI delta-v (Part B), capped by
     capability (never silently substituted with NO_MANEUVER, PRD §13).
  5. Apply delta-v to GROUND TRUTH and propagate it at `physics_dt_s`
     resolution across the full control step (not just the endpoints --
     relative velocities ~2km/s traverse ~120km in one 60s control step).
  6. Ground-truth collision check over that fine-grained window (PRD §18).
  7. Form a fresh belief AFTER the physical step and assess all pairs again
     -- this is both the "post" assessment feeding this step's reward (PRD
     §21/§27 R_safety and the coordination term) AND next step's "pre".
  8. Mission deviation (PRD §25) compares the post-step belief against a
     deterministically-evolved (zero-noise-after-reset) shadow belief, at
     the same instant -- never ground truth (Q14's leakage-boundary choice).
  9. Assemble each agent's reward (Part C) and this step's observations
     (next step's "pre" observations) together.

Ground truth is used ONLY for actuation (converting a maneuver intent to an
ECI delta-v relative to the satellite's own true state, step 4), propagation,
and collision detection (steps 5-6) above; everything feeding observations
or rewards is belief-derived, enforced structurally by `sensing.belief`'s own
TypeError guards.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatten
from pettingzoo import ParallelEnv

from orbit_guard.conjunction.collision import detect_collision
from orbit_guard.conjunction.screening import generate_candidate_pairs
from orbit_guard.physics.dynamics import state_derivative
from orbit_guard.physics.integrator import propagate
from orbit_guard.physics.mission_deviation import compute_mission_deviation, compute_weighted_mission_deviation
from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.rl.conjunction_assessment import assess_all_pairs, build_neighbor_candidates
from orbit_guard.rl.observation import build_observation
from orbit_guard.rl.reward import (
    compute_coordination_term,
    compute_delta_v_term,
    compute_mission_term,
    compute_pairwise_margin,
    compute_reward,
    compute_safety_term,
)
from orbit_guard.safety.maneuver_mapper import ManeuverIntent, map_intent_to_delta_v
from orbit_guard.scenarios.conjunction_generator import DEFAULT_DISTRIBUTION_MIX, generate_conjunction_scenario
from orbit_guard.scenarios.constellation import MIN_SUPPORTED_N, fit_shell_distribution, load_tle_snapshot
from orbit_guard.scenarios.small_n_scenario import generate_small_n_conjunction_scenario
from orbit_guard.sensing.belief import GroundTruthState, observe, form_belief
from orbit_guard.sensing.noise import NoiseModel

_DEFAULT_TLE_PATH = Path(__file__).resolve().parents[2] / "data" / "TLE_snapshot_53deg_shell.csv"
_DEFAULT_CAPABILITY = SatelliteCapability(
    max_delta_v_per_action_mps=0.1,
    remaining_delta_v_mps=2.0,
    maneuver_enabled=True,
    mission_deviation_limit=1.0,
)
_DEFAULT_NOISE_MODEL = NoiseModel(position_sigma_m=100.0, velocity_sigma_mps=0.1)
_NUM_ACTIONS = len(ManeuverIntent)
_K_NEIGHBORS = 4
_HARD_BODY_RADIUS_M = 5.0
_REWARD_WEIGHTS = {"w1": 1.0, "w2": 0.3, "w3": 0.1, "w4": 0.5, "collision_penalty": -100.0}

# Duplicated raw shape numbers rather than imported from rl.observation -- Part
# B's frozen contract (tests/rl/test_observation.py) fixes these shapes via
# literals, not named constants, so there is nothing to import; matches
# tests/rl/stub_env.py's own documented rationale for the same duplication.
_OWN_DIM = 10
_NEIGHBOR_DIM = 10
_GLOBAL_SUMMARY_DIM = 4


class OrbitGuardEnv(ParallelEnv):
    """Multi-agent satellite collision-avoidance environment (PRD §19)."""

    metadata = {"name": "orbit_guard_env_v0", "is_parallelizable": True}

    def __init__(
        self,
        num_agents: int = 2,
        episode_horizon_s: float = 21_600.0,
        physics_dt_s: float = 10.0,
        control_dt_s: float = 60.0,
        lookahead_s: float = 1800.0,
        k_neighbors: int = _K_NEIGHBORS,
        hard_body_radius_m: float = _HARD_BODY_RADIUS_M,
        capability_template: SatelliteCapability = _DEFAULT_CAPABILITY,
        distribution_mix=DEFAULT_DISTRIBUTION_MIX,
        noise_model: NoiseModel = _DEFAULT_NOISE_MODEL,
        tle_path: str | Path = _DEFAULT_TLE_PATH,
        reward_weights: dict | None = None,
    ):
        if num_agents < 2:
            raise ValueError("OrbitGuard is a multi-agent env; num_agents must be >= 2")
        if control_dt_s <= 0 or physics_dt_s <= 0 or control_dt_s < physics_dt_s:
            raise ValueError("control_dt_s and physics_dt_s must be > 0 with control_dt_s >= physics_dt_s")

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
        self._reward_weights = reward_weights or _REWARD_WEIGHTS
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
        self._shadow_states: list[np.ndarray] = []
        self._last_observations: dict = {}
        self._pending_assessments = None
        self._collided_agents: set[int] = set()

    def observation_space(self, agent):
        return self._observation_space

    def action_space(self, agent):
        return self._action_space

    def reset(self, seed=None, options=None):
        self._rng = np.random.default_rng(seed)
        scenario_seed = int(self._rng.integers(0, 2**31 - 1))

        if self._num_agents < MIN_SUPPORTED_N:
            scenario = generate_small_n_conjunction_scenario(
                self._num_agents, scenario_seed, self._shell, self._capability_template,
                self._episode_horizon_s, self._physics_dt_s, self._distribution_mix,
            )
        else:
            scenario = generate_conjunction_scenario(
                self._num_agents, scenario_seed, self._shell, self._capability_template,
                self._episode_horizon_s, self._physics_dt_s, self._distribution_mix,
            )

        self._ground_truth = [sat.state.as_vector() for sat in scenario.satellites]
        self._capabilities = [sat.capability for sat in scenario.satellites]
        self.agents = list(self.possible_agents)
        self._step_count = 0
        self._collided_agents = set()

        self._sensing_rng = np.random.default_rng(int(self._rng.integers(0, 2**31 - 1)))
        self._shadow_states = [
            form_belief(observe(GroundTruthState(gt), self._sensing_rng, self._noise_model), self._noise_model).state_vector
            for gt in self._ground_truth
        ]

        beliefs = self._current_beliefs()
        self._pending_assessments = assess_all_pairs(beliefs, self._lookahead_s, self._physics_dt_s)
        observations = self._observations_from(beliefs, self._pending_assessments, self._shadow_states)
        self._last_observations = dict(observations)
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        pre_assessments = self._pending_assessments
        applied_delta_v = self._apply_actions(actions)
        fine_trajectories = self._propagate_ground_truth(applied_delta_v)
        newly_collided = self._check_collisions(fine_trajectories)
        self._collided_agents |= newly_collided
        self._step_count += 1
        truncated_now = self._step_count >= self._horizon_steps
        self._advance_shadow_states()

        post_beliefs = self._current_beliefs()
        post_assessments = assess_all_pairs(post_beliefs, self._lookahead_s, self._physics_dt_s)
        self._pending_assessments = post_assessments

        rewards = self._compute_rewards(
            actions, applied_delta_v, pre_assessments, post_assessments, post_beliefs, self._shadow_states, newly_collided
        )

        observations = self._observations_from(post_beliefs, post_assessments, self._shadow_states)
        self._last_observations.update(observations)

        terminations = {agent: self._agent_index[agent] in newly_collided for agent in self.agents}
        truncations = {agent: truncated_now for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self.agents = [agent for agent in self.agents if not (terminations[agent] or truncations[agent])]
        return observations, rewards, terminations, truncations, infos

    def state(self) -> np.ndarray:
        flattened = [flatten(self._observation_space, self._last_observations[agent]) for agent in self.possible_agents]
        return np.concatenate(flattened).astype(np.float32)

    def render(self):
        raise NotImplementedError("OrbitGuardEnv has no rendering.")

    def _current_beliefs(self):
        beliefs = []
        for gt in self._ground_truth:
            observation = observe(GroundTruthState(gt), self._sensing_rng, self._noise_model)
            beliefs.append(form_belief(observation, self._noise_model))
        return beliefs

    def _advance_shadow_states(self) -> None:
        """Advance the no-maneuver shadow one control step (PRD §25, Q9).

        Propagating incrementally (vs. from t=0 every call) is bit-identical
        to a from-scratch propagation of the same total step count -- RK4 is
        a pure step-wise recurrence (`physics.integrator.propagate` is just
        repeated `rk4_step` calls) -- but turns an O(step_count) per-step cost
        into O(1), avoiding a quadratic episode cost.
        """
        self._shadow_states = [
            propagate(state, self._physics_dt_s, self._substeps_per_control, state_derivative)[-1]
            for state in self._shadow_states
        ]

    def _observations_from(self, beliefs, assessments, shadow_states) -> dict:
        observations = {}
        for agent in self.agents:
            idx = self._agent_index[agent]
            components = compute_mission_deviation(beliefs[idx].state_vector, shadow_states[idx])
            own_mission_deviation = compute_weighted_mission_deviation(components)
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

    def _apply_actions(self, actions: dict) -> list[np.ndarray]:
        applied_delta_v = [np.zeros(3) for _ in range(self._num_agents)]
        for agent, action in actions.items():
            if action not in range(_NUM_ACTIONS):
                raise ValueError(f"action {action} out of range for agent {agent}")
            idx = self._agent_index[agent]
            cap = self._capabilities[idx]
            gt = self._ground_truth[idx]
            proposed = map_intent_to_delta_v(
                ManeuverIntent(action), gt[:3], gt[3:6], base_delta_v_mps=cap.max_delta_v_per_action_mps
            )
            proposed_mag = float(np.linalg.norm(proposed))

            if not cap.maneuver_enabled or proposed_mag < 1e-12:
                applied_mag = 0.0
                applied = np.zeros(3)
            else:
                applied_mag = min(proposed_mag, cap.remaining_delta_v_mps)
                applied = proposed * (applied_mag / proposed_mag)

            applied_delta_v[idx] = applied
            self._capabilities[idx] = cap.with_remaining_delta_v(cap.remaining_delta_v_mps - applied_mag)
        return applied_delta_v

    def _propagate_ground_truth(self, applied_delta_v: list[np.ndarray]) -> list[np.ndarray]:
        fine_trajectories = []
        for idx in range(self._num_agents):
            gt = self._ground_truth[idx]
            state_after_burn = np.concatenate([gt[:3], gt[3:6] + applied_delta_v[idx]])
            trajectory = propagate(state_after_burn, self._physics_dt_s, self._substeps_per_control, state_derivative)
            fine_trajectories.append(trajectory)
            self._ground_truth[idx] = trajectory[-1]
        return fine_trajectories

    def _check_collisions(self, fine_trajectories: list[np.ndarray]) -> set[int]:
        newly_collided: set[int] = set()
        for i, j in generate_candidate_pairs(self._num_agents):
            result = detect_collision(
                fine_trajectories[i], fine_trajectories[j], self._physics_dt_s,
                self._hard_body_radius_m, self._hard_body_radius_m,
            )
            if result.collided:
                newly_collided.add(i)
                newly_collided.add(j)
        return newly_collided

    def _compute_rewards(
        self, actions, applied_delta_v, pre_assessments, post_assessments, post_beliefs, shadow_states, newly_collided
    ) -> dict:
        total_risk_pre = sum(a.risk_proxy for a in pre_assessments)
        total_risk_post = sum(a.risk_proxy for a in post_assessments)
        coordination_term = compute_coordination_term(total_risk_pre, total_risk_post)

        rewards = {}
        for agent in actions:
            idx = self._agent_index[agent]
            margins = [
                compute_pairwise_margin(a.d_min_m, a.d_safe_m)
                for a in post_assessments
                if idx in (a.i, a.j)
            ]
            safety_term = compute_safety_term(margins)

            components = compute_mission_deviation(post_beliefs[idx].state_vector, shadow_states[idx])
            mission_term = compute_mission_term(components.d_a, components.d_e, components.d_i, components.d_m)

            delta_v_term = compute_delta_v_term(applied_delta_v[idx])
            collided = idx in newly_collided

            rewards[agent] = compute_reward(
                safety_term, mission_term, delta_v_term, coordination_term, collided, **self._reward_weights
            )
        return rewards
