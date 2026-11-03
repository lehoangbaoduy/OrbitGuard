"""Shield-active OrbitGuardEnv (PRD §28 "training-time shield behavior", P4-M6).

A SEPARATE class, not a flag on `OrbitGuardEnv`: the base environment never
imports `safety.shield`, so a no-shield training run structurally cannot call
it (tests/rl/test_shielded_environment.py pins that). "Adaptive MAPPO without
shield" is therefore a different training run built from `OrbitGuardEnv`, never
an evaluation-time toggle on shielded weights.

Semantics (PRD §28):
  * the realized transition applies the shield-APPROVED joint action;
  * the safety and coordination reward terms are scored against the predicted
    outcome of the RAW joint proposal from the decision-time beliefs, so an
    agent is still penalized for proposing something unsafe even when the
    shield prevented the bad outcome. This is the ONE estimator both arms use
    (CORE-2, `pair_assessment.assess_proposal`): the base environment computes
    it itself, and here it is read from `ShieldResult.proposal_evaluation`,
    which the shield builds with the same functions -- equal by construction,
    so the shield-vs-no-shield ablation differs only in the shield;
  * the delta-v term is charged on the raw request (Q21), so the shield's
    own corrective burns are free to the agent -- it is not punished for the
    shield's work, and not rewarded for relying on it either;
  * the mission term stays on the realized trajectory.

The shield's MISSION_LIMIT input is NOT the belief-vs-shadow deviation the
mission term and the observation use (Q14). That value is mostly sensing noise:
under the noise_increase shift, satellites that never burned exceed the limit
from noise alone. The shield instead reads the noise-free deviation of each
satellite's maneuvered shadow (the reset shadow propagated with its own applied
burns) from the no-maneuver shadow (`safety.maneuver_shadow`). It is exactly 0
for a satellite that never burned.

The PRD §40 step records (`step_records()`, CORE-4c) take shield_intervened from
the ShieldResult and shield_reason from the raw proposal's violations involving
each satellite, and only on a rejected step.
"""

from __future__ import annotations

from orbit_guard.config import env_kwargs_from_config, load_config
from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.episode_metrics import ShieldStep
from orbit_guard.safety.maneuver_shadow import (
    advance_maneuvered_shadows,
    maneuver_mission_deviations,
    no_maneuver_shadows,
)
from orbit_guard.safety.shield import JointActionShield, ShieldResult


class ShieldedOrbitGuardEnv(OrbitGuardEnv):
    """`OrbitGuardEnv` whose actions pass through the joint-action safety shield."""

    def __init__(
        self,
        *args,
        shield_exact_max_agents: int | None = None,
        shield_beam_width: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        defaults = env_kwargs_from_config(load_config(), shielded=True)
        exact_max_agents = defaults["shield_exact_max_agents"] if shield_exact_max_agents is None else shield_exact_max_agents
        beam_width = defaults["shield_beam_width"] if shield_beam_width is None else shield_beam_width
        self._shield = JointActionShield(
            lookahead_s=self._lookahead_s,
            physics_dt_s=self._physics_dt_s,
            margin_fn=self._margin_fn,
            phase_fraction=self._phase_fraction,
            collision_distance_m=2.0 * self._hard_body_radius_m,
            exact_max_agents=exact_max_agents,
            beam_width=beam_width,
            density_radius_m=self._density_radius_m,
            density_ref_count=self._density_ref_count,
            mission_weights=self._mission_weights,
            **self._risk_params,
        )
        self.last_shield_result: ShieldResult | None = None
        self._maneuvered_shadows = no_maneuver_shadows(self._num_agents)

    def reset(self, seed=None, options=None):
        self._maneuvered_shadows = no_maneuver_shadows(self._num_agents)
        return super().reset(seed=seed, options=options)

    def step(self, actions):
        shadow_before = tuple(self._shadow_states)
        result = super().step(actions)
        self._maneuvered_shadows = advance_maneuvered_shadows(
            self._maneuvered_shadows, shadow_before, self.last_step["approved"], self.last_step["applied_dv_mps"],
            [c.max_delta_v_per_action_mps for c in self._capabilities],
            phase_fraction=self._phase_fraction, physics_dt_s=self._physics_dt_s, substeps=self._substeps_per_control,
        )
        return result

    def maneuver_mission_deviations(self) -> tuple[float, ...]:
        """Each satellite's noise-free maneuver-induced mission deviation: the shield's MISSION_LIMIT input."""
        return maneuver_mission_deviations(self._maneuvered_shadows, self._shadow_states, self._mission_weights)

    def _approve_actions(self, actions: dict):
        proposals = [actions.get(agent, 0) for agent in self.possible_agents]
        controllable = [agent in actions for agent in self.possible_agents]
        result = self._shield.filter(
            self._pending_beliefs,
            proposals,
            self._capabilities,
            controllable=controllable,
            mission_deviations=self.maneuver_mission_deviations(),
            baseline_predictions=self._pending_predictions,
        )
        self.last_shield_result = result
        approved = {agent: result.approved[self._agent_index[agent]] for agent in actions}
        return approved, result

    def _outcome_assessments(self, actions: dict, decision: ShieldResult):
        """The shield already predicted the raw proposal with the shared estimator; reuse it, never recompute."""
        return list(decision.proposal_evaluation.pairs)

    def _shield_step(self, decision: ShieldResult) -> ShieldStep:
        return ShieldStep(
            interventions=sum(decision.intervened),
            proposal_accepted=decision.accepted,
            approved_feasible=decision.feasible,
        )

    def _shield_record(self, decision: ShieldResult, index: int) -> tuple[bool, list[str]]:
        """§40 shield fields: the violation kinds involving `index` in the raw proposal, only when it was rejected."""
        if decision.accepted:
            return bool(decision.intervened[index]), []
        kinds = {v.kind.value for v in decision.proposal_evaluation.violations if index in v.agents}
        return bool(decision.intervened[index]), sorted(kinds)

    def _step_infos(self, decision: ShieldResult) -> dict:
        repair_mode = decision.repair.stats.mode if decision.repair is not None else None
        return {
            agent: {
                "shield": {
                    "proposed": decision.proposed[self._agent_index[agent]],
                    "approved": decision.approved[self._agent_index[agent]],
                    "intervened": decision.intervened[self._agent_index[agent]],
                    "accepted": decision.accepted,
                    "feasible": decision.feasible,
                    "repair_mode": repair_mode,
                }
            }
            for agent in self.agents
        }
