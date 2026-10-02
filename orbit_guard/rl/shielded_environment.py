"""Shield-active OrbitGuardEnv (PRD §28 "training-time shield behavior", P4-M6).

A SEPARATE class, not a flag on `OrbitGuardEnv`: the base environment never
imports `safety.shield`, so a no-shield training run structurally cannot call
it (tests/rl/test_shielded_environment.py pins that). "Adaptive MAPPO without
shield" is therefore a different training run built from `OrbitGuardEnv`, never
an evaluation-time toggle on shielded weights.

Semantics (PRD §28):
  * the realized transition applies the shield-APPROVED joint action;
  * the safety and coordination reward terms are scored against the shield's
    predicted outcome of each agent's RAW proposal, so an agent is still
    penalized for proposing something unsafe even when the shield prevented the
    bad outcome;
  * the delta-v term is charged on the raw request (Q21), so the shield's
    own corrective burns are free to the agent -- it is not punished for the
    shield's work, and not rewarded for relying on it either;
  * the mission term stays on the realized trajectory.
"""

from __future__ import annotations

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.episode_metrics import ShieldStep
from orbit_guard.safety.shield import JointActionShield, ShieldResult


class ShieldedOrbitGuardEnv(OrbitGuardEnv):
    """`OrbitGuardEnv` whose actions pass through the joint-action safety shield."""

    def __init__(self, *args, shield_exact_max_agents: int = 4, shield_beam_width: int = 50, **kwargs):
        super().__init__(*args, **kwargs)
        self._shield = JointActionShield(
            lookahead_s=self._lookahead_s,
            physics_dt_s=self._physics_dt_s,
            margin_fn=self._margin_fn,
            base_delta_v_mps=self._capability_template.max_delta_v_per_action_mps,
            collision_distance_m=2.0 * self._hard_body_radius_m,
            exact_max_agents=shield_exact_max_agents,
            beam_width=shield_beam_width,
        )
        self.last_shield_result: ShieldResult | None = None

    def _approve_actions(self, actions: dict):
        proposals = [actions.get(agent, 0) for agent in self.possible_agents]
        controllable = [agent in actions for agent in self.possible_agents]
        result = self._shield.filter(
            self._pending_beliefs,
            proposals,
            self._capabilities,
            controllable=controllable,
            mission_deviations=self._pending_mission_deviations,
            baseline_predictions=self._pending_predictions,
        )
        self.last_shield_result = result
        approved = {agent: result.approved[self._agent_index[agent]] for agent in actions}
        return approved, result

    def _outcome_assessments(self, post_assessments, decision: ShieldResult):
        return list(decision.proposal_evaluation.pairs)

    def _shield_step(self, decision: ShieldResult) -> ShieldStep:
        return ShieldStep(
            interventions=sum(decision.intervened),
            proposal_accepted=decision.accepted,
            approved_feasible=decision.feasible,
        )

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
