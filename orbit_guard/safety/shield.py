"""Joint-action safety shield (PRD §29, P4-M2).

All satellites propose an intent simultaneously (§26); the shield judges the
JOINT set before anything is applied:

  1. map every intent to a delta-v (the frozen §23 mapper) from the belief
     state, sized by each satellite's OWN `max_delta_v_per_action_mps`;
  2. capability validation -- per-action limit, remaining budget,
     `maneuver_enabled` -- violating burns are rejected, never silently capped;
  3. short-horizon consequence prediction -- each satellite's *belief* state
     plus its burn is propagated over the lookahead window; satellite dynamics
     are uncoupled, so each (satellite, intent) trajectory is propagated at
     most once per call and every joint assignment is then scored by pairwise
     lookups on those cached trajectories. The RAW proposal is predicted with
     `pair_assessment.predict_proposal_trajectories`/`assess_trajectories`,
     the same estimator the no-shield environment scores its rewards with
     (CORE-2), so `proposal_evaluation.pairs` is that estimator's output;
  4. safety checks per pair (collision, new conjunction, risk worsened,
     `d_safe` violated) and per agent (mission-deviation limit);
  5. accept the joint proposal if nothing is violated;
  6. otherwise repair with the bounded search of `repair_search` over only the
     agents involved in a violation (exact <= 4, beam width 50 above);
  7. return the approved joint action.

Belief-only: every input is an `AgentBelief` (PRD §14) -- ground truth never
enters. The margin is an injected `MarginFn`, so Fixed-Margin is a swap, not a
fork. The shield is deterministic and is a simulated safety filter, never a
formal guarantee.

Design decisions beyond the PRD text (see docs/OrbitGuard_progress.md, Q22-Q24):
  * Uninvolved agents KEEP their proposed action during repair instead of being
    reset to NO_MANEUVER: resetting could silently re-create the very
    conjunction their burn was already resolving.
  * When no feasible joint alternative exists the least risky capability-valid
    assignment is approved and `feasible=False` is reported -- NO_MANEUVER is
    never an automatic fallback.
  * Non-controllable (already collided) satellites are propagated as
    no-maneuver debris and are never repaired.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from orbit_guard.conjunction.pair_assessment import (
    DEFAULT_DENSITY_RADIUS_M,
    DEFAULT_DENSITY_REF_COUNT,
    PairAssessment,
    assess_pair,
    assess_trajectories,
    intent_delta_v,
    pair_density,
    predict_all,
    predict_proposal_trajectories,
    prediction_steps,
)
from orbit_guard.conjunction.risk import DEFAULT_EPSILON_M, DEFAULT_K, DEFAULT_Z0
from orbit_guard.conjunction.screening import generate_candidate_pairs
from orbit_guard.physics.dynamics import state_derivative_batch
from orbit_guard.physics.integrator import propagate_batch
from orbit_guard.physics.mission_deviation import (
    compute_mission_deviation,
    compute_weighted_mission_deviation,
)
from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.safety.adaptive_margin import MarginFn, make_adaptive_margin
from orbit_guard.safety.maneuver_mapper import ManeuverIntent
from orbit_guard.safety.repair_search import (
    DEFAULT_BEAM_WIDTH,
    DEFAULT_EXACT_MAX_AGENTS,
    DEFAULT_RISK_TOLERANCE,
    NO_MANEUVER_INTENT,
    AssignmentScore,
    RepairOutcome,
    search_repair,
)
from orbit_guard.sensing.belief import (
    AgentBelief,
    PredictedState,
    combined_relative_sigma,
)

_NUM_INTENTS = len(ManeuverIntent)
_LIMIT_SLACK = 1.0e-12
DEFAULT_COLLISION_DISTANCE_M = 10.0


class ViolationKind(Enum):
    CAPABILITY = "capability"
    MISSION_LIMIT = "mission_limit"
    COLLISION = "collision"
    NEW_CONJUNCTION = "new_conjunction"
    RISK_WORSENED = "risk_worsened"
    D_SAFE = "d_safe"


@dataclass(frozen=True)
class Violation:
    kind: ViolationKind
    agents: tuple[int, ...]


@dataclass(frozen=True)
class JointEvaluation:
    """The shield's predicted consequences of one full joint assignment."""

    pairs: tuple[PairAssessment, ...]
    violations: tuple[Violation, ...]
    total_risk: float
    total_delta_v_mps: float
    shortfall_m: float = 0.0  # summed depth inside the safety margin, over the evaluated pairs


@dataclass(frozen=True)
class ShieldResult:
    proposed: tuple[int, ...]
    approved: tuple[int, ...]
    intervened: tuple[bool, ...]
    accepted: bool  # the joint proposal had no violation
    feasible: bool  # the approved joint action has no violation
    proposal_evaluation: JointEvaluation  # the RAW proposal's predicted outcome (feeds training rewards)
    approved_evaluation: JointEvaluation
    repair: RepairOutcome | None


class _JointContext:
    """Per-call caches: delta-v, trajectories, pair assessments for (agent, intent) combinations."""

    def __init__(self, shield, beliefs, capabilities, mission_deviations, baseline_predictions):
        self._shield = shield
        self.beliefs = beliefs
        self.capabilities = capabilities
        self._mission_deviations = mission_deviations
        self.num_agents = len(beliefs)
        self.all_pairs = generate_candidate_pairs(self.num_agents)
        self._positions = np.array([b.state_vector[:3] for b in beliefs])
        self._delta_v: dict[tuple[int, int], np.ndarray] = {}
        self._trajectories = {(i, NO_MANEUVER_INTENT): p.trajectory for i, p in enumerate(baseline_predictions)}
        self._pairs: dict[tuple[int, int, int, int], PairAssessment] = {}
        self._pair_geometry: dict[tuple[int, int], tuple[float, float]] = {}

    def delta_v(self, agent: int, intent: int) -> np.ndarray:
        key = (agent, intent)
        if key not in self._delta_v:
            self._delta_v[key] = intent_delta_v(
                self.beliefs[agent], intent,
                self.capabilities[agent].max_delta_v_per_action_mps, self._shield.phase_fraction,
            )
        return self._delta_v[key]

    def seed_proposal(self, assignment: Sequence[int], baseline_predictions) -> None:
        """Cache the raw proposal's trajectories and pair assessments from the shared estimator (CORE-2)."""
        trajectories = predict_proposal_trajectories(
            self.beliefs, assignment, [c.max_delta_v_per_action_mps for c in self.capabilities],
            phase_fraction=self._shield.phase_fraction, lookahead_s=self._shield.lookahead_s,
            physics_dt_s=self._shield.physics_dt_s, baseline_predictions=baseline_predictions,
        )
        for agent, intent in enumerate(assignment):
            self._trajectories[(agent, intent)] = trajectories[agent]
        shield = self._shield
        assessments = assess_trajectories(
            self.beliefs, trajectories, shield.physics_dt_s, shield.margin_fn,
            density_radius_m=shield.density_radius_m, density_ref_count=shield.density_ref_count,
            risk_epsilon_m=shield.risk_epsilon_m, risk_z0=shield.risk_z0, risk_k=shield.risk_k,
        )
        for (i, j), assessment in zip(self.all_pairs, assessments, strict=True):
            self._pairs[(i, assignment[i], j, assignment[j])] = assessment

    def prefetch(self, keys: Sequence[tuple[int, int]]) -> None:
        """Propagate every not-yet-cached (agent, intent) trajectory together, in one vectorized call."""
        missing = sorted({key for key in keys if key not in self._trajectories})
        if not missing:
            return
        states = np.array([self.beliefs[agent].state_vector for agent, _ in missing])
        for row, (agent, intent) in enumerate(missing):
            states[row, 3:6] += self.delta_v(agent, intent)
        trajectories = propagate_batch(states, self._shield.physics_dt_s, self._shield.n_steps, state_derivative_batch)
        for key, trajectory in zip(missing, trajectories, strict=True):
            self._trajectories[key] = trajectory

    def trajectory(self, agent: int, intent: int) -> np.ndarray:
        if (agent, intent) not in self._trajectories:
            self.prefetch([(agent, intent)])
        return self._trajectories[(agent, intent)]

    def capability_ok(self, agent: int, intent: int) -> bool:
        if intent == NO_MANEUVER_INTENT:
            return True
        capability = self.capabilities[agent]
        magnitude = float(np.linalg.norm(self.delta_v(agent, intent)))
        return (
            capability.maneuver_enabled
            and magnitude <= capability.max_delta_v_per_action_mps + _LIMIT_SLACK
            and magnitude <= capability.remaining_delta_v_mps + _LIMIT_SLACK
        )

    def mission_ok(self, agent: int, intent: int) -> bool:
        if intent == NO_MANEUVER_INTENT:
            return True
        candidate_end = self.trajectory(agent, intent)[-1]
        baseline_end = self.trajectory(agent, NO_MANEUVER_INTENT)[-1]
        deviation = compute_weighted_mission_deviation(
            compute_mission_deviation(candidate_end, baseline_end), **self._shield.mission_weights
        )
        limit = self.capabilities[agent].mission_deviation_limit
        return self._mission_deviations[agent] + deviation <= limit + _LIMIT_SLACK

    def pair(self, i: int, intent_i: int, j: int, intent_j: int) -> PairAssessment:
        key = (i, intent_i, j, intent_j)
        if key not in self._pairs:
            if (i, j) not in self._pair_geometry:
                self._pair_geometry[(i, j)] = (
                    combined_relative_sigma(self.beliefs[i], self.beliefs[j]),
                    pair_density(i, j, self._positions, self._shield.density_radius_m, self._shield.density_ref_count),
                )
            sigma_rel_m, rho = self._pair_geometry[(i, j)]
            self._pairs[key] = assess_pair(
                i, j, self.trajectory(i, intent_i), self.trajectory(j, intent_j),
                self._shield.physics_dt_s, sigma_rel_m, rho, self._shield.margin_fn,
                risk_epsilon_m=self._shield.risk_epsilon_m, risk_z0=self._shield.risk_z0, risk_k=self._shield.risk_k,
            )
        return self._pairs[key]

    def evaluate(
        self,
        assignment: Sequence[int],
        pairs: Sequence[tuple[int, int]] | None = None,
        agents: Sequence[int] | None = None,
    ) -> JointEvaluation:
        pair_list = self.all_pairs if pairs is None else pairs
        agent_list = range(self.num_agents) if agents is None else agents
        assessments, violations = [], []
        for i, j in pair_list:
            assessment = self.pair(i, assignment[i], j, assignment[j])
            assessments.append(assessment)
            baseline = self.pair(i, NO_MANEUVER_INTENT, j, NO_MANEUVER_INTENT)
            moving = assignment[i] != NO_MANEUVER_INTENT or assignment[j] != NO_MANEUVER_INTENT
            violations.extend(self._shield.pair_violations(assessment, baseline, moving))
        for agent in agent_list:
            if not self.capability_ok(agent, assignment[agent]):
                violations.append(Violation(ViolationKind.CAPABILITY, (agent,)))
            elif not self.mission_ok(agent, assignment[agent]):
                violations.append(Violation(ViolationKind.MISSION_LIMIT, (agent,)))
        total_delta_v = sum(float(np.linalg.norm(self.delta_v(a, assignment[a]))) for a in range(self.num_agents))
        return JointEvaluation(
            pairs=tuple(assessments),
            violations=tuple(violations),
            total_risk=float(sum(a.risk_proxy for a in assessments)),
            total_delta_v_mps=total_delta_v,
            shortfall_m=float(sum(max(0.0, a.d_safe_m - a.d_min_m) for a in assessments)),
        )


class JointActionShield:
    """Deterministic joint-action safety filter (PRD §29)."""

    def __init__(
        self,
        lookahead_s: float = 1800.0,
        physics_dt_s: float = 10.0,
        *,
        margin_fn: MarginFn | None = None,
        phase_fraction: float = 0.5,
        collision_distance_m: float = DEFAULT_COLLISION_DISTANCE_M,
        exact_max_agents: int = DEFAULT_EXACT_MAX_AGENTS,
        beam_width: int = DEFAULT_BEAM_WIDTH,
        risk_tolerance: float = DEFAULT_RISK_TOLERANCE,
        density_radius_m: float = DEFAULT_DENSITY_RADIUS_M,
        density_ref_count: float = DEFAULT_DENSITY_REF_COUNT,
        risk_epsilon_m: float = DEFAULT_EPSILON_M,
        risk_z0: float = DEFAULT_Z0,
        risk_k: float = DEFAULT_K,
        mission_weights: Mapping[str, float] | None = None,
    ):
        self.lookahead_s = lookahead_s
        self.physics_dt_s = physics_dt_s
        self.n_steps = prediction_steps(lookahead_s, physics_dt_s)
        self.margin_fn = margin_fn if margin_fn is not None else make_adaptive_margin()
        self.phase_fraction = phase_fraction
        self.collision_distance_m = collision_distance_m
        self.exact_max_agents = exact_max_agents
        self.beam_width = beam_width
        self.risk_tolerance = risk_tolerance
        self.density_radius_m = density_radius_m
        self.density_ref_count = density_ref_count
        self.risk_epsilon_m = risk_epsilon_m
        self.risk_z0 = risk_z0
        self.risk_k = risk_k
        self.mission_weights = dict(mission_weights or {})

    def pair_violations(self, assessment: PairAssessment, baseline: PairAssessment, moving: bool) -> list[Violation]:
        agents = (assessment.i, assessment.j)
        violations = []
        if assessment.d_min_m < self.collision_distance_m:
            violations.append(Violation(ViolationKind.COLLISION, agents))
        inside_margin = assessment.d_min_m < assessment.d_safe_m
        if inside_margin and moving and baseline.d_min_m >= baseline.d_safe_m:
            violations.append(Violation(ViolationKind.NEW_CONJUNCTION, agents))
        if moving and assessment.risk_proxy > baseline.risk_proxy + self.risk_tolerance:
            violations.append(Violation(ViolationKind.RISK_WORSENED, agents))
        if inside_margin:
            violations.append(Violation(ViolationKind.D_SAFE, agents))
        return violations

    def filter(
        self,
        beliefs: Sequence[AgentBelief],
        proposals: Sequence[int],
        capabilities: Sequence[SatelliteCapability],
        *,
        controllable: Sequence[bool] | None = None,
        mission_deviations: Sequence[float] | None = None,
        baseline_predictions: Sequence[PredictedState] | None = None,
    ) -> ShieldResult:
        """Judge the joint `proposals` (one intent per satellite) and return the approved joint action."""
        n = len(beliefs)
        if not all(isinstance(b, AgentBelief) for b in beliefs):
            raise TypeError("JointActionShield works on AgentBelief only -- never ground truth (PRD §14)")
        if len(proposals) != n or len(capabilities) != n:
            raise ValueError(f"need one proposal and one capability per satellite ({n}); got {len(proposals)}, {len(capabilities)}")
        proposed = tuple(int(ManeuverIntent(p)) for p in proposals)
        active = tuple(bool(c) for c in controllable) if controllable is not None else (True,) * n
        deviations = tuple(float(d) for d in mission_deviations) if mission_deviations is not None else (0.0,) * n
        if len(active) != n or len(deviations) != n:
            raise ValueError("controllable and mission_deviations must have one entry per satellite")

        effective = tuple(p if is_active else NO_MANEUVER_INTENT for p, is_active in zip(proposed, active, strict=True))
        predictions = baseline_predictions if baseline_predictions is not None else predict_all(beliefs, self.lookahead_s, self.physics_dt_s)
        context = _JointContext(self, list(beliefs), list(capabilities), deviations, predictions)

        context.seed_proposal(effective, predictions)
        proposal_evaluation = context.evaluate(effective)
        affected = sorted({a for v in proposal_evaluation.violations for a in v.agents if active[a]})
        if not proposal_evaluation.violations or not affected:
            return self._result(proposed, effective, active, proposal_evaluation, proposal_evaluation, None)

        outcome = self._repair(context, effective, affected, proposal_evaluation)
        approved_evaluation = context.evaluate(outcome.assignment)
        return self._result(proposed, outcome.assignment, active, proposal_evaluation, approved_evaluation, outcome)

    def _repair(self, context: _JointContext, effective, affected, proposal_evaluation) -> RepairOutcome:
        affected_set = set(affected)
        candidates = {
            agent: tuple(intent for intent in range(_NUM_INTENTS) if context.capability_ok(agent, intent))
            for agent in affected
        }
        touching = [pair for pair in context.all_pairs if affected_set & set(pair)]
        context.prefetch([(agent, intent) for agent, intents in candidates.items() for intent in intents])

        def score(assignment) -> AssignmentScore:
            evaluation = context.evaluate(assignment, touching, affected)
            return AssignmentScore(
                not evaluation.violations, evaluation.total_risk, evaluation.total_delta_v_mps, evaluation.shortfall_m
            )

        priority = dict.fromkeys(affected, 0.0)
        risk_by_pair = {(a.i, a.j): a.risk_proxy for a in proposal_evaluation.pairs}
        for violation in proposal_evaluation.violations:
            if len(violation.agents) == 2:
                for agent in set(violation.agents) & affected_set:
                    priority[agent] = max(priority[agent], risk_by_pair[violation.agents])
        order = sorted(affected, key=lambda agent: (-priority[agent], agent))

        return search_repair(
            effective, affected, candidates, score,
            order=order, exact_max_agents=self.exact_max_agents,
            beam_width=self.beam_width, risk_tolerance=self.risk_tolerance,
        )

    @staticmethod
    def _result(proposed, approved, active, proposal_evaluation, approved_evaluation, repair) -> ShieldResult:
        return ShieldResult(
            proposed=proposed,
            approved=tuple(approved),
            intervened=tuple(is_active and a != p for is_active, a, p in zip(active, approved, proposed, strict=True)),
            accepted=not proposal_evaluation.violations,
            feasible=not approved_evaluation.violations,
            proposal_evaluation=proposal_evaluation,
            approved_evaluation=approved_evaluation,
            repair=repair,
        )
