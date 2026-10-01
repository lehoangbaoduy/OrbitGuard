"""Acceptance Test D (Phase 3 GATE, PRD §44): training-pipeline smoke test.

This is deliberately CHEAP and CI-sized: one training iteration on the
trivial `StubOrbitGuardEnv` (no physics, no real reward), just to prove the
mechanical wiring -- env registration, `MultiAgentEnv` flattening,
`PPOConfig.multi_agent(...)` with a single shared-weights policy
(parameter sharing), one `.train()` call, a checkpoint round-trip --
actually runs end-to-end against this project's pinned Ray/RLlib version.

**Parameter sharing alone is NOT CTDE.** A shared policy with each agent
seeing only its own local observation is IPPO (independent, decentralized
critic) -- PRD §28 requires MAPPO/CTDE: the same parameter-shared actor,
PLUS a CENTRALIZED CRITIC that sees global state during training (never at
decentralized execution). This smoke test only proves the mechanical loop
runs; it does not build or exercise a centralized critic. Part C's real
`rl/training.py` must still implement one, using `StubOrbitGuardEnv.state()`
(see tests/rl/stub_env.py) as the frozen global-state interface -- verify
the current RLlib new-API-stack pattern for a centralized critic against
live docs, the same discipline already applied to the Dict-observation
encoder issue below. Do not treat this smoke test as satisfying the CTDE
requirement by itself.

**This test is NOT the real Phase 3 convergence check.** The actual
2-satellite sanity run against the real `OrbitGuardEnv` (does the shared
policy learn ANYTHING sensible on a real physics scenario) is milestone
P3-M5 -- a separate, human-reviewed, non-CI-gated task, run and read by a
person, not asserted on in CI. Do not extend this test with reward-value or
convergence assertions; a trivial random-noise env has no reward signal
worth asserting on.

Uses `tests/rl/rllib_integration.FlattenedParallelPettingZooEnv`, a helper
pre-verified against this project's installed Ray 2.58 on 2026-09-30 (see
that module's docstring for why a plain `ParallelPettingZooEnv` alone fails
on the current RLlib API stack). This file lives under `tests/`, so this
acceptance test may import it directly -- but Part C's PRODUCTION
`orbit_guard/rl/training.py` must not import test infrastructure. Part C
should port/extend this same verified flatten pattern into a production
module (e.g. `orbit_guard/rl/rllib_env.py`, an explicitly allowed third
file beyond `reward.py`/`training.py` -- see the Phase 3 Antigravity brief)
rather than reach into `tests.rl.rllib_integration` from production code.
"""

from __future__ import annotations

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from tests.rl.rllib_integration import FlattenedParallelPettingZooEnv
from tests.rl.stub_env import StubOrbitGuardEnv

_ENV_NAME = "acceptance_d_stub_orbit_guard"


def _env_creator(config):
    return FlattenedParallelPettingZooEnv(StubOrbitGuardEnv(num_agents=2))


def _build_smoke_config() -> PPOConfig:
    register_env(_ENV_NAME, _env_creator)
    return (
        PPOConfig()
        .environment(_ENV_NAME)
        .multi_agent(
            policies={"shared_policy"},
            policy_mapping_fn=lambda agent_id, episode, **kw: "shared_policy",
        )
        .env_runners(num_env_runners=0)
        .training(train_batch_size=64, minibatch_size=32, num_epochs=1)
    )


def test_one_training_iteration_runs_without_error_on_the_stub_env():
    algo = _build_smoke_config().build_algo()
    try:
        result = algo.train()
        assert "env_runners" in result or "num_env_steps_sampled_lifetime" in result
    finally:
        algo.stop()


def test_checkpoint_save_and_restore_round_trips_without_error(tmp_path):
    algo = _build_smoke_config().build_algo()
    try:
        algo.train()
        checkpoint_path = algo.save_to_path(str(tmp_path / "checkpoint"))
    finally:
        algo.stop()

    restored_algo = _build_smoke_config().build_algo()
    try:
        restored_algo.restore_from_path(checkpoint_path)
    finally:
        restored_algo.stop()


def test_shared_actor_weights_are_parameter_shared_across_agents():
    # Necessary but NOT sufficient for CTDE (see module docstring) --
    # confirms parameter sharing only, not the centralized critic.
    config = _build_smoke_config()
    assert set(config.policies) == {"shared_policy"}
    assert config.policy_mapping_fn("sat_0", None) == "shared_policy"
    assert config.policy_mapping_fn("sat_1", None) == "shared_policy"
