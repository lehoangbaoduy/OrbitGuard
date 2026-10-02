"""Acceptance Test D against the REAL `OrbitGuardEnv` (PRD §38, Phase 3 gate).

Mechanical half of the gate with real physics: train -> checkpoint -> restore
-> evaluate on the actual environment, through the same production code path
(`orbit_guard.rl.training`) with no code changes in between. CI-sized (tiny
batch, short episodes). It deliberately asserts nothing about reward values
or convergence -- the genuine 2-satellite learning check is a separate,
human-read run (see `orbit_guard.rl.training.main`).
"""

from __future__ import annotations

import math

from orbit_guard.rl.environment import OrbitGuardEnv
from orbit_guard.rl.evaluation import evaluate
from orbit_guard.rl.training import TrainingSettings, build_algo, restore, train

_SETTINGS = TrainingSettings(train_batch_size=64, minibatch_size=32, num_epochs=1, seed=0)


def _env_factory(config: dict) -> OrbitGuardEnv:
    return OrbitGuardEnv(
        num_agents=2, episode_horizon_s=300.0, control_dt_s=60.0, physics_dt_s=10.0, lookahead_s=120.0
    )


def test_train_checkpoint_restore_evaluate_on_the_real_environment(tmp_path):
    algo = build_algo(_env_factory, _SETTINGS, env_name="acceptance_d_real_env")
    try:
        report = train(algo, num_iterations=1, checkpoint_dir=tmp_path / "checkpoint")
    finally:
        algo.stop()
    assert report.env_steps_sampled >= _SETTINGS.train_batch_size

    restored = restore(_env_factory, _SETTINGS, report.checkpoint_path)
    try:
        evaluation = evaluate(restored, _env_factory, num_episodes=2, seed=0)
    finally:
        restored.stop()

    assert evaluation.episodes == 2
    assert evaluation.mean_episode_steps > 0
    assert evaluation.mean_delta_v_used_mps >= 0
    assert math.isfinite(evaluation.mean_return_per_agent)
