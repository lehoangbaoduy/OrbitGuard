"""The training CLI writes a provenance sidecar run_config.json next to the checkpoint (CORE-1)."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest
from orbit_guard.rl import training
from orbit_guard.rl.evaluation import EvaluationReport
from orbit_guard.rl.training import (
    ACTOR_WEIGHTS_FILENAME,
    RUN_CONFIG_FILENAME,
    TrainingReport,
)


def _report() -> EvaluationReport:
    return EvaluationReport(
        episodes=1, mean_return_per_agent=0.0, collision_episodes=0, maneuver_rate=0.0,
        threat_maneuver_rate=0.0, calm_maneuver_rate=0.0, threat_step_fraction=0.0,
        mean_delta_v_used_mps=0.0, mean_budget_fraction_used=0.0, unsafe_episodes=0, mean_episode_steps=1.0,
    )


class _FakeAlgo:
    def stop(self) -> None:
        return None


def test_training_cli_writes_a_provenance_sidecar_beside_the_actor_weights(tmp_path, monkeypatch):
    checkpoint = tmp_path / "run"
    overlay = tmp_path / "tuned.yaml"
    overlay.write_text(yaml.safe_dump({"reward": {"w4": 1.5}, "adaptive_margin": {"alpha": 2.0}}))

    def fake_train(algo, num_iterations, checkpoint_dir=None, on_iteration=None):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (Path(checkpoint_dir) / ACTOR_WEIGHTS_FILENAME).write_bytes(b"")
        return TrainingReport(iterations=num_iterations, episode_return_means=(), env_steps_sampled=0,
                              checkpoint_path=str(checkpoint_dir))

    monkeypatch.setattr(training, "build_algo", lambda env_factory, settings: _FakeAlgo())
    monkeypatch.setattr(training, "train", fake_train)
    monkeypatch.setattr("orbit_guard.rl.evaluation.evaluate_policy", lambda *a, **k: _report())
    monkeypatch.setattr("orbit_guard.rl.evaluation.evaluate", lambda *a, **k: _report())
    monkeypatch.setattr("orbit_guard.rl.evaluation.gate_verdict", lambda *a, **k: "PASS")

    training.main([
        "--agents", "3", "--iterations", "1", "--eval-episodes", "1", "--num-env-runners", "0",
        "--checkpoint-dir", str(checkpoint), "--overlay", str(overlay), "--w4", "1.2",
    ])

    sidecar = json.loads((checkpoint / RUN_CONFIG_FILENAME).read_text())
    assert RUN_CONFIG_FILENAME == "run_config.json"
    assert (checkpoint / ACTOR_WEIGHTS_FILENAME).exists()
    assert sidecar["config_path"] == str(DEFAULT_CONFIG_PATH)
    assert sidecar["overlay_path"] == str(overlay)
    assert sidecar["config_digest"] == config_digest(DEFAULT_CONFIG_PATH, overlay)
    assert sidecar["num_agents"] == 3
    assert sidecar["seed"] == 0
    assert sidecar["shield"] is True
    assert sidecar["reward_weights"]["w4"] == 1.2
    assert sidecar["reward_weights"]["w3"] == 5.0
    assert sidecar["margin"]["kind"] == "adaptive"
    assert sidecar["margin"]["alpha"] == 2.0
    assert sidecar["training"]["learning_rate"] == 5.0e-4
    assert "git_commit" in sidecar
