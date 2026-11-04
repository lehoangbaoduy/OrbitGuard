"""The training CLI writes a provenance sidecar run_config.json next to the checkpoint (CORE-1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest
from orbit_guard.rl import training
from orbit_guard.rl.evaluation import EvaluationReport
from orbit_guard.rl.training import (
    ACTOR_WEIGHTS_FILENAME,
    RUN_CONFIG_FILENAME,
    TrainingReport,
    TrainingSettings,
    restore,
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


def test_the_training_cli_passes_the_critic_input_to_the_settings_and_the_sidecar(tmp_path, monkeypatch):
    checkpoint = tmp_path / "run"
    built_with = {}

    def fake_build_algo(env_factory, settings):
        built_with["settings"] = settings
        return _FakeAlgo()

    def fake_train(algo, num_iterations, checkpoint_dir=None, on_iteration=None):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        return TrainingReport(iterations=num_iterations, episode_return_means=(), env_steps_sampled=0,
                              checkpoint_path=str(checkpoint_dir))

    monkeypatch.setattr(training, "build_algo", fake_build_algo)
    monkeypatch.setattr(training, "train", fake_train)
    monkeypatch.setattr("orbit_guard.rl.evaluation.evaluate_policy", lambda *a, **k: _report())
    monkeypatch.setattr("orbit_guard.rl.evaluation.evaluate", lambda *a, **k: _report())
    monkeypatch.setattr("orbit_guard.rl.evaluation.gate_verdict", lambda *a, **k: "PASS")

    training.main([
        "--agents", "2", "--iterations", "1", "--eval-episodes", "1", "--num-env-runners", "0",
        "--checkpoint-dir", str(checkpoint), "--critic-input", "agent_global",
    ])

    sidecar = json.loads((checkpoint / RUN_CONFIG_FILENAME).read_text())
    assert built_with["settings"].critic_input == "agent_global"
    assert sidecar["training"]["critic_input"] == "agent_global"


class _RestorableAlgo:
    def __init__(self) -> None:
        self.restored_from = None

    def restore_from_path(self, path: str) -> None:
        self.restored_from = path


def _capture_build_algo(monkeypatch) -> dict:
    built_with = {}

    def fake_build_algo(env_factory, settings):
        built_with["settings"] = settings
        built_with["algo"] = _RestorableAlgo()
        return built_with["algo"]

    monkeypatch.setattr(training, "build_algo", fake_build_algo)
    return built_with


def _write_sidecar(checkpoint: Path, payload: dict) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / RUN_CONFIG_FILENAME).write_text(json.dumps(payload))


def test_restore_rebuilds_with_the_critic_input_recorded_in_the_sidecar(tmp_path, monkeypatch, capsys):
    checkpoint = tmp_path / "run"
    _write_sidecar(checkpoint, {"training": {"critic_input": "agent_global"}})
    requested = TrainingSettings(critic_input="global")
    built_with = _capture_build_algo(monkeypatch)

    restore(lambda config: None, requested, checkpoint)

    assert built_with["settings"].critic_input == "agent_global"
    assert requested.critic_input == "global"  # the caller's settings object is not mutated
    assert built_with["algo"].restored_from == str(checkpoint)
    assert "agent_global" in capsys.readouterr().err


def test_restore_keeps_the_settings_when_the_checkpoint_has_no_sidecar(tmp_path, monkeypatch, capsys):
    requested = TrainingSettings(critic_input="agent_global")
    built_with = _capture_build_algo(monkeypatch)

    restore(lambda config: None, requested, tmp_path)

    assert built_with["settings"] is requested
    assert capsys.readouterr().err == ""


def test_restore_keeps_the_settings_when_the_sidecar_predates_the_critic_input_record(tmp_path, monkeypatch):
    _write_sidecar(tmp_path, {"training": {"learning_rate": 5.0e-4}})
    requested = TrainingSettings(critic_input="agent_global")
    built_with = _capture_build_algo(monkeypatch)

    restore(lambda config: None, requested, tmp_path)

    assert built_with["settings"] is requested


def test_restore_refuses_an_unreadable_sidecar_instead_of_guessing_the_critic_input(tmp_path, monkeypatch):
    (tmp_path / RUN_CONFIG_FILENAME).write_text("{not json")
    _capture_build_algo(monkeypatch)

    with pytest.raises(ValueError, match="unreadable"):
        restore(lambda config: None, TrainingSettings(), tmp_path)


def test_training_refuses_to_write_a_sidecar_when_the_overlay_changed_mid_run(tmp_path, monkeypatch):
    checkpoint = tmp_path / "run"
    overlay = tmp_path / "tuned.yaml"
    overlay.write_text(yaml.safe_dump({"reward": {"w4": 1.5}}))

    def fake_train(algo, num_iterations, checkpoint_dir=None, on_iteration=None):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (Path(checkpoint_dir) / ACTOR_WEIGHTS_FILENAME).write_bytes(b"")
        overlay.write_text(yaml.safe_dump({"reward": {"w4": 9.9}}))  # tuned under the run's feet
        return TrainingReport(iterations=num_iterations, episode_return_means=(), env_steps_sampled=0,
                              checkpoint_path=str(checkpoint_dir))

    monkeypatch.setattr(training, "build_algo", lambda env_factory, settings: _FakeAlgo())
    monkeypatch.setattr(training, "train", fake_train)
    monkeypatch.setattr("orbit_guard.rl.evaluation.evaluate_policy", lambda *a, **k: _report())
    monkeypatch.setattr("orbit_guard.rl.evaluation.evaluate", lambda *a, **k: _report())
    monkeypatch.setattr("orbit_guard.rl.evaluation.gate_verdict", lambda *a, **k: "PASS")

    with pytest.raises(RuntimeError, match="changed during training"):
        training.main([
            "--agents", "2", "--iterations", "1", "--eval-episodes", "1", "--num-env-runners", "0",
            "--checkpoint-dir", str(checkpoint), "--overlay", str(overlay),
        ])

    assert (checkpoint / ACTOR_WEIGHTS_FILENAME).exists()
    assert not (checkpoint / RUN_CONFIG_FILENAME).exists()
