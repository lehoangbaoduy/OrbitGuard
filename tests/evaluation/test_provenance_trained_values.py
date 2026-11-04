"""Checkpoint provenance covers the margin and reward weights a model was trained with, not just config_digest.

The training CLI's --margin-alpha/--margin-beta/--margin-gamma-margin and --w1..--w4/--collision-penalty flags
change the training environment without changing config_digest, but the evaluation environment is rebuilt from
config + overlay only. A checkpoint whose run_config.json records a margin or reward weights other than the ones
the evaluation environment for its model will build is therefore refused (wave-4 review, TUNED_FLAGS finding).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest
from orbit_guard.evaluation.experiment import (
    CheckpointProvenanceError,
    check_checkpoint_provenance,
    check_registry_provenance,
)
from orbit_guard.evaluation.replay_recording import EpisodePlan, check_plan_provenance

# Literal values of configs/default.yaml, written out so the test does not derive them with the code under test.
_DEFAULT_ADAPTIVE = {
    "kind": "adaptive", "d_base_m": 1000.0, "d_min_m": 200.0, "d_max_m": 5000.0,
    "alpha": 1.0, "beta": 0.5, "gamma": 1.0,
}
_DEFAULT_FIXED = {"kind": "fixed", "d_safe_m": 1142.0}
_DEFAULT_WEIGHTS = {"w1": 1.0, "w2": 0.3, "w3": 5.0, "w4": 0.5, "collision_penalty": -100.0}


def _checkpoint(tmp_path: Path, *, margin: dict | None, weights: dict | None, overlay: Path | None = None,
                name: str = "run", shield: bool = True) -> Path:
    run = tmp_path / name
    run.mkdir()
    (run / "actor_weights.npz").write_bytes(b"weights")
    record: dict = {"config_digest": config_digest(DEFAULT_CONFIG_PATH, overlay),
                    "overlay_path": None if overlay is None else str(overlay), "shield": shield, "num_agents": 15,
                    "seed": 0, "training": {"critic_input": "agent_global"}}
    if margin is not None:
        record["margin"] = margin
    if weights is not None:
        record["reward_weights"] = weights
    (run / "run_config.json").write_text(json.dumps(record))
    return run / "actor_weights.npz"


def _shielded(model: str) -> bool:
    return model != "adaptive_mappo_no_shield"


def _check(weights: Path, model: str | None, overlay: Path | None = None) -> None:
    check_checkpoint_provenance(weights, config_path=DEFAULT_CONFIG_PATH, overlay_path=overlay, model=model)


@pytest.mark.parametrize(("model", "margin"), [
    ("adaptive_mappo", _DEFAULT_ADAPTIVE),
    ("adaptive_mappo_no_shield", _DEFAULT_ADAPTIVE),
    ("fixed_margin_mappo", _DEFAULT_FIXED),
])
def test_a_sidecar_matching_the_evaluation_environment_is_accepted(tmp_path, model, margin):
    _check(_checkpoint(tmp_path, margin=margin, weights=_DEFAULT_WEIGHTS, shield=_shielded(model)), model)


@pytest.mark.parametrize(("model", "margin"), [
    ("adaptive_mappo", {**_DEFAULT_ADAPTIVE, "alpha": 0.5}),
    ("adaptive_mappo", {**_DEFAULT_ADAPTIVE, "gamma": 2.0}),
    ("adaptive_mappo_no_shield", {**_DEFAULT_ADAPTIVE, "beta": 0.25}),
    ("fixed_margin_mappo", {"kind": "fixed", "d_safe_m": 1283.0}),
    ("fixed_margin_mappo", _DEFAULT_ADAPTIVE),  # an adaptive checkpoint registered as the fixed model
    ("adaptive_mappo", _DEFAULT_FIXED),  # and the other way round
])
def test_a_sidecar_whose_margin_differs_is_refused_with_overlay_advice(tmp_path, model, margin):
    weights = _checkpoint(tmp_path, margin=margin, weights=_DEFAULT_WEIGHTS, shield=_shielded(model))

    with pytest.raises(CheckpointProvenanceError, match="margin") as raised:
        _check(weights, model)

    assert "overlay" in str(raised.value)


@pytest.mark.parametrize("model", ["adaptive_mappo", "fixed_margin_mappo"])
@pytest.mark.parametrize("change", [{"w4": 1.5}, {"w3": 2.5}, {"w1": 2.0}, {"collision_penalty": -50.0}])
def test_a_sidecar_whose_reward_weights_differ_is_refused(tmp_path, model, change):
    margin = _DEFAULT_FIXED if model == "fixed_margin_mappo" else _DEFAULT_ADAPTIVE
    weights = _checkpoint(tmp_path, margin=margin, weights={**_DEFAULT_WEIGHTS, **change})

    with pytest.raises(CheckpointProvenanceError, match="reward weights") as raised:
        _check(weights, model)

    assert "overlay" in str(raised.value)


@pytest.mark.parametrize("missing", ["margin", "reward_weights"])
def test_a_sidecar_without_the_recorded_margin_or_weights_is_refused(tmp_path, missing):
    values = {"margin": _DEFAULT_ADAPTIVE, "weights": _DEFAULT_WEIGHTS}
    values["weights" if missing == "reward_weights" else "margin"] = None
    weights = _checkpoint(tmp_path, **values)

    with pytest.raises(CheckpointProvenanceError, match=missing):
        _check(weights, "adaptive_mappo")


def test_without_a_model_the_recorded_margin_kind_is_checked_against_the_config(tmp_path):
    _check(_checkpoint(tmp_path, margin=_DEFAULT_FIXED, weights=_DEFAULT_WEIGHTS, name="fixed"), None)
    _check(_checkpoint(tmp_path, margin=_DEFAULT_ADAPTIVE, weights=_DEFAULT_WEIGHTS, name="adaptive"), None)
    tuned = _checkpoint(tmp_path, margin={**_DEFAULT_ADAPTIVE, "alpha": 0.5}, weights=_DEFAULT_WEIGHTS, name="tuned")

    with pytest.raises(CheckpointProvenanceError, match="margin"):
        _check(tuned, None)


def test_tuned_values_carried_by_the_overlay_are_accepted(tmp_path):
    overlay = tmp_path / "tuned.yaml"
    overlay.write_text(yaml.safe_dump({"reward": {"w4": 1.5}, "adaptive_margin": {"alpha": 0.5}}))
    weights = _checkpoint(
        tmp_path, margin={**_DEFAULT_ADAPTIVE, "alpha": 0.5}, weights={**_DEFAULT_WEIGHTS, "w4": 1.5}, overlay=overlay,
    )

    _check(weights, "adaptive_mappo", overlay)
    with pytest.raises(CheckpointProvenanceError):  # the same overlay, but the sidecar recorded the defaults
        _check(_checkpoint(tmp_path, margin=_DEFAULT_ADAPTIVE, weights=_DEFAULT_WEIGHTS, overlay=overlay,
                           name="stale"), "adaptive_mappo", overlay)


def test_the_registry_check_compares_each_checkpoint_with_its_own_model(tmp_path):
    fixed = _checkpoint(tmp_path, margin=_DEFAULT_FIXED, weights=_DEFAULT_WEIGHTS, name="fixed")
    adaptive = _checkpoint(tmp_path, margin=_DEFAULT_ADAPTIVE, weights=_DEFAULT_WEIGHTS, name="adaptive")

    check_registry_provenance(
        {"fixed_margin_mappo": {0: str(fixed)}, "adaptive_mappo": {0: str(adaptive)}}, config_path=DEFAULT_CONFIG_PATH,
    )
    with pytest.raises(CheckpointProvenanceError, match="margin"):
        check_registry_provenance({"fixed_margin_mappo": {0: str(adaptive)}}, config_path=DEFAULT_CONFIG_PATH)


def test_the_replay_recorder_refuses_a_checkpoint_trained_with_another_margin(tmp_path):
    good = _checkpoint(tmp_path, margin=_DEFAULT_ADAPTIVE, weights=_DEFAULT_WEIGHTS, name="good")
    bad = _checkpoint(tmp_path, margin={**_DEFAULT_ADAPTIVE, "alpha": 0.5}, weights=_DEFAULT_WEIGHTS, name="bad",
                      shield=False)

    check_plan_provenance([EpisodePlan("normal", 0, "trained", True, "adaptive_mappo", good)])
    with pytest.raises(CheckpointProvenanceError, match="margin"):
        check_plan_provenance([EpisodePlan("normal", 0, "trained", False, "adaptive_mappo_no_shield", bad)])


# --- end to end: the sidecar the training CLI really writes ----------------------------------------------------


def _train(tmp_path: Path, name: str, *flags: str) -> Path:
    training = pytest.importorskip("orbit_guard.rl.training")
    from orbit_guard.rl.evaluation import EvaluationReport

    report = EvaluationReport(
        episodes=1, mean_return_per_agent=0.0, collision_episodes=0, maneuver_rate=0.0,
        threat_maneuver_rate=0.0, calm_maneuver_rate=0.0, threat_step_fraction=0.0,
        mean_delta_v_used_mps=0.0, mean_budget_fraction_used=0.0, unsafe_episodes=0, mean_episode_steps=1.0,
    )

    class _Algo:
        def stop(self) -> None:
            return None

    def fake_train(algo, num_iterations, checkpoint_dir=None, on_iteration=None):
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        (Path(checkpoint_dir) / training.ACTOR_WEIGHTS_FILENAME).write_bytes(b"")
        return training.TrainingReport(iterations=num_iterations, episode_return_means=(), env_steps_sampled=0,
                                       checkpoint_path=str(checkpoint_dir))

    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(training, "build_algo", lambda env_factory, settings: _Algo())
        patch.setattr(training, "train", fake_train)
        patch.setattr("orbit_guard.rl.evaluation.evaluate_policy", lambda *a, **k: report)
        patch.setattr("orbit_guard.rl.evaluation.evaluate", lambda *a, **k: report)
        patch.setattr("orbit_guard.rl.evaluation.gate_verdict", lambda *a, **k: "PASS")
        checkpoint = tmp_path / name
        training.main(["--agents", "3", "--iterations", "1", "--eval-episodes", "1", "--num-env-runners", "0",
                       "--checkpoint-dir", str(checkpoint), *flags])
    finally:
        patch.undo()
    return checkpoint / training.ACTOR_WEIGHTS_FILENAME


@pytest.mark.parametrize(("model", "flags"), [
    ("adaptive_mappo", ("--shield", "--margin", "adaptive")),
    ("adaptive_mappo_no_shield", ("--no-shield", "--margin", "adaptive")),
    ("fixed_margin_mappo", ("--shield", "--margin", "fixed")),
])
def test_a_checkpoint_trained_from_config_alone_passes(tmp_path, model, flags):
    _check(_train(tmp_path, "run", *flags), model)


@pytest.mark.parametrize(("model", "flags", "match"), [
    ("adaptive_mappo", ("--margin", "adaptive", "--margin-alpha", "0.5"), "margin"),
    ("adaptive_mappo_no_shield", ("--no-shield", "--margin-gamma-margin", "2.0"), "margin"),
    ("adaptive_mappo", ("--w4", "1.5"), "reward weights"),
    ("fixed_margin_mappo", ("--margin", "fixed", "--w3", "2.5"), "reward weights"),
])
def test_a_checkpoint_trained_with_cli_overrides_is_refused(tmp_path, model, flags, match):
    weights = _train(tmp_path, "run", *flags)

    with pytest.raises(CheckpointProvenanceError, match=match):
        _check(weights, model)


def test_the_shield_and_n_the_training_cli_records_are_compared(tmp_path):
    no_shield = _train(tmp_path, "run", "--no-shield", "--margin", "adaptive")  # --agents 3

    check_checkpoint_provenance(no_shield, model="adaptive_mappo_no_shield", n_agents=3)
    with pytest.raises(CheckpointProvenanceError, match="shield"):
        check_checkpoint_provenance(no_shield, model="adaptive_mappo", n_agents=3)
    with pytest.raises(CheckpointProvenanceError, match="3 agents"):
        check_checkpoint_provenance(no_shield, model="adaptive_mappo_no_shield", n_agents=15)
