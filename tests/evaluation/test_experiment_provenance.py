"""Overlay plumbing and checkpoint provenance in the experiment CLI (PRD §33, §40; CORE-1 config_digest).

A model trained under a tuned overlay must be evaluated under that same overlay: the run_config.json
sidecar that rl/training.py writes next to actor_weights.npz records the config_digest it trained with,
and the experiment refuses any checkpoint whose digest differs from the one it is about to evaluate with.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbit_guard.config import DEFAULT_CONFIG_PATH
from orbit_guard.evaluation import experiment
from orbit_guard.evaluation.experiment import (
    CheckpointProvenanceError,
    build_registry,
    build_tasks,
    check_checkpoint_provenance,
)
from tests.evaluation.sidecar_fixtures import sidecar_record


def _overlay(tmp_path: Path, w3: float = 1.5) -> Path:
    path = tmp_path / f"tuned_{w3}.yaml"
    path.write_text(f"reward:\n  w3: {w3}\n")
    return path


def _run_dir(
    tmp_path: Path, name: str, overlay: Path | None, *, sidecar: bool = True, margin_kind: str = "adaptive",
    shield: bool = True,
) -> Path:
    run = tmp_path / "runs" / name
    run.mkdir(parents=True)
    (run / "actor_weights.npz").write_bytes(b"weights")
    if sidecar:
        record = {
            "config_path": str(DEFAULT_CONFIG_PATH),
            "overlay_path": None if overlay is None else str(overlay),
            **sidecar_record(overlay, margin_kind=margin_kind, shield=shield),
        }
        (run / "run_config.json").write_text(json.dumps(record))
    return run


_VARIANTS = {"adaptive": ("adaptive", True), "fixed": ("fixed", True), "noshield": ("adaptive", False)}


def _all_runs(tmp_path: Path, overlay: Path | None) -> Path:
    """The nine finished queue runs (three variants x seeds 0-2), all trained under `overlay`; returns the runs dir."""
    for variant, (margin_kind, shield) in _VARIANTS.items():
        for seed in (0, 1, 2):
            _run_dir(tmp_path, f"n15_{variant}_s{seed}", overlay, margin_kind=margin_kind, shield=shield)
    return tmp_path / "runs"


def _registry_file(tmp_path: Path, run: Path) -> Path:
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"adaptive_mappo": {"0": str(run / "actor_weights.npz")}}))
    return path


def test_the_sidecar_filename_matches_the_one_training_writes():
    training = pytest.importorskip("orbit_guard.rl.training")

    assert experiment.RUN_CONFIG_FILENAME == training.RUN_CONFIG_FILENAME


def test_tasks_carry_the_overlay_they_will_be_evaluated_with(tmp_path):
    overlay = _overlay(tmp_path)

    tasks = build_tasks(("rule_based",), {"normal": (1, 2)}, {}, n_agents=3, experiment_id="e", overlay_path=overlay)

    assert {task.overlay_path for task in tasks} == {str(overlay)}


def test_tasks_default_to_no_overlay():
    tasks = build_tasks(("no_maneuver",), {"normal": (1,)}, {}, n_agents=3, experiment_id="e")

    assert tasks[0].overlay_path is None


def test_a_checkpoint_trained_with_an_overlay_is_refused_without_it(tmp_path):
    run = _run_dir(tmp_path, "n15_adaptive_s0", _overlay(tmp_path))

    with pytest.raises(CheckpointProvenanceError, match=r"tuned_1\.5\.yaml"):
        check_checkpoint_provenance(run / "actor_weights.npz", config_path=DEFAULT_CONFIG_PATH, overlay_path=None)


def test_a_checkpoint_trained_without_an_overlay_is_refused_with_one(tmp_path):
    run = _run_dir(tmp_path, "n15_adaptive_s0", None)

    with pytest.raises(CheckpointProvenanceError, match="config_digest"):
        check_checkpoint_provenance(
            run / "actor_weights.npz", config_path=DEFAULT_CONFIG_PATH, overlay_path=_overlay(tmp_path)
        )


def test_the_matching_overlay_is_accepted_even_from_another_path(tmp_path):
    overlay = _overlay(tmp_path)
    run = _run_dir(tmp_path, "n15_adaptive_s0", overlay)
    copy = tmp_path / "copied_tuned.yaml"
    copy.write_bytes(overlay.read_bytes())

    check_checkpoint_provenance(run / "actor_weights.npz", config_path=DEFAULT_CONFIG_PATH, overlay_path=copy)


def test_a_checkpoint_without_a_sidecar_is_accepted(tmp_path):
    run = _run_dir(tmp_path, "n15_adaptive_s0", None, sidecar=False)

    check_checkpoint_provenance(run / "actor_weights.npz", config_path=DEFAULT_CONFIG_PATH, overlay_path=None)


def test_a_sidecar_without_a_digest_is_refused(tmp_path):
    run = _run_dir(tmp_path, "n15_adaptive_s0", None)
    (run / "run_config.json").write_text(json.dumps({"overlay_path": None}))

    with pytest.raises(CheckpointProvenanceError, match="config_digest"):
        check_checkpoint_provenance(run / "actor_weights.npz", config_path=DEFAULT_CONFIG_PATH, overlay_path=None)


def test_build_tasks_refuses_a_registry_trained_under_another_config(tmp_path):
    run = _run_dir(tmp_path, "n15_adaptive_s0", _overlay(tmp_path))
    registry = {"adaptive_mappo": {0: str(run / "actor_weights.npz")}}

    with pytest.raises(CheckpointProvenanceError):
        build_tasks(("adaptive_mappo",), {"normal": (1,)}, registry, n_agents=3, experiment_id="e")


def test_the_registry_command_refuses_mismatched_checkpoints(tmp_path):
    runs = _all_runs(tmp_path, _overlay(tmp_path))

    with pytest.raises(CheckpointProvenanceError):
        experiment.main(["registry", "--runs-dir", str(runs), "--out", str(tmp_path / "reg.json")])


def test_the_registry_command_accepts_the_training_overlay(tmp_path):
    overlay = _overlay(tmp_path)
    runs = _all_runs(tmp_path, overlay)
    out = tmp_path / "reg.json"

    experiment.main(["registry", "--runs-dir", str(runs), "--out", str(out), "--overlay", str(overlay)])

    assert set(json.loads(out.read_text())) == {"adaptive_mappo", "fixed_margin_mappo", "adaptive_mappo_no_shield"}


def test_the_run_command_passes_config_and_overlay_to_every_task(tmp_path, monkeypatch):
    overlay = _overlay(tmp_path)
    registry = _registry_file(tmp_path, _run_dir(tmp_path, "n15_adaptive_s0", overlay))
    captured: list = []
    monkeypatch.setattr(experiment, "run_tasks", lambda tasks, out, **kwargs: captured.extend(tasks) or [])
    monkeypatch.setattr(experiment, "read_records", lambda path: [])

    experiment.main(
        [
            "run", "--split", "validation", "--max-seeds", "2", "--models", "rule_based", "adaptive_mappo",
            "--registry", str(registry), "--out", str(tmp_path / "e.jsonl"), "--overlay", str(overlay),
            "--config", str(DEFAULT_CONFIG_PATH),
        ]
    )

    assert len(captured) == 4
    assert {task.overlay_path for task in captured} == {str(overlay)}
    assert {task.config_path for task in captured} == {str(DEFAULT_CONFIG_PATH)}


def test_the_run_command_refuses_a_checkpoint_trained_under_another_overlay(tmp_path, monkeypatch):
    registry = _registry_file(tmp_path, _run_dir(tmp_path, "n15_adaptive_s0", _overlay(tmp_path)))
    monkeypatch.setattr(experiment, "run_tasks", lambda *a, **k: pytest.fail("must refuse before running"))

    with pytest.raises(CheckpointProvenanceError):
        experiment.main(
            [
                "run", "--split", "validation", "--max-seeds", "1", "--models", "adaptive_mappo",
                "--registry", str(registry), "--out", str(tmp_path / "e.jsonl"),
            ]
        )


def test_the_freeze_command_refuses_a_checkpoint_trained_under_another_overlay(tmp_path):
    runs = _all_runs(tmp_path, _overlay(tmp_path))
    registry = tmp_path / "reg.json"
    registry.write_text(json.dumps(
        {model: {str(seed): str(path) for seed, path in seeds.items()} for model, seeds in build_registry(runs).items()}
    ))

    with pytest.raises(CheckpointProvenanceError):
        experiment.main(["freeze", "--registry", str(registry), "--out", str(tmp_path / "frozen.json")])

    assert not (tmp_path / "frozen.json").exists()
