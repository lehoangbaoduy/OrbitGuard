"""The replay recorder refuses checkpoints it cannot trace (review2, medium: stale sidecar-less runs were replayed).

A trained checkpoint without its run_config.json sidecar is refused, as the freeze refuses it. The script needs an
explicit --runs-dir (its old default pointed at the retired checkpoints), and with --freeze-snapshot every replayed
actor's sha256 must equal the frozen checkpoint of the same model and training seed.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from orbit_guard.evaluation.experiment import CheckpointProvenanceError
from orbit_guard.evaluation.freeze import SNAPSHOT_FORMAT
from orbit_guard.evaluation.replay_recording import (
    EpisodePlan,
    build_plan,
    check_plan_provenance,
)
from orbit_guard.rl.actor_policy import save_actor_weights
from tests.evaluation.sidecar_fixtures import sidecar_record

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "record_replay_traces.py"
_DIMS = {"obs_dim": 54, "num_actions": 6}
_VARIANTS = {"adaptive": ("adaptive_mappo", True), "noshield": ("adaptive_mappo_no_shield", False)}


def _weights(path: Path, seed: int = 5) -> Path:
    from orbit_guard.evaluation.replay_recording import untrained_actor_weights

    path.parent.mkdir(parents=True, exist_ok=True)
    save_actor_weights(untrained_actor_weights(**_DIMS, init_seed=seed), path)  # stands in for a trained actor
    return path


def _runs(tmp_path: Path, *, sidecars: bool = True, trained_seed: int = 0) -> Path:
    runs = tmp_path / "runs"
    for variant, (_model, shield) in _VARIANTS.items():
        weights = _weights(runs / f"n15_{variant}_s{trained_seed}" / "actor_weights.npz")
        if sidecars:
            (weights.parent / "run_config.json").write_text(json.dumps(sidecar_record(None, shield=shield)))
    return runs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(tmp_path: Path, checkpoints: dict[str, Path]) -> Path:
    """A freeze snapshot holding only what the replay check reads: the frozen actor weights hashes."""
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({
        "format": SNAPSHOT_FORMAT,
        "checkpoints": {name: {"actor_weights_sha256": _sha256(file)} for name, file in checkpoints.items()},
    }))
    return path


def _trained_items(runs: Path, trained_seed: int = 0) -> list[EpisodePlan]:
    return [item for item in build_plan(runs, trained_seed=trained_seed) if item.policy == "trained"]


def test_a_trained_checkpoint_without_a_sidecar_is_refused(tmp_path):
    runs = _runs(tmp_path, sidecars=False)

    with pytest.raises(CheckpointProvenanceError, match=r"n15_adaptive_s0.*no run_config\.json"):
        check_plan_provenance(_trained_items(runs))


def test_untrained_items_need_no_sidecar(tmp_path):
    untrained = [item for item in build_plan(tmp_path / "nothing") if item.policy == "untrained"]

    check_plan_provenance(untrained)


def test_the_plan_records_which_training_seed_supplies_each_trained_actor(tmp_path):
    assert {item.training_seed for item in _trained_items(tmp_path, trained_seed=2)} == {2}
    assert {item.training_seed for item in build_plan(tmp_path) if item.policy == "untrained"} == {None}


def test_replay_weights_equal_to_the_frozen_checkpoints_are_accepted(tmp_path):
    runs = _runs(tmp_path)
    frozen = _snapshot(tmp_path, {
        "adaptive_mappo/0": runs / "n15_adaptive_s0" / "actor_weights.npz",
        "adaptive_mappo_no_shield/0": runs / "n15_noshield_s0" / "actor_weights.npz",
    })

    check_plan_provenance(_trained_items(runs), freeze_snapshot=frozen)


def test_replay_weights_other_than_the_frozen_ones_are_refused(tmp_path):
    runs = _runs(tmp_path)
    retrained = _weights(tmp_path / "other" / "actor_weights.npz", seed=9)
    frozen = _snapshot(tmp_path, {
        "adaptive_mappo/0": retrained,
        "adaptive_mappo_no_shield/0": runs / "n15_noshield_s0" / "actor_weights.npz",
    })

    with pytest.raises(CheckpointProvenanceError, match=r"adaptive_mappo/0.*frozen"):
        check_plan_provenance(_trained_items(runs), freeze_snapshot=frozen)


def test_a_checkpoint_the_freeze_does_not_hold_is_refused(tmp_path):
    runs = _runs(tmp_path)
    frozen = _snapshot(tmp_path, {"adaptive_mappo/0": runs / "n15_adaptive_s0" / "actor_weights.npz"})

    with pytest.raises(CheckpointProvenanceError, match=r"adaptive_mappo_no_shield/0.*not in the freeze"):
        check_plan_provenance(_trained_items(runs), freeze_snapshot=frozen)


def _script():
    spec = importlib.util.spec_from_file_location("record_replay_traces_gate", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_script_requires_an_explicit_runs_dir(tmp_path, monkeypatch, capsys):
    script = _script()
    monkeypatch.setattr(script, "record_all", lambda *a, **k: pytest.fail("must not record"))

    with pytest.raises(SystemExit):
        script.main(["--out", str(tmp_path / "out")])

    assert "--runs-dir" in capsys.readouterr().err


def test_the_script_refuses_sidecar_less_checkpoints_before_recording(tmp_path, monkeypatch, capsys):
    script = _script()
    monkeypatch.setattr(script, "record_all", lambda *a, **k: pytest.fail("must not record"))

    with pytest.raises(SystemExit):
        script.main(["--runs-dir", str(_runs(tmp_path, sidecars=False)), "--out", str(tmp_path / "out")])

    assert "run_config.json" in capsys.readouterr().err


def test_the_script_checks_and_forwards_the_freeze_snapshot(tmp_path, monkeypatch, capsys):
    script, calls = _script(), []
    runs = _runs(tmp_path)
    monkeypatch.setattr(script, "record_all", lambda plan, out, **kwargs: calls.append(kwargs) or [])
    stale = _snapshot(tmp_path, {
        "adaptive_mappo/0": _weights(tmp_path / "other" / "actor_weights.npz", seed=9),
        "adaptive_mappo_no_shield/0": runs / "n15_noshield_s0" / "actor_weights.npz",
    })

    with pytest.raises(SystemExit):
        script.main(["--runs-dir", str(runs), "--out", str(tmp_path / "out"), "--freeze-snapshot", str(stale)])
    assert not calls and "frozen" in capsys.readouterr().err

    stale.unlink()
    good = _snapshot(tmp_path, {
        "adaptive_mappo/0": runs / "n15_adaptive_s0" / "actor_weights.npz",
        "adaptive_mappo_no_shield/0": runs / "n15_noshield_s0" / "actor_weights.npz",
    })
    script.main(["--runs-dir", str(runs), "--out", str(tmp_path / "out"), "--freeze-snapshot", str(good)])
    assert calls and calls[0]["freeze_snapshot"] == good
