"""Replay recording under a tuned overlay (replay fix a).

A model trained under a tuned overlay must be replayed in the environment that overlay defines, not in the default
config: the overlay reaches the environment, the configuration digest and overlay path land in the trace meta, and a
checkpoint whose run_config.json sidecar records a different config_digest is refused before anything is recorded.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest
from orbit_guard.evaluation import replay_recording
from orbit_guard.evaluation.experiment import CheckpointProvenanceError
from orbit_guard.evaluation.replay_recording import (
    EpisodePlan,
    record_all,
    record_plan_item,
    untrained_actor_weights,
)
from orbit_guard.rl.actor_policy import save_actor_weights
from orbit_guard.visualization.episodes import discover_episodes
from tests.evaluation.replay_fixtures import ENV_KWARGS
from tests.evaluation.sidecar_fixtures import sidecar_record

_DIMS = {"obs_dim": 54, "num_actions": 6}
_W3 = 7.25  # distinctive: neither the default (5.0) nor any grid value
_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "record_replay_traces.py"


def _overlay(tmp_path: Path, body: str = f"reward:\n  w3: {_W3}\n") -> Path:
    path = tmp_path / "tuned.yaml"
    path.write_text(body)
    return path


def _checkpoint(tmp_path: Path, *, trained_with: Path | None, sidecar: bool = True) -> Path:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    weights = run / "actor_weights.npz"
    save_actor_weights(untrained_actor_weights(**_DIMS, init_seed=5), weights)  # stands in for a trained actor
    if sidecar:
        record = {**sidecar_record(trained_with, num_agents=2), "overlay_path": str(trained_with)}
        (run / "run_config.json").write_text(json.dumps(record))
    return weights


def _trained(weights: Path) -> EpisodePlan:
    return EpisodePlan("normal", 0, "trained", True, "adaptive_mappo", weights)


def _record(tmp_path: Path, item: EpisodePlan, **kwargs) -> Path:
    return record_plan_item(item, num_agents=2, out_dir=tmp_path / "out", init_seed=0, env_kwargs=ENV_KWARGS, **kwargs)


@pytest.fixture
def built_envs(monkeypatch):
    """Spy on the recorder's env factory: the real one runs, and every env it builds is kept for inspection."""
    envs, real = [], replay_recording.build_env_factory

    def spy(*args, **kwargs):
        factory = real(*args, **kwargs)

        def make(config):
            env = factory(config)
            envs.append(env)
            return env

        return make

    monkeypatch.setattr(replay_recording, "build_env_factory", spy)
    return envs


def test_a_replay_recorded_with_an_overlay_runs_in_the_overlays_environment(tmp_path, built_envs):
    overlay = _overlay(tmp_path)
    _record(tmp_path, _trained(_checkpoint(tmp_path, trained_with=overlay)), overlay_path=overlay)
    (env,) = built_envs
    assert env._reward_weights["w3"] == pytest.approx(_W3)


def test_without_an_overlay_the_replay_runs_in_the_default_environment(tmp_path, built_envs):
    _record(tmp_path, _trained(_checkpoint(tmp_path, trained_with=None)))
    (env,) = built_envs
    assert env._reward_weights["w3"] != pytest.approx(_W3)


def test_the_meta_records_the_overlay_path_and_the_config_digest(tmp_path):
    overlay = _overlay(tmp_path)
    path = _record(tmp_path, _trained(_checkpoint(tmp_path, trained_with=overlay)), overlay_path=overlay)
    (entry,) = discover_episodes(path.parent).entries
    assert entry.meta.overlay_path == str(overlay)
    assert entry.meta.config_digest == config_digest(DEFAULT_CONFIG_PATH, overlay)


def test_without_an_overlay_the_meta_records_the_base_config_digest(tmp_path):
    path = _record(tmp_path, EpisodePlan("normal", 0, "untrained", False, "untrained_policy", None))
    (entry,) = discover_episodes(path.parent).entries
    assert entry.meta.overlay_path is None
    assert entry.meta.config_digest == config_digest(DEFAULT_CONFIG_PATH, None)


def test_a_checkpoint_trained_with_an_overlay_is_refused_without_it(tmp_path):
    weights = _checkpoint(tmp_path, trained_with=_overlay(tmp_path))
    with pytest.raises(CheckpointProvenanceError, match="config_digest"):
        _record(tmp_path, _trained(weights))
    assert not list(tmp_path.glob("out/*.npz"))


def test_a_checkpoint_trained_without_an_overlay_is_refused_with_one(tmp_path):
    weights = _checkpoint(tmp_path, trained_with=None)
    with pytest.raises(CheckpointProvenanceError):
        _record(tmp_path, _trained(weights), overlay_path=_overlay(tmp_path))


def test_a_checkpoint_without_a_sidecar_is_refused(tmp_path):
    # review2: the stale ~/orbitguard_runs checkpoints have no sidecar and were silently replayed as final models.
    weights = _checkpoint(tmp_path, trained_with=None, sidecar=False)
    with pytest.raises(CheckpointProvenanceError, match=r"run_config\.json"):
        _record(tmp_path, _trained(weights), overlay_path=_overlay(tmp_path))
    assert not list(tmp_path.glob("out/*.npz"))


def test_a_replayed_checkpoint_must_have_been_trained_as_its_training_seed(tmp_path):
    weights = _checkpoint(tmp_path, trained_with=None)  # the sidecar records seed 0
    item = EpisodePlan("normal", 0, "trained", True, "adaptive_mappo", weights, training_seed=1)
    with pytest.raises(CheckpointProvenanceError, match="seed"):
        _record(tmp_path, item)
    assert not list(tmp_path.glob("out/*.npz"))


def test_record_all_refuses_before_recording_any_episode(tmp_path):
    weights = _checkpoint(tmp_path, trained_with=_overlay(tmp_path))
    plan = [EpisodePlan("normal", 0, "untrained", False, "untrained_policy", None), _trained(weights)]
    with pytest.raises(CheckpointProvenanceError):
        record_all(plan, tmp_path / "out", num_agents=2, env_kwargs=ENV_KWARGS, progress=lambda _line: None)
    assert not list(tmp_path.glob("out/*.npz")), "no episode is recorded before every checkpoint is checked"


def test_record_all_threads_the_overlay_to_every_episode(tmp_path, built_envs):
    overlay = _overlay(tmp_path)
    weights = _checkpoint(tmp_path, trained_with=overlay)
    plan = [EpisodePlan("normal", 0, "untrained", False, "untrained_policy", None), _trained(weights)]
    paths = record_all(
        plan, tmp_path / "out", num_agents=2, env_kwargs=ENV_KWARGS, overlay_path=overlay, progress=lambda _line: None
    )
    assert len(paths) == 2 and [env._reward_weights["w3"] for env in built_envs] == pytest.approx([_W3, _W3])


def test_the_untrained_actor_takes_its_architecture_from_the_overlay(tmp_path):
    overlay = _overlay(tmp_path, "rl:\n  actor_hidden_layers: [32, 16]\n")
    layers = untrained_actor_weights(**_DIMS, init_seed=0, overlay_path=overlay).layers
    assert [w.shape for w, _ in layers] == [(32, 54), (16, 32), (6, 16)]


def _script():
    spec = importlib.util.spec_from_file_location("record_replay_traces_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_weights(runs: Path, *, trained_with: Path | None) -> None:
    for variant in ("adaptive", "noshield"):
        run = runs / f"n15_{variant}_s0"
        run.mkdir(parents=True)
        save_actor_weights(untrained_actor_weights(**_DIMS, init_seed=5), run / "actor_weights.npz")
        record = sidecar_record(trained_with, shield=variant == "adaptive")
        (run / "run_config.json").write_text(json.dumps(record))


def test_the_script_passes_overlay_to_the_recorder(tmp_path, monkeypatch):
    script, overlay, calls = _script(), _overlay(tmp_path), []
    _all_weights(tmp_path / "runs", trained_with=overlay)
    monkeypatch.setattr(script, "record_all", lambda plan, out, **kwargs: calls.append(kwargs) or [])
    script.main(["--runs-dir", str(tmp_path / "runs"), "--out", str(tmp_path / "out"), "--overlay", str(overlay), "--unfrozen"])
    assert calls and calls[0]["overlay_path"] == overlay


def test_the_script_refuses_a_mismatched_checkpoint_before_recording(tmp_path, monkeypatch, capsys):
    script, calls = _script(), []
    _all_weights(tmp_path / "runs", trained_with=_overlay(tmp_path))
    monkeypatch.setattr(script, "record_all", lambda plan, out, **kwargs: calls.append(kwargs) or [])
    with pytest.raises(SystemExit):
        script.main(["--runs-dir", str(tmp_path / "runs"), "--out", str(tmp_path / "out"), "--unfrozen"])
    assert not calls and "config_digest" in capsys.readouterr().err
