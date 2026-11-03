"""`python -m orbit_guard.rl.run_provenance <run dirs>`: which queue runs are done under THIS config (review2, low).

queue_n15.sh used to skip any run dir holding actor_weights.npz, so a RUNS dir of checkpoints trained under another
config (or before sidecars existed) was silently reused. A run is "done" only if its run_config.json records the
config_digest of the config + overlay the queue trains with; "todo" if it has no weights; otherwise it is foreign
and the helper exits non-zero so the queue refuses to overwrite or reuse it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest
from orbit_guard.rl.run_provenance import FOREIGN_EXIT_CODE, main, run_dir_status


def _run(tmp_path: Path, name: str, *, weights: bool = True, sidecar: dict | str | None = None) -> Path:
    run = tmp_path / name
    run.mkdir()
    if weights:
        (run / "actor_weights.npz").write_bytes(b"w")
    if sidecar is not None:
        (run / "run_config.json").write_text(sidecar if isinstance(sidecar, str) else json.dumps(sidecar))
    return run


def _overlay(tmp_path: Path) -> Path:
    path = tmp_path / "tuned.yaml"
    path.write_text("reward:\n  w3: 2.5\n")
    return path


def test_a_missing_or_weightless_run_is_todo(tmp_path):
    assert run_dir_status(tmp_path / "absent")[0] == "todo"
    assert run_dir_status(_run(tmp_path, "partial", weights=False))[0] == "todo"


def test_a_run_trained_under_the_same_config_and_overlay_is_done(tmp_path):
    overlay = _overlay(tmp_path)
    run = _run(tmp_path, "run", sidecar={"config_digest": config_digest(DEFAULT_CONFIG_PATH, overlay)})

    assert run_dir_status(run, overlay_path=overlay)[0] == "done"
    assert run_dir_status(run)[0] == "foreign"  # the same run, but the queue now trains without the overlay


@pytest.mark.parametrize("sidecar", [None, "not json", {"overlay_path": None}, {"config_digest": "x" * 64}])
def test_weights_without_a_matching_sidecar_are_foreign(tmp_path, sidecar):
    status, reason = run_dir_status(_run(tmp_path, "run", sidecar=sidecar))

    assert status == "foreign" and "run" in reason


def test_the_cli_prints_one_status_per_run_and_fails_on_a_foreign_run(tmp_path, capsys):
    done = _run(tmp_path, "done", sidecar={"config_digest": config_digest(DEFAULT_CONFIG_PATH, None)})
    todo = tmp_path / "todo"

    main([str(done), str(todo)])
    assert capsys.readouterr().out.splitlines() == [f"done {done}", f"todo {todo}"]

    stale = _run(tmp_path, "stale")
    with pytest.raises(SystemExit) as exited:
        main([str(done), str(stale)])
    assert exited.value.code == FOREIGN_EXIT_CODE
    assert "stale" in capsys.readouterr().err


def test_an_empty_overlay_argument_means_no_overlay(tmp_path, capsys):
    done = _run(tmp_path, "done", sidecar={"config_digest": config_digest(DEFAULT_CONFIG_PATH, None)})

    main([str(done), "--overlay", ""])

    assert capsys.readouterr().out.strip() == f"done {done}"
