"""scripts/run_experiments.sh honours OVERLAY on every experiment step (PRD §33; CORE-1 overlay plumbing).

The script is run for real with a fake `uv` first on PATH that only logs its argv, so no episode,
freeze or result is computed; the test checks which flags each pipeline step would have received.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_experiments.sh"
_SEPARATOR = "\x1f"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run_script(
    tmp_path: Path, extra_env: dict[str, str], args: list[str] | None = None, *, cwd: Path | None = None,
    check: bool = True,
) -> list[list[str]]:
    """Run the script with a logging fake uv; `args` defaults to the positional RUNS and OUT dirs under tmp_path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "uv_calls.log"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text("#!/bin/sh\nprintf '%s\\037' \"$@\" >> \"$UV_LOG\"\nprintf '\\n' >> \"$UV_LOG\"\n")
    fake_uv.chmod(0o755)
    env = {
        **os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "UV_LOG": str(log), "WORKERS": "1",
        **extra_env,
    }
    runs, out = tmp_path / "runs", tmp_path / "out"
    runs.mkdir(exist_ok=True)
    positional = [str(runs), str(out)] if args is None else args
    subprocess.run(["bash", str(_SCRIPT), *positional], env=env, check=check, capture_output=True, text=True, cwd=cwd)
    return [line.rstrip(_SEPARATOR).split(_SEPARATOR) for line in log.read_text().splitlines()] if log.exists() else []


def _step(call: list[str]) -> str:
    return call[call.index("orbit_guard.evaluation.experiment") + 1]


def _flag(call: list[str], flag: str) -> str:
    return call[call.index(flag) + 1]


def _experiment_calls(calls: list[list[str]]) -> list[list[str]]:
    return [call for call in calls if "orbit_guard.evaluation.experiment" in call]


def test_every_experiment_step_receives_the_overlay(tmp_path):
    overlay = tmp_path / "tuned.yaml"

    calls = _experiment_calls(_run_script(tmp_path, {"OVERLAY": str(overlay)}))

    steps = [call[call.index("orbit_guard.evaluation.experiment") + 1] for call in calls]
    assert steps == ["registry", "run", "freeze", "run"]
    for call in calls:
        assert "--overlay" in call, call
        assert call[call.index("--overlay") + 1] == str(overlay)


def test_without_an_overlay_no_step_passes_one(tmp_path):
    env = {"OVERLAY": ""}

    calls = _experiment_calls(_run_script(tmp_path, env))

    assert len(calls) == 4
    assert all("--overlay" not in call for call in calls)


def test_runs_comes_from_the_environment_when_no_argument_is_given(tmp_path):
    # queue_n15.sh documents RUNS=... as an env var; the experiments must read the same runs dir from it.
    tuned = tmp_path / "runs_tuned"

    calls = _experiment_calls(_run_script(tmp_path, {"RUNS": str(tuned), "OVERLAY": ""}, args=[]))

    registry = next(call for call in calls if _step(call) == "registry")
    assert _flag(registry, "--runs-dir") == str(tuned)
    assert _flag(registry, "--out") == str(tuned / "experiments" / "registry.json")


def test_a_positional_runs_dir_overrides_the_environment(tmp_path):
    calls = _experiment_calls(_run_script(tmp_path, {"RUNS": str(tmp_path / "ignored"), "OVERLAY": ""}))

    registry = next(call for call in calls if _step(call) == "registry")
    assert _flag(registry, "--runs-dir") == str(tmp_path / "runs")


def test_an_existing_freeze_is_verified_by_the_test_run_instead_of_rewritten(tmp_path):
    # An interrupted final-test sweep resumes: the one-way freeze is not attempted again (it would refuse), and
    # `run --split test` re-verifies the existing snapshot before reading the test pool.
    out = tmp_path / "out"
    out.mkdir()
    (out / "frozen.json").write_text("{}")

    calls = _experiment_calls(_run_script(tmp_path, {"OVERLAY": ""}))

    assert [_step(call) for call in calls] == ["registry", "run", "run"]
    test_run = calls[-1]
    assert _flag(test_run, "--split") == "test" and _flag(test_run, "--freeze-snapshot") == str(out / "frozen.json")


def test_relative_paths_resolve_against_the_callers_directory_not_the_repo(tmp_path):
    # The script cd's to the repo root before the python steps; relative inputs must not follow it there.
    caller = tmp_path / "caller"
    (caller / "runs").mkdir(parents=True)

    calls = _experiment_calls(_run_script(
        tmp_path, {"OVERLAY": "tuned.yaml"}, args=["runs", "out"], cwd=caller,
    ))

    registry = next(call for call in calls if _step(call) == "registry")
    assert _flag(registry, "--runs-dir") == str(caller / "runs")
    assert _flag(registry, "--out") == str(caller / "out" / "registry.json")
    assert {_flag(call, "--overlay") for call in calls} == {str(caller / "tuned.yaml")}


def test_out_comes_from_the_environment_when_no_argument_is_given(tmp_path):
    out = tmp_path / "final_out"

    calls = _experiment_calls(_run_script(tmp_path, {"RUNS": str(tmp_path / "runs"), "OUT": str(out), "OVERLAY": ""}, args=[]))

    registry = next(call for call in calls if _step(call) == "registry")
    assert _flag(registry, "--out") == str(out / "registry.json")


def test_a_torn_frozen_snapshot_stops_the_pipeline_before_the_test_run(tmp_path):
    # A freeze killed mid-write leaves frozen.json unparseable; skipping the freeze then must not pass silently.
    out = tmp_path / "out"
    out.mkdir()
    (out / "frozen.json").write_text('{"format": 3, "checkpo')

    calls = _experiment_calls(_run_script(tmp_path, {"OVERLAY": ""}, check=False))

    assert [_step(call) for call in calls] == ["registry", "run"]
    assert all(_flag(call, "--split") != "test" for call in calls if _step(call) == "run")
