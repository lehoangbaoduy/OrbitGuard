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


def _run_script(tmp_path: Path, extra_env: dict[str, str]) -> list[list[str]]:
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
    runs.mkdir()
    subprocess.run(["bash", str(_SCRIPT), str(runs), str(out)], env=env, check=True, capture_output=True, text=True)
    return [line.rstrip(_SEPARATOR).split(_SEPARATOR) for line in log.read_text().splitlines()]


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
