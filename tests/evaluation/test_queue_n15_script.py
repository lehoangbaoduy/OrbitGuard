"""scripts/queue_n15.sh: keeps going through the queue, but a failed run makes the whole queue fail (review2).

The script runs for real with a fake `uv` first on PATH. Training calls are only logged (and fail for the run
directories named in FAIL_RUNS); no model is trained.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "queue_n15.sh"
_SEPARATOR = "\x1f"
_FAKE_UV = """#!/bin/sh
case " $* " in
  *" orbit_guard.rl.training "*)
    printf '%s\\037' "$@" >> "$UV_LOG"; printf '\\n' >> "$UV_LOG"
    for name in $FAIL_RUNS; do
      case " $* " in *"/$name "*) exit 1 ;; esac
    done
    exit 0 ;;
esac
shift  # run
shift  # python
exec "$REAL_PYTHON" "$@"
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def run_queue(tmp_path: Path, runs: Path, **extra_env: str) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "uv_calls.log"
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(_FAKE_UV)
    fake_uv.chmod(0o755)
    env = {
        **os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "UV_LOG": str(log),
        "REAL_PYTHON": sys.executable, "RUNS": str(runs), "OVERLAY": "", "FAIL_RUNS": "", **extra_env,
    }
    completed = subprocess.run(
        ["bash", str(_SCRIPT)], env=env, capture_output=True, text=True, timeout=300, check=False
    )
    calls = [line.rstrip(_SEPARATOR).split(_SEPARATOR) for line in log.read_text().splitlines()] if log.exists() else []
    return completed, calls


def trained_runs(calls: list[list[str]]) -> list[str]:
    return [Path(call[call.index("--checkpoint-dir") + 1]).name for call in calls]


def test_a_clean_queue_trains_all_nine_runs_and_exits_zero(tmp_path):
    completed, calls = run_queue(tmp_path, tmp_path / "runs")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(trained_runs(calls)) == 9


def test_a_failed_run_does_not_stop_the_queue_but_fails_it_with_a_summary(tmp_path):
    completed, calls = run_queue(tmp_path, tmp_path / "runs", FAIL_RUNS="n15_noshield_s1 n15_fixed_s2")

    assert completed.returncode != 0
    assert len(trained_runs(calls)) == 9  # every run after the failure still trained
    summary = completed.stdout.strip().splitlines()[-1]
    assert "2 of 9" in summary and "noshield seed 1" in summary and "fixed seed 2" in summary
