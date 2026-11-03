"""scripts/queue_n15.sh: keeps going through the queue, but a failed run makes the whole queue fail (review2), and an
existing run is skipped only when its run_config.json records the queue's own config_digest; any other finished
run makes the queue refuse to start (use a fresh RUNS) rather than reuse or overwrite it.

The script runs for real with a fake `uv` first on PATH. Training calls are only logged (and fail for the run
directories named in FAIL_RUNS); no model is trained. Other `uv run python ...` calls (the run-dir status helper)
are forwarded to the real interpreter.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "queue_n15.sh"
_SEPARATOR = "\x1f"
_FAKE_UV = """#!/bin/sh
case " $* " in
  *" orbit_guard.rl.training "*)
    printf '%s\\037' "$@" >> "$UV_LOG"; printf '\\n' >> "$UV_LOG"
    printf '%s\\n' "$RAY_TMPDIR" >> "$UV_LOG.ray_tmpdir"
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


def ray_tmpdirs(tmp_path: Path) -> set[str]:
    return set((tmp_path / "uv_calls.log.ray_tmpdir").read_text().splitlines())


def test_without_a_ray_tmpdir_the_queue_puts_ray_on_disk_under_runs(tmp_path, monkeypatch):
    monkeypatch.delenv("RAY_TMPDIR", raising=False)
    runs = tmp_path / "runs"

    completed, _ = run_queue(tmp_path, runs)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert ray_tmpdirs(tmp_path) == {f"{runs}/ray_tmp"}


def test_an_inherited_ray_tmpdir_is_kept(tmp_path):
    # Ray's socket paths under a deep RUNS dir exceed AF_UNIX's 107 bytes; a short RAY_TMPDIR must win.
    completed, _ = run_queue(tmp_path, tmp_path / "runs", RAY_TMPDIR="/short/ray_tmp")

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert ray_tmpdirs(tmp_path) == {"/short/ray_tmp"}


def test_a_failed_run_does_not_stop_the_queue_but_fails_it_with_a_summary(tmp_path):
    completed, calls = run_queue(tmp_path, tmp_path / "runs", FAIL_RUNS="n15_noshield_s1 n15_fixed_s2")

    assert completed.returncode != 0
    assert len(trained_runs(calls)) == 9  # every run after the failure still trained
    summary = completed.stdout.strip().splitlines()[-1]
    assert "2 of 9" in summary and "noshield seed 1" in summary and "fixed seed 2" in summary


# --- reuse of an existing RUNS dir: only runs trained under the queue's own config + overlay are skipped -------


def _finished(runs: Path, name: str, sidecar: dict | None) -> Path:
    run = runs / name
    run.mkdir(parents=True)
    (run / "actor_weights.npz").write_bytes(b"trained earlier")
    if sidecar is not None:
        (run / "run_config.json").write_text(json.dumps(sidecar))
    return run


def _digest(overlay: Path | None = None) -> dict:
    return {"config_digest": config_digest(DEFAULT_CONFIG_PATH, overlay)}


def test_a_run_trained_under_the_same_config_is_skipped(tmp_path):
    runs = tmp_path / "runs"
    _finished(runs, "n15_adaptive_s0", _digest())

    completed, calls = run_queue(tmp_path, runs)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "n15_adaptive_s0" not in trained_runs(calls) and len(trained_runs(calls)) == 8


def test_a_run_is_skipped_only_under_the_overlay_it_was_trained_with(tmp_path):
    runs, overlay = tmp_path / "runs", tmp_path / "tuned.yaml"
    overlay.write_text("reward:\n  w3: 2.5\n")
    _finished(runs, "n15_fixed_s1", _digest(overlay))

    completed, calls = run_queue(tmp_path, runs, OVERLAY=str(overlay))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert len(trained_runs(calls)) == 8

    (tmp_path / "uv_calls.log").unlink()
    completed, calls = run_queue(tmp_path, runs)  # the same runs dir, but now without the overlay
    assert completed.returncode != 0 and not calls


@pytest.mark.parametrize("sidecar", [None, {"config_digest": "0" * 64}], ids=["no-sidecar", "other-digest"])
def test_a_foreign_finished_run_is_neither_reused_nor_overwritten(tmp_path, sidecar):
    runs = tmp_path / "runs"
    stale = _finished(runs, "n15_noshield_s2", sidecar)

    completed, calls = run_queue(tmp_path, runs)

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0
    assert not calls, "nothing is trained into a runs dir holding foreign checkpoints"
    assert (stale / "actor_weights.npz").read_bytes() == b"trained earlier"
    assert "fresh RUNS" in output and "n15_noshield_s2" in output


def test_a_failing_status_helper_is_not_reported_as_foreign_runs(tmp_path):
    runs = tmp_path / "runs"
    _finished(runs, "n15_adaptive_s0", _digest())

    completed, calls = run_queue(tmp_path, runs, OVERLAY=str(tmp_path / "missing.yaml"))

    output = completed.stdout + completed.stderr
    assert completed.returncode != 0 and not calls
    assert "status helper failed" in output and "fresh RUNS" not in output


def test_a_partial_run_without_weights_is_cleared_and_retrained(tmp_path):
    runs = tmp_path / "runs"
    partial = runs / "n15_adaptive_s1"
    partial.mkdir(parents=True)
    (partial / "rllib_state.pkl").write_bytes(b"interrupted")

    completed, calls = run_queue(tmp_path, runs)

    assert completed.returncode == 0
    assert "n15_adaptive_s1" in trained_runs(calls) and not (partial / "rllib_state.pkl").exists()
