"""Evaluation workers import neither Ray nor torch (~660 MB per spawn worker otherwise).

`run_tasks` pickles `run_episode_task` into spawn workers, which re-import `evaluation.runner` and with it
`evaluation.baselines` and `rl.evaluation`; learned policies run from exported NumPy actor weights.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_PROBE = """
import json, sys
import orbit_guard.evaluation.runner, orbit_guard.evaluation.baselines, orbit_guard.evaluation.trained_values
print(json.dumps(sorted(m for m in ("ray", "torch") if m in sys.modules)))
"""


def test_the_modules_an_evaluation_worker_imports_load_neither_ray_nor_torch():
    repo_root = Path(__file__).resolve().parents[2]

    completed = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=repo_root, capture_output=True, text=True, timeout=300, check=True
    )

    assert json.loads(completed.stdout) == []
