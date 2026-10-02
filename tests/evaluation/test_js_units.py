"""Runs the `node --test` unit tests of the replay stage's pure JS helpers against payloads built from REAL recorded
fixture episodes (so the JS is checked on the same arrays the browser receives, never on hand-made data)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from orbit_guard.visualization.derived.status import (
    EVENT_NAMES,
    PAIR_STATUS_NAMES,
    SAT_STATUS_NAMES,
)
from orbit_guard.visualization.payload import build_payload
from orbit_guard.visualization.source import ReplaySource
from tests.evaluation.replay_fixtures import (
    capability_trace,
    depletion_trace,
    fleet_collision_trace,
    full_size_trace,
    shielded_trace,
)

JS_TESTS = Path(__file__).parents[1] / "js"
ALLOW_NO_NODE = os.environ.get("OG_ALLOW_NO_NODE") == "1"  # opt-in: the JS tests must never be dropped silently
STAGE_JS = Path(__file__).parents[2] / "orbit_guard" / "visualization" / "assets"


def _fixture(trace) -> dict:
    source = ReplaySource(trace)
    return {
        "payload": build_payload(source),
        "expected": {
            "positions_km": (trace.positions_m / 1000.0).astype(np.float32).ravel().tolist(),
            "approved": trace.approved.ravel().tolist(),
            "sat_status": source.sat_status.ravel().tolist(),
        },
    }


@pytest.fixture(scope="module")
def fixtures_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("js_fixtures")
    for name, trace in (("shielded", shielded_trace()), ("fleet_collision", fleet_collision_trace()), ("depletion", depletion_trace()), ("capability", capability_trace()), ("full_size", full_size_trace())):
        (root / f"{name}.json").write_text(json.dumps(_fixture(trace), allow_nan=False))
    return root


def test_js_unit_tests_pass(fixtures_dir):
    if shutil.which("node") is None:
        if ALLOW_NO_NODE:
            pytest.skip("node is not installed (OG_ALLOW_NO_NODE=1)")
        pytest.fail("node is required for the replay stage's JS tests; install it or set OG_ALLOW_NO_NODE=1")
    completed = subprocess.run(
        [shutil.which("node"), "--test", *map(str, sorted(JS_TESTS.glob("*.test.mjs")))],
        env={**os.environ, "OG_FIXTURES": str(fixtures_dir)},
        capture_output=True, text=True, timeout=180, check=False,
    )

    assert completed.returncode == 0, completed.stdout[-4000:] + completed.stderr[-2000:]


def _constant_pair(source: str, left: str, right: str) -> tuple[int, int]:
    match = re.search(rf"const {left} = (\d+), {right} = (\d+)", source)
    assert match, f"{left}/{right} constants not found"
    return int(match.group(1)), int(match.group(2))


def test_js_status_and_event_codes_match_the_python_vocabulary():
    """The scene's literal codes must equal the ones the payload's arrays use (derived/status.py)."""
    source = (STAGE_JS / "13_og_scene_dynamic.js").read_text()
    assert re.search(r"const STATUS_COLLIDED = (\d+)", source).group(1) == str(SAT_STATUS_NAMES.index("collided"))
    maneuver, shield = _constant_pair(source, "EVENT_MANEUVER", "EVENT_SHIELD")
    assert (maneuver, shield) == (EVENT_NAMES.index("maneuver"), EVENT_NAMES.index("shield_intervention"))
    assert f"EVENT_COLLISION = {EVENT_NAMES.index('collision')}" in source
    inside, high = _constant_pair(source, "PAIR_INSIDE", "PAIR_HIGH_RISK")
    assert (inside, high) == (PAIR_STATUS_NAMES.index("inside_d_safe"), PAIR_STATUS_NAMES.index("high_risk"))
    sat_inside, sat_high = _constant_pair(source, "SAT_INSIDE", "SAT_HIGH_RISK")
    assert (sat_inside, sat_high) == (SAT_STATUS_NAMES.index("inside_d_safe"), SAT_STATUS_NAMES.index("high_risk"))
