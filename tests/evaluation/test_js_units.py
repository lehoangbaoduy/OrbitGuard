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


# Every status and event code the browser hard-codes, by the name the asset uses, with the Python vocabulary's value.
VOCAB_CODES = {
    "STATUS_COLLIDED": SAT_STATUS_NAMES.index("collided"),
    "SAT_INSIDE": SAT_STATUS_NAMES.index("inside_d_safe"),
    "SAT_HIGH_RISK": SAT_STATUS_NAMES.index("high_risk"),
    "PAIR_INSIDE": PAIR_STATUS_NAMES.index("inside_d_safe"),
    "PAIR_HIGH_RISK": PAIR_STATUS_NAMES.index("high_risk"),
    "PAIR_COLLISION": PAIR_STATUS_NAMES.index("collision"),
    "EVENT_MANEUVER": EVENT_NAMES.index("maneuver"),
    "EVENT_SHIELD": EVENT_NAMES.index("shield_intervention"),
    "EVENT_COLLISION": EVENT_NAMES.index("collision"),
}
CODE_ASSETS = ("13_og_scene_dynamic.js", "20_og_panels.js", "21_og_models.js")  # every asset that hard-codes codes; test_every_asset_that_hard_codes_them keeps this complete
CODE_DEFINITION = re.compile(r"\b(?:STATUS_COLLIDED|(?:PAIR|SAT|EVENT)_[A-Z_]+)\s*=\s*\d|\bEVENT = \{")


def _declared_codes(source: str) -> dict[str, int]:
    """`NAME = 3` declarations, and `EVENT = { maneuver: 1, ... }` object literals as `EVENT_MANEUVER` etc."""
    codes = {name: int(value) for name, value in re.findall(r"\b([A-Z][A-Z_]*)\s*=\s*(\d+)\b", source)}
    for body in re.findall(r"\bEVENT = \{([^}]*)\}", source):
        codes.update({f"EVENT_{name.upper()}": int(value) for name, value in re.findall(r"(\w+):\s*(\d+)", body)})
    return codes


def _assert_codes_match(source: str) -> None:
    declared = _declared_codes(source)
    for name, value in declared.items():
        if name in VOCAB_CODES:
            assert value == VOCAB_CODES[name], f"{name} = {value} in the asset; the Python vocabulary says {VOCAB_CODES[name]}"
    for name in VOCAB_CODES:
        if re.search(rf"\b{name}\b", source):
            assert name in declared, f"{name} is used by the asset but never declared"


@pytest.mark.parametrize("asset", CODE_ASSETS)
def test_js_status_and_event_codes_match_the_python_vocabulary(asset):
    """The asset's literal codes must equal the ones the payload's arrays use (derived/status.py)."""
    _assert_codes_match((STAGE_JS / asset).read_text())


def test_every_asset_that_hard_codes_the_codes_is_checked_against_the_vocabulary():
    hard_coding = {path.name for path in STAGE_JS.glob("*.js") if CODE_DEFINITION.search(path.read_text())}

    assert hard_coding == set(CODE_ASSETS), "an asset hard-codes status/event codes but is not in CODE_ASSETS"
