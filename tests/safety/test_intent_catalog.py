"""The action catalog is the single source of maneuver labels, delta-v and exhaust direction for the replay UI
(redesign real-data rule 5). It must agree with the environment's own `map_intent_to_delta_v`, and importing it
must not drag Ray or torch into the replay import closure (FR16)."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from orbit_guard.safety.maneuver_mapper import (
    DEFAULT_BASE_DELTA_V_MPS,
    DEFAULT_PHASE_FRACTION,
    ManeuverIntent,
    intent_catalog,
    map_intent_to_delta_v,
)

_POSITION = np.array([6.9e6, 0.0, 0.0])
_VELOCITY = np.array([0.0, 7.6e3, 0.0])


def test_catalog_has_one_entry_per_intent_in_index_order():
    catalog = intent_catalog()
    assert [spec.index for spec in catalog] == [int(intent) for intent in ManeuverIntent]
    assert [spec.name for spec in catalog] == [intent.name for intent in ManeuverIntent]


def test_labels_are_derived_from_the_enum_names():
    labels = {spec.index: spec.label for spec in intent_catalog()}
    assert labels[0] == "NO MANEUVER"
    assert labels[1] == "RAISE ORBIT"
    assert labels[3] == "PHASE FORWARD"
    assert labels[5] == "RADIAL ADJUSTMENT"


def test_short_labels_are_unique_and_short():
    shorts = [spec.short for spec in intent_catalog()]
    assert len(set(shorts)) == len(shorts)
    assert all(len(short) <= 7 for short in shorts)


@pytest.mark.parametrize("base,fraction", [(DEFAULT_BASE_DELTA_V_MPS, DEFAULT_PHASE_FRACTION), (0.2, 0.25), (0.05, 1.0)])
def test_delta_v_matches_the_mapper_for_every_intent(base, fraction):
    for spec in intent_catalog(base, fraction):
        expected = float(
            np.linalg.norm(
                map_intent_to_delta_v(spec.index, _POSITION, _VELOCITY, base_delta_v_mps=base, phase_fraction=fraction)
            )
        )
        assert spec.delta_v_mps == pytest.approx(expected)


def test_defaults_match_the_mappers_own_defaults():
    delta_v = float(np.linalg.norm(map_intent_to_delta_v(ManeuverIntent.RAISE_ORBIT, _POSITION, _VELOCITY)))
    assert delta_v == pytest.approx(DEFAULT_BASE_DELTA_V_MPS)
    phase = float(np.linalg.norm(map_intent_to_delta_v(ManeuverIntent.PHASE_FORWARD, _POSITION, _VELOCITY)))
    assert phase == pytest.approx(DEFAULT_BASE_DELTA_V_MPS * DEFAULT_PHASE_FRACTION)


def test_exhaust_is_opposite_to_the_burn_in_rtn():
    """exhaust_rtn is (radial, tangential, normal); a prograde burn throws exhaust backwards."""
    exhaust = {spec.name: spec.exhaust_rtn for spec in intent_catalog()}
    assert exhaust["NO_MANEUVER"] == (0.0, 0.0, 0.0)
    assert exhaust["RAISE_ORBIT"] == (0.0, -1.0, 0.0)
    assert exhaust["PHASE_BACKWARD"] == (0.0, -1.0, 0.0)
    assert exhaust["LOWER_ORBIT"] == (0.0, 1.0, 0.0)
    assert exhaust["PHASE_FORWARD"] == (0.0, 1.0, 0.0)
    assert exhaust["RADIAL_ADJUSTMENT"] == (-1.0, 0.0, 0.0)


def test_exhaust_agrees_with_the_burn_direction_of_the_mapper():
    """Cross-check the table against physics: burn direction in ECI == -exhaust rotated from RTN."""
    for spec in intent_catalog():
        burn = map_intent_to_delta_v(spec.index, _POSITION, _VELOCITY)
        if not np.linalg.norm(burn):
            continue
        radial, tangential = _POSITION / np.linalg.norm(_POSITION), _VELOCITY / np.linalg.norm(_VELOCITY)
        exhaust_eci = spec.exhaust_rtn[0] * radial + spec.exhaust_rtn[1] * tangential
        assert np.dot(burn / np.linalg.norm(burn), exhaust_eci) == pytest.approx(-1.0)


def test_catalog_is_immutable():
    catalog = intent_catalog()
    assert isinstance(catalog, tuple)
    with pytest.raises(AttributeError):
        catalog[0].label = "x"  # frozen dataclass


def test_importing_the_catalog_pulls_in_no_ray_or_torch():
    code = (
        "import sys; import orbit_guard.safety.maneuver_mapper as m; m.intent_catalog(); "
        "bad = [n for n in ('ray', 'torch') if n in sys.modules]; sys.exit(1 if bad else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
