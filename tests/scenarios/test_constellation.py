from pathlib import Path

import numpy as np
import pytest

from orbit_guard.physics.state import SatelliteCapability
from orbit_guard.scenarios.constellation import (
    MAX_SUPPORTED_N,
    MIN_SUPPORTED_N,
    ShellDistribution,
    fit_shell_distribution,
    load_tle_snapshot,
    sample_constellation,
)

_TLE_PATH = Path(__file__).resolve().parents[2] / "data" / "TLE_snapshot_53deg_shell.csv"

_CAPABILITY_TEMPLATE = SatelliteCapability(
    max_delta_v_per_action_mps=0.1,
    remaining_delta_v_mps=2.0,
    maneuver_enabled=True,
    mission_deviation_limit=1.0,
)


def test_load_tle_snapshot_reads_the_frozen_file_correctly():
    data = load_tle_snapshot(_TLE_PATH)
    assert len(data["norad_id"]) == 3002
    assert len(set(data["norad_id"].tolist())) == 3002  # no duplicates
    for column in ("inclination_deg", "altitude_km", "eccentricity", "arg_periapsis_deg"):
        assert column in data
        assert len(data[column]) == 3002


def test_load_tle_snapshot_raises_on_missing_file():
    with pytest.raises(FileNotFoundError):
        load_tle_snapshot(Path("/nonexistent/path.csv"))


def test_fit_shell_distribution_matches_documented_prd_ranges():
    # PRD §16's "53.04-53.17 deg / 460-465 km" ranges are rounded prose
    # summaries of the real data, not exact bounds -- the actual CSV's tails
    # extend slightly past them (observed: incl. min 53.0355, altitude max
    # 465.30). The CSV is ground truth here; tolerances below reflect that
    # rounding rather than re-deriving stricter bounds from the PRD text.
    data = load_tle_snapshot(_TLE_PATH)
    shell = fit_shell_distribution(data)

    assert shell.inclination_deg_min == pytest.approx(53.04, abs=0.01)
    assert shell.inclination_deg_max == pytest.approx(53.17, abs=0.01)
    assert shell.altitude_km_min == pytest.approx(460.0, abs=0.5)
    assert shell.altitude_km_max == pytest.approx(465.0, abs=0.5)
    assert shell.eccentricity_max < 0.0008
    assert shell.altitude_km_mean == pytest.approx(463.0, abs=1.0)
    assert shell.inclination_deg_mean == pytest.approx(53.16, abs=0.02)


def test_sample_constellation_rejects_n_below_supported_range():
    shell = _toy_shell_distribution()
    with pytest.raises(ValueError):
        sample_constellation(MIN_SUPPORTED_N - 1, seed=1, shell=shell, capability_template=_CAPABILITY_TEMPLATE)


def test_sample_constellation_rejects_n_above_supported_range():
    shell = _toy_shell_distribution()
    with pytest.raises(ValueError):
        sample_constellation(MAX_SUPPORTED_N + 1, seed=1, shell=shell, capability_template=_CAPABILITY_TEMPLATE)


@pytest.mark.parametrize("n", [MIN_SUPPORTED_N, 15, MAX_SUPPORTED_N])
def test_sample_constellation_produces_exactly_n_satellites(n):
    shell = _toy_shell_distribution()
    satellites = sample_constellation(n, seed=42, shell=shell, capability_template=_CAPABILITY_TEMPLATE)
    assert len(satellites) == n


def test_sample_constellation_is_deterministic_under_fixed_seed():
    shell = _toy_shell_distribution()
    first = sample_constellation(15, seed=7, shell=shell, capability_template=_CAPABILITY_TEMPLATE)
    second = sample_constellation(15, seed=7, shell=shell, capability_template=_CAPABILITY_TEMPLATE)

    for sat_a, sat_b in zip(first, second):
        assert sat_a.state.position_m == pytest.approx(sat_b.state.position_m)
        assert sat_a.state.velocity_mps == pytest.approx(sat_b.state.velocity_mps)


def test_sample_constellation_differs_across_seeds():
    shell = _toy_shell_distribution()
    first = sample_constellation(15, seed=1, shell=shell, capability_template=_CAPABILITY_TEMPLATE)
    second = sample_constellation(15, seed=2, shell=shell, capability_template=_CAPABILITY_TEMPLATE)

    assert any(
        sat_a.state.position_m != pytest.approx(sat_b.state.position_m)
        for sat_a, sat_b in zip(first, second)
    )


def test_sampled_satellites_have_finite_state_and_requested_capability():
    shell = _toy_shell_distribution()
    satellites = sample_constellation(15, seed=3, shell=shell, capability_template=_CAPABILITY_TEMPLATE)

    for sat in satellites:
        assert np.all(np.isfinite(sat.state.as_vector()))
        assert sat.capability == _CAPABILITY_TEMPLATE


def test_sampled_altitudes_stay_within_shell_bounds():
    shell = _toy_shell_distribution()
    satellites = sample_constellation(20, seed=5, shell=shell, capability_template=_CAPABILITY_TEMPLATE)

    from orbit_guard.physics.constants import R_EARTH_EQUATORIAL_M

    for sat in satellites:
        radius_m = np.linalg.norm(sat.state.position_m)
        altitude_km = (radius_m - R_EARTH_EQUATORIAL_M) / 1000.0
        assert shell.altitude_km_min - 1.0 <= altitude_km <= shell.altitude_km_max + 1.0


def _toy_shell_distribution() -> ShellDistribution:
    return ShellDistribution(
        altitude_km_mean=463.0,
        altitude_km_std=1.0,
        altitude_km_min=460.0,
        altitude_km_max=465.0,
        inclination_deg_mean=53.16,
        inclination_deg_std=0.02,
        inclination_deg_min=53.04,
        inclination_deg_max=53.17,
        eccentricity_mean=0.0002,
        eccentricity_std=0.0001,
        eccentricity_min=0.0,
        eccentricity_max=0.0008,
    )
