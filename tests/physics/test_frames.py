import numpy as np
import pytest

from orbit_guard.physics.constants import MU_EARTH_M3_S2
from orbit_guard.physics.frames import (
    cartesian_to_coe,
    coe_to_cartesian,
    eci_vector_to_rtn,
    rtn_vector_to_eci,
    solve_kepler_equation,
)


def test_solve_kepler_equation_zero_eccentricity_gives_mean_equals_eccentric_anomaly():
    E = solve_kepler_equation(mean_anomaly_rad=1.234, eccentricity=0.0)
    assert E == pytest.approx(1.234)


def test_solve_kepler_equation_satisfies_keplers_equation():
    M = 2.1
    e = 0.15
    E = solve_kepler_equation(mean_anomaly_rad=M, eccentricity=e)
    assert E - e * np.sin(E) == pytest.approx(M, abs=1e-10)


@pytest.mark.parametrize("mean_anomaly_deg", [0.0, 45.0, 90.0, 180.0, 270.0])
def test_coe_to_cartesian_round_trips_through_cartesian_to_coe(mean_anomaly_deg):
    semi_major_axis_m = 6_838_588.0
    eccentricity = 0.0005
    inclination_rad = np.radians(53.06)
    raan_rad = np.radians(120.0)
    arg_periapsis_rad = np.radians(60.0)
    mean_anomaly_rad = np.radians(mean_anomaly_deg)

    r, v = coe_to_cartesian(
        semi_major_axis_m, eccentricity, inclination_rad, raan_rad, arg_periapsis_rad, mean_anomaly_rad
    )
    (
        a_out,
        e_out,
        i_out,
        raan_out,
        argp_out,
        M_out,
    ) = cartesian_to_coe(r, v)

    assert a_out == pytest.approx(semi_major_axis_m, rel=1e-8)
    assert e_out == pytest.approx(eccentricity, abs=1e-9)
    assert i_out == pytest.approx(inclination_rad, abs=1e-9)
    assert raan_out == pytest.approx(raan_rad, abs=1e-8)
    assert argp_out == pytest.approx(arg_periapsis_rad, abs=1e-6)
    assert M_out == pytest.approx(mean_anomaly_rad, abs=1e-6)


def test_coe_to_cartesian_circular_equatorial_like_orbit_matches_expected_radius():
    semi_major_axis_m = 7_000_000.0
    r, v = coe_to_cartesian(semi_major_axis_m, 0.0001, np.radians(53.0), 0.0, 0.0, 0.0)
    assert np.linalg.norm(r) == pytest.approx(semi_major_axis_m, rel=1e-3)
    # circular-orbit speed check
    expected_speed = np.sqrt(MU_EARTH_M3_S2 / semi_major_axis_m)
    assert np.linalg.norm(v) == pytest.approx(expected_speed, rel=1e-3)


def test_cartesian_to_coe_raises_on_near_equatorial_orbit():
    # Node vector is undefined at zero inclination -- must fail loudly, not silently.
    position = np.array([7_000_000.0, 0.0, 0.0])
    speed = np.sqrt(MU_EARTH_M3_S2 / 7_000_000.0)
    velocity = np.array([0.0, speed, 0.0])  # inclination == 0
    with pytest.raises(ValueError):
        cartesian_to_coe(position, velocity)


def test_cartesian_to_coe_raises_on_near_circular_orbit():
    # Eccentricity vector (and hence argument of periapsis) is undefined for
    # an exactly circular orbit -- must fail loudly, not silently.
    r, v = coe_to_cartesian(7_000_000.0, 0.0, np.radians(53.0), np.radians(10.0), 0.0, 0.0)
    with pytest.raises(ValueError):
        cartesian_to_coe(r, v)


def test_cartesian_to_coe_raises_on_near_parabolic_orbit():
    # Specific energy ~0 (escape velocity) makes semi-major axis undefined.
    # Velocity is inclined (not purely in the x-y plane) so this reaches the
    # energy check rather than tripping the earlier near-equatorial guard.
    radius_m = 7_000_000.0
    position = np.array([radius_m, 0.0, 0.0])
    escape_speed = np.sqrt(2 * MU_EARTH_M3_S2 / radius_m)
    velocity = escape_speed * np.array([0.0, np.cos(np.radians(30.0)), np.sin(np.radians(30.0))])
    with pytest.raises(ValueError):
        cartesian_to_coe(position, velocity)


def test_coe_to_cartesian_round_trips_with_arg_periapsis_past_180_degrees():
    # Exercises the e_vec[2] < 0 quadrant-disambiguation branch in
    # cartesian_to_coe, not reached by the primary round-trip test's argp=60.
    semi_major_axis_m = 6_838_588.0
    eccentricity = 0.0008
    inclination_rad = np.radians(53.06)
    raan_rad = np.radians(200.0)
    arg_periapsis_rad = np.radians(200.0)
    mean_anomaly_rad = np.radians(50.0)

    r, v = coe_to_cartesian(
        semi_major_axis_m, eccentricity, inclination_rad, raan_rad, arg_periapsis_rad, mean_anomaly_rad
    )
    a_out, e_out, i_out, raan_out, argp_out, M_out = cartesian_to_coe(r, v)

    assert a_out == pytest.approx(semi_major_axis_m, rel=1e-8)
    assert e_out == pytest.approx(eccentricity, abs=1e-9)
    assert i_out == pytest.approx(inclination_rad, abs=1e-9)
    assert raan_out == pytest.approx(raan_rad, abs=1e-8)
    assert argp_out == pytest.approx(arg_periapsis_rad, abs=1e-6)
    assert M_out == pytest.approx(mean_anomaly_rad, abs=1e-6)


def test_eci_to_rtn_basis_is_orthonormal_and_right_handed():
    position = np.array([6_838_588.0, 1_000_000.0, 500_000.0])
    velocity = np.array([-1000.0, 7200.0, 200.0])

    r_unit = eci_vector_to_rtn(position / np.linalg.norm(position), position, velocity)
    t_unit = eci_vector_to_rtn(velocity, position, velocity)

    # R basis vector expressed in RTN must be exactly [1, 0, 0].
    assert r_unit == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
    # Transverse direction has zero radial and normal component (planar, tangential).
    assert t_unit[2] == pytest.approx(0.0, abs=1e-6)


def test_eci_rtn_round_trip_is_identity():
    position = np.array([6_838_588.0, -2_100_000.0, 3_300_000.0])
    velocity = np.array([2200.0, 6800.0, -900.0])
    vec_eci = np.array([123.4, -56.7, 89.0])

    vec_rtn = eci_vector_to_rtn(vec_eci, position, velocity)
    vec_eci_restored = rtn_vector_to_eci(vec_rtn, position, velocity)

    assert vec_eci_restored == pytest.approx(vec_eci, abs=1e-9)


def test_rtn_radial_direction_matches_position_unit_vector():
    position = np.array([5_000_000.0, 3_000_000.0, 2_000_000.0])
    velocity = np.array([-3000.0, 5000.0, 1000.0])
    r_hat_expected = position / np.linalg.norm(position)

    r_hat_in_rtn = np.array([1.0, 0.0, 0.0])
    r_hat_in_eci = rtn_vector_to_eci(r_hat_in_rtn, position, velocity)

    assert r_hat_in_eci == pytest.approx(r_hat_expected, abs=1e-9)
