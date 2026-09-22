import numpy as np
import pytest

from orbit_guard.physics.constants import J2, MU_EARTH_M3_S2, R_EARTH_EQUATORIAL_M
from orbit_guard.physics.dynamics import (
    j2_acceleration,
    state_derivative,
    total_acceleration,
    two_body_acceleration,
    two_body_state_derivative,
)


@pytest.mark.parametrize(
    "position_m",
    [
        (7000000.0, 0.0, 0.0),
        (0.0, 6900000.0, 0.0),
        (4000000.0, 4000000.0, 3000000.0),
    ],
)
def test_two_body_acceleration_matches_inverse_square_law(position_m):
    position = np.array(position_m)
    accel = two_body_acceleration(position)
    r = np.linalg.norm(position)

    expected_magnitude = MU_EARTH_M3_S2 / r**2
    assert np.linalg.norm(accel) == pytest.approx(expected_magnitude, rel=1e-12)

    # Acceleration must point exactly toward Earth's center (anti-parallel to r).
    r_hat = position / r
    accel_hat = accel / np.linalg.norm(accel)
    assert accel_hat == pytest.approx(-r_hat, abs=1e-12)


def test_two_body_acceleration_raises_on_zero_position():
    with pytest.raises(ValueError):
        two_body_acceleration(np.array([0.0, 0.0, 0.0]))


def test_two_body_acceleration_raises_on_non_finite_position():
    with pytest.raises(ValueError):
        two_body_acceleration(np.array([np.nan, 0.0, 0.0]))


def test_j2_acceleration_has_no_out_of_plane_component_on_the_equatorial_plane():
    position = np.array([7000000.0, 1000000.0, 0.0])
    accel = j2_acceleration(position)
    assert accel[2] == pytest.approx(0.0, abs=1e-15)


def test_j2_acceleration_matches_independently_written_reference_formula():
    # Independent re-derivation of the standard closed-form J2 acceleration
    # (Vallado / Curtis), written separately from orbit_guard.physics.dynamics
    # so this test can catch a transcription bug in the implementation.
    position = np.array([4200000.0, -3100000.0, 5300000.0])
    x, y, z = position
    r = np.linalg.norm(position)
    coeff = -1.5 * J2 * MU_EARTH_M3_S2 * R_EARTH_EQUATORIAL_M**2 / r**5
    z_over_r_sq = (z / r) ** 2
    expected = np.array(
        [
            coeff * x * (1 - 5 * z_over_r_sq),
            coeff * y * (1 - 5 * z_over_r_sq),
            coeff * z * (3 - 5 * z_over_r_sq),
        ]
    )
    accel = j2_acceleration(position)
    assert accel == pytest.approx(expected, rel=1e-12)


def test_j2_acceleration_at_pole_has_no_radial_in_plane_component():
    # By symmetry, directly over the pole the J2 perturbation has zero x/y component.
    position = np.array([0.0, 0.0, 6900000.0])
    accel = j2_acceleration(position)
    assert accel[0] == pytest.approx(0.0, abs=1e-15)
    assert accel[1] == pytest.approx(0.0, abs=1e-15)


def test_total_acceleration_is_sum_of_two_body_and_j2():
    position = np.array([6900000.0, 200000.0, 100000.0])
    total = total_acceleration(position)
    expected = two_body_acceleration(position) + j2_acceleration(position)
    assert total == pytest.approx(expected, rel=1e-12)


def test_j2_acceleration_much_smaller_than_two_body_at_leo_altitude():
    # Sanity bound: J2 is a perturbation, not a dominant term, at LEO altitudes.
    position = np.array([6838588.0, 0.0, 0.0])
    two_body = np.linalg.norm(two_body_acceleration(position))
    j2 = np.linalg.norm(j2_acceleration(position))
    assert j2 / two_body < 1e-2
    assert j2 / two_body > 1e-5


def test_state_derivative_returns_velocity_followed_by_acceleration():
    position = np.array([6838588.0, 0.0, 0.0])
    velocity = np.array([0.0, 7600.0, 0.0])
    state_vector = np.concatenate([position, velocity])

    derivative = state_derivative(state_vector)

    assert derivative[0:3] == pytest.approx(velocity)
    assert derivative[3:6] == pytest.approx(total_acceleration(position))


def test_state_derivative_raises_loudly_on_non_finite_state():
    bad_state = np.array([np.inf, 0.0, 0.0, 0.0, 7600.0, 0.0])
    with pytest.raises((ValueError, FloatingPointError)):
        state_derivative(bad_state)


def test_two_body_state_derivative_excludes_j2_term():
    position = np.array([6838588.0, 0.0, 500000.0])
    velocity = np.array([0.0, 7600.0, 0.0])
    state_vector = np.concatenate([position, velocity])

    derivative = two_body_state_derivative(state_vector)

    assert derivative[0:3] == pytest.approx(velocity)
    assert derivative[3:6] == pytest.approx(two_body_acceleration(position))
    # Must differ from the full (two-body + J2) derivative off the equatorial plane.
    assert derivative[3:6] != pytest.approx(total_acceleration(position))


def test_two_body_state_derivative_raises_loudly_on_non_finite_state():
    bad_state = np.array([np.nan, 0.0, 0.0, 0.0, 7600.0, 0.0])
    with pytest.raises((ValueError, FloatingPointError)):
        two_body_state_derivative(bad_state)
