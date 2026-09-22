"""TLE-derived constellation generator (PRD §16-17).

Loads the frozen CelesTrak Starlink 53-degree-shell snapshot (already
pre-parsed into orbital elements -- never a raw-TLE parser, never re-pulled),
fits per-column sampling distributions, and samples fresh synthetic
constellation instances of N=10-20 satellites per episode seed.

Per PRD §16, the snapshot is used only to derive realistic *sampling
distributions* -- no satellite is propagated from its real TLE epoch. Each
sampled satellite is seeded directly as a Cartesian state via
orbit_guard.physics.frames.coe_to_cartesian at a single simulator-internal
t=0 (see orbit_guard.physics.frames module docstring for the full frame-
convention rationale).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orbit_guard.physics.constants import R_EARTH_EQUATORIAL_M
from orbit_guard.physics.frames import coe_to_cartesian
from orbit_guard.physics.state import SatelliteCapability, SatelliteState

MIN_SUPPORTED_N = 10
MAX_SUPPORTED_N = 20

_NUMERIC_COLUMNS = (
    "norad_id",
    "epoch_year",
    "epoch_day",
    "inclination_deg",
    "raan_deg",
    "eccentricity",
    "arg_periapsis_deg",
    "mean_anomaly_deg",
    "mean_motion_rev_day",
    "semi_major_axis_km",
    "altitude_km",
)


def load_tle_snapshot(path: str | Path) -> dict[str, np.ndarray]:
    """Load the pre-parsed orbital-elements CSV. Raises if the file is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"TLE snapshot not found at {path}")

    columns: dict[str, list[float]] = {name: [] for name in _NUMERIC_COLUMNS}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for name in _NUMERIC_COLUMNS:
                columns[name].append(float(row[name]))

    return {name: np.array(values, dtype=np.float64) for name, values in columns.items()}


@dataclass(frozen=True)
class ShellDistribution:
    """Per-column marginal sampling distribution fit from the frozen snapshot (PRD §16)."""

    altitude_km_mean: float
    altitude_km_std: float
    altitude_km_min: float
    altitude_km_max: float
    inclination_deg_mean: float
    inclination_deg_std: float
    inclination_deg_min: float
    inclination_deg_max: float
    eccentricity_mean: float
    eccentricity_std: float
    eccentricity_min: float
    eccentricity_max: float


def fit_shell_distribution(tle_data: dict[str, np.ndarray]) -> ShellDistribution:
    """Fit marginal (mean/std/min/max) distributions directly from the loaded columns."""
    altitude = tle_data["altitude_km"]
    inclination = tle_data["inclination_deg"]
    eccentricity = tle_data["eccentricity"]

    return ShellDistribution(
        altitude_km_mean=float(np.mean(altitude)),
        altitude_km_std=float(np.std(altitude)),
        altitude_km_min=float(np.min(altitude)),
        altitude_km_max=float(np.max(altitude)),
        inclination_deg_mean=float(np.mean(inclination)),
        inclination_deg_std=float(np.std(inclination)),
        inclination_deg_min=float(np.min(inclination)),
        inclination_deg_max=float(np.max(inclination)),
        eccentricity_mean=float(np.mean(eccentricity)),
        eccentricity_std=float(np.std(eccentricity)),
        eccentricity_min=float(np.min(eccentricity)),
        eccentricity_max=float(np.max(eccentricity)),
    )


@dataclass(frozen=True)
class GeneratedSatellite:
    state: SatelliteState
    capability: SatelliteCapability


def sample_constellation(
    n: int,
    seed: int,
    shell: ShellDistribution,
    capability_template: SatelliteCapability,
) -> tuple[GeneratedSatellite, ...]:
    """Sample n fresh synthetic satellites (PRD §16-17).

    Altitude, inclination, and eccentricity are drawn from the fitted shell
    distribution and clipped to its observed [min, max] envelope, so no
    synthetic satellite falls outside the real shell's physical range. RAAN,
    argument of periapsis, and mean anomaly are drawn uniformly over
    [0, 2*pi) -- the snapshot itself shows RAAN spanning the full range
    (§16), and conjunction-specific phase correlation is layered on
    separately by the conjunction generator (Phase 2, §17), not here.
    """
    if not (MIN_SUPPORTED_N <= n <= MAX_SUPPORTED_N):
        raise ValueError(
            f"N={n} is outside the supported range [{MIN_SUPPORTED_N}, {MAX_SUPPORTED_N}]"
        )

    rng = np.random.default_rng(seed)
    satellites = []
    for _ in range(n):
        altitude_km = float(
            np.clip(
                rng.normal(shell.altitude_km_mean, shell.altitude_km_std),
                shell.altitude_km_min,
                shell.altitude_km_max,
            )
        )
        inclination_deg = float(
            np.clip(
                rng.normal(shell.inclination_deg_mean, shell.inclination_deg_std),
                shell.inclination_deg_min,
                shell.inclination_deg_max,
            )
        )
        eccentricity = float(
            np.clip(
                abs(rng.normal(shell.eccentricity_mean, shell.eccentricity_std)),
                shell.eccentricity_min,
                shell.eccentricity_max,
            )
        )
        raan_rad = float(rng.uniform(0.0, 2 * np.pi))
        arg_periapsis_rad = float(rng.uniform(0.0, 2 * np.pi))
        mean_anomaly_rad = float(rng.uniform(0.0, 2 * np.pi))

        semi_major_axis_m = R_EARTH_EQUATORIAL_M + altitude_km * 1000.0
        r, v = coe_to_cartesian(
            semi_major_axis_m,
            eccentricity,
            np.radians(inclination_deg),
            raan_rad,
            arg_periapsis_rad,
            mean_anomaly_rad,
        )

        satellites.append(
            GeneratedSatellite(
                state=SatelliteState(position_m=tuple(r), velocity_mps=tuple(v)),
                capability=capability_template,
            )
        )

    return tuple(satellites)
