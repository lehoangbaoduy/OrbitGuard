"""run_config.json contents for test checkpoints: the digest plus the margin and reward weights training records.

Checkpoint provenance refuses a sidecar without a recorded margin and reward weights, so fixtures that stand in
for a real training run write the values the training CLI would have recorded from the same config + overlay.
(tests/evaluation/test_provenance_trained_values.py checks those values against literals.)
"""

from __future__ import annotations

from pathlib import Path

from orbit_guard.config import DEFAULT_CONFIG_PATH, config_digest, load_config
from orbit_guard.evaluation.trained_values import evaluation_values


def sidecar_record(
    overlay_path: str | Path | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    margin_kind: str = "adaptive",
) -> dict:
    """The provenance fields of a run trained from `config_path` + `overlay_path` with no CLI overrides."""
    return {
        "config_digest": config_digest(config_path, overlay_path),
        **evaluation_values(load_config(config_path, overlay_path), margin_kind),
    }
