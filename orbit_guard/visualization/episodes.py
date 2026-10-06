"""Discover recorded episodes from `trace.meta` (redesign real-data rules 3 and 4).

A directory of `.npz` traces becomes an `EpisodeIndex` keyed by (scenario_id, policy, shield). Only each file's
JSON header is read, so indexing is cheap; file names are never consulted. A combination with no trace is
reported as unavailable so the UI can disable that option ("no trace recorded") instead of inventing data.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from orbit_guard.visualization.replay_meta import (
    ReplayMeta,
    ReplayMetaError,
    validate_replay_meta,
)
from orbit_guard.visualization.summary import SummaryTotals, summary_totals

# Mirrors the key order of `shifts:` in configs/default.yaml (tests/evaluation/test_episodes.py guards drift).
CONDITION_ORDER = ("normal", "noise_increase", "density_increase", "simultaneous_conjunctions", "capability_reduction")
_POLICY_ORDER = ("untrained", "trained")
_CONDITION_LABELS = {"simultaneous_conjunctions": "Simultaneous conjunctions"}


class NoEpisodesError(ValueError):
    """The episode source is not a directory."""


def condition_label(condition: str) -> str:
    if condition in _CONDITION_LABELS:
        return _CONDITION_LABELS[condition]
    words = condition.replace("_", " ")
    return words[:1].upper() + words[1:]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    condition: str
    seed: int
    label: str


@dataclass(frozen=True)
class EpisodeEntry:
    path: Path
    meta: ReplayMeta


_MAX_HEADER_BYTES = 8 * 1024 * 1024  # a real header is a few kB; refuse anything absurd before parsing it


def _read_header(path: Path) -> dict:
    """The trace's JSON header; every failure becomes a `ReplayMetaError` naming the file."""
    try:
        with zipfile.ZipFile(path) as archive:  # the stored size is known before anything is decompressed
            stored = next((info.file_size for info in archive.infolist() if info.filename == "header.npy"), None)
        if stored is not None and stored > _MAX_HEADER_BYTES:
            raise ReplayMetaError(f"{path.name}: header is implausibly large ({stored} bytes)")
        with np.load(path, allow_pickle=False) as data:
            if "header" not in data.files:
                raise ReplayMetaError(f"{path.name}: not a replay trace (no header); keep weights and other arrays out of the traces directory")
            raw = data["header"]
            if raw.nbytes > _MAX_HEADER_BYTES:
                raise ReplayMetaError(f"{path.name}: header is implausibly large ({raw.nbytes} bytes)")
            header = json.loads(str(raw))
    except ReplayMetaError:
        raise
    except (ValueError, OSError, TypeError, zipfile.BadZipFile, EOFError) as error:
        raise ReplayMetaError(f"{path.name}: cannot read as a replay trace ({type(error).__name__}: {str(error)[:120]})") from error
    if not isinstance(header, dict):
        raise ReplayMetaError(f"{path.name}: header is not a JSON object")
    return header


def read_meta(path: Path) -> ReplayMeta:
    """Validated meta from a trace's header, naming the file on failure. Reads no arrays."""
    header = _read_header(path)
    try:
        return validate_replay_meta(header.get("meta", {}))
    except ReplayMetaError as error:
        raise ReplayMetaError(f"{path.name}: {error}") from error


def read_episode_summary(path: Path) -> SummaryTotals:
    """Recorded totals from a trace's header (for the BEFORE/AFTER strip bars). Reads no arrays."""
    header = _read_header(path)
    try:
        return summary_totals(header.get("summary", {}), validate_replay_meta(header.get("meta", {})))
    except ReplayMetaError as error:
        raise ReplayMetaError(f"{path.name}: {error}") from error


def _condition_rank(condition: str) -> int:
    return CONDITION_ORDER.index(condition) if condition in CONDITION_ORDER else len(CONDITION_ORDER)


class EpisodeIndex:
    def __init__(self, entries: tuple[EpisodeEntry, ...]):
        self.entries = entries
        self._by_key = {(e.meta.scenario_id, e.meta.policy, e.meta.shield): e for e in entries}

    def find(self, scenario_id: str, policy: str, shield: bool) -> EpisodeEntry | None:
        return self._by_key.get((scenario_id, policy, shield))

    def scenarios(self) -> tuple[Scenario, ...]:
        seen = {e.meta.scenario_id: e.meta for e in self.entries}
        ordered = sorted(seen.values(), key=lambda m: (_condition_rank(m.condition), m.condition, m.seed))
        return tuple(Scenario(m.scenario_id, m.condition, m.seed, condition_label(m.condition)) for m in ordered)

    def policies(self) -> tuple[str, ...]:
        """BEFORE and AFTER training are always offered (a missing one is disabled, EC-03); baselines only when recorded."""
        present = {e.meta.policy for e in self.entries}
        return tuple(_POLICY_ORDER) + tuple(sorted(present - set(_POLICY_ORDER)))

    def availability(self, scenario_id: str) -> dict[tuple[str, bool], bool]:
        return {
            (policy, shield): (scenario_id, policy, shield) in self._by_key
            for policy in self.policies()
            for shield in (True, False)
        }


def _build(paths: list[Path]) -> EpisodeIndex:
    entries: dict[tuple[str, str, bool], EpisodeEntry] = {}
    for path in paths:
        meta = read_meta(path)
        key = (meta.scenario_id, meta.policy, meta.shield)
        if key in entries:
            raise ReplayMetaError(f"duplicate episode {key}: {entries[key].path.name} and {path.name}")
        entries[key] = EpisodeEntry(path, meta)
    return EpisodeIndex(tuple(entries.values()))


def discover_episodes(directory: str | Path) -> EpisodeIndex:
    """Index every `.npz` trace in `directory` (an empty directory gives an empty index)."""
    root = Path(directory)
    if not root.is_dir():
        raise NoEpisodesError(f"{root} is not a directory")
    return _build(sorted(root.glob("*.npz")))


def index_from_trace(path: str | Path) -> EpisodeIndex:
    """Single-trace mode (`--trace FILE`): the same meta requirements apply."""
    return _build([Path(path)])
