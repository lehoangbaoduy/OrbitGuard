"""Reference collision note from an evaluation log (real-data rule 6).

The note is read from a JSONL log written by `orbit_guard.evaluation.runner`; nothing is typed in. With no log,
or no matching records, there is no note, and a model the log does not contain is never mentioned. Records are
matched on model, condition and (when given) fleet size, so a mixed log never describes the wrong population.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any


class ReferenceLogError(ValueError):
    """The reference log holds a line that is not a valid evaluation record."""


def _records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ReferenceLogError(f"{path.name}:{number}: not valid JSON ({error.msg})") from error
            if not isinstance(record, dict):
                raise ReferenceLogError(f"{path.name}:{number}: expected a JSON object per line")
            yield number, record


def _collided(path: Path, number: int, record: dict[str, Any]) -> bool:
    count = record.get("collision_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ReferenceLogError(f"{path.name}:{number}: collision_count must be a non-negative integer")
    return count > 0


def reference_note(log_path: str | Path | None, condition: str, models: Sequence[str], *, n_agents: int | None = None) -> str | None:
    if log_path is None:
        return None
    path = Path(log_path)
    if not path.is_file():
        raise FileNotFoundError(f"reference log {path} not found")
    wanted = set(models)
    tally: dict[str, list[bool]] = {model: [] for model in wanted}
    for number, record in _records(path):
        if record.get("model_name") not in wanted or record.get("shift_type") != condition or record.get("status") != "ok":
            continue
        if n_agents is not None and record.get("n_satellites") != n_agents:
            continue
        tally[record["model_name"]].append(_collided(path, number, record))
    parts = [f"{model} {sum(rows)}/{len(rows)}" for model in dict.fromkeys(models) if (rows := tally[model])]
    return " · ".join(parts) + " episodes with a collision" if parts else None
