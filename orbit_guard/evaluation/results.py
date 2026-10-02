"""Result tables and figures, generated from the episode log alone (PRD §36, §49; P6-M3/M4).

Nothing here is hand-typed: `write_results(records, out_dir)` reads logged
episode records and writes the markdown results table, the failure-taxonomy
table, and the primary-endpoint figure. Every episode can carry several of the
nine §36 labels; they are counted separately, never collapsed into one flag.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import plotly.graph_objects as go

from orbit_guard.evaluation.baselines import MODEL_NAMES
from orbit_guard.evaluation.experiment import format_report, summarize
from orbit_guard.evaluation.shifts import CONDITION_NAMES
from orbit_guard.rl import episode_metrics as em

ALL_FAILURE_LABELS = (
    em.LABEL_COLLISION,
    em.LABEL_UNSAFE_REMAINED,
    em.LABEL_SHIELD_REJECTED,
    em.LABEL_MISSION_COSTLY,
    em.LABEL_BUDGET_EXHAUSTED,
    em.LABEL_UNABLE_TO_MANEUVER,
    em.LABEL_NEW_CONJUNCTION,
    em.LABEL_WORSENED,
    em.LABEL_SIMULATION_FAILURE,
)
_PALETTE = ["#7d8ba1", "#ffb020", "#b794f6", "#3dd6e6", "#4ade80"]


def failure_taxonomy(records: Sequence[Mapping]) -> dict[str, dict[str, int]]:
    """Per model, the number of episodes carrying each failure label (all nine categories always present)."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        counts[record["model_name"]].update(set(record.get("failure_labels", ())))
    return {model: {label: counts[model][label] for label in ALL_FAILURE_LABELS} for model in counts}


def taxonomy_markdown(table: Mapping[str, Mapping[str, int]]) -> str:
    models = sorted(table, key=lambda m: (MODEL_NAMES.index(m) if m in MODEL_NAMES else len(MODEL_NAMES), m))
    lines = ["| failure mode (PRD §36) | " + " | ".join(models) + " |", "|---|" + "---:|" * len(models)]
    for label in ALL_FAILURE_LABELS:
        lines.append(f"| {label} | " + " | ".join(str(table[model][label]) for model in models) + " |")
    lines += ["", "Counts are episodes carrying the label; one episode may carry several."]
    return "\n".join(lines)


def collision_rate_figure(records: Sequence[Mapping]) -> go.Figure:
    """Primary endpoint: collision-episode rate by condition, one bar group per model, ±std across training seeds."""
    rows = summarize(records)
    figure = go.Figure()
    conditions = [c for c in CONDITION_NAMES if any(row["condition"] == c for row in rows)]
    for index, model in enumerate(m for m in MODEL_NAMES if any(row["model"] == m for row in rows)):
        by_condition = {row["condition"]: row["collision_rate"] for row in rows if row["model"] == model}
        figure.add_trace(
            go.Bar(
                name=model, x=conditions,
                y=[by_condition[c].mean if c in by_condition else None for c in conditions],
                error_y={"type": "data", "array": [by_condition[c].std if c in by_condition else 0.0 for c in conditions]},
                marker_color=_PALETTE[index % len(_PALETTE)],
            )
        )
    figure.update_layout(
        barmode="group", title="Collision-episode rate by condition (mean ± std across training seeds)",
        yaxis={"title": "episodes with a collision", "range": [0, 1]}, template="plotly_dark",
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def write_results(records: Sequence[Mapping], out_dir: str | Path) -> list[Path]:
    """Write the results table, failure taxonomy and primary-endpoint figure; return the paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = out / "results_table.md"
    taxonomy = out / "failure_taxonomy.md"
    figure = out / "collision_rate.html"
    table.write_text(format_report(records) + "\n")
    taxonomy.write_text(taxonomy_markdown(failure_taxonomy(records)) + "\n")
    collision_rate_figure(records).write_html(figure, include_plotlyjs="cdn")
    return [table, taxonomy, figure]


def main(argv: list[str] | None = None) -> None:
    import argparse

    from orbit_guard.evaluation.runner import read_records

    parser = argparse.ArgumentParser(description="Generate result tables/figures from an episode log.")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    for path in write_results(read_records(args.episodes), args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
