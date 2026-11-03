"""Record the 12 before/after replay episodes (see orbit_guard/evaluation/replay_recording.py).

    uv run python scripts/record_replay_traces.py --runs-dir <RUNS> [--out ~/orbitguard_runs/replay_traces]
        [--overlay <tuned.yaml>] [--freeze-snapshot <RUNS>/experiments/frozen.json]

Needs the trained actors written by the N=15 trainings: $RUNS/n15_adaptive_s<k>/actor_weights.npz (shield on) and
$RUNS/n15_noshield_s<k>/actor_weights.npz (shield off), each with the run_config.json sidecar training writes.
--runs-dir has no default: name the runs dir of the final models (e.g. ~/orbitguard_runs_tuned), never the
retired ones. Light on CPU (about 20 s per episode, one core).
Pass --overlay with the tuned overlay the actors were trained with: a checkpoint with no run_config.json, or whose
sidecar records a different config_digest, is refused before any episode is recorded. With --freeze-snapshot, each
replayed actor must also be byte-identical (sha256) to the frozen checkpoint of the same model and training seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from orbit_guard.evaluation.experiment import CheckpointProvenanceError
from orbit_guard.evaluation.replay_recording import (
    DEFAULT_INIT_SEED,
    build_plan,
    check_plan_provenance,
    record_all,
)

_RUNS = Path.home() / "orbitguard_runs"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, required=True, help="runs dir holding n15_<variant>_s<k>/")
    parser.add_argument("--out", type=Path, default=_RUNS / "replay_traces")
    parser.add_argument("--agents", type=int, default=15)
    parser.add_argument("--trained-seed", type=int, default=0, help="which N=15 training seed supplies the trained actors")
    parser.add_argument("--init-seed", type=int, default=DEFAULT_INIT_SEED, help="init seed of the untrained actor")
    parser.add_argument("--overlay", type=Path, default=None, help="tuned overlay yaml the trained actors were trained with")
    parser.add_argument(
        "--freeze-snapshot", type=Path, default=None,
        help="frozen.json of the final experiment: replay only the frozen actors (sha256 must match)",
    )
    args = parser.parse_args(argv)
    plan = build_plan(args.runs_dir, trained_seed=args.trained_seed)
    missing = [str(item.weights_path) for item in plan if item.weights_path and not item.weights_path.is_file()]
    if missing:
        parser.error("missing trained actor weights:\n  " + "\n  ".join(sorted(set(missing))))
    try:
        check_plan_provenance(plan, overlay_path=args.overlay, freeze_snapshot=args.freeze_snapshot)
    except CheckpointProvenanceError as error:
        parser.error(str(error))
    paths = record_all(
        plan, args.out, num_agents=args.agents, init_seed=args.init_seed, overlay_path=args.overlay,
        freeze_snapshot=args.freeze_snapshot,
    )
    print(f"wrote {len(paths)} traces to {args.out}")


if __name__ == "__main__":
    main()
