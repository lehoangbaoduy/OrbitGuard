"""Record the 12 before/after replay episodes (see orbit_guard/evaluation/replay_recording.py).

    uv run python scripts/record_replay_traces.py --runs-dir <RUNS> --out <fresh dir>
        [--overlay <tuned.yaml>] (--freeze-snapshot <OUT>/frozen.json | --unfrozen)

Needs the trained actors written by the N=15 trainings: $RUNS/n15_adaptive_s<k>/actor_weights.npz (shield on) and
$RUNS/n15_noshield_s<k>/actor_weights.npz (shield off), each with the run_config.json sidecar training writes.
--runs-dir has no default: name the runs dir of the final models (e.g. ~/orbitguard_runs_tuned), never the
retired ones. Light on CPU (about 20 s per episode, one core).
Pass --overlay with the tuned overlay the actors were trained with: a checkpoint with no run_config.json, or whose
sidecar records a different config_digest, is refused before any episode is recorded. Final traces need
--freeze-snapshot: the whole freeze is re-verified first (every checkpoint in --runs-dir, config + overlay, pools and
code, as the final-test run does), and each replayed actor must be byte-identical (sha256) to the frozen checkpoint
of the same model and training seed. --unfrozen records a preview without that check. --out must be empty or new,
so a partial re-recording can never mix with older traces.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from orbit_guard.config import DEFAULT_CONFIG_PATH
from orbit_guard.evaluation.experiment import CheckpointProvenanceError, build_registry
from orbit_guard.evaluation.freeze import FrozenConfigError, verify_freeze
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
    parser.add_argument("--unfrozen", action="store_true", help="record a preview without --freeze-snapshot")
    args = parser.parse_args(argv)
    if args.freeze_snapshot is None and not args.unfrozen:
        parser.error("final traces need --freeze-snapshot <OUT>/frozen.json (or pass --unfrozen for a preview)")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"--out {args.out} is not empty: record into a fresh directory")
    plan = build_plan(args.runs_dir, trained_seed=args.trained_seed)
    missing = [str(item.weights_path) for item in plan if item.weights_path and not item.weights_path.is_file()]
    if missing:
        parser.error("missing trained actor weights:\n  " + "\n  ".join(sorted(set(missing))))
    if args.freeze_snapshot is not None:
        checkpoints = {
            f"{model}/{seed}": weights for model, seeds in build_registry(args.runs_dir).items()
            for seed, weights in seeds.items()
        }
        try:
            verify_freeze(args.freeze_snapshot, checkpoints, DEFAULT_CONFIG_PATH, args.overlay)
        except FrozenConfigError as error:
            parser.error(f"the freeze does not verify: {error}")
    try:
        check_plan_provenance(
            plan, overlay_path=args.overlay, freeze_snapshot=args.freeze_snapshot, num_agents=args.agents
        )
    except CheckpointProvenanceError as error:
        parser.error(str(error))
    paths = record_all(
        plan, args.out, num_agents=args.agents, init_seed=args.init_seed, overlay_path=args.overlay,
        freeze_snapshot=args.freeze_snapshot,
    )
    print(f"wrote {len(paths)} traces to {args.out}")


if __name__ == "__main__":
    main()
