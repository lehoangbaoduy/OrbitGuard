#!/usr/bin/env bash
# Post-training pipeline (Phase 4 exit evidence -> Phase 5 sweep -> Phase 6 artifacts).
# Run AFTER the nine N=15 trainings (adaptive/fixed/noshield x seeds 0-2) have written
# $RUNS/n15_<variant>_s<seed>/actor_weights.npz. Usage: scripts/run_experiments.sh [RUNS_DIR] [OUT_DIR]
# RUNS_DIR defaults to $RUNS (the variable queue_n15.sh reads), then ~/orbitguard_runs; OUT_DIR to $OUT, then
# $RUNS/experiments. Relative paths are taken from the caller's directory.
# OVERLAY=<tuned.yaml> evaluates under the tuned overlay the models were trained with (CORE-1); it is passed
# to every experiment step, and a checkpoint whose run_config.json records another config digest is refused.
# Resumable: re-running after an interruption skips logged episodes, and an existing $OUT/frozen.json is not
# re-frozen (the freeze is one-way); the final-test run verifies it instead, so nothing can change in between.
set -euo pipefail
RUNS="${1:-${RUNS:-$HOME/orbitguard_runs}}"
OUT="${2:-${OUT:-$RUNS/experiments}}"
WORKERS="${WORKERS:-6}"
OVERLAY="${OVERLAY:-}"
# Absolute before the cd below, so relative inputs do not end up resolved against the repo root.
RUNS="$(realpath -m "$RUNS")"
OUT="$(realpath -m "$OUT")"
if [ -n "$OVERLAY" ]; then OVERLAY="$(realpath -m "$OVERLAY")"; fi
echo "== RUNS=$RUNS OUT=$OUT OVERLAY=${OVERLAY:-none}"
overlay_flags=()
if [ -n "$OVERLAY" ]; then overlay_flags=(--overlay "$OVERLAY"); fi
mkdir -p "$OUT"
cd "$(dirname "$0")/.."

uv run python -m orbit_guard.evaluation.experiment registry --runs-dir "$RUNS" --out "$OUT/registry.json" \
  ${overlay_flags[@]+"${overlay_flags[@]}"}

# 1. Validation-split pass (safe to look at; used for checkpoint sanity and tuning decisions).
uv run python -m orbit_guard.evaluation.experiment run --split validation --max-seeds 10 \
  --registry "$OUT/registry.json" --out "$OUT/validation_episodes.jsonl" --workers "$WORKERS" --experiment-id validation \
  ${overlay_flags[@]+"${overlay_flags[@]}"}

# 2. FREEZE configs + checkpoints, then (and only then) the final-test pool can be read. The freeze is one-way
#    (it refuses an existing $OUT/frozen.json) and refuses uncommitted changes under orbit_guard/, configs/,
#    scripts/ or data/: commit everything there first, including a tuned overlay kept under configs/. docs/ is
#    exempt. Every checkpoint must carry the run_config.json written by training, trained from a clean tree at
#    one commit with no change to orbit_guard/, configs/ or data/ since. If $OUT/frozen.json already exists (a
#    resumed sweep), it is kept: step 3's verify_freeze checks it against this registry, config and code.
[ -f configs/test_seeds.json ] || uv run python -c "from orbit_guard.evaluation.test_seed_pool import write_test_pool; write_test_pool('configs/test_seeds.json')"
if [ -f "$OUT/frozen.json" ] && ! python3 -c 'import json, sys; json.load(open(sys.argv[1]))' "$OUT/frozen.json" 2>/dev/null; then
  # A freeze killed mid-write. No test episode can exist yet (the test run verifies the snapshot first).
  echo "!! $OUT/frozen.json is not valid JSON (a torn freeze): delete it, note it in the run log, and re-run" >&2
  exit 1
elif [ -f "$OUT/frozen.json" ]; then
  echo "== $OUT/frozen.json exists: not re-freezing (one-way); the final-test run verifies it"
else
  uv run python -m orbit_guard.evaluation.experiment freeze --registry "$OUT/registry.json" --out "$OUT/frozen.json" \
    ${overlay_flags[@]+"${overlay_flags[@]}"}
fi

# 3. Final-test sweep: 5 models x 5 conditions x 10 seeds (x 3 training seeds for learned models).
uv run python -m orbit_guard.evaluation.experiment run --split test --freeze-snapshot "$OUT/frozen.json" \
  --registry "$OUT/registry.json" --out "$OUT/test_episodes.jsonl" --workers "$WORKERS" --experiment-id final \
  ${overlay_flags[@]+"${overlay_flags[@]}"}

# 4. Tables + figures + failure taxonomy, straight from the log.
uv run python -m orbit_guard.evaluation.results --episodes "$OUT/test_episodes.jsonl" --out-dir "$OUT/results"
