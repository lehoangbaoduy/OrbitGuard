#!/usr/bin/env bash
# Post-training pipeline (Phase 4 exit evidence -> Phase 5 sweep -> Phase 6 artifacts).
# Run AFTER the nine N=15 trainings (adaptive/fixed/noshield x seeds 0-2) have written
# $RUNS/n15_<variant>_s<seed>/actor_weights.npz. Usage: scripts/run_experiments.sh [RUNS_DIR] [OUT_DIR]
# OVERLAY=<tuned.yaml> evaluates under the tuned overlay the models were trained with (CORE-1); it is passed
# to every experiment step, and a checkpoint whose run_config.json records another config digest is refused.
set -euo pipefail
RUNS="${1:-$HOME/orbitguard_runs}"
OUT="${2:-$RUNS/experiments}"
WORKERS="${WORKERS:-6}"
OVERLAY="${OVERLAY:-}"
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

# 2. FREEZE configs + checkpoints, then (and only then) the final-test pool can be read.
[ -f configs/test_seeds.json ] || uv run python -c "from orbit_guard.evaluation.test_seed_pool import write_test_pool; write_test_pool('configs/test_seeds.json')"
uv run python -m orbit_guard.evaluation.experiment freeze --registry "$OUT/registry.json" --out "$OUT/frozen.json" \
  ${overlay_flags[@]+"${overlay_flags[@]}"}

# 3. Final-test sweep: 5 models x 5 conditions x 10 seeds (x 3 training seeds for learned models).
uv run python -m orbit_guard.evaluation.experiment run --split test --freeze-snapshot "$OUT/frozen.json" \
  --registry "$OUT/registry.json" --out "$OUT/test_episodes.jsonl" --workers "$WORKERS" --experiment-id final \
  ${overlay_flags[@]+"${overlay_flags[@]}"}

# 4. Tables + figures + failure taxonomy, straight from the log.
uv run python -m orbit_guard.evaluation.results --episodes "$OUT/test_episodes.jsonl" --out-dir "$OUT/results"
