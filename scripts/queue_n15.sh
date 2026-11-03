#!/usr/bin/env bash
# Resumable queue of the nine N=15 trainings (adaptive / fixed-margin / no-shield x seeds 0-2).
# Skips any run that already has actor_weights.npz, so it is safe to re-launch after an
# interruption or reboot. Logs live on disk under $RUNS/logs (NOT /tmp, which is a RAM tmpfs).
#   monitor:  tail -f ~/orbitguard_runs/logs/queue_n15.log
#   stop:     kill $(cat ~/orbitguard_runs/logs/queue_n15.pid)   # then stop the python training child
#   then:     scripts/run_experiments.sh
# Retraining with a tuned grid point goes ONLY through the frozen, hashed overlay (tuning.grid_search writes
# tuned.yaml, with the Fixed-Margin constant re-measured for tuned margin coefficients; every run gets
# --overlay "$OVERLAY", so the fixed arm trains with it too), into a fresh runs dir:
#   RUNS=~/orbitguard_runs_tuned OVERLAY=~/orbitguard_runs/grid/tuned.yaml scripts/queue_n15.sh
# Never pass reward weights (--w1..--w4, --collision-penalty) or margin coefficients (--margin-alpha/-beta/
# -gamma-margin) through TUNED_FLAGS: evaluation rebuilds the env from config + overlay only, and the experiment
# and replay recorder refuse a checkpoint whose run_config.json records other values. TUNED_FLAGS is deprecated
# for tuning; it remains only for training-only settings that never reach the evaluation env, e.g.
#   TUNED_FLAGS="--entropy-coeff 0.01"   (prefer the overlay's rl: block, which config_digest records)
set -uo pipefail
RUNS="${RUNS:-$HOME/orbitguard_runs}"
ITERATIONS="${ITERATIONS:-50}"
TUNED_FLAGS="${TUNED_FLAGS:-}"  # deprecated: extra training-only flags for every run (word-split on purpose); see above
OVERLAY="${OVERLAY:-}"          # optional tuned overlay yaml, merged over configs/default.yaml for every run
overlay_flags=()
if [ -n "$OVERLAY" ]; then overlay_flags=(--overlay "$OVERLAY"); fi
mkdir -p "$RUNS/logs"
echo $$ > "$RUNS/logs/queue_n15.pid"
export RAY_TMPDIR="$RUNS/ray_tmp"
cd "$(dirname "$0")/.."

for seed in 0 1 2; do
  for variant in adaptive fixed noshield; do
    run="$RUNS/n15_${variant}_s${seed}"
    if [ -f "$run/actor_weights.npz" ]; then echo "== skip $variant seed $seed (done)"; continue; fi
    case $variant in
      adaptive) flags="--shield --margin adaptive" ;;
      fixed)    flags="--shield --margin fixed" ;;
      noshield) flags="--no-shield --margin adaptive" ;;
    esac
    rm -rf "$run"  # a partial checkpoint from an interrupted run
    echo "== N=15 $variant seed $seed $(date +%T)"
    uv run python -u -m orbit_guard.rl.training --agents 15 --iterations "$ITERATIONS" --num-env-runners 6 \
      --seed "$seed" $flags $TUNED_FLAGS ${overlay_flags[@]+"${overlay_flags[@]}"} --eval-episodes 2 --eval-seed 5000 --checkpoint-dir "$run" \
      > "$RUNS/logs/n15_${variant}_s${seed}.log" 2>&1 \
      || echo "!! FAILED $variant seed $seed (see logs/n15_${variant}_s${seed}.log)"
  done
done
echo "== ALL N15 DONE $(date +%T)"
