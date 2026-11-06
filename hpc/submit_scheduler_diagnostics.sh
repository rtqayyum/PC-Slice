#!/usr/bin/env bash
# Run from the repository root: bash hpc/submit_scheduler_diagnostics.sh pilot
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-pilot}"
case "$MODE" in pilot|development) ;; *) echo "Use pilot or development" >&2; exit 2;; esac
[[ -x envs/pcslice_v3/bin/python ]] || { echo "Missing envs/pcslice_v3; see docs/REPRODUCIBILITY.md" >&2; exit 2; }
mkdir -p logs results
export PCSLICE_ROOT="$PWD" PCSLICE_MODE="$MODE"
# Prevent duplicate submissions for the same user, project and diagnostic job name.
EXISTING="$(squeue -h -u "$(id -un)" -n pcslice-scheduler -o '%i' 2>/dev/null || true)"
[[ -z "$EXISTING" ]] || { echo "Scheduler job already active: $EXISTING"; exit 2; }
PARTITION="$(scontrol show partition longq -o)"
[[ "$PARTITION" =~ MaxTime=([^[:space:]]+) ]] || { echo "Unable to read longq time limit"; exit 2; }
MAXTIME="${BASH_REMATCH[1]}"
TIME="04:00:00"
if [[ "$MODE" == development ]]; then TIME="1-00:00:00"; fi
# If the requested time is above partition MaxTime, Slurm may reject it.
echo "Requested time=$TIME; longq MaxTime=$MAXTIME; 4 CPUs, 0 GPUs"
sbatch --time="$TIME" --output="$PCSLICE_ROOT/logs/pcslice_scheduler_%j.out" \
 --error="$PCSLICE_ROOT/logs/pcslice_scheduler_%j.err" \
 hpc/run_scheduler_diagnostics.slurm
