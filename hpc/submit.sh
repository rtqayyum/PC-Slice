#!/usr/bin/env bash
# Explicit manual submission only: one GPU and exactly 20 CPUs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results
PCSLICE_ROOT="$PWD"
PCSLICE_CONFIG=$(scontrol show partition gpuq -o)
[[ "$PCSLICE_CONFIG" =~ MaxTime=([^[:space:]]+) ]] || { echo 'Cannot read gpuq MaxTime' >&2; exit 2; }
PCSLICE_MAXTIME="${BASH_REMATCH[1]}"
[[ "$PCSLICE_MAXTIME" != INFINITE && "$PCSLICE_MAXTIME" != UNLIMITED ]] || PCSLICE_MAXTIME=0
[[ -x envs/pcslice_v3/bin/python ]] || { echo 'Prepare the environment in docs/REPRODUCIBILITY.md first.' >&2; exit 2; }
PCSLICE_ACTIVE=$(squeue -h -u "$(id -un)" -p gpuq -n pcslice_comparison -o '%i')
[[ -z "$PCSLICE_ACTIVE" ]] || { echo "Existing project job: $PCSLICE_ACTIVE; no duplicate submitted."; exit 0; }
echo ">> pre-flight ok: pcslice_comparison, 20 CPUs, 1 GPU, gpuq time=$PCSLICE_MAXTIME"
export PCSLICE_ROOT
sbatch --time="$PCSLICE_MAXTIME" --chdir="$PCSLICE_ROOT" \
  --output="$PCSLICE_ROOT/logs/pcslice_comparison_%j.out" \
  --error="$PCSLICE_ROOT/logs/pcslice_comparison_%j.err" hpc/run.slurm
