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
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
envs/pcslice_v3/bin/python - <<'CHECK'
import torch, transformers, scipy
from src.surgical_campaign import load_config, preflight_models
from src.prepare_surgical_sources import verify
assert torch.__version__.startswith('2.7.1'), torch.__version__
assert transformers.__version__ == '4.51.3', transformers.__version__
assert scipy.__version__ == '1.16.3', scipy.__version__
config=load_config('config/surgical_task_v3.json')
verify(config['source_root'])
preflight_models(config)
print('>> inference/solver imports, source hashes, checkpoints and tokenizer budgets verified')
CHECK
PCSLICE_ACTIVE=$(squeue -h -u "$(id -un)" -p gpuq -n pcslice_comparison -o '%i')
[[ -z "$PCSLICE_ACTIVE" ]] || { echo "Existing project job: $PCSLICE_ACTIVE; no duplicate submitted."; exit 0; }
echo ">> pre-flight ok: pcslice_comparison, 20 CPUs, 1 GPU, gpuq time=$PCSLICE_MAXTIME"
export PCSLICE_ROOT
sbatch --time="$PCSLICE_MAXTIME" --chdir="$PCSLICE_ROOT" \
  --output="$PCSLICE_ROOT/logs/pcslice_comparison_%j.out" \
  --error="$PCSLICE_ROOT/logs/pcslice_comparison_%j.err" hpc/run.slurm
