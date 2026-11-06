#!/usr/bin/env bash
#SBATCH --job-name=pcslice-scheduler
#SBATCH --partition=longq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
set -euo pipefail
cd "${PCSLICE_ROOT:?Set PCSLICE_ROOT at submission}"
unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source envs/pcslice_v3/bin/activate
export PCSLICE_MODE="${PCSLICE_MODE:-pilot}"
case "$PCSLICE_MODE" in pilot|development) ;; *) echo "Bad mode: $PCSLICE_MODE" >&2; exit 2;; esac
OUTPUT="$PCSLICE_ROOT/results/scheduler_${PCSLICE_MODE}_${SLURM_JOB_ID}"
echo "START mode=$PCSLICE_MODE output=$OUTPUT commit=$(git rev-parse HEAD 2>/dev/null || echo unavailable)"
srun --ntasks=1 --cpus-per-task=4 python -u -m src.surgical_scheduler_experiments --mode "$PCSLICE_MODE" --output-dir "$OUTPUT"
