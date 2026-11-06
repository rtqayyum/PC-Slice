#!/usr/bin/env bash
#SBATCH --job-name=pcslice_comparison
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
# MaxTime is read at submission; no guessed wall time.
set -euo pipefail
cd "$PCSLICE_ROOT"
unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX PYTHONOPTIMIZE
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source envs/pcslice_v3/bin/activate
[[ "$SLURM_CPUS_PER_TASK" = 20 ]]
PCSLICE_RUN="$PWD/results/comparison_${SLURM_JOB_ID}"
srun --ntasks=1 --cpus-per-task=20 --gres=gpu:1 python -u -m src.surgical_v3_gate \
 --workers 20 --output-dir "$PCSLICE_RUN"
