#!/usr/bin/env bash
#SBATCH --job-name=oopsie-final
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=disable
unset PYOPENGL_PLATFORM || true
cd "$SAFE_PILOTS_PROJECT_ROOT"
"$SAFE_PILOTS_PROJECT_ROOT/.venv/bin/python" -m safe_diffusion_cl_pilots.pilots.aggregate \
  --project-root "$SAFE_PILOTS_PROJECT_ROOT" \
  --run-root "$SAFE_PILOTS_RUN_ROOT" \
  --seeds "$SAFE_PILOTS_SEEDS"
