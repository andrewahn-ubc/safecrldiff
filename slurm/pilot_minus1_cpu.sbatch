#!/usr/bin/env bash
#SBATCH --job-name=oopsie-pm1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd "$SAFE_PILOTS_PROJECT_ROOT"
"$SAFE_PILOTS_PROJECT_ROOT/.venv/bin/python" -m safe_diffusion_cl_pilots.pilots.pilot_minus1 \
  --project-root "$SAFE_PILOTS_PROJECT_ROOT" \
  --run-root "$SAFE_PILOTS_RUN_ROOT" \
  --seeds "$SAFE_PILOTS_SEEDS"

