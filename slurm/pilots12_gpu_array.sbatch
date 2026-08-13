#!/usr/bin/env bash
#SBATCH --job-name=oopsie-p12
#SBATCH --gpus=a100:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
cd "$SAFE_PILOTS_PROJECT_ROOT"
IFS=',' read -r -a pilot_seeds <<< "$SAFE_PILOTS_SEEDS"
seed="${pilot_seeds[$SLURM_ARRAY_TASK_ID]}"
if ! "$SAFE_PILOTS_PROJECT_ROOT/.venv/bin/python" scripts/gpu_gate.py \
  --run-root "$SAFE_PILOTS_RUN_ROOT" --seed "$seed"; then
  echo "Pilot 0 is red; seed $seed exits before importing Torch or initializing CUDA."
  exit 0
fi
"$SAFE_PILOTS_PROJECT_ROOT/.venv/bin/python" -m safe_diffusion_cl_pilots.pilots.seed_smoke \
  --run-root "$SAFE_PILOTS_RUN_ROOT" \
  --seed "$seed"
"$SAFE_PILOTS_PROJECT_ROOT/.venv/bin/python" -m safe_diffusion_cl_pilots.pilots.pilot1 \
  --project-root "$SAFE_PILOTS_PROJECT_ROOT" \
  --run-root "$SAFE_PILOTS_RUN_ROOT" \
  --seed "$seed" \
  --seeds "$SAFE_PILOTS_SEEDS"
"$SAFE_PILOTS_PROJECT_ROOT/.venv/bin/python" -m safe_diffusion_cl_pilots.pilots.pilot2 \
  --project-root "$SAFE_PILOTS_PROJECT_ROOT" \
  --run-root "$SAFE_PILOTS_RUN_ROOT" \
  --seed "$seed" \
  --seeds "$SAFE_PILOTS_SEEDS"
if [[ "$seed" == "0" ]]; then
  if ! "$SAFE_PILOTS_PROJECT_ROOT/.venv/bin/python" -m safe_diffusion_cl_pilots.pilots.postprocess_videos \
    --run-root "$SAFE_PILOTS_RUN_ROOT"; then
    echo "Diagnostic video postprocessing failed; scientific metrics and gates are unchanged." >&2
  fi
fi
