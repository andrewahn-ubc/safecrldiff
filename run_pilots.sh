#!/usr/bin/env bash
set -euo pipefail

launcher_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<EOF
Usage: bash run_pilots.sh [OPTIONS]

Defaults:
  --account       def-mijungp
  --project-root  directory containing this launcher
  --run-root      same as --project-root
  --seeds         0,1,2

Common options:
  --dry-run
  --force-from pilot-minus1|pilot0|pilots12|aggregate
  --account ACCOUNT
  --project-root PATH
  --run-root PATH
  --seeds 0,1,2
EOF
}

account="${SAFE_PILOTS_ACCOUNT:-def-mijungp}"
project_root="${SAFE_PILOTS_PROJECT_ROOT:-$launcher_root}"
run_root="${SAFE_PILOTS_RUN_ROOT:-}"
seeds="0,1,2"
dry_run=0
local_smoke=0
force_from=""
pm1_cpus=""
p0_cpus=""
gpu_cpus=""
aggregate_cpus=""
pm1_walltime=""
p0_walltime=""
gpu_walltime=""
aggregate_walltime=""
smoke_episodes_per_file="10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --account) account="$2"; shift 2 ;;
    --project-root) project_root="$2"; shift 2 ;;
    --run-root) run_root="$2"; shift 2 ;;
    --seeds) seeds="$2"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    --local-smoke) local_smoke=1; shift ;;
    --force-from) force_from="$2"; shift 2 ;;
    --pilot-minus1-cpus) pm1_cpus="$2"; shift 2 ;;
    --pilot0-cpus) p0_cpus="$2"; shift 2 ;;
    --gpu-cpus) gpu_cpus="$2"; shift 2 ;;
    --aggregate-cpus) aggregate_cpus="$2"; shift 2 ;;
    --pilot-minus1-walltime) pm1_walltime="$2"; shift 2 ;;
    --pilot0-walltime) p0_walltime="$2"; shift 2 ;;
    --gpu-walltime) gpu_walltime="$2"; shift 2 ;;
    --aggregate-walltime) aggregate_walltime="$2"; shift 2 ;;
    --smoke-episodes-per-file) smoke_episodes_per_file="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run_root="${run_root:-$project_root}"
project_root="$(cd "$project_root" && pwd)"
if [[ ! -f "$project_root/configs/narval.yaml" || ! -f "$project_root/pyproject.toml" ]]; then
  echo "--project-root is not a safecrldiff checkout: $project_root" >&2
  exit 2
fi
mkdir -p "$run_root" "$run_root/logs/slurm" "$run_root/results" "$run_root/artifacts"
run_root="$(cd "$run_root" && pwd)"
if [[ ! "$seeds" =~ ^-?[0-9]+(,-?[0-9]+)*$ ]]; then
  echo "--seeds must be comma-separated integers" >&2
  exit 2
fi
IFS=',' read -r -a seed_array <<< "$seeds"
seed_count="${#seed_array[@]}"
array_max="$((seed_count - 1))"

yaml_value() {
  awk -v section="$1" -v key="$2" '
    $0 == section ":" { inside=1; next }
    inside && /^[^ ]/ { inside=0 }
    inside && $1 == key ":" { gsub(/\"/, "", $2); print $2; exit }
  ' "$project_root/configs/narval.yaml"
}

pm1_cpus="${pm1_cpus:-$(yaml_value pilot_minus1 cpus)}"
p0_cpus="${p0_cpus:-$(yaml_value pilot0 cpus)}"
gpu_cpus="${gpu_cpus:-$(yaml_value pilots12 cpus)}"
aggregate_cpus="${aggregate_cpus:-$(yaml_value aggregate cpus)}"
pm1_walltime="${pm1_walltime:-$(yaml_value pilot_minus1 walltime)}"
p0_walltime="${p0_walltime:-$(yaml_value pilot0 walltime)}"
gpu_walltime="${gpu_walltime:-$(yaml_value pilots12 walltime)}"
aggregate_walltime="${aggregate_walltime:-$(yaml_value aggregate walltime)}"
array_concurrency="$(yaml_value pilots12 array_concurrency)"

if [[ $dry_run -eq 0 && $local_smoke -eq 0 ]]; then
  cluster_name="${SLURM_CLUSTER_NAME:-}"
  host_name="$(hostname -f 2>/dev/null || hostname)"
  if [[ "$cluster_name" != *narval* && "$host_name" != *narval* ]]; then
    echo "This command must run on Narval; use --local-smoke for an explicitly local smoke run." >&2
    exit 2
  fi
fi

export SAFE_PILOTS_PROJECT_ROOT="$project_root"
export SAFE_PILOTS_RUN_ROOT="$run_root"
export SAFE_PILOTS_SEEDS="$seeds"
export SAFE_PILOTS_SMOKE_EPISODES_PER_FILE="$smoke_episodes_per_file"
cd "$project_root"

echo "Account:      $account"
echo "Project root: $project_root"
echo "Run root:     $run_root"
echo "Seeds:        $seeds"

if [[ $dry_run -eq 1 ]]; then
  echo "sbatch --parsable --account $account --cpus-per-task $pm1_cpus --time $pm1_walltime --output $run_root/logs/slurm/%x-%j.out slurm/pilot_minus1_cpu.sbatch"
  echo "sbatch --parsable --account $account --cpus-per-task $p0_cpus --time $p0_walltime --kill-on-invalid-dep=yes --dependency afterok:<PILOT_MINUS1_JOB_ID> --output $run_root/logs/slurm/%x-%j.out slurm/pilot0_cpu.sbatch"
  echo "sbatch --parsable --account $account --cpus-per-task $gpu_cpus --time $gpu_walltime --array 0-${array_max}%${array_concurrency} --kill-on-invalid-dep=yes --dependency afterok:<PILOT0_JOB_ID> --output $run_root/logs/slurm/%x-%A_%a.out slurm/pilots12_gpu_array.sbatch"
  echo "sbatch --parsable --account $account --cpus-per-task $aggregate_cpus --time $aggregate_walltime --dependency afterany:<PILOT_MINUS1_JOB_ID>:<PILOT0_JOB_ID>:<PILOTS12_JOB_ID> --output $run_root/logs/slurm/%x-%j.out slurm/aggregate_cpu.sbatch"
  echo "Final report: $run_root/results/GO_NO_GO_REPORT.md"
  exit 0
fi

if [[ $dry_run -eq 0 ]]; then
  # Loading this on every invocation makes the environment inherited by Slurm
  # jobs self-contained even when an existing venv is reused from a fresh shell.
  if command -v module >/dev/null 2>&1; then
    module load python/3.10
  fi
  # Keep package/model/build caches off the small home quota. These variables
  # are exported through sbatch as well, so compute nodes remain home-quota safe.
  cache_root="$run_root/.cache"
  temporary_root="$run_root/.tmp"
  mkdir -p \
    "$cache_root/pip" \
    "$cache_root/huggingface" \
    "$cache_root/torch" \
    "$cache_root/rustup" \
    "$cache_root/cargo" \
    "$cache_root/matplotlib" \
    "$temporary_root"
  export PIP_CACHE_DIR="$cache_root/pip"
  export HF_HOME="$cache_root/huggingface"
  export TORCH_HOME="$cache_root/torch"
  export RUSTUP_HOME="$cache_root/rustup"
  export CARGO_HOME="$cache_root/cargo"
  export MPLCONFIGDIR="$cache_root/matplotlib"
  export XDG_CACHE_HOME="$cache_root"
  export TMPDIR="$temporary_root"
  export PYTHONNOUSERSITE=1
  export PIP_DISABLE_PIP_VERSION_CHECK=1
  # Bootstrap and preflight use low-dimensional simulator state only. Avoid
  # selecting GLFW (which needs a display) or EGL (which needs a GPU) on the
  # shared Narval login node. GPU Slurm jobs select EGL explicitly.
  if [[ $local_smoke -eq 0 ]]; then
    export MUJOCO_GL=disable
    unset PYOPENGL_PLATFORM || true
  fi
  if command -v diskusage_report >/dev/null 2>&1; then
    diskusage_report || true
  fi
  if [[ ! -x "$project_root/.venv/bin/python" ]]; then
    if ! command -v python3.10 >/dev/null 2>&1; then
      echo "Python 3.10 is required; load a Narval Python 3.10 module first." >&2
      exit 2
    fi
    python3.10 -m venv "$project_root/.venv"
  fi
  "$project_root/.venv/bin/python" scripts/bootstrap.py \
    --project-root "$project_root" --run-root "$run_root"
  "$project_root/.venv/bin/python" -m pytest -q
  if [[ -x "$project_root/.venv/bin/ruff" ]]; then
    "$project_root/.venv/bin/ruff" check .
  else
    echo "Ruff wheel unavailable on this platform; skipping cluster lint (source build forbidden)."
  fi
  "$project_root/.venv/bin/python" -m compileall -q src
  "$project_root/.venv/bin/python" scripts/preflight.py
  if [[ -n "$force_from" ]]; then
    "$project_root/.venv/bin/python" scripts/prepare_force.py \
      --run-root "$run_root" --from-stage "$force_from" --seeds "$seeds"
  fi
fi

if [[ $local_smoke -eq 1 ]]; then
  "$project_root/.venv/bin/python" -m safe_diffusion_cl_pilots.pilots.pilot_minus1 --project-root "$project_root" --run-root "$run_root" --seeds "$seeds"
  "$project_root/.venv/bin/python" -m safe_diffusion_cl_pilots.pilots.pilot0 --project-root "$project_root" --run-root "$run_root" --seeds "$seeds"
  echo "Local smoke stages finished; GPU Pilots 1–2 are intentionally not launched outside Narval."
  exit 0
fi

submit_or_print() {
  sbatch --parsable "$@"
}

pm1_id="COMPLETED"
if [[ ! -f "$run_root/results/pilot_minus1/_SUCCESS" || "$force_from" == "pilot-minus1" ]]; then
  pm1_id="$(submit_or_print --account "$account" --cpus-per-task "$pm1_cpus" --time "$pm1_walltime" --output "$run_root/logs/slurm/%x-%j.out" --error "$run_root/logs/slurm/%x-%j.err" slurm/pilot_minus1_cpu.sbatch)"
fi

p0_id="COMPLETED"
if [[ ! -f "$run_root/results/pilot0/_SUCCESS" || "$force_from" == "pilot-minus1" || "$force_from" == "pilot0" ]]; then
  p0_args=(--account "$account" --cpus-per-task "$p0_cpus" --time "$p0_walltime" --output "$run_root/logs/slurm/%x-%j.out" --error "$run_root/logs/slurm/%x-%j.err" --kill-on-invalid-dep=yes)
  if [[ "$pm1_id" != "COMPLETED" ]]; then
    p0_args+=(--dependency "afterok:$pm1_id")
  fi
  p0_id="$(submit_or_print "${p0_args[@]}" slurm/pilot0_cpu.sbatch)"
fi

gpu_complete=1
for seed in "${seed_array[@]}"; do
  if [[ ! -f "$run_root/results/seed_${seed}/pilot1/diffusion/_SUCCESS" || ! -f "$run_root/results/seed_${seed}/pilot2/gaussian/_SUCCESS" ]]; then
    gpu_complete=0
  fi
done
gpu_id="COMPLETED"
if [[ $gpu_complete -eq 0 || "$force_from" == "pilot-minus1" || "$force_from" == "pilot0" || "$force_from" == "pilots12" ]]; then
  gpu_args=(--account "$account" --cpus-per-task "$gpu_cpus" --time "$gpu_walltime" --output "$run_root/logs/slurm/%x-%A_%a.out" --error "$run_root/logs/slurm/%x-%A_%a.err" --array "0-${array_max}%${array_concurrency}" --kill-on-invalid-dep=yes)
  if [[ "$p0_id" != "COMPLETED" ]]; then
    gpu_args+=(--dependency "afterok:$p0_id")
  fi
  gpu_id="$(submit_or_print "${gpu_args[@]}" slurm/pilots12_gpu_array.sbatch)"
fi

aggregate_id="COMPLETED"
if [[ ! -f "$run_root/results/aggregate/_SUCCESS" || -n "$force_from" || "$pm1_id" != "COMPLETED" || "$p0_id" != "COMPLETED" || "$gpu_id" != "COMPLETED" ]]; then
  dependency_ids=()
  [[ "$pm1_id" != "COMPLETED" ]] && dependency_ids+=("$pm1_id")
  [[ "$p0_id" != "COMPLETED" ]] && dependency_ids+=("$p0_id")
  [[ "$gpu_id" != "COMPLETED" ]] && dependency_ids+=("$gpu_id")
  aggregate_args=(--account "$account" --cpus-per-task "$aggregate_cpus" --time "$aggregate_walltime" --output "$run_root/logs/slurm/%x-%j.out" --error "$run_root/logs/slurm/%x-%j.err")
  if [[ ${#dependency_ids[@]} -gt 0 ]]; then
    dependency_csv="$(IFS=:; echo "${dependency_ids[*]}")"
    aggregate_args+=(--dependency "afterany:$dependency_csv")
  fi
  aggregate_id="$(submit_or_print "${aggregate_args[@]}" slurm/aggregate_cpu.sbatch)"
fi

echo "Pilot -1 job: $pm1_id"
echo "Pilot 0 job: $p0_id"
echo "Pilots 1-2 array job: $gpu_id"
echo "Aggregation job: $aggregate_id"
echo "Pilot -1 report: $run_root/results/pilot_minus1/report.md"
echo "Final report: $run_root/results/GO_NO_GO_REPORT.md"
