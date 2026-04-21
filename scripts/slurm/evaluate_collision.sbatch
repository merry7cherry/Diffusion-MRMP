#!/bin/bash
#SBATCH -A <allocation>
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/smd_evaluate_collision_%j.bootstrap.out
#SBATCH --export=NONE

set -euo pipefail

: "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before submitting this job.}"
CURRENT_USER="${USER:-$(id -un)}"
source "/home/${CURRENT_USER}/${SMD_PROJECT_NAME}/scripts/uva/runtime_env.sh"
module purge
module load miniforge
source "/home/${CURRENT_USER}/${SMD_PROJECT_NAME}/scripts/uva/load_miniforge.sh"
source activate "${SMD_ENV_ROOT}"

mkdir -p "${SMD_DATA_ROOT}" "${SMD_RUNS_ROOT}" "${SMD_RUNS_ROOT}/slurm"
LOG_PATH="${SMD_RUNS_ROOT}/slurm/evaluate_collision_${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "${LOG_PATH}") 2>&1

export PYTHONPATH="${SMD_PROJECT_ROOT}:${PYTHONPATH:-}"
RESULTS_DIR="${SMD_RESULTS_DIR:-${SMD_RUNS_ROOT}/inference}"

python -m smd.tasks.evaluate_collision --results-dir "${RESULTS_DIR}"
