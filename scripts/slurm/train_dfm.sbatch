#!/bin/bash
#SBATCH -A <allocation>
#SBATCH --partition=gpu
#SBATCH --constraint=a40|a6000|a100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/smd_train_dfm_%j.bootstrap.out
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
LOG_PATH="${SMD_RUNS_ROOT}/slurm/train_dfm_${SLURM_JOB_ID:-manual}.out"
exec > >(tee -a "${LOG_PATH}") 2>&1

export PYTHONPATH="${SMD_PROJECT_ROOT}:${PYTHONPATH:-}"
CONFIG_PATH="${SMD_CONFIG:-${SMD_PROJECT_ROOT}/configs/experiments/planar_disk_dfm.yaml}"

python -m smd.tasks.train_dfm --config "${CONFIG_PATH}"
