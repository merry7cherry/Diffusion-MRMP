#!/bin/bash

set -euo pipefail

: "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before running this script.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_env.sh"
module purge
module load miniforge
source "${SCRIPT_DIR}/load_miniforge.sh"

ENV_ROOT="${SMD_ENV_ROOT}"
PROJECT_ROOT="${SMD_PROJECT_ROOT}"

conda create -y -p "${ENV_ROOT}" python=3.8.20
source activate "${ENV_ROOT}"
python -m pip install --upgrade pip
python -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e "${PROJECT_ROOT}"
python -m pip install pytest
