#!/bin/bash

set -euo pipefail

: "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before running this script.}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/runtime_env.sh"

mkdir -p "${SMD_PROJECT_ROOT}"
rsync -av --exclude ".git/" "${SCRIPT_DIR}/../../" "${SMD_PROJECT_ROOT}/"
