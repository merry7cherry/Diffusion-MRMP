#!/bin/bash

unset CONDA_DEFAULT_ENV
unset CONDA_PREFIX
unset CONDA_PROMPT_MODIFIER
unset CONDA_SHLVL

if [[ -f "/usr/share/miniforge/etc/profile.d/conda.sh" ]]; then
  source "/usr/share/miniforge/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
