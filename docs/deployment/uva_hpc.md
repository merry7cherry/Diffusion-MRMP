# UVA HPC Deployment

This repository now treats canonical UVA HPC execution as the supported runtime model.

The fixed filesystem contract is:

- code: `/home/$USER/{project_name}`
- data: `/scratch/$USER/{project_name}/data`
- runs: `/scratch/$USER/{project_name}/runs`
- env: `/home/$USER/envs/{project_name}`

The public naming knob is `SMD_PROJECT_NAME`. All runtime roots are derived from it:

- `SMD_PROJECT_ROOT`
- `SMD_DATA_ROOT`
- `SMD_RUNS_ROOT`
- `SMD_ENV_ROOT`

## Recommended Workflow

1. Export the project name for the session.

```bash
export SMD_PROJECT_NAME=Diffusion-MRMP
```

2. Sync the repo into the canonical home checkout.

```bash
bash scripts/uva/bootstrap_home_checkout.sh
```

3. Create the Miniforge environment.

```bash
bash /home/$USER/${SMD_PROJECT_NAME}/scripts/uva/setup_miniforge_env.sh
```

4. Submit the shipped jobs with explicit export.

```bash
sbatch --export=SMD_PROJECT_NAME="${SMD_PROJECT_NAME}" /home/$USER/${SMD_PROJECT_NAME}/scripts/slurm/train_dfm.sbatch
sbatch --export=SMD_PROJECT_NAME="${SMD_PROJECT_NAME}" /home/$USER/${SMD_PROJECT_NAME}/scripts/slurm/infer_multi_agent.sbatch
sbatch --export=SMD_PROJECT_NAME="${SMD_PROJECT_NAME}" /home/$USER/${SMD_PROJECT_NAME}/scripts/slurm/evaluate_collision.sbatch
```

## Entry Points

- `python -m smd.tasks.train_dfm --config ...`
- `python -m smd.tasks.infer_multi_agent ...`
- `python -m smd.tasks.evaluate_collision --results-dir ...`

The Slurm templates call these module entrypoints directly and do not depend on the submission directory.
