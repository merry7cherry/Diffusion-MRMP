# Simultaneous Multi-Robot Motion Planning with Projected DFM

**[Paper](https://arxiv.org/pdf/2502.03607)** | **[Project Page](https://multi-robot-constrained-diffusion.github.io/)** | **[arXiv](https://arxiv.org/abs/2502.03607)**

This repository now keeps the multi-robot planner, projection, collision checking, and evaluation stack from `Diffusion-MRMP`, but replaces the trajectory generator with a trajectory-native Drift Flow Matching backend.



## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Citation](#citation)



## Overview

This repository contains the official implementation of our method for **simultaneous multi-robot motion planning with projected diffusion models**. Our approach enables efficient and collision-free path planning for multiple robots operating in shared environments.

### Key Features

- **Scalable**: Handles 9 robots simultaneously
- **Efficient**: Leverages diffusion models with projection-based constraints
- **Flexible**: Works across various environment configurations (empty, basic, dense, rooms, shelf.)
- **Generalizable**: Handles unseen scenarios



## Installation

### Canonical UVA Runtime

The supported runtime model is canonical UVA HPC execution.

- code: `/home/$USER/{project_name}`
- data: `/scratch/$USER/{project_name}/data`
- runs: `/scratch/$USER/{project_name}/runs`
- env: `/home/$USER/envs/{project_name}`

The public naming knob is `SMD_PROJECT_NAME`. All runtime roots are derived from it.

### Setup Instructions

1. Export the project name:

```bash
export SMD_PROJECT_NAME=Diffusion-MRMP
```

2. Sync the repo into the canonical home checkout:

```bash
bash scripts/uva/bootstrap_home_checkout.sh
```

3. Create the Miniforge environment:

```bash
bash /home/$USER/${SMD_PROJECT_NAME}/scripts/uva/setup_miniforge_env.sh
```

4. Submit the shipped jobs with explicit export:

```bash
sbatch --export=SMD_PROJECT_NAME="${SMD_PROJECT_NAME}" /home/$USER/${SMD_PROJECT_NAME}/scripts/slurm/train_dfm.sbatch
sbatch --export=SMD_PROJECT_NAME="${SMD_PROJECT_NAME}" /home/$USER/${SMD_PROJECT_NAME}/scripts/slurm/infer_multi_agent.sbatch
sbatch --export=SMD_PROJECT_NAME="${SMD_PROJECT_NAME}" /home/$USER/${SMD_PROJECT_NAME}/scripts/slurm/evaluate_collision.sbatch
```

See [docs/deployment/uva_hpc.md](/Users/cherryma/Desktop/NeurIPS%202026/Diffusion-MRMP/docs/deployment/uva_hpc.md) for the canonical deployment contract.

## Quick Start

### Train a DFM trajectory generator

```bash
python -m smd.tasks.train_dfm --config configs/experiments/planar_disk_dfm.yaml
```

The training task writes its resolved artifacts to:

- `{SMD_RUNS_ROOT}/trained_models/{model_id}`

### Run multi-agent inference

```bash
python -m smd.tasks.infer_multi_agent --start-index 0 --end-index 1
```

Inference outputs are written under `{SMD_RUNS_ROOT}/inference`.

### Evaluate collision feasibility

```bash
python -m smd.tasks.evaluate_collision --results-dir /scratch/$USER/${SMD_PROJECT_NAME}/runs/inference
```



## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{liang2025simultaneous,
  author    = {Liang, Jinhao and Christopher, Jacob K. and Koenig, Sven and Fioretto, Ferdinando},
  title     = {Simultaneous Multi-Robot Motion Planning with Projected Diffusion Models},
  journal   = {Forty-second International Conference on Machine Learning},
  year      = {2025},
}
```



## Acknowledgments

This codebase is built upon [MMD (Multi-Robot Motion Planning with Diffusion Models)](https://github.com/yoraish/mmd) by Shaoul et al. We thank the authors for providing a robust foundation for multi-robot motion planning research.

We also acknowledge the contributions from [MPD (Motion Planning Diffusion: Learning and Planning of Robot Motions with Diffusion Models)](https://github.com/jacarvalho/mpd-public) for foundational work on motion planning with diffusion models.



## Contact

For questions or issues regarding this implementation, please contact:

**Jinhao Liang**  

Email: [jliang@email.virginia.edu](mailto:jliang@email.virginia.edu)

Alternatively, feel free to open an issue on GitHub.
