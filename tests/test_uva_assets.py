from __future__ import annotations

import subprocess
from pathlib import Path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_uva_asset_files_exist() -> None:
    expected = [
        Path("scripts/uva/runtime_env.sh"),
        Path("scripts/uva/load_miniforge.sh"),
        Path("scripts/uva/setup_miniforge_env.sh"),
        Path("scripts/uva/bootstrap_home_checkout.sh"),
        Path("scripts/slurm/train_dfm.sbatch"),
        Path("scripts/slurm/infer_multi_agent.sbatch"),
        Path("scripts/slurm/evaluate_collision.sbatch"),
        Path("docs/deployment/uva_hpc.md"),
        Path("configs/experiments/planar_disk_dfm.yaml"),
    ]
    for path in expected:
        assert path.exists(), path


def test_shell_assets_are_parseable() -> None:
    for path in [
        Path("scripts/uva/runtime_env.sh"),
        Path("scripts/uva/load_miniforge.sh"),
        Path("scripts/uva/setup_miniforge_env.sh"),
        Path("scripts/uva/bootstrap_home_checkout.sh"),
        Path("scripts/slurm/train_dfm.sbatch"),
        Path("scripts/slurm/infer_multi_agent.sbatch"),
        Path("scripts/slurm/evaluate_collision.sbatch"),
    ]:
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_runtime_env_exports_canonical_roots() -> None:
    text = _read(Path("scripts/uva/runtime_env.sh"))
    assert ': "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before sourcing scripts/uva/runtime_env.sh.}"' in text
    assert 'export SMD_PROJECT_ROOT="/home/${CURRENT_USER}/${SMD_PROJECT_NAME}"' in text
    assert 'export SMD_DATA_ROOT="/scratch/${CURRENT_USER}/${SMD_PROJECT_NAME}/data"' in text
    assert 'export SMD_RUNS_ROOT="/scratch/${CURRENT_USER}/${SMD_PROJECT_NAME}/runs"' in text
    assert 'export SMD_ENV_ROOT="/home/${CURRENT_USER}/envs/${SMD_PROJECT_NAME}"' in text


def test_slurm_templates_require_explicit_project_name() -> None:
    for path in [
        Path("scripts/slurm/train_dfm.sbatch"),
        Path("scripts/slurm/infer_multi_agent.sbatch"),
        Path("scripts/slurm/evaluate_collision.sbatch"),
    ]:
        text = _read(path)
        assert "#SBATCH --export=NONE" in text
        assert ': "${SMD_PROJECT_NAME:?Set SMD_PROJECT_NAME before submitting this job.}"' in text
        assert 'source "/home/${CURRENT_USER}/${SMD_PROJECT_NAME}/scripts/uva/runtime_env.sh"' in text
        assert 'source "/home/${CURRENT_USER}/${SMD_PROJECT_NAME}/scripts/uva/load_miniforge.sh"' in text
        assert 'source activate "${SMD_ENV_ROOT}"' in text
