from __future__ import annotations

from pathlib import Path


def test_official_runtime_files_do_not_use_legacy_relative_roots() -> None:
    files = [
        Path("smd/datasets/trajectories.py"),
        Path("smd/config/smd_experiment_configs.py"),
        Path("scripts/inference/inference_multi_agent.py"),
        Path("scripts/inference/launch_smd_composite_experiment.py"),
        Path("is_collision.py"),
    ]
    banned_fragments = [
        "data_trajectories",
        "data_trained_models",
        "instances_data",
        "init4proj_data",
        "results_test",
        "git.Repo(",
        "../../",
    ]
    for path in files:
        text = path.read_text(encoding="utf-8")
        for fragment in banned_fragments:
            assert fragment not in text, f"{fragment!r} found in {path}"
