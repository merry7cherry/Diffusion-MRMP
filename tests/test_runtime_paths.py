from __future__ import annotations

from pathlib import Path


def test_resolve_runtime_paths_builds_canonical_uva_roots(monkeypatch) -> None:
    from smd.runtime import RuntimePaths

    monkeypatch.setenv("USER", "tester")
    runtime = RuntimePaths.from_project_name("demo-project")

    assert runtime.project_name == "demo-project"
    assert runtime.project_root == Path("/home/tester/demo-project")
    assert runtime.data_root == Path("/scratch/tester/demo-project/data")
    assert runtime.runs_root == Path("/scratch/tester/demo-project/runs")
    assert runtime.env_root == Path("/home/tester/envs/demo-project")


def test_expand_runtime_path_resolves_placeholders_and_relative_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from smd.runtime import RuntimePaths, expand_runtime_path

    monkeypatch.setenv("USER", "tester")
    submit_dir = tmp_path / "submit"
    config_dir = tmp_path / "configs"
    submit_dir.mkdir()
    config_dir.mkdir()
    monkeypatch.chdir(submit_dir)

    runtime = RuntimePaths.from_project_name("demo-project")
    config_path = config_dir / "train.yaml"
    config_path.write_text("runtime:\n  project_name: demo-project\n", encoding="utf-8")

    assert expand_runtime_path(
        "{runs_root}/outputs/model_a",
        runtime=runtime,
        source_path=config_path,
    ) == Path("/scratch/tester/demo-project/runs/outputs/model_a")
    assert expand_runtime_path(
        "../artifacts/checkpoint.pt",
        runtime=runtime,
        source_path=config_path,
    ) == (config_dir.parent / "artifacts" / "checkpoint.pt").resolve()
