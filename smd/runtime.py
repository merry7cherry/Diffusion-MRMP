from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _current_user() -> str:
    return os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"


@dataclass(frozen=True)
class RuntimePaths:
    project_name: str
    project_root: Path
    data_root: Path
    runs_root: Path
    env_root: Path

    @classmethod
    def from_project_name(cls, project_name: str, user: str | None = None) -> "RuntimePaths":
        if project_name is None or not str(project_name).strip():
            raise ValueError("project_name must be a non-empty string")
        active_user = user or _current_user()
        return cls(
            project_name=str(project_name),
            project_root=Path("/home") / active_user / str(project_name),
            data_root=Path("/scratch") / active_user / str(project_name) / "data",
            runs_root=Path("/scratch") / active_user / str(project_name) / "runs",
            env_root=Path("/home") / active_user / "envs" / str(project_name),
        )

    def as_placeholders(self) -> dict[str, str]:
        return {
            "project_name": self.project_name,
            "project_root": str(self.project_root),
            "data_root": str(self.data_root),
            "runs_root": str(self.runs_root),
            "env_root": str(self.env_root),
        }

    @property
    def trajectories_root(self) -> Path:
        return self.data_root / "trajectories"

    @property
    def instances_root(self) -> Path:
        return self.data_root / "instances"

    @property
    def init4proj_root(self) -> Path:
        return self.data_root / "init4proj"

    @property
    def trained_models_root(self) -> Path:
        return self.runs_root / "trained_models"

    @property
    def inference_root(self) -> Path:
        return self.runs_root / "inference"

    @property
    def evaluation_root(self) -> Path:
        return self.runs_root / "evaluation"


def require_project_name(env_var: str = "SMD_PROJECT_NAME") -> str:
    value = os.environ.get(env_var)
    if value is None or not value.strip():
        raise RuntimeError(f"Set {env_var} before using the canonical SMD runtime.")
    return value


def runtime_from_env(env_var: str = "SMD_PROJECT_NAME") -> RuntimePaths:
    return RuntimePaths.from_project_name(require_project_name(env_var))


def expand_runtime_path(
    value: str | os.PathLike[str],
    *,
    runtime: RuntimePaths,
    source_path: str | os.PathLike[str] | None = None,
) -> Path:
    raw = Path(os.path.expandvars(str(value).format(**runtime.as_placeholders()))).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    if source_path is None:
        return raw.resolve()
    source = Path(source_path)
    source_dir = source if source.is_dir() else source.parent
    return (source_dir / raw).resolve()
