from __future__ import annotations

import os
from pathlib import Path

import torch

from smd.trainer.train_loaders import get_model
from smd.utils.loading import load_params_from_yaml


def resolve_model_checkpoint(model_dir: str | os.PathLike[str]) -> Path:
    checkpoints_dir = Path(model_dir) / "checkpoints"
    preferred = [
        checkpoints_dir / "ema_model_current_state_dict.pth",
        checkpoints_dir / "model_current_state_dict.pth",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing DFM checkpoint under {checkpoints_dir}")


def load_dfm_training_args(model_dir: str | os.PathLike[str]) -> dict:
    args_path = Path(model_dir) / "args.yaml"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing training args: {args_path}")
    return load_params_from_yaml(str(args_path))


def load_dfm_model_from_model_dir(model_dir: str | os.PathLike[str], *, tensor_args: dict) -> torch.nn.Module:
    args = load_dfm_training_args(model_dir)
    if args.get("generator_backend") != "dfm":
        raise ValueError(f"Unsupported generator backend in {model_dir}: {args.get('generator_backend')!r}")

    model = get_model(
        model_class=args["model_class"],
        tensor_args=tensor_args,
        state_dim=int(args["state_dim"]),
        velocity_model_class=str(args.get("velocity_model_class", "TrajectoryVelocityMLP")),
        hidden_dim=int(args.get("hidden_dim", 128)),
        n_hidden_layers=int(args.get("n_hidden_layers", 2)),
        context_dim=int(args.get("context_dim", 0)),
        n_sampling_steps=int(args.get("n_sampling_steps", 16)),
        sigma_data=float(args.get("sigma_data", 1.0)),
        sigma_noise=float(args.get("sigma_noise", 1.0)),
    )
    checkpoint_path = resolve_model_checkpoint(model_dir)
    model.load_state_dict(torch.load(checkpoint_path, map_location=tensor_args["device"]))
    model.eval()
    return model
