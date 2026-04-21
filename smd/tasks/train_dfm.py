from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import wandb
import yaml

from smd.losses import GaussianDiffusionLoss
from smd.runtime import RuntimePaths
from smd.trainer import get_dataset, get_model, get_summary, train
from smd.utils.loading import load_params_from_yaml


def _resolve_config(config_path: str | Path) -> tuple[dict, RuntimePaths]:
    payload = load_params_from_yaml(str(config_path))
    runtime = RuntimePaths.from_project_name(payload["runtime"]["project_name"])
    return payload, runtime


def train_from_config(config_path: str | Path) -> Path:
    payload, runtime = _resolve_config(config_path)
    dataset_cfg = dict(payload.get("dataset", {}))
    model_cfg = dict(payload.get("model", {}))
    training_cfg = dict(payload.get("training", {}))
    output_cfg = dict(payload.get("output", {}))

    model_id = str(output_cfg["model_id"])
    model_dir = runtime.trained_models_root / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    tensor_args = {
        "device": training_cfg.get("device", "cpu"),
        "dtype": torch.float32,
    }

    train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
        dataset_class=str(dataset_cfg.get("dataset_class", "TrajectoryDataset")),
        dataset_subdir=str(dataset_cfg["dataset_subdir"]),
        batch_size=int(dataset_cfg.get("batch_size", 32)),
        val_set_size=float(dataset_cfg.get("val_set_size", 0.05)),
        use_extra_objects=bool(dataset_cfg.get("use_extra_objects", True)),
        obstacle_cutoff_margin=float(dataset_cfg.get("obstacle_cutoff_margin", 0.01)),
        tensor_args=tensor_args,
    )

    dataset = train_subset.dataset
    model = get_model(
        model_class=str(model_cfg.get("model_class", "TrajectoryDFMModel")),
        tensor_args=tensor_args,
        state_dim=int(dataset.state_dim),
        velocity_model_class=str(model_cfg.get("velocity_model_class", "TrajectoryVelocityMLP")),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
        n_hidden_layers=int(model_cfg.get("n_hidden_layers", 2)),
        context_dim=int(model_cfg.get("context_dim", 0)),
        n_sampling_steps=int(model_cfg.get("n_sampling_steps", 16)),
        sigma_data=float(model_cfg.get("sigma_data", 1.0)),
        sigma_noise=float(model_cfg.get("sigma_noise", 1.0)),
    )

    args_payload = {
        "generator_backend": "dfm",
        "model_class": str(model_cfg.get("model_class", "TrajectoryDFMModel")),
        "velocity_model_class": str(model_cfg.get("velocity_model_class", "TrajectoryVelocityMLP")),
        "dataset_class": str(dataset_cfg.get("dataset_class", "TrajectoryDataset")),
        "dataset_subdir": str(dataset_cfg["dataset_subdir"]),
        "batch_size": int(dataset_cfg.get("batch_size", 32)),
        "val_set_size": float(dataset_cfg.get("val_set_size", 0.05)),
        "use_extra_objects": bool(dataset_cfg.get("use_extra_objects", True)),
        "obstacle_cutoff_margin": float(dataset_cfg.get("obstacle_cutoff_margin", 0.01)),
        "state_dim": int(dataset.state_dim),
        "n_sampling_steps": int(model_cfg.get("n_sampling_steps", 16)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 128)),
        "n_hidden_layers": int(model_cfg.get("n_hidden_layers", 2)),
        "context_dim": int(model_cfg.get("context_dim", 0)),
        "sigma_data": float(model_cfg.get("sigma_data", 1.0)),
        "sigma_noise": float(model_cfg.get("sigma_noise", 1.0)),
        "use_ema": bool(training_cfg.get("use_ema", False)),
        "ema_decay": float(training_cfg.get("ema_decay", 0.995)),
    }
    with (model_dir / "args.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(args_payload, handle, sort_keys=True)
    shutil.copyfile(str(config_path), model_dir / "train_config.yaml")

    loss_fn = GaussianDiffusionLoss().loss_fn
    summary_fn = None
    if training_cfg.get("summary_class"):
        summary_fn = get_summary(summary_class=str(training_cfg["summary_class"]))

    wandb.init(
        project="smd-dfm",
        name=model_id,
        mode=str(training_cfg.get("wandb_mode", "disabled")),
        reinit=True,
    )
    try:
        train(
            model=model,
            train_dataloader=train_dataloader,
            train_subset=train_subset,
            val_dataloader=val_dataloader,
            val_subset=val_subset,
            epochs=int(training_cfg.get("epochs", 1)),
            lr=float(training_cfg.get("lr", 1e-4)),
            steps_til_summary=int(training_cfg.get("steps_til_summary", 100)),
            steps_til_checkpoint=int(training_cfg.get("steps_til_checkpoint", 100)),
            model_dir=str(model_dir),
            loss_fn=loss_fn,
            val_loss_fn=loss_fn,
            summary_fn=summary_fn,
            clip_grad=bool(training_cfg.get("clip_grad", True)),
            clip_grad_max_norm=float(training_cfg.get("clip_grad_max_norm", 1.0)),
            use_ema=bool(training_cfg.get("use_ema", False)),
            ema_decay=float(training_cfg.get("ema_decay", 0.995)),
            update_ema_every=int(training_cfg.get("update_ema_every", 10)),
            use_amp=bool(training_cfg.get("use_amp", False)),
            debug=bool(training_cfg.get("debug", False)),
            tensor_args=tensor_args,
        )
    finally:
        wandb.finish()

    return model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a trajectory-native DFM generator.")
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
    args = parser.parse_args()

    model_dir = train_from_config(args.config)
    print(model_dir.resolve())


if __name__ == "__main__":
    main()
