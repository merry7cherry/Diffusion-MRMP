from __future__ import annotations

import types

import torch


class _ZeroVelocity(torch.nn.Module):
    def forward(self, x, t, h, context=None):  # noqa: D401
        return torch.zeros_like(x)


class _ContextRecordingVelocity(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_context = None

    def forward(self, x, t, h, context=None):  # noqa: D401
        self.last_context = None if context is None else context.detach().clone()
        return torch.zeros_like(x)


class _OutputRecordingVelocity(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)
        self.last_output = None

    def forward(self, x, t, h, context=None):  # noqa: D401
        del t, h, context
        self.last_output = torch.full_like(x, self.value)
        return self.last_output


def test_sample_grouped_timesteps_share_values_within_group() -> None:
    from smd.models.trajectory_dfm import sample_grouped_timesteps

    t, r = sample_grouped_timesteps(
        batch_size=8,
        groups_per_batch=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        P_mean_t=-1.0,
        P_std_t=2.5,
        P_mean_r=1.0,
        P_std_r=2.5,
    )

    assert t.shape == (8, 1)
    assert r.shape == (8, 1)
    assert torch.all(t <= r)
    assert torch.allclose(t[0:2], t[0:1].expand_as(t[0:2]))
    assert torch.allclose(t[2:4], t[2:3].expand_as(t[2:4]))
    assert torch.allclose(r[4:6], r[4:5].expand_as(r[4:6]))
    assert torch.allclose(r[6:8], r[6:7].expand_as(r[6:8]))


def test_compute_split_v0_drift_is_finite_and_shape_stable() -> None:
    from smd.models.trajectory_dfm import compute_split_v0_drift

    gen = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 1.0]],
            [[0.5, -0.5], [1.5, 0.5]],
        ],
        dtype=torch.float32,
    )
    pos = gen + 2.0

    drift = compute_split_v0_drift(
        gen,
        pos,
        kernel_temp_pos=1.0,
        kernel_temp_neg=1.0,
        num_sinkhorn_iters=8,
    )

    assert drift.shape == gen.shape
    assert torch.isfinite(drift).all()


def test_gaussian_diffusion_loss_passes_task_tensor_context_to_dfm_model() -> None:
    from smd.losses.gaussian_diffusion_loss import GaussianDiffusionLoss
    from smd.models.trajectory_dfm import TrajectoryDFMModel

    velocity = _ContextRecordingVelocity()
    model = TrajectoryDFMModel(
        velocity_model=velocity,
        n_sampling_steps=4,
        context_dim=4,
        groups_per_batch=2,
    )

    class DatasetStub:
        field_key_traj = "traj"
        field_key_task = "task"

    input_dict = {
        "traj_normalized": torch.randn(2, 5, 2),
        "task_normalized": torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
            ],
            dtype=torch.float32,
        ),
        "hard_conds": {
            0: torch.zeros(2, 2),
            4: torch.ones(2, 2),
        },
    }

    GaussianDiffusionLoss.loss_fn(model, input_dict, DatasetStub())

    assert velocity.last_context is not None
    assert torch.allclose(velocity.last_context, input_dict["task_normalized"])


def test_trajectory_dfm_loss_does_not_clamp_velocity_tensor_at_endpoints() -> None:
    from smd.models.trajectory_dfm import TrajectoryDFMModel

    velocity = _OutputRecordingVelocity(7.0)
    model = TrajectoryDFMModel(
        velocity_model=velocity,
        n_sampling_steps=4,
    )

    x = torch.randn(4, 6, 2)
    hard_conds = {
        0: torch.zeros(4, 2),
        5: torch.ones(4, 2),
    }

    model.loss(x, context=None, hard_conds=hard_conds)

    assert velocity.last_output is not None
    assert torch.allclose(velocity.last_output[:, 0, :], torch.full((4, 2), 7.0))
    assert torch.allclose(velocity.last_output[:, -1, :], torch.full((4, 2), 7.0))


def test_trajectory_dfm_loss_returns_scalar_and_metrics() -> None:
    from smd.models.trajectory_dfm import TrajectoryDFMModel

    model = TrajectoryDFMModel(
        velocity_model=_ZeroVelocity(),
        n_sampling_steps=4,
        groups_per_batch=3,
    )

    x = torch.randn(3, 6, 2)
    hard_conds = {
        0: torch.zeros(3, 2),
        5: torch.ones(3, 2),
    }

    loss, info = model.loss(x, context=None, hard_conds=hard_conds)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "dfm_loss" in info


def test_trajectory_dfm_run_inference_applies_hard_conditions_and_hooks() -> None:
    from smd.models.trajectory_dfm import TrajectoryDFMModel

    def projection_fn(x, projection_info, hard_conds, first_projection, init_traj4proj, proj_params):
        projected = x.clone()
        projected[:, 1:-1, :] = 0.5
        return projected, 1.0

    class ShiftGuide:
        def __call__(self, x):
            return torch.full_like(x, 0.25)

    model = TrajectoryDFMModel(
        velocity_model=_ZeroVelocity(),
        n_sampling_steps=3,
        projection_fn=projection_fn,
    )

    chain = model.run_inference(
        context=None,
        hard_conds={0: torch.zeros(2), 4: torch.ones(2)},
        n_samples=2,
        horizon=5,
        return_chain=True,
        guide=ShiftGuide(),
        n_guide_steps=1,
        dataset=object(),
        init_traj4proj=None,
        proj_params={"projection_step": [1]},
    )

    assert chain.shape == (4, 2, 5, 2)
    assert torch.allclose(chain[:, :, 0, :], torch.zeros_like(chain[:, :, 0, :]))
    assert torch.allclose(chain[:, :, -1, :], torch.ones_like(chain[:, :, -1, :]))
    assert torch.allclose(chain[-1, :, 1:-1, :], torch.full_like(chain[-1, :, 1:-1, :], 0.5))


def test_trajectory_dfm_run_local_inference_uses_seed_trajectory() -> None:
    from smd.models.trajectory_dfm import TrajectoryDFMModel

    model = TrajectoryDFMModel(
        velocity_model=_ZeroVelocity(),
        n_sampling_steps=2,
    )

    seed = torch.full((2, 4, 2), 0.75)
    chain = model.run_local_inference(
        seed_trajectory_b=seed,
        n_noising_steps=1,
        n_denoising_steps=2,
        context=None,
        hard_conds={0: torch.zeros(2), 3: torch.ones(2)},
        n_samples=2,
        return_chain=True,
    )

    assert chain.shape == (3, 2, 4, 2)
    assert torch.allclose(chain[:, :, 0, :], torch.zeros_like(chain[:, :, 0, :]))
    assert torch.allclose(chain[:, :, -1, :], torch.ones_like(chain[:, :, -1, :]))


def test_trajectory_dataset_skips_instance_bound_env_for_training(monkeypatch, tmp_path) -> None:
    from smd.datasets import trajectories as trajectories_module

    class UnexpectedEnv:
        def __init__(self, *args, **kwargs):
            raise AssertionError("environment construction should be skipped without instance context")

    class UnexpectedTask:
        def __init__(self, *args, **kwargs):
            raise AssertionError("planning task should be skipped without instance context")

    class UnexpectedVisualizer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("visualizer should be skipped without instance context")

    class FakeRobot:
        def __init__(self, tensor_args=None):
            self.tensor_args = tensor_args

        def get_position(self, x):
            return x

    def fake_load_params(path: str):
        if path.endswith("metadata.yaml"):
            return {
                "env_id": "EnvEmptyNoWait2D",
                "robot_id": "RobotCompositeThreePlanarDisk",
            }
        return {
            "threshold_start_goal_pos": 1,
        }

    def fake_load_trajectories(self):
        self.fields[self.field_key_traj] = torch.tensor(
            [[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]]],
            dtype=torch.float32,
        )
        self.fields[self.field_key_task] = torch.tensor(
            [[0.0, 0.0, 1.5, 1.5]],
            dtype=torch.float32,
        )
        self.map_task_id_to_trajectories_id[0] = torch.tensor([0])
        self.map_trajectory_id_to_task_id[0] = 0

    monkeypatch.setattr(trajectories_module, "_dataset_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(trajectories_module, "load_params_from_yaml", fake_load_params)
    monkeypatch.setattr(
        trajectories_module,
        "environments",
        types.SimpleNamespace(
            EnvEmptyNoWait2DExtraObjects=UnexpectedEnv,
            EnvEmptyNoWait2D=UnexpectedEnv,
        ),
    )
    monkeypatch.setattr(
        trajectories_module,
        "robots",
        types.SimpleNamespace(RobotCompositeThreePlanarDisk=FakeRobot),
    )
    monkeypatch.setattr(trajectories_module, "PlanningTask", UnexpectedTask)
    monkeypatch.setattr(trajectories_module, "PlanningVisualizer", UnexpectedVisualizer)
    monkeypatch.setattr(
        trajectories_module.TrajectoryDatasetBase,
        "load_trajectories",
        fake_load_trajectories,
    )

    dataset = trajectories_module.TrajectoryDataset(
        dataset_subdir="demo",
        tensor_args={"device": "cpu"},
    )

    sample = dataset[0]

    assert dataset.env is None
    assert dataset.task is None
    assert dataset.planner_visualizer is None
    assert sample["traj_normalized"].shape == (4, 2)
    assert 0 in sample["hard_conds"]
    assert 3 in sample["hard_conds"]


def test_merge_dataset_loader_kwargs_overrides_training_args() -> None:
    from smd.trainer.train_loaders import merge_dataset_loader_kwargs

    tensor_args = {"device": "cpu"}
    merged = merge_dataset_loader_kwargs(
        {
            "dataset_class": "FromArgs",
            "use_extra_objects": False,
            "obstacle_cutoff_margin": 0.5,
            "dataset_subdir": "demo-set",
        },
        dataset_class="TrajectoryDataset",
        use_extra_objects=True,
        obstacle_cutoff_margin=0.01,
        tensor_args=tensor_args,
        instance_idx=7,
        map_name="instances_simple",
    )

    assert merged["dataset_class"] == "TrajectoryDataset"
    assert merged["use_extra_objects"] is True
    assert merged["obstacle_cutoff_margin"] == 0.01
    assert merged["dataset_subdir"] == "demo-set"
    assert merged["tensor_args"] is tensor_args
    assert merged["instance_idx"] == 7
    assert merged["map_name"] == "instances_simple"


def test_planner_modules_import_dataset_loader_merge_helper() -> None:
    import smd.planners.multi_agent.smd_composite as smd_composite_module
    import smd.planners.single_agent.mpd as mpd_module
    import smd.planners.single_agent.mpd_ensemble as mpd_ensemble_module

    assert callable(smd_composite_module.merge_dataset_loader_kwargs)
    assert callable(mpd_module.merge_dataset_loader_kwargs)
    assert callable(mpd_ensemble_module.merge_dataset_loader_kwargs)


def test_projection_layout_uses_position_only_multi_agent_state() -> None:
    from smd.projection.projection import _resolve_projection_layout

    class FakeRobot:
        n_agents = 3

    class FakeProjectionInfo:
        robot = FakeRobot()

    x = torch.zeros((1, 64, 6))
    hard_conds = {
        0: torch.zeros((1, 6)),
        63: torch.ones((1, 6)),
    }

    start, goal, num_agents, horizons, traj_index = _resolve_projection_layout(
        x,
        FakeProjectionInfo(),
        hard_conds,
    )

    assert start.shape == (6,)
    assert goal.shape == (6,)
    assert num_agents == 3
    assert horizons == 64
    assert traj_index.tolist() == list(range(64))


def test_evaluate_collision_infers_num_agents_from_result_dir() -> None:
    from pathlib import Path

    import numpy as np

    from smd.tasks.evaluate_collision import infer_num_agents

    path_file = Path(
        "/tmp/runs/inference/demo/instance_name___Example/num_agents___3/planner___SMDComposite/0/paths.npy"
    )
    map_info = {"map_name": "instances_simple", "instance_idx": 0}
    paths_data = np.zeros((64, 64, 6), dtype=float)

    assert infer_num_agents(path_file, map_info, paths_data) == 3
