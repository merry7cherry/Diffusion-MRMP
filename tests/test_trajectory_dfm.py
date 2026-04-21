from __future__ import annotations

import torch


class _ZeroVelocity(torch.nn.Module):
    def forward(self, x, t, h, context=None):  # noqa: D401
        return torch.zeros_like(x)


def test_trajectory_dfm_loss_returns_scalar_and_metrics() -> None:
    from smd.models.trajectory_dfm import TrajectoryDFMModel

    model = TrajectoryDFMModel(
        velocity_model=_ZeroVelocity(),
        n_sampling_steps=4,
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
