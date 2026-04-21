from __future__ import annotations

from copy import copy

import einops
import torch
import torch.nn as nn


def apply_hard_conditioning(x: torch.Tensor, conditions: dict[int, torch.Tensor] | None) -> torch.Tensor:
    if conditions is None:
        return x
    for t, value in conditions.items():
        x[:, t, :] = value.clone()
    return x


def guide_gradient_steps(
    x: torch.Tensor,
    *,
    hard_conds: dict[int, torch.Tensor] | None = None,
    guide=None,
    n_guide_steps: int = 1,
) -> torch.Tensor:
    if guide is None:
        return x
    for _ in range(n_guide_steps):
        x = x + guide(x)
        x = apply_hard_conditioning(x, hard_conds)
    return x


class TrajectoryVelocityMLP(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        hidden_dim: int = 128,
        n_hidden_layers: int = 2,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.context_dim = int(context_dim)

        input_dim = self.state_dim + 2 + self.context_dim
        layer_dims = [input_dim, *([hidden_dim] * n_hidden_layers), self.state_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != self.state_dim:
                layers.append(nn.SiLU())
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, horizon, _ = x.shape
        t_features = t.reshape(batch_size, 1, -1).expand(batch_size, horizon, -1)
        h_features = h.reshape(batch_size, 1, -1).expand(batch_size, horizon, -1)
        pieces = [x, t_features, h_features]
        if context is not None:
            if context.ndim == 2:
                context = context.unsqueeze(1).expand(batch_size, horizon, -1)
            elif context.ndim != 3:
                raise ValueError(f"context must have shape [B, C] or [B, H, C], got {tuple(context.shape)}")
            pieces.append(context)
        return self.network(torch.cat(pieces, dim=-1))


class TrajectoryDFMModel(nn.Module):
    def __init__(
        self,
        *,
        velocity_model: nn.Module | None = None,
        velocity_model_class: str = "TrajectoryVelocityMLP",
        n_sampling_steps: int = 16,
        sigma_data: float = 1.0,
        sigma_noise: float = 1.0,
        projection_fn=None,
        context_model=None,
        **kwargs,
    ) -> None:
        super().__init__()
        velocity_kwargs = dict(kwargs)
        self.state_dim = int(kwargs.get("state_dim", getattr(velocity_model, "state_dim", 2)))
        if velocity_model is None:
            velocity_kwargs.setdefault("state_dim", self.state_dim)
            velocity_model = globals()[velocity_model_class](**velocity_kwargs)
        self.velocity_model = velocity_model
        self.n_sampling_steps = int(n_sampling_steps)
        self.sigma_data = float(sigma_data)
        self.sigma_noise = float(sigma_noise)
        self.projection_fn = projection_fn
        self.context_model = context_model
        self.submodules: dict[str, nn.Module] = {}

    def _prepare_hard_conds(
        self,
        hard_conds: dict[int, torch.Tensor] | None,
        *,
        n_samples: int,
        device: torch.device,
    ) -> dict[int, torch.Tensor]:
        prepared: dict[int, torch.Tensor] = {}
        if hard_conds is None:
            return prepared
        for key, value in copy(hard_conds).items():
            tensor = value.to(device=device, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = einops.repeat(tensor, "d -> b d", b=n_samples)
            elif tensor.ndim == 2 and tensor.shape[0] == 1 and n_samples > 1:
                tensor = tensor.expand(n_samples, -1)
            elif tensor.ndim != 2 or tensor.shape[0] != n_samples:
                raise ValueError(
                    f"hard condition {key} must have shape [D] or [B, D] with B={n_samples}, got {tuple(tensor.shape)}"
                )
            prepared[int(key)] = tensor
        return prepared

    def _prepare_context(self, context, *, n_samples: int):
        if context is None:
            return None
        if self.context_model is not None:
            context = self.context_model(context)
        if isinstance(context, torch.Tensor):
            if context.ndim == 1:
                context = einops.repeat(context, "d -> b d", b=n_samples)
            elif context.ndim == 2 and context.shape[0] == 1 and n_samples > 1:
                context = context.expand(n_samples, -1)
        return context

    def warmup(self, horizon: int = 64, device: str | torch.device = "cpu") -> None:
        sample = torch.randn((2, horizon, self.state_dim), device=device)
        t = torch.zeros((2, 1), device=device)
        h = torch.full((2, 1), 1.0 / max(1, self.n_sampling_steps), device=device)
        self.velocity_model(sample, t, h, context=None)

    def loss(self, x: torch.Tensor, context=None, hard_conds=None, *args):
        batch_size = x.shape[0]
        device = x.device
        prepared_hard_conds = self._prepare_hard_conds(hard_conds, n_samples=batch_size, device=device)
        prepared_context = self._prepare_context(context, n_samples=batch_size)

        x_target = apply_hard_conditioning(x.clone(), prepared_hard_conds)
        x0 = torch.randn_like(x_target) * self.sigma_noise
        t = torch.rand((batch_size, 1), device=device, dtype=x.dtype)
        r = torch.rand((batch_size, 1), device=device, dtype=x.dtype)
        t, r = torch.minimum(t, r), torch.maximum(t, r)
        h = r - t

        t_view = t.unsqueeze(-1)
        x_t = (1.0 - t_view) * x0 + t_view * x_target
        x_t = apply_hard_conditioning(x_t, prepared_hard_conds)
        target_velocity = x_target - x0
        predicted_velocity = self.velocity_model(x_t, t, h, prepared_context)
        predicted_velocity = apply_hard_conditioning(predicted_velocity, prepared_hard_conds)

        loss = torch.mean((predicted_velocity - target_velocity) ** 2)
        info = {
            "dfm_loss": float(loss.detach().item()),
            "mean_horizon": float(h.detach().mean().item()),
        }
        return loss, info

    def _projection_callback(self):
        if self.projection_fn is not None:
            return self.projection_fn
        from smd.projection.projection import apply_projection_alm

        return apply_projection_alm

    def _sample_rollout(
        self,
        *,
        x: torch.Tensor,
        context,
        hard_conds: dict[int, torch.Tensor] | None,
        num_steps: int,
        return_chain: bool,
        guide=None,
        n_guide_steps: int = 1,
        dataset=None,
        init_traj4proj=None,
        proj_params=None,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        prepared_context = self._prepare_context(context, n_samples=batch_size)
        prepared_hard_conds = self._prepare_hard_conds(hard_conds, n_samples=batch_size, device=x.device)

        x = apply_hard_conditioning(x, prepared_hard_conds)
        chain = [x.clone()] if return_chain else None

        projection_steps = set((proj_params or {}).get("projection_step", []))
        projection_enabled = self.projection_fn is not None or proj_params is not None
        projection_callback = self._projection_callback() if projection_enabled else None

        for step_idx in range(num_steps):
            t_value = float(step_idx) / float(max(1, num_steps))
            r_value = float(step_idx + 1) / float(max(1, num_steps))
            t = torch.full((batch_size, 1), t_value, device=x.device, dtype=x.dtype)
            r = torch.full((batch_size, 1), r_value, device=x.device, dtype=x.dtype)
            h = r - t
            h_view = h.unsqueeze(-1)

            velocity = self.velocity_model(x, t, h, prepared_context)
            x = x + h_view * velocity
            x = apply_hard_conditioning(x, prepared_hard_conds)
            x = guide_gradient_steps(
                x,
                hard_conds=prepared_hard_conds,
                guide=guide,
                n_guide_steps=n_guide_steps,
            )

            remaining_steps = num_steps - step_idx - 1
            should_project = projection_enabled and (remaining_steps in projection_steps or remaining_steps == 0)
            if should_project and projection_callback is not None:
                first_projection = 1 if step_idx == 0 else 0
                x, _ = projection_callback(
                    x,
                    dataset,
                    prepared_hard_conds,
                    first_projection,
                    init_traj4proj,
                    proj_params or {},
                )
                x = apply_hard_conditioning(x, prepared_hard_conds)

            if return_chain:
                chain.append(x.clone())

        if return_chain:
            return torch.stack(chain, dim=0)
        return x

    @torch.no_grad()
    def run_inference(self, context=None, hard_conds=None, n_samples: int = 1, return_chain: bool = False, **kwargs):
        horizon = int(kwargs.get("horizon", 64))
        device = None
        if hard_conds:
            device = next(iter(hard_conds.values())).device
        if device is None:
            device = next(self.velocity_model.parameters(), torch.empty(0)).device
            if device.type == "cpu" and not any(True for _ in self.velocity_model.parameters()):
                device = torch.device("cpu")

        x = torch.randn((n_samples, horizon, self.state_dim), device=device, dtype=torch.float32) * self.sigma_noise
        return self._sample_rollout(
            x=x,
            context=context,
            hard_conds=hard_conds,
            num_steps=self.n_sampling_steps,
            return_chain=return_chain,
            guide=kwargs.get("guide"),
            n_guide_steps=int(kwargs.get("n_guide_steps", 1)),
            dataset=kwargs.get("dataset"),
            init_traj4proj=kwargs.get("init_traj4proj"),
            proj_params=kwargs.get("proj_params"),
        )

    @torch.no_grad()
    def run_local_inference(
        self,
        seed_trajectory_b: torch.Tensor,
        n_noising_steps: int,
        n_denoising_steps: int,
        context=None,
        hard_conds=None,
        n_samples: int = 1,
        return_chain: bool = False,
        **kwargs,
    ):
        seed = seed_trajectory_b.to(dtype=torch.float32)
        if seed.shape[0] == 1 and n_samples > 1:
            seed = seed.expand(n_samples, -1, -1).clone()
        elif seed.shape[0] != n_samples:
            raise ValueError(f"seed_trajectory_b batch must match n_samples={n_samples}, got {seed.shape[0]}")

        if n_noising_steps and n_noising_steps > 0:
            noise_scale = float(n_noising_steps) / float(max(1, self.n_sampling_steps))
            seed = seed + noise_scale * torch.randn_like(seed) * self.sigma_noise

        return self._sample_rollout(
            x=seed,
            context=context,
            hard_conds=hard_conds,
            num_steps=int(n_denoising_steps),
            return_chain=return_chain,
            guide=kwargs.get("guide"),
            n_guide_steps=int(kwargs.get("n_guide_steps", 1)),
            dataset=kwargs.get("dataset"),
            init_traj4proj=kwargs.get("init_traj4proj"),
            proj_params=kwargs.get("proj_params"),
        )
