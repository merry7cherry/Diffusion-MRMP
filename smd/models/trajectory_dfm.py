from __future__ import annotations

import math
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


def logit_normal_timestep_sample(
    P_mean: float,
    P_std: float,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    random_values = torch.randn((num_samples,), device=device, dtype=dtype)
    sampled = torch.sigmoid(random_values * P_std + P_mean)
    return torch.clamp(sampled, min=0.0, max=1.0)


def sample_grouped_timesteps(
    *,
    batch_size: int,
    groups_per_batch: int,
    device: torch.device,
    dtype: torch.dtype,
    P_mean_t: float,
    P_std_t: float,
    P_mean_r: float,
    P_std_r: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if groups_per_batch <= 0:
        raise ValueError(f"groups_per_batch must be positive, got {groups_per_batch}")
    if batch_size % groups_per_batch != 0:
        raise ValueError(
            f"batch_size must be divisible by groups_per_batch, got batch_size={batch_size}, "
            f"groups_per_batch={groups_per_batch}"
        )

    group_size = batch_size // groups_per_batch
    t_groups = logit_normal_timestep_sample(P_mean_t, P_std_t, groups_per_batch, device, dtype)
    r_groups = logit_normal_timestep_sample(P_mean_r, P_std_r, groups_per_batch, device, dtype)
    t_groups = torch.minimum(t_groups, r_groups)
    r_groups = torch.maximum(t_groups, r_groups)
    t = t_groups.view(groups_per_batch, 1, 1).expand(groups_per_batch, group_size, 1).reshape(batch_size, 1)
    r = r_groups.view(groups_per_batch, 1, 1).expand(groups_per_batch, group_size, 1).reshape(batch_size, 1)
    return t, r


def reshape_batch_to_groups(batch_tensor: torch.Tensor, groups_per_batch: int) -> torch.Tensor:
    if batch_tensor.ndim < 2:
        raise ValueError("batch_tensor must have shape [N, ...]")
    if groups_per_batch <= 0:
        raise ValueError(f"groups_per_batch must be positive, got {groups_per_batch}")
    if batch_tensor.shape[0] % groups_per_batch != 0:
        raise ValueError(
            f"batch size {batch_tensor.shape[0]} must be divisible by groups_per_batch={groups_per_batch}"
        )
    group_size = batch_tensor.shape[0] // groups_per_batch
    return batch_tensor.reshape(groups_per_batch, group_size, *batch_tensor.shape[1:])


def flatten_group_features(group_tensor: torch.Tensor) -> torch.Tensor:
    if group_tensor.ndim < 3:
        raise ValueError("group_tensor must have shape [G, B, ...]")
    return group_tensor.reshape(group_tensor.shape[0], group_tensor.shape[1], -1)


def adaptive_matching_loss(
    predicted_points: torch.Tensor,
    target_points: torch.Tensor,
    norm_eps: float,
    norm_p: float,
) -> torch.Tensor:
    terms = (predicted_points - target_points) ** 2
    terms = terms.reshape(terms.shape[0], -1).sum(dim=1)
    adaptive_weight = (terms.detach() + norm_eps) ** norm_p
    return torch.mean(terms / adaptive_weight)


def _expand_marginals(
    marginals: torch.Tensor,
    *,
    expected_size: int,
    leading_shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    marginals = marginals.to(device=device, dtype=dtype)
    if marginals.ndim == 1:
        if marginals.shape[0] != expected_size:
            raise ValueError(f"{name} must have shape [{expected_size}], got {tuple(marginals.shape)}")
        view_shape = (1,) * len(leading_shape) + (expected_size,)
        return marginals.reshape(view_shape).expand(*leading_shape, expected_size)
    if marginals.shape[:-1] != leading_shape or marginals.shape[-1] != expected_size:
        raise ValueError(
            f"{name} must have shape {leading_shape + (expected_size,)}, got {tuple(marginals.shape)}"
        )
    return marginals


def _sinkhorn_from_logits(
    logits: torch.Tensor,
    *,
    row_marginals: torch.Tensor,
    col_marginals: torch.Tensor,
    num_iters: int = 20,
    eps: float = 1e-12,
    return_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if logits.ndim < 2:
        raise ValueError(f"logits must have shape [..., N_src, N_tgt], got {tuple(logits.shape)}")
    if num_iters <= 0:
        raise ValueError(f"num_iters must be > 0, got {num_iters}")

    num_source_points, num_target_points = logits.shape[-2], logits.shape[-1]
    orig_dtype = logits.dtype
    work_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
    if return_dtype is None:
        return_dtype = orig_dtype

    logits = logits.to(dtype=work_dtype)
    leading_shape = logits.shape[:-2]
    row_marginals = _expand_marginals(
        row_marginals,
        expected_size=num_source_points,
        leading_shape=leading_shape,
        device=logits.device,
        dtype=work_dtype,
        name="row_marginals",
    )
    col_marginals = _expand_marginals(
        col_marginals,
        expected_size=num_target_points,
        leading_shape=leading_shape,
        device=logits.device,
        dtype=work_dtype,
        name="col_marginals",
    )

    log_row_marginals = torch.log(row_marginals.clamp_min(eps))
    log_col_marginals = torch.log(col_marginals.clamp_min(eps))
    log_u = torch.zeros_like(logits[..., :, 0])
    log_v = torch.zeros_like(logits[..., 0, :])

    for _ in range(int(num_iters)):
        log_u = log_row_marginals - torch.logsumexp(logits + log_v.unsqueeze(-2), dim=-1)
        log_v = log_col_marginals - torch.logsumexp(logits + log_u.unsqueeze(-1), dim=-2)

    plan = torch.exp(logits + log_u.unsqueeze(-1) + log_v.unsqueeze(-2)).clamp_min(0.0)
    return plan.to(dtype=return_dtype)


def _row_normalize_plan(plan: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return plan / plan.sum(dim=-1, keepdim=True).clamp_min(eps)


def _compute_sinkhorn_barycentric_projection(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    *,
    kernel_temp: float,
    num_iters: int,
    eps: float,
) -> torch.Tensor:
    if source_points.ndim != 3 or target_points.ndim != 3:
        raise ValueError("source_points and target_points must have shape [G, N, D]")
    if source_points.shape != target_points.shape:
        raise ValueError(
            f"source_points and target_points must share shape [G, N, D], "
            f"got {tuple(source_points.shape)} and {tuple(target_points.shape)}"
        )

    orig_dtype = source_points.dtype
    work_dtype = torch.float32 if source_points.dtype in (torch.float16, torch.bfloat16) else source_points.dtype
    source_points = source_points.to(dtype=work_dtype)
    target_points = target_points.to(dtype=work_dtype)

    row_marginals = torch.full(
        (source_points.shape[1],),
        1.0 / float(source_points.shape[1]),
        device=source_points.device,
        dtype=work_dtype,
    )
    col_marginals = torch.full(
        (target_points.shape[1],),
        1.0 / float(target_points.shape[1]),
        device=target_points.device,
        dtype=work_dtype,
    )
    distances = torch.cdist(source_points, target_points)
    logits = -distances / float(kernel_temp)
    plan = _sinkhorn_from_logits(
        logits,
        row_marginals=row_marginals,
        col_marginals=col_marginals,
        num_iters=num_iters,
        eps=eps,
    )
    weights = _row_normalize_plan(plan, eps=eps)
    return torch.matmul(weights, target_points).to(dtype=orig_dtype)


def compute_split_v0_drift(
    gen: torch.Tensor,
    pos: torch.Tensor,
    *,
    kernel_temp_pos: float = 1.0,
    kernel_temp_neg: float = 1.0,
    num_sinkhorn_iters: int = 20,
    eps: float = 1e-12,
) -> torch.Tensor:
    if gen.ndim != 3 or pos.ndim != 3:
        raise ValueError("gen and pos must have shape [G, B, D]")
    if gen.shape != pos.shape:
        raise ValueError(f"gen and pos must share shape [G, B, D], got {tuple(gen.shape)} and {tuple(pos.shape)}")

    pos_proj = _compute_sinkhorn_barycentric_projection(
        gen,
        pos,
        kernel_temp=kernel_temp_pos,
        num_iters=num_sinkhorn_iters,
        eps=eps,
    )
    neg_proj = _compute_sinkhorn_barycentric_projection(
        gen,
        gen,
        kernel_temp=kernel_temp_neg,
        num_iters=num_sinkhorn_iters,
        eps=eps,
    )
    return pos_proj - neg_proj


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
        if self.context_dim > 0:
            if context is None:
                raise ValueError("Task-conditioned DFM requires a context tensor during training and inference")
            if context.ndim == 2:
                if context.shape[-1] != self.context_dim:
                    raise ValueError(
                        f"context must have last dimension {self.context_dim}, got {tuple(context.shape)}"
                    )
                context = context.unsqueeze(1).expand(batch_size, horizon, -1)
            elif context.ndim == 3:
                if context.shape[-1] != self.context_dim:
                    raise ValueError(
                        f"context must have last dimension {self.context_dim}, got {tuple(context.shape)}"
                    )
            else:
                raise ValueError(f"context must have shape [B, C] or [B, H, C], got {tuple(context.shape)}")
            pieces.append(context)
        elif context is not None:
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
        groups_per_batch: int = 4,
        P_mean_t: float = -1.0,
        P_std_t: float = 2.5,
        P_mean_r: float = 1.0,
        P_std_r: float = 2.5,
        kernel_temp_pos: float = 1.0,
        kernel_temp_neg: float = 1.0,
        sinkhorn_iters: int = 20,
        norm_eps: float = 1e-4,
        norm_p: float = 0.0,
        projection_fn=None,
        context_model=None,
        **kwargs,
    ) -> None:
        super().__init__()
        velocity_kwargs = dict(kwargs)
        self.state_dim = int(kwargs.get("state_dim", getattr(velocity_model, "state_dim", 2)))
        self.context_dim = int(kwargs.get("context_dim", getattr(velocity_model, "context_dim", 0)))
        if velocity_model is None:
            velocity_kwargs.setdefault("state_dim", self.state_dim)
            velocity_kwargs.setdefault("context_dim", self.context_dim)
            velocity_model = globals()[velocity_model_class](**velocity_kwargs)
        self.velocity_model = velocity_model
        self.n_sampling_steps = int(n_sampling_steps)
        self.sigma_data = float(sigma_data)
        self.sigma_noise = float(sigma_noise)
        self.groups_per_batch = int(groups_per_batch)
        self.P_mean_t = float(P_mean_t)
        self.P_std_t = float(P_std_t)
        self.P_mean_r = float(P_mean_r)
        self.P_std_r = float(P_std_r)
        self.kernel_temp_pos = float(kernel_temp_pos)
        self.kernel_temp_neg = float(kernel_temp_neg)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.norm_eps = float(norm_eps)
        self.norm_p = float(norm_p)
        self.projection_fn = projection_fn
        self.context_model = context_model
        self.expects_task_tensor_context = self.context_dim > 0
        self.submodules: dict[str, nn.Module] = {}

    def _prepare_hard_conds(
        self,
        hard_conds: dict[int, torch.Tensor] | None,
        *,
        n_samples: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[int, torch.Tensor]:
        prepared: dict[int, torch.Tensor] = {}
        if hard_conds is None:
            return prepared
        for key, value in copy(hard_conds).items():
            tensor = value.to(device=device, dtype=dtype)
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

    def _prepare_context(
        self,
        context,
        *,
        n_samples: int,
        device: torch.device,
        dtype: torch.dtype,
        allow_dummy: bool = False,
    ):
        if context is None:
            if self.context_dim == 0:
                return None
            if allow_dummy:
                return torch.zeros((n_samples, self.context_dim), device=device, dtype=dtype)
            raise ValueError("Task-conditioned DFM requires task_normalized context")
        if self.context_model is not None:
            context = self.context_model(context)
        if not isinstance(context, torch.Tensor):
            raise ValueError("TrajectoryDFMModel expects context as a tensor for task-conditioned DFM")
        context = context.to(device=device, dtype=dtype)
        if context.ndim == 1:
            context = einops.repeat(context, "d -> b d", b=n_samples)
        elif context.ndim == 2 and context.shape[0] == 1 and n_samples > 1:
            context = context.expand(n_samples, -1)
        elif context.ndim != 2:
            raise ValueError(f"context must have shape [D] or [B, D], got {tuple(context.shape)}")
        if context.shape[0] != n_samples:
            raise ValueError(f"context batch must match n_samples={n_samples}, got {context.shape[0]}")
        if self.context_dim > 0 and context.shape[-1] != self.context_dim:
            raise ValueError(
                f"context last dimension must be {self.context_dim}, got {context.shape[-1]}"
            )
        return context

    def warmup(self, horizon: int = 64, device: str | torch.device = "cpu") -> None:
        sample = torch.randn((2, horizon, self.state_dim), device=device)
        t = torch.zeros((2, 1), device=device)
        h = torch.full((2, 1), 1.0 / max(1, self.n_sampling_steps), device=device)
        context = self._prepare_context(
            None,
            n_samples=2,
            device=sample.device,
            dtype=sample.dtype,
            allow_dummy=True,
        )
        self.velocity_model(sample, t, h, context=context)

    def loss(self, x: torch.Tensor, context=None, hard_conds=None, *args):
        del args
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        prepared_hard_conds = self._prepare_hard_conds(
            hard_conds,
            n_samples=batch_size,
            device=device,
            dtype=dtype,
        )
        prepared_context = self._prepare_context(
            context,
            n_samples=batch_size,
            device=device,
            dtype=dtype,
        )

        x1 = apply_hard_conditioning(x.clone(), prepared_hard_conds)
        x0 = torch.randn_like(x1) * self.sigma_noise
        t, r = sample_grouped_timesteps(
            batch_size=batch_size,
            groups_per_batch=self.groups_per_batch,
            device=device,
            dtype=dtype,
            P_mean_t=self.P_mean_t,
            P_std_t=self.P_std_t,
            P_mean_r=self.P_mean_r,
            P_std_r=self.P_std_r,
        )
        h = r - t

        t_view = t.unsqueeze(-1)
        r_view = r.unsqueeze(-1)
        h_view = h.unsqueeze(-1)
        x_t = (1.0 - t_view) * x0 + t_view * x1
        x_r = (1.0 - r_view) * x0 + r_view * x1
        x_t = apply_hard_conditioning(x_t, prepared_hard_conds)
        x_r = apply_hard_conditioning(x_r, prepared_hard_conds)

        predicted_velocity = self.velocity_model(x_t, t, h, prepared_context)
        x_r_pred = x_t + h_view * predicted_velocity
        x_r_pred = apply_hard_conditioning(x_r_pred, prepared_hard_conds)

        x_r_pred_groups = reshape_batch_to_groups(x_r_pred, self.groups_per_batch)
        x_r_groups = reshape_batch_to_groups(x_r, self.groups_per_batch)
        gen = flatten_group_features(x_r_pred_groups.detach())
        pos = flatten_group_features(x_r_groups)
        drift = compute_split_v0_drift(
            gen,
            pos,
            kernel_temp_pos=self.kernel_temp_pos,
            kernel_temp_neg=self.kernel_temp_neg,
            num_sinkhorn_iters=self.sinkhorn_iters,
        )
        target_x_r = (gen + drift).detach().reshape_as(x_r_pred_groups).reshape_as(x_r_pred)
        target_x_r = apply_hard_conditioning(target_x_r, prepared_hard_conds)

        loss = adaptive_matching_loss(
            x_r_pred,
            target_x_r,
            self.norm_eps,
            self.norm_p,
        )
        info = {
            "dfm_loss": float(loss.detach().item()),
            "drift_loss": float(loss.detach().item()),
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
        prepared_context = self._prepare_context(
            context,
            n_samples=batch_size,
            device=x.device,
            dtype=x.dtype,
        )
        prepared_hard_conds = self._prepare_hard_conds(
            hard_conds,
            n_samples=batch_size,
            device=x.device,
            dtype=x.dtype,
        )

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
