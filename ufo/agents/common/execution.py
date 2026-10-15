# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""Fixed CUDA forward compilation, manual gradient averaging and actor isolation."""

from contextlib import contextmanager, nullcontext

import torch

from ufo.agents.common.networks import tree_get_batch_size
from ufo.training.distributed import average_gradients


def make_adam(params, device, *, lr, weight_decay):
    return torch.optim.Adam(
        params,
        lr=lr,
        weight_decay=weight_decay,
        fused=True if device == "cuda" else None,
    )


@contextmanager
def _frozen_parameters(modules):
    parameters = tuple(p for module in modules for p in module.parameters())
    flags = tuple(p.requires_grad for p in parameters)
    try:
        for p in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(parameters, flags):
            p.requires_grad_(flag)


class AgentExecutionMixin:
    @property
    def optimizer_dict(self):
        return {name: optimizer.state_dict() for name, optimizer in self.optimizers.items()}

    def compile_network_forwards(self):
        """Compile CUDA forwards; losses, sampling and gradient penalties stay eager."""
        if self.device != "cuda" or getattr(self, "_network_forwards_compiled", False):
            return
        for name in ("actor", "forward_map", "backward_map", "critic", "aux_critic", "goal_encoder"):
            for prefix in ("_", "_target_"):
                module = getattr(self._model, prefix + name, None)
                if module is not None:
                    module.forward = torch.compile(module.forward, mode="default")
        self._network_forwards_compiled = True

    def _sync_gradients(self, parameters):
        if not hasattr(self, "_gradient_bucket_cache"):
            self._gradient_bucket_cache = {}
        average_gradients(parameters, bucket_cache=self._gradient_bucket_cache)

    def _run_actor_update(self, *args, **kwargs):
        dependencies = [
            getattr(self._model, name)
            for name in ("_forward_map", "_critic", "_aux_critic", "_discriminator")
            if hasattr(self._model, name)
        ]
        # Gradients still flow through the critic's action input to the actor.
        with _frozen_parameters(dependencies):
            return self.update_actor(*args, **kwargs)


def gradient_penalty_wgan(
    discriminator,
    real_obs: torch.Tensor | dict[str, torch.Tensor],
    real_z: torch.Tensor,
    fake_obs: torch.Tensor | dict[str, torch.Tensor],
    fake_z: torch.Tensor,
    *,
    autocast_context=None,
) -> torch.Tensor:
    batch_size = tree_get_batch_size(real_obs)
    alpha = torch.rand(batch_size, 1, device=real_z.device)
    interpolated_obs = {}
    interpolated_obs_list = []
    if isinstance(real_obs, torch.Tensor):
        if real_obs.shape != fake_obs.shape:
            raise ValueError(f"Shape mismatch: {real_obs.shape} vs {fake_obs.shape}")
        interpolated_obs = (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True)
        interpolated_obs_list.append(interpolated_obs)
    else:
        for key in real_obs.keys():
            real_obs_tensor = real_obs[key]
            fake_obs_tensor = fake_obs[key]
            if isinstance(real_obs_tensor, torch.Tensor):
                interpolated_obs[key] = (alpha * real_obs_tensor + (1 - alpha) * fake_obs_tensor).requires_grad_(True)
                interpolated_obs_list.append(interpolated_obs[key])
            else:
                raise ValueError(f"Unsupported type for key {key}: {type(real_obs_tensor)}")
    interpolated_z = alpha * real_z + (1 - alpha) * fake_z
    interpolated_z = interpolated_z.requires_grad_(True)
    with autocast_context if autocast_context is not None else nullcontext():
        d_interpolates = discriminator.compute_logits(interpolated_obs, interpolated_z)
    if autocast_context is not None:
        d_interpolates = d_interpolates.float()
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolated_obs_list + [interpolated_z],
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        # Observation entries outside the discriminator's filter have no gradient.
        allow_unused=True,
    )
    gradients = [g for g in gradients if g is not None]
    cat_gradients = torch.cat(gradients, dim=1)
    return ((cat_gradients.norm(2, dim=1) - 1) ** 2).mean()
