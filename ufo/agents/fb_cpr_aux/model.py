# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils._pytree import tree_map

from ufo.agents.common.base import BaseModel
from ufo.agents.common.networks import (
    eval_mode,
    tree_get_batch_size,
)

from .config import FBcprAuxModelConfig


class SharedCritic(nn.Module):
    def __init__(self, disc, aux):
        super().__init__()
        self.input_filter = disc.input_filter
        self.num_parallel = disc.num_parallel
        self.embed_z = disc.embed_z
        self.embed_sa = disc.embed_sa
        self.trunk = nn.Sequential(*list(disc.Fs.children())[:-1])
        self.head_disc = disc.Fs[-1]
        self.head_aux = aux.Fs[-1]

    def forward(self, obs, z, action):
        obs = self.input_filter(obs)
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embed = self.embed_z(torch.cat([obs, z], dim=-1))
        sa_embed = self.embed_sa(torch.cat([obs, action], dim=-1))
        features = self.trunk(torch.cat([sa_embed, z_embed], dim=-1))
        return self.head_disc(features), self.head_aux(features)


class FBcprAuxModel(BaseModel):
    config_class = FBcprAuxModelConfig

    def __init__(self, obs_space, action_dim, cfg: FBcprAuxModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.cfg: FBcprAuxModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device

        # create networks
        self._backward_map = arch.b.build(obs_space, arch.z_dim)
        self._forward_map = arch.f.build(obs_space, arch.z_dim, action_dim)
        self._actor = arch.actor.build(obs_space, arch.z_dim, action_dim)
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)

        self._discriminator = cfg.archi.discriminator.build(obs_space, cfg.archi.z_dim)
        self._critic = cfg.archi.critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        self._aux_critic = cfg.archi.critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        if cfg.archi.shared_critic:
            self._critic = SharedCritic(self._critic, self._aux_critic)
            del self._aux_critic
        self._aux_reward_normalizer = cfg.norm_aux_reward.build()

        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        # create TARGET networks
        self._target_backward_map = copy.deepcopy(self._backward_map)
        self._target_forward_map = copy.deepcopy(self._forward_map)
        self._target_critic = copy.deepcopy(self._critic)
        if not self.cfg.archi.shared_critic:
            self._target_aux_critic = copy.deepcopy(self._aux_critic)

    def _normalize(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        return self._backward_map(self._normalize(obs))

    @torch.no_grad()
    def actor(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, std: float):
        return self._actor(self._normalize(obs), z, std)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z):
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean.float()
        return dist.sample().float()  # TODO we upcast to float32 to make sure the action can be converted to numpy later

    def reward_inference(
        self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        batch_size = tree_get_batch_size(next_obs)
        num_batches = int(np.ceil(batch_size / self.cfg.inference_batch_size))
        z = 0
        wr = reward if weight is None else reward * weight
        for i in range(num_batches):
            start_idx, end_idx = i * self.cfg.inference_batch_size, (i + 1) * self.cfg.inference_batch_size
            next_obs_slice = tree_map(lambda x: x[start_idx:end_idx].to(self.device), next_obs)
            B = self.backward_map(next_obs_slice)
            z += torch.matmul(wr[start_idx:end_idx].to(self.device).T, B)
        return self.project_z(z)

    def reward_wr_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor], reward: torch.Tensor) -> torch.Tensor:
        return self.reward_inference(next_obs, reward, F.softmax(10 * reward, dim=0))

    def goal_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self.backward_map(next_obs)
        return self.project_z(z)

    def tracking_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.goal_inference(next_obs)
