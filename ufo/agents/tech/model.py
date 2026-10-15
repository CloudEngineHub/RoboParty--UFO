# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""TeCH temporal embedding, policy and critics."""

import copy
import math

import torch
import torch.nn.functional as F

from ufo.agents.common.base import BaseModel
from ufo.agents.common.networks import (
    eval_mode,
)

from .config import TeCHModelConfig


class TeCHModel(BaseModel):
    config_class = TeCHModelConfig

    def __init__(self, obs_space, action_dim, cfg: TeCHModelConfig):
        super().__init__(obs_space, action_dim, cfg)
        self.cfg: TeCHModelConfig = cfg
        arch = self.cfg.archi
        self.device = self.cfg.device

        self._goal_encoder = arch.goal_encoder.build(obs_space, arch.z_dim)
        self._actor = arch.actor.build(obs_space, arch.z_dim, action_dim)
        self._critic = arch.critic.build(obs_space, arch.z_dim, action_dim, output_dim=1)
        self._obs_normalizer = self.cfg.obs_normalizer.build(obs_space)

        self._discriminator = cfg.archi.discriminator.build(obs_space, cfg.archi.z_dim)
        self._aux_critic = cfg.archi.aux_critic.build(obs_space, cfg.archi.z_dim, action_dim, output_dim=1)
        self._aux_reward_normalizer = cfg.norm_aux_reward.build()

        self.train(False)
        self.requires_grad_(False)
        self.to(self.cfg.device)

    def _prepare_for_train(self) -> None:
        self._target_critic = copy.deepcopy(self._critic)
        self._target_aux_critic = copy.deepcopy(self._aux_critic)

    def _normalize(self, obs: torch.Tensor | dict[str, torch.Tensor]):
        with torch.no_grad(), eval_mode(self._obs_normalizer):
            return self._obs_normalizer(obs)

    def sample_z(self, size: int, device: str = "cpu") -> torch.Tensor:
        z = torch.randn((size, self.cfg.archi.z_dim), dtype=torch.float32, device=device)
        return self.project_z(z)

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.archi.z_geometry == "poincare_ball":
            radius = float(self.cfg.archi.poincare_radius)
            eps = float(self.cfg.archi.poincare_eps)
            z_norm = torch.linalg.norm(z, dim=-1, keepdim=True).clamp_min(eps)
            return radius * torch.tanh(z_norm / radius) * (z / z_norm)
        if self.cfg.archi.norm_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z

    def encode_goal(self, goal_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z = self._goal_encoder(self._normalize(goal_obs))
        return self.project_z(z)

    def actor(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, std: float):
        return self._actor(self._normalize(obs), z, std)

    def act(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, mean: bool = True) -> torch.Tensor:
        dist = self.actor(obs, z, self.cfg.actor_std)
        if mean:
            return dist.mean.float()
        return dist.sample().float()

    def goal_inference(self, goal_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        z_raw = self.backward_map(goal_obs)
        return self.project_z(z_raw)

    @torch.no_grad()
    def backward_map(self, obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self._goal_encoder(self._normalize(obs))

    def tracking_inference(self, next_obs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        return self.goal_inference(next_obs)
