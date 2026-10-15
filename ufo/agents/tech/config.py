# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Algorithm configuration and training defaults, kept in one place."""

import typing as tp
from typing import TYPE_CHECKING, Literal

import pydantic

from ufo.agents.common.base import BaseConfig, BaseModelConfig
from ufo.agents.common.inputs import ObsNormalizerConfig
from ufo.agents.common.networks import (
    ActorArchiConfig,
    BackwardArchiConfig,
    DiscriminatorArchiConfig,
    ForwardArchiConfig,
    RewardNormalizerConfig,
)
from ufo.agents.common.presets import ACTOR, AUX_REWARD_SCALES, AUX_REWARDS, BACKWARD, DISCRIMINATOR, FORWARD, OBS_NORMALIZER

DEFAULT_UPDATE_Z_EVERY_STEP = 10

if TYPE_CHECKING:
    from .agent import TeCHAgent
    from .model import TeCHModel


class TeCHModelArchiConfig(BaseConfig):
    z_dim: int = 256
    norm_z: bool = True
    z_geometry: tp.Literal["hypersphere", "poincare_ball"] = "hypersphere"
    poincare_radius: float = 1.0
    poincare_eps: float = 1e-6
    goal_encoder: BackwardArchiConfig = pydantic.Field(BACKWARD, discriminator="name")
    actor: ActorArchiConfig = pydantic.Field(ACTOR, discriminator="name")
    critic: ForwardArchiConfig = pydantic.Field(FORWARD, discriminator="name")
    discriminator: DiscriminatorArchiConfig = pydantic.Field(
        DISCRIMINATOR, discriminator="name"
    )
    aux_critic: ForwardArchiConfig = FORWARD


class TeCHModelConfig(BaseModelConfig):
    name: tp.Literal["TeCHModel"] = "TeCHModel"
    archi: TeCHModelArchiConfig = TeCHModelArchiConfig()
    obs_normalizer: ObsNormalizerConfig = OBS_NORMALIZER
    seq_length: int = 8
    actor_std: float = 0.05
    norm_aux_reward: RewardNormalizerConfig = RewardNormalizerConfig(scale=True)

    def build(self, obs_space, action_dim) -> "TeCHModel":
        from .model import TeCHModel

        return TeCHModel(obs_space, action_dim, self)


class TeCHAgentTrainConfig(BaseConfig):

    lr_goal_encoder: float = 8e-7
    lr_actor: float = 5e-5
    lr_critic: float = 5e-5
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    batch_size: int = 1024
    discount: float = 0.98
    stddev_clip: float = 0.3
    actor_pessimism_penalty: float = 0.5
    critic_pessimism_penalty: float = 0.5
    critic_target_tau: float = 0.005
    train_goal_ratio: float = 0.2
    expert_asm_ratio: float = 0.6
    relabel_ratio: float | None = 0.4
    use_mix_rollout: bool = True
    update_z_every_step: int = DEFAULT_UPDATE_Z_EVERY_STEP
    z_buffer_size: int = 8192
    rollout_expert_trajectories: bool = True
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.5
    freeze_goal_encoder_after_pretrain: bool = False
    lr_discriminator: float = 2e-6
    weight_decay_discriminator: float = 0.0
    grad_penalty_discriminator: float = 10.0
    disc_reward_coef: float = 0.5
    reg_coeff_disc: float = 0.03
    scale_reg: bool = True
    lr_aux_critic: float = 2e-5
    reg_coeff_aux: float = 0.01
    aux_critic_pessimism_penalty: float = 0.5
    use_tech_pretrain: bool = True
    tech_pretrain_env_steps: int = 200000
    dual_reg: bool = True
    dual_lam_init: float = 3000.0
    lr_dual_lam: float = 4e-5
    dual_slack: float = 1.0
    tech_softplus_scale: float = 500.0
    tech_softplus_beta: float = 0.01
    tech_te_during_rl: bool = True
    tech_reward_scale: float = 1.0
    goal_encoder_lr_schedule: Literal["none", "linear", "cosine"] = "none"
    goal_encoder_lr_schedule_steps: int = 0
    goal_encoder_lr_min: float = 1e-6


class TeCHAgentConfig(BaseConfig):
    name: Literal["TeCHAgent"] = "TeCHAgent"
    model: TeCHModelConfig = TeCHModelConfig()
    train: TeCHAgentTrainConfig = TeCHAgentTrainConfig()
    aux_rewards: list[str] = AUX_REWARDS
    aux_rewards_scaling: dict[str, float] = AUX_REWARD_SCALES

    def build(self, obs_space, action_dim: int) -> "TeCHAgent":
        return self.object_class(obs_space=obs_space, action_dim=action_dim, cfg=self)

    @property
    def object_class(self):
        from .agent import TeCHAgent

        return TeCHAgent


TRAIN_RUNTIME = {
    "log_every_steps": 384000,
    "update_agent_every": 1024,
    "num_seed_steps": 10240,
    "num_agent_updates": 128,
    "eval_every_steps": 3200000,
}


def build_tech_agent(
    device: str = "cuda",
    update_z_every_step: int = DEFAULT_UPDATE_Z_EVERY_STEP,
) -> TeCHAgentConfig:
    return TeCHAgentConfig(
        model=TeCHModelConfig(device=device),
        train=TeCHAgentTrainConfig(update_z_every_step=int(update_z_every_step)),
    )
