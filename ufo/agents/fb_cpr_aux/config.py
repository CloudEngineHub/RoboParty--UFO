# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""Algorithm configuration and training defaults, kept in one place."""

import typing as tp
from typing import TYPE_CHECKING

import pydantic
from omegaconf import OmegaConf

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

DEFAULT_UPDATE_Z_EVERY_STEP = 100

_FORWARD = FORWARD.model_copy(update={"hidden_dim": 1024})
_ACTOR = ACTOR.model_copy(update={"hidden_dim": 1024})

CARTWHEEL_SAFE_REWARDS = {
    "agent": {
        "aux_rewards": ["penalty_torques", "penalty_action_rate", "limits_dof_pos", "limits_torque"],
        "aux_rewards_scaling": {
            "penalty_action_rate": -0.03, "limits_dof_pos": -10.0, "penalty_torques": 0.0, "limits_torque": 0.0,
        },
    },
    "task": {
        "penalty_undesired_contact": 0.0, "penalty_feet_ori": 0.0,
        "feet_heading_alignment": 0.0, "penalty_slippage": 0.0,
        "penalty_ankle_roll": 0.0, "penalty_action_rate": -0.1,
    },
}

if TYPE_CHECKING:
    from .agent import FBcprAuxAgent
    from .model import FBcprAuxModel


class FBcprAuxModelArchiConfig(BaseConfig):
    z_dim: int = 256
    norm_z: bool = True
    f: ForwardArchiConfig = pydantic.Field(_FORWARD, discriminator="name")
    b: BackwardArchiConfig = pydantic.Field(BACKWARD, discriminator="name")
    actor: ActorArchiConfig = _ACTOR
    critic: ForwardArchiConfig = pydantic.Field(_FORWARD, discriminator="name")
    discriminator: DiscriminatorArchiConfig = pydantic.Field(
        DISCRIMINATOR, discriminator="name"
    )
    shared_critic: bool = True


class FBcprAuxModelConfig(BaseModelConfig):
    name: tp.Literal["FBcprAuxModel"] = "FBcprAuxModel"

    archi: FBcprAuxModelArchiConfig = FBcprAuxModelArchiConfig()
    obs_normalizer: ObsNormalizerConfig = OBS_NORMALIZER
    inference_batch_size: int = 500_000
    seq_length: int = 8
    actor_std: float = 0.05
    norm_aux_reward: RewardNormalizerConfig = RewardNormalizerConfig(scale=True)

    def build(self, obs_space, action_dim) -> "FBcprAuxModel":
        from .model import FBcprAuxModel

        return FBcprAuxModel(obs_space, action_dim, self)


class FBcprAuxAgentTrainConfig(BaseConfig):
    use_bf16_amp: bool = False
    lr_f: float = 3e-4
    lr_b: float = 1e-5
    lr_actor: float = 3e-4
    weight_decay: float = 0.0
    clip_grad_norm: float = 0.0
    fb_target_tau: float = 0.01
    ortho_coef: float = 100.0
    train_goal_ratio: float = 0.2
    fb_pessimism_penalty: float = 0.0
    actor_pessimism_penalty: float = 0.5
    stddev_clip: float = 0.3
    q_loss_coef: float = 0.0
    batch_size: int = 1024
    discount: float = 0.98
    use_mix_rollout: bool = True
    update_z_every_step: int = DEFAULT_UPDATE_Z_EVERY_STEP
    z_buffer_size: int = 8192
    rollout_expert_trajectories: bool = True
    rollout_expert_trajectories_length: int = 250
    rollout_expert_trajectories_percentage: float = 0.5
    lr_discriminator: float = 1e-5
    lr_critic: float = 3e-4
    critic_target_tau: float = 0.005
    critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 0.05
    scale_reg: bool = True
    expert_asm_ratio: float = 0.6
    relabel_ratio: float | None = 0.8
    grad_penalty_discriminator: float = 10.0
    weight_decay_discriminator: float = 0.0
    lr_aux_critic: float = 3e-4
    reg_coeff_aux: float = 0.02
    aux_critic_pessimism_penalty: float = 0.5


class FBcprAuxAgentConfig(BaseConfig):
    name: tp.Literal["FBcprAuxAgent"] = "FBcprAuxAgent"

    model: FBcprAuxModelConfig = FBcprAuxModelConfig()
    train: FBcprAuxAgentTrainConfig = FBcprAuxAgentTrainConfig()
    aux_rewards: list[str] = AUX_REWARDS
    aux_rewards_scaling: dict[str, float] = AUX_REWARD_SCALES

    def build(self, obs_space, action_dim: int) -> "FBcprAuxAgent":
        return self.object_class(
            obs_space=obs_space,
            action_dim=action_dim,
            cfg=self,
        )

    @property
    def object_class(self):
        from .agent import FBcprAuxAgent

        return FBcprAuxAgent


TRAIN_RUNTIME = {
    "log_every_steps": 384000,
    "update_agent_every": 1024,
    "num_seed_steps": 10240,
    "num_agent_updates": 16,
    "eval_every_steps": 3200000,
}


def build_fb_agent(
    device: str = "cuda",
    update_z_every_step: int = DEFAULT_UPDATE_Z_EVERY_STEP,
    lr_scale: float = 1.0,
    clip_grad_norm: float = 0.0,
    network_width: int | None = None,
    shared_critic: bool | None = None,
    *,
    overrides: dict | None = None,
    cartwheel_aux_safe: bool = False,
) -> FBcprAuxAgentConfig:
    if network_width is not None and network_width not in (1024, 1536, 2048):
        raise ValueError("network_width must be 1024, 1536 or 2048")
    if cartwheel_aux_safe and {"aux_rewards", "aux_rewards_scaling"}.intersection(overrides or {}):
        raise ValueError("--cartwheel-aux-safe cannot override a robot's explicit FB reward baseline")
    defaults = FBcprAuxAgentConfig(
        model=FBcprAuxModelConfig(device=device),
        train=FBcprAuxAgentTrainConfig(update_z_every_step=int(update_z_every_step), clip_grad_norm=clip_grad_norm),
    )
    merged = OmegaConf.to_container(OmegaConf.merge(defaults.model_dump(), overrides or {}))
    # Validate robot input before CLI overrides can hide an invalid field.
    values = FBcprAuxAgentConfig.model_validate(merged).model_dump()
    # Robot baselines precede explicit topology choices; scale each final learning rate once.
    archi = values["model"]["archi"]
    if network_width is not None:
        for key in ("f", "actor", "critic"):
            archi[key]["hidden_dim"] = network_width
    if shared_critic is not None:
        archi["shared_critic"] = shared_critic
    for key in ("lr_f", "lr_b", "lr_actor", "lr_discriminator", "lr_critic", "lr_aux_critic"):
        values["train"][key] *= lr_scale
    if cartwheel_aux_safe:
        values.update(CARTWHEEL_SAFE_REWARDS["agent"])
    return FBcprAuxAgentConfig.model_validate(values)
