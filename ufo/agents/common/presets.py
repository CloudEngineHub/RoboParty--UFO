"""Shared FB/TeCH preset values; algorithm-specific settings stay in each preset."""

from ufo.agents.common.inputs import BatchNormNormalizerConfig, DictInputFilterConfig, ObsNormalizerConfig
from ufo.agents.common.networks import ActorArchiConfig, BackwardArchiConfig, DiscriminatorArchiConfig, ForwardArchiConfig

FORWARD = ForwardArchiConfig(
    name="ForwardArchi",
    hidden_dim=2048,
    hidden_layers=6,
    num_parallel=2,
    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state", "last_action", "history_actor"]),
)

BACKWARD = BackwardArchiConfig(
    hidden_dim=256, hidden_layers=1, norm=True,
    input_filter=DictInputFilterConfig(key=["state", "privileged_state"]),
)

ACTOR = ActorArchiConfig(
    name="actor", hidden_dim=2048, hidden_layers=6, embedding_layers=2,
    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "last_action", "history_actor"]),
)

DISCRIMINATOR = DiscriminatorArchiConfig(
    name="DiscriminatorArchi", hidden_dim=1024, hidden_layers=3,
    input_filter=DictInputFilterConfig(name="DictInputFilterConfig", key=["state", "privileged_state"]),
)

OBS_NORMALIZER = ObsNormalizerConfig(
    name="ObsNormalizerConfig",
    normalizers={
        "state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
        "privileged_state": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
        "last_action": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
        "history_actor": BatchNormNormalizerConfig(name="BatchNormNormalizerConfig", momentum=0.01),
    },
    allow_mismatching_keys=True,
)

AUX_REWARD_SCALES = {
    "penalty_action_rate": -0.1,
    "penalty_feet_ori": -0.4,
    "penalty_ankle_roll": -4.0,
    "limits_dof_pos": -10.0,
    "penalty_slippage": -2.0,
    "penalty_undesired_contact": -1.0,
    "penalty_torques": 0.0,
    "limits_torque": 0.0,
}

AUX_REWARDS = [
    "penalty_torques", "penalty_action_rate", "limits_dof_pos", "limits_torque",
    "penalty_undesired_contact", "penalty_feet_ori", "penalty_ankle_roll", "penalty_slippage",
]
