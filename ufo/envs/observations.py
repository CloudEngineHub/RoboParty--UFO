"""Observation history, humanoid coordinate transforms and policy inputs.

Noise draws follow actor-field insertion order, then history. The caller updates
history only after the observation values are returned.
"""

from collections import OrderedDict

import torch
from loguru import logger
from torch import Tensor
from torch.utils._pytree import tree_map

from ufo.math import calc_heading_quat_inv, my_quat_rotate, quat_mul, quat_to_tan_norm


class ObservationHistory:
    def __init__(self, num_envs, history_config, obs_dims, device):
        self.device = device
        self.history = {}

        self.buffer_config = {}
        for aux_key, aux_config in history_config.items():
            for obs_key, obs_num in aux_config.items():
                if obs_key in self.buffer_config:
                    self.buffer_config[obs_key] = max(self.buffer_config[obs_key], obs_num)
                else:
                    self.buffer_config[obs_key] = obs_num

        for key in self.buffer_config.keys():
            self.history[key] = torch.zeros(num_envs, self.buffer_config[key], obs_dims[key], device=self.device)

        logger.info("History Handler Initialized")
        for key, value in self.buffer_config.items():
            logger.info(f"Key: {key}, Value: {value}")

    def reset(self, reset_ids):
        if len(reset_ids) == 0:
            return
        assert set(self.buffer_config.keys()) == set(self.history.keys()), (
            f"History keys mismatch\n{self.buffer_config.keys()}\n{self.history.keys()}"
        )
        for key in self.history.keys():
            self.history[key][reset_ids] *= 0.0

    def add(self, key: str, value: Tensor, env_ids=None):
        assert key in self.history.keys(), f"Key {key} not found in history"
        ids = slice(None) if env_ids is None else env_ids
        val = self.history[key][ids].clone()
        self.history[key][ids, 1:] = val[:, :-1]
        self.history[key][ids, 0] = value

    def query(self, key: str):
        assert key in self.history.keys(), f"Key {key} not found in history"
        return self.history[key].clone()


@torch.jit.script
def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Dict[str, Tensor]
    obs_dict = OrderedDict()
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot_inv = calc_heading_quat_inv(root_rot, w_last=True)

    if root_height_obs:
        obs_dict["root_height"] = root_h

    heading_rot_inv_expand = heading_rot_inv.unsqueeze(-2)
    heading_rot_inv_expand = heading_rot_inv_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot_inv = heading_rot_inv_expand.reshape(
        heading_rot_inv_expand.shape[0] * heading_rot_inv_expand.shape[1], heading_rot_inv_expand.shape[2]
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = my_quat_rotate(flat_heading_rot_inv, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:]

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot_inv, flat_body_rot, w_last=True)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot, w_last=True)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])

    if not local_root_obs:
        root_rot_obs = quat_to_tan_norm(root_rot, w_last=True)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])

    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = my_quat_rotate(flat_heading_rot_inv, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

    obs_dict["local_body_pos"] = local_body_pos
    obs_dict["local_body_rot"] = local_body_rot_obs
    obs_dict["local_body_vel"] = local_body_vel
    obs_dict["local_body_ang_vel"] = local_body_ang_vel

    return obs_dict


def scale_observation(key, value, *, config, is_evaluating, noise_multiplier):
    scale = float(config.obs_scales.get(key, 1.0))
    noise_scale = 0.0 if is_evaluating else float(config.noise_scales.get(key, 0.0))
    noise_scale *= noise_multiplier
    if noise_scale != 0.0:
        value = value + (torch.rand_like(value) * 2.0 - 1.0) * noise_scale
    return value * scale


def actor_observation_values(values, *, history, history_lengths, config, is_evaluating, noise_multiplier):
    options = dict(config=config, is_evaluating=is_evaluating, noise_multiplier=noise_multiplier)
    raw = {key: scale_observation(key, value, **options) for key, value in values.items()}
    history_tensors = []
    for key in sorted(history_lengths):
        value = history[key][:, : history_lengths[key]]
        if config.get("history_oldest_first", False):
            value = value.flip(1)
        history_tensors.append(value.reshape(value.shape[0], -1))
    raw["history_actor"] = scale_observation("history_actor", torch.cat(history_tensors, dim=1), **options)
    return raw


def pack_observation(raw, episode_length, *, to_numpy, include_last_action, include_history_actor):
    obs = {
        "state": torch.cat([raw["dof_pos"], raw["dof_vel"], raw["projected_gravity"], raw["base_ang_vel"]], dim=-1),
        "privileged_state": raw["max_local_self"],
    }
    if include_last_action:
        obs["last_action"] = raw["actions"]
    obs["time"] = episode_length.unsqueeze(-1)
    if include_history_actor:
        obs["history_actor"] = raw["history_actor"]
    if to_numpy:
        obs = tree_map(lambda value: value.detach().cpu().numpy(), obs)
    return obs
