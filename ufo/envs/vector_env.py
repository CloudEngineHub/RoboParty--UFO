from __future__ import annotations

from typing import Any, Dict, Union

import gymnasium
import numpy as np
import torch
from gymnasium import Env
from gymnasium.vector import VectorEnv
from torch.utils._pytree import tree_map

from ufo.math import (
    xyzw_to_wxyz,
)

from .core import MJLabCore


class MJLabVectorEnv(VectorEnv):
    """Gymnasium interface for UFO observations, actions and transition metadata."""

    def __init__(
        self,
        env: MJLabCore,
        *,
        include_last_action: bool = True,
        include_history_actor: bool = True,
    ):
        super().__init__()
        self._env = env
        self.spec = None
        self.num_envs = env.num_envs
        self.include_last_action = include_last_action
        self.include_history_actor = include_history_actor

        self.single_action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(env.num_dof,), dtype=np.float32)
        action_space_shape = (self.num_envs,) + self.single_action_space.shape
        self.action_space = gymnasium.spaces.Box(
            low=np.tile(self.single_action_space.low, (self.num_envs, 1)),
            high=np.tile(self.single_action_space.high, (self.num_envs, 1)),
            shape=action_space_shape,
            dtype=np.float32,
        )
        example_observation, _ = self.reset()
        observation_spaces = {}
        for key, value in example_observation.items():
            observation_spaces[key] = gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=value.shape, dtype=value.dtype)
        self.observation_space = gymnasium.spaces.Dict(observation_spaces)

    @property
    def single_observation_space(self):
        single_obs_spaces = {}
        for key, space in self.observation_space.spaces.items():
            single_obs_spaces[key] = gymnasium.spaces.Box(low=space.low[0], high=space.high[0], shape=space.shape[1:], dtype=space.dtype)
        return gymnasium.spaces.Dict(single_obs_spaces)

    @property
    def device(self):
        return self.base_env.device

    @property
    def base_env(self) -> Env:
        return self._env

    @property
    def unwrapped(self):
        return self.base_env

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        to_numpy: bool = True,
        target_states: dict[str, torch.Tensor] | None = None,
    ):
        del seed, options
        self.base_env.reset_all(target_states=target_states)
        observation = self.base_env.get_observation(
            to_numpy=to_numpy,
            include_last_action=self.include_last_action,
            include_history_actor=self.include_history_actor,
        )
        qpos, qvel = self._get_qpos_qvel(to_numpy=to_numpy)
        return observation, {"qpos": qpos, "qvel": qvel}

    def _get_qpos_qvel(self, to_numpy: bool = True):
        base_pos_wxyz = torch.cat([self._env.robot_root_states[:, :3], xyzw_to_wxyz(self._env.robot_root_states[:, 3:7])], dim=-1)
        qpos = torch.cat([base_pos_wxyz, self._env.dof_pos], dim=-1)
        qvel = torch.cat([self._env.robot_root_states[:, 7:10], self._env.base_ang_vel, self._env.dof_vel], dim=-1)
        if to_numpy:
            return qpos.detach().cpu().numpy(), qvel.detach().cpu().numpy()
        return qpos, qvel

    def step(self, actions: Union[torch.Tensor, np.ndarray, Dict], to_numpy: bool = True):
        if isinstance(actions, dict):
            actions = actions["actions"]
        if isinstance(actions, np.ndarray):
            actions = torch.tensor(actions, device=self._env.device, dtype=torch.float32)
        _, reward, reset, new_info = self.base_env.step(actions)
        time_outs = new_info["time_outs"].bool()
        terminated = torch.logical_and(reset.bool(), ~time_outs)
        truncated = time_outs
        observation = self.base_env.get_observation(
            to_numpy=False,
            include_last_action=self.include_last_action,
            include_history_actor=self.include_history_actor,
        )
        qpos, qvel = self._get_qpos_qvel(to_numpy=False)
        if torch.any(reset):
            env_ids = reset.nonzero(as_tuple=False).flatten()
            new_info["final_env_ids"] = env_ids
            new_info["final_observation"] = tree_map(lambda value: value[env_ids].clone(), observation)
            new_info["final_qpos"], new_info["final_qvel"] = qpos[env_ids], qvel[env_ids]
            self.base_env.reset_finished(env_ids)
            reset_observation = self.base_env.get_observation(
                to_numpy=False, include_last_action=self.include_last_action,
                include_history_actor=self.include_history_actor, env_ids=env_ids,
            )
            for key in observation:
                observation[key][env_ids] = reset_observation[key]
            qpos, qvel = self._get_qpos_qvel(to_numpy=False)
        new_info["qpos"] = qpos
        new_info["qvel"] = qvel
        if to_numpy:
            observation = tree_map(lambda value: value.detach().cpu().numpy(), observation)
            reward = reward.detach().cpu().numpy()
            terminated = terminated.detach().cpu().numpy()
            truncated = truncated.detach().cpu().numpy()
            new_info = tree_map(lambda value: value.detach().cpu().numpy(), new_info)
        return observation, reward, terminated, truncated, new_info

    def close(self):
        return self.base_env.close()

    def render(self):
        return self.base_env.mjlab_env.render()
