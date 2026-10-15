"""Motion observations for expert buffers, evaluation and inference."""

import numpy as np
import torch

from ufo.agents.common.buffers.trajectory import ExpertTrajectoryBuffer
from ufo.envs.observations import compute_humanoid_observations_max, scale_observation
from ufo.math import quat_rotate_inverse


def reference_base_ang_vel(env, base_quat: torch.Tensor, world_ang_vel: torch.Tensor) -> torch.Tensor:
    """Match simulator actor-observation preprocessing for base angular velocity."""
    local_ang_vel = quat_rotate_inverse(base_quat, world_ang_vel, w_last=True)
    return scale_observation(
        "base_ang_vel", local_ang_vel, config=env.config.obs, is_evaluating=True, noise_multiplier=0.0,
    )


def _motion_times(env, motion_id):
    return torch.arange(int(np.ceil((env._motion_lib._motion_lengths[motion_id] / env.dt).cpu()))).to(env.device) * env.dt


def _motion_observations(env, motion_state, ref_dof_pos, *, root_height_obs, use_obs_filter):
    obs_dict = compute_humanoid_observations_max(
        motion_state["rg_pos_t"],
        motion_state["rg_rot_t"],
        motion_state["body_vel_t"],
        motion_state["body_ang_vel_t"],
        local_root_obs=True,
        root_height_obs=root_height_obs,
    )
    # Reference inputs share simulator scales but never observation noise.
    scale_options = dict(config=env.config.obs, is_evaluating=True, noise_multiplier=0.0)
    max_local_self_obs = scale_observation(
        "max_local_self", torch.cat([v for v in obs_dict.values()], dim=-1), **scale_options,
    )
    state = None
    if use_obs_filter:
        base_quat = motion_state["rg_rot_t"][:, 0]
        ref_ang_vel = reference_base_ang_vel(env, base_quat, motion_state["body_ang_vel_t"][:, 0])
        projected_gravity = quat_rotate_inverse(base_quat, env.gravity_vec[0:1].repeat(max_local_self_obs.shape[0], 1), w_last=True)
        state = torch.cat([
            scale_observation("dof_pos", ref_dof_pos, **scale_options),
            scale_observation("dof_vel", motion_state["dof_vel"], **scale_options),
            scale_observation("projected_gravity", projected_gravity, **scale_options),
            ref_ang_vel,
        ], dim=-1)
    return state, max_local_self_obs


def get_backward_observation(env, motion_id, use_root_height_obs: bool = False, velocity_multiplier: float = 1.0):
    motion_state = env._motion_lib.get_motion_state(motion_id, _motion_times(env, motion_id))
    motion_state = {
        **motion_state,
        "body_vel_t": motion_state["body_vel_t"] * velocity_multiplier,
        "body_ang_vel_t": motion_state["body_ang_vel_t"] * velocity_multiplier,
        "dof_vel": motion_state["dof_vel"] * velocity_multiplier,
    }
    ref_dof_pos = motion_state["dof_pos"] - env.default_dof_pos[0]
    state, max_local_self_obs = _motion_observations(
        env,
        motion_state,
        ref_dof_pos,
        root_height_obs=use_root_height_obs,
        use_obs_filter=env.config.obs.use_obs_filter,
    )
    if env.config.obs.use_obs_filter:
        bogus_actions = torch.zeros_like(ref_dof_pos)
        ref_dict = {
            "ref_dof_vel": motion_state["dof_vel"],
            "dof_pos": motion_state["dof_pos"],
            "ref_body_pos": motion_state["rg_pos_t"],
            "ref_body_rots": motion_state["rg_rot_t"],
            "ref_body_vels": motion_state["body_vel_t"],
            "ref_body_angular_vels": motion_state["body_ang_vel_t"],
        }
        ufo_obs = {"state": state, "last_action": bogus_actions, "privileged_state": max_local_self_obs}
        return ufo_obs, ref_dict
    return max_local_self_obs, {}


def get_goal_observation(env, motion_id: int, *, use_root_height_obs: bool = False):
    """Select a dataset motion, then read its local slot in the loaded subset."""
    env.set_is_evaluating(motion_id)
    local_ids = (env._motion_lib._curr_motion_ids == motion_id).nonzero().flatten()
    if not local_ids.numel():
        raise ValueError(f"Goal motion {motion_id} is absent from the loaded motion subset")
    return get_backward_observation(
        env, int(local_ids[0]), use_root_height_obs=use_root_height_obs, velocity_multiplier=0,
    )


def load_expert_trajectories_from_motion_lib(env, agent_cfg, device="cpu"):
    """Load expert trajectories directly from an environment motion library."""
    env._motion_lib.load_motions_for_training()
    episodes = []
    file_names = []
    for i in range(env._motion_lib._num_unique_motions):
        motion_times = _motion_times(env, i)
        motion_id = torch.tensor([i]).to(env.device).repeat(motion_times.shape[0])
        motion_res = env._motion_lib.get_motion_state(motion_id, motion_times)
        file_names.append(env._motion_lib._motion_data_keys[i])

        ref_dof_pos = motion_res["dof_pos"] - env.default_dof_pos[0]
        state, max_local_self_obs = _motion_observations(
            env,
            motion_res,
            ref_dof_pos,
            root_height_obs=env.config.obs.root_height_obs,
            use_obs_filter=True,
        )
        bogus_actions = ref_dof_pos * 0

        curr_motion_len = state.shape[0]
        truncated = torch.zeros(curr_motion_len, dtype=bool).to(env.device)
        truncated[-1] = True

        assert max_local_self_obs.shape[0] == curr_motion_len, (
            f"{env._motion_lib._motion_data_keys[i]}: {max_local_self_obs.shape[0]} vs {curr_motion_len}"
        )
        assert bogus_actions.shape[0] == curr_motion_len, (
            f"{env._motion_lib._motion_data_keys[i]}: {bogus_actions.shape[0]} vs {curr_motion_len}"
        )

        ep = {
            "observation": {
                "state": state,
                "last_action": bogus_actions,
                "privileged_state": max_local_self_obs,
            },
            "terminated": torch.zeros(curr_motion_len, dtype=bool).to(env.device),
            "truncated": truncated,
            "motion_id": torch.ones(curr_motion_len, dtype=torch.long) * i,
        }
        episodes.append(ep)

    expert_buffer = ExpertTrajectoryBuffer(
        episodes=episodes,
        seq_length=agent_cfg.model.seq_length,
        device=device,
    )

    assert expert_buffer.storage["observation"]["state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["observation"]["last_action"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["observation"]["privileged_state"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["terminated"].shape[0] == expert_buffer.storage["truncated"].shape[0]
    assert expert_buffer.storage["motion_id"].shape[0] == expert_buffer.storage["truncated"].shape[0]

    expert_buffer.file_names = file_names
    return expert_buffer
