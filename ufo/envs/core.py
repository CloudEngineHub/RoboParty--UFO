from __future__ import annotations

import random

import numpy as np
import torch
from omegaconf import OmegaConf

from ufo.math import (
    my_quat_rotate,
    quat_from_angle_axis,
    quat_mul,
    quat_rotate_inverse,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from ufo.motion.library import MotionLib

from .config import _action_target_scale, _default_joint_pos, _small_random_quaternions, _to_list
from .observations import ObservationHistory, actor_observation_values, compute_humanoid_observations_max, pack_observation
from .rewards import (
    RoboPartyRewardHistory,
    combine_rewards,
    compute_aux_rewards,
    compute_com_angular_momentum,
    compute_roboparty_aux_rewards,
)


class MJLabCore:
    """UFO task semantics over mjlab-owned physics and episode counters.

    mjlab steps physics without auto-reset. reset_idx is the sole reset entry:
    it calls the public mjlab reset, writes the sampled motion state, clears
    UFO action/history state, and refreshes the ordered observation snapshot.
    """

    @property
    def episode_length_buf(self):
        return self.mjlab_env.episode_length_buf

    def __init__(self, task_config, mjlab_env) -> None:
        if mjlab_env.cfg.auto_reset:
            raise ValueError("UFO owns task reset; mjlab auto_reset must be False")
        self.config = task_config
        self.mjlab_env = mjlab_env
        self.robot = mjlab_env.scene["robot"]
        self.device = str(mjlab_env.device)
        self.num_envs = int(mjlab_env.num_envs)
        self.dt = float(mjlab_env.step_dt)

        self.dof_names = tuple(_to_list(task_config.robot.dof_names))
        self.body_names = tuple(_to_list(task_config.robot.body_names))
        self.num_dof = len(self.dof_names)
        self.num_bodies = len(self.body_names)
        self.env_origins = mjlab_env.scene.env_origins

        mjlab_joint_names = tuple(self.robot.joint_names)
        mjlab_body_names = tuple(self.robot.body_names)
        missing_joints = [name for name in self.dof_names if name not in mjlab_joint_names]
        missing_bodies = [name for name in self.body_names if name not in mjlab_body_names]
        if missing_joints:
            raise ValueError(f"MJLab robot asset is missing joints from UFO config: {missing_joints}")
        if missing_bodies:
            raise ValueError(f"MJLab robot asset is missing bodies from UFO config: {missing_bodies}")
        self._joint_ids = torch.tensor([mjlab_joint_names.index(name) for name in self.dof_names], device=self.device, dtype=torch.long)
        self._body_ids = torch.tensor([mjlab_body_names.index(name) for name in self.body_names], device=self.device, dtype=torch.long)

        self._contact_sensor = mjlab_env.scene.sensors.get("body_contact")
        # Sensor/body ordering is fixed for the lifetime of this scene. Keep the
        # last sensor for each body, matching the old sequential assignments.
        contact_ids = {}
        if self._contact_sensor is not None:
            for index, primary_name in enumerate(self._contact_sensor.primary_names):
                name = primary_name.rsplit("/", 1)[-1]
                if name in self.body_names:
                    contact_ids[self.body_names.index(name)] = index
        self._contact_body_ids = torch.tensor(list(contact_ids), device=self.device, dtype=torch.long)
        self._contact_sensor_ids = torch.tensor(list(contact_ids.values()), device=self.device, dtype=torch.long)

        action_term = self.mjlab_env.action_manager.get_term("actions")
        action_target_names = tuple(action_term.target_names)
        if len(action_target_names) != self.num_dof or set(action_target_names) != set(self.dof_names):
            raise ValueError(
                "MJLab action target joints do not match UFO dof_names: "
                f"target_names={list(action_target_names)}, dof_names={list(self.dof_names)}"
            )
        self._action_term_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in action_target_names], device=self.device, dtype=torch.long
        )
        if action_target_names != self.dof_names:
            print(
                f"[INFO] MJLab action target order differs from UFO dof order: action_target_names={list(action_target_names)}",
                flush=True,
            )

        self.default_dof_pos = _default_joint_pos(task_config).to(self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.default_dof_pos_offset = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.action_target_scale = _action_target_scale(task_config).to(self.device).unsqueeze(0)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.forward_vec = torch.tensor([1.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)

        lower = torch.tensor(_to_list(task_config.robot.dof_pos_lower_limit_list), dtype=torch.float32, device=self.device)
        upper = torch.tensor(_to_list(task_config.robot.dof_pos_upper_limit_list), dtype=torch.float32, device=self.device)
        limit_scale = float(task_config.rewards.reward_limit.soft_dof_pos_limit)
        center = (lower + upper) * 0.5
        radius = (upper - lower) * 0.5 * limit_scale
        self.dof_pos_limits = torch.stack((center - radius, center + radius), dim=-1)
        self.torque_limits = torch.tensor(_to_list(task_config.robot.dof_effort_limit_list), device=self.device, dtype=torch.float32)
        self.dof_vel_limits = torch.tensor(_to_list(task_config.robot.dof_vel_limit_list), device=self.device, dtype=torch.float32)

        self.actions = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self._roboparty_rewards = task_config.rewards.set_reward == "roboparty"
        if self._roboparty_rewards:
            self._reward_history = RoboPartyRewardHistory(self.dt)
            self._normal_contact_sensor = mjlab_env.scene.sensors["normal_contact"]
            self._global_body_ids = self.robot.indexing.body_ids[self._body_ids]
            self.rated_effort = torch.tensor(task_config.robot.rated_effort, device=self.device)
            self.rated_velocity = torch.tensor(task_config.robot.rated_velocity, device=self.device)

        self._init_reward_scales()
        self._validate_aux_reward_semantics(task_config)
        self.feet_indices = torch.tensor(
            [self.body_names.index(name) for name in task_config.robot.contact_bodies], device=self.device, dtype=torch.long
        )
        if self._roboparty_rewards:
            self._undesired_body_indices = torch.tensor(
                [i for i, name in enumerate(self.body_names) if name not in task_config.robot.contact_bodies],
                device=self.device,
            )
        self.torso_index = self.body_names.index(task_config.robot.torso_name)
        penalized = []
        for pattern in _to_list(task_config.robot.penalize_contacts_on):
            penalized.extend([i for i, name in enumerate(self.body_names) if pattern in name])
        self.penalised_contact_indices = torch.tensor(sorted(set(penalized)), device=self.device, dtype=torch.long)
        self.left_ankle_dof_indices = torch.tensor(
            [self.dof_names.index(n) for n in task_config.robot.left_ankle_dof_names], device=self.device
        )
        self.right_ankle_dof_indices = torch.tensor(
            [self.dof_names.index(n) for n in task_config.robot.right_ankle_dof_names], device=self.device
        )

        self._init_motion_extend()
        self.is_evaluating = False
        self.average_episode_length = 0.0
        self.last_episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.num_compute_average_epl = float(self.config.rewards.num_compute_average_epl)
        self.add_noise_curriculum = bool(self.config.obs.get("add_noise_curriculum", False))
        self.current_noise_curriculum_value = float(self.config.obs.get("noise_initial_value", 1.0))
        self._init_motion_lib()
        self.history_handler = ObservationHistory(self.num_envs, task_config.obs.obs_auxiliary, task_config.obs.obs_dims, self.device)
        if task_config.domain_rand.get("default_dof_pos_startup_only", False):
            self._randomize_default_dof_pos_offset(torch.arange(self.num_envs, device=self.device))

        self._refresh_state()

    def _init_reward_scales(self) -> None:
        self.reward_scales = dict(OmegaConf.to_container(self.config.rewards.reward_scales, resolve=True))
        self.reward_scales = {k: float(v) * self.dt for k, v in self.reward_scales.items() if float(v) != 0.0}
        self.reward_penalty_reward_names = set(_to_list(self.config.rewards.reward_penalty_reward_names))
        self.use_reward_penalty_curriculum = bool(self.config.rewards.reward_penalty_curriculum)
        self.reward_penalty_scale = float(self.config.rewards.reward_initial_penalty_scale)

    def _validate_aux_reward_semantics(self, task_config) -> None:
        contact_bodies = _to_list(task_config.robot.get("contact_bodies", None))
        if len(contact_bodies) < 2:
            raise ValueError(
                "robot.contact_bodies must contain at least 2 bodies because the current MJLab reward "
                "implementation computes biped foot auxiliary terms unconditionally"
            )

        if "penalty_ankle_roll" in self.reward_scales:
            missing_fields = []
            if len(_to_list(task_config.robot.get("left_ankle_dof_names", None))) < 2:
                missing_fields.append("robot.left_ankle_dof_names")
            if len(_to_list(task_config.robot.get("right_ankle_dof_names", None))) < 2:
                missing_fields.append("robot.right_ankle_dof_names")
            if missing_fields:
                raise ValueError(
                    f"{', '.join(missing_fields)} must contain at least 2 joints because reward 'penalty_ankle_roll' is enabled"
                )

    def _update_average_episode_length(self, env_ids: torch.Tensor) -> None:
        if self.is_evaluating or len(env_ids) == 0:
            return
        current = torch.mean(self.last_episode_length_buf[env_ids].float()).item()
        ratio = min(float(len(env_ids)) / max(self.num_compute_average_epl, 1.0), 1.0)
        self.average_episode_length = self.average_episode_length * (1.0 - ratio) + current * ratio

    def _update_reward_penalty_curriculum(self) -> None:
        if not self.use_reward_penalty_curriculum:
            return
        if self.average_episode_length < float(self.config.rewards.reward_penalty_level_down_threshold):
            self.reward_penalty_scale *= 1.0 - float(self.config.rewards.reward_penalty_degree)
        elif self.average_episode_length > float(self.config.rewards.reward_penalty_level_up_threshold):
            self.reward_penalty_scale *= 1.0 + float(self.config.rewards.reward_penalty_degree)
        self.reward_penalty_scale = float(
            np.clip(
                self.reward_penalty_scale,
                float(self.config.rewards.reward_min_penalty_scale),
                float(self.config.rewards.reward_max_penalty_scale),
            )
        )

    def _update_obs_noise_curriculum(self) -> None:
        if not self.add_noise_curriculum:
            return
        if self.average_episode_length < float(self.config.obs.noise_curriculum_level_down_threshold):
            self.current_noise_curriculum_value *= 1.0 - float(self.config.obs.noise_curriculum_degree)
        elif self.average_episode_length > float(self.config.obs.noise_curriculum_level_up_threshold):
            self.current_noise_curriculum_value *= 1.0 + float(self.config.obs.noise_curriculum_degree)
        self.current_noise_curriculum_value = float(
            np.clip(
                self.current_noise_curriculum_value,
                float(self.config.obs.noise_value_min),
                float(self.config.obs.noise_value_max),
            )
        )

    def _init_motion_lib(self) -> None:
        self.config.robot.motion.step_dt = self.dt
        self._motion_lib = MotionLib(self.config.robot.motion, num_envs=self.num_envs, device=self.device)
        self._motion_lib.load_motions_for_training(max_num_seqs=self.num_envs)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_start_times = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._resample_motion_time_and_ids(torch.arange(self.num_envs, device=self.device))

    def _init_motion_extend(self) -> None:
        extend_parent_ids, extend_pos, extend_rot = [], [], []
        for item in _to_list(self.config.robot.motion.extend_config):
            extend_parent_ids.append(self.body_names.index(item["parent_name"]))
            extend_pos.append(item["pos"])
            extend_rot.append(item["rot"])
        self.num_extend_bodies = len(extend_parent_ids)
        if self.num_extend_bodies:
            self.extend_body_parent_ids = torch.tensor(extend_parent_ids, device=self.device, dtype=torch.long)
            self.extend_body_pos_in_parent = torch.tensor(extend_pos, device=self.device, dtype=torch.float32).repeat(self.num_envs, 1, 1)
            self.extend_body_rot_in_parent_wxyz = torch.tensor(extend_rot, device=self.device, dtype=torch.float32).repeat(
                self.num_envs, 1, 1
            )
            self.extend_body_rot_in_parent_xyzw = self.extend_body_rot_in_parent_wxyz[:, :, [1, 2, 3, 0]]
            self.body_names = tuple(
                list(self.body_names) + [item["joint_name"] for item in _to_list(self.config.robot.motion.extend_config)]
            )
        else:
            self.extend_body_parent_ids = torch.empty(0, device=self.device, dtype=torch.long)

    def _resample_motion_time_and_ids(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.motion_ids[env_ids] = self._motion_lib.sample_motions(len(env_ids))
        if self.is_evaluating and not self.config.enforce_randomize_motion_start_eval:
            self.motion_start_times[env_ids] = 0.0
        else:
            self.motion_start_times[env_ids] = self._motion_lib.sample_time(self.motion_ids[env_ids])

    def _randomize_default_dof_pos_offset(self, env_ids: torch.Tensor) -> None:
        if bool(self.config.domain_rand.get("randomize_default_dof_pos", False)):
            offset_range = self.config.domain_rand.default_dof_pos_noise_range
            self.default_dof_pos_offset[env_ids] = torch.empty(
                len(env_ids), self.num_dof, device=self.device, dtype=torch.float32
            ).uniform_(float(offset_range[0]), float(offset_range[1]))
        else:
            self.default_dof_pos_offset[env_ids] = 0.0

    def _refresh_state(self) -> None:
        data = self.robot.data
        # Tensor indices already create owned snapshots in UFO joint/body order.
        self.dof_pos = data.joint_pos[:, self._joint_ids]
        self.dof_vel = data.joint_vel[:, self._joint_ids]
        root_pose_w = data.root_link_pose_w.clone()
        root_vel_w = data.root_link_vel_w.clone()
        self.base_quat = wxyz_to_xyzw(root_pose_w[:, 3:7])
        self.robot_root_states = torch.cat([root_pose_w[:, :3], self.base_quat, root_vel_w], dim=-1)
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, root_vel_w[:, 3:6], w_last=True)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec, w_last=True)
        body_pose = data.body_link_pose_w[:, self._body_ids]
        body_vel = data.body_link_vel_w[:, self._body_ids]
        self.body_pos = body_pose[..., :3]
        self.body_rot = wxyz_to_xyzw(body_pose[..., 3:7])
        self.body_vel = body_vel[..., :3]
        self.body_ang_vel = body_vel[..., 3:6]
        self.torques = data.qfrc_actuator[:, self._joint_ids]
        self.contact_forces = self._read_contact_forces()

    def _read_contact_forces(self) -> torch.Tensor:
        forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        if self._contact_sensor is not None:
            source = self._contact_sensor.data.force
            if source is not None:
                # mjlab reports force exerted BY the body; UFO expects force ON it.
                forces[:, self._contact_body_ids] = -source[:, self._contact_sensor_ids]
        return forces

    def _extend_body_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.num_extend_bodies == 0:
            return self.body_pos, self.body_rot, self.body_vel, self.body_ang_vel
        rotated_pos = my_quat_rotate(
            self.body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_pos_in_parent.reshape(-1, 3),
        ).view(self.num_envs, -1, 3)
        extend_pos = rotated_pos + self.body_pos[:, self.extend_body_parent_ids]
        extend_rot = quat_mul(
            self.body_rot[:, self.extend_body_parent_ids].reshape(-1, 4),
            self.extend_body_rot_in_parent_xyzw.reshape(-1, 4),
            w_last=True,
        ).view(self.num_envs, -1, 4)
        extend_ang_vel = self.body_ang_vel[:, self.extend_body_parent_ids]
        extend_vel = self.body_vel[:, self.extend_body_parent_ids] + torch.cross(
            extend_ang_vel, self.extend_body_pos_in_parent.view(self.num_envs, -1, 3), dim=2
        )
        return (
            torch.cat([self.body_pos, extend_pos], dim=1),
            torch.cat([self.body_rot, extend_rot], dim=1),
            torch.cat([self.body_vel, extend_vel], dim=1),
            torch.cat([self.body_ang_vel, extend_ang_vel], dim=1),
        )

    def _compute_privileged_obs(self) -> None:
        body_pos, body_rot, body_vel, body_ang_vel = self._extend_body_state()
        obs_dict = compute_humanoid_observations_max(
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel,
            local_root_obs=True,
            root_height_obs=bool(self.config.obs.get("root_height_obs", True)),
        )
        self._max_local_self = torch.cat([v for v in obs_dict.values()], dim=-1)

    def _raw_actor_obs(self, env_ids=None) -> dict[str, torch.Tensor]:
        self._compute_privileged_obs()
        history_config = self.config.obs.obs_auxiliary["history_actor"]
        ids = slice(None) if env_ids is None else env_ids
        raw = actor_observation_values(
            {
                "actions": self.actions[ids],
                "base_ang_vel": self.base_ang_vel[ids],
                "dof_pos": (self.dof_pos - (self.default_dof_pos + self.default_dof_pos_offset))[ids],
                "dof_vel": self.dof_vel[ids],
                "projected_gravity": self.projected_gravity[ids],
                "max_local_self": self._max_local_self[ids],
            },
            # Observation assembly copies history before add/reset can change it.
            history={key: value[ids] for key, value in self.history_handler.history.items()},
            history_lengths=history_config,
            config=self.config.obs,
            is_evaluating=self.is_evaluating,
            noise_multiplier=self.current_noise_curriculum_value if self.add_noise_curriculum else 1.0,
        )
        # Observe the previous history before inserting this frame exactly once.
        for key in history_config:
            self.history_handler.add(key, raw[key], env_ids=env_ids)
        return raw

    def get_observation(self, *, to_numpy: bool = True, include_last_action: bool = True, include_history_actor: bool = True, env_ids=None):
        return pack_observation(
            self._raw_actor_obs(env_ids),
            self.episode_length_buf if env_ids is None else self.episode_length_buf[env_ids],
            to_numpy=to_numpy,
            include_last_action=include_last_action,
            include_history_actor=include_history_actor,
        )

    def _compute_reward(self, reset_mask=None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if self._roboparty_rewards:
            if reset_mask is None:
                reset_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            data = self.robot.data
            pose = data.body_com_pose_w[:, self._body_ids]
            velocity = data.body_com_vel_w[:, self._body_ids]
            momentum = compute_com_angular_momentum(
                pose[..., :3], velocity[..., :3], velocity[..., 3:], pose[..., 3:],
                data.model.body_mass[:, self._global_body_ids],
                torch.diag_embed(data.model.body_inertia[:, self._global_body_ids]),
            )
            aux = compute_roboparty_aux_rewards(
                normal_forces_w_history=self._normal_contact_sensor.data,
                feet_indices=self.feet_indices,
                undesired_body_indices=self._undesired_body_indices,
                body_pos=self.body_pos, body_vel=self.body_vel,
                torques=self.torques, last_actions=self.last_actions, actions=self.actions,
                dof_pos_limits=self.dof_pos_limits, dof_pos=self.dof_pos, dof_vel=self.dof_vel,
                joint_acc=data.joint_acc[:, self._joint_ids],
                rated_velocity=self.rated_velocity, rated_effort=self.rated_effort,
                angular_momentum=momentum, history=self._reward_history,
                step=self.mjlab_env.common_step_counter, reset_mask=reset_mask,
            )
        else:
            aux = compute_aux_rewards(
                contact_forces=self.contact_forces,
                feet_indices=self.feet_indices,
                torques=self.torques,
                last_actions=self.last_actions,
                actions=self.actions,
                dof_pos_limits=self.dof_pos_limits,
                dof_pos=self.dof_pos,
                dof_vel_limits=self.dof_vel_limits,
                dof_vel=self.dof_vel,
                torque_limits=self.torque_limits,
                penalised_contact_indices=self.penalised_contact_indices,
                left_ankle_dof_indices=self.left_ankle_dof_indices,
                right_ankle_dof_indices=self.right_ankle_dof_indices,
                body_rot=self.body_rot,
                gravity_vec=self.gravity_vec,
                body_vel=self.body_vel,
                forward_vec=self.forward_vec,
                base_quat=self.base_quat,
                reward_limits=self.config.rewards.reward_limit,
            )
        reward = combine_rewards(
            aux,
            self.reward_scales,
            self.reward_penalty_reward_names,
            use_curriculum=self.use_reward_penalty_curriculum,
            penalty_scale=self.reward_penalty_scale,
            num_envs=self.num_envs,
            device=self.device,
        )
        return reward, aux

    def _normalized_action(self, actions: torch.Tensor) -> torch.Tensor:
        if bool(self.config.robot.control.normalize_action):
            actions = (
                actions * float(self.config.robot.control.normalize_action_to) / float(self.config.robot.control.normalize_action_from)
            )
        return torch.clamp(actions, -float(self.config.robot.control.action_clip_value), float(self.config.robot.control.action_clip_value))

    def _mjlab_action_input(self) -> torch.Tensor:
        action_indices = self._action_term_dof_indices
        return self.actions[:, action_indices] + self.default_dof_pos_offset[:, action_indices] / torch.clamp(
            self.action_target_scale[:, action_indices], min=1.0e-6
        )

    def step(self, actions: torch.Tensor):
        actions = actions.to(self.device, dtype=torch.float32)
        self.last_actions[:] = self.actions
        self.actions[:] = self._normalized_action(actions)
        mjlab_actions = self._mjlab_action_input()
        _, _, terminated, time_outs, _ = self.mjlab_env.step(mjlab_actions)
        self._refresh_state()
        reset = torch.logical_or(terminated.bool(), time_outs.bool())
        reward, aux = self._compute_reward(reset if self._roboparty_rewards else None)
        aux_snapshot = {k: v.detach() for k, v in aux.items()}
        # The vector adapter captures the successor before resetting finished lanes.
        return None, reward, reset, {"time_outs": time_outs.bool().clone(), "aux_rewards": aux_snapshot}

    def reset_finished(self, env_ids: torch.Tensor) -> None:
        self.last_episode_length_buf[env_ids] = self.episode_length_buf[env_ids]
        self._update_average_episode_length(env_ids)
        self._update_reward_penalty_curriculum()
        self._update_obs_noise_curriculum()
        self.reset_idx(env_ids)

    def reset_all(self, target_states: dict[str, torch.Tensor] | None = None):
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_idx(env_ids, target_states=target_states)
        return None, {}

    def reset_idx(self, env_ids: torch.Tensor, target_states: dict[str, torch.Tensor] | None = None) -> None:
        if len(env_ids) == 0:
            return
        self.mjlab_env.reset(env_ids=env_ids)
        if not self.config.domain_rand.get("default_dof_pos_startup_only", False):
            self._randomize_default_dof_pos_offset(env_ids)
        if target_states is not None:
            root_xyzw = target_states["root_states"][env_ids].to(self.device, dtype=torch.float32)
            dof_state = target_states["dof_states"][env_ids].to(self.device, dtype=torch.float32)
            joint_pos = dof_state[..., 0]
            joint_vel = dof_state[..., 1]
        else:
            self._resample_motion_time_and_ids(env_ids)
            motion_times = self.motion_start_times[env_ids]
            motion_res = self._motion_lib.get_motion_state(self.motion_ids[env_ids], motion_times, offset=self.env_origins[env_ids])
            root_pos = motion_res["root_pos"]
            root_rot = motion_res["root_rot"]
            root_vel = motion_res["root_vel"]
            root_ang_vel = motion_res["root_ang_vel"]
            if self.config.get("lie_down_init", False):
                mask = torch.rand(len(env_ids), device=self.device) < float(getattr(self.config, "lie_down_init_prob", 0.0))
                if torch.any(mask):
                    root_pos = root_pos.clone()
                    root_rot = root_rot.clone()
                    root_pos[mask, 2] = 0.5
                    sign = 1 if random.random() < 0.5 else -1
                    rot_quat = quat_from_angle_axis(
                        torch.tensor(sign * (-torch.pi / 2), device=self.device),
                        torch.tensor([1.0, 0.0, 0.0], device=self.device),
                        w_last=True,
                    )
                    root_rot[mask] = quat_mul(rot_quat.expand_as(root_rot[mask]), root_rot[mask], w_last=True)
            root_pos = root_pos + torch.randn_like(root_pos) * float(self.config.init_noise_scale.root_pos) * float(
                self.config.noise_to_initial_level
            )
            root_rot = quat_mul(
                _small_random_quaternions(
                    len(env_ids),
                    float(self.config.init_noise_scale.root_rot) * 3.14 / 180.0 * float(self.config.noise_to_initial_level),
                    self.device,
                ),
                root_rot,
                w_last=True,
            )
            root_vel = root_vel + torch.randn_like(root_vel) * float(self.config.init_noise_scale.root_vel) * float(
                self.config.noise_to_initial_level
            )
            root_ang_vel = root_ang_vel + torch.randn_like(root_ang_vel) * float(self.config.init_noise_scale.root_ang_vel) * float(
                self.config.noise_to_initial_level
            )
            root_xyzw = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
            joint_pos = motion_res["dof_pos"] + torch.randn_like(motion_res["dof_pos"]) * float(
                self.config.init_noise_scale.dof_pos
            ) * float(self.config.noise_to_initial_level)
            joint_vel = motion_res["dof_vel"] + torch.randn_like(motion_res["dof_vel"]) * float(
                self.config.init_noise_scale.dof_vel
            ) * float(self.config.noise_to_initial_level)

        root_wxyz = torch.cat([root_xyzw[:, :3], xyzw_to_wxyz(root_xyzw[:, 3:7]), root_xyzw[:, 7:13]], dim=-1)
        self.robot.write_root_state_to_sim(root_wxyz, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self._joint_ids, env_ids=env_ids)
        self.mjlab_env.scene.write_data_to_sim()
        self.mjlab_env.sim.forward()
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.history_handler.reset(env_ids)
        if self._roboparty_rewards:
            self._reward_history.reset(env_ids)
        self._refresh_state()

    def set_is_evaluating(self, global_rank: int = 0):
        self.is_evaluating = True
        self.begin_seq_motion_samples(global_rank)

    def begin_seq_motion_samples(self, global_rank: int = 0):
        self._motion_lib.load_motions_for_evaluation(start_idx=global_rank * self.num_envs)
        self.reset_all()

    def set_is_training(self):
        self.is_evaluating = False
        self.resample_motion()

    def resample_motion(self):
        self._motion_lib.load_motions_for_training(max_num_seqs=self.num_envs)
        self.reset_all()

    def close(self):
        return self.mjlab_env.close()
