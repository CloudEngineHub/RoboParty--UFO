"""Auxiliary reward kernels and RoboParty's reset-aware temporal penalties."""

import math

import torch

from ufo.math import my_quat_rotate, quat_rotate_inverse, quaternion_to_matrix, wrap_to_pi


def compute_aux_rewards(
    *,
    contact_forces: torch.Tensor,
    feet_indices: torch.Tensor,
    torques: torch.Tensor,
    last_actions: torch.Tensor,
    actions: torch.Tensor,
    dof_pos_limits: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel_limits: torch.Tensor,
    dof_vel: torch.Tensor,
    torque_limits: torch.Tensor,
    penalised_contact_indices: torch.Tensor,
    left_ankle_dof_indices: torch.Tensor,
    right_ankle_dof_indices: torch.Tensor,
    body_rot: torch.Tensor,
    gravity_vec: torch.Tensor,
    body_vel: torch.Tensor,
    forward_vec: torch.Tensor,
    base_quat: torch.Tensor,
    reward_limits,
) -> dict[str, torch.Tensor]:
    aux: dict[str, torch.Tensor] = {}
    contact = contact_forces
    foot_contact = contact[:, feet_indices, 2] > 1.0
    aux["penalty_torques"] = torch.sum(torch.square(torques), dim=1)
    aux["penalty_action_rate"] = torch.sum(torch.square(last_actions - actions), dim=1)
    lower, upper = dof_pos_limits[:, 0], dof_pos_limits[:, 1]
    aux["limits_dof_pos"] = torch.sum((-(dof_pos - lower).clip(max=0.0)) + ((dof_pos - upper).clip(min=0.0)), dim=1)
    vel_limit = dof_vel_limits * float(reward_limits.soft_dof_vel_limit)
    aux["limits_dof_vel"] = torch.sum((torch.abs(dof_vel) - vel_limit).clip(min=0.0, max=1.0), dim=1)
    torque_limit = torque_limits * float(reward_limits.soft_torque_limit)
    aux["limits_torque"] = torch.sum((torch.abs(torques) - torque_limit).clip(min=0.0), dim=1)
    if len(penalised_contact_indices) > 0:
        undesired = torch.any(torch.abs(contact[:, penalised_contact_indices, :]) > 1.0, dim=(1, 2))
    else:
        undesired = torch.zeros(dof_pos.shape[0], dtype=torch.bool, device=dof_pos.device)
    aux["penalty_undesired_contact"] = undesired.float()
    left_ankle_roll = dof_pos[:, left_ankle_dof_indices[1:2]]
    right_ankle_roll = dof_pos[:, right_ankle_dof_indices[1:2]]
    aux["penalty_ankle_roll"] = torch.sum(torch.square(left_ankle_roll) + torch.square(right_ankle_roll), dim=1)
    left_quat = body_rot[:, feet_indices[0]]
    right_quat = body_rot[:, feet_indices[1]]
    left_gravity = quat_rotate_inverse(left_quat, gravity_vec, w_last=True)
    right_gravity = quat_rotate_inverse(right_quat, gravity_vec, w_last=True)
    aux["penalty_feet_ori"] = (
        torch.sum(torch.square(left_gravity[:, :2]), dim=1).sqrt() * foot_contact[:, 0]
        + torch.sum(torch.square(right_gravity[:, :2]), dim=1).sqrt() * foot_contact[:, 1]
    )
    foot_vel = body_vel[:, feet_indices, :2]
    aux["penalty_slippage"] = torch.sum(torch.norm(foot_vel, dim=-1) * (torch.norm(contact[:, feet_indices, :], dim=-1) > 1.0), dim=1)
    forward_left = my_quat_rotate(left_quat, forward_vec)
    forward_right = my_quat_rotate(right_quat, forward_vec)
    root_forward = my_quat_rotate(base_quat, forward_vec)
    heading_root = torch.atan2(root_forward[:, 1], root_forward[:, 0])
    aux["feet_heading_alignment"] = torch.abs(wrap_to_pi(torch.atan2(forward_left[:, 1], forward_left[:, 0]) - heading_root)) + torch.abs(
        wrap_to_pi(torch.atan2(forward_right[:, 1], forward_right[:, 0]) - heading_root)
    )

    return aux


def combine_rewards(aux, scales, penalty_names, *, use_curriculum, penalty_scale, num_envs, device):
    """Apply dt-scaled weights in configured order, preserving auxiliary values."""
    reward = torch.zeros(num_envs, device=device)
    for name, scale in scales.items():
        if name not in aux:
            continue
        rew = aux[name] * scale
        if name in penalty_names and use_curriculum:
            rew *= penalty_scale
        reward += rew
    return reward


def _finite_bounded_penalty(value: torch.Tensor, maximum: float) -> torch.Tensor:
    return torch.nan_to_num(value, nan=maximum, posinf=maximum, neginf=0.0).clamp(max=maximum)


def compute_com_angular_momentum(
    body_com_pos: torch.Tensor,
    body_com_lin_vel: torch.Tensor,
    body_com_ang_vel: torch.Tensor,
    body_com_quat_wxyz: torch.Tensor,
    body_mass: torch.Tensor,
    body_inertia: torch.Tensor,
) -> torch.Tensor:
    """Orbital plus spin momentum about total COM, using live inertial frames."""
    total_mass = body_mass.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    com_pos = (body_com_pos * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass
    com_vel = (body_com_lin_vel * body_mass.unsqueeze(-1)).sum(dim=1) / total_mass
    orbital = torch.cross(body_com_pos - com_pos.unsqueeze(1), body_com_lin_vel - com_vel.unsqueeze(1), dim=-1)
    rotation = quaternion_to_matrix(body_com_quat_wxyz)
    inertia_world = rotation @ (body_inertia @ rotation.transpose(-1, -2))
    spin = (inertia_world @ body_com_ang_vel.unsqueeze(-1)).squeeze(-1)
    return (orbital * body_mass.unsqueeze(-1) + spin).sum(dim=1)


class RoboPartyRewardHistory:
    """Keep the source task's jerk warm-up and finite momentum history."""

    def __init__(self, step_dt: float):
        if not math.isfinite(step_dt) or step_dt <= 0.0:
            raise ValueError("RoboParty reward step_dt must be finite and positive")
        self.step_dt = step_dt
        self.previous_action = None
        self.previous_previous_action = None
        self.action_steps = None
        self.previous_momentum = None
        self.momentum_valid = None
        self._step = None
        self._cache = None

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        indices = slice(None) if env_ids is None else env_ids
        if self.action_steps is not None:
            self.action_steps[indices] = -1
            self.previous_action[indices] = 0.0
            self.previous_previous_action[indices] = 0.0
        if self.momentum_valid is not None:
            self.momentum_valid[indices] = False
            self.previous_momentum[indices] = 0.0
        self._step = None

    def compute(self, actions: torch.Tensor, momentum: torch.Tensor, *, step: int, reset_mask: torch.Tensor | None) -> dict[str, torch.Tensor]:
        if self._step == step:
            return self._cache
        if reset_mask is None:
            reset_mask = torch.zeros(actions.shape[0], dtype=torch.bool, device=actions.device)
        if self.previous_action is None:
            self.previous_action = actions.clone()
            self.previous_previous_action = actions.clone()
            self.action_steps = torch.zeros(actions.shape[0], dtype=torch.long, device=actions.device)
            jerk_penalty = torch.zeros(actions.shape[0], device=actions.device)
        else:
            new_episode = self.action_steps < 0
            clear = reset_mask | new_episode
            self.previous_action[clear] = actions[clear]
            self.previous_previous_action[clear] = actions[clear]
            self.action_steps[reset_mask] = 0
            jerk = actions - 2.0 * self.previous_action + self.previous_previous_action
            self.previous_previous_action = self.previous_action.clone()
            self.previous_action = actions.clone()
            self.action_steps += 1
            jerk_penalty = torch.where(self.action_steps >= 2, jerk.square().sum(dim=-1), 0.0)
        current_finite = torch.isfinite(momentum).all(dim=-1)
        if self.previous_momentum is None:
            self.previous_momentum = torch.where(current_finite.unsqueeze(-1), momentum, torch.zeros_like(momentum)).clone()
            self.momentum_valid = current_finite.clone()
            rate_penalty = torch.zeros(momentum.shape[0], dtype=momentum.dtype, device=momentum.device)
        else:
            valid_rate = self.momentum_valid & current_finite & ~reset_mask
            rate = (momentum - self.previous_momentum) / self.step_dt
            rate_penalty = torch.where(valid_rate, _finite_bounded_penalty(rate.square().sum(dim=-1), 1e4), 0.0)
            self.previous_momentum = torch.where(current_finite.unsqueeze(-1), momentum, self.previous_momentum).clone()
            self.momentum_valid = current_finite & ~reset_mask
        self._step = step
        self._cache = {"action_jerk_l2": jerk_penalty, "angular_momentum_rate_l2": rate_penalty}
        return self._cache


def compute_roboparty_aux_rewards(
    *,
    normal_forces_w_history: torch.Tensor,
    feet_indices: torch.Tensor,
    undesired_body_indices: torch.Tensor,
    body_pos: torch.Tensor,
    body_vel: torch.Tensor,
    torques: torch.Tensor,
    last_actions: torch.Tensor,
    actions: torch.Tensor,
    dof_pos_limits: torch.Tensor,
    dof_pos: torch.Tensor,
    dof_vel: torch.Tensor,
    joint_acc: torch.Tensor,
    rated_velocity: torch.Tensor,
    rated_effort: torch.Tensor,
    angular_momentum: torch.Tensor,
    history: RoboPartyRewardHistory,
    step: int,
    reset_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The RP1 UFO task's thirteen raw penalties, before agent-side weights."""
    if normal_forces_w_history.ndim != 4 or normal_forces_w_history.shape[1] < 3:
        raise ValueError("RoboParty rewards require normal-force history [env, history >= 3, body, xyz]")
    aux = {
        "penalty_torques": torques.square().sum(dim=1),
        "penalty_action_rate": (last_actions - actions).square().sum(dim=1),
    }
    lower, upper = dof_pos_limits[..., 0], dof_pos_limits[..., 1]
    aux["limits_dof_pos"] = ((lower - dof_pos).clamp(min=0.0) + (dof_pos - upper).clamp(min=0.0)).sum(dim=1)
    forces = normal_forces_w_history[:, :3, undesired_body_indices, :].norm(dim=-1)
    contact = (forces > 1.0).any(dim=1)
    height_gate = torch.sigmoid((body_pos[:, undesired_body_indices, 2] - 0.30) / 0.05)
    aux["height_gated_undesired_contacts"] = (height_gate * contact.float()).sum(dim=1)
    slip_contact = normal_forces_w_history[:, :, feet_indices, :].norm(dim=-1).amax(dim=1) > 1.0
    aux["penalty_slippage"] = (body_vel[:, feet_indices, :2].norm(dim=-1) * slip_contact).sum(dim=1)
    aux["joint_acc_l2"] = joint_acc.square().sum(dim=1)
    aux["joint_velocity_l2"] = dof_vel.square().sum(dim=1)
    aux["joint_energy"] = (torques * dof_vel).abs().sum(dim=1)
    aux["joint_rated_velocity"] = torch.relu(dof_vel.abs() / rated_velocity - 1.0).square().sum(dim=1)
    aux["joint_rated_effort"] = torch.relu(torques.abs() / rated_effort - 1.0).square().sum(dim=1)
    aux.update(history.compute(actions, angular_momentum, step=step, reset_mask=reset_mask))
    aux["yaw_angular_momentum_l2"] = _finite_bounded_penalty(angular_momentum[:, 2].square(), 1e3)
    return aux
