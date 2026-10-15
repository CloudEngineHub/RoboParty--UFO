"""Analytic RP1 reward checks, including reset boundaries and live inertias."""

import math
import unittest
from types import SimpleNamespace

import torch

from ufo.envs.rewards import RoboPartyRewardHistory, compute_aux_rewards, compute_com_angular_momentum, compute_roboparty_aux_rewards


class RoboPartyRewardsTest(unittest.TestCase):
    def test_com_momentum_includes_orbital_spin_and_rotated_live_inertia(self):
        pos = torch.tensor([[[-1.0, 0, 0], [1.0, 0, 0]]])
        vel = torch.tensor([[[0.0, -1, 0], [0, 1, 0]]])
        angular = torch.tensor([[[1.0, 0, 1], [0, 0, 2]]])
        quat = torch.tensor([[[math.sqrt(0.5), 0, 0, math.sqrt(0.5)], [1, 0, 0, 0]]])
        masses = torch.ones(1, 2)
        inertia = torch.diag_embed(torch.tensor([[[1.0, 2, 3], [1, 1, 1]]]))
        momentum = compute_com_angular_momentum(pos, vel, angular, quat, masses, inertia)
        torch.testing.assert_close(momentum, torch.tensor([[2.0, 0, 7]]), atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(compute_com_angular_momentum(pos, vel, angular, quat, masses * 2, inertia * 2), momentum * 2)
        torch.testing.assert_close(compute_com_angular_momentum(pos + 10, vel + 3, angular, quat, masses, inertia), momentum)

    def test_history_warmup_cache_partial_reset_and_nonfinite_recovery(self):
        history = RoboPartyRewardHistory(0.02)
        reset = torch.zeros(4, dtype=torch.bool)
        for step, action in enumerate((0.0, 1.0, 3.0)):
            values = history.compute(torch.full((4, 2), action), torch.zeros(4, 3), step=step, reset_mask=reset)
            torch.testing.assert_close(values["action_jerk_l2"], torch.full((4,), 2.0 if step == 2 else 0.0))
            self.assertIs(history.compute(torch.zeros(4, 2), torch.ones(4, 3), step=step, reset_mask=reset), values)
        snapshot = {name: value.clone() for name, value in values.items()}
        momentum = torch.tensor([[float("nan"), 0, 0], [float("inf"), 0, 0], [1.0, 0, 0], [1.0, 0, 0]])
        reset[2] = True
        values = history.compute(torch.full((4, 2), 7.0), momentum, step=3, reset_mask=reset)
        torch.testing.assert_close(values["angular_momentum_rate_l2"], torch.tensor([0.0, 0, 0, 2500]))
        self.assertTrue(torch.isfinite(history.previous_momentum).all())
        self.assertEqual(values["action_jerk_l2"][2], 0)
        history.reset(torch.tensor([0, 2]))
        reset.zero_()
        values = history.compute(torch.full((4, 2), 12.0), torch.zeros(4, 3), step=4, reset_mask=reset)
        torch.testing.assert_close(values["angular_momentum_rate_l2"], torch.tensor([0.0, 0, 0, 2500]))
        self.assertEqual(values["action_jerk_l2"][0], 0)
        self.assertEqual(values["action_jerk_l2"][2], 0)
        torch.testing.assert_close(snapshot["action_jerk_l2"], torch.full((4,), 2.0))

    def test_all_raw_terms_and_normal_contact_history(self):
        force = torch.zeros(1, 4, 4, 3)
        force[0, 1, 0, 2] = 2.0
        force[0, 0, 1, 2] = 1.0  # Strict >1 threshold.
        force[0, 2, 2, 2] = 2.0
        force[0, 3, 3, 2] = 2.0  # Too old for undesired contact.
        pos = torch.zeros(1, 4, 3)
        pos[..., 2] = 0.3
        vel = torch.tensor([[[3.0, 4, 100], [0, 8, 100], [0, 0, 0], [0, 0, 0]]])
        kwargs = dict(
            normal_forces_w_history=force,
            feet_indices=torch.tensor([0, 1]),
            undesired_body_indices=torch.tensor([2, 3]),
            body_pos=pos,
            body_vel=vel,
            torques=torch.tensor([[2.0, -4.0]]),
            last_actions=torch.zeros(1, 2),
            actions=torch.tensor([[1.0, 2.0]]),
            dof_pos_limits=torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
            dof_pos=torch.tensor([[-1.5, 2.0]]),
            dof_vel=torch.tensor([[2.0, -3.0]]),
            joint_acc=torch.tensor([[3.0, 4.0]]),
            rated_velocity=torch.tensor([1.0, 1.0]),
            rated_effort=torch.tensor([1.0, 2.0]),
            angular_momentum=torch.tensor([[0.0, 0.0, 100.0]]),
            history=RoboPartyRewardHistory(0.02),
            step=0,
            reset_mask=torch.zeros(1, dtype=torch.bool),
        )
        aux = compute_roboparty_aux_rewards(**kwargs)
        expected = dict(
            penalty_torques=20.0,
            penalty_action_rate=5.0,
            limits_dof_pos=1.5,
            height_gated_undesired_contacts=0.5,
            penalty_slippage=5.0,
            joint_acc_l2=25.0,
            joint_velocity_l2=13.0,
            joint_energy=16.0,
            joint_rated_velocity=5.0,
            joint_rated_effort=2.0,
            action_jerk_l2=0.0,
            angular_momentum_rate_l2=0.0,
            yaw_angular_momentum_l2=1000.0,
        )
        self.assertEqual(aux.keys(), expected.keys())
        for name, value in expected.items():
            torch.testing.assert_close(aux[name], torch.tensor([value]), rtol=0, atol=0)
        kwargs["normal_forces_w_history"] = force[:, :2]
        with self.assertRaisesRegex(ValueError, "history"):
            compute_roboparty_aux_rewards(**kwargs)
        for dt in (0.0, -1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RoboPartyRewardHistory(dt)





class G1SlippageTest(unittest.TestCase):
    def test_horizontal_slip_preserves_contact_gate_and_ignores_vertical_motion(self):
        force = torch.zeros(3, 2, 3)
        force[0, :, 2] = torch.tensor([2.0, 1.0])
        force[1, :, 2] = 2.0
        # The last environment has no contact despite moving horizontally.
        vel = torch.tensor([[[3., 4., 100.], [0., 8., 100.]],
                            [[0., 0., 100.], [0., 0., -100.]],
                            [[3., 4., 0.], [0., 8., 0.]]])
        zeros = torch.zeros(3, 4)
        quat = torch.tensor([0., 0., 0., 1.]).repeat(3, 2, 1)
        aux = compute_aux_rewards(
            contact_forces=force, feet_indices=torch.tensor([0, 1]),
            torques=zeros, last_actions=zeros, actions=zeros,
            dof_pos_limits=torch.tensor([[-1., 1.]]).repeat(4, 1),
            dof_pos=zeros, dof_vel_limits=torch.ones(4), dof_vel=zeros,
            torque_limits=torch.ones(4), penalised_contact_indices=torch.tensor([], dtype=torch.long),
            left_ankle_dof_indices=torch.tensor([0, 1]), right_ankle_dof_indices=torch.tensor([2, 3]),
            body_rot=quat, gravity_vec=torch.tensor([0., 0., -1.]).repeat(3, 1),
            body_vel=vel, forward_vec=torch.tensor([1., 0., 0.]).repeat(3, 1),
            base_quat=quat[:, 0],
            reward_limits=SimpleNamespace(soft_dof_vel_limit=.95, soft_torque_limit=.95),
        )
        torch.testing.assert_close(aux["penalty_slippage"], torch.tensor([5., 0., 0.]))


if __name__ == "__main__":
    unittest.main()
