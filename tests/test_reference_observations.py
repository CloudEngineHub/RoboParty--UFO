from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from ufo.envs.observations import actor_observation_values, compute_humanoid_observations_max, pack_observation
from ufo.inference.runtime import _target_states_from_obs
from ufo.math import quat_rotate_inverse
from ufo.motion.reference import get_backward_observation, load_expert_trajectories_from_motion_lib, reference_base_ang_vel
from ufo.training.evaluation import (
    get_backward_observation as get_mjlab_eval_backward_observation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class ReferenceObservationsTest(unittest.TestCase):
    def test_reference_base_ang_vel_matches_simulator_contract(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(obs_scales={"base_ang_vel": 0.25})))
        sqrt_half = 2.0**-0.5
        base_quat = torch.tensor(
            [
                [0.0, 0.0, sqrt_half, sqrt_half],
                [sqrt_half, 0.0, 0.0, sqrt_half],
            ],
            dtype=torch.float32,
        )
        world_ang_vel = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [-2.0, 4.0, 1.0],
            ],
            dtype=torch.float32,
        )

        expected = quat_rotate_inverse(base_quat, world_ang_vel, w_last=True) * 0.25
        actual = reference_base_ang_vel(env, base_quat, world_ang_vel)

        self.assertTrue(torch.allclose(actual, expected))

    def test_reference_base_ang_vel_defaults_to_unit_scale(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(obs_scales={})))
        base_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
        world_ang_vel = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)

        actual = reference_base_ang_vel(env, base_quat, world_ang_vel)

        self.assertTrue(torch.equal(actual, world_ang_vel))

    def test_mjlab_eval_backward_observation_requires_obs_filter(self) -> None:
        env = SimpleNamespace(config=SimpleNamespace(obs=SimpleNamespace(use_obs_filter=False)))

        with self.assertRaisesRegex(ValueError, "requires obs.use_obs_filter=True"):
            get_mjlab_eval_backward_observation(env, motion_id=0, include_last_action=False)

    def test_reference_paths_do_not_directly_assign_world_base_ang_vel(self) -> None:
        paths = [
            "ufo/motion/reference.py",
        ]
        for relative_path in paths:
            with self.subTest(path=relative_path):
                tree = ast.parse((REPO_ROOT / relative_path).read_text())
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    assigns_ref_ang_vel = any(isinstance(target, ast.Name) and target.id == "ref_ang_vel" for target in node.targets)
                    if assigns_ref_ang_vel:
                        self.assertNotEqual(ast.unparse(node.value), "ref_body_angular_vels[:, 0]", relative_path)

    def test_reference_backward_helpers_use_zero_fake_actions(self) -> None:
        paths = [
            "ufo/motion/reference.py",
        ]
        for relative_path in paths:
            with self.subTest(path=relative_path):
                tree = ast.parse((REPO_ROOT / relative_path).read_text())
                zero_action_assignments = 0
                for node in ast.walk(tree):
                    if not isinstance(node, ast.Assign):
                        continue
                    assigns_bogus_actions = any(isinstance(target, ast.Name) and target.id == "bogus_actions" for target in node.targets)
                    if not assigns_bogus_actions:
                        continue
                    value = ast.unparse(node.value)
                    self.assertNotEqual(value, "ref_dof_pos", relative_path)
                    zero_action_assignments += value == "torch.zeros_like(ref_dof_pos)"
                self.assertGreaterEqual(
                    zero_action_assignments,
                    1,
                    relative_path,
                )


    def test_reference_encoding_preserves_velocity_scaling_and_reset_states(self) -> None:
        frames, bodies, dofs = 4, 2, 3
        rotation = torch.zeros(frames, bodies, 4)
        rotation[..., -1] = 1
        motion = {
            "rg_pos_t": torch.arange(frames * bodies * 3, dtype=torch.float32).reshape(frames, bodies, 3) / 10,
            "rg_rot_t": rotation,
            "body_vel_t": torch.full((frames, bodies, 3), 2.0),
            "body_ang_vel_t": torch.full((frames, bodies, 3), 4.0),
            "dof_pos": torch.full((frames, dofs), 0.75),
            "dof_vel": torch.full((frames, dofs), 6.0),
        }
        env = SimpleNamespace(
            device="cpu", dt=0.02, default_dof_pos=torch.full((1, dofs), 0.25),
            gravity_vec=torch.tensor([[0.0, 0.0, -1.0]]),
            config=SimpleNamespace(obs=SimpleNamespace(use_obs_filter=True, obs_scales={
                "base_ang_vel": 0.25, "dof_pos": 0.5, "dof_vel": 2.0, "projected_gravity": 0.5,
            })),
            _motion_lib=SimpleNamespace(_motion_lengths=torch.tensor([0.08]), get_motion_state=lambda ids, times: motion),
        )
        for multiplier in (-1.0, 0.0, 0.5, 1.0, 2.0):
            with self.subTest(velocity_multiplier=multiplier):
                observation, reference = get_backward_observation(env, 0, velocity_multiplier=multiplier)
                expected = torch.cat([
                    torch.full((frames, dofs), 0.25), torch.full((frames, dofs), 12.0 * multiplier),
                    env.gravity_vec.repeat(frames, 1) * 0.5, torch.full((frames, 3), multiplier),
                ], dim=-1)
                torch.testing.assert_close(observation["state"], expected, rtol=0, atol=0)
                torch.testing.assert_close(observation["last_action"], torch.zeros(frames, dofs), rtol=0, atol=0)
                torch.testing.assert_close(reference["dof_pos"], motion["dof_pos"], rtol=0, atol=0)
                torch.testing.assert_close(reference["ref_dof_vel"], motion["dof_vel"] * multiplier, rtol=0, atol=0)
                torch.testing.assert_close(reference["ref_body_vels"], motion["body_vel_t"] * multiplier, rtol=0, atol=0)
                torch.testing.assert_close(reference["ref_body_angular_vels"], motion["body_ang_vel_t"] * multiplier, rtol=0, atol=0)

    def test_expert_and_reference_keep_height_filter_and_episode_contracts(self) -> None:
        frames, bodies, dofs = 4, 2, 3
        rotation = torch.zeros(frames, bodies, 4)
        rotation[..., 2:] = 2.0**-0.5
        motion = {
            "rg_pos_t": torch.arange(frames * bodies * 3, dtype=torch.float32).reshape(frames, bodies, 3) / 10,
            "rg_rot_t": rotation,
            "body_vel_t": torch.full((frames, bodies, 3), 2.0),
            "body_ang_vel_t": torch.tensor([1.0, -2.0, 3.0]).repeat(frames, bodies, 1),
            "dof_pos": torch.tensor([-0.75, 0.25, 0.75]).repeat(frames, 1),
            "dof_vel": torch.full((frames, dofs), 6.0),
        }
        original_motion = {key: value.clone() for key, value in motion.items()}
        library = SimpleNamespace(
            _motion_lengths=torch.tensor([0.08, 0.08]),
            _num_unique_motions=2,
            _motion_data_keys=["first", "second"],
            load_motions_for_training=Mock(),
            get_motion_state=Mock(side_effect=lambda ids, times: motion),
        )
        env = SimpleNamespace(
            device="cpu", dt=0.02, default_dof_pos=torch.full((1, dofs), 0.25),
            gravity_vec=torch.tensor([[0.0, 0.0, -1.0]]), _motion_lib=library,
            config=SimpleNamespace(obs=SimpleNamespace(
                use_obs_filter=True, root_height_obs=False, obs_scales={"base_ang_vel": 0.25},
            )),
        )
        agent_cfg = SimpleNamespace(model=SimpleNamespace(seq_length=2))
        for root_height in (False, True):
            for obs_filter in (False, True):
                with self.subTest(root_height=root_height, obs_filter=obs_filter):
                    env.config.obs.root_height_obs = root_height
                    env.config.obs.use_obs_filter = obs_filter
                    expert = load_expert_trajectories_from_motion_lib(env, agent_cfg)
                    reference, reset = get_backward_observation(env, 0, use_root_height_obs=root_height)
                    alternate, _ = get_backward_observation(env, 0, use_root_height_obs=not root_height)
                    storage = expert.storage
                    observation = storage["observation"]
                    privileged = reference["privileged_state"] if obs_filter else reference
                    alternate_privileged = alternate["privileged_state"] if obs_filter else alternate
                    torch.testing.assert_close(observation["privileged_state"][:frames], privileged, rtol=0, atol=0)
                    if root_height:
                        torch.testing.assert_close(privileged[:, 0], motion["rg_pos_t"][:, 0, 2], rtol=0, atol=0)
                        torch.testing.assert_close(privileged[:, 1:], alternate_privileged, rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(privileged, alternate_privileged[:, 1:], rtol=0, atol=0)
                    if obs_filter:
                        torch.testing.assert_close(observation["state"][:frames], reference["state"], rtol=0, atol=0)
                        self.assertFalse(torch.signbit(reference["last_action"]).any())
                        self.assertEqual(reset["dof_pos"].data_ptr(), motion["dof_pos"].data_ptr())
                    else:
                        self.assertEqual(reset, {})
                    self.assertTrue(torch.equal(observation["last_action"], torch.zeros(frames * 2, dofs)))
                    self.assertTrue(torch.signbit(observation["last_action"][:, 0]).all())
                    self.assertEqual(storage["truncated"].tolist(), [False, False, False, True] * 2)
                    self.assertEqual(storage["terminated"].tolist(), [False] * (frames * 2))
                    self.assertEqual(storage["motion_id"].tolist(), [0] * frames + [1] * frames)
                    self.assertEqual(storage["done"].shape, (frames * 2, 1))
                    self.assertEqual(expert.file_names, ["first", "second"])
                    self.assertEqual(expert.seq_length, 2)
                    for key, value in motion.items():
                        torch.testing.assert_close(value, original_motion[key], rtol=0, atol=0)
        self.assertEqual(library.load_motions_for_training.call_count, 4)


@pytest.mark.parametrize("scales", [
    {},
    *[dict.fromkeys(("dof_pos", "dof_vel", "projected_gravity", "base_ang_vel", "max_local_self"), scale)
      for scale in (0.5, 1.0, 2.0)],
    {"dof_pos": 0.5, "dof_vel": 2.0, "projected_gravity": 3.0, "base_ang_vel": 0.25, "max_local_self": 0.75},
])
@pytest.mark.parametrize("use_obs_filter", [False, True])
def test_reference_and_expert_match_clean_simulator_with_custom_scales(scales, use_obs_filter):
    frames, dofs = 4, 3
    rotation = torch.tensor([0.0, 0.0, 2.0**-0.5, 2.0**-0.5]).repeat(frames, 2, 1)
    motion = {
        "rg_pos_t": torch.arange(frames * 6, dtype=torch.float32).reshape(frames, 2, 3) / 10,
        "rg_rot_t": rotation,
        "body_vel_t": torch.full((frames, 2, 3), 2.0),
        "body_ang_vel_t": torch.tensor([1.0, 2.0, 3.0]).repeat(frames, 2, 1),
        "dof_pos": torch.tensor([-0.75, 0.25, 0.75]).repeat(frames, 1),
        "dof_vel": torch.full((frames, dofs), 2.0),
    }
    original = {key: value.clone() for key, value in motion.items()}
    env = SimpleNamespace(
        device="cpu", dt=0.02, default_dof_pos=torch.full((1, dofs), 0.25),
        gravity_vec=torch.tensor([[0.0, 0.0, -1.0]]), is_evaluating=False,
        config=OmegaConf.create({"obs": {
            "use_obs_filter": use_obs_filter, "root_height_obs": True, "obs_scales": scales,
            "noise_scales": dict.fromkeys((*scales, "dof_pos", "dof_vel", "base_ang_vel"), 10.0),
        }}),
        _motion_lib=SimpleNamespace(
            _motion_lengths=torch.tensor([0.08]), _num_unique_motions=1, _motion_data_keys=["sample"],
            load_motions_for_training=Mock(), get_motion_state=lambda ids, times: motion,
        ),
    )
    privileged = compute_humanoid_observations_max(
        motion["rg_pos_t"], rotation, motion["body_vel_t"], motion["body_ang_vel_t"], True, True,
    )
    values = {
        "actions": torch.zeros(frames, dofs), "dof_pos": motion["dof_pos"] - env.default_dof_pos,
        "dof_vel": motion["dof_vel"], "projected_gravity": env.gravity_vec.repeat(frames, 1),
        # A 90-degree yaw maps world (1, 2, 3) to body (2, -1, 3).
        "base_ang_vel": torch.tensor([2.0, -1.0, 3.0]).repeat(frames, 1),
        "max_local_self": torch.cat(list(privileged.values()), dim=-1),
    }
    raw = actor_observation_values(
        values, history={"dof_pos": torch.zeros(frames, 2, dofs)}, history_lengths={"dof_pos": 2},
        config=env.config.obs, is_evaluating=True, noise_multiplier=1.0,
    )
    expected = pack_observation(raw, torch.zeros(frames), to_numpy=False, include_last_action=True, include_history_actor=False)
    rng = torch.get_rng_state()
    reference, reset = get_backward_observation(env, 0, use_root_height_obs=True)
    expert = load_expert_trajectories_from_motion_lib(env, SimpleNamespace(model=SimpleNamespace(seq_length=2)))
    assert torch.equal(torch.get_rng_state(), rng)
    for key in ("state", "privileged_state", "last_action"):
        torch.testing.assert_close(expert.storage["observation"][key], expected[key], atol=1e-6, rtol=1e-6)
        if use_obs_filter:
            torch.testing.assert_close(reference[key], expected[key], atol=1e-6, rtol=1e-6)
    if use_obs_filter:
        evaluated, _ = get_mjlab_eval_backward_observation(env, 0, include_last_action=True)
        torch.testing.assert_close(evaluated, reference, rtol=0, atol=0)
        target = _target_states_from_obs(reset, "cpu", num_dof=dofs)
        torch.testing.assert_close(target["dof_states"][0, :, 0], motion["dof_pos"][0], rtol=0, atol=0)
        torch.testing.assert_close(target["dof_states"][0, :, 1], motion["dof_vel"][0], rtol=0, atol=0)
        torch.testing.assert_close(target["root_states"][0, 7:10], motion["body_vel_t"][0, 0], rtol=0, atol=0)
        torch.testing.assert_close(target["root_states"][0, 10:13], motion["body_ang_vel_t"][0, 0], rtol=0, atol=0)
    else:
        torch.testing.assert_close(reference, expected["privileged_state"], rtol=0, atol=0)
    for key in motion:
        torch.testing.assert_close(motion[key], original[key], rtol=0, atol=0)
