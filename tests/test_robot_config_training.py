from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from ufo.envs.config import MJLabEnvConfig, _action_target_scale, make_mjlab_ufo_env_cfg, resolve_task_config
from ufo.envs.core import MJLabCore
from ufo.inference.runtime import (
    _target_states_from_obs,
)
from ufo.inference.tracking_inference import _expert_qpos_from_obs
from ufo.inference.tracking_inference import (
    parse_args as parse_tracking_args,
)
from ufo.tools.robot_config import load_robot_training_spec, select_robot_config
from ufo.train import build_mjlab_config
from ufo.train import parse_args as parse_train_args


def _write_tiny_robot_with_training(root: Path, *, missing_actuator_joint: bool = False) -> Path:
    xml_path = root / "tiny_train.xml"
    xml_path.write_text(
        """
<mujoco model="tiny_train">
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="link1" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.1"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint1_motor" joint="joint1"/>
    <motor name="joint2_motor" joint="joint2"/>
  </actuator>
</mujoco>
""".strip()
    )
    joint2_block = (
        []
        if missing_actuator_joint
        else [
            "      joint2:",
            "        effort_limit: 2.0",
            "        velocity_limit: 20.0",
            "        armature: 0.02",
            "        friction: 0.002",
        ]
    )
    robot_config = root / "tiny_train.yaml"
    robot_config.write_text(
        "\n".join(
            [
                "name: tiny_train",
                "xml_path: tiny_train.xml",
                "base_body: base",
                "root_quat_order: xyzw",
                "coordinate_system: z_up",
                "dof_unit: rad",
                "control_joints:",
                "  mode: all_actuated",
                "feet: [link2]",
                "hands: []",
                "key_bodies: [base, link1, link2]",
                "default_dof_pos: {}",
                "training:",
                "  task_overrides: {}",
                "  semantics:",
                "    contact_bodies: [link2]",
                "    undesired_contact_bodies: [base]",
                "    torso_name: base",
                "    left_ankle_dof_names: []",
                "    right_ankle_dof_names: []",
                "  init_state:",
                "    pos: [0.0, 0.0, 1.0]",
                "    rot: [0.0, 0.0, 0.0, 1.0]",
                "    lin_vel: [0.0, 0.0, 0.0]",
                "    ang_vel: [0.0, 0.0, 0.0]",
                "    default_joint_angles:",
                "      joint1: 0.0",
                "      joint2: 0.0",
                "  control:",
                "    action_scale: 0.25",
                "    action_clip_value: 5.0",
                "    normalize_action_to: 5.0",
                "    effort_limit: [1.0, 2.0]",
                "    velocity_limit: [10.0, 20.0]",
                "    stiffness: {joint1: 1.0, joint2: 2.0}",
                "    damping: {joint1: 0.1, joint2: 0.2}",
                "  actuator:",
                "    source: yaml",
                "    joints:",
                "      joint1:",
                "        effort_limit: 1.0",
                "        velocity_limit: 10.0",
                "        armature: 0.01",
                "        friction: 0.001",
                *joint2_block,
            ]
        )
    )
    return robot_config


class RobotConfigTrainingTest(unittest.TestCase):
    def test_old_g1_default_builds_cfg(self) -> None:
        cfg = build_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
        )
        self.assertTrue(str(cfg.env.robot_config_path).endswith("configs/robots/g1_29dof.yaml"))
        self.assertTrue(str(cfg.env.mjcf_path).endswith("ufo/data/robots/g1/g1_29dof.xml"))

    def test_g1_fb_defaults_and_overrides_preserve_reward_preset(self) -> None:
        kwargs = dict(device="cpu", work_dir="/tmp/ufo_unit", num_envs=1,
                      num_env_steps=1, seed=1, use_wandb=False, wandb_run_name=None)
        cfg = build_mjlab_config(**kwargs)
        self.assertTrue(cfg.agent.model.archi.shared_critic)
        for name in ("f", "actor", "critic"):
            self.assertEqual(getattr(cfg.agent.model.archi, name).hidden_dim, 1024)
        self.assertFalse(cfg.agent.train.use_bf16_amp)
        self.assertEqual(cfg.prioritization_scale, 2.0)
        other = build_mjlab_config(**kwargs, network_width=2048, shared_critic=False)
        self.assertFalse(other.agent.model.archi.shared_critic)
        for name in ("f", "actor", "critic"):
            self.assertEqual(getattr(other.agent.model.archi, name).hidden_dim, 2048)
        safe = build_mjlab_config(**kwargs, cartwheel_aux_safe=True)
        self.assertTrue(safe.agent.model.archi.shared_critic)
        self.assertEqual(safe.agent.aux_rewards_scaling.get("penalty_feet_ori", 0), 0)
        self.assertEqual(safe.agent.aux_rewards, ["penalty_torques", "penalty_action_rate", "limits_dof_pos", "limits_torque"])
        self.assertEqual(safe.agent.aux_rewards_scaling, {
            "penalty_action_rate": -0.03, "limits_dof_pos": -10.0, "penalty_torques": 0.0, "limits_torque": 0.0,
        })
        self.assertEqual(safe.env.task_overrides["rewards"]["reward_scales"]["penalty_action_rate"], -0.1)
        self.assertEqual(safe.env.task_overrides["rewards"]["reward_scales"]["penalty_feet_ori"], 0.0)
        self.assertEqual(build_mjlab_config(**kwargs).model_dump(), cfg.model_dump())

    def test_explicit_g1_robot_config_builds_cfg(self) -> None:
        cfg = build_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_unit",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            robot_config="configs/robots/g1_29dof.yaml",
        )
        self.assertTrue(str(cfg.env.robot_config_path).endswith("configs/robots/g1_29dof.yaml"))

    def test_manifest_robot_config_is_used_when_cli_missing(self) -> None:
        argv = [
            "train.py",
            "--agent",
            "fb",
            "--data-manifest",
            "configs/data/example_robot_state_auto_build.yaml",
            "--gpu-ids",
            "single",
            "--smoke",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_train_args()
        self.assertTrue(str(args.robot_config).endswith("configs/robots/g1_29dof.yaml"))

    def test_cli_manifest_robot_config_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir))
            with self.assertRaisesRegex(ValueError, "does not match data manifest robot_config"):
                select_robot_config(tiny_robot, "configs/robots/g1_29dof.yaml")

    def test_tracking_manifest_robot_config_is_used_when_cli_missing(self) -> None:
        argv = [
            "tracking_inference.py",
            "--model-folder",
            "/tmp/ufo_unit_model",
            "--data-manifest",
            "configs/data/example_robot_state_auto_build.yaml",
            "--dataset",
            "g1_robot_state_sample",
            "--export-onnx",
            "false",
        ]
        with patch.object(sys, "argv", argv):
            args = parse_tracking_args()
        self.assertTrue(str(args.robot_config).endswith("configs/robots/g1_29dof.yaml"))

    def test_aux_rewards_require_two_contact_bodies_unconditionally(self) -> None:
        core = object.__new__(MJLabCore)
        core.reward_scales = {}
        cfg = OmegaConf.create(
            {
                "robot": {
                    "contact_bodies": ["left_foot"],
                    "left_ankle_dof_names": ["left_ankle_pitch_joint", "left_ankle_roll_joint"],
                    "right_ankle_dof_names": ["right_ankle_pitch_joint", "right_ankle_roll_joint"],
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "robot.contact_bodies.*biped foot auxiliary terms unconditionally"):
            core._validate_aux_reward_semantics(cfg)

    def test_aux_ankle_reward_requires_both_ankle_fields(self) -> None:
        core = object.__new__(MJLabCore)
        core.reward_scales = {"penalty_ankle_roll": -1.0}
        cfg = OmegaConf.create(
            {
                "robot": {
                    "contact_bodies": ["left_foot", "right_foot"],
                    "left_ankle_dof_names": ["left_ankle_pitch_joint"],
                    "right_ankle_dof_names": [],
                }
            }
        )
        with self.assertRaisesRegex(
            ValueError,
            "robot.left_ankle_dof_names, robot.right_ankle_dof_names.*penalty_ankle_roll",
        ):
            core._validate_aux_reward_semantics(cfg)

    def test_mjlab_action_input_reorders_policy_actions_to_action_term_order(self) -> None:
        core = object.__new__(MJLabCore)
        core.actions = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
        core.default_dof_pos_offset = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        core.action_target_scale = torch.tensor([[1.0, 2.0, 1.0, 4.0]])
        core._action_term_dof_indices = torch.tensor([2, 0, 3, 1])

        action_input = core._mjlab_action_input()

        torch.testing.assert_close(action_input, torch.tensor([[33.0, 11.0, 41.0, 21.0]]))

    def test_native_and_core_action_scales_preserve_precision_joint_order_and_effort_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            training = load_robot_training_spec(_write_tiny_robot_with_training(Path(tmpdir))).to_env_dict()
            env = MJLabEnvConfig(device="cpu", motion_data_path="unused.pkl", robot_training=training)
            task = resolve_task_config(env, 1)
            task.robot.control.stiffness = {"2": 7.3, "joint": 3.7}
            task.robot.control.action_scale = 0.2718281828459045
            efforts, gains = {"joint1": 10.1, "joint2": 13.9}, {"joint1": 3.7, "joint2": 7.3}
            for order in (("joint1", "joint2"), ("joint2", "joint1")):
                for rescale in (False, True):
                    for effort_scale in (1.0, 3.0):
                        with self.subTest(order=order, rescale=rescale, effort_scale=effort_scale):
                            task.robot.dof_names = list(order)
                            task.robot.dof_effort_limit_list = [efforts[name] for name in order]
                            task.robot.dof_effort_limit_scale = effort_scale
                            task.robot.control.action_rescale = rescale
                            expected = [
                                0.2718281828459045 * (efforts[name] / gains[name] if rescale else 1.0) for name in order
                            ]
                            native = make_mjlab_ufo_env_cfg(
                                task, num_envs=1, seed=1, mjcf_path=training["robot"]["xml_path"], auto_reset=False, robot_training=training
                            )
                            self.assertEqual(list(native.actions["actions"].scale), list(order))
                            self.assertEqual(list(native.actions["actions"].scale.values()), expected)
                            core_scale = _action_target_scale(task)
                            self.assertEqual(core_scale.dtype, torch.float32)
                            torch.testing.assert_close(core_scale, torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0)
                            # Native action config retains Python float precision; Core converts only at its tensor boundary.
                            self.assertNotEqual(core_scale[0].item(), expected[0])
                            native_efforts = [actuator.effort_limit for actuator in native.scene.entities["robot"].articulation.actuators]
                            self.assertEqual(native_efforts, [training["actuator"]["joints"][name]["effort_limit"] for name in order])

    def test_action_rescale_rejects_nonpositive_stiffness_but_fixed_scale_accepts_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            training = load_robot_training_spec(_write_tiny_robot_with_training(Path(tmpdir))).to_env_dict()
            env = MJLabEnvConfig(device="cpu", motion_data_path="unused.pkl", robot_training=training)
            task = resolve_task_config(env, 1)
            for stiffness in (0.0, -1.0):
                with self.subTest(stiffness=stiffness):
                    task.robot.control.stiffness["joint1"] = stiffness
                    task.robot.control.action_rescale = True
                    message = f"Cannot action_rescale joint joint1: stiffness={stiffness}"
                    with self.assertRaisesRegex(ValueError, message):
                        _action_target_scale(task)
                    with self.assertRaisesRegex(ValueError, message):
                        make_mjlab_ufo_env_cfg(
                            task, num_envs=1, seed=1, mjcf_path=training["robot"]["xml_path"], auto_reset=False, robot_training=training
                        )
                    task.robot.control.action_rescale = False
                    torch.testing.assert_close(_action_target_scale(task), torch.full((2,), 0.25), rtol=0, atol=0)
                    native = make_mjlab_ufo_env_cfg(
                        task, num_envs=1, seed=1, mjcf_path=training["robot"]["xml_path"], auto_reset=False, robot_training=training
                    )
                    self.assertEqual(native.actions["actions"].scale, {"joint1": 0.25, "joint2": 0.25})

    def test_yaml_actuator_missing_joint_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_with_training(Path(tmpdir), missing_actuator_joint=True)
            with self.assertRaisesRegex(ValueError, "missing parameters for joint 'joint2'"):
                load_robot_training_spec(tiny_robot)

    def test_tracking_shapes_follow_num_dof(self) -> None:
        obs = {
            "ref_body_pos": torch.zeros(4, 1, 3),
            "ref_body_rots": torch.zeros(4, 1, 4),
            "ref_body_vels": torch.zeros(4, 1, 3),
            "ref_body_angular_vels": torch.zeros(4, 1, 3),
            "dof_pos": torch.zeros(4, 2),
            "ref_dof_vel": torch.ones(4, 2),
        }
        obs["ref_body_rots"][..., 3] = 1.0
        qpos = _expert_qpos_from_obs(obs, num_dof=2, dof_qpos_order_indices=torch.tensor([0, 1]).numpy())
        self.assertEqual(qpos.shape, (4, 9))
        target = _target_states_from_obs(obs, device="cpu", num_dof=2)
        self.assertEqual(tuple(target["dof_states"].shape), (1, 2, 2))


@pytest.mark.parametrize("ranges", [{"unknown": [0.0, 1.0]}, {"joint1": [float("nan"), 1.0]}, {"joint1": [2.0, 1.0]}])
def test_native_training_joint_limits_are_validated(tmp_path, ranges):
    from omegaconf import OmegaConf

    from ufo.tools.robot_config import load_robot_training_spec

    path = _write_tiny_robot_with_training(tmp_path)
    cfg = OmegaConf.load(path)
    cfg.training.joint_ranges = ranges
    OmegaConf.save(cfg, path)
    with pytest.raises(ValueError, match="training.joint_ranges"):
        load_robot_training_spec(path)

def test_training_reads_robot_yaml_once_and_warns_for_draft(tmp_path, capsys):
    from omegaconf import OmegaConf
    raw = OmegaConf.load("configs/robots/g1_29dof.yaml")
    raw.xml_path = str(Path(raw.xml_path).resolve())
    path = tmp_path / "draft.yaml"
    raw.metadata = {"review_status": "draft"}
    OmegaConf.save(raw, path)
    with patch.object(OmegaConf, "load", wraps=OmegaConf.load) as load:
        build_mjlab_config(
            device="cpu", work_dir=str(tmp_path), num_envs=2, num_env_steps=16,
            seed=1, use_wandb=False, wandb_run_name=None, robot_config=path, data_path="matching.pkl",
        )
    load.assert_called_once_with(path.resolve())
    assert "auto-generated draft" in capsys.readouterr().out


if __name__ == "__main__":
    unittest.main()
