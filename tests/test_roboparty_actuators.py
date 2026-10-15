"""CPU checks for the RoboParty motor mapping and command-delay contract."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch
from mjlab.actuator import ActuatorCmd, DcMotorActuatorCfg
from mjlab.actuator.dc_actuator import dc_motor_clip
from mjlab.actuator.fused_group import FusedActuatorGroup
from mjlab.entity import Entity
from omegaconf import OmegaConf

from ufo.envs.actuators import RoboPartyActuatorCfg
from ufo.envs.config import MJLabEnvConfig, _action_target_scale_values, make_mjlab_ufo_env_cfg, resolve_task_config
from ufo.tools.robot_config import load_robot_training_spec


class RoboPartyActuatorTest(unittest.TestCase):
    def test_four_quadrant_peak_plateau_and_overspeed_braking(self):
        for peak, speed, keep in ((145.53, 13.61357, 8.37758), (56.0, 20.94395, 15.70796), (28.0, 20.94395, 15.70796)):
            peak, speed, keep = (torch.tensor(v, dtype=torch.float64) for v in (peak, speed, keep))
            saturation = peak / (1 - keep / speed)
            corner = speed * (1 + peak / saturation)
            velocity = torch.stack((speed * 0, keep, speed, (speed + corner) / 2, corner, corner * 2))
            positive = dc_motor_clip(torch.full_like(velocity, 1e6), saturation, speed, peak, velocity)
            expected = torch.stack((peak, peak, peak * 0, -peak / 2, -peak, -peak))
            torch.testing.assert_close(positive, expected)
            negative = dc_motor_clip(torch.full_like(velocity, -1e6), saturation, speed, peak, velocity)
            torch.testing.assert_close(negative, -peak.expand_as(velocity))
            reverse = dc_motor_clip(torch.full_like(velocity, -1e6), saturation, speed, peak, -velocity)
            torch.testing.assert_close(reverse, -expected)
            coast = dc_motor_clip(torch.zeros_like(velocity), saturation, speed, peak, velocity)
            torch.testing.assert_close(coast, torch.minimum(expected, torch.zeros_like(expected)))

    def test_fused_delay_holds_until_partial_reset_and_never_delays_feedback(self):
        acts = []
        for index in range(2):
            cfg = RoboPartyActuatorCfg(
                target_names_expr=(f"joint{index}",),
                stiffness=1.0,
                damping=0.1,
                effort_limit=2.0,
                saturation_effort=3.0,
                velocity_limit=10.0,
                delay_min_lag=0,
                delay_max_lag=2,
            )
            act = cfg.build(None, [index], [f"joint{index}"])
            act._target_ids = torch.tensor([index])
            act._ctrl_ids = torch.tensor([index])
            for name in act.param_names:
                setattr(act, name, torch.ones(64, 1))
            acts.append(act)
        group, remaining = FusedActuatorGroup.process(acts)
        self.assertFalse(remaining)
        group.initialize(64, "cpu")
        self.assertIs(acts[0]._delay_buffer, acts[1]._delay_buffer)
        torch.manual_seed(123)
        for act in acts:
            act.reset()
        before = acts[0]._delay_buffer.current_lags.clone()
        self.assertEqual(set(before.tolist()), {0, 1, 2})
        for step in range(6):
            target = torch.full((64, 2), float(step))
            feedback = target + 100
            cmd = ActuatorCmd(target, target + 10, target + 20, feedback, feedback)
            delayed = acts[0].apply_delay(cmd)
            expected = (step - before).clamp_min(0).float()[:, None].expand(-1, 2)
            torch.testing.assert_close(delayed.position_target, expected)
            torch.testing.assert_close(delayed.velocity_target, expected + 10)
            torch.testing.assert_close(delayed.effort_target, expected + 20)
            self.assertIs(delayed.pos, feedback)
            self.assertIs(delayed.vel, feedback)
            torch.testing.assert_close(acts[0]._delay_buffer.current_lags, before)
        reset_ids = torch.arange(0, 64, 2)
        for act in acts:
            act.reset(reset_ids)
        torch.testing.assert_close(acts[0]._delay_buffer.current_lags[1::2], before[1::2])
        self.assertTrue(torch.any(acts[0]._delay_buffer.current_lags[reset_ids] != before[reset_ids]))

    def test_robot_config_mapping_action_scaling_and_passive_friction(self):
        g1_path = Path(__file__).parents[1] / "configs/robots/g1_29dof.yaml"
        g1 = load_robot_training_spec(g1_path)
        config = OmegaConf.to_container(OmegaConf.load(g1_path), resolve=True)
        config["xml_path"] = g1.robot.xml_path
        control = config["training"]["control"]
        control.update(action_scale=0.4, action_rescale=False, normalize_action_to=8.0, action_clip_value=8.0)
        params = dict(
            effort_limit=56.0,
            velocity_limit=20.94395,
            keep_velocity=15.70796,
            rated_effort=19.0,
            rated_velocity=6.28319,
            armature=0.01,
            friction=0.1,
            viscous_friction=0.02,
        )
        config["training"]["actuator"] = dict(
            source="yaml", model="roboparty", min_delay=0, max_delay=2, joints={name: dict(params) for name in g1.robot.control_joint_names}
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "robot.yaml"
            OmegaConf.save(config=OmegaConf.create(config), f=path)
            spec = load_robot_training_spec(path)
            env = MJLabEnvConfig(motion_data_path="unused", robot_training=spec.to_env_dict())
            task = resolve_task_config(env, 1)
            self.assertEqual(_action_target_scale_values(task), [0.4] * len(g1.robot.control_joint_names))
            policy = torch.tensor([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
            offsets = torch.clamp(policy * task.robot.control.normalize_action_to, -8.0, 8.0) * 0.4
            torch.testing.assert_close(offsets, torch.tensor([-3.2, -3.2, 0.0, 1.6, 3.2, 3.2]))
            scene = make_mjlab_ufo_env_cfg(
                task, num_envs=1, seed=1, mjcf_path=spec.robot.xml_path, auto_reset=False, robot_training=spec.to_env_dict()
            )
            robot = scene.scene.entities["robot"]
            for actuator in robot.articulation.actuators:
                self.assertIsInstance(actuator, RoboPartyActuatorCfg)
                self.assertAlmostEqual(actuator.saturation_effort, 56.0 / (1 - 15.70796 / 20.94395))
                self.assertEqual((actuator.delay_min_lag, actuator.delay_max_lag, actuator.delay_hold_prob), (0, 2, 1.0))
            model = Entity(robot).spec.compile()
            for name in spec.robot.control_joint_names:
                dof = model.joint(name).dofadr[0]
                self.assertAlmostEqual(model.dof_frictionloss[dof], 0.1)
                self.assertAlmostEqual(model.dof_damping[dof], 0.02)
                self.assertAlmostEqual(model.dof_armature[dof], 0.01)
            for key, value in (
                ("keep_velocity", 20.94395),
                ("rated_effort", 57),
                ("rated_velocity", 21),
                ("friction", -0.1),
                ("viscous_friction", float("nan")),
            ):
                bad = copy.deepcopy(config)
                bad["training"]["actuator"]["joints"][g1.robot.control_joint_names[0]][key] = value
                OmegaConf.save(config=OmegaConf.create(bad), f=path)
                with self.assertRaisesRegex(ValueError, key):
                    load_robot_training_spec(path)

    def test_g1_keeps_native_dc_motor_and_effort_rescaling(self):
        path = Path(__file__).parents[1] / "configs/robots/g1_29dof.yaml"
        spec = load_robot_training_spec(path)
        self.assertTrue(spec.action_rescale)
        env = MJLabEnvConfig(motion_data_path="unused", robot_training=spec.to_env_dict())
        task = resolve_task_config(env, 1)
        scene = make_mjlab_ufo_env_cfg(
            task, num_envs=1, seed=1, mjcf_path=spec.robot.xml_path, auto_reset=False, robot_training=spec.to_env_dict()
        )
        for actuator, scale, effort in zip(
            scene.scene.entities["robot"].articulation.actuators, _action_target_scale_values(task), spec.effort_limits
        ):
            self.assertIs(type(actuator), DcMotorActuatorCfg)
            self.assertEqual(actuator.saturation_effort, actuator.effort_limit)
            self.assertEqual(actuator.delay_max_lag, 0)
            self.assertIsNone(actuator.viscous_damping)
            self.assertAlmostEqual(scale, spec.action_scale * effort / actuator.stiffness)


if __name__ == "__main__":
    unittest.main()
