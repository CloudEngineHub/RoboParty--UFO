"""Check RP1's startup parameter mapping and velocity replacement semantics."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ufo.envs.config import (
    MJLabEnvConfig,
    _push_by_replacing_velocity,
    _randomize_mass_and_inertia,
    make_mjlab_ufo_env_cfg,
    resolve_task_config,
)
from ufo.tools.robot_config import load_robot_training_spec


class RoboPartyDomainRandomizationTest(unittest.TestCase):
    def test_push_replaces_all_six_velocity_axes_for_selected_worlds(self):
        robot = SimpleNamespace(write_root_com_velocity_to_sim=Mock())
        env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot": robot})
        ids = torch.tensor([0, 2])
        ranges = {name: (float(index), float(index)) for index, name in enumerate(("x", "y", "z", "roll", "pitch", "yaw"))}
        _push_by_replacing_velocity(env, ids, ranges)
        call = robot.write_root_com_velocity_to_sim.call_args
        torch.testing.assert_close(call.args[0], torch.arange(6, dtype=torch.float32).repeat(2, 1))
        self.assertIs(call.kwargs["env_ids"], ids)

    def test_mass_scaling_updates_inertia_without_touching_other_worlds_or_bodies(self):
        default_mass = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mass = default_mass.repeat(3, 1)
        inertia = torch.ones(3, 4, 3)
        model = SimpleNamespace(body_mass=mass, body_inertia=inertia)
        robot = SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([1, 2, 3], dtype=torch.int)))
        env = SimpleNamespace(
            num_envs=3,
            device="cpu",
            scene={"robot": robot},
            sim=SimpleNamespace(model=model, get_default_field=lambda name: default_mass, per_world_default_fields=set()),
        )
        with self.assertWarns(UserWarning):
            _randomize_mass_and_inertia(env, torch.tensor([1]), (2.0, 2.0), SceneEntityCfg("robot", body_ids=[1]), "scale")
        expected_mass = default_mass.repeat(3, 1)
        expected_mass[1, 2] = 6.0
        torch.testing.assert_close(mass, expected_mass)
        expected_inertia = torch.ones_like(inertia)
        expected_inertia[1, 2] = 2.0
        torch.testing.assert_close(inertia, expected_inertia)
        with self.assertWarns(UserWarning):
            _randomize_mass_and_inertia(env, torch.tensor([1]), (-1.0, -1.0), SceneEntityCfg("robot", body_ids=[1]), "add")
        self.assertEqual(mass[1, 2], 2.0)
        torch.testing.assert_close(inertia[1, 2], torch.full((3,), 2.0 / 3.0))

    def test_rp1_preset_resolves_every_event_and_keeps_historic_physics(self):
        path = Path(__file__).parents[1] / "configs/robots/rp1_3.yaml"
        spec = load_robot_training_spec(path)
        env = MJLabEnvConfig(motion_data_path="unused", robot_training=spec.to_env_dict(), task_overrides=spec.task_overrides)
        task = resolve_task_config(env, 4)
        cfg = make_mjlab_ufo_env_cfg(
            task, num_envs=4, seed=1, mjcf_path=spec.robot.xml_path, auto_reset=False, robot_training=spec.to_env_dict()
        )
        robot = Entity(cfg.scene.entities["robot"])
        model = robot.spec.compile()
        self.assertEqual(model.nu, 24)
        for event in cfg.events.values():
            if "asset_cfg" in event.params:
                event.params["asset_cfg"].resolve({"robot": robot})
        events = cfg.events
        self.assertEqual(
            events["random_link_mass"].params["asset_cfg"].body_names,
            [name for name in robot.body_names if name.startswith(("left_", "right_"))],
        )
        self.assertEqual(events["random_base_mass"].params["asset_cfg"].body_names, ["waist_yaw_link"])
        self.assertEqual(events["random_pd_gain"].params["kp_range"], (0.8, 1.2))
        self.assertEqual(events["random_joint_friction"].params["ranges"], (0.8, 1.2))
        self.assertEqual(events["random_joint_armature"].params["ranges"], (0.8, 1.2))
        self.assertIs(events["push_robots"].func, _push_by_replacing_velocity)
        self.assertEqual(events["push_robots"].params["velocity_range"]["yaw"], (-0.78, 0.78))
        self.assertEqual(cfg.sim.mujoco.tolerance, 1e-6)
        self.assertEqual(cfg.sim.mujoco.iterations, 100)
        self.assertEqual(cfg.sim.mujoco.ls_iterations, 50)
        self.assertEqual(cfg.sim.mujoco.integrator, "implicitfast")
        self.assertEqual(cfg.sim.mujoco.cone, "pyramidal")
        self.assertEqual([sensor.name for sensor in cfg.scene.sensors], ["body_contact", "normal_contact"])


if __name__ == "__main__":
    unittest.main()
