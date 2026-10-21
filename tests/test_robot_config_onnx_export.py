from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from ufo.envs.config import MJLabEnvConfig
from ufo.tools.export_onnx import _deployment_contract, export_meta_policy_as_onnx, export_policy_model
from ufo.tools.robot_config import load_robot_training_spec


class _FakeSpace:
    def __init__(self, dim: int) -> None:
        self.shape = (int(dim),)


class _FakeInputFilter:
    def __init__(self, dim: int) -> None:
        self.output_space = _FakeSpace(dim)


class _FakeActor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int) -> None:
        super().__init__()
        self.input_filter = _FakeInputFilter(input_dim)
        self.action_dim = action_dim

    def forward(self, actor_dict, ctx, std):
        self.last_actor_dict_shapes = {key: tuple(value.shape) for key, value in actor_dict.items()}
        self.last_ctx_shape = tuple(ctx.shape)
        return SimpleNamespace(mean=torch.zeros(ctx.shape[0], self.action_dim, dtype=ctx.dtype, device=ctx.device))


class _FakePolicyModel(nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        last_action_dim: int,
        history_actor_dim: int = 0,
        z_dim: int = 16,
        action_dim: int = 29,
        actor_input_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        if actor_input_keys is None:
            actor_input_keys = ["state", "last_action"]
        self.obs_space = {
            "state": _FakeSpace(state_dim),
            "last_action": _FakeSpace(last_action_dim),
            "history_actor": _FakeSpace(history_actor_dim),
        }
        self.action_dim = int(action_dim)
        actor_input_dim = sum(int(self.obs_space[key].shape[0]) for key in actor_input_keys)
        self._actor = _FakeActor(actor_input_dim, action_dim)
        self._obs_normalizer = nn.Identity()
        self.cfg = SimpleNamespace(
            actor_std=0.05,
            archi=SimpleNamespace(
                z_dim=int(z_dim),
                actor=SimpleNamespace(input_filter=SimpleNamespace(key=list(actor_input_keys))),
            )
        )
        self.last_actor_dict_shapes: dict[str, tuple[int, ...]] = {}
        self.last_ctx_shape: tuple[int, ...] | None = None



def _fake_robot_training(num_dof: int) -> SimpleNamespace:
    return SimpleNamespace(
        config_path=Path("/tmp/fake_robot.yaml"),
        robot=SimpleNamespace(
            name="fake_robot",
            xml_path="/tmp/fake_robot.xml",
            control_joint_names=[f"joint_{idx}" for idx in range(num_dof)],
        ),
    )


class RobotConfigOnnxExportTest(unittest.TestCase):
    def test_deployment_contract_uses_saved_observation_and_motor_settings(self):
        root = Path(__file__).resolve().parents[1]
        spec = load_robot_training_spec(root / 'configs/robots/g1_29dof.yaml')
        snapshot = spec.to_env_dict()
        gain_key = next(key for key in snapshot['stiffness'] if key in 'waist_yaw_joint')
        snapshot['stiffness'][gain_key] = 321.
        env = MJLabEnvConfig(motion_data_path='unused.pkl', robot_training=snapshot,
            task_overrides={'obs': {'obs_scales': {'base_ang_vel': .5, 'dof_pos': 2., 'history_actor': 3.}}})
        model = _FakePolicyModel(state_dim=64, last_action_dim=29, history_actor_dim=372,
            action_dim=29, actor_input_keys=['state', 'last_action', 'history_actor'])
        contract = _deployment_contract(model, env)
        obs = contract['deployment']['observation']['policy']
        self.assertEqual(obs['base_ang_vel']['scale'], .5)
        self.assertEqual(obs['base_ang_vel_history']['scale'], 1.5)
        self.assertEqual(obs['dof_pos_minus_default_history']['scale'], 6.)
        self.assertEqual(contract['observation_order'][:5], ['dof_pos_minus_default', 'dof_vel', 'projected_gravity', 'base_ang_vel', 'prev_actions'])
        self.assertFalse(contract['history_includes_current'])
        self.assertFalse(contract['history_oldest_first'])
        self.assertEqual(contract['deployment']['joint_kp']['waist_yaw_joint'], 321.)

    def test_rp1_export_preserves_oldest_first_history(self):
        root = Path(__file__).resolve().parents[1]
        spec = load_robot_training_spec(root / "configs/robots/rp1_3.yaml")
        env = MJLabEnvConfig(motion_data_path="unused.pkl", robot_training=spec.to_env_dict(), task_overrides=spec.task_overrides)
        model = _FakePolicyModel(state_dim=54, last_action_dim=24, history_actor_dim=312,
                                 action_dim=24, actor_input_keys=["state", "last_action", "history_actor"])
        with tempfile.TemporaryDirectory() as tmpdir, patch("ufo.tools.export_onnx.torch.onnx.export"):
            metadata = export_policy_model(model, Path(tmpdir), spec, env_config=env)
            saved = json.loads((Path(tmpdir) / "_FakePolicyModel.meta.json").read_text())
            self.assertTrue((Path(tmpdir) / "_FakePolicyModel.deploy.yaml").is_file())
        self.assertTrue(metadata["history_oldest_first"])
        self.assertTrue(saved["history_oldest_first"])
        self.assertFalse(saved["history_includes_current"])
        self.assertEqual(saved["control_joint_names"], list(spec.robot.control_joint_names))

    def test_invalid_contract_fails_before_writing_policy(self):
        model = _FakePolicyModel(state_dim=64, last_action_dim=29)
        with tempfile.TemporaryDirectory() as tmpdir, \
             patch("ufo.tools.export_onnx._deployment_contract", side_effect=ValueError("invalid contract")), \
             patch("ufo.tools.export_onnx.torch.onnx.export") as export:
            with self.assertRaisesRegex(ValueError, "invalid contract"):
                export_policy_model(model, Path(tmpdir), _fake_robot_training(29), env_config=object())
            export.assert_not_called()
            self.assertEqual(list(Path(tmpdir).iterdir()), [])

    def _export_with_mock(self, model: _FakePolicyModel, output_name: str = "FakePolicyModel.onnx"):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        patcher = patch("ufo.tools.export_onnx.torch.onnx.export")
        onnx_export = patcher.start()
        self.addCleanup(patcher.stop)
        metadata = export_meta_policy_as_onnx(model, Path(tmpdir.name), output_name)
        return metadata, onnx_export

    def test_g1_29dof_wrapper_uses_inferred_dimensions(self) -> None:
        model = _FakePolicyModel(state_dim=64, last_action_dim=29, z_dim=32, action_dim=29)
        metadata, onnx_export = self._export_with_mock(model)

        self.assertEqual(metadata["actor_input_keys"], ["state", "last_action"])
        self.assertEqual(metadata["actor_input_dims"], {"state": 64, "last_action": 29})
        self.assertEqual(metadata["actor_obs_dim"], 125)
        wrapper = onnx_export.call_args.args[0]
        example_input = onnx_export.call_args.args[1]
        self.assertEqual(tuple(example_input.shape), (1, 125))

        action = wrapper(torch.randn(2, 125))
        self.assertEqual(tuple(action.shape), (2, 29))
        self.assertEqual(wrapper.actor.last_actor_dict_shapes["state"], (2, 64))
        self.assertEqual(wrapper.actor.last_actor_dict_shapes["last_action"], (2, 29))
        self.assertEqual(wrapper.actor.last_ctx_shape, (2, 32))

    def test_non_g1_action_dims_export_without_g1_only_error(self) -> None:
        for action_dim in (23, 27):
            model = _FakePolicyModel(state_dim=40 + action_dim, last_action_dim=action_dim, action_dim=action_dim)
            metadata, _onnx_export = self._export_with_mock(model, output_name=f"FakePolicyModel{action_dim}.onnx")
            self.assertEqual(metadata["output_action_dim"], action_dim)
            self.assertEqual(metadata["actor_input_dims"]["last_action"], action_dim)

    def test_export_meta_policy_signature_does_not_accept_use_29dof(self) -> None:
        signature = inspect.signature(export_meta_policy_as_onnx)
        self.assertNotIn("use_29dof", signature.parameters)
        model = _FakePolicyModel(state_dim=64, last_action_dim=29)
        with self.assertRaises(TypeError):
            export_meta_policy_as_onnx(model, "/tmp", "policy.onnx", use_29dof=True)  # type: ignore[call-arg]

    def test_history_actor_key_contributes_to_actor_obs_dim(self) -> None:
        model = _FakePolicyModel(
            state_dim=52,
            last_action_dim=23,
            history_actor_dim=112,
            z_dim=24,
            action_dim=23,
            actor_input_keys=["state", "last_action", "history_actor"],
        )
        metadata, onnx_export = self._export_with_mock(model)

        self.assertEqual(
            metadata["actor_input_dims"],
            {"state": 52, "last_action": 23, "history_actor": 112},
        )
        self.assertEqual(metadata["actor_obs_dim"], 52 + 23 + 112 + 24)
        wrapper = onnx_export.call_args.args[0]
        wrapper(torch.randn(3, metadata["actor_obs_dim"]))
        self.assertEqual(wrapper.actor.last_actor_dict_shapes["history_actor"], (3, 112))

    def test_missing_actor_input_key_errors_clearly(self) -> None:
        model = _FakePolicyModel(
            state_dim=52,
            last_action_dim=23,
            history_actor_dim=112,
            actor_input_keys=["state", "last_action", "history_actor"],
        )
        del model.obs_space["history_actor"]
        with self.assertRaisesRegex(KeyError, "history_actor.*obs_space"):
            export_meta_policy_as_onnx(model, "/tmp", "policy.onnx")

    def test_tracking_inference_has_no_g1_only_num_dof_export_block(self) -> None:
        source = Path(__file__).resolve().parents[1] / "ufo" / "inference" / "tracking_inference.py"
        text = source.read_text()
        self.assertNotIn("num_dof != 29", text)
        self.assertNotIn("ONNX export currently supports only G1 29-DOF", text)

    def test_goal_and_reward_share_the_exporter(self) -> None:
        from types import SimpleNamespace
        from unittest.mock import patch

        from ufo.inference.goal_inference import export_policy_model as goal_export
        from ufo.inference.reward_inference import export_policy_model as reward_export
        self.assertIs(goal_export, reward_export)
        model = SimpleNamespace(cfg=SimpleNamespace(archi=SimpleNamespace(z_dim=12)))
        with tempfile.TemporaryDirectory() as tmpdir, patch("ufo.tools.export_onnx.export_meta_policy_as_onnx") as export:
            export.return_value = {"output_action_dim": 27}
            goal_export(model, Path(tmpdir), _fake_robot_training(27))
            export.assert_called_once_with(model, Path(tmpdir), "SimpleNamespace.onnx", z_dim=12)

    def test_tracking_policy_export_writes_robot_metadata_json(self) -> None:
        model = _FakePolicyModel(
            state_dim=52,
            last_action_dim=27,
            history_actor_dim=100,
            z_dim=12,
            action_dim=27,
            actor_input_keys=["state", "last_action", "history_actor"],
        )
        with tempfile.TemporaryDirectory() as tmpdir, patch("ufo.tools.export_onnx.torch.onnx.export"):
            metadata = export_policy_model(model, Path(tmpdir), _fake_robot_training(27))
            metadata_path = Path(tmpdir) / "_FakePolicyModel.meta.json"
            self.assertTrue(metadata_path.exists())
            saved = json.loads(metadata_path.read_text())

        self.assertEqual(metadata["robot_name"], "fake_robot")
        self.assertEqual(saved["robot_name"], "fake_robot")
        self.assertEqual(saved["num_dof"], 27)
        self.assertEqual(saved["control_joint_names"][0], "joint_0")
        self.assertEqual(saved["actor_input_dims"]["last_action"], 27)
        self.assertEqual(saved["z_dim"], 12)
        self.assertEqual(saved["actor_obs_dim"], 52 + 27 + 100 + 12)
        self.assertEqual(saved["output_action_dim"], 27)


if __name__ == "__main__":
    unittest.main()
