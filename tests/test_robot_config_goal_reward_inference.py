from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mujoco
import numpy as np
import pytest
import torch

from ufo.agents.common.buffers.trajectory import ReplayBuffer, get_idxs
from ufo.inference.goal_inference import (
    _find_goal_json,
    load_and_validate_goal_json,
)
from ufo.inference.goal_inference import (
    parse_args as parse_goal_args,
)
from ufo.inference.reward_inference import (
    DEFAULT_TASKS,
    RewardInference,
    _is_non_g1_locomotion_task,
    _load_replay_buffer,
    _resolve_reward_tasks,
    write_g1_mjlab_relabel_xml,
    write_mjlab_relabel_xml,
)
from ufo.inference.runtime import MujocoQposRenderer, _target_states_from_obs, render_policy_frame, video_writer
from ufo.tools.evaluate_mae import evaluate_goal_joint_mae
from ufo.tools.robot_config import load_robot_training_spec

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reward_export_contains_numpy_and_still_runs_rollout(tmp_path, monkeypatch):
    import joblib

    from ufo.inference import reward_inference as module

    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint/'config.json').write_text(json.dumps({'model': {'name': 'FBcprAuxModel'}}))
    spec = load_robot_training_spec(REPO_ROOT/'configs/robots/g1_29dof.yaml')
    env = Mock()
    env.reset.return_value = ({'state': torch.zeros(1,64)}, {})
    env.step.return_value = ({'state': torch.zeros(1,64)}, 0, torch.tensor([False]), torch.tensor([False]), {})
    cfg = SimpleNamespace(build=lambda **kw: (env, {}), mjcf_path=spec.robot.xml_path)
    model = Mock()
    monkeypatch.setattr(module, 'load_mjlab_env_cfg', lambda *a, **kw: (cfg, spec))
    monkeypatch.setattr(module, 'load_model_from_checkpoint_dir', lambda *a, **kw: model)
    monkeypatch.setattr(module, '_load_replay_buffer', lambda *a, **kw: (object(), tmp_path))
    monkeypatch.setattr(module, 'write_g1_mjlab_relabel_xml', lambda *a: Path(spec.robot.xml_path))
    monkeypatch.setattr(module, 'RewardInference', lambda **kw: SimpleNamespace(reward_inference=lambda **kw: torch.ones(1,256)))
    monkeypatch.setattr(module, '_default_standing_target_states', lambda *a, **kw: {})
    module.run_reward_inference(model_folder=tmp_path, data_path=None, robot_config=None,
        device='cpu', save_mp4=False, disable_dr=True, disable_obs_noise=True,
        episode_length=1, num_samples=1, n_inferences=1, skip_rollouts=False,
        tasks=['stand'], buffer_rank=0, buffer_path=None, max_workers=1,
        process_executor=False, render_size=32, camera_distance=3., camera_azimuth=0.,
        camera_elevation=0., fps=50, max_episode_length_s=1., export_onnx=False)
    saved = joblib.load(tmp_path/'reward_inference/reward_locomotion.pkl')
    assert isinstance(saved['stand'][0], np.ndarray)
    np.testing.assert_array_equal(saved['stand'][0], np.ones((1,256)))
    assert isinstance(model.act.call_args.args[1], torch.Tensor)
    env.close.assert_called_once()


def test_video_stream_encodes_in_order_and_preserves_previous_output_on_failure(tmp_path):
    import mediapy as media

    destination = tmp_path / "rollout.mp4"
    with video_writer(destination, shape=(32, 64), fps=10) as writer:
        for level in (0, 64, 128, 255):
            writer.add_image(np.full((32, 64, 3), level, dtype=np.uint8))
            assert not destination.exists()
    frames = media.read_video(str(destination))
    assert frames.shape == (4, 32, 64, 3)
    np.testing.assert_allclose(frames.mean(axis=(1, 2, 3)), [0, 64, 128, 255], atol=3)
    original = destination.read_bytes()
    with pytest.raises(RuntimeError, match="rollout failed"):
        with video_writer(destination, shape=(32, 64), fps=10) as writer:
            writer.add_image(np.zeros((32, 64, 3), dtype=np.uint8))
            raise RuntimeError("rollout failed")
    assert destination.read_bytes() == original
    with pytest.raises(ValueError, match="dimensions"):
        with video_writer(destination, shape=(32, 64), fps=10) as writer:
            writer.add_image(np.zeros((16, 16, 3), dtype=np.uint8))
    assert destination.read_bytes() == original
    with video_writer(None, shape=(32, 64), fps=10) as writer:
        assert writer is None
    assert list(tmp_path.iterdir()) == [destination]


def _write_tiny_robot_config(root: Path) -> Path:
    xml_path = root / "tiny_robot.xml"
    xml_path.write_text(
        """
<mujoco model="tiny_robot">
  <worldbody>
    <body name="base" pos="0 0 0.8">
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
    <motor name="old_joint1_motor" joint="joint1"/>
  </actuator>
</mujoco>
""".strip()
    )
    robot_config = root / "tiny_robot.yaml"
    robot_config.write_text(
        "\n".join(
            [
                "name: tiny_robot",
                "xml_path: tiny_robot.xml",
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
            ]
        )
    )
    return robot_config


def _fake_robot_training(name: str, num_dof: int) -> SimpleNamespace:
    return SimpleNamespace(
        robot=SimpleNamespace(
            name=name,
            control_joint_names=[f"joint_{idx}" for idx in range(num_dof)],
        )
    )


class RobotConfigGoalRewardInferenceTest(unittest.TestCase):
    def test_policy_frames_use_qpos_and_own_valid_rgb_pixels(self) -> None:
        renderer = MujocoQposRenderer.__new__(MujocoQposRenderer)
        renderer.model = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><body><freejoint/><geom type="sphere" size="0.1"/></body></worldbody></mujoco>'
        )
        renderer.data = mujoco.MjData(renderer.model)
        renderer.camera = mujoco.MjvCamera()
        pixels = np.arange(72, dtype=np.uint8).reshape(4, 6, 3)[:, ::-1]
        renderer.renderer = SimpleNamespace(update_scene=Mock(), render=Mock(return_value=pixels))
        qpos = np.asarray([[1, 2, 0.5, 1, 0, 0, 0]], dtype=np.float64)
        env = SimpleNamespace(_get_qpos_qvel=Mock(return_value=(qpos, None)), render=Mock())
        expected = pixels.copy()
        actual = render_policy_frame(env, renderer)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(renderer.data.qpos, qpos[0])
        np.testing.assert_array_equal(renderer.camera.lookat, [1, 2, 0.75])
        self.assertEqual(actual.dtype, np.uint8)
        self.assertTrue(actual.flags.c_contiguous)
        pixels[:] = 0
        np.testing.assert_array_equal(actual, expected)
        env.render.assert_not_called()
        for invalid in (None, {}, np.zeros((4, 6)), np.zeros((4, 6, 4)), np.zeros((4, 6, 3), dtype=np.float32)):
            with self.subTest(frame=type(invalid)), self.assertRaisesRegex(ValueError, "MuJoCo RGB frame"):
                renderer.renderer.render.return_value = invalid
                render_policy_frame(env, renderer)

    def test_reward_replay_defaults_to_requested_rank_and_never_legacy_buffers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            buffers = root / "checkpoint" / "buffers"
            for index, name in enumerate(("train_reduced", "train", "train_rank_0", "train_rank_3")):
                dataset = ReplayBuffer(capacity=4, n_dim=2, output_key_tp1=[])
                dataset.extend({"observation": torch.full((2, 2, 1), float(index)), "done": torch.zeros(2, 2, 1, dtype=torch.bool)})
                dataset.save(buffers / name)
                (buffers / name / "snapshot.json").write_text(json.dumps({"id": name, "global_time": 16, "world_size": 8}))
            for rank, value in ((0, 2.0), (3, 3.0)):
                dataset, selected = _load_replay_buffer(root, buffer_rank=rank, buffer_path=None)
                self.assertEqual(selected, buffers / f"train_rank_{rank}")
                torch.testing.assert_close(dataset.get_full_buffer()["observation"], torch.full((4, 1), value), rtol=0, atol=0)
            with self.assertRaises(FileNotFoundError):
                _load_replay_buffer(root, buffer_rank=1, buffer_path=None)
            dataset, selected = _load_replay_buffer(root, buffer_rank=0, buffer_path=buffers / "train_reduced")
            self.assertEqual(selected, buffers / "train_reduced")
            torch.testing.assert_close(dataset.get_full_buffer()["observation"], torch.zeros(4, 1), rtol=0, atol=0)

    def test_reward_replay_accepts_complete_explicit_trajectory_without_matching_model_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            replay_path = root / "other_replay"
            dataset = ReplayBuffer(capacity=4, n_dim=2)
            dataset.extend({"observation": {"x": torch.ones(3, 2, 1)}, "done": torch.zeros(3, 2, 1, dtype=torch.bool)})
            dataset.save(replay_path)
            (replay_path / "snapshot.json").write_text(json.dumps({"id": "complete", "global_time": 16, "world_size": 1}))
            (root / "checkpoint").mkdir()
            (root / "checkpoint" / "train_status.json").write_text(json.dumps({"global_time": 32, "replay_id": None}))
            config_path = replay_path / "config.json"
            config = json.loads(config_path.read_text())
            for class_name in ("ReplayBuffer", "TrajectoryDictBufferMultiDim"):
                with self.subTest(class_name=class_name):
                    config["__target__"] = f"ufo.agents.common.buffers.trajectory.{class_name}"
                    config_path.write_text(json.dumps(config))
                    loaded, selected = _load_replay_buffer(root, buffer_rank=7, buffer_path=replay_path)
                    self.assertIsInstance(loaded, ReplayBuffer)
                    self.assertEqual(selected, replay_path)
                    self.assertEqual(len(loaded), 3)

    def test_reward_replay_rejects_incomplete_or_changed_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            replay_path = root / "checkpoint" / "buffers" / "train_rank_0"
            replay_path.mkdir(parents=True)
            marker = replay_path / "snapshot.json"
            (replay_path / "config.json").write_text(json.dumps({
                "__target__": "ufo.agents.common.buffers.trajectory.ReplayBuffer"
            }))
            for content in (None, "{", "null", '{}', '{"id":"x","global_time":0,"world_size":true}'):
                with self.subTest(content=content), patch("ufo.inference.reward_inference.ReplayBuffer.load") as load:
                    if content is not None:
                        marker.write_text(content)
                    with self.assertRaisesRegex(ValueError, "snapshot"):
                        _load_replay_buffer(root, buffer_rank=0, buffer_path=None)
                    load.assert_not_called()
            marker.write_text(json.dumps({"id": "old", "global_time": 16, "world_size": 1}))

            def overwrite(*args, **kwargs):
                marker.write_text(json.dumps({"id": "new", "global_time": 32, "world_size": 1}))
                return object()

            with patch("ufo.inference.reward_inference.ReplayBuffer.load", side_effect=overwrite):
                with self.assertRaisesRegex(RuntimeError, "snapshot changed"):
                    _load_replay_buffer(root, buffer_rank=0, buffer_path=None)

    def test_reward_relabel_passes_fp32_rewards_and_full_buffer_to_model(self) -> None:
        states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        data = {
            "action": torch.zeros(2, 0),
            "next": {"qpos": torch.zeros(2, 0), "qvel": torch.zeros(2, 0), "observation": {"state": states}},
        }

        def inference(next_obs, reward):
            self.assertEqual(reward.dtype, torch.float32)
            self.assertEqual(reward.device.type, "cpu")
            return reward.T @ next_obs["state"]

        wrapper = RewardInference(
            model=SimpleNamespace(device="cpu", reward_wr_inference=inference),
            inference_dataset=SimpleNamespace(size=lambda: 2, get_full_buffer=lambda: data),
            num_samples_per_inference=2,
            max_workers=1,
            env_model=mujoco.MjModel.from_xml_string("<mujoco/>"),
        )
        rewards = torch.tensor([[1.0], [2.0]], dtype=torch.float64).numpy()
        with patch("ufo.inference.reward_inference.relabel", return_value=rewards) as relabel:
            result = wrapper.reward_inference("zero")
        torch.testing.assert_close(result, torch.tensor([[7.0, 10.0]]), rtol=0, atol=0)
        self.assertEqual(relabel.call_args.args[1].shape, (2, 0))

    def test_reward_relabel_sampling_keeps_rng_and_nested_transition_fields(self) -> None:
        dataset = ReplayBuffer(
            capacity=8, n_dim=2, end_key="done", output_key_t=["action"], output_key_tp1=["observation"],
        )
        values = torch.arange(12, dtype=torch.float32).reshape(6, 2, 1)
        dataset.extend({
            "action": values + 30, "observation": {"state": values}, "qpos": values + 10,
            "qvel": values + 20, "done": torch.zeros(6, 2, 1, dtype=torch.bool),
        })
        dataset._get_idxs = get_idxs
        wrapper = RewardInference(
            model=SimpleNamespace(device="cpu", reward_wr_inference=lambda next_obs, reward: reward.T @ next_obs["state"]),
            inference_dataset=dataset, num_samples_per_inference=4, max_workers=1,
            env_model=mujoco.MjModel.from_xml_string("<mujoco/>"),
        )
        rewards = np.arange(4, dtype=np.float64).reshape(4, 1)
        with torch.random.fork_rng(devices=[]), patch("ufo.inference.reward_inference.relabel", return_value=rewards) as relabel:
            torch.manual_seed(18)
            actual = wrapper.reward_inference("zero")
            actual_rng = torch.get_rng_state()
            torch.manual_seed(18)
            expected = dataset.sample(4)
            torch.testing.assert_close(actual_rng, torch.get_rng_state(), rtol=0, atol=0)
        for argument, key in zip(relabel.call_args.args[1:4], ("qpos", "qvel", "action")):
            field = expected[key] if key == "action" else expected["next"][key]
            np.testing.assert_array_equal(argument, field.numpy())
        torch.testing.assert_close(actual, torch.tensor(rewards, dtype=torch.float32).T @ expected["next"]["observation"]["state"])

    def test_goal_target_states_follow_num_dof(self) -> None:
        obs = {
            "ref_body_pos": torch.zeros(4, 1, 3),
            "ref_body_rots": torch.zeros(4, 1, 4),
            "ref_body_vels": torch.zeros(4, 1, 3),
            "ref_body_angular_vels": torch.zeros(4, 1, 3),
            "dof_pos": torch.arange(27, dtype=torch.float32).repeat(4, 1),
            "ref_dof_vel": torch.ones(4, 27),
        }
        obs["ref_body_rots"][..., 3] = 1.0
        target = _target_states_from_obs(obs, device="cpu", num_dof=27)
        self.assertEqual(tuple(target["dof_states"].shape), (1, 27, 2))
        self.assertTrue(torch.equal(target["dof_states"][0, :, 0], torch.arange(27, dtype=torch.float32)))

    def test_non_g1_goal_json_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "Non-G1 goal inference requires --goal-json"):
            _find_goal_json(None, num_dof=27, robot_name="tiny_robot")

    def test_goal_json_dimension_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = [
                ("dof_pos", [0.0] * 26, "expected dimension 27"),
                ("qpos", [0.0] * 33, "expected dimension 34"),
            ]
            for key, value, pattern in cases:
                goal_json = root / f"{key}.json"
                goal_json.write_text(json.dumps([{"motion_id": 0, "frames": [0], key: value}]))
                with self.subTest(key=key):
                    with self.assertRaisesRegex(ValueError, pattern):
                        load_and_validate_goal_json(goal_json, num_dof=27)

    def test_goal_mae_validates_saved_robot_before_loading_model_or_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "checkpoint").mkdir()
            goal_path = root / "goal.json"
            env_cfg = SimpleNamespace(
                robot_training={"robot": {"name": "tiny_robot", "control_joint_names": ["a", "b"]}},
                build=Mock(),
            )
            cases = [
                (None, ValueError, "Non-G1 goal inference requires --goal-json", False),
                ([{"dof_pos": [0.0]}], ValueError, "expected dimension 2", False),
                ({"frames": [0]}, ValueError, "must be a list", False),
                ([{"motion_id": 0, "frames": [0], "dof_pos": [0.0, 0.0]}], RuntimeError, "checkpoint reached", True),
            ]
            for goals, error, message, valid in cases:
                if goals is not None:
                    goal_path.write_text(json.dumps(goals))
                with (
                    self.subTest(goals=goals),
                    patch("ufo.tools.evaluate_mae.load_mjlab_env_cfg", return_value=(env_cfg, False)),
                    patch("ufo.tools.evaluate_mae.load_model_from_checkpoint_dir", side_effect=RuntimeError("checkpoint reached")) as load,
                    self.assertRaisesRegex(error, message),
                ):
                    evaluate_goal_joint_mae(
                        model_folder=root, data_path=None, goal_json=goal_path if goals is not None else None,
                        device="cpu", steps=1, disable_dr=True, disable_obs_noise=True,
                        max_episode_length_s=10.0, output=None,
                    )
                self.assertEqual(load.call_count, int(valid))
                env_cfg.build.assert_not_called()

    def test_goal_cli_manifest_robot_config_mismatch_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tiny_robot = _write_tiny_robot_config(Path(tmpdir))
            argv = [
                "goal_inference.py",
                "--model-folder",
                "/tmp/ufo_unit_model",
                "--data-manifest",
                "configs/data/example_robot_state_auto_build.yaml",
                "--dataset",
                "g1_robot_state_sample",
                "--robot-config",
                str(tiny_robot),
            ]
            with patch.object(sys, "argv", argv):
                with self.assertRaisesRegex(ValueError, "does not match data manifest robot_config"):
                    parse_goal_args()

    def test_g1_default_goal_json_fallback_is_preserved(self) -> None:
        goal_json = _find_goal_json(None, num_dof=29, robot_name="g1_29dof")
        self.assertTrue(goal_json.name.endswith("goal_frames_lafan29dof.json"))
        goals = load_and_validate_goal_json(goal_json, num_dof=29)
        self.assertTrue(goals)

    def test_generic_relabel_xml_has_one_motor_per_control_joint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            robot_config = _write_tiny_robot_config(root)
            robot_xml = root / "tiny_robot.xml"
            output = write_mjlab_relabel_xml(
                robot_xml,
                root / "out",
                ["joint1", "joint2"],
                "tiny_robot",
                root_body_name="base",
            )
            model = mujoco.MjModel.from_xml_path(str(output))
            data = mujoco.MjData(model)
            self.assertEqual(model.nu, 2)
            self.assertEqual(data.ctrl.size, 2)
            text = output.read_text()
            self.assertIn('body="pelvis"', text)
            self.assertEqual(text.count("<motor "), 2)
            self.assertTrue(robot_config.exists())

    def test_g1_relabel_uses_config_joint_order_and_preserves_torso_sensors(self) -> None:
        robot = load_robot_training_spec(REPO_ROOT / "configs/robots/g1_29dof.yaml").robot
        joints = list(reversed(robot.control_joint_names))
        with tempfile.TemporaryDirectory() as tmpdir:
            output = write_g1_mjlab_relabel_xml(Path(robot.xml_path), Path(tmpdir), joints)
            model = mujoco.MjModel.from_xml_path(str(output))
            self.assertEqual([model.joint(int(j)).name for j in model.actuator_trnid[:, 0]], joints)
            torso_velocity = model.sensor("torso_link_subtreelinvel").id
            self.assertEqual(model.sensor_objtype[torso_velocity], mujoco.mjtObj.mjOBJ_BODY)
            self.assertEqual(model.sensor_objid[torso_velocity], model.body("torso_link").id)
            for name in ("frame_vel", "upvector_torso", "imu-angular-velocity"):
                sensor = model.sensor(name).id
                self.assertEqual(model.sensor_objtype[sensor], mujoco.mjtObj.mjOBJ_SITE)
                self.assertEqual(model.sensor_objid[sensor], model.site("imu_in_torso").id)
            with self.assertRaisesRegex(ValueError, "at least one joint"):
                write_g1_mjlab_relabel_xml(Path(robot.xml_path), Path(tmpdir), [])

    def test_non_g1_default_tasks_are_locomotion_only(self) -> None:
        tasks, mode = _resolve_reward_tasks(None, _fake_robot_training("tiny_robot", 27))
        self.assertEqual(mode, "non-G1 locomotion-only tasks")
        self.assertTrue(tasks)
        self.assertTrue(all(_is_non_g1_locomotion_task(task) for task in tasks))

    def test_non_g1_rejects_g1_semantics_tasks(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Task raisearms-l-l requires G1-style arm/body semantics and is not enabled for robot tiny_robot",
        ):
            _resolve_reward_tasks(["move-ego-0-0.3", "raisearms-l-l"], _fake_robot_training("tiny_robot", 27))
        with self.assertRaisesRegex(ValueError, "requires G1-style arm/body semantics"):
            _resolve_reward_tasks(["move-ego-low-0.7-0.3-l-l"], _fake_robot_training("tiny_robot", 27))

    def test_g1_default_tasks_are_preserved(self) -> None:
        self.assertIn("raisearms-l-l", DEFAULT_TASKS)
        self.assertIn("sitonground", DEFAULT_TASKS)
        tasks, mode = _resolve_reward_tasks(None, _fake_robot_training("g1_29dof", 29))
        self.assertEqual(mode, "G1 full tasks")
        self.assertEqual(tasks, DEFAULT_TASKS)


@pytest.mark.parametrize("preload_all", [False, True])
def test_goal_inference_selects_dataset_motion_in_subset_and_full_library(tmp_path, monkeypatch, preload_all):
    from test_motion_loading import build_sample_library

    from ufo.envs.core import MJLabCore
    from ufo.inference import goal_inference as goal

    library, _ = build_sample_library(tmp_path, mixed=False)
    library.num_envs = 1
    if preload_all:
        library.load_all_motions()
    env = MJLabCore.__new__(MJLabCore)
    env._motion_lib, env.num_envs, env.device, env.dt = library, 1, "cpu", 0.02
    env.reset_all = Mock()
    env.default_dof_pos = torch.zeros(1, 29)
    env.gravity_vec = torch.tensor([[0., 0., -1.]])
    env.config = SimpleNamespace(obs=SimpleNamespace(use_obs_filter=True, obs_scales={"base_ang_vel": 1.0}))
    wrapped = SimpleNamespace(_env=env, close=Mock(), reset=Mock(return_value=({}, {})),
                              step=Mock(return_value=({}, 0, False, False, {})))
    robot_path = REPO_ROOT / "configs/robots/g1_29dof.yaml"
    cfg = SimpleNamespace(build=lambda **kw: (wrapped, {}), motion_data_path="motions.pkl", mjcf_path="robot.xml", root_height_obs=False)
    monkeypatch.setattr(goal, "load_mjlab_env_cfg", lambda *a, **kw: (cfg, load_robot_training_spec(robot_path)))
    model = Mock()
    model.goal_inference.side_effect = lambda obs: obs["state"][:, :1]
    model.act.return_value = torch.zeros(1, 29)
    monkeypatch.setattr(goal, "load_model_from_checkpoint_dir", lambda *a, **kw: model)
    renderer = Mock()
    monkeypatch.setattr(goal, "MujocoQposRenderer", lambda *a, **kw: renderer)
    monkeypatch.setattr(goal, "render_policy_frame", lambda *a: np.zeros((8, 8, 3), dtype=np.uint8))
    (tmp_path / "checkpoint").mkdir()
    targets = [{"motion_id": mid, "motion_name": f"clip_{mid}", "frames": [frame]} for mid, frame in ((2, 2), (0, 1), (1, 4))]
    path = tmp_path / "goals.json"
    path.write_text(json.dumps(targets))
    goal.run_goal_inference(
        model_folder=tmp_path, data_path=None, robot_config=robot_path, goal_json=path, device="cpu", save_mp4=True,
        disable_dr=True, disable_obs_noise=True, episode_len=3, goal_switch_interval=1,
        render_size=8, camera_distance=3.0, camera_azimuth=120.0, camera_elevation=-15.0,
        fps=50, max_episode_length_s=10000.0, export_onnx=False,
    )
    import joblib

    latents = joblib.load(tmp_path / "goal_inference/goal_reaching.pkl")
    for mid, frame in ((2, 2), (0, 1), (1, 4)):
        np.testing.assert_allclose(latents[f"clip_{mid}_{frame}"], [[frame * 0.01]], atol=1e-6)
    # The video starts from the first requested motion, whose world X is 2.
    target = wrapped.reset.call_args.kwargs["target_states"]
    assert float(target["root_states"][0, 0]) == 2.0
    assert wrapped.step.call_count == 3
    import mediapy as media

    assert media.read_video(str(tmp_path / "goal_inference/videos/goal.mp4")).shape == (3, 8, 8, 3)
    wrapped.close.assert_called_once()
    renderer.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
