"""Contracts at the boundary between task state and observation/reward kernels."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from ufo.envs.core import MJLabCore
from ufo.envs.observations import ObservationHistory
from ufo.envs.rewards import combine_rewards


def test_extended_body_velocity_matches_world_position_derivative():
    core = MJLabCore.__new__(MJLabCore)
    core.num_envs, core.num_extend_bodies = 4, 2
    core.extend_body_parent_ids = torch.tensor([1, 0])
    core.extend_body_pos_in_parent = torch.tensor([[0.0, 0.0, 0.35], [0.0, 0.0, 0.0]], dtype=torch.float64).repeat(4, 1, 1)
    core.extend_body_rot_in_parent_xyzw = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64).repeat(4, 2, 1)
    positions = torch.zeros(4, 2, 3, dtype=torch.float64)
    core.body_vel = torch.tensor([[-1.0, -2.0, -3.0], [0.1, 0.2, 0.3]], dtype=torch.float64).repeat(4, 1, 1)
    core.body_ang_vel = torch.zeros_like(core.body_vel)
    speeds = torch.tensor([1.0, 1.0, 2.0, 0.0], dtype=torch.float64)
    core.body_ang_vel[:, 1, 1] = speeds
    angles = torch.tensor([0.0, torch.pi / 2, -torch.pi / 2, torch.pi], dtype=torch.float64)

    def at_time(time):
        core.body_pos = positions + time * core.body_vel
        core.body_rot = core.extend_body_rot_in_parent_xyzw.clone()
        angle = angles + time * speeds
        core.body_rot[:, 1, 1] = torch.sin(angle / 2)
        core.body_rot[:, 1, 3] = torch.cos(angle / 2)
        return core._extend_body_state()

    body_pos, _, body_vel, body_ang_vel = at_time(0.0)
    expected = core.body_vel[:, 1] + torch.tensor(
        [[0.35, 0.0, 0.0], [0.0, 0.0, -0.35], [0.0, 0.0, 0.7], [0.0, 0.0, 0.0]], dtype=torch.float64
    )
    torch.testing.assert_close(body_vel[:, 2], expected, atol=1e-10, rtol=1e-10)
    # An extension at its parent's origin inherits that parent's velocity.
    torch.testing.assert_close(body_vel[:, 3], core.body_vel[:, 0])
    torch.testing.assert_close(body_vel[:, :2], core.body_vel)
    torch.testing.assert_close(body_ang_vel[:, 2:], core.body_ang_vel[:, core.extend_body_parent_ids])
    torch.testing.assert_close(body_pos[1, 2], torch.tensor([0.35, 0.0, 0.0], dtype=torch.float64), atol=1e-10, rtol=1e-10)
    dt = 1e-5
    finite_difference = (at_time(dt)[0] - at_time(-dt)[0]) / (2 * dt)
    torch.testing.assert_close(body_vel, finite_difference, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("enabled,length,initial,expected", [
    (False, 10, 0.5, 0.5), (True, 10, 0.5, 0.45), (True, 41, 0.5, 0.5),
    (True, 50, 0.5, 0.55), (True, 50, 1.0, 1.0), (True, 10, 0.1, 0.1),
])
def test_noise_curriculum_changes_only_noise_multiplier(enabled, length, initial, expected):
    core = MJLabCore.__new__(MJLabCore)
    core.config = OmegaConf.create({"obs": {
        "noise_curriculum_degree": 0.1,
        "noise_curriculum_level_down_threshold": 40,
        "noise_curriculum_level_up_threshold": 42,
        "noise_value_min": 0.1, "noise_value_max": 1.0,
    }})
    core.add_noise_curriculum = enabled
    core.average_episode_length = length
    core.current_noise_curriculum_value = initial
    core._update_obs_noise_curriculum()
    assert core.current_noise_curriculum_value == pytest.approx(expected)


def test_observation_history_uses_previous_scaled_frame_and_resets_selected_rows():
    core = MJLabCore.__new__(MJLabCore)
    core.config = OmegaConf.create(
        {
            "obs": {
                "obs_scales": {"dof_pos": 2.0},
                "noise_scales": {},
                "obs_auxiliary": {"history_actor": {"dof_pos": 2}},
            }
        }
    )
    core.history_handler = ObservationHistory(2, core.config.obs.obs_auxiliary, {"dof_pos": 1}, "cpu")
    core.is_evaluating = False
    core.add_noise_curriculum = False
    core.current_noise_curriculum_value = 1.0
    core._compute_privileged_obs = lambda: None
    core.actions = torch.zeros(2, 1)
    core.dof_pos = torch.tensor([[1.0], [3.0]])
    core.default_dof_pos = torch.zeros(2, 1)
    core.default_dof_pos_offset = torch.tensor([[0.25], [0.5]])
    core.dof_vel = torch.zeros(2, 1)
    core.base_ang_vel = torch.zeros(2, 3)
    core.projected_gravity = torch.zeros(2, 3)
    core._max_local_self = torch.zeros(2, 4)
    core.mjlab_env = SimpleNamespace(episode_length_buf=torch.tensor([4, 7]))
    first = core.get_observation(to_numpy=False)
    torch.testing.assert_close(first["state"][:, 0], torch.tensor([1.5, 5.0]))
    assert torch.count_nonzero(first["history_actor"]) == 0
    snapshots = [{key: first[key].clone() for key in ("state", "history_actor", "last_action")}]
    core.dof_pos.add_(1.0)
    second = core.get_observation(to_numpy=False)
    torch.testing.assert_close(second["history_actor"], torch.tensor([[1.5, 0.0], [5.0, 0.0]]))
    snapshots.append({key: second[key].clone() for key in snapshots[0]})
    core.history_handler.reset(torch.tensor([0]))
    third = core.get_observation(to_numpy=True, include_last_action=False)
    assert "last_action" not in third
    torch.testing.assert_close(torch.from_numpy(third["history_actor"]), torch.tensor([[0.0, 0.0], [7.0, 5.0]]))
    assert third["time"].tolist() == [[4], [7]]
    for obs, snapshot in zip((first, second), snapshots):
        for key, value in snapshot.items():
            torch.testing.assert_close(obs[key], value, rtol=0, atol=0)


def test_history_copies_input_and_query_survives_update_and_reset():
    history = ObservationHistory(2, {"actor": {"value": 2}}, {"value": 1}, "cpu")
    value = torch.tensor([[1.0], [2.0]])
    history.add("value", value)
    value.add_(10.0)
    snapshot = history.query("value")
    expected = torch.tensor([[[1.0], [0.0]], [[2.0], [0.0]]])
    torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)
    history.add("value", value)
    history.reset(torch.tensor([0]))
    torch.testing.assert_close(snapshot, expected, rtol=0, atol=0)
    torch.testing.assert_close(history.query("value"), torch.tensor([[[0.0], [0.0]], [[12.0], [2.0]]]))


def test_reward_weighting_keeps_auxiliary_values_unscaled():
    aux = {"penalty": torch.tensor([2.0, 3.0]), "other": torch.tensor([4.0, 5.0])}
    before = {key: value.clone() for key, value in aux.items()}
    result = combine_rewards(
        aux,
        {"penalty": -0.1, "other": 0.2, "unimplemented": 9.0},
        {"penalty"},
        use_curriculum=True,
        penalty_scale=0.5,
        num_envs=2,
        device="cpu",
    )
    torch.testing.assert_close(result, torch.tensor([0.7, 0.85]))
    for key in aux:
        torch.testing.assert_close(aux[key], before[key], atol=0, rtol=0)


def test_step_auxiliary_rewards_survive_reset_and_next_step():
    core = MJLabCore.__new__(MJLabCore)
    core._roboparty_rewards = False
    core.num_envs, core.device = 2, "cpu"
    core.config = OmegaConf.create({
        "robot": {"control": {"normalize_action": False, "action_clip_value": 100.0}},
        "rewards": {"reward_limit": {"soft_dof_vel_limit": 0.9, "soft_torque_limit": 0.9}},
    })
    core.actions = torch.zeros(2, 4)
    core.last_actions = torch.zeros_like(core.actions)
    core.torques = torch.full((2, 4), 2.0, requires_grad=True)
    core.dof_pos = torch.full((2, 4), 2.0)
    core.dof_vel = torch.full((2, 4), 2.0)
    core.dof_pos_limits = torch.tensor([[-1.0, 1.0]]).repeat(4, 1)
    core.dof_vel_limits = torch.ones(4)
    core.torque_limits = torch.ones(4)
    core.feet_indices = torch.tensor([0, 1])
    core.penalised_contact_indices = torch.tensor([2])
    core.left_ankle_dof_indices = torch.tensor([0, 1])
    core.right_ankle_dof_indices = torch.tensor([2, 3])
    core.contact_forces = torch.full((2, 3, 3), 2.0)
    core.body_vel = torch.ones(2, 3, 3)
    core.body_rot = torch.tensor([0.0, 0.3826834, 0.0, 0.9238795]).repeat(2, 3, 1)
    core.base_quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(2, 1)
    core.gravity_vec = torch.tensor([0.0, 0.0, -1.0]).repeat(2, 1)
    core.forward_vec = torch.tensor([1.0, 0.0, 0.0]).repeat(2, 1)
    core.reward_scales = {"penalty_torques": -0.1}
    core.reward_penalty_reward_names = set()
    core.use_reward_penalty_curriculum = False
    core.reward_penalty_scale = 1.0
    core.last_episode_length_buf = torch.zeros(2, dtype=torch.long)
    core._refresh_state = lambda: None
    core._mjlab_action_input = lambda: core.actions
    core._update_average_episode_length = lambda ids: None
    core._update_reward_penalty_curriculum = lambda: None
    core._update_obs_noise_curriculum = lambda: None
    core.mjlab_env = SimpleNamespace(
        episode_length_buf=torch.tensor([4, 7]),
        step=lambda actions: (None, None, torch.tensor([False, False]), torch.tensor([True, False]), {}),
    )

    def reset(env_ids):
        with torch.no_grad():
            for name in ("actions", "last_actions", "torques", "dof_pos", "dof_vel", "contact_forces", "body_vel"):
                getattr(core, name)[env_ids] = 0.0

    core.reset_idx = reset
    expected_reward, expected_aux = core._compute_reward()
    expected_aux = {name: value.detach().clone() for name, value in expected_aux.items()}
    _, reward, reset_mask, info = core.step(torch.zeros_like(core.actions))
    assert reset_mask.tolist() == [True, False]
    core.reset_finished(reset_mask.nonzero(as_tuple=False).flatten())
    assert torch.count_nonzero(core.torques[0]) == 0
    torch.testing.assert_close(reward, expected_reward, rtol=0, atol=0)
    core.step(torch.ones_like(core.actions))
    for name, value in info["aux_rewards"].items():
        assert not value.requires_grad
        torch.testing.assert_close(value, expected_aux[name], rtol=0, atol=0)


def test_contact_batch_preserves_order_sign_and_owned_snapshot():
    core = MJLabCore.__new__(MJLabCore)
    core.num_envs, core.num_bodies, core.device = 2, 4, "cpu"
    source = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3)
    core._contact_sensor = SimpleNamespace(data=SimpleNamespace(force=source))
    core._contact_body_ids = torch.tensor([2, 0])
    core._contact_sensor_ids = torch.tensor([4, 1])
    expected = torch.zeros(2, 4, 3)
    for body, sensor in [(2, 4), (0, 1)]:
        expected[:, body] = -source[:, sensor]
    actual = core._read_contact_forces()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    source.add_(100)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(core._read_contact_forces(), actual)
    core._contact_sensor.data.force = None
    assert torch.count_nonzero(core._read_contact_forces()) == 0
    core._contact_sensor = None
    assert torch.count_nonzero(core._read_contact_forces()) == 0


def test_state_refresh_keeps_ordered_snapshots_independent_of_simulator_storage():
    core = MJLabCore.__new__(MJLabCore)
    core.num_envs, core.num_bodies, core.device = 2, 3, "cpu"
    core._joint_ids = torch.tensor([3, 0, 2])
    core._body_ids = torch.tensor([2, 0, 3])
    core.gravity_vec = torch.tensor([[0.0, 0.0, -1.0]]).repeat(2, 1)
    core._contact_sensor = None
    values = {
        "joint_pos": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "joint_vel": torch.arange(8, 16, dtype=torch.float32).reshape(2, 4),
        "qfrc_actuator": torch.arange(16, 24, dtype=torch.float32).reshape(2, 4),
        "root_link_pose_w": torch.tensor([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        "root_link_vel_w": torch.arange(12, dtype=torch.float32).reshape(2, 6),
        "body_link_pose_w": torch.tensor([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]).repeat(2, 4, 1),
        "body_link_vel_w": torch.arange(48, dtype=torch.float32).reshape(2, 4, 6),
    }
    values["body_link_pose_w"][..., :3] = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    # Match simulator data that can expose strided views into larger buffers.
    values = {name: value.repeat_interleave(2, dim=-1)[..., ::2] for name, value in values.items()}
    core.robot = SimpleNamespace(data=SimpleNamespace(**values))
    core._refresh_state()
    expected = {
        "dof_pos": values["joint_pos"][:, core._joint_ids],
        "dof_vel": values["joint_vel"][:, core._joint_ids],
        "torques": values["qfrc_actuator"][:, core._joint_ids],
        "body_pos": values["body_link_pose_w"][:, core._body_ids, :3],
        "body_vel": values["body_link_vel_w"][:, core._body_ids, :3],
        "body_ang_vel": values["body_link_vel_w"][:, core._body_ids, 3:],
    }
    for name, value in expected.items():
        torch.testing.assert_close(getattr(core, name), value, rtol=0, atol=0)
    snapshots = {name: getattr(core, name).clone() for name in expected}
    for value in values.values():
        value.add_(100)
    for name, value in snapshots.items():
        torch.testing.assert_close(getattr(core, name), value, rtol=0, atol=0)
    simulator_snapshots = {name: value.clone() for name, value in values.items()}
    for name in expected:
        getattr(core, name).zero_()
    for name, value in simulator_snapshots.items():
        torch.testing.assert_close(values[name], value, rtol=0, atol=0)
    core._refresh_state()
    torch.testing.assert_close(core.dof_pos, values["joint_pos"][:, core._joint_ids], rtol=0, atol=0)
