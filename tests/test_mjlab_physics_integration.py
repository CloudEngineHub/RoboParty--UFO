"""Opt-in GPU checks: UFO_TEST_MOTION_FILE=/path/motions.pkl pytest -q this_file.

No agent or training checkpoint is required; tests exercise the real simulator.
"""

import os

import numpy as np
import pytest
import torch

MOTION_FILE = os.environ.get("UFO_TEST_MOTION_FILE")
pytestmark = pytest.mark.skipif(not MOTION_FILE or not torch.cuda.is_available(), reason="requires GPU and UFO_TEST_MOTION_FILE")


@pytest.fixture(scope="module", params=[False, True], ids=["dr_off", "dr_on"])
def env(request, tmp_path_factory):
    from ufo.train import build_mjlab_config

    cfg = build_mjlab_config(
        device="cuda:0",
        work_dir=str(tmp_path_factory.mktemp("physics")),
        num_envs=8,
        num_env_steps=32,
        seed=173,
        use_wandb=False,
        wandb_run_name=None,
        data_path=MOTION_FILE,
        disable_dr=not request.param,
        disable_obs_noise=True,
    )
    vector, info = cfg.env.build(num_envs=8)
    from omegaconf import OmegaConf

    assert info["task_config"] is not None
    assert OmegaConf.to_yaml(info["task_config"])
    vector._test_dr_enabled = request.param
    yield vector
    vector.close()


def _same(a, b):
    torch.testing.assert_close(a.clone(), b.clone(), atol=0, rtol=0)


def test_partial_reset_preserves_other_worlds_and_clears_timeout(env):
    core = env._env
    native = core.mjlab_env
    env.reset(to_numpy=False)
    for _ in range(4):
        env.step(torch.zeros_like(core.actions), to_numpy=False)
    reset_ids = torch.tensor([1, 5], device=core.device)
    keep = torch.tensor([0, 2, 3, 4, 6, 7], device=core.device)
    # Produce real pending manual resets through the public step interface.
    native.episode_length_buf[reset_ids] = native.max_episode_length
    _, _, _, truncated, _ = native.step(torch.zeros_like(core.actions))
    assert truncated[reset_ids].all()
    core._refresh_state()
    fields = {name: getattr(native.sim.data, name).clone() for name in ("qpos", "qvel")}
    core_fields = {
        name: getattr(core, name).clone()
        for name in ("actions", "last_actions", "motion_ids", "motion_start_times", "default_dof_pos_offset", "episode_length_buf")
    }
    histories = {name: value.clone() for name, value in core.history_handler.history.items()}
    timers = [value.clone() for value in native.event_manager._interval_term_time_left]
    core.reset_idx(reset_ids)
    for name, before in fields.items():
        _same(getattr(native.sim.data, name)[keep], before[keep])
    for name, before in core_fields.items():
        _same(getattr(core, name)[keep], before[keep])
    for name, before in histories.items():
        _same(core.history_handler.history[name][keep], before[keep])
        assert torch.count_nonzero(core.history_handler.history[name][reset_ids]) == 0
    for after, before in zip(native.event_manager._interval_term_time_left, timers):
        _same(after[keep], before[keep])
        lower, upper = core.config.domain_rand.push_interval_s
        assert ((after[reset_ids] >= lower) & (after[reset_ids] <= upper)).all()
    assert core.episode_length_buf.data_ptr() == native.episode_length_buf.data_ptr()
    assert (native.episode_length_buf[reset_ids] == 0).all()
    assert torch.count_nonzero(core.actions[reset_ids]) == 0
    # Must succeed without UFO touching mjlab's private pending-reset flags.
    native.step(torch.zeros_like(core.actions))


def test_domain_randomization_changes_actual_model_fields_and_targets(env):
    import mujoco

    core = env._env
    native = core.mjlab_env
    model = native.sim.mj_model
    cfg = core.config.domain_rand
    robot_bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + n) for n in core.body_names[: core.num_bodies]]
    assert min(robot_bodies) >= 0
    geoms = np.flatnonzero(np.isin(model.geom_bodyid, robot_bodies)).tolist()
    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + core.config.robot.torso_name)
    mass = native.sim.model.body_mass.clone()
    com = native.sim.model.body_ipos.clone()
    friction = native.sim.model.geom_friction.clone()
    baseline_mass = torch.as_tensor(model.body_mass.copy(), device=core.device, dtype=mass.dtype)
    baseline_com = torch.as_tensor(model.body_ipos.copy(), device=core.device, dtype=com.dtype)
    baseline_friction = torch.as_tensor(model.geom_friction.copy(), device=core.device, dtype=friction.dtype)
    if env._test_dr_enabled:
        ratio = mass[:, robot_bodies] / baseline_mass[robot_bodies]
        lower, upper = cfg.link_mass_range
        assert ((ratio >= lower - 1e-6) & (ratio <= upper + 1e-6)).all()
        assert ratio.std(dim=0).max() > 0.001
        delta = com[:, torso] - baseline_com[torso]
        for axis, key in enumerate(("x", "y", "z")):
            lower, upper = cfg.base_com_range[key]
            assert ((delta[:, axis] >= lower - 1e-6) & (delta[:, axis] <= upper + 1e-6)).all()
            assert delta[:, axis].std() > 1e-4
        mu = friction[:, geoms, 0]
        lower, upper = cfg.friction_range
        assert ((mu >= lower - 1e-6) & (mu <= upper + 1e-6)).all()
        assert mu.std(dim=0).max() > 0.001
        assert core.default_dof_pos_offset.abs().max() > 1e-4
        lower, upper = cfg.default_dof_pos_noise_range
        assert ((core.default_dof_pos_offset >= lower) & (core.default_dof_pos_offset <= upper)).all()
        # Translating the normalized input must yield the configured angle offset.
        core.actions.zero_()
        idx = core._action_term_dof_indices
        torch.testing.assert_close(core._mjlab_action_input() * core.action_target_scale[:, idx], core.default_dof_pos_offset[:, idx])
        native.action_manager.process_action(core._mjlab_action_input())
        native.action_manager.apply_action()
        torch.testing.assert_close(
            core.robot.data.joint_pos_target[:, core._joint_ids],
            core.default_dof_pos + core.default_dof_pos_offset,
        )
        before = {name: getattr(native.sim.model, name).clone() for name in ("body_mass", "body_ipos", "geom_friction")}
        core.reset_idx(torch.tensor([0, 3], device=core.device))
        for name, value in before.items():
            _same(getattr(native.sim.model, name), value)  # startup DR is not reset DR
    else:
        torch.testing.assert_close(mass, baseline_mass.expand_as(mass))
        torch.testing.assert_close(com, baseline_com.expand_as(com))
        torch.testing.assert_close(friction, baseline_friction.expand_as(friction))
        assert torch.count_nonzero(core.default_dof_pos_offset) == 0
        assert not native.event_manager._interval_term_time_left


def test_interval_push_is_scoped_and_matches_configured_bounds(env):
    if not env._test_dr_enabled:
        pytest.skip("push is disabled")
    core = env._env
    native = core.mjlab_env
    timers = native.event_manager._interval_term_time_left
    assert len(timers) == 1
    timers[0].fill_(10.0)
    timers[0][[1, 5]] = 0.0
    before = core.robot.data.root_link_vel_w.clone()
    native.event_manager.apply("interval", dt=core.dt)
    native.sim.forward()
    after = core.robot.data.root_link_vel_w.clone()
    delta = after - before
    _same(delta[[0, 2, 3, 4, 6, 7]], torch.zeros_like(delta[[0, 2, 3, 4, 6, 7]]))
    assert delta[[1, 5], :2].abs().max() > 1e-5
    assert delta[[1, 5], :2].abs().max() <= core.config.domain_rand.max_push_vel_xy + 1e-6
    assert delta[[1, 5], 3:].abs().max() <= core.config.domain_rand.max_push_ang_vel + 1e-6
    torch.testing.assert_close(delta[:, 2], torch.zeros_like(delta[:, 2]), atol=1e-6, rtol=0)


def test_contact_forces_and_rewards_match_independent_contact_sum(env):
    import mujoco
    import mujoco_warp
    import warp as wp

    core = env._env
    native = core.mjlab_env
    env.reset(to_numpy=False)
    model = native.sim.mj_model
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + n) for n in core.body_names[: core.num_bodies]]
    seen_contact = False
    seen_feet = False
    seen_undesired = False
    seen_orientation_penalty = False
    for step in range(100):
        _, reward, _, _, info = env.step(torch.zeros_like(core.actions), to_numpy=False)
        if step % 10:
            continue
        # Independently sum per-contact wrenches into force ON each body.
        data = native.sim.wp_data
        ncon = int(data.nacon.numpy()[0])
        worlds = data.contact.worldid.numpy()[:ncon]
        ids = np.flatnonzero(worlds == 0).astype(np.int32)
        forces = np.zeros((model.nbody, 3))
        if len(ids):
            with wp.ScopedDevice(core.device):
                contact_ids = wp.array(ids, dtype=wp.int32)
                wrenches = wp.zeros(len(ids), dtype=wp.spatial_vector)
                mujoco_warp.contact_force(native.sim.wp_model, data, contact_ids, True, wrenches)
                world_forces = wrenches.numpy()[:, :3]
            pairs = data.contact.geom.numpy()[ids]
            for pair, world_force in zip(pairs, world_forces):
                forces[model.geom_bodyid[pair[0]]] -= world_force
                forces[model.geom_bodyid[pair[1]]] += world_force
        actual = core.contact_forces[0].detach().cpu().numpy()
        expected = forces[body_ids]
        np.testing.assert_allclose(actual, expected, atol=0.05, rtol=2e-3)
        seen_contact |= np.linalg.norm(expected, axis=-1).max() > 1
        feet = core.feet_indices.cpu().numpy()
        penalized = core.penalised_contact_indices.cpu().numpy()
        foot_mask = np.linalg.norm(expected[feet], axis=-1) > 1
        seen_feet |= foot_mask.any()
        expected_slip = (np.linalg.norm(core.body_vel[0, core.feet_indices, :2].cpu().numpy(), axis=-1) * foot_mask).sum()
        undesired = float((np.abs(expected[penalized]) > 1).any())
        seen_undesired |= bool(undesired)
        _, aux = core._compute_reward()
        np.testing.assert_allclose(aux["penalty_slippage"][0].item(), expected_slip, atol=1e-5)
        assert aux["penalty_undesired_contact"][0].item() == undesired
        from scipy.spatial.transform import Rotation

        rotations = Rotation.from_quat(core.body_rot[0, core.feet_indices].cpu().numpy())
        gravity_in_feet = rotations.inv().apply(np.tile([0.0, 0.0, -1.0], (len(feet), 1)))
        expected_orientation = (np.linalg.norm(gravity_in_feet[:, :2], axis=-1) * (expected[feet, 2] > 1)).sum()
        np.testing.assert_allclose(aux["penalty_feet_ori"][0].item(), expected_orientation, atol=1e-5)
        seen_orientation_penalty |= expected_orientation > 1e-4
        weighted_reward = sum(
            value
            * core.reward_scales[name]
            * (core.reward_penalty_scale if name in core.reward_penalty_reward_names and core.use_reward_penalty_curriculum else 1.0)
            for name, value in aux.items()
            if name in core.reward_scales
        )
        torch.testing.assert_close(reward, weighted_reward)

        # The rollout must receive the same reward terms, before any next reset.
        torch.testing.assert_close(info["aux_rewards"]["penalty_slippage"].flatten(), aux["penalty_slippage"])
        assert torch.isfinite(reward).all()
    assert seen_contact and seen_feet and seen_undesired and seen_orientation_penalty, (
        seen_contact,
        seen_feet,
        seen_undesired,
        seen_orientation_penalty,
    )
