"""Opt-in: UFO_TEST_RP1_MOTION_FILE=/path/to/rp1.pkl pytest -q this_file."""

import os

import mujoco
import numpy as np
import pytest
import torch

MOTION_FILE = os.environ.get("UFO_TEST_RP1_MOTION_FILE")
pytestmark = pytest.mark.skipif(not MOTION_FILE or not torch.cuda.is_available(), reason="requires GPU and UFO_TEST_RP1_MOTION_FILE")


@pytest.fixture(scope="module", params=[False, True], ids=["dr_off", "dr_on"])
def env(request, tmp_path_factory):
    from ufo.train import build_mjlab_config

    cfg = build_mjlab_config(
        device="cuda:0",
        work_dir=str(tmp_path_factory.mktemp("rp1_physics")),
        num_envs=8,
        num_env_steps=32,
        seed=173,
        use_wandb=False,
        wandb_run_name=None,
        data_path=MOTION_FILE,
        robot_config="configs/robots/rp1_3.yaml",
        disable_dr=not request.param,
        disable_obs_noise=True,
    )
    vector, _ = cfg.env.build(num_envs=8)
    vector._test_dr_enabled = request.param
    vector.reset(to_numpy=False)
    yield vector
    vector.close()


def test_startup_randomization_ranges_and_live_inertia(env):
    core = env._env
    sim = core.mjlab_env.sim
    model = sim.mj_model
    body_ids = core._global_body_ids.long()
    mass = sim.model.body_mass[:, body_ids]
    defaults = torch.as_tensor(model.body_mass.copy(), device=core.device, dtype=mass.dtype)[body_ids]
    inertia = sim.model.body_inertia[:, body_ids]
    default_inertia = torch.as_tensor(model.body_inertia.copy(), device=core.device, dtype=inertia.dtype)[body_ids]
    com = sim.model.body_ipos[:, body_ids]
    default_com = torch.as_tensor(model.body_ipos.copy(), device=core.device, dtype=com.dtype)[body_ids]
    dof_ids = core.robot.indexing.joint_v_adr[core._joint_ids]
    armature = sim.model.dof_armature[:, dof_ids]
    dry_friction = sim.model.dof_frictionloss[:, dof_ids]
    armature_default = torch.as_tensor(model.dof_armature.copy(), device=core.device, dtype=armature.dtype)[dof_ids]
    friction_default = torch.as_tensor(model.dof_frictionloss.copy(), device=core.device, dtype=dry_friction.dtype)[dof_ids]
    geom_ids = core.robot.indexing.geom_ids
    geom_friction = sim.model.geom_friction[:, geom_ids, 0]
    geom_default = torch.as_tensor(model.geom_friction.copy(), device=core.device, dtype=geom_friction.dtype)[geom_ids, 0]
    if env._test_dr_enabled:
        limb_ids = [i for i, name in enumerate(core.body_names) if name.startswith(("left_", "right_"))]
        ratio = mass[:, limb_ids] / defaults[limb_ids]
        assert ((ratio >= 0.8 - 1e-6) & (ratio <= 1.2 + 1e-6)).all()
        assert ratio.std(dim=0).max() > 1e-3
        torso = core.body_names.index("waist_yaw_link")
        assert ((mass[:, torso] - defaults[torso]).abs() <= 2.0 + 1e-6).all()
        assert ((com[:, torso] - default_com[torso]).abs() <= 0.04 + 1e-6).all()
        assert (com[:, torso] - default_com[torso]).std(dim=0).max() > 1e-3
        torch.testing.assert_close(inertia, default_inertia * (mass / defaults).unsqueeze(-1))
        assert ((geom_friction >= 0.3) & (geom_friction <= 1.6)).all()
        assert geom_friction.std(dim=0).max() > 1e-3
        for values, initial in ((armature, armature_default), (dry_friction, friction_default)):
            scale = values / initial
            assert ((scale >= 0.8 - 1e-6) & (scale <= 1.2 + 1e-6)).all()
            assert scale.std(dim=0).max() > 1e-3
    else:
        torch.testing.assert_close(mass, defaults.expand_as(mass))
        torch.testing.assert_close(inertia, default_inertia.expand_as(inertia))
        torch.testing.assert_close(com, default_com.expand_as(com))
        torch.testing.assert_close(armature, armature_default.expand_as(armature))
        torch.testing.assert_close(dry_friction, friction_default.expand_as(dry_friction))
        torch.testing.assert_close(geom_friction, geom_default.expand_as(geom_friction))
    for actuator in core.robot.actuators:
        for value, baseline in ((actuator.stiffness, actuator.default_stiffness), (actuator.damping, actuator.default_damping)):
            if env._test_dr_enabled:
                ratio = value / baseline
                assert ((ratio >= 0.8 - 1e-6) & (ratio <= 1.2 + 1e-6)).all()
                assert ratio.std(dim=0).max() > 1e-3
            else:
                torch.testing.assert_close(value, baseline)


def test_delay_partial_reset_and_startup_offset_persistence(env):
    core = env._env
    env.reset(to_numpy=False)
    actuator = core.robot.actuators[0]
    buffer = actuator._delay_buffer
    assert all(other._delay_buffer is buffer for other in core.robot.actuators)
    lags = buffer.current_lags.clone()
    assert ((lags >= 0) & (lags <= 2)).all()
    offsets = core.default_dof_pos_offset.clone()
    if env._test_dr_enabled:
        assert 0 < offsets.abs().max() <= 0.01
    else:
        assert torch.count_nonzero(offsets) == 0
    expected_keys = {
        "penalty_torques",
        "penalty_action_rate",
        "limits_dof_pos",
        "height_gated_undesired_contacts",
        "penalty_slippage",
        "joint_acc_l2",
        "joint_velocity_l2",
        "joint_energy",
        "joint_rated_velocity",
        "joint_rated_effort",
        "action_jerk_l2",
        "angular_momentum_rate_l2",
        "yaw_angular_momentum_l2",
    }
    for _ in range(8):
        _, reward, _, _, info = env.step(torch.zeros_like(core.actions), to_numpy=False)
        assert torch.isfinite(reward).all()
        assert info["aux_rewards"].keys() == expected_keys
        assert all(torch.isfinite(value).all() for value in info["aux_rewards"].values())
        torch.testing.assert_close(buffer.current_lags, lags, atol=0, rtol=0)
    keep = torch.tensor([0, 2, 3, 4, 6, 7], device=core.device)
    reset = torch.tensor([1, 5], device=core.device)
    before = core.mjlab_env.sim.data.qpos[keep].clone()
    core.reset_idx(reset)
    torch.testing.assert_close(core.mjlab_env.sim.data.qpos[keep], before, atol=0, rtol=0)
    torch.testing.assert_close(buffer.current_lags[keep], lags[keep], atol=0, rtol=0)
    torch.testing.assert_close(core.default_dof_pos_offset, offsets, atol=0, rtol=0)
    assert torch.count_nonzero(core._normal_contact_sensor.data[reset]) == 0
    assert (core._reward_history.action_steps[reset] == -1).all()
    assert not core._reward_history.momentum_valid[reset].any()


def test_normal_contact_sensor_matches_independent_constraint_sum(env):
    core = env._env
    native = core.mjlab_env
    sensor = core._normal_contact_sensor
    env.reset(to_numpy=False)
    found = False
    for _ in range(100):
        env.step(torch.zeros_like(core.actions), to_numpy=False)
        if sensor.data[:, 0, :, 2].sum(dim=1).max() > 1.0:
            found = True
            break
    assert found, "No upward terrain contact was measured in 100 policy steps"
    # Compare before the policy step's final forward() replaces the contact data.
    native.action_manager.apply_action()
    native.scene.write_data_to_sim()
    native.sim.step()
    native.scene.update(dt=native.physics_dt)
    model = native.sim.mj_model
    mapping = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot/" + name): i for i, name in enumerate(core.body_names)}
    expected = np.zeros((core.num_envs, core.num_bodies, 3), dtype=np.float64)
    friction = native.sim.model.geom_friction[..., 0].cpu().numpy()
    ground_coefficients = []
    raw = sensor.physics.struct
    ncontact = int(raw.nacon.numpy()[0])
    assert ncontact <= raw.naconmax
    contact_worlds = raw.contact.worldid.numpy()[:ncontact]
    addresses = raw.contact.efc_address.numpy()[:ncontact]
    dimensions = raw.contact.dim.numpy()[:ncontact]
    geoms = raw.contact.geom.numpy()[:ncontact]
    frames = raw.contact.frame.numpy()[:ncontact]
    contact_friction = raw.contact.friction.numpy()[:ncontact]
    efc_forces = raw.efc.force.numpy()
    # Reconstruct from every explicit constraint address. get_data_into remaps
    # inactive contacts too, so its exported contact addresses are not an oracle.
    pyramidal = model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
    for contact_id, world in enumerate(contact_worlds):
        normal_force = 0.0
        if addresses[contact_id, 0] >= 0:
            edges = max(1, 2 * (int(dimensions[contact_id]) - 1)) if pyramidal else 1
            rows = addresses[contact_id, :edges]
            assert ((rows >= 0) & (rows < efc_forces.shape[1])).all()
            normal_force = efc_forces[world, rows].astype(np.float64).sum()
        normal = frames[contact_id, 0].astype(np.float64) * normal_force
        pair = geoms[contact_id]
        robot_geoms = [int(geom) for geom in pair if geom >= 0 and int(model.geom_bodyid[geom]) in mapping]
        if len(robot_geoms) == 1:
            expected_mu = friction[world if friction.shape[0] > 1 else 0, robot_geoms[0]]
            np.testing.assert_allclose(contact_friction[contact_id, 0], expected_mu, atol=1e-6, rtol=1e-6)
            ground_coefficients.append(float(contact_friction[contact_id, 0]))
        for side, geom in enumerate(pair):
            body = mapping.get(int(model.geom_bodyid[geom])) if geom >= 0 else None
            if body is not None:
                expected[world, body] += (2 * side - 1) * normal
    np.testing.assert_allclose(sensor.data[:, 0].cpu().numpy(), expected, rtol=1e-4, atol=1e-3)
    assert ground_coefficients
    print(f"RP1 dr={env._test_dr_enabled} ground contact friction range={min(ground_coefficients):.6f}..{max(ground_coefficients):.6f}")
