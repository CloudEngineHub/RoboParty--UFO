"""Imported motion derivatives survive clipping, heading changes and interpolation."""

import joblib
import numpy as np
import pytest
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation

from ufo.motion.formats import RobotStateMotion, clip_ufo_motion_dict, robot_state_to_ufo_motion
from ufo.motion.library import MotionLib, _dof_vel_from_dof_pos
from ufo.tools.robot_config import robot_spec_from_config


@pytest.fixture
def imported_motion(tmp_path):
    xml = tmp_path / "robot.xml"
    xml.write_text("""<mujoco><compiler angle="radian"/><worldbody>
      <body name="base"><freejoint name="root"/><geom type="sphere" size="0.1"/>
        <body name="knee" pos="0 0 -0.2"><joint name="knee" axis="0 -1 0" range="-3 3"/>
          <geom type="sphere" size="0.05"/>
          <body name="ankle" pos="0 0 -0.2"><joint name="ankle" axis="1 0 0.1" range="-1 1"/>
            <geom type="sphere" size="0.05"/>
          </body>
        </body>
      </body></worldbody><actuator><motor name="knee" joint="knee"/><motor name="ankle" joint="ankle"/></actuator></mujoco>""")
    spec = robot_spec_from_config(dict(name="rp1_axes", xml_path=str(xml), base_body="base"), tmp_path / "robot.yaml")
    positions = np.array([[0, 0], [0.03, 0.01], [0.11, -0.02], [0.06, 0.04], [0.02, 0.01]], dtype=np.float32)
    state = RobotStateMotion(
        "clip", np.tile([[0, 0, 0.8]], (5, 1)), np.tile([[0, 0, 0, 1]], (5, 1)), positions[:, ::-1], 50, joint_names=["ankle", "knee"]
    )
    motion = robot_state_to_ufo_motion(state, spec)
    motion.update(
        dof_vel=np.gradient(positions, axis=0) * 50,
        body_vel=np.broadcast_to([1, 2, 3], (5, 3, 3)).astype(np.float32).copy(),
        body_ang_vel=np.broadcast_to([4, 5, 6], (5, 3, 3)).astype(np.float32).copy(),
    )
    cfg = EasyDict(
        asset=EasyDict(assetRoot=str(tmp_path), assetFileName="robot.xml"), extend_config=[], motion_file=str(tmp_path / "motions.pkl")
    )
    joblib.dump({"clip": motion}, cfg.motion_file)
    return MotionLib(cfg, 1, "cpu"), motion, spec


def test_imported_joint_axes_and_derivatives_are_preserved(imported_motion):
    library, motion, spec = imported_motion
    positions = motion["dof_pos"]
    np.testing.assert_allclose(motion["pose_aa"][:, 1], positions[:, 0, None] * np.array(spec.joint_axes["knee"]))
    np.testing.assert_allclose(motion["pose_aa"][:, 2], positions[:, 1, None] * np.array(spec.joint_axes["ankle"]), atol=1e-8)
    library.load_all_motions()
    torch.testing.assert_close(library.dof_pos, torch.from_numpy(positions))
    torch.testing.assert_close(library.dvs, torch.from_numpy(motion["dof_vel"]), rtol=0, atol=0)
    torch.testing.assert_close(library.gvs, torch.from_numpy(motion["body_vel"]), rtol=0, atol=0)
    state = library.get_motion_state(torch.tensor([0]), torch.tensor([0.01]))
    torch.testing.assert_close(state["dof_vel"][0], torch.from_numpy(motion["dof_vel"][:2].mean(axis=0)))
    assert library._motion_num_frames.tolist() == [5]
    torch.testing.assert_close(library._motion_dt, torch.tensor([0.02]))


def test_clipping_keeps_source_boundary_derivatives(imported_motion):
    library, motion, _ = imported_motion
    clips = clip_ufo_motion_dict({"clip": motion}, clip_seconds=0.06, stride_seconds=0.06, min_clip_seconds=0)
    assert [len(clip["dof_pos"]) for clip in clips.values()] == [3, 2]
    np.testing.assert_array_equal(np.concatenate([clip["dof_vel"] for clip in clips.values()]), motion["dof_vel"])
    converted = library.load_motion_with_skeleton(list(clips.values()), None, -1)
    torch.testing.assert_close(converted[1].dof_vels, torch.from_numpy(motion["dof_vel"][3:]), rtol=0, atol=0)


def test_heading_rotates_explicit_world_velocities(imported_motion):
    library, motion, _ = imported_motion
    heading = Rotation.from_euler("z", 90, degrees=True)
    transformed = library.load_motion_with_skeleton([motion], heading.as_quat(), -1)[0]
    for field, target in [("body_vel", "global_velocity"), ("body_ang_vel", "global_angular_velocity")]:
        expected = heading.apply(motion[field].reshape(-1, 3)).reshape(5, 3, 3)
        np.testing.assert_allclose(transformed[target].numpy(), expected, atol=1e-6)
    torch.testing.assert_close(transformed.dof_vels, torch.from_numpy(motion["dof_vel"]))


def test_missing_derivatives_keeps_existing_fk_and_forward_difference(imported_motion):
    library, motion, _ = imported_motion
    legacy = {key: value for key, value in motion.items() if key not in {"dof_vel", "body_vel", "body_ang_vel"}}
    expected = library.kinematics.fk_batch(
        torch.from_numpy(motion["pose_aa"][None]), torch.from_numpy(motion["root_trans_offset"][None]), dt=0.02
    )
    actual = library.load_motion_with_skeleton([legacy], None, -1)[0]
    torch.testing.assert_close(actual.global_velocity, expected.global_velocity[0], rtol=0, atol=0)
    torch.testing.assert_close(actual.global_angular_velocity, expected.global_angular_velocity[0], rtol=0, atol=0)
    torch.testing.assert_close(actual.dof_vels, _dof_vel_from_dof_pos(torch.from_numpy(motion["dof_pos"]), 0.02), rtol=0, atol=0)


@pytest.mark.parametrize("field", ["dof_vel", "body_vel", "body_ang_vel"])
@pytest.mark.parametrize("malformed", ["shape", "nan"])
def test_rejects_malformed_imported_derivatives(imported_motion, field, malformed):
    library, motion, _ = imported_motion
    motion[field] = motion[field][:-1] if malformed == "shape" else np.full_like(motion[field], np.nan)
    with pytest.raises(ValueError, match=field):
        library.load_motion_with_skeleton([motion], None, -1)


def test_rejects_body_derivatives_with_unmapped_virtual_bodies(imported_motion):
    library, motion, _ = imported_motion
    library.kinematics.cfg.extend_config = [EasyDict(joint_name="virtual")]
    with pytest.raises(ValueError, match="motion_extensions"):
        library.load_motion_with_skeleton([motion], None, -1)
