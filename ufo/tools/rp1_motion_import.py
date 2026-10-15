"""Import the pinned RP1 BeyondMimic dataset into UFO without resampling.

The source stores link poses and COM velocities. Transform velocities to link
origins before writing explicit derivatives consumed by UFO's motion library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mujoco
import numpy as np
import torch
from easydict import EasyDict
from scipy.spatial.transform import Rotation

from ufo.motion.formats import RobotStateMotion, clip_ufo_motion_dict, robot_state_to_ufo_motion, validate_ufo_motion_dict
from ufo.motion.kinematics import HumanoidKinematics
from ufo.tools.robot_config import load_robot_spec


def sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def quaternion_error(actual, expected):
    a = actual / np.linalg.norm(actual, axis=-1, keepdims=True)
    b = expected / np.linalg.norm(expected, axis=-1, keepdims=True)
    return float(np.max(4 * np.arcsin(np.clip(np.minimum(np.linalg.norm(a - b, axis=-1), np.linalg.norm(a + b, axis=-1)) / 2, 0, 1))))


def convert(manifest_path, config_path, output_path, train_output_path, report_path):
    for path in (output_path, train_output_path, report_path):
        require(not Path(path).exists(), f"Refusing to overwrite {path}")
    require(len({Path(path).resolve() for path in (output_path, train_output_path, report_path)}) == 3, "Output paths must be distinct")
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    meta = manifest["npz"]
    records = manifest["clips"]
    spec = load_robot_spec(config_path)
    require(meta["body_quaternion_order"] == spec.root_quat_order == "xyzw", "Expected XYZW quaternions")
    require(spec.coordinate_system == "z_up" and spec.dof_unit == "rad", "Expected Z-up and radians")
    joints, bodies = meta["joint_order"], meta["body_order"]
    require(len(joints) == len(set(joints)) and set(joints) == set(spec.control_joint_names), "Joint names do not match robot")
    require(len(bodies) == len(set(bodies)) and set(bodies) == set(spec.body_names), "Body names do not match robot")
    require(spec.body_names[0] == spec.base_body, "UFO requires the root body at index zero")
    require(len(records) == manifest["clip_count"] == meta["clip_count"], "Clip count disagrees with manifest")
    model = mujoco.MjModel.from_xml_path(spec.xml_path)
    sim = mujoco.MjData(model)
    parser = HumanoidKinematics(
        EasyDict(asset=EasyDict(assetRoot=str(spec.xml_file.parent), assetFileName=spec.xml_file.name), extend_config=[])
    )
    require(parser.body_names == spec.body_names, "UFO FK body order differs from RobotSpec")
    root_id = bodies.index(spec.base_body)
    body_order = [bodies.index(name) for name in spec.body_names]
    joint_order = [joints.index(name) for name in spec.control_joint_names]
    body_ids = np.array([model.body(name).id for name in bodies])
    qpos_ids = [spec.joint_qpos_addr[name] for name in joints]
    qvel_ids = [spec.joint_dof_addr[name] for name in joints]
    require(spec.free_joint is not None, "RP1 requires a free root joint")
    root_qpos = spec.joint_qpos_addr[spec.free_joint]
    root_qvel = spec.joint_dof_addr[spec.free_joint]
    motions, audits = {}, []
    torch.set_num_threads(1)
    for record in records:
        name = record["name"]
        require(name not in motions, f"Duplicate motion name: {name}")
        path = (manifest_path.parent / record["npz"]).resolve()
        require(path.is_relative_to(manifest_path.parent), f"NPZ path escapes dataset: {path}")
        require(sha256(path) == record["npz_sha256"], f"{name}: source NPZ hash mismatch")
        with np.load(path, allow_pickle=False) as loaded:
            fields = {key: np.asarray(loaded[key]) for key in meta["keys"]}
        fps = fields["fps"]
        require(fps.size == 1 and np.isfinite(fps).all() and float(fps.item()) == float(meta["output_fps"]), f"{name}: invalid fps")
        fps = float(fps.item())
        frames = record["npz_frames"]
        require(frames >= 3, f"{name}: at least three frames required")
        shapes = {
            "joint_pos": (frames, len(joints)),
            "joint_vel": (frames, len(joints)),
            "body_pos_w": (frames, len(bodies), 3),
            "body_quat_w": (frames, len(bodies), 4),
            "body_lin_vel_w": (frames, len(bodies), 3),
            "body_ang_vel_w": (frames, len(bodies), 3),
        }
        for key, shape in shapes.items():
            require(fields[key].shape == shape and np.isfinite(fields[key]).all(), f"{name}: invalid {key} shape/values")
        quaternions = fields["body_quat_w"]
        require(np.max(np.abs(np.linalg.norm(quaternions, axis=-1) - 1)) < 1e-5, f"{name}: non-unit quaternion")
        rotations = Rotation.from_quat(quaternions.reshape(-1, 4))
        offsets = rotations.apply(np.broadcast_to(model.body_ipos[body_ids], (frames, len(bodies), 3)).reshape(-1, 3)).reshape(
            frames, len(bodies), 3
        )
        link_velocity = (fields["body_lin_vel_w"] - np.cross(fields["body_ang_vel_w"], offsets)).astype(np.float32)
        gradient_error = float(np.max(np.abs(np.gradient(fields["joint_pos"], axis=0) * fps - fields["joint_vel"])))
        require(gradient_error < 1e-5, f"{name}: joint velocity is not the documented reference derivative")
        motion = robot_state_to_ufo_motion(
            RobotStateMotion(
                name,
                fields["body_pos_w"][:, root_id],
                quaternions[:, root_id],
                fields["joint_pos"],
                fps,
                joint_names=joints,
                source=manifest["dataset"],
            ),
            spec,
        )
        motion.update(
            dof_vel=fields["joint_vel"][:, joint_order].copy(),
            body_vel=link_velocity[:, body_order].copy(),
            body_ang_vel=fields["body_ang_vel_w"][:, body_order].copy(),
        )
        motion["metadata"].update(
            source_npz=record["npz"],
            source_npz_sha256=record["npz_sha256"],
            source_manifest_sha256=sha256(manifest_path),
            body_velocity_reference="link_origin",
            body_velocity_frame="world",
            source_body_velocity_reference="center_of_mass",
            joint_velocity_definition="numpy.gradient(joint_pos, axis=0) * fps, one-sided endpoints",
            body_velocity_conversion="v_link = v_com - omega_world cross (R_body_world @ body_ipos)",
            frames_preserved=frames,
        )
        sampled = np.unique(np.linspace(0, frames - 1, min(128, frames), dtype=np.int64))
        errors = dict(position_m=0.0, rotation_rad=0.0, linear_velocity_m_s=0.0, angular_velocity_rad_s=0.0)
        for frame in sampled:
            sim.qpos[root_qpos : root_qpos + 3] = fields["body_pos_w"][frame, root_id]
            sim.qpos[root_qpos + 3 : root_qpos + 7] = quaternions[frame, root_id, [3, 0, 1, 2]]
            sim.qpos[qpos_ids] = fields["joint_pos"][frame]
            sim.qvel[root_qvel : root_qvel + 3] = link_velocity[frame, root_id]
            sim.qvel[root_qvel + 3 : root_qvel + 6] = (
                Rotation.from_quat(quaternions[frame, root_id]).inv().apply(fields["body_ang_vel_w"][frame, root_id])
            )
            sim.qvel[qvel_ids] = fields["joint_vel"][frame]
            mujoco.mj_forward(model, sim)
            velocity = np.empty((len(bodies), 6))
            for index, body_id in enumerate(body_ids):
                mujoco.mj_objectVelocity(model, sim, mujoco.mjtObj.mjOBJ_XBODY, body_id, velocity[index], 0)
            errors["position_m"] = max(errors["position_m"], float(np.max(np.abs(sim.xpos[body_ids] - fields["body_pos_w"][frame]))))
            errors["rotation_rad"] = max(errors["rotation_rad"], quaternion_error(sim.xquat[body_ids][:, [1, 2, 3, 0]], quaternions[frame]))
            errors["linear_velocity_m_s"] = max(
                errors["linear_velocity_m_s"], float(np.max(np.abs(velocity[:, 3:] - link_velocity[frame])))
            )
            errors["angular_velocity_rad_s"] = max(
                errors["angular_velocity_rad_s"], float(np.max(np.abs(velocity[:, :3] - fields["body_ang_vel_w"][frame])))
            )
        with torch.no_grad():
            fk = parser.fk_batch(
                torch.from_numpy(motion["pose_aa"][sampled][None]), torch.from_numpy(motion["root_trans_offset"][sampled][None]), dt=1 / fps
            )
        errors["ufo_position_m"] = float(np.max(np.abs(fk.global_translation[0].numpy() - fields["body_pos_w"][sampled][:, body_order])))
        errors["ufo_rotation_rad"] = quaternion_error(fk.global_rotation[0].numpy(), quaternions[sampled][:, body_order])
        for key, error in errors.items():
            require(error < (2e-4 if "velocity" in key else 2e-5), f"{name}: FK {key} error {error} exceeds tolerance")
        require(np.array_equal(motion["dof_pos"], fields["joint_pos"][:, joint_order]), f"{name}: joint positions changed")
        audits.append(
            dict(
                name=name,
                frames=frames,
                fps=fps,
                npz_sha256=record["npz_sha256"],
                fk_sampled_frames=len(sampled),
                errors=errors,
                joint_gradient_max_error=gradient_error,
            )
        )
        motions[name] = motion
        print(json.dumps(audits[-1]), flush=True)
    require(sum(item["frames"] for item in audits) == meta["total_frames"], "Total frame count mismatch")
    validate_ufo_motion_dict(motions, manifest["dataset"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    require(not output_path.exists(), f"Refusing to overwrite {output_path}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    joblib.dump(motions, temporary)
    temporary.replace(output_path)
    train_output_path = Path(train_output_path)
    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    training = clip_ufo_motion_dict(motions, clip_seconds=10, stride_seconds=10, min_clip_seconds=1)
    temporary = train_output_path.with_suffix(train_output_path.suffix + ".tmp")
    joblib.dump(training, temporary)
    temporary.replace(train_output_path)
    report = dict(
        dataset=manifest["dataset"],
        source_manifest=str(manifest_path),
        source_manifest_sha256=sha256(manifest_path),
        robot_config=str(Path(config_path).resolve()),
        robot_config_sha256=sha256(config_path),
        robot_xml_sha256=sha256(spec.xml_path),
        output=str(output_path.resolve()),
        output_sha256=sha256(output_path),
        train_output=str(train_output_path.resolve()),
        train_output_sha256=sha256(train_output_path),
        train_clips=len(training),
        train_frames=sum(len(motion["dof_pos"]) for motion in training.values()),
        training_windows={"clip_seconds": 10, "stride_seconds": 10, "min_clip_seconds": 1, "keep_short": True},
        clips=len(motions),
        frames=sum(item["frames"] for item in audits),
        joint_order=spec.control_joint_names,
        body_order=spec.body_names,
        velocity_semantics=motions[records[0]["name"]]["metadata"],
        audit=audits,
    )
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", required=True, type=Path)
    cli.add_argument("--robot-config", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--train-output", required=True, type=Path)
    cli.add_argument("--report", required=True, type=Path)
    args = cli.parse_args()
    convert(args.manifest, args.robot_config, args.output, args.train_output, args.report)
