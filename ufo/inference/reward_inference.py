"""Reward inference, task rewards, and replay relabeling for UFO policies.

G1 supports the full task set; other robots support root/locomotion tasks.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import abc
import argparse
import dataclasses
import functools
import inspect
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import joblib
import mujoco
import numpy as np
import torch
from dm_control.utils import rewards

from ufo.agents.common.buffers.trajectory import ReplayBuffer
from ufo.agents.common.checkpoint import checkpoint_load_device, load_model_from_checkpoint_dir, resolve_checkpoint_dir
from ufo.inference.runtime import (
    MujocoQposRenderer,
    add_bool_arg,
    add_robot_config_manifest_args,
    load_mjlab_env_cfg,
    render_policy_frame,
    resolve_inference_data_and_robot_args,
    video_writer,
)
from ufo.tools.export_onnx import export_policy_model


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.iter("body"):
        if body.get("name") == name:
            return body
    return None


def _first_worldbody_body(root: ET.Element) -> ET.Element | None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        return None
    return worldbody.find("body")


def _absolutize_compiler_paths(root: ET.Element, source_xml: Path) -> None:
    compiler = root.find("compiler")
    if compiler is None:
        return
    for attr in ("meshdir", "texturedir", "assetdir"):
        raw_path = compiler.get(attr)
        if not raw_path:
            continue
        asset_path = Path(raw_path)
        if not asset_path.is_absolute():
            compiler.set(attr, str((source_xml.parent / asset_path).resolve()))


def _write_relabel_xml(tree: ET.ElementTree, output_path: Path, control_joint_names: list[str]) -> Path:
    if not control_joint_names:
        raise ValueError("control_joint_names must contain at least one joint")
    root = tree.getroot()
    for actuator in list(root.findall("actuator")):
        root.remove(actuator)
    actuator_root = ET.SubElement(root, "actuator")
    for joint_name in control_joint_names:
        ET.SubElement(actuator_root, "motor", {"name": f"{joint_name}_motor", "joint": joint_name, "gear": "1"})
    tree.write(output_path, encoding="utf-8", xml_declaration=False)
    model = mujoco.MjModel.from_xml_path(str(output_path))
    if int(model.nu) != len(control_joint_names):
        raise RuntimeError(f"Expected relabel XML ctrl dim {len(control_joint_names)}, got model.nu={model.nu}: {output_path}")
    data = mujoco.MjData(model)
    if int(data.ctrl.size) != len(control_joint_names):
        raise RuntimeError(f"Expected relabel XML data.ctrl dim {len(control_joint_names)}, got {data.ctrl.size}: {output_path}")
    return output_path


def write_mjlab_relabel_xml(
    robot_xml: Path,
    output_dir: Path,
    control_joint_names: list[str],
    robot_name: str,
    *,
    root_body_name: str | None = None,
) -> Path:
    """Create a relabel-only MuJoCo XML with one motor per controlled joint."""

    if not control_joint_names:
        raise ValueError("control_joint_names must contain at least one joint")

    source_xml = Path(robot_xml).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_xml)
    root = tree.getroot()
    _absolutize_compiler_paths(root, source_xml)

    pelvis_body = _find_body(root, "pelvis")
    if pelvis_body is None and root_body_name is not None:
        pelvis_body = _find_body(root, root_body_name)
    if pelvis_body is None:
        pelvis_body = _first_worldbody_body(root)
    if pelvis_body is None:
        raise ValueError(f"Robot XML has no worldbody body for reward relabeling: {source_xml}")
    if pelvis_body.get("name") != "pelvis" and _find_body(root, "pelvis") is None:
        pelvis_body.set("name", "pelvis")

    site_name = "ufo_relabel_imu_site"
    if not any(site.get("name") == site_name for site in root.iter("site")):
        ET.SubElement(
            pelvis_body,
            "site",
            {
                "name": site_name,
                "pos": "0 0 0",
                "size": "0.01",
            },
        )

    for sensor in list(root.findall("sensor")):
        root.remove(sensor)
    sensor_root = ET.SubElement(root, "sensor")
    ET.SubElement(sensor_root, "subtreelinvel", {"name": "torso_link_subtreelinvel", "body": "pelvis"})
    ET.SubElement(sensor_root, "framezaxis", {"name": "upvector_torso", "objtype": "site", "objname": site_name})
    ET.SubElement(sensor_root, "gyro", {"name": "imu-angular-velocity", "site": site_name})

    safe_robot_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(robot_name)).strip("._") or "robot"
    return _write_relabel_xml(tree, output_dir / f"{safe_robot_name}_reward_relabel.xml", control_joint_names)


def write_g1_mjlab_relabel_xml(source_xml: Path, output_dir: Path, control_joint_names: list[str]) -> Path:
    """Use policy joint order for relabel controls while retaining G1 torso sensors."""

    source_xml = Path(source_xml).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(source_xml)
    root = tree.getroot()

    _absolutize_compiler_paths(root, source_xml)

    sensor_root = root.find("sensor")
    if sensor_root is None:
        sensor_root = ET.SubElement(root, "sensor")
    sensor_names = {sensor.get("name") for sensor in list(sensor_root) if sensor.get("name")}

    def add_sensor(tag: str, name: str, **attrs: str) -> None:
        if name in sensor_names:
            return
        ET.SubElement(sensor_root, tag, {"name": name, **attrs})
        sensor_names.add(name)

    add_sensor("subtreelinvel", "torso_link_subtreelinvel", body="torso_link")
    add_sensor("framelinvel", "frame_vel", objtype="site", objname="imu_in_torso")
    add_sensor("framezaxis", "upvector_torso", objtype="site", objname="imu_in_torso")
    add_sensor("gyro", "imu-angular-velocity", site="imu_in_torso")

    return _write_relabel_xml(tree, output_dir / "g1_mjlab_reward_relabel.xml", control_joint_names)


COORD_TO_INDEX = {"x": 0, "y": 1, "z": 2}


ALIGNMENT_BOUNDS = {"x": (-0.1, 0.1), "z": (0.9, float("inf")), "y": (-0.1, 0.1)}


REWARD_LIMITS = {
    # "l": [0, 0.5, 0.2], # get down to the knew
    "l": [0.6, 0.8, 0.2],
    "m": [1.0, float("inf"), 0.1],
    # "h": [1.3, float("inf"), 0.1],  # TODO this won't work, hard to find positions with hands above 1.3
}


def add_visual_arrow(renderer, point1, point2, rgba):
    """Adds an arrow to an mjvScene."""
    if renderer is None:
        return
    if not isinstance(rgba, np.ndarray):
        rgba = np.array(rgba).astype(np.float32)
    if not isinstance(point1, np.ndarray):
        point1 = np.array(point1).astype(np.float32)
    if not isinstance(point2, np.ndarray):
        point2 = np.array(point2).astype(np.float32)
    scene = renderer.scene
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.02,
        point1,
        point2,
    )


def add_arrow_from_xpos_to_direction(renderer, xpos, vector, rgba):
    point1 = xpos
    point2 = xpos + vector
    add_visual_arrow(renderer, point1, point2, rgba)


def rot2eul(R: np.ndarray):
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


def get_xpos(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1
    xpos = data.xpos[index].copy()
    return xpos


def get_xmat(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert index > -1
    xmat = data.xmat[index].reshape((3, 3)).copy()
    return xmat


def get_torso_upright(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    _index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    assert _index > -1
    _upright = data.xmat[_index][-2]
    return _upright


def get_center_of_mass_linvel(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    chest_subtree_linvel_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "torso_link_subtreelinvel")  # in global coordinate
    start = model.sensor_adr[chest_subtree_linvel_index]
    end = start + model.sensor_dim[chest_subtree_linvel_index]
    center_of_mass_velocity = data.sensordata[start:end].copy()
    return center_of_mass_velocity


def get_sensor_data(model: mujoco.MjModel, data: mujoco.MjData, name: str):
    chest_gyro_index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)  # in global coordinate
    assert chest_gyro_index > -1
    start = model.sensor_adr[chest_gyro_index]
    end = start + model.sensor_dim[chest_gyro_index]
    sensord = data.sensordata[start:end].copy()
    return sensord


class RewardFunction(abc.ABC):
    @abc.abstractmethod
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float: ...

    @staticmethod
    @abc.abstractmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]: ...

    def __call__(
        self,
        model: mujoco.MjModel,
        qpos: np.ndarray,
        qvel: np.ndarray,
        ctrl: np.ndarray,
    ):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        mujoco.mj_forward(model, data)
        return self.compute(model, data)


@dataclasses.dataclass
class ZeroReward(RewardFunction):
    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        return 0.0

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name.lower() in ["none", "zero", "rewardfree"]:
            return ZeroReward()
        return None


@dataclasses.dataclass
class LocomotionReward(RewardFunction):
    move_speed: float = 5
    # Head height of G1 robot after "Default" reset is 1.22
    stand_height: float = 0.5
    move_angle: float = 0
    egocentric_target: bool = True
    stay_low: bool = False

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        # head_height = data.geom("head").xpos[-1]
        # torso_upright = get_torso_upright(model, data)
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        if self.move_angle is not None:
            move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]

        if self.stay_low:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height * 0.95, self.stand_height * 1.05),
                margin=self.stand_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        # upright = rewards.tolerance(
        #             torso_upright,
        #             bounds=(-0.1, 0.1),
        #             margin=0.8,
        #             value_at_margin=0,
        #             sigmoid="linear",
        #         )
        # gravity_upright = rewards.tolerance(
        #     -gravity[-1],
        #     bounds=(0.9, float("inf")),
        #     sigmoid="linear",
        #     margin=1.9,
        #     value_at_margin=0,
        # )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        if 0 <= self.move_speed <= 0.01:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.2).mean()
            angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
            dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
            return stand_reward * dont_move * dont_rotate
        else:
            vel = center_of_mass_velocity[[0, 1]]
            com_velocity = np.linalg.norm(vel)
            move = rewards.tolerance(
                com_velocity,
                bounds=(
                    self.move_speed - 0.1 * self.move_speed,
                    self.move_speed + 0.1 * self.move_speed,
                ),
                margin=self.move_speed / 2,
                value_at_margin=0.5,
                sigmoid="gaussian",
            )
            move = (5 * move + 1) / 6
            # move in a specific direction
            if np.isclose(com_velocity, 0.0) or move_angle is None:
                angle_reward = 1.0
            else:
                direction = vel / (com_velocity + 1e-6)
                target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
                dot = target_direction.dot(direction)
                angle_reward = (dot + 1.0) / 2.0
            reward = stand_reward * move * angle_reward
            return reward

    def render(self, renderer, model, data):
        if renderer is None:
            return
        pelvis_xpos = get_xpos(model, data, "pelvis")
        com_velocity = get_center_of_mass_linvel(model, data)
        com_velocity[2] = 0  # ignore vertical velocity

        move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]
        target_direction = np.array([np.cos(move_angle), np.sin(move_angle), 0])
        target_direction = target_direction * self.move_speed

        # Visualize center of mass velocity
        add_arrow_from_xpos_to_direction(renderer, pelvis_xpos, com_velocity, (0, 1, 0, 1))

        # Visualize target direction
        add_arrow_from_xpos_to_direction(renderer, pelvis_xpos, target_direction, (1, 0, 0, 1))

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed = float(match.group(1)), float(match.group(2))
            return LocomotionReward(move_angle=move_angle, move_speed=move_speed)
        pattern = r"^move-ego-low(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            stand_height, move_angle, move_speed = float(match.group(1)), float(match.group(2)), float(match.group(3))
            print(stand_height, move_angle, move_speed)
            return LocomotionReward(move_angle=move_angle, move_speed=move_speed, stay_low=True, stand_height=stand_height)
        return None


@dataclasses.dataclass
class JumpReward(RewardFunction):
    jump_height: float = 1.4
    max_velocity: float = 5.0

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> float:
        head_height = data.geom("head").xpos[-1]
        chest_upright = get_torso_upright(model, data)
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)

        jumping = rewards.tolerance(
            head_height,
            bounds=(self.jump_height, self.jump_height + 0.1),
            margin=self.jump_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upright = rewards.tolerance(
            chest_upright,
            bounds=(0.9, float("inf")),
            sigmoid="linear",
            margin=1.9,
            value_at_margin=0,
        )
        up_velocity = rewards.tolerance(
            center_of_mass_velocity[-1],
            bounds=(self.max_velocity, float("inf")),
            sigmoid="linear",
            margin=self.max_velocity,
            value_at_margin=0,
        )
        reward = jumping * upright * up_velocity
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^jump-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            jump_height = float(match.group(1))
            return JumpReward(jump_height=jump_height)
        return None

    def render(self, renderer, model, data):
        if renderer is None:
            return
        head_xpos = data.geom("head").xpos
        target_height = np.array([head_xpos[0], head_xpos[1], self.jump_height])
        current_height = np.array([head_xpos[0], head_xpos[1], head_xpos[2]])

        # Visualize target height
        color = (0, 1, 0, 1) if head_xpos[2] >= self.jump_height else (1, 0, 0, 1)
        add_visual_arrow(renderer, current_height, target_height, color)


@dataclasses.dataclass
class RotationReward(RewardFunction):
    axis: str = "x"
    target_ang_velocity: float = 5.0
    # Note: pelvis height is 0.8 exactly after reset with default pose
    stand_pelvis_height: float = 0.8

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="pelvis")
        torso_rotation = pelvis_xmat[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")

        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)


        targ_av = np.abs(self.target_ang_velocity)
        move = rewards.tolerance(
            direction * angular_velocity[COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )

        aligned = rewards.tolerance(
            torso_rotation[COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        reward = move * height_reward * aligned
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^rotate-(x|y|z)-(-?\d+\.*\d*)-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            axis, target_ang_velocity, stand_pelvis_height = (
                match.group(1),
                float(match.group(2)),
                float(match.group(3)),
            )
            return RotationReward(
                axis=axis,
                target_ang_velocity=target_ang_velocity,
                stand_pelvis_height=stand_pelvis_height,
            )
        return None


@dataclasses.dataclass
class ArmsReward(RewardFunction):
    # head height of the character in T-pose, this is used as a reference to compute whether the character is standing
    left_pose: str = "m"
    right_pose: str = "m"
    # target height for standing pose
    stand_height: float = 0.5

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        standing = rewards.tolerance(
            root_height,
            bounds=(self.stand_height, float("inf")),
            margin=self.stand_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.2).mean()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
        # dont_move = rewards.tolerance(data.qvel, margin=0.5).mean()
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        return stand_reward * dont_move * left_arm * right_arm * dont_rotate

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^raisearms-(l|m|h|x)-(l|m|h|x)"
        match = re.search(pattern, name)
        if match:
            left_pose, right_pose = match.group(1), match.group(2)
            return ArmsReward(left_pose=left_pose, right_pose=right_pose)
        return None


@dataclasses.dataclass
class SitOnGroundReward(RewardFunction):
    pelvis_height_th: float = 0
    constrained_knees: bool = False
    knees_not_on_ground: bool = False

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        left_knee_pos = get_xpos(model, data, name="left_knee_link")[-1]
        right_knee_pos = get_xpos(model, data, name="right_knee_link")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.5).mean()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
        dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
        pelvis_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.pelvis_height_th, self.pelvis_height_th + 0.1),
            sigmoid="linear",
            margin=0.7,
            value_at_margin=0,
        )
        knee_reward = 1
        if self.constrained_knees:
            knee_reward *= rewards.tolerance(
                left_knee_pos,
                bounds=(0, 0.1),
                sigmoid="linear",
                margin=0.7,
                value_at_margin=0,
            )
            knee_reward *= rewards.tolerance(
                right_knee_pos,
                bounds=(0, 0.1),
                sigmoid="linear",
                margin=0.7,
                value_at_margin=0,
            )
        if self.knees_not_on_ground:
            knee_reward *= rewards.tolerance(
                left_knee_pos,
                bounds=(0.2, 1),
                sigmoid="linear",
                margin=0.1,
                value_at_margin=0,
            )
            knee_reward *= rewards.tolerance(
                right_knee_pos,
                bounds=(0.2, 1),
                sigmoid="linear",
                margin=0.1,
                value_at_margin=0,
            )
        return cost_orientation * dont_move * dont_rotate * pelvis_reward * (2 * knee_reward + 1) / 3

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name == "sitonground":
            pelvis_height_th = 0
            return SitOnGroundReward(pelvis_height_th=pelvis_height_th, constrained_knees=True)
        pattern = r"^crouch-(\d+\.*\d*)$"
        match = re.search(pattern, name)
        if match:
            pelvis_height_th = float(match.group(1))
            return SitOnGroundReward(pelvis_height_th=pelvis_height_th, knees_not_on_ground=True)
        return None


@dataclasses.dataclass
class ToTheKnee(RewardFunction):
    stand_height: float = 0.4

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        left_limits = [0, 0.4, 0.2]
        right_limits = [0, 0.4, 0.2]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        standing = rewards.tolerance(
            root_height,
            bounds=(self.stand_height, float("inf")),
            margin=self.stand_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation
        dont_move = rewards.tolerance(center_of_mass_velocity, margin=0.2).mean()
        # dont_move = rewards.tolerance(data.qvel, margin=0.5).mean()
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5
        return stand_reward * dont_move * left_arm * right_arm

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        if name.lower() == "totheknee":
            return ToTheKnee()


@dataclasses.dataclass
class MoveArmsReward(RewardFunction):
    move_speed: float = 5
    # Head height of G1 robot after "Default" reset is 1.22
    stand_height: float = 0.5
    move_angle: float = 0
    egocentric_target: bool = True
    low_height: float = 0.5
    stay_low: bool = False  # TODO this currently doesn't work because arm heights are expressed in global coords
    left_pose: str = "m"
    right_pose: str = "m"

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        root_height = get_xpos(model, data, "pelvis")[-1]
        center_of_mass_velocity = get_center_of_mass_linvel(model, data)
        if self.move_angle is not None:
            move_angle = np.deg2rad(self.move_angle)
        if self.egocentric_target:
            pelvis_xmat = get_xmat(model, data, name="pelvis")
            euler = rot2eul(pelvis_xmat)
            move_angle = move_angle + euler[-1]

        # STANDING HEIGHT
        if self.stay_low:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.low_height / 2, self.low_height),
                margin=self.low_height / 2,
                value_at_margin=0.01,
                sigmoid="linear",
            )
        else:
            standing = rewards.tolerance(
                root_height,
                bounds=(self.stand_height, float("inf")),
                margin=self.stand_height,
                value_at_margin=0.01,
                sigmoid="linear",
            )

        # STANDING STRAIGHT
        upvector_torso = get_sensor_data(model, data, "upvector_torso")
        cost_orientation = rewards.tolerance(
            np.sum(np.square(upvector_torso - np.array([0.073, 0.0, 1.0]))),
            bounds=(0, 0.1),
            margin=3,
            value_at_margin=0,
            sigmoid="linear",
        )
        stand_reward = standing * cost_orientation

        # SMALL CONTROL

        # ARM POSES
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        if self.move_speed == 0:
            horizontal_velocity = center_of_mass_velocity[[0, 1]]
            dont_move = rewards.tolerance(horizontal_velocity, margin=0.2).mean()
            angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")
            dont_rotate = rewards.tolerance(angular_velocity, margin=0.1).mean()
            reward = stand_reward * dont_move * dont_rotate * left_arm * right_arm
            return reward
        else:
            vel = center_of_mass_velocity[[0, 1]]
            com_velocity = np.linalg.norm(vel)
            move = rewards.tolerance(
                com_velocity,
                bounds=(
                    self.move_speed - 0.1 * self.move_speed,
                    self.move_speed + 0.1 * self.move_speed,
                ),
                margin=self.move_speed / 2,
                value_at_margin=0.5,
                sigmoid="gaussian",
            )
            move = (5 * move + 1) / 6
            # move in a specific direction
            if np.isclose(com_velocity, 0.0) or move_angle is None:
                angle_reward = 1.0
            else:
                direction = vel / (com_velocity + 1e-6)
                target_direction = np.array([np.cos(move_angle), np.sin(move_angle)])
                dot = target_direction.dot(direction)
                angle_reward = (dot + 1.0) / 2.0
            reward = stand_reward * move * angle_reward * left_arm * right_arm
            return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^move-arms-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, left_pose, right_pose = float(match.group(1)), float(match.group(2)), match.group(3), match.group(4)
            return MoveArmsReward(move_angle=move_angle, move_speed=move_speed, left_pose=left_pose, right_pose=right_pose)
        pattern = r"^move-ego-low-(-?\d+\.*\d*)-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            move_angle, move_speed, left_pose, right_pose = float(match.group(1)), float(match.group(2)), match.group(3), match.group(4)
            return MoveArmsReward(move_angle=move_angle, move_speed=move_speed, stay_low=True, left_pose=left_pose, right_pose=right_pose)
        return None


@dataclasses.dataclass
class SpinArmsReward(RewardFunction):
    axis: str = "z"
    target_ang_velocity: float = 5.0
    stand_pelvis_height: float = 0.5
    left_pose: str = "m"
    right_pose: str = "m"

    def compute(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
    ) -> float:
        pelvis_height = get_xpos(model, data, name="pelvis")[-1]
        pelvis_xmat = get_xmat(model, data, name="pelvis")
        torso_rotation = pelvis_xmat[2, :].ravel()
        angular_velocity = get_sensor_data(model, data, "imu-angular-velocity")

        # PELVIS HEIGHT
        height_reward = rewards.tolerance(
            pelvis_height,
            bounds=(self.stand_pelvis_height, float("inf")),
            margin=self.stand_pelvis_height,
            value_at_margin=0.01,
            sigmoid="linear",
        )
        direction = np.sign(self.target_ang_velocity)

        # SMALL CONTROL

        # SPINNING
        targ_av = np.abs(self.target_ang_velocity)
        move = rewards.tolerance(
            direction * angular_velocity[COORD_TO_INDEX[self.axis]],
            bounds=(targ_av, targ_av + 5),
            margin=targ_av / 2,
            value_at_margin=0,
            sigmoid="linear",
        )

        # UPRIGHT
        aligned = rewards.tolerance(
            torso_rotation[COORD_TO_INDEX[self.axis]],
            bounds=ALIGNMENT_BOUNDS[self.axis],
            sigmoid="linear",
            margin=0.9,
            value_at_margin=0,
        )

        # ARM POSES
        left_limits = REWARD_LIMITS[self.left_pose]
        right_limits = REWARD_LIMITS[self.right_pose]
        left_height = data.body("left_wrist_roll_link").xpos[-1]
        right_height = data.body("right_wrist_roll_link").xpos[-1]
        left_arm = rewards.tolerance(
            left_height,
            bounds=(left_limits[0], left_limits[1]),
            margin=left_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        left_arm = (4 * left_arm + 1) / 5
        right_arm = rewards.tolerance(
            right_height,
            bounds=(right_limits[0], right_limits[1]),
            margin=right_limits[2],
            value_at_margin=0,
            sigmoid="linear",
        )
        right_arm = (4 * right_arm + 1) / 5

        reward = move * height_reward * aligned * left_arm * right_arm
        return reward

    @staticmethod
    def reward_from_name(name: str) -> Optional["RewardFunction"]:
        pattern = r"^spin-arms-(-?\d+\.*\d*)-(l|m|h|x)-(l|m|h|x)$"
        match = re.search(pattern, name)
        if match:
            target_ang_velocity, left_pose, right_pose = (
                float(match.group(1)),
                match.group(2),
                match.group(3),
            )
            return SpinArmsReward(
                target_ang_velocity=target_ang_velocity,
                left_pose=left_pose,
                right_pose=right_pose,
            )
        return None


def make_reward_from_name(name: str | None) -> RewardFunction:
    for _class_name, reward_cls in inspect.getmembers(sys.modules[__name__], inspect.isclass):
        if not issubclass(reward_cls, RewardFunction) or inspect.isabstract(reward_cls):
            continue
        reward_obj = reward_cls.reward_from_name(name)
        if reward_obj is not None:
            return reward_obj
    raise ValueError(f"Unknown reward name: {name}")


@dataclasses.dataclass(kw_only=True)
class RewardInference:
    model: Any
    inference_dataset: Any
    num_samples_per_inference: int
    max_workers: int
    process_executor: bool = False
    process_context: str = "spawn"
    env_model: str | mujoco.MjModel = "ufo/data/robots/g1/g1_29dof.xml"

    def reward_inference(self, task: str) -> torch.Tensor:
        if isinstance(self.env_model, str):
            self.env_model = mujoco.MjModel.from_xml_path(self.env_model)

        if isinstance(self.inference_dataset, ReplayBuffer):
            if "qpos" not in self.inference_dataset.output_key_tp1:
                self.inference_dataset.output_key_tp1.append("qpos")
            if "qvel" not in self.inference_dataset.output_key_tp1:
                self.inference_dataset.output_key_tp1.append("qvel")

        if self.num_samples_per_inference >= self.inference_dataset.size() and hasattr(self.inference_dataset, "get_full_buffer"):
            data = self.inference_dataset.get_full_buffer()
        else:
            data = self.inference_dataset.sample(self.num_samples_per_inference)

        qpos = data["next"]["qpos"]
        qvel = data["next"]["qvel"]
        action = data["action"]
        if isinstance(qpos, torch.Tensor):
            qpos = qpos.cpu().detach().numpy()
            qvel = qvel.cpu().detach().numpy()
            action = action.cpu().detach().numpy()

        rewards = relabel(
            self.env_model,
            qpos,
            qvel,
            action,
            make_reward_from_name(task),
            max_workers=self.max_workers,
            process_executor=self.process_executor,
            process_context=self.process_context,
        )

        reward = torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
        return self.model.reward_wr_inference(next_obs=data["next"]["observation"], reward=reward).reshape(1, -1)


def _relabel_worker(
    x,
    model: mujoco.MjModel,
    reward_fn: RewardFunction,
):
    qpos, qvel, action = x
    assert len(qpos.shape) > 1
    assert qvel.shape[0] == qpos.shape[0]
    assert qvel.shape[0] == action.shape[0]
    rewards = np.zeros((qpos.shape[0], 1))
    for i in range(qpos.shape[0]):
        rewards[i] = reward_fn(model, qpos[i], qvel[i], action[i])
    return rewards


def relabel(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    qvel: np.ndarray,
    action: np.ndarray,
    reward_fn: RewardFunction,
    max_workers: int = 5,
    process_executor: bool = False,
    process_context: str = "spawn",
):
    chunk_size = int(np.ceil(qpos.shape[0] / max_workers))
    args = [(qpos[i : i + chunk_size], qvel[i : i + chunk_size], action[i : i + chunk_size]) for i in range(0, qpos.shape[0], chunk_size)]
    if max_workers == 1:
        result = [_relabel_worker(args[0], model=model, reward_fn=reward_fn)]
    elif process_executor:
        import multiprocessing

        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context(process_context),
        ) as exe:
            f = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
            result = exe.map(f, args)
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as exe:
            f = functools.partial(_relabel_worker, model=model, reward_fn=reward_fn)
            result = exe.map(f, args)

    return np.concatenate([r for r in result])


DEFAULT_TASKS = [
    "move-ego-0-0",
    "move-ego-low0.5-0-0",
    "move-ego-0-0.7",
    "move-ego-0-0.3",
    "move-ego-90-0.3",
    "move-ego-180-0.3",
    "move-ego--90-0.3",
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
    "raisearms-l-l",
    "raisearms-l-m",
    "raisearms-m-l",
    "raisearms-m-m",
    "move-arms-0-0.7-m-m",
    "move-arms-90-0.7-m-m",
    "move-arms-180-0.4-m-m",
    "move-arms--90-0.7-m-m",
    "move-arms-0-0.7-l-m",
    "move-arms-90-0.7-l-m",
    "move-arms-180-0.4-l-m",
    "move-arms--90-0.7-l-m",
    "move-arms-0-0.7-m-l",
    "move-arms-90-0.7-m-l",
    "move-arms-180-0.4-m-l",
    "move-arms--90-0.7-m-l",
    "move-arms-0-0.7-l-l",
    "move-arms-90-0.7-l-l",
    "move-arms-180-0.4-l-l",
    "move-arms--90-0.7-l-l",
    "spin-arms-5-l-l",
    "spin-arms--5-l-l",
    "spin-arms-5-l-m",
    "spin-arms--5-l-m",
    "spin-arms-5-m-l",
    "spin-arms--5-m-l",
    "crouch-0",
    "crouch-0.25",
    "sitonground",
]


NON_G1_LOCOMOTION_TASKS = [task for task in DEFAULT_TASKS if task.startswith("move-ego-") or task.startswith("rotate-z-")]


def _is_g1_reward_robot(robot_training) -> bool:
    return robot_training.robot.name == "g1_29dof" and len(robot_training.robot.control_joint_names) == 29


def _is_non_g1_locomotion_task(task: str) -> bool:
    patterns = (
        r"^move-ego-(-?\d+\.*\d*)-(-?\d+\.*\d*)$",
        r"^move-ego-low(-?\d+\.*\d*)-(-?\d+\.*\d*)-(-?\d+\.*\d*)$",
        r"^rotate-z-(-?\d+\.*\d*)-(\d+\.*\d*)$",
    )
    return any(re.search(pattern, task) for pattern in patterns)


def _resolve_reward_tasks(tasks: list[str] | None, robot_training) -> tuple[list[str], str]:
    if _is_g1_reward_robot(robot_training):
        return list(tasks or DEFAULT_TASKS), "G1 full tasks"

    selected_tasks = list(tasks or NON_G1_LOCOMOTION_TASKS)
    for task in selected_tasks:
        if not _is_non_g1_locomotion_task(task):
            raise ValueError(f"Task {task} requires G1-style arm/body semantics and is not enabled for robot {robot_training.robot.name}.")
    return selected_tasks, "non-G1 locomotion-only tasks"


def _default_standing_target_states(wrapped_env, device: str) -> dict[str, torch.Tensor]:
    core_env = wrapped_env._env
    num_envs = int(core_env.num_envs)
    env_device = core_env.device

    init_state = core_env.config.robot.init_state
    root_pos = torch.as_tensor(init_state.pos, device=env_device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    if hasattr(core_env, "env_origins"):
        root_pos = root_pos + core_env.env_origins.to(device=env_device, dtype=torch.float32)
    root_rot_xyzw = torch.as_tensor(init_state.rot, device=env_device, dtype=torch.float32).unsqueeze(0).repeat(num_envs, 1)
    root_lin_vel = torch.zeros((num_envs, 3), device=env_device, dtype=torch.float32)
    root_ang_vel = torch.zeros((num_envs, 3), device=env_device, dtype=torch.float32)
    root_state_xyzw = torch.cat([root_pos, root_rot_xyzw, root_lin_vel, root_ang_vel], dim=-1)

    dof_state = torch.zeros((num_envs, core_env.num_dof, 2), device=env_device, dtype=torch.float32)
    dof_state[..., 0] = core_env.default_dof_pos.to(device=env_device, dtype=torch.float32)
    return {
        "root_states": root_state_xyzw.to(device=device, dtype=torch.float32),
        "dof_states": dof_state.to(device=device, dtype=torch.float32),
    }


def _read_replay_snapshot(buffer_path: Path) -> dict:
    marker = buffer_path / "snapshot.json"
    try:
        snapshot = json.loads(marker.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"Replay buffer snapshot is missing or incomplete: {marker}") from error
    if (
        not isinstance(snapshot, dict)
        or not isinstance(snapshot.get("id"), str)
        or not snapshot["id"]
        or type(snapshot.get("global_time")) is not int
        or snapshot["global_time"] < 0
        or type(snapshot.get("world_size")) is not int
        or snapshot["world_size"] < 1
    ):
        raise ValueError(f"Invalid replay buffer snapshot: {marker}")
    return snapshot


def _load_replay_buffer(
    model_folder: Path,
    *,
    buffer_rank: int,
    buffer_path: Path | None,
) -> tuple[object, Path]:
    if buffer_path is None:
        buffer_path = resolve_checkpoint_dir(model_folder / "checkpoint") / "buffers" / f"train_rank_{buffer_rank}"
    buffer_path = buffer_path.expanduser().resolve()
    if not buffer_path.is_dir():
        raise FileNotFoundError(f"Missing replay buffer path: {buffer_path}")
    snapshot = _read_replay_snapshot(buffer_path)

    with (buffer_path / "config.json").open() as stream:
        config = json.load(stream)
    # The class rename leaves the replay layout unchanged, including existing runs.
    if config.get("__target__") not in (
        "ufo.agents.common.buffers.trajectory.ReplayBuffer",
        "ufo.agents.common.buffers.trajectory.TrajectoryDictBufferMultiDim",
    ):
        raise ValueError("Reward inference requires the current trajectory replay format")
    dataset = ReplayBuffer.load(buffer_path, device="cpu")
    if _read_replay_snapshot(buffer_path) != snapshot:
        raise RuntimeError(f"Replay buffer snapshot changed while loading: {buffer_path}")
    return dataset, buffer_path


def _require_fb_checkpoint(model_folder: Path) -> None:
    with (resolve_checkpoint_dir(Path(model_folder) / "checkpoint") / "config.json").open() as stream:
        config = json.load(stream)
    if config["model"]["name"] != "FBcprAuxModel":
        raise ValueError("Reward inference requires an FB checkpoint; TeCH supports tracking and goal inference only.")


def run_reward_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    robot_config: Path | None,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    episode_length: int,
    num_samples: int,
    n_inferences: int,
    skip_rollouts: bool,
    tasks: list[str],
    buffer_rank: int,
    buffer_path: Path | None,
    max_workers: int,
    process_executor: bool,
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    fps: int,
    max_episode_length_s: float,
    export_onnx: bool,
) -> None:
    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = resolve_checkpoint_dir(model_folder / "checkpoint")
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    _require_fb_checkpoint(model_folder)
    env_cfg, robot_training = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=robot_config,
        device=device,

        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=max_episode_length_s,
    )
    robot_xml = Path(robot_training.robot.xml_path).expanduser().resolve()
    if not robot_xml.exists():
        raise FileNotFoundError(f"Missing robot XML: {robot_xml}")
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    is_g1 = _is_g1_reward_robot(robot_training)
    tasks, task_support_mode = _resolve_reward_tasks(tasks, robot_training)

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()
    if not hasattr(model, "_backward_map"):
        raise ValueError("Reward inference requires an FB checkpoint. TeCH supports tracking and goal inference, not reward-conditioned inference.")

    if export_onnx:
        export_policy_model(model, model_folder / "exported", robot_training, env_config=env_cfg)

    print("[INFO] Loading replay buffer...", end=" ", flush=True)
    start_t = time.time()
    dataset, loaded_buffer_path = _load_replay_buffer(model_folder, buffer_rank=buffer_rank, buffer_path=buffer_path)
    print(f"done in {time.time() - start_t:.2f}s")
    print(f"[INFO] Replay buffer={loaded_buffer_path}")
    if hasattr(dataset, "size"):
        print(f"[INFO] Replay buffer sampled transition count={dataset.size()}")

    output_dir = model_folder / "reward_inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    if is_g1:
        relabel_xml = write_g1_mjlab_relabel_xml(robot_xml, output_dir, control_joint_names)
    else:
        relabel_xml = write_mjlab_relabel_xml(
            robot_xml,
            output_dir,
            control_joint_names,
            robot_training.robot.name,
            root_body_name=robot_training.robot.base_body,
        )

    reward_eval_agent = RewardInference(
        model=model,
        inference_dataset=dataset,
        num_samples_per_inference=int(num_samples),
        max_workers=int(max_workers),
        process_executor=bool(process_executor),
        env_model=str(relabel_xml),
    )

    print(f"[INFO] UFO reward inference model_folder={model_folder}")
    print(f"[INFO] Robot={robot_training.robot.name}")
    print(f"[INFO] Robot config={Path(robot_training.config_path).expanduser().resolve()}")
    print(f"[INFO] num_dof={num_dof}")
    print(f"[INFO] Reward source XML={robot_xml}")
    print(f"[INFO] Reward relabel XML={relabel_xml}")
    print(f"[INFO] task support mode={task_support_mode}")
    print(f"[INFO] device={device} save_mp4={save_mp4} skip_rollouts={skip_rollouts}")
    print(f"[INFO] tasks={tasks}")

    z_dict: dict[str, list[np.ndarray]] = {}
    output_path = output_dir / "reward_locomotion.pkl"
    for inference_idx in range(int(n_inferences)):
        for task in tasks:
            print(f"[INFO] Started reward inference {inference_idx + 1}/{n_inferences} for {task}...", end=" ", flush=True)
            start_t = time.time()
            z = reward_eval_agent.reward_inference(task=task)
            z_dict.setdefault(task, []).append(z.detach().cpu().numpy())
            print(f"done in {time.time() - start_t:.2f}s")
            joblib.dump(z_dict, output_path)
            print(f"[INFO] Saved reward embeddings: {output_path}")

    if skip_rollouts:
        return

    wrapped_env, _ = env_cfg.build(num_envs=1)
    renderer = None
    try:
        print(f"[INFO] Generating rollout videos with XML={env_cfg.mjcf_path}")
        if save_mp4:
            renderer = MujocoQposRenderer(
                robot_xml,
                render_size=render_size,
                camera_distance=camera_distance,
                camera_azimuth=camera_azimuth,
                camera_elevation=camera_elevation,
                expected_qpos_size=7 + num_dof,
            )
        for task in tasks:
            video_path = video_dir / f"{task}.mp4"
            with video_writer(video_path if save_mp4 else None, shape=(render_size, render_size), fps=fps) as writer:
                for z_cpu in z_dict[task]:
                    z = torch.as_tensor(z_cpu, device=device, dtype=torch.float32)
                    target_states = _default_standing_target_states(wrapped_env, device=device)
                    observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)
                    print("[INFO] Reset reward rollout to default standing pose.")
                    if save_mp4:
                        frame = render_policy_frame(wrapped_env, renderer)
                        writer.add_image(frame)
                    for step in range(int(episode_length)):
                        action = model.act(observation, z, mean=True)
                        observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                        if save_mp4:
                            frame = render_policy_frame(wrapped_env, renderer)
                            writer.add_image(frame)
                        if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                            print(f"[INFO] Task {task} episode ended at step={step}; stopping this rollout.")
                            break
            if save_mp4:
                print(f"[INFO] Saved reward rollout video for {task}: {video_path}")
    finally:
        if renderer is not None:
            renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFO reward inference.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    add_robot_config_manifest_args(parser, purpose="reward inference")

    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save policy rollout MP4s.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=150_000)
    parser.add_argument("--n-inferences", type=int, default=1)
    add_bool_arg(parser, "--skip-rollouts", False, "Only compute reward embeddings; do not create rollout videos.")
    parser.add_argument("--list-tasks", action="store_true", help="List tasks supported by the checkpoint robot without loading replay.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Optional task subset. Defaults to the full locomotion task list.")
    parser.add_argument("--buffer-rank", type=int, default=0, help="Rank-local replay buffer shard to use, e.g. train_rank_0.")
    parser.add_argument("--buffer-path", type=Path, default=None, help="Explicit replay buffer directory; overrides --buffer-rank.")
    parser.add_argument("--max-workers", type=int, default=24)
    add_bool_arg(parser, "--process-executor", True, "Use ProcessPoolExecutor for reward relabel workers.")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    add_bool_arg(parser, "--export-onnx", False, "Export ONNX next to the checkpoint before inference.")
    return resolve_inference_data_and_robot_args(parser.parse_args(), parser)


def main() -> None:
    args = parse_args()
    if args.list_tasks:
        _require_fb_checkpoint(args.model_folder)
        _, robot_training = load_mjlab_env_cfg(
            args.model_folder, data_path=args.data_path, robot_config=args.robot_config,
            device="cpu", disable_dr=True, disable_obs_noise=True, max_episode_length_s=args.max_episode_length_s,
        )
        tasks, support = _resolve_reward_tasks(None, robot_training)
        print(support)
        print("\n".join(tasks))
        return
    run_reward_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,

        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        episode_length=args.episode_length,
        num_samples=args.num_samples,
        n_inferences=args.n_inferences,
        skip_rollouts=args.skip_rollouts,
        tasks=args.tasks,
        buffer_rank=args.buffer_rank,
        buffer_path=args.buffer_path,
        max_workers=args.max_workers,
        process_executor=args.process_executor,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        fps=args.fps,
        max_episode_length_s=args.max_episode_length_s,
        export_onnx=args.export_onnx,
    )


if __name__ == "__main__":
    main()
