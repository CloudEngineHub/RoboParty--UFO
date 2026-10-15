import copy
import xml.etree.ElementTree as ETree
from collections import OrderedDict
from pathlib import Path

import numpy as np
import scipy.ndimage.filters as filters
import torch
from easydict import EasyDict
from loguru import logger

from ufo.math import (
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quat_angle_axis,
    quat_inverse,
    quat_mul_norm,
    quaternion_to_matrix,
    wxyz_to_xyzw,
)


class HumanoidKinematics:
    def __init__(self, cfg, device=torch.device("cpu")):
        self.cfg = cfg
        self.asset_root = Path(cfg.asset.assetRoot)
        self.asset_file = cfg.asset.assetFileName
        self.mjcf_file = self.asset_root / self.asset_file

        tree = ETree.parse(self.mjcf_file)

        joints = sorted([j.attrib["name"] for j in tree.getroot().find("worldbody").findall(".//joint")])
        motors = sorted([m.attrib["name"] for m in tree.getroot().find("actuator")])

        assert len(motors) > 0, "No motors found in the mjcf file"

        self.num_dof = len(motors)

        mjcf_data = self.from_mjcf(self.mjcf_file)
        self.body_names = copy.deepcopy(mjcf_data["node_names"])
        self._parents = mjcf_data["parent_indices"]

        self.body_names_augment = copy.deepcopy(mjcf_data["node_names"])
        self._offsets = mjcf_data["local_translation"][None,].to(device)

        self._local_rotation = mjcf_data["local_rotation"][None,].to(device)
        self.actuated_joints_idx = np.array([self.body_names.index(k) for k, v in mjcf_data["body_to_joint"].items()])

        for m in motors:
            if m not in joints:
                print(m)

        for extend_config in cfg.extend_config:
            self.body_names_augment += [extend_config.joint_name]
            self._parents = torch.cat([self._parents, torch.tensor([self.body_names.index(extend_config.parent_name)]).to(device)], dim=0)
            self._offsets = torch.cat([self._offsets, torch.tensor([[extend_config.pos]]).to(device)], dim=1)
            self._local_rotation = torch.cat([self._local_rotation, torch.tensor([[extend_config.rot]]).to(device)], dim=1)

        self.num_bodies = len(self.body_names)
        self.num_bodies_augment = len(self.body_names_augment)

        self._local_rotation_mat = quaternion_to_matrix(self._local_rotation).float()  # w, x, y ,z
        if getattr(cfg, "has_mesh", False):
            raise ValueError("Mesh height correction is retired; preprocess motion heights before training")

    def from_mjcf(self, path):
        # function from Poselib:
        tree = ETree.parse(path)
        xml_doc_root = tree.getroot()
        xml_world_body = xml_doc_root.find("worldbody")
        if xml_world_body is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")
        # assume this is the root
        xml_body_root = xml_world_body.find("body")
        if xml_body_root is None:
            raise ValueError("MJCF parsed incorrectly please verify it.")


        node_names = []
        parent_indices = []
        local_translation = []
        local_rotation = []
        joints_range = []
        body_to_joint = OrderedDict()

        # recursively adding all nodes into the skel_tree
        def _add_xml_node(xml_node, parent_index, node_index):
            node_name = xml_node.attrib.get("name")
            # parse the local translation into float list
            pos = np.fromstring(xml_node.attrib.get("pos", "0 0 0"), dtype=float, sep=" ")
            quat = np.fromstring(xml_node.attrib.get("quat", "1 0 0 0"), dtype=float, sep=" ")
            node_names.append(node_name)
            parent_indices.append(parent_index)
            local_translation.append(pos)
            local_rotation.append(quat)
            curr_index = node_index
            node_index += 1
            all_joints = xml_node.findall("joint")  # joints need to remove the first 6 joints
            if len(all_joints) == 6:
                all_joints = all_joints[6:]

            for joint in all_joints:
                if joint.attrib.get("range") is not None:
                    joints_range.append(np.fromstring(joint.attrib.get("range"), dtype=float, sep=" "))
                else:
                    if not joint.attrib.get("type") == "free":
                        joints_range.append([-np.pi, np.pi])
            for joint_node in xml_node.findall("joint"):
                body_to_joint[node_name] = joint_node.attrib.get("name")

            for next_node in xml_node.findall("body"):
                node_index = _add_xml_node(next_node, curr_index, node_index)

            return node_index

        _add_xml_node(xml_body_root, -1, 0)
        assert len(joints_range) == self.num_dof
        return {
            "node_names": node_names,
            "parent_indices": torch.from_numpy(np.array(parent_indices, dtype=np.int32)),
            "local_translation": torch.from_numpy(np.array(local_translation, dtype=np.float32)),
            "local_rotation": torch.from_numpy(np.array(local_rotation, dtype=np.float32)),
            "body_to_joint": body_to_joint,
        }

    def fk_batch(self, pose, trans, dt=1 / 30):
        device = pose.device
        B, seq_len = pose.shape[:2]

        pose = pose[..., : len(self._parents), :]  # G1 fitted joints might have extra joints

        if self.num_bodies_augment > 0 and pose.shape[2] < self.num_bodies + self.num_bodies_augment:
            pose = torch.cat([pose, torch.zeros(B, seq_len, self.num_bodies_augment - pose.shape[2], pose.shape[3]).to(device)], dim=2)

        pose_quat = axis_angle_to_quaternion(pose.clone())
        pose_mat = quaternion_to_matrix(pose_quat)

        pose_mat = pose_mat.reshape(B, seq_len, -1, 3, 3)
        wbody_pos, wbody_mat = self.forward_kinematics_batch(pose_mat[:, :, 1:], pose_mat[:, :, 0:1], trans)

        return_dict = EasyDict()

        wbody_rot = wxyz_to_xyzw(matrix_to_quaternion(wbody_mat))
        if len(self.cfg.extend_config) > 0:
            return_dict.global_velocity_extend = self._compute_velocity(wbody_pos, dt)
            return_dict.global_angular_velocity_extend = self._compute_angular_velocity(wbody_rot, dt)

            return_dict.global_translation_extend = wbody_pos.clone()
            return_dict.global_rotation_extend = wbody_rot

            wbody_pos = wbody_pos[..., : self.num_bodies, :]
            wbody_rot = wbody_rot[..., : self.num_bodies, :]

        return_dict.global_translation = wbody_pos
        return_dict.global_rotation = wbody_rot
        rigidbody_linear_velocity = self._compute_velocity(
            wbody_pos, dt
        )  # Isaac gym is [x, y, z, w]. All the previous functions are [w, x, y, z]
        rigidbody_angular_velocity = self._compute_angular_velocity(wbody_rot, dt)
        return_dict.global_angular_velocity = rigidbody_angular_velocity
        return_dict.global_velocity = rigidbody_linear_velocity

        if len(self.cfg.extend_config) > 0:
            return_dict.dof_pos = pose.sum(dim=-1)[
                ..., 1 : self.num_bodies
            ]  # you can sum it up since unitree's each joint has 1 dof. Last two are for hands. doesn't really matter.
        else:
            if not len(self.actuated_joints_idx) == len(self.body_names):
                return_dict.dof_pos = pose.sum(dim=-1)[..., self.actuated_joints_idx]
            else:
                return_dict.dof_pos = pose.sum(dim=-1)[..., 1:]

        dof_vel = (return_dict.dof_pos[:, 1:] - return_dict.dof_pos[:, :-1]) / dt
        return_dict.dof_vels = torch.cat([dof_vel, dof_vel[:, -2:-1]], dim=1)
        return_dict.fps = int(1 / dt)

        return return_dict

    def forward_kinematics_batch(self, rotations, root_rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where B = batch size, J = number of joints):
         -- rotations: (B, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (B, 3) tensor describing the root joint positions.
        Output: joint positions (B, J, 3)
        """

        device, dtype = root_rotations.device, root_rotations.dtype
        B, seq_len = rotations.size()[0:2]
        J = self._offsets.shape[1]
        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets[:, None].expand(B, seq_len, J, 3).to(device).type(dtype)

        for i in range(J):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(root_rotations)
            else:
                try:
                    jpos = (
                        torch.matmul(rotations_world[self._parents[i]][:, :, 0], expanded_offsets[:, :, i, :, None]).squeeze(-1)
                        + positions_world[self._parents[i]]
                    )
                    rot_mat = torch.matmul(
                        rotations_world[self._parents[i]],
                        torch.matmul(self._local_rotation_mat[:, (i) : (i + 1)], rotations[:, :, (i - 1) : i, :]),
                    )
                except Exception as e:
                    logger.error(f"Error at joint index {i}")
                    logger.error(f"Parent index: {self._parents[i]}")
                    logger.error(f"Error details: {str(e)}")
                # rot_mat = torch.matmul(rotations_world[self._parents[i]], rotations[:, :, (i - 1):i, :])

                positions_world.append(jpos)
                rotations_world.append(rot_mat)

        positions_world = torch.stack(positions_world, dim=2)
        rotations_world = torch.cat(rotations_world, dim=2)
        return positions_world, rotations_world

    @staticmethod
    def _compute_velocity(p, time_delta):
        velocity = np.gradient(p.numpy(), axis=-3) / time_delta
        return torch.from_numpy(filters.gaussian_filter1d(velocity, 2, axis=-3, mode="nearest")).to(p)

    @staticmethod
    def _compute_angular_velocity(r, time_delta: float):
        # assume the second last dimension is the time axis
        diff_quat_data = torch.zeros_like(r)
        diff_quat_data[..., 3] = 1
        diff_quat_data[..., :-1, :, :] = quat_mul_norm(r[..., 1:, :, :], quat_inverse(r[..., :-1, :, :], w_last=True), w_last=True)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data, w_last=True)
        angular_velocity = diff_axis * diff_angle.unsqueeze(-1) / time_delta
        return torch.from_numpy(filters.gaussian_filter1d(angular_velocity.numpy(), 2, axis=-3, mode="nearest"))
