"""Inspect robot XML and generate configuration drafts, optionally using URDF hints."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import mujoco
import yaml
from loguru import logger
from omegaconf import OmegaConf

from ufo.tools.robot_config import _mj_name, load_robot_spec, load_robot_training_spec

_SEMANTIC_TOKENS = (
    "left",
    "right",
    "hip",
    "knee",
    "ankle",
    "shoulder",
    "elbow",
    "wrist",
    "foot",
    "toe",
    "hand",
    "torso",
    "pelvis",
    "chest",
    "trunk",
)


@dataclass(frozen=True)
class UrdfJointInfo:
    name: str
    type: str
    parent: str | None = None
    child: str | None = None
    axis: list[float] | None = None
    limit_lower: float | None = None
    limit_upper: float | None = None
    limit_effort: float | None = None
    limit_velocity: float | None = None
    dynamics_damping: float | None = None
    dynamics_friction: float | None = None
    mimic_joint: str | None = None
    mimic_multiplier: float | None = None
    mimic_offset: float | None = None
    semantic_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UrdfLinkInfo:
    name: str
    visual_meshes: list[str] = field(default_factory=list)
    collision_meshes: list[str] = field(default_factory=list)
    semantic_hints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class UrdfTransmissionInfo:
    name: str
    joint_name: str | None = None
    actuator_name: str | None = None


@dataclass(frozen=True)
class UrdfRobotInfo:
    source_path: str
    joints: dict[str, UrdfJointInfo]
    links: dict[str, UrdfLinkInfo]
    transmissions: list[UrdfTransmissionInfo]
    warnings: list[str] = field(default_factory=list)


def _tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _children(element: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(element) if _tag_name(child.tag) == name]


def _first_child(element: ET.Element, name: str) -> ET.Element | None:
    for child in list(element):
        if _tag_name(child.tag) == name:
            return child
    return None


def _to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _axis(value: str | None) -> list[float] | None:
    if not value:
        return None
    parts = value.split()
    if len(parts) != 3:
        return None
    try:
        return [float(item) for item in parts]
    except ValueError:
        return None


def _semantic_hints(*names: str | None) -> list[str]:
    text = " ".join(name or "" for name in names).lower()
    hints = [token for token in _SEMANTIC_TOKENS if token in text]
    if text.startswith("l_") or "_l_" in text:
        hints.append("left")
    if text.startswith("r_") or "_r_" in text:
        hints.append("right")
    return sorted(set(hints))


def _mesh_paths(element: ET.Element, section_name: str) -> list[str]:
    paths: list[str] = []
    for section in _children(element, section_name):
        geometry = _first_child(section, "geometry")
        if geometry is None:
            continue
        mesh = _first_child(geometry, "mesh")
        if mesh is None:
            continue
        filename = mesh.attrib.get("filename")
        if filename:
            paths.append(str(filename))
    return paths


def _parse_joint(element: ET.Element) -> UrdfJointInfo | None:
    name = element.attrib.get("name")
    if not name:
        return None
    parent = _first_child(element, "parent")
    child = _first_child(element, "child")
    axis = _first_child(element, "axis")
    limit = _first_child(element, "limit")
    dynamics = _first_child(element, "dynamics")
    mimic = _first_child(element, "mimic")
    parent_link = parent.attrib.get("link") if parent is not None else None
    child_link = child.attrib.get("link") if child is not None else None
    return UrdfJointInfo(
        name=str(name),
        type=str(element.attrib.get("type", "")),
        parent=parent_link,
        child=child_link,
        axis=_axis(axis.attrib.get("xyz") if axis is not None else None),
        limit_lower=_to_float(limit.attrib.get("lower") if limit is not None else None),
        limit_upper=_to_float(limit.attrib.get("upper") if limit is not None else None),
        limit_effort=_to_float(limit.attrib.get("effort") if limit is not None else None),
        limit_velocity=_to_float(limit.attrib.get("velocity") if limit is not None else None),
        dynamics_damping=_to_float(dynamics.attrib.get("damping") if dynamics is not None else None),
        dynamics_friction=_to_float(dynamics.attrib.get("friction") if dynamics is not None else None),
        mimic_joint=mimic.attrib.get("joint") if mimic is not None else None,
        mimic_multiplier=_to_float(mimic.attrib.get("multiplier") if mimic is not None else None),
        mimic_offset=_to_float(mimic.attrib.get("offset") if mimic is not None else None),
        semantic_hints=_semantic_hints(name, parent_link, child_link),
    )


def _parse_link(element: ET.Element) -> UrdfLinkInfo | None:
    name = element.attrib.get("name")
    if not name:
        return None
    return UrdfLinkInfo(
        name=str(name),
        visual_meshes=_mesh_paths(element, "visual"),
        collision_meshes=_mesh_paths(element, "collision"),
        semantic_hints=_semantic_hints(name),
    )


def _parse_transmission(element: ET.Element) -> UrdfTransmissionInfo:
    joint = _first_child(element, "joint")
    actuator = _first_child(element, "actuator")
    return UrdfTransmissionInfo(
        name=str(element.attrib.get("name", "")),
        joint_name=joint.attrib.get("name") if joint is not None else None,
        actuator_name=actuator.attrib.get("name") if actuator is not None else None,
    )


def parse_urdf_robot(path: str | Path) -> UrdfRobotInfo:
    urdf_path = Path(path).expanduser()
    root = ET.parse(urdf_path).getroot()
    warnings: list[str] = []
    if _tag_name(root.tag) != "robot":
        warnings.append(f"URDF root tag is {_tag_name(root.tag)!r}, expected 'robot'.")
    joints: dict[str, UrdfJointInfo] = {}
    links: dict[str, UrdfLinkInfo] = {}
    transmissions: list[UrdfTransmissionInfo] = []
    for child in list(root):
        tag = _tag_name(child.tag)
        if tag == "joint":
            joint = _parse_joint(child)
            if joint is not None:
                if joint.name in joints:
                    warnings.append(f"Duplicate URDF joint {joint.name!r}; keeping the last definition.")
                joints[joint.name] = joint
        elif tag == "link":
            link = _parse_link(child)
            if link is not None:
                if link.name in links:
                    warnings.append(f"Duplicate URDF link {link.name!r}; keeping the last definition.")
                links[link.name] = link
        elif tag == "transmission":
            transmissions.append(_parse_transmission(child))
    return UrdfRobotInfo(
        source_path=str(urdf_path),
        joints=joints,
        links=links,
        transmissions=transmissions,
        warnings=warnings,
    )


def _ordered_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if str(value)))


def _contains_any(name: str, tokens: Iterable[str]) -> bool:
    lower = name.lower()
    return any(token in lower for token in tokens)


def _is_left(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("left") or "_left" in lower or lower.startswith("l_") or "_l_" in lower


def _is_right(name: str) -> bool:
    lower = name.lower()
    return lower.startswith("right") or "_right" in lower or lower.startswith("r_") or "_r_" in lower


def _read_actuator_joints(model: mujoco.MjModel) -> tuple[list[str], list[str]]:
    joints: list[str] = []
    non_joint_actuators: list[str] = []
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            non_joint_actuators.append(_mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id))
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        if joint_id < 0:
            non_joint_actuators.append(_mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id))
            continue
        joints.append(_mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    return joints, non_joint_actuators


def infer_control_joints_from_xml(model: mujoco.MjModel) -> tuple[list[str], list[str]]:
    """Return unique joint-transmission actuator joints in MuJoCo actuator order."""
    joints, non_joint_actuators = _read_actuator_joints(model)
    joints = _ordered_unique(joints)
    warnings: list[str] = []
    if not joints:
        raise ValueError("MuJoCo XML has no joint-transmission actuators; cannot infer control joints for training.")
    if non_joint_actuators:
        warnings.append(f"Ignored non-joint actuators while inferring control joints: {non_joint_actuators}")
    if len(joints) < model.nu:
        warnings.append("Control joints were inferred only from joint-transmission actuators.")
    return joints, warnings


def infer_body_names_from_xml(model: mujoco.MjModel) -> list[str]:
    return [_mj_name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(1, model.nbody)]


def infer_freejoint_base_body(model: mujoco.MjModel) -> tuple[str, str | None, list[str]]:
    joint_name, body_name = _freejoint_body(model)
    if joint_name is not None:
        return body_name, joint_name, []
    body_names = infer_body_names_from_xml(model)
    fallback = body_names[0] if body_names else ""
    return (
        fallback,
        None,
        [
            f"No freejoint found in XML; using first non-world body as base_body={fallback!r}. Training a floating humanoid usually requires a freejoint."
        ],
    )


def _joint_id_map(model: mujoco.MjModel) -> dict[str, int]:
    return {_mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, idx): idx for idx in range(model.njnt)}


def _actuator_for_joint(model: mujoco.MjModel, joint_name: str) -> int | None:
    joint_ids = _joint_id_map(model)
    joint_id = joint_ids[joint_name]
    for actuator_id in range(model.nu):
        if int(model.actuator_trntype[actuator_id]) != int(mujoco.mjtTrn.mjTRN_JOINT):
            continue
        if int(model.actuator_trnid[actuator_id, 0]) == joint_id:
            return actuator_id
    return None


def infer_joint_ranges_from_xml(
    model: mujoco.MjModel, control_joints: list[str]
) -> tuple[dict[str, list[float]], list[str], list[list[float]]]:
    joint_ids = _joint_id_map(model)
    ranges: dict[str, list[float]] = {}
    lower: list[float] = []
    upper: list[float] = []
    warnings: list[str] = []
    for joint_name in control_joints:
        joint_id = joint_ids[joint_name]
        if int(model.jnt_limited[joint_id]):
            lo, hi = (float(model.jnt_range[joint_id, 0]), float(model.jnt_range[joint_id, 1]))
        else:
            lo, hi = -3.141592653589793, 3.141592653589793
            warnings.append(f"Joint {joint_name!r} has no XML range; using [-pi, pi] draft limits.")
        ranges[joint_name] = [lo, hi]
        lower.append(lo)
        upper.append(hi)
    return ranges, warnings, [lower, upper]


def infer_actuator_limits_from_xml(
    model: mujoco.MjModel,
    control_joints: list[str],
    *,
    effort_limit_default: float = 80.0,
    velocity_limit_default: float = 20.0,
    armature_default: float = 0.01,
    friction_default: float = 0.0,
) -> tuple[list[float], list[float], dict[str, dict[str, float]], list[str]]:
    joint_ids = _joint_id_map(model)
    warnings: list[str] = []
    effort_limits: list[float] = []
    velocity_limits: list[float] = []
    actuator_joints: dict[str, dict[str, float]] = {}
    for joint_name in control_joints:
        joint_id = joint_ids[joint_name]
        effort_limit, effort_source = _xml_effort_limit_for_joint(model, joint_name)
        if effort_source == "XML actuator ctrlrange":
            warnings.append(f"Joint {joint_name!r} actuator has no force range; using ctrlrange as draft effort_limit.")
        if effort_limit is None or effort_limit <= 0.0:
            effort_limit = float(effort_limit_default)
            warnings.append(f"Joint {joint_name!r} has no actuator force/ctrl range; using effort_limit_default={effort_limit_default}.")
        velocity_limit = float(velocity_limit_default)
        warnings.append(
            f"Joint {joint_name!r} velocity limit is not reliably represented in MJCF; using velocity_limit_default={velocity_limit_default}."
        )
        dof_addr = int(model.jnt_dofadr[joint_id])
        armature = float(model.dof_armature[dof_addr]) if dof_addr >= 0 else float(armature_default)
        friction = float(model.dof_frictionloss[dof_addr]) if dof_addr >= 0 else float(friction_default)
        if armature == 0.0:
            armature = float(armature_default)
            warnings.append(f"Joint {joint_name!r} has zero XML armature; using armature_default={armature_default}.")
        if friction == 0.0 and friction_default != 0.0:
            friction = float(friction_default)
            warnings.append(f"Joint {joint_name!r} has zero XML frictionloss; using friction_default={friction_default}.")
        effort_limits.append(float(effort_limit))
        velocity_limits.append(velocity_limit)
        actuator_joints[joint_name] = {
            "effort_limit": float(effort_limit),
            "velocity_limit": velocity_limit,
            "armature": armature,
            "friction": friction,
        }
    return effort_limits, velocity_limits, actuator_joints, warnings


def _positive(value: float | None) -> bool:
    return value is not None and float(value) > 0.0


def _xml_effort_limit_for_joint(model: mujoco.MjModel, joint_name: str) -> tuple[float | None, str | None]:
    actuator_id = _actuator_for_joint(model, joint_name)
    if actuator_id is None:
        return None, None
    force_range = [float(v) for v in model.actuator_forcerange[actuator_id].tolist()]
    if bool(model.actuator_forcelimited[actuator_id]) and force_range[0] != force_range[1]:
        return max(abs(force_range[0]), abs(force_range[1])), "XML actuator forcerange"
    ctrl_range = [float(v) for v in model.actuator_ctrlrange[actuator_id].tolist()]
    if bool(model.actuator_ctrllimited[actuator_id]) and ctrl_range[0] != ctrl_range[1]:
        return max(abs(ctrl_range[0]), abs(ctrl_range[1])), "XML actuator ctrlrange"
    return None, None


def _xml_dof_value(model: mujoco.MjModel, joint_name: str, values, default: float) -> float:
    joint_id = _joint_id_map(model)[joint_name]
    dof_addr = int(model.jnt_dofadr[joint_id])
    if dof_addr < 0:
        return float(default)
    return float(values[dof_addr])


def _urdf_joint_for_xml_joint(
    urdf_info: UrdfRobotInfo,
    xml_joint_name: str,
    joint_name_map: dict[str, str] | None,
) -> tuple[str | None, UrdfJointInfo | None]:
    mapped_name = (joint_name_map or {}).get(xml_joint_name, xml_joint_name)
    joint = urdf_info.joints.get(mapped_name)
    return mapped_name, joint


def _urdf_semantic_hints_for_joints(urdf_info: UrdfRobotInfo, urdf_joint_names: list[str]) -> dict[str, dict[str, list[str]]]:
    joint_hints: dict[str, list[str]] = {}
    link_hints: dict[str, list[str]] = {}
    for joint_name in _ordered_unique(urdf_joint_names):
        joint = urdf_info.joints.get(joint_name)
        if joint is None:
            continue
        if joint.semantic_hints:
            joint_hints[joint_name] = list(joint.semantic_hints)
        for link_name in (joint.parent, joint.child):
            if link_name is None:
                continue
            link = urdf_info.links.get(link_name)
            if link is not None and link.semantic_hints:
                link_hints[link_name] = list(link.semantic_hints)
    return {"joints": joint_hints, "links": link_hints}


def _right_counterpart_name(name: str) -> str | None:
    lower = name.lower()
    replacements = (
        ("left_", "right_"),
        ("l_", "r_"),
        ("_left_", "_right_"),
        ("_l_", "_r_"),
    )
    for old, new in replacements:
        if old in lower:
            start = lower.index(old)
            return name[:start] + new + name[start + len(old) :]
    if lower.startswith("left"):
        return "right" + name[4:]
    return None


def build_symmetric_dofs_idx_draft(control_joints: list[str]) -> tuple[dict[str, Any], list[str]]:
    index_by_name = {name: idx for idx, name in enumerate(control_joints)}
    pairs: list[list[str]] = []
    index_pairs: list[list[int]] = []
    unmatched: list[str] = []
    for joint_name in control_joints:
        if not _is_left(joint_name):
            continue
        counterpart = _right_counterpart_name(joint_name)
        if counterpart is None or counterpart not in index_by_name:
            unmatched.append(joint_name)
            continue
        pairs.append([joint_name, counterpart])
        index_pairs.append([index_by_name[joint_name], index_by_name[counterpart]])
    warnings: list[str] = []
    if not pairs:
        warnings.append("Could not infer any left/right symmetric DOF pairs from XML control_joint_names.")
    if unmatched:
        warnings.append(f"Could not infer right-side symmetric counterparts for joints: {unmatched}")
    return {"pairs": pairs, "index_pairs": index_pairs, "unmatched_left_joints": unmatched}, warnings


def merge_xml_urdf_training_hints(
    *,
    xml_model: mujoco.MjModel,
    xml_control_joints: list[str],
    urdf_info: UrdfRobotInfo,
    joint_name_map: dict[str, str] | None = None,
    prefer_urdf_limits: bool = False,
    prefer_urdf_dynamics: bool = True,
    effort_limit_default: float = 80.0,
    velocity_limit_default: float = 20.0,
    armature_default: float = 0.01,
    friction_default: float = 0.0,
    emit_symmetry_draft: bool = True,
    range_warning_threshold: float = 1.0e-4,
) -> dict[str, Any]:
    """Merge optional URDF hints without changing XML-controlled layout."""

    joint_ids = _joint_id_map(xml_model)
    joint_ranges: dict[str, list[float]] = {}
    lower: list[float] = []
    upper: list[float] = []
    effort_limits: list[float] = []
    velocity_limits: list[float] = []
    actuator_joints: dict[str, dict[str, float]] = {}
    warnings: list[str] = [
        *urdf_info.warnings,
        "MuJoCo XML remains the source of truth for qpos/qvel/action layout and actuator order; URDF is used only for auxiliary draft hints.",
    ]
    matched_urdf_joints: list[str] = []

    for joint_name in xml_control_joints:
        joint_id = joint_ids[joint_name]
        urdf_joint_name, urdf_joint = _urdf_joint_for_xml_joint(urdf_info, joint_name, joint_name_map)
        if urdf_joint is None:
            warnings.append(f"No URDF joint match for XML control joint {joint_name!r}; XML/default hints are used.")
        else:
            matched_urdf_joints.append(str(urdf_joint_name))

        xml_limited = bool(xml_model.jnt_limited[joint_id])
        xml_lo = float(xml_model.jnt_range[joint_id, 0]) if xml_limited else -3.141592653589793
        xml_hi = float(xml_model.jnt_range[joint_id, 1]) if xml_limited else 3.141592653589793
        urdf_has_range = urdf_joint is not None and urdf_joint.limit_lower is not None and urdf_joint.limit_upper is not None
        if prefer_urdf_limits and urdf_has_range:
            urdf_lo = float(urdf_joint.limit_lower)
            urdf_hi = float(urdf_joint.limit_upper)
            if xml_limited and (abs(xml_lo - urdf_lo) > range_warning_threshold or abs(xml_hi - urdf_hi) > range_warning_threshold):
                warnings.append(
                    f"URDF limits for {joint_name!r} differ from XML limits: XML=({xml_lo}, {xml_hi}), "
                    f"URDF=({urdf_lo}, {urdf_hi}); using URDF because prefer_urdf_limits=True."
                )
            lo, hi = urdf_lo, urdf_hi
        elif xml_limited:
            lo, hi = xml_lo, xml_hi
            if urdf_has_range and (
                abs(xml_lo - float(urdf_joint.limit_lower)) > range_warning_threshold
                or abs(xml_hi - float(urdf_joint.limit_upper)) > range_warning_threshold
            ):
                warnings.append(
                    f"URDF limits for {joint_name!r} differ from XML limits; keeping XML limits because MuJoCo XML is the default source of truth."
                )
        elif urdf_has_range:
            lo, hi = float(urdf_joint.limit_lower), float(urdf_joint.limit_upper)
            warnings.append(f"Joint {joint_name!r} has no XML range; using URDF limits from {urdf_joint_name!r}.")
        else:
            lo, hi = xml_lo, xml_hi
            warnings.append(f"Joint {joint_name!r} has no XML or URDF range; using [-pi, pi] draft limits.")
        joint_ranges[joint_name] = [lo, hi]
        lower.append(lo)
        upper.append(hi)

        xml_effort, xml_effort_source = _xml_effort_limit_for_joint(xml_model, joint_name)
        if urdf_joint is not None and _positive(urdf_joint.limit_effort):
            effort = float(urdf_joint.limit_effort)
            warnings.append(f"Joint {joint_name!r} effort_limit from URDF joint {urdf_joint_name!r}.")
        elif xml_effort is not None and xml_effort > 0.0:
            effort = float(xml_effort)
            if xml_effort_source == "XML actuator ctrlrange":
                warnings.append(f"Joint {joint_name!r} actuator has no force range; using ctrlrange as draft effort_limit.")
        else:
            effort = float(effort_limit_default)
            warnings.append(f"Joint {joint_name!r} has no URDF/XML effort limit; using effort_limit_default={effort_limit_default}.")
        effort_limits.append(effort)

        if urdf_joint is not None and _positive(urdf_joint.limit_velocity):
            velocity = float(urdf_joint.limit_velocity)
            warnings.append(f"Joint {joint_name!r} velocity_limit from URDF joint {urdf_joint_name!r}.")
        else:
            velocity = float(velocity_limit_default)
            warnings.append(f"Joint {joint_name!r} has no URDF velocity limit; using velocity_limit_default={velocity_limit_default}.")
        velocity_limits.append(velocity)

        armature = _xml_dof_value(xml_model, joint_name, xml_model.dof_armature, armature_default)
        if armature == 0.0:
            armature = float(armature_default)
            warnings.append(f"Joint {joint_name!r} has zero XML armature; using armature_default={armature_default}.")
        xml_friction = _xml_dof_value(xml_model, joint_name, xml_model.dof_frictionloss, friction_default)
        xml_damping = _xml_dof_value(xml_model, joint_name, xml_model.dof_damping, 0.0)
        if prefer_urdf_dynamics and urdf_joint is not None and urdf_joint.dynamics_friction is not None:
            friction = float(urdf_joint.dynamics_friction)
            warnings.append(f"Joint {joint_name!r} friction from URDF dynamics.")
        else:
            friction = float(xml_friction)
            if friction == 0.0 and friction_default != 0.0:
                friction = float(friction_default)
                warnings.append(f"Joint {joint_name!r} has zero XML frictionloss; using friction_default={friction_default}.")
        if prefer_urdf_dynamics and urdf_joint is not None and urdf_joint.dynamics_damping is not None:
            physical_damping = float(urdf_joint.dynamics_damping)
            warnings.append(f"Joint {joint_name!r} physical damping from URDF dynamics.")
        else:
            physical_damping = float(xml_damping)

        actuator_joints[joint_name] = {
            "effort_limit": effort,
            "velocity_limit": velocity,
            "armature": float(armature),
            "friction": float(friction),
            "damping": float(physical_damping),
        }

    symmetric_draft: dict[str, Any] | None = None
    if emit_symmetry_draft:
        symmetric_draft, symmetry_warnings = build_symmetric_dofs_idx_draft(xml_control_joints)
        warnings.extend(symmetry_warnings)
        warnings.append("symmetric_dofs_idx is emitted as draft metadata only; current training/inference does not consume it.")

    return {
        "joint_ranges": joint_ranges,
        "lower_upper": [lower, upper],
        "effort_limits": effort_limits,
        "velocity_limits": velocity_limits,
        "actuator_joints": actuator_joints,
        "warnings": warnings,
        "metadata": {
            "urdf_source": urdf_info.source_path,
            "matched_urdf_joints": _ordered_unique(matched_urdf_joints),
            "semantic_hints": _urdf_semantic_hints_for_joints(urdf_info, matched_urdf_joints),
            "merge_policy": {
                "source_of_truth": "mujoco_xml",
                "urdf_used_for": [
                    "effort_limit",
                    "velocity_limit",
                    "damping",
                    "friction",
                    "semantic_hints",
                    "symmetric_dofs_idx_draft",
                ],
                "prefer_urdf_limits": bool(prefer_urdf_limits),
                "prefer_urdf_dynamics": bool(prefer_urdf_dynamics),
            },
            "symmetric_dofs_idx_draft": symmetric_draft,
        },
    }


def _keyframe_qpos(model: mujoco.MjModel, keyframe_names: tuple[str, ...]) -> tuple[list[float], str | None]:
    requested = [name for name in keyframe_names if name]
    for preferred in requested:
        for key_id in range(model.nkey):
            name = _mj_name(model, mujoco.mjtObj.mjOBJ_KEY, key_id)
            if name == preferred:
                return [float(v) for v in model.key_qpos[key_id].tolist()], name
    for key_id in range(model.nkey):
        name = _mj_name(model, mujoco.mjtObj.mjOBJ_KEY, key_id)
        if name in {"stand", "default", "home", "init"}:
            return [float(v) for v in model.key_qpos[key_id].tolist()], name
    return [float(v) for v in model.qpos0.tolist()], None


def _freejoint_pose_from_qpos(model: mujoco.MjModel, qpos: list[float]) -> tuple[list[float], list[float], list[str]]:
    warnings: list[str] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        qpos_addr = int(model.jnt_qposadr[joint_id])
        pos = [float(v) for v in qpos[qpos_addr : qpos_addr + 3]]
        quat_wxyz = [float(v) for v in qpos[qpos_addr + 3 : qpos_addr + 7]]
        quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        return pos, quat_xyzw, warnings
    warnings.append("No freejoint qpos found; using default init_state pos/rot.")
    return [0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], warnings


def infer_default_pose_from_xml(
    model: mujoco.MjModel,
    control_joints: list[str],
    keyframe_names: tuple[str, ...] = ("stand", "default", "home", "init"),
) -> tuple[dict[str, float], dict[str, list[float]], list[str]]:
    qpos, keyframe_name = _keyframe_qpos(model, keyframe_names)
    warnings: list[str] = []
    if keyframe_name is None:
        warnings.append("No stand/default/home/init keyframe found; using model.qpos0 for default pose.")
    else:
        warnings.append(f"Default pose inferred from MuJoCo keyframe {keyframe_name!r}; review before training.")
    joint_ids = _joint_id_map(model)
    default_angles: dict[str, float] = {}
    for joint_name in control_joints:
        joint_id = joint_ids[joint_name]
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}:
            warnings.append(f"Control joint {joint_name!r} is not hinge/slide; using first qpos element as draft default.")
        qpos_addr = int(model.jnt_qposadr[joint_id])
        default_angles[joint_name] = float(qpos[qpos_addr]) if qpos_addr < len(qpos) else 0.0
    pos, rot, pose_warnings = _freejoint_pose_from_qpos(model, qpos)
    warnings.extend(pose_warnings)
    init_state = {"pos": pos, "rot": rot, "lin_vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]}
    return default_angles, init_state, warnings


def _pd_gains(control_joints: list[str], pd_template: str) -> tuple[dict[str, float], dict[str, float], list[str]]:
    warnings = [f"PD gains generated from {pd_template!r} draft template; tune stiffness/damping before training."]
    stiffness: dict[str, float] = {}
    damping: dict[str, float] = {}
    for joint in control_joints:
        lower = joint.lower()
        if "waist" in lower or "torso" in lower:
            kp, kd = 120.0, 4.0
        elif "hip" in lower or "knee" in lower:
            kp, kd = 80.0, 3.0
        elif "ankle" in lower:
            kp, kd = 30.0, 1.5
        elif "shoulder" in lower or "elbow" in lower:
            kp, kd = 20.0, 1.0
        elif "wrist" in lower or "hand" in lower:
            kp, kd = 10.0, 0.5
        else:
            kp, kd = 40.0, 2.0
        stiffness[joint] = kp
        damping[joint] = kd
    return stiffness, damping, warnings


def build_robot_training_yaml_draft(
    *,
    model: mujoco.MjModel,
    name: str,
    xml_path: str | Path,
    semantics: dict[str, Any],
    default_keyframe: str = "stand",
    actuator_source: str = "yaml",
    action_scale: float = 0.25,
    action_clip_value: float = 5.0,
    normalize_action_to: float = 5.0,
    effort_limit_default: float = 80.0,
    velocity_limit_default: float = 20.0,
    armature_default: float = 0.01,
    friction_default: float = 0.0,
    pd_template: str = "humanoid",
    review_status: str = "draft",
    urdf_info: UrdfRobotInfo | None = None,
    urdf_joint_name_map: dict[str, str] | None = None,
    prefer_urdf_limits: bool = False,
    prefer_urdf_dynamics: bool = True,
    emit_symmetry_draft: bool = True,
) -> dict[str, Any]:
    warnings: list[str] = []
    control_joints, control_warnings = infer_control_joints_from_xml(model)
    warnings.extend(control_warnings)
    base_body = semantics["base_body"]
    warnings.extend(semantics["warnings"])
    key_bodies = semantics["key_bodies"]
    urdf_metadata: dict[str, Any] = {}
    if urdf_info is not None:
        merged_hints = merge_xml_urdf_training_hints(
            xml_model=model,
            xml_control_joints=control_joints,
            urdf_info=urdf_info,
            joint_name_map=urdf_joint_name_map,
            prefer_urdf_limits=prefer_urdf_limits,
            prefer_urdf_dynamics=prefer_urdf_dynamics,
            effort_limit_default=effort_limit_default,
            velocity_limit_default=velocity_limit_default,
            armature_default=armature_default,
            friction_default=friction_default,
            emit_symmetry_draft=emit_symmetry_draft,
        )
        joint_ranges = dict(merged_hints["joint_ranges"])
        effort_limits = list(merged_hints["effort_limits"])
        velocity_limits = list(merged_hints["velocity_limits"])
        actuator_joints = dict(merged_hints["actuator_joints"])
        warnings.extend(list(merged_hints["warnings"]))
        urdf_metadata = dict(merged_hints["metadata"])
        urdf_metadata["merged_joint_ranges"] = joint_ranges
    else:
        joint_ranges, range_warnings, _lower_upper = infer_joint_ranges_from_xml(model, control_joints)
        warnings.extend(range_warnings)
        effort_limits, velocity_limits, actuator_joints, actuator_warnings = infer_actuator_limits_from_xml(
            model,
            control_joints,
            effort_limit_default=effort_limit_default,
            velocity_limit_default=velocity_limit_default,
            armature_default=armature_default,
            friction_default=friction_default,
        )
        warnings.extend(actuator_warnings)
    default_angles, init_state, pose_warnings = infer_default_pose_from_xml(
        model,
        control_joints,
        keyframe_names=(default_keyframe, "stand", "default", "home", "init"),
    )
    warnings.extend(pose_warnings)
    stiffness, damping, pd_warnings = _pd_gains(control_joints, pd_template)
    warnings.extend(pd_warnings)
    if actuator_source != "yaml":
        warnings.append(f"actuator.source={actuator_source!r} was requested; non-yaml sources are not recommended for new robots.")
    if len(semantics.get("contact_bodies") or []) < 2:
        warnings.append(
            "contact_bodies has fewer than two entries; smoke may run but biped auxiliary rewards can require manual contact body setup."
        )
    if urdf_info is None:
        warnings.append("symmetric_dofs_idx was not generated because reliable symmetry inference is robot-specific.")
    warnings.append(
        "This config is XML-derived draft output. Review semantic fields, PD gains, default pose, actuator parameters, contact bodies, and reward/termination-related fields before formal training."
    )
    metadata = {
        "generated_from_xml": str(xml_path),
        "generated_by": "ufo.tools.robot_inspect",
        "review_status": review_status,
        "warnings": _ordered_unique(warnings),
    }
    if urdf_metadata:
        metadata.update(urdf_metadata)
    return {
        "name": name,
        "xml_path": str(xml_path),
        "base_body": base_body,
        "root_quat_order": "xyzw",
        "coordinate_system": "z_up",
        "dof_unit": "rad",
        "control_joints": {"mode": "explicit", "names": control_joints},
        "feet": list(semantics.get("feet") or []),
        "hands": list(semantics.get("hands") or []),
        "key_bodies": key_bodies,
        "default_dof_pos": {},
        "training": {
            "task_overrides": {},
            "joint_ranges": joint_ranges,
            "semantics": {
                "contact_bodies": list(semantics.get("contact_bodies") or []),
                "undesired_contact_bodies": list(semantics.get("undesired_contact_bodies") or []),
                "torso_name": str(semantics.get("torso_name") or base_body),
                "left_ankle_dof_names": list(semantics.get("left_ankle_dof_names") or []),
                "right_ankle_dof_names": list(semantics.get("right_ankle_dof_names") or []),
            },
            "init_state": {
                **init_state,
                "default_joint_angles": default_angles,
            },
            "control": {
                "action_scale": float(action_scale),
                "action_clip_value": float(action_clip_value),
                "normalize_action_to": float(normalize_action_to),
                "effort_limit_scale": 1.0,
                "effort_limit": effort_limits,
                "velocity_limit": velocity_limits,
                "stiffness": stiffness,
                "damping": damping,
            },
            "actuator": {
                "source": actuator_source,
                "joints": actuator_joints,
            },
        },
        "metadata": metadata,
    }


class _IndentedSafeDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False):  # noqa: ANN201 - PyYAML hook signature.
        return super().increase_indent(flow, False)



def _joint_names(model: mujoco.MjModel) -> list[str]:
    return [_mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(model.njnt)]


def _freejoint_body(model: mujoco.MjModel) -> tuple[str | None, str | None]:
    for idx in range(model.njnt):
        if int(model.jnt_type[idx]) == int(mujoco.mjtJoint.mjJNT_FREE):
            joint_name = _mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
            body_name = _mj_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[idx]))
            return joint_name, body_name
    return None, None


def _body_parent_map(model: mujoco.MjModel) -> dict[str, str | None]:
    names = [_mj_name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(model.nbody)]
    parents: dict[str, str | None] = {}
    for idx in range(1, model.nbody):
        parent_id = int(model.body_parentid[idx])
        parents[names[idx]] = None if parent_id == 0 else names[parent_id]
    return parents


def _body_depths(model: mujoco.MjModel) -> dict[str, int]:
    names = [_mj_name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(model.nbody)]
    depths: dict[str, int] = {}
    for idx in range(1, model.nbody):
        depth = 0
        parent_id = int(model.body_parentid[idx])
        while parent_id != 0:
            depth += 1
            parent_id = int(model.body_parentid[parent_id])
        depths[names[idx]] = depth
    return depths


def _leaf_bodies(model: mujoco.MjModel) -> set[str]:
    names = [_mj_name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(model.nbody)]
    parents = {int(model.body_parentid[idx]) for idx in range(1, model.nbody)}
    return {names[idx] for idx in range(1, model.nbody) if idx not in parents}


def _in_xml_order(values: list[str], xml_order: list[str]) -> list[str]:
    seen = set()
    allowed = set(values)
    result: list[str] = []
    for name in xml_order:
        if name in allowed and name not in seen:
            result.append(name)
            seen.add(name)
    return result


def _side_matches(lower_name: str, side_tokens: tuple[str, ...]) -> bool:
    if side_tokens == ("",):
        return True
    return any(token in lower_name for token in side_tokens)


def _select_side_candidate(
    body_names: list[str],
    depths: dict[str, int],
    leaves: set[str],
    *,
    side_tokens: tuple[str, ...],
    keywords: tuple[str, ...],
    preferred: tuple[str, ...],
) -> str | None:
    best: tuple[float, int, str] | None = None
    for idx, name in enumerate(body_names):
        lower = name.lower()
        if not _side_matches(lower, side_tokens):
            continue
        if not any(keyword in lower for keyword in keywords):
            continue
        keyword_score = max((len(keyword) for keyword in keywords if keyword in lower), default=0)
        preferred_score = 5 if any(token in lower for token in preferred) else 0
        leaf_score = 3 if name in leaves else 0
        score = preferred_score + keyword_score + leaf_score + depths.get(name, 0) * 0.1
        candidate = (score, -idx, name)
        if best is None or candidate > best:
            best = candidate
    return best[2] if best is not None else None


def _guess_pair(body_names: list[str], depths: dict[str, int], leaves: set[str], *, kind: str) -> list[str]:
    if kind == "feet":
        keywords = ("foot", "toe", "ankle")
        preferred = ("foot", "toe", "ankle_roll")
    elif kind == "hands":
        keywords = ("hand", "wrist", "gripper", "palm")
        preferred = ("hand", "palm", "gripper", "wrist_yaw")
    else:
        raise ValueError(f"Unsupported semantic pair kind={kind}")
    left = _select_side_candidate(body_names, depths, leaves, side_tokens=("left", "l_"), keywords=keywords, preferred=preferred)
    right = _select_side_candidate(body_names, depths, leaves, side_tokens=("right", "r_"), keywords=keywords, preferred=preferred)
    guessed = [name for name in (left, right) if name is not None]
    if guessed:
        return _in_xml_order(guessed, body_names)
    fallback = _select_side_candidate(body_names, depths, leaves, side_tokens=("",), keywords=keywords, preferred=preferred)
    return [fallback] if fallback is not None else []


def _parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false value, got {value!r}")


def _load_urdf_joint_name_map(path: str | Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    map_path = Path(path).expanduser()
    if not map_path.exists():
        raise FileNotFoundError(f"URDF joint name map does not exist: {map_path}")
    if map_path.suffix.lower() == ".json":
        data = json.loads(map_path.read_text())
    else:
        data = yaml.safe_load(map_path.read_text())
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("--urdf-joint-name-map must contain a mapping of XML joint name to URDF joint name")
    return {str(key): str(value) for key, value in data.items()}


def _save_yaml(path: Path, config: dict[str, Any], *, force: bool, header: str | None = None) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Output file already exists: {path}. Use --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(config, Dumper=_IndentedSafeDumper, sort_keys=False, default_flow_style=False)
    if header:
        text = header.rstrip() + "\n" + text
    path.write_text(text)


def inspect_mujoco_xml(xml_path: str | Path, *, name: str | None = None) -> dict[str, Any]:
    path = Path(xml_path).expanduser()
    model = mujoco.MjModel.from_xml_path(str(path))
    body_names = infer_body_names_from_xml(model)
    base_body, free_joint, warnings = infer_freejoint_base_body(model)
    return {
        "name": name or path.stem,
        "xml_path": str(xml_path),
        "model": model,
        "body_names": body_names,
        "joint_names": _joint_names(model),
        "actuated_joint_names": _read_actuator_joints(model)[0],
        "free_joint": free_joint,
        "free_body": base_body if free_joint is not None else None,
        "base_body": base_body,
        "warnings": warnings,
        "body_parent": _body_parent_map(model),
        "body_depths": _body_depths(model),
        "leaf_bodies": _leaf_bodies(model),
    }


def infer_robot_semantics(
    inspection: dict[str, Any],
    *,
    base_body: str | None = None,
    feet: list[str] | None = None,
    hands: list[str] | None = None,
    key_bodies: list[str] | None = None,
) -> dict[str, Any]:
    body_names = list(inspection["body_names"])
    depths = dict(inspection["body_depths"])
    leaves = set(inspection["leaf_bodies"])
    inferred_base = base_body or inspection["base_body"]
    inferred_feet = feet if feet is not None else _guess_pair(body_names, depths, leaves, kind="feet")
    inferred_hands = hands if hands is not None else _guess_pair(body_names, depths, leaves, kind="hands")
    warnings = list(inspection["warnings"])
    if len(inferred_feet) < 2:
        warnings.append("Fewer than two contact/feet bodies were inferred; biped rewards/terminations may need manual configuration.")
    if not inferred_hands:
        warnings.append("No hand bodies were inferred from names.")
    for warning in warnings:
        logger.warning(warning)
    torso_candidates = [name for name in body_names if _contains_any(name, ("torso", "trunk", "chest"))]
    if not torso_candidates:
        torso_candidates = [name for name in body_names if "waist" in name.lower()]
    torso_name = torso_candidates[0] if torso_candidates else inferred_base
    joint_names = inspection["actuated_joint_names"]
    if key_bodies is not None:
        inferred_key_bodies = key_bodies
    else:
        torso_like = [name for name in body_names if any(token in name.lower() for token in ("torso", "chest", "trunk", "waist"))]
        inferred_key_bodies = _in_xml_order([inferred_base, *torso_like, *inferred_feet, *inferred_hands], body_names)
    return {
        "base_body": inferred_base, "feet": inferred_feet, "hands": inferred_hands, "key_bodies": inferred_key_bodies,
        "contact_bodies": list(inferred_feet),
        "undesired_contact_bodies": [name for name in body_names if _contains_any(name, ("pelvis", "hip", "shoulder", "torso", "trunk", "chest"))],
        "torso_name": torso_name,
        "left_ankle_dof_names": _ordered_unique([name for name in joint_names if _is_left(name) and "ankle" in name.lower()]),
        "right_ankle_dof_names": _ordered_unique([name for name in joint_names if _is_right(name) and "ankle" in name.lower()]),
        "warnings": warnings,
    }


def write_robot_yaml(
    out_path: str | Path,
    inspection: dict[str, Any],
    semantics: dict[str, Any],
    *,
    root_quat_order: str = "xyzw",
    coordinate_system: str = "z_up",
    dof_unit: str = "rad",
    force: bool = False,
) -> Path:
    path = Path(out_path).expanduser()
    if path.exists() and not force:
        raise FileExistsError(f"Robot YAML already exists: {path}. Use --force to overwrite it.")
    config = {
        "name": inspection["name"],
        "xml_path": inspection["xml_path"],
        "base_body": semantics["base_body"],
        "root_quat_order": root_quat_order,
        "coordinate_system": coordinate_system,
        "dof_unit": dof_unit,
        "control_joints": {"mode": "all_actuated"},
        "feet": list(semantics["feet"]),
        "hands": list(semantics["hands"]),
        "key_bodies": list(semantics["key_bodies"]),
        "default_dof_pos": {},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), path)
    load_robot_spec(path)
    return path


def write_robot_training_yaml(
    out_path: str | Path,
    inspection: dict[str, Any],
    semantics: dict[str, Any],
    *,
    args: argparse.Namespace,
    urdf_info: UrdfRobotInfo | None = None,
    urdf_joint_name_map: dict[str, str] | None = None,
) -> Path:
    path = Path(out_path).expanduser()
    config = build_robot_training_yaml_draft(
        model=inspection["model"],
        name=str(inspection["name"]),
        xml_path=str(inspection["xml_path"]),
        semantics=semantics,
        default_keyframe=args.default_keyframe,
        actuator_source=args.actuator_source,
        action_scale=args.action_scale,
        action_clip_value=args.action_clip_value,
        normalize_action_to=args.normalize_action_to,
        effort_limit_default=args.effort_limit_default,
        velocity_limit_default=args.velocity_limit_default,
        armature_default=args.armature_default,
        friction_default=args.friction_default,
        pd_template=args.pd_template,
        review_status=args.review_status,
        urdf_info=urdf_info,
        urdf_joint_name_map=urdf_joint_name_map,
        prefer_urdf_limits=bool(args.prefer_urdf_limits),
        prefer_urdf_dynamics=bool(args.prefer_urdf_dynamics),
        emit_symmetry_draft=bool(args.emit_symmetry_draft),
    )
    _save_yaml(path, config, force=bool(args.force))
    load_robot_training_spec(path)
    return path


def _print_summary(inspection: dict[str, Any], semantics: dict[str, Any]) -> None:
    model = inspection["model"]
    print("Robot XML summary")
    print(f"  name: {inspection['name']}")
    print(f"  nq/nv/nu: {model.nq}/{model.nv}/{model.nu}")
    print(f"  bodies: {len(inspection['body_names'])}")
    print(f"  joints: {len(inspection['joint_names'])}")
    print(f"  actuators: {model.nu}")
    print(f"  freejoint: {inspection.get('free_joint')} on body {inspection.get('free_body')}")
    print(f"  base_body: {semantics['base_body']}")
    print(f"  actuated_joints: {inspection['actuated_joint_names']}")
    print(f"  guessed feet: {semantics['feet']}")
    print(f"  guessed hands: {semantics['hands']}")
    print(f"  guessed key_bodies: {semantics['key_bodies']}")
    print("  Please inspect feet/hands/key_bodies before large-scale training.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True, help="MuJoCo XML path")
    parser.add_argument("--name", required=True, help="Robot name for the generated YAML")
    parser.add_argument("--out", required=True, help="Output robot YAML path")
    parser.add_argument("--base-body", default=None, help="Override base_body")
    parser.add_argument("--feet", default=None, help="Comma-separated feet body override")
    parser.add_argument("--hands", default=None, help="Comma-separated hands body override")
    parser.add_argument("--key-bodies", default=None, help="Comma-separated key_bodies override")
    parser.add_argument("--root-quat-order", default="xyzw", choices=["xyzw", "wxyz"])
    parser.add_argument("--coordinate-system", default="z_up")
    parser.add_argument("--dof-unit", default="rad")
    parser.add_argument("--with-training", action="store_true", help="Also write a draft training section into the Robot YAML.")
    parser.add_argument(
        "--urdf", default=None, help="Optional URDF used only to enrich draft hardware limits, dynamics, and semantic hints."
    )
    parser.add_argument("--urdf-joint-name-map", default=None, help="Optional YAML/JSON mapping from XML joint names to URDF joint names.")
    parser.add_argument(
        "--prefer-urdf-limits",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
        help="Allow URDF joint lower/upper limits to override XML limits with warnings. Default: false.",
    )
    parser.add_argument(
        "--prefer-urdf-dynamics",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help="Prefer URDF dynamics damping/friction when available. Default: true.",
    )
    parser.add_argument(
        "--emit-symmetry-draft",
        nargs="?",
        const=True,
        default=True,
        type=_parse_bool,
        help="Emit left/right symmetric DOF pairs as draft metadata. Default: true.",
    )
    parser.add_argument("--default-keyframe", default="stand", help="Preferred MuJoCo keyframe name for default pose inference.")
    parser.add_argument("--actuator-source", default="yaml", choices=["yaml", "g1_mode15", "g1-mode_15"])
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--action-clip-value", type=float, default=5.0)
    parser.add_argument("--normalize-action-to", type=float, default=5.0)
    parser.add_argument("--effort-limit-default", type=float, default=80.0)
    parser.add_argument("--velocity-limit-default", type=float, default=20.0)
    parser.add_argument("--armature-default", type=float, default=0.01)
    parser.add_argument("--friction-default", type=float, default=0.0)
    parser.add_argument("--pd-template", default="humanoid", choices=["humanoid"])
    parser.add_argument("--review-status", default="draft")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args(argv)
    if args.actuator_source != "yaml":
        logger.warning("Non-yaml actuator sources are legacy/G1-specific. Use --actuator-source yaml for new robots.")

    urdf_info = parse_urdf_robot(args.urdf) if args.urdf is not None else None
    urdf_joint_name_map = _load_urdf_joint_name_map(args.urdf_joint_name_map)
    if args.urdf is not None and not args.with_training:
        logger.warning("--urdf was provided without --with-training; XML-only Robot YAML output is unchanged.")

    inspection = inspect_mujoco_xml(args.xml, name=args.name)
    semantics = infer_robot_semantics(
        inspection,
        base_body=args.base_body,
        feet=_parse_csv_list(args.feet),
        hands=_parse_csv_list(args.hands),
        key_bodies=_parse_csv_list(args.key_bodies),
    )
    if args.with_training:
        out_path = write_robot_training_yaml(
            args.out,
            inspection,
            semantics,
            args=args,
            urdf_info=urdf_info,
            urdf_joint_name_map=urdf_joint_name_map,
        )
    else:
        out_path = write_robot_yaml(
            args.out,
            inspection,
            semantics,
            root_quat_order=args.root_quat_order,
            coordinate_system=args.coordinate_system,
            dof_unit=args.dof_unit,
            force=args.force,
        )
    _print_summary(inspection, semantics)
    print(f"Wrote robot YAML: {out_path}")
    if args.with_training:
        robot_config = OmegaConf.to_container(OmegaConf.load(out_path), resolve=True)
        warnings = list((robot_config.get("metadata") or {}).get("warnings") or [])
        print(
            "Robot training draft generated. Review semantics, default pose, PD gains, actuator parameters, contact bodies, and reward/termination-related fields before formal training."
        )
        for warning in warnings:
            print(f"  warning: {warning}")


if __name__ == "__main__":
    main()
