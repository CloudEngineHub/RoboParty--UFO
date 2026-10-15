"""Robot motor calibration, parameter expansion and mjlab actuator behavior."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from typing import Any

import torch
from mjlab.actuator import DcMotorActuator, DcMotorActuatorCfg


@dataclass(kw_only=True)
class RoboPartyActuatorCfg(DcMotorActuatorCfg):
    delay_hold_prob: float = 1.0

    def build(self, entity, target_ids: list[int], target_names: list[str]) -> RoboPartyActuator:
        return RoboPartyActuator(self, entity, target_ids, target_names)


class RoboPartyActuator(DcMotorActuator):
    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        if self._delay_buffer is not None:
            # Isaac DelayedPD samples once per reset; mjlab normally samples every step.
            indices = slice(None) if env_ids is None else env_ids
            lags = self._delay_buffer.current_lags[indices]
            self.set_lags(
                torch.randint(self.cfg.delay_min_lag, self.cfg.delay_max_lag + 1, lags.shape, device=lags.device),
                env_ids,
            )


G1_MJLAB_ACTUATOR_SOURCE = "g1-mode_15"


def _reflected_inertia_from_two_stage_planetary(
    rotor_inertia: tuple[float, float, float],
    gear_ratio: tuple[float, float, float],
) -> float:
    """Compute reflected motor inertia constants for Unitree G1 motors."""
    return rotor_inertia[0] * (gear_ratio[1] * gear_ratio[2]) ** 2 + rotor_inertia[1] * gear_ratio[2] ** 2 + rotor_inertia[2]


_ARMATURE_5020 = _reflected_inertia_from_two_stage_planetary((0.139e-4, 0.017e-4, 0.169e-4), (1, 1 + (46 / 18), 1 + (56 / 16)))
_ARMATURE_7520_14 = _reflected_inertia_from_two_stage_planetary((0.489e-4, 0.098e-4, 0.533e-4), (1, 4.5, 1 + (48 / 22)))
_ARMATURE_7520_22 = _reflected_inertia_from_two_stage_planetary((0.489e-4, 0.109e-4, 0.738e-4), (1, 4.5, 5))
_ARMATURE_5010 = _reflected_inertia_from_two_stage_planetary((0.084e-4, 0.015e-4, 0.068e-4), (1, 4, 4))


def _g1_mjlab_mode15_actuator_params(dof_names: tp.Sequence[str]) -> dict[str, list[float]]:
    """Return per-DOF G1 mode-15 motor params in UFO order.

    The constants are vendored here so training does not depend on an external
    asset package or download path. Kp/Kd remain UFO values; this only
    supplies motor effort, velocity reference, armature and dry friction.
    """

    efforts: list[float] = []
    velocities: list[float] = []
    armatures: list[float] = []
    frictions: list[float] = []

    for joint_name in dof_names:
        if "_hip_pitch_joint" in joint_name:
            effort, velocity, armature = 139.0, 20.0, _ARMATURE_7520_22
        elif "_hip_yaw_joint" in joint_name or joint_name == "waist_yaw_joint":
            effort, velocity, armature = 88.0, 32.0, _ARMATURE_7520_14
        elif "_hip_roll_joint" in joint_name or "_knee_joint" in joint_name:
            effort, velocity, armature = 139.0, 20.0, _ARMATURE_7520_22
        elif "_ankle_pitch_joint" in joint_name or "_ankle_roll_joint" in joint_name:
            effort, velocity, armature = 50.0, 37.0, 2.0 * _ARMATURE_5020
        elif joint_name in ("waist_pitch_joint", "waist_roll_joint"):
            effort, velocity, armature = 50.0, 37.0, 2.0 * _ARMATURE_5020
        elif (
            "_shoulder_pitch_joint" in joint_name
            or "_shoulder_roll_joint" in joint_name
            or "_shoulder_yaw_joint" in joint_name
            or "_elbow_joint" in joint_name
            or "_wrist_roll_joint" in joint_name
        ):
            effort, velocity, armature = 25.0, 37.0, _ARMATURE_5020
        elif "_wrist_pitch_joint" in joint_name or "_wrist_yaw_joint" in joint_name:
            effort, velocity, armature = 13.4, 27.0, _ARMATURE_5010
        else:
            raise ValueError(f"No G1 mode-15 actuator parameters for joint: {joint_name}")

        efforts.append(effort)
        velocities.append(velocity)
        armatures.append(armature)
        frictions.append(0.01)

    return {
        "effort_limit": efforts,
        "velocity_limit": velocities,
        "armature": armatures,
        "friction": frictions,
    }


def _actuator_params_from_training(
    dof_names: tp.Sequence[str], robot_training: dict[str, Any] | None
) -> tuple[str, dict[str, list[float]]]:
    if not robot_training:
        return G1_MJLAB_ACTUATOR_SOURCE, _g1_mjlab_mode15_actuator_params(dof_names)
    actuator = dict(robot_training.get("actuator") or {})
    source = str(actuator.get("source", G1_MJLAB_ACTUATOR_SOURCE))
    if source in {"g1_mode15", "g1-mode_15"}:
        return source, _g1_mjlab_mode15_actuator_params(dof_names)
    if source != "yaml":
        raise ValueError(f"Unsupported training.actuator.source={source!r}")
    joints = actuator.get("joints")
    if not isinstance(joints, dict):
        raise ValueError("training.actuator.source=yaml requires training.actuator.joints")
    params = {"effort_limit": [], "velocity_limit": [], "armature": [], "friction": []}
    if actuator.get("model") == "roboparty":
        params.update({"keep_velocity": [], "rated_effort": [], "rated_velocity": [], "viscous_friction": []})
    for joint_name in dof_names:
        joint_params = joints.get(joint_name)
        if not isinstance(joint_params, dict):
            raise ValueError(f"training.actuator.joints is missing parameters for joint {joint_name!r}")
        for key in params:
            if key not in joint_params:
                raise ValueError(f"training.actuator.joints.{joint_name} is missing '{key}'")
            params[key].append(float(joint_params[key]))
    return source, params
