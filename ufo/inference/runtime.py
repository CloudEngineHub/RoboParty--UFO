"""Shared configuration, initialization and rendering for inference entrypoints."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mediapy as media
import mujoco
import numpy as np
import torch

import ufo
from ufo.envs.config import MJLabEnvConfig
from ufo.motion.manifest import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from ufo.tools.robot_config import (
    BUILTIN_ROBOTS,
    RobotTrainingSpec,
    builtin_robot_config,
    load_robot_training_spec,
    select_robot_config,
)

if getattr(ufo, "__file__", None) is not None:
    UFO_DIR = Path(ufo.__file__).parent
else:
    UFO_DIR = Path(__file__).resolve().parent


PROJECT_ROOT = UFO_DIR.parent


@contextmanager
def video_writer(path: Path | None, *, shape: tuple[int, int], fps: int):
    """Stream frames; publish the video only after the rollout and encoder succeed."""
    if path is None:
        yield None
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}_", suffix=".mp4", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        with media.VideoWriter(str(temporary), shape=shape, fps=fps) as writer:
            yield writer
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    parser.add_argument(
        name,
        nargs="?",
        const=True,
        default=default,
        type=str2bool,
        help=help_text,
    )


def add_robot_config_manifest_args(parser: argparse.ArgumentParser, *, purpose: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--robot", choices=tuple(BUILTIN_ROBOTS), help="Validate the checkpoint against this robot.")
    group.add_argument("--robot-config", type=Path, default=None, help="Matching robot YAML; otherwise read from the saved run.")
    parser.add_argument("--data-manifest", type=Path, default=None, help=f"Motion data manifest. Use with --dataset for {purpose}.")
    parser.add_argument("--dataset", default=None, help="Dataset name inside --data-manifest.")
    parser.add_argument("--rebuild-motion-cache", action="store_true", help="Rebuild manifest-generated motion pkl cache.")


def resolve_inference_data_and_robot_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    manifest_robot_config = None
    if args.data_manifest is not None:
        if args.data_path is not None:
            parser.error("--data-manifest and --data-path cannot be used together")
        if args.dataset is None:
            parser.error("--dataset is required when --data-manifest is provided")
        manifest_robot_config = prepare_manifest_robot_config_path(args.data_manifest)
        args.data_path = Path(
            prepare_manifest_dataset_path(
                args.data_manifest,
                args.dataset,
                split="inference",
                rebuild_cache=bool(args.rebuild_motion_cache),
            )
        )
    if args.robot is not None:
        args.robot_config = builtin_robot_config(args.robot)
    if args.robot_config is not None or manifest_robot_config is not None:
        args.robot_config = select_robot_config(args.robot_config, manifest_robot_config)
    return args


def load_mjlab_env_cfg(
    model_folder: Path,
    *,
    data_path: Path | None,
    robot_config: Path | None = None,
    device: str,
    disable_dr: bool,
    disable_obs_noise: bool,
    max_episode_length_s: float,
) -> tuple[MJLabEnvConfig, RobotTrainingSpec]:
    with (model_folder / "config.json").open("r") as f:
        config = json.load(f)

    env_config = MJLabEnvConfig.model_validate(config["env"]).model_dump()
    env_config["device"] = device
    if robot_config is None:
        robot_config = env_config["robot_config_path"]
        if not robot_config:
            raise ValueError("Saved run has no robot configuration; supply --robot-config.")
    robot_config = select_robot_config(robot_config)
    training_spec = load_robot_training_spec(robot_config)
    current = training_spec.to_env_dict()["robot"]
    expected = env_config.get("robot_training", {}).get("robot") if env_config.get("robot_training") else None
    if not expected:
        raise ValueError("Saved run has no robot snapshot; use its matching archived code.")
    for key in ("name", "control_joint_names", "body_names", "base_body", "key_bodies", "feet", "hands", "joint_ranges"):
        if current[key] != expected[key]:
            raise ValueError(f"Robot does not match checkpoint: {key}")
    # Use saved control/observation settings; the selected YAML only locates matching assets.
    env_config["robot_config_path"] = str(robot_config)
    env_config["mjcf_path"] = training_spec.robot.xml_path
    env_config["robot_training"]["robot"]["xml_path"] = training_spec.robot.xml_path
    env_config["disable_domain_randomization"] = disable_dr
    env_config["disable_obs_noise"] = disable_obs_noise
    env_config["auto_reset"] = False
    env_config["max_episode_length_s"] = max_episode_length_s

    if data_path is not None:
        env_config["motion_data_path"] = str(data_path.expanduser().resolve())
        env_config["data_mix_weights"] = None
    else:
        saved_paths = env_config["motion_data_path"]
        paths = saved_paths if isinstance(saved_paths, list) else [saved_paths]
        resolved = []
        for path in paths:
            motion_path = Path(path).expanduser()
            if not motion_path.is_absolute() and not motion_path.exists() and (PROJECT_ROOT / motion_path).exists():
                motion_path = PROJECT_ROOT / motion_path
            resolved.append(str(motion_path))
        env_config["motion_data_path"] = resolved if isinstance(saved_paths, list) else resolved[0]


    return MJLabEnvConfig(**env_config), training_spec


class MujocoQposRenderer:
    """Pure MuJoCo renderer for qpos playback from an MJCF."""

    def __init__(
        self,
        xml_path: Path,
        render_size: int = 480,
        *,
        camera_distance: float = 3.0,
        camera_azimuth: float = 135.0,
        camera_elevation: float = -18.0,
        expected_qpos_size: int | None = None,
    ):
        spec = mujoco.MjSpec.from_file(str(xml_path))
        spec.worldbody.add_geom(
            name="inference_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            pos=[0.0, 0.0, 0.0],
            size=[20.0, 20.0, 0.02],
            rgba=[0.45, 0.47, 0.50, 1.0],
            contype=0,
            conaffinity=0,
        )
        spec.worldbody.add_light(
            name="inference_key_light",
            pos=[0.0, -3.0, 4.0],
            dir=[0.2, 0.5, -1.0],
            diffuse=[0.8, 0.8, 0.8],
            ambient=[0.35, 0.35, 0.35],
            specular=[0.1, 0.1, 0.1],
        )
        self.model = spec.compile()
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), int(render_size))
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), int(render_size))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = float(camera_distance)
        self.camera.azimuth = float(camera_azimuth)
        self.camera.elevation = float(camera_elevation)
        if expected_qpos_size is not None and self.model.nq != int(expected_qpos_size):
            raise ValueError(f"Expected renderer nq={expected_qpos_size}, got nq={self.model.nq}")

    def render_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.size != self.model.nq:
            raise ValueError(f"Expected qpos size {self.model.nq}, got {qpos.size}")
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.camera.lookat[:] = [float(qpos[0]), float(qpos[1]), max(float(qpos[2]), 0.75)]
        self.renderer.update_scene(self.data, camera=self.camera)
        frame = self.renderer.render()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError("Expected MuJoCo RGB frame with shape (height, width, 3)")
        if frame.dtype != np.uint8:
            raise ValueError(f"Expected MuJoCo RGB frame dtype uint8, got {frame.dtype}")
        # Video frame lists own their pixels even if a renderer reuses its output.
        return frame.copy(order="C")

    def close(self) -> None:
        self.renderer.close()


def policy_qpos_from_env(wrapped_env: Any, *, expected_qpos_size: int) -> np.ndarray:
    qpos, _qvel = wrapped_env._get_qpos_qvel(to_numpy=True)
    qpos = np.asarray(qpos)
    if qpos.ndim == 2:
        qpos = qpos[0]
    qpos = qpos.reshape(-1)
    if qpos.size != int(expected_qpos_size):
        raise ValueError(f"Expected MJLab policy qpos size {expected_qpos_size}, got shape {qpos.shape}")
    return qpos


def render_policy_frame(wrapped_env: Any, renderer: MujocoQposRenderer) -> np.ndarray:
    return renderer.render_qpos(policy_qpos_from_env(wrapped_env, expected_qpos_size=renderer.model.nq))


def _target_states_from_obs(obs_dict: dict[str, torch.Tensor], device: str, *, num_dof: int) -> dict[str, torch.Tensor]:
    root_state_xyzw = torch.cat(
        [
            obs_dict["ref_body_pos"][0, 0],
            obs_dict["ref_body_rots"][0, 0],
            obs_dict["ref_body_vels"][0, 0],
            obs_dict["ref_body_angular_vels"][0, 0],
        ],
        dim=-1,
    ).to(device=device, dtype=torch.float32)
    dof_state = torch.zeros((int(num_dof), 2), device=device, dtype=torch.float32)
    dof_state[:, 0] = obs_dict["dof_pos"][0].to(device=device, dtype=torch.float32)
    dof_state[:, 1] = obs_dict["ref_dof_vel"][0].to(device=device, dtype=torch.float32)
    return {"root_states": root_state_xyzw.unsqueeze(0), "dof_states": dof_state.unsqueeze(0)}
