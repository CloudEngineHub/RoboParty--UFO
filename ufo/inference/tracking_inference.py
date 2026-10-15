"""Track reference motion sequences and export their latents and videos."""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from torch.utils._pytree import tree_map

from ufo.agents.common.checkpoint import checkpoint_load_device, load_model_from_checkpoint_dir
from ufo.inference.runtime import (
    MujocoQposRenderer,
    _target_states_from_obs,
    add_bool_arg,
    add_robot_config_manifest_args,
    load_mjlab_env_cfg,
    render_policy_frame,
    resolve_inference_data_and_robot_args,
    video_writer,
)
from ufo.motion.reference import get_backward_observation
from ufo.tools.export_onnx import export_tracking_onnx


def _resize_nearest(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    if frame.shape[:2] == (height, width):
        return frame
    y_idx = np.linspace(0, frame.shape[0] - 1, height).astype(np.int64)
    x_idx = np.linspace(0, frame.shape[1] - 1, width).astype(np.int64)
    return frame[y_idx[:, None], x_idx[None, :]]


def _control_to_qpos_order_indices(robot_training: Any) -> tuple[np.ndarray, list[str]]:
    control_joint_names = list(robot_training.robot.control_joint_names)
    qpos_joint_names = sorted(control_joint_names, key=lambda joint: robot_training.robot.joint_qpos_addr[joint])
    qpos_addrs = [int(robot_training.robot.joint_qpos_addr[joint]) for joint in qpos_joint_names]
    if len(set(qpos_addrs)) != len(qpos_addrs):
        raise ValueError(f"Duplicate MuJoCo qpos addresses for control joints: {list(zip(qpos_joint_names, qpos_addrs))}")
    index_by_control_joint = {joint: idx for idx, joint in enumerate(control_joint_names)}
    return np.asarray([index_by_control_joint[joint] for joint in qpos_joint_names], dtype=np.int64), qpos_joint_names


def _expert_qpos_from_obs(
    obs_dict: dict[str, torch.Tensor],
    *,
    num_dof: int,
    dof_qpos_order_indices: np.ndarray,
) -> np.ndarray:
    root_pos = obs_dict["ref_body_pos"][:, 0].detach().cpu().numpy()
    root_quat_wxyz = np.roll(obs_dict["ref_body_rots"][:, 0].detach().cpu().numpy(), 1, axis=-1)
    # MotionLib stores dof_pos in policy/control-joint order. MuJoCo qpos playback
    # expects hinge joints sorted by qpos address, which can differ for robots such
    # as X2 where actuator order places head joints before arm joints.
    dof_pos = obs_dict["dof_pos"].detach().cpu().numpy()[:, dof_qpos_order_indices]
    qpos = np.concatenate([root_pos, root_quat_wxyz, dof_pos], axis=-1)
    expected = 7 + int(num_dof)
    if qpos.shape[-1] != expected:
        raise ValueError(f"Expected expert qpos shape (*, {expected}), got {qpos.shape}")
    return qpos


def select_motion_ids(keys: list[str], names: list[str]) -> list[int]:
    ids = []
    for name in names:
        matches = [i for i, key in enumerate(keys) if str(key) == name]
        if not matches:
            matches = [i for i, key in enumerate(keys) if Path(str(key)).stem == name]
        if len(matches) != 1:
            raise ValueError(f"Motion {name!r} has {len(matches)} matches; use --list-motions and --motion-list.")
        ids.append(matches[0])
    return ids


def run_tracking_inference(
    *,
    model_folder: Path,
    data_path: Path | None,
    robot_config: Path | None,
    device: str,
    save_mp4: bool,
    disable_dr: bool,
    disable_obs_noise: bool,
    motion_list: list[int],
    render_size: int,
    camera_distance: float,
    camera_azimuth: float,
    camera_elevation: float,
    fps: int,
    max_steps: int | None,
    log_every_steps: int,
    max_episode_length_s: float,
    export_onnx: bool,
    motion_names: list[str] | None = None,
) -> None:
    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

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
    dof_qpos_order_indices, qpos_joint_names = _control_to_qpos_order_indices(robot_training)

    model_load_device = checkpoint_load_device(device)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=model_load_device)
    model.to(device)
    model.eval()

    if export_onnx:
        export_tracking_onnx(model, model_folder / "exported", robot_training, env_config=env_cfg)

    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env

    output_dir = model_folder / "tracking_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] UFO tracking inference model_folder={model_folder}")
    print(f"[INFO] Rollout XML={env_cfg.mjcf_path}")
    print(f"[INFO] Motion data={env_cfg.motion_data_path}")
    print(f"[INFO] Expert renderer XML={robot_xml}")
    if qpos_joint_names != control_joint_names:
        print(f"[INFO] Expert qpos joint order differs from control order: {qpos_joint_names}")
    print(f"[INFO] device={device} disable_dr={disable_dr} disable_obs_noise={disable_obs_noise} save_mp4={save_mp4}")

    if motion_names:
        motion_list = select_motion_ids(list(env._motion_lib._motion_data_keys), motion_names)
    env._motion_lib.load_all_motions()
    env.is_evaluating = True
    expert_renderer = (
        MujocoQposRenderer(
            robot_xml,
            render_size=render_size,
            camera_distance=camera_distance,
            camera_azimuth=camera_azimuth,
            camera_elevation=camera_elevation,
            expected_qpos_size=7 + num_dof,
        )
        if save_mp4
        else None
    )
    try:
        for motion_id in motion_list:
            backward_obs, obs_dict = get_backward_observation(env, motion_id, use_root_height_obs=env_cfg.root_height_obs)
            z = model.tracking_inference(tree_map(lambda x: x[1:].to(device) if hasattr(x, "to") else x, backward_obs))
            joblib.dump(z.detach().cpu().numpy(), output_dir / f"zs_{motion_id}.pkl")
            print(f"[INFO] Saved z embedding: {output_dir / f'zs_{motion_id}.pkl'}")

            target_states = _target_states_from_obs(obs_dict, device=device, num_dof=num_dof)
            observation, _ = wrapped_env.reset(to_numpy=False, target_states=target_states)
            episode_len = int(z.shape[0])
            if max_steps is not None:
                episode_len = min(episode_len, int(max_steps))
            expert_qpos = _expert_qpos_from_obs(
                obs_dict,
                num_dof=num_dof,
                dof_qpos_order_indices=dof_qpos_order_indices,
            )
            video_path = output_dir / f"tracking_{motion_id}.mp4"
            frames_written = 0

            print(f"[INFO] Running policy rollout for motion_id={motion_id}, steps={episode_len}", flush=True)
            with video_writer(video_path if save_mp4 else None, shape=(render_size, 2 * render_size), fps=fps) as writer:
                for step in range(episode_len):
                    action = model.act(observation, z[step].unsqueeze(0), mean=True)
                    observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)

                    if save_mp4:
                        policy_frame = render_policy_frame(wrapped_env, expert_renderer)
                        expert_frame = expert_renderer.render_qpos(expert_qpos[min(step + 1, len(expert_qpos) - 1)])
                        expert_frame = _resize_nearest(expert_frame, policy_frame.shape[0], policy_frame.shape[1])
                        writer.add_image(np.concatenate([expert_frame, policy_frame], axis=1))
                        frames_written += 1

                    if step == 0 or (step + 1) == episode_len or (log_every_steps > 0 and (step + 1) % log_every_steps == 0):
                        print(f"[INFO] motion_id={motion_id} rollout/render progress {step + 1}/{episode_len}", flush=True)

                    if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
                        print(f"[INFO] Episode ended at step={step}; stopping rollout for motion_id={motion_id}")
                        break

                if save_mp4 and not frames_written:
                    raise RuntimeError(f"No frames were rendered for motion_id={motion_id}")
            if save_mp4:
                print(f"[INFO] Saved side-by-side video: {video_path}")
    finally:
        if expert_renderer is not None:
            expert_renderer.close()
        wrapped_env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UFO tracking inference with MuJoCo expert rendering.")
    parser.add_argument("--model-folder", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    add_robot_config_manifest_args(parser, purpose="tracking inference")

    parser.add_argument("--device", default="cuda:0")
    add_bool_arg(parser, "--save-mp4", False, "Save side-by-side expert/policy MP4.")
    add_bool_arg(parser, "--disable-dr", False, "Disable domain randomization.")
    add_bool_arg(parser, "--disable-obs-noise", False, "Disable observation noise.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--motion-list", type=int, nargs="+", default=[0])
    selection.add_argument("--motion-name", nargs="+", default=None, help="Exact motion name or unambiguous filename stem.")
    parser.add_argument("--list-motions", action="store_true", help="List dataset motion IDs/names without loading a policy or simulator.")
    parser.add_argument("--render-size", type=int, default=480)
    parser.add_argument("--camera-distance", type=float, default=3.0)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap on rollout/video frames for quick previews.")
    parser.add_argument(
        "--log-every-steps", type=int, default=100, help="Print rollout/render progress every N steps; 0 disables periodic logs."
    )
    parser.add_argument("--max-episode-length-s", type=float, default=10000.0)
    add_bool_arg(parser, "--export-onnx", True, "Export ONNX next to the checkpoint before inference.")
    return resolve_inference_data_and_robot_args(parser.parse_args(), parser)


def main() -> None:
    args = parse_args()
    if args.list_motions:
        cfg, _ = load_mjlab_env_cfg(args.model_folder, data_path=args.data_path, robot_config=args.robot_config,
                                  device="cpu", disable_dr=True, disable_obs_noise=True, max_episode_length_s=10000)
        paths = cfg.motion_data_path if isinstance(cfg.motion_data_path, list) else [cfg.motion_data_path]
        names = []
        for path in paths:
            if not Path(path).is_file():
                raise ValueError("--list-motions requires motion dictionary PKL files; pass --data-path FILE.pkl.")
            data = joblib.load(path)
            if not isinstance(data, dict):
                raise ValueError("--list-motions requires a motion dictionary PKL.")
            names.extend(data)
        for index, name in enumerate(names):
            print(f"{index}\t{name}")
        return
    run_tracking_inference(
        model_folder=args.model_folder,
        data_path=args.data_path,
        robot_config=args.robot_config,

        device=args.device,
        save_mp4=args.save_mp4,
        disable_dr=args.disable_dr,
        disable_obs_noise=args.disable_obs_noise,
        motion_list=args.motion_list,
        motion_names=args.motion_name,
        render_size=args.render_size,
        camera_distance=args.camera_distance,
        camera_azimuth=args.camera_azimuth,
        camera_elevation=args.camera_elevation,
        fps=args.fps,
        max_steps=args.max_steps,
        log_every_steps=args.log_every_steps,
        max_episode_length_s=args.max_episode_length_s,
        export_onnx=args.export_onnx,
    )


if __name__ == "__main__":
    main()
