"""Motion loading keeps source order, FK topology and raw joint trajectories."""

from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import torch

from ufo.envs.config import resolve_task_config
from ufo.motion.kinematics import HumanoidKinematics
from ufo.motion.library import MotionLib
from ufo.train import build_mjlab_config


def build_sample_library(tmp_path, mixed, raw_dof=True, extensions=True):
    cfg = build_mjlab_config(
        device="cpu", work_dir=str(tmp_path), num_envs=3, num_env_steps=16,
        seed=1, use_wandb=False, wandb_run_name=None,
        robot_config=Path("configs/robots/g1_29dof.yaml").resolve(),
    )
    motion_cfg = resolve_task_config(cfg.env, 3).robot.motion
    if not extensions:
        motion_cfg.extend_config = []
    from ufo.motion.kinematics import HumanoidKinematics

    parser = HumanoidKinematics(motion_cfg)
    clips = {}
    for index in range(3):
        frames = 5 + index
        dof = np.arange(frames, dtype=np.float32)[:, None] * np.full((1, parser.num_dof), 0.01, dtype=np.float32)
        clips[f"clip_{index}"] = {
            "root_trans_offset": np.tile(np.asarray([[index, 0, 0.8]], dtype=np.float32), (frames, 1)),
            "pose_aa": np.zeros((frames, len(parser.body_names), 3), dtype=np.float32),
            "dof_pos": dof, "fps": 50.0,
            "action": np.zeros_like(dof),
        }
    if not raw_dof:
        for clip in clips.values():
            clip["pose_aa"][:, 1:, 0] = clip["dof_pos"]
            del clip["dof_pos"]
    first = tmp_path / "first.pkl"
    joblib.dump({"clip_0": clips["clip_0"]} if mixed else clips, first)
    motion_cfg.motion_file = str(first)
    if mixed:
        second = tmp_path / "second.pkl"
        joblib.dump({k: v for k, v in clips.items() if k != "clip_0"}, second)
        motion_cfg.motion_file = [str(first), str(second)]
        motion_cfg.motion_file_weights = [0.25, 0.75]
    return MotionLib(motion_cfg, num_envs=3, device="cpu"), clips


@pytest.mark.parametrize("mixed", [False, True])
def test_motion_load_and_reload_preserve_order_and_joint_state(tmp_path, mixed):
    library, clips = build_sample_library(tmp_path, mixed)
    for _ in range(2):
        library.load_all_motions()
        assert library.curr_motion_keys == list(clips)
        assert len(library.kinematics.body_names) == 30
        assert library.length_starts.tolist() == [0, 5, 11]
        torch.testing.assert_close(library.dof_pos, torch.from_numpy(np.concatenate([v["dof_pos"] for v in clips.values()])))
        state = library.get_motion_state(torch.arange(3), torch.full((3,), 0.02))
        torch.testing.assert_close(state["root_pos"][:, 0], torch.arange(3, dtype=torch.float32))
        torch.testing.assert_close(state["dof_pos"], torch.full_like(state["dof_pos"], 0.01))
        torch.testing.assert_close(state["dof_vel"], torch.full_like(state["dof_vel"], 0.5))
        expected = [0.25, 0.375, 0.375] if mixed else [1 / 3] * 3
        torch.testing.assert_close(library._sampling_prob, torch.tensor(expected))


@pytest.mark.parametrize("extensions", [False, True])
def test_axis_angle_motion_without_raw_joints_preserves_fk_dofs(tmp_path, extensions):
    library, _ = build_sample_library(tmp_path, mixed=False, raw_dof=False, extensions=extensions)
    for _ in range(2):
        library.load_all_motions()
        state = library.get_motion_state(torch.arange(3), torch.full((3,), 0.02))
        torch.testing.assert_close(state["dof_pos"], torch.full((3, 29), 0.01))
        torch.testing.assert_close(state["dof_vel"], torch.full((3, 29), 0.5))
        bodies = len(library.kinematics.body_names_augment)
        assert state["rg_pos_t"].shape == (3, bodies, 3)
        assert state["rg_rot_t"].shape == (3, bodies, 4)


def test_directory_motion_loads_each_file_once_and_preserves_joints(tmp_path):
    library, clips = build_sample_library(tmp_path, mixed=False)
    directory = tmp_path / "clips"
    directory.mkdir()
    for name, clip in clips.items():
        joblib.dump({name: clip}, directory / f"{name}.pkl")
    library.m_cfg.motion_file = str(directory)
    with patch("ufo.motion.library.joblib.load", wraps=joblib.load) as load:
        library = MotionLib(library.m_cfg, num_envs=3, device="cpu")
        load.assert_not_called()
        library.load_all_motions()
        assert load.call_count == len(clips)
    ordered = [clips[Path(name).stem]["dof_pos"] for name in library.curr_motion_keys]
    torch.testing.assert_close(library.dof_pos, torch.from_numpy(np.concatenate(ordered)))


@pytest.mark.parametrize("invalid", ["empty_directory", "non_dict", "missing_key", "corrupt"])
def test_invalid_motion_sources_still_fail(tmp_path, invalid):
    library, _ = build_sample_library(tmp_path, mixed=False)
    source = tmp_path / "invalid"
    source.mkdir()
    if invalid == "empty_directory":
        match = "No motions found"
    elif invalid == "non_dict":
        source = source / "clip.pkl"
        joblib.dump([], source)
        match = "must contain a dict"
    elif invalid == "missing_key":
        joblib.dump({"other": {}}, source / "clip.pkl")
        match = "must contain key 'clip'"
    else:
        (source / "clip.pkl").write_bytes(b"")
    library.m_cfg.motion_file = str(source)
    if invalid == "corrupt":
        library = MotionLib(library.m_cfg, num_envs=3, device="cpu")
        with pytest.raises(EOFError):
            library.load_all_motions()
    else:
        with pytest.raises(ValueError, match=match):
            MotionLib(library.m_cfg, num_envs=3, device="cpu").load_all_motions()


def test_retired_height_option_is_rejected_before_loading_data():
    with pytest.raises(ValueError, match="fix_height is retired"):
        MotionLib({"fix_height": 0}, num_envs=1, device="cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("frames", [1, 7])
def test_angular_velocity_preserves_identity_endpoint_and_dtype(dtype, frames):
    from scipy.ndimage import gaussian_filter1d

    angles = torch.arange(frames, dtype=dtype) * 0.1
    rotations = torch.zeros((2, frames, 3, 4), dtype=dtype)
    rotations[..., 2] = torch.sin(angles[None, :, None] / 2)
    rotations[..., 3] = torch.cos(angles[None, :, None] / 2)
    actual = HumanoidKinematics._compute_angular_velocity(rotations, 0.02)
    expected = np.zeros((2, frames, 3, 3), dtype=np.float32 if dtype == torch.float32 else np.float64)
    expected[:, :-1, :, 2] = 5.0
    expected = torch.from_numpy(gaussian_filter1d(expected, 2, axis=1, mode="nearest"))
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
