"""Tracking sequence MAE and goal-reaching final-pose MAE."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils._pytree import tree_map

from ufo.agents.common.checkpoint import checkpoint_load_device, load_model_from_checkpoint_dir
from ufo.inference.goal_inference import _find_goal_json, load_and_validate_goal_json, validate_goal_motions
from ufo.inference.runtime import _target_states_from_obs, add_bool_arg, load_mjlab_env_cfg
from ufo.motion.reference import get_backward_observation, get_goal_observation


def evaluate_tracking_joint_mae(args: argparse.Namespace) -> Path:
    for name in ("chunk_envs", "log_every_prep", "log_every_step"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    model_folder = args.model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint dir: {checkpoint_dir}")

    data_path = args.data_path.expanduser().resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    output_path = (
        args.output.expanduser().resolve() if args.output is not None else model_folder / "tracking_inference" / "joint_pos_mae_stats.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pass 1: single-env prep for per-motion z/targets.
    prep_cfg, _ = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=None,
        device=args.device,

        disable_dr=True,
        disable_obs_noise=True,
        max_episode_length_s=args.max_episode_length_s,
    )
    prep_env_wrapped, _ = prep_cfg.build(num_envs=1)
    prep_env = prep_env_wrapped._env

    model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(args.device))
    model.to(args.device)
    model.eval()

    num_dof = int(prep_env.num_dof)
    prep_env._motion_lib.load_all_motions()
    num_motions = int(prep_env._motion_lib._num_unique_motions)
    num_chunks = math.ceil(num_motions / args.chunk_envs)
    print(f"[EVAL] total_motions={num_motions}, chunk_envs={args.chunk_envs}, num_chunks={num_chunks}")

    all_errs: list[np.ndarray] = []
    per_motion: dict[str, dict[str, float | int | None]] = {}

    try:
        with torch.no_grad():
            for chunk_idx, start in enumerate(range(0, num_motions, args.chunk_envs), start=1):
                end = min(start + args.chunk_envs, num_motions)
                motion_ids = list(range(start, end))

                chunk_zs: list[torch.Tensor] = []
                chunk_target_dofs: list[torch.Tensor] = []
                chunk_root_states: list[torch.Tensor] = []
                chunk_dof_states: list[torch.Tensor] = []
                chunk_lengths: list[int] = []

                for i, motion_id in enumerate(motion_ids, start=1):
                    backward_obs, obs_dict = get_backward_observation(prep_env, motion_id, use_root_height_obs=prep_cfg.root_height_obs)
                    z = model.tracking_inference(
                        tree_map(
                            lambda x: x[1:].to(args.device) if hasattr(x, "to") else x,
                            backward_obs,
                        ),
                    )
                    target_states = _target_states_from_obs(obs_dict, device=args.device, num_dof=num_dof)

                    chunk_zs.append(z)
                    chunk_target_dofs.append(obs_dict["dof_pos"].to(device=args.device, dtype=torch.float32))
                    chunk_root_states.append(target_states["root_states"][0])
                    chunk_dof_states.append(target_states["dof_states"][0])
                    chunk_lengths.append(int(z.shape[0]))

                    if i % args.log_every_prep == 0 or i == len(motion_ids):
                        print(f"[EVAL] chunk {chunk_idx}/{num_chunks} prep {i}/{len(motion_ids)}")

                run_env_wrapped, _ = prep_cfg.build(num_envs=len(motion_ids))
                run_env = run_env_wrapped._env
                try:
                    target_states = {
                        "root_states": torch.stack(chunk_root_states, dim=0),
                        "dof_states": torch.stack(chunk_dof_states, dim=0),
                    }
                    observation, _ = run_env_wrapped.reset(to_numpy=False, target_states=target_states)

                    if any(length == 0 for length in chunk_lengths):
                        raise ValueError("Tracking MAE requires at least one transition per motion")
                    max_len = max(chunk_lengths)
                    steps = torch.arange(max_len, device=args.device)
                    contexts = torch.stack([z[steps.clamp_max(len(z) - 1)] for z in chunk_zs])
                    targets = torch.stack([q[(steps + 1).clamp_max(len(q) - 1)] for q in chunk_target_dofs])
                    errors = torch.empty((len(motion_ids), max_len), device=args.device)

                    for step in range(max_len):
                        ctx_batch = contexts[:, step]
                        action = model.act(observation, ctx_batch, mean=True)
                        observation, _reward, _terminated, _truncated, _info = run_env_wrapped.step(action, to_numpy=False)

                        errors[:, step] = (run_env.dof_pos.to(torch.float32) - targets[:, step]).abs().mean(dim=-1)

                        if (step + 1) % args.log_every_step == 0 or step == 0 or step + 1 == max_len:
                            done = sum(1 for i in range(len(motion_ids)) if step + 1 >= chunk_lengths[i])
                            print(f"[EVAL] chunk {chunk_idx}/{num_chunks} step {step + 1}/{max_len}, finished={done}/{len(motion_ids)}")
                finally:
                    run_env_wrapped.close()

                errors = errors.cpu().numpy()
                for i, motion_id in enumerate(motion_ids):
                    arr = errors[i, :chunk_lengths[i]].astype(np.float64)
                    per_motion[str(motion_id)] = {
                        "mean_mae": float(arr.mean()) if arr.size else None,
                        "std_mae": float(arr.std()) if arr.size else None,
                        "num_steps": int(arr.size),
                    }
                    all_errs.append(arr)

                print(f"[EVAL] chunk {chunk_idx}/{num_chunks} done")
    finally:
        prep_env_wrapped.close()

    arr_all = np.concatenate(all_errs) if all_errs else np.empty(0, dtype=np.float64)
    if arr_all.size == 0:
        raise RuntimeError("No MAE samples collected.")

    summary = {
        "policy": model_folder.name,
        "model_folder": str(model_folder),
        "data_path": str(data_path),
        "metric": "joint_pos_mae_abs_mean_over_dof",
        "chunk_envs": int(args.chunk_envs),
        "model_type": type(model).__name__,
        "global_mean": float(arr_all.mean()),
        "global_std": float(arr_all.std()),
        "num_total_steps": int(arr_all.size),
        "per_motion": per_motion,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"[EVAL] saved: {output_path}")
    print(
        json.dumps(
            {
                "policy": summary["policy"],
                "global_mean": summary["global_mean"],
                "global_std": summary["global_std"],
                "num_total_steps": summary["num_total_steps"],
                "chunk_envs": summary["chunk_envs"],
            }
        )
    )
    return output_path



def _goal_name(goal: dict[str, Any], frame_idx: int) -> str:
    return f"{goal.get('motion_name', 'motion')}_{int(frame_idx)}"


def _unique_name(name: str, existing: dict[str, Any]) -> str:
    if name not in existing:
        return name
    suffix = 1
    while f"{name}_{suffix}" in existing:
        suffix += 1
    return f"{name}_{suffix}"


@torch.no_grad()
def _goal_z(model: torch.nn.Module, goal_obs: dict[str, torch.Tensor]) -> torch.Tensor:
    if not hasattr(model, "goal_inference"):
        raise TypeError("Loaded policy does not expose goal_inference(); goal MAE currently requires an FB-style goal encoder.")
    z = model.goal_inference(goal_obs)
    if z.ndim != 2:
        raise ValueError(f"Expected goal latent shape [batch, z_dim], got {tuple(z.shape)}")
    return z


def evaluate_goal_joint_mae(
    *,
    model_folder: Path,
    data_path: Path | None,
    goal_json: Path | None,
    device: str,
    steps: int,
    disable_dr: bool,
    disable_obs_noise: bool,
    max_episode_length_s: float,
    output: Path | None,
) -> Path:
    if steps <= 0:
        raise ValueError("--steps must be positive")

    model_folder = model_folder.expanduser().resolve()
    checkpoint_dir = model_folder / "checkpoint"
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_dir}")

    if data_path is not None:
        data_path = data_path.expanduser().resolve()
        if not data_path.exists():
            raise FileNotFoundError(f"Missing data file: {data_path}")

    output_path = (
        output.expanduser().resolve() if output is not None else model_folder / "goal_inference" / f"goal_joint_mae_{int(steps)}steps.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env_cfg, _ = load_mjlab_env_cfg(
        model_folder,
        data_path=data_path,
        robot_config=None,
        device=device,

        disable_dr=disable_dr,
        disable_obs_noise=disable_obs_noise,
        max_episode_length_s=max_episode_length_s,
    )
    if not env_cfg.robot_training:
        raise ValueError("Native task configuration requires robot_training")
    robot = env_cfg.robot_training["robot"]
    num_dof = len(robot["control_joint_names"])
    goal_json_path = _find_goal_json(goal_json, num_dof=num_dof, robot_name=robot["name"])
    goals_to_evaluate = load_and_validate_goal_json(
        goal_json_path, num_dof=num_dof, robot_name=robot["name"], control_joint_names=robot["control_joint_names"],
    )

    model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(device))
    model.to(device)
    model.eval()

    wrapped_env, _ = env_cfg.build(num_envs=1)
    env = wrapped_env._env
    num_dof = int(env.num_dof)

    print(f"[GOAL-MAE] model_folder={model_folder}", flush=True)
    print(f"[GOAL-MAE] data_path={env_cfg.motion_data_path}", flush=True)
    print(f"[GOAL-MAE] goal_json={goal_json_path}", flush=True)
    print(f"[GOAL-MAE] steps={steps}, num_dof={num_dof}, device={device}", flush=True)

    per_goal: dict[str, dict[str, Any]] = {}
    final_maes: list[float] = []

    try:
        validate_goal_motions(goals_to_evaluate, env._motion_lib._motion_data_keys)
        with torch.no_grad():
            for goal_idx, goal in enumerate(goals_to_evaluate, start=1):
                motion_id = int(goal["motion_id"])
                backward_obs, obs_dict = get_goal_observation(
                    env,
                    motion_id,
                    use_root_height_obs=env_cfg.root_height_obs,
                )
                num_frames = int(obs_dict["dof_pos"].shape[0])

                for frame_idx_raw in goal["frames"]:
                    frame_idx = int(frame_idx_raw)
                    name = _unique_name(_goal_name(goal, frame_idx), per_goal)
                    if not 0 <= frame_idx < num_frames:
                        per_goal[name] = {
                            "motion_id": motion_id,
                            "motion_name": goal.get("motion_name"),
                            "frame_idx": frame_idx,
                            "num_frames": num_frames,
                            "skipped": True,
                            "reason": "frame_idx_out_of_range",
                        }
                        print(f"[GOAL-MAE] skip {name}: frame_idx={frame_idx}, num_frames={num_frames}", flush=True)
                        continue

                    goal_observation = {key: value[frame_idx][None, ...] for key, value in backward_obs.items()}
                    goal_observation = tree_map(
                        lambda x: torch.as_tensor(x, device=device, dtype=torch.float32),
                        goal_observation,
                    )
                    z = _goal_z(model, goal_observation)
                    target_dof = obs_dict["dof_pos"][frame_idx].to(device=device, dtype=torch.float32)
                    target_states = _target_states_from_obs(obs_dict, device=device, num_dof=num_dof)
                    observation, _info = wrapped_env.reset(to_numpy=False, target_states=target_states)

                    reached_steps = 0
                    terminated_early = False
                    truncated_early = False
                    for step in range(int(steps)):
                        action = model.act(observation, z, mean=True)
                        observation, _reward, terminated, truncated, _info = wrapped_env.step(action, to_numpy=False)
                        reached_steps = step + 1
                        terminated_early = bool(torch.as_tensor(terminated).any())
                        truncated_early = bool(torch.as_tensor(truncated).any())
                        if terminated_early or truncated_early:
                            break

                    final_dof = env.dof_pos[0].to(device=device, dtype=torch.float32)
                    if final_dof.shape[0] != target_dof.shape[0]:
                        raise ValueError(f"final_dof shape {tuple(final_dof.shape)} != target_dof shape {tuple(target_dof.shape)}")
                    final_mae = float((final_dof - target_dof).abs().mean().detach().cpu().item())
                    final_maes.append(final_mae)

                    per_goal[name] = {
                        "motion_id": motion_id,
                        "motion_name": goal.get("motion_name"),
                        "frame_idx": frame_idx,
                        "num_frames": num_frames,
                        "steps_requested": int(steps),
                        "steps_executed": int(reached_steps),
                        "terminated_early": terminated_early,
                        "truncated_early": truncated_early,
                        "final_joint_mae": final_mae,
                    }
                    print(
                        f"[GOAL-MAE] {goal_idx}/{len(goals_to_evaluate)} {name}: final_joint_mae={final_mae:.6f}, steps={reached_steps}",
                        flush=True,
                    )
    finally:
        wrapped_env.close()

    if not final_maes:
        raise RuntimeError("No goal MAE samples collected.")

    arr = np.asarray(final_maes, dtype=np.float64)
    summary = {
        "policy": model_folder.name,
        "model_folder": str(model_folder),
        "data_path": str(data_path) if data_path is not None else str(env_cfg.motion_data_path),
        "goal_json": str(goal_json_path),
        "metric": "goal_final_joint_pos_mae_abs_mean_over_dof",
        "steps": int(steps),
        "num_goals": int(arr.size),
        "global_mean": float(arr.mean()),
        "global_std": float(arr.std()),
        "global_min": float(arr.min()),
        "global_max": float(arr.max()),
        "per_goal": per_goal,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    print(f"[GOAL-MAE] saved: {output_path}", flush=True)
    print(
        json.dumps(
            {
                "global_mean": summary["global_mean"],
                "global_std": summary["global_std"],
                "num_goals": summary["num_goals"],
                "steps": summary["steps"],
            }
        ),
        flush=True,
    )
    return output_path



def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    tracking = commands.add_parser("tracking", help="Per-step tracking joint-position MAE.")
    goal = commands.add_parser("goal", help="Final joint-position MAE after reaching each goal.")
    for command in (tracking, goal):
        command.add_argument("--model-folder", type=Path, required=True)
        command.add_argument("--data-path", type=Path, required=command is tracking)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--max-episode-length-s", type=float, default=10000.0)
        command.add_argument("--output", type=Path)
    tracking.add_argument("--chunk-envs", type=int, default=600)
    tracking.add_argument("--log-every-prep", type=int, default=100)
    tracking.add_argument("--log-every-step", type=int, default=300)
    goal.add_argument("--goal-json", type=Path)
    goal.add_argument("--steps", type=int, default=500)
    add_bool_arg(goal, "--disable-dr", True, "Disable domain randomization.")
    add_bool_arg(goal, "--disable-obs-noise", True, "Disable observation noise.")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.mode == "tracking":
        evaluate_tracking_joint_mae(args)
    else:
        kwargs = vars(args).copy()
        del kwargs["mode"]
        evaluate_goal_joint_mae(**kwargs)


if __name__ == "__main__":
    main()
