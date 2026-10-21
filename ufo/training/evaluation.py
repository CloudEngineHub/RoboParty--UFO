import collections
import numbers
import random
from typing import Any

import numpy as np
import ot
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_map
from tqdm import tqdm

from ufo.agents.common.base import BaseConfig, BaseModel


def get_backward_observation(env, motion_id, include_last_action, velocity_multiplier: float = 1.0):
    """Use the same reference encoder inputs as standalone inference."""
    from ufo.motion.reference import get_backward_observation as build_reference

    if not env.config.obs.use_obs_filter:
        raise ValueError("UFO tracking evaluation requires obs.use_obs_filter=True")
    obs, reference = build_reference(
        env,
        motion_id,
        use_root_height_obs=env.config.obs.root_height_obs,
        velocity_multiplier=velocity_multiplier,
    )
    if not include_last_action:
        obs.pop("last_action")
    return obs, reference


class TrackingEvaluationConfig(BaseConfig):
    name_in_logs: str = "ufo_tracking_eval"

    n_episodes_per_motion: int = 1
    include_results_from_all_envs: bool = False  # If True, include results from all num_envs envs, not just first num_motions

    disable_tqdm: bool = True


    def build(self):
        return TrackingEvaluation(self)


def distributed_motion_ids(num_motions: int, rank: int, world_size: int) -> list[int]:
    """Assign each motion to exactly one rank, including uneven/empty shards."""
    if num_motions < 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid motion count, rank, or world size")
    return list(range(rank, num_motions, world_size))


def merge_tracking_metrics(shards, *, num_motions, n_episodes_per_motion, include_results_from_all_envs):
    """Merge per-motion records before computing global means and stds."""
    metrics = {}
    counts = collections.Counter()
    for rank, shard in enumerate(shards):
        for key, metric in shard.items():
            motion_id = int(metric["motion_id"])
            if not 0 <= motion_id < num_motions or motion_id % len(shards) != rank:
                raise RuntimeError(f"Unexpected motion_id={motion_id} from rank {rank}")
            if key in metrics:
                raise RuntimeError(f"Duplicate tracking metric: {key!r}")
            metrics[key] = metric
            counts[motion_id] += 1
    missing = sorted(set(range(num_motions)) - counts.keys())
    if missing:
        raise RuntimeError(f"Tracking evaluation missed motion IDs: {missing[:8]}")
    for motion_id, count in counts.items():
        if count < n_episodes_per_motion or (not include_results_from_all_envs and count != n_episodes_per_motion):
            raise RuntimeError(f"Unexpected result count for motion_id={motion_id}: {count}")
    return dict(sorted(metrics.items(), key=lambda item: (int(item[1]["motion_id"]), item[0])))


class TrackingEvaluation:
    def __init__(self, config: TrackingEvaluationConfig):
        self.cfg = config

    def run(self, *, timestep, agent_or_model, logger, env: Any, distributed: bool = False):
        if distributed and not dist.is_initialized():
            raise RuntimeError("Distributed tracking evaluation requires an initialized process group")
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1

        num_motions = int(env._env._motion_lib._num_unique_motions)
        self.motion_ids = distributed_motion_ids(num_motions, rank, world_size)
        n_envs = env._env.num_envs
        print(
            f"[tracking_eval] rank={rank}/{world_size} motions={len(self.motion_ids)}/{num_motions} envs={n_envs}",
            flush=True,
        )

        # Keep the training physics/DR, but use noise-free evaluation observations.
        env._env.is_evaluating = True
        try:
            # "Parallelization" is handled on the worker level.
            metrics = {}
            for repetition_i in range(self.cfg.n_episodes_per_motion):
                run_metrics = {}
                # If we have less envs than motions, we run iteratively over the motions
                for motion_id_chunk_start in range(0, len(self.motion_ids), n_envs):
                    motion_id_chunk = self.motion_ids[motion_id_chunk_start : motion_id_chunk_start + n_envs]
                    motion_chunk_results = _evaluate_motion_batch(
                        motion_id_chunk, agent_or_model,
                        env=env,
                        disable_tqdm=self.cfg.disable_tqdm,
                        include_results_from_all_envs=self.cfg.include_results_from_all_envs,
                    )
                    for k, v in motion_chunk_results.items():
                        assert k not in run_metrics, "Tried to override existing metric"
                        run_metrics[k] = v

                if self.cfg.n_episodes_per_motion == 1:
                    # Just return metrics, do not change names
                    metrics = run_metrics
                else:
                    # Append run number to metric names
                    for k, v in run_metrics.items():
                        metrics[f"{k}_repetition#{repetition_i}"] = v
        finally:
            # Restore even on failure and on ranks with no assigned motions.
            env._env.set_is_training()

        if distributed:
            shards = [None] * world_size if rank == 0 else None
            dist.gather_object(metrics, shards, dst=0)
            merge_error = [None]
            if rank == 0:
                try:
                    metrics = merge_tracking_metrics(
                        shards,
                        num_motions=num_motions,
                        n_episodes_per_motion=self.cfg.n_episodes_per_motion,
                        include_results_from_all_envs=self.cfg.include_results_from_all_envs,
                    )
                except RuntimeError as exc:
                    merge_error[0] = str(exc)
            # Fail on every rank instead of leaving peers waiting at the next
            # training barrier if coverage/duplicate validation fails on rank 0.
            dist.broadcast_object_list(merge_error, src=0)
            if merge_error[0] is not None:
                raise RuntimeError(merge_error[0])
            if rank != 0:
                return {}, None

        aggregate = collections.defaultdict(list)
        wandb_dict = {}
        for _, metr in metrics.items():
            for k, v in metr.items():
                if isinstance(v, numbers.Number):
                    aggregate[k].append(v)
        for k, v in aggregate.items():
            wandb_dict[k] = np.mean(v)
            wandb_dict[f"{k}#std"] = np.std(v)

        if logger is not None and metrics:
            for k, v in metrics.items():
                v["motion_name"] = k
                v["timestep"] = timestep
            fields = set().union(*(metric.keys() for metric in metrics.values()))
            logger.log({field: [metric.get(field, "") for metric in metrics.values()] for field in fields})

        return metrics, wandb_dict



def group_assign_motions_to_envs_with_map(motion_ids, num_envs, device=None):
    motion_ids = torch.tensor(motion_ids, device=device)
    num_motions = len(motion_ids)

    env_idxs = list(range(num_envs))
    # Shuffle the environment indices so repeated tracking evaluations do not
    # bind the same motion to the same per-env randomized state every time.
    random.shuffle(env_idxs)
    shuffled_env_idxs = torch.tensor(env_idxs, device=device, dtype=torch.long)

    assigned = motion_ids[shuffled_env_idxs % num_motions]

    motion_to_envs = {}
    for env_id, motion_id in enumerate(assigned.tolist()):
        motion_to_envs.setdefault(motion_id, []).append(env_id)

    return assigned, motion_to_envs


@torch.no_grad()
def _evaluate_motion_batch(motion_ids, agent, env: Any, disable_tqdm: bool = False, include_results_from_all_envs=False):
    model = agent if isinstance(agent, BaseModel) else agent._model
    core = env._env
    library = core._motion_lib
    if not library.all_motions_loaded:
        library.all_motions_loaded = True
        library.load_motions(random_sample=False, num_motions_to_load=library._num_unique_motions, start_idx=0)

    num_envs = core.num_envs
    assigned, motion_to_envs = group_assign_motions_to_envs_with_map(motion_ids, num_envs)
    contexts, targets, joint_targets = {}, {}, {}
    dof_states = torch.zeros((num_envs, core.num_dof, 2), device=core.device, dtype=core.dof_pos.dtype)
    root_states = torch.empty((num_envs, 13), device=core.device)
    for motion_id in motion_ids:
        obs, reference = get_backward_observation(core, motion_id, include_last_action=env.include_last_action)
        contexts[motion_id] = model.tracking_inference(tree_map(lambda x: x[1:], obs))
        targets[motion_id] = {"state": obs["state"].cpu()}
        joint_targets[motion_id] = reference["dof_pos"].clone()
        env_ids = motion_to_envs[motion_id]
        dof_states[env_ids, :, 0] = reference["dof_pos"][0]
        dof_states[env_ids, :, 1] = reference["ref_dof_vel"][0]
        root_states[env_ids] = torch.cat(
            [
                reference["ref_body_pos"][0, 0],
                reference["ref_body_rots"][0, 0],
                reference["ref_body_vels"][0, 0],
                reference["ref_body_angular_vels"][0, 0],
            ]
        )

    lengths = torch.tensor([len(contexts[m]) for m in motion_ids], device=core.device)
    max_length = max(len(contexts[m]) for m in motion_ids)
    padded = torch.zeros(
        (len(motion_ids), max_length, contexts[motion_ids[0]].shape[-1]), device=core.device, dtype=contexts[motion_ids[0]].dtype
    )
    for index, motion_id in enumerate(motion_ids):
        padded[index, : len(contexts[motion_id])] = contexts[motion_id]
    motion_index = {motion_id: index for index, motion_id in enumerate(motion_ids)}
    env_motion_index = torch.tensor([motion_index[m] for m in assigned.tolist()], device=core.device)
    env_lengths = lengths[env_motion_index]

    physics_cfg = core.mjlab_env.cfg
    training_episode_length = physics_cfg.episode_length_s
    physics_cfg.episode_length_s = max(training_episode_length, (max_length + 1) * core.dt)
    try:
        observation, _ = env.reset(target_states={"dof_states": dof_states, "root_states": root_states}, to_numpy=False)
        assert torch.allclose(core.dof_pos, dof_states[..., 0])

        # Metrics consume only joint positions and the joint-position prefix of
        # state. Retain snapshots on device and transfer once after the rollout.
        num_dofs = dof_states.shape[1]
        observation_states = [observation["state"][..., :num_dofs].clone()]
        joint_positions = [core.dof_pos.clone()]
        for step in tqdm(range(max_length), desc="Tracking Evaluation", disable=disable_tqdm):
            context = padded[env_motion_index, step % env_lengths]
            action = agent.act(observation, context, mean=True)
            observation, _, terminated, truncated, _ = env.step(action, to_numpy=False)
            if torch.any(terminated | truncated):
                raise RuntimeError("Tracking evaluation reset before the reference ended; refusing invalid metrics")
            observation_states.append(observation["state"][..., :num_dofs].clone())
            joint_positions.append(core.dof_pos.clone())
    finally:
        physics_cfg.episode_length_s = training_episode_length
    observation_states = torch.stack(observation_states).cpu()
    joint_positions = torch.stack(joint_positions)

    metrics = {}
    for motion_id, env_ids in motion_to_envs.items():
        count = len(contexts[motion_id]) + 1
        for repetition, env_id in enumerate(env_ids):
            result = _calc_metrics(
                {
                    "tracking_target": targets[motion_id],
                    "motion_id": motion_id,
                    "motion_file": library.curr_motion_keys[motion_id],
                    "observation": {"state": observation_states[:count, env_id]},
                    "joint_pos": joint_positions[:count, env_id],
                    "target_joint_pos": joint_targets[motion_id],
                }
            )
            key, metric = next(iter(result.items()))
            if include_results_from_all_envs:
                metrics[f"{key}_repetition#{repetition}"] = metric
            else:
                metrics[key] = metric
                break
    return metrics


def distance_matrix(X: torch.Tensor, Y: torch.Tensor):
    X_norm = X.pow(2).sum(1).reshape(-1, 1)
    Y_norm = Y.pow(2).sum(1).reshape(1, -1)
    val = X_norm + Y_norm - 2 * torch.matmul(X, Y.T)
    return torch.sqrt(torch.clamp(val, min=0))


def emd_numpy(next_obs: torch.Tensor, tracking_target: torch.Tensor, prefix=""):
    # keep only pose part of the observations
    agent_obs = next_obs.to("cpu")
    tracked_obs = tracking_target.to("cpu")
    # compute optimal transport cost
    cost_matrix = distance_matrix(agent_obs, tracked_obs).cpu().detach().numpy()
    X_pot = np.ones(agent_obs.shape[0]) / agent_obs.shape[0]
    Y_pot = np.ones(tracked_obs.shape[0]) / tracked_obs.shape[0]
    transport_cost = ot.emd2(X_pot, Y_pot, cost_matrix, numItermax=100000)
    return {f"{prefix}emd": transport_cost}


def distance_proximity(next_obs: torch.Tensor, tracking_target: torch.Tensor, bound: float = 2.0, margin: float = 2, prefix=""):
    stats = {}
    dist = torch.norm(next_obs - tracking_target, dim=-1)
    in_bounds_mask = dist <= bound
    out_bounds_mask = dist > bound + margin
    stats[f"{prefix}proximity"] = (in_bounds_mask + ((bound + margin - dist) / margin) * (~in_bounds_mask) * (~out_bounds_mask)).mean()
    stats[f"{prefix}distance"] = dist.mean()
    return stats


def _calc_metrics(ep):
    metr = {}
    observation_state = torch.as_tensor(ep["observation"]["state"], dtype=torch.float32)
    target_state = torch.as_tensor(ep["tracking_target"]["state"], dtype=torch.float32)
    joint_pos = torch.as_tensor(ep["joint_pos"], dtype=torch.float32)
    target_joint_pos = torch.as_tensor(ep["target_joint_pos"], dtype=torch.float32)
    num_dofs = target_joint_pos.shape[-1] if target_joint_pos.ndim > 0 else 0

    shape_errors = []
    if num_dofs <= 0:
        shape_errors.append("num_dofs must be greater than zero")
    if observation_state.ndim != 2:
        shape_errors.append("observation state must be two-dimensional [T, D]")
    elif observation_state.shape[-1] < num_dofs:
        shape_errors.append("observation state feature dimension is smaller than num_dofs")
    if target_state.ndim != 2:
        shape_errors.append("tracking target state must be two-dimensional [T, D]")
    elif target_state.shape[-1] < num_dofs:
        shape_errors.append("tracking target state feature dimension is smaller than num_dofs")
    if joint_pos.ndim != 2:
        shape_errors.append("joint_pos must be two-dimensional [T, D]")
    elif joint_pos.shape[-1] <= 0:
        shape_errors.append("joint_pos feature dimension must be greater than zero")
    if target_joint_pos.ndim != 2:
        shape_errors.append("target_joint_pos must be two-dimensional [T, D]")
    elif target_joint_pos.shape[-1] <= 0:
        shape_errors.append("target_joint_pos feature dimension must be greater than zero")
    if joint_pos.ndim == 2 and target_joint_pos.ndim == 2 and joint_pos.shape != target_joint_pos.shape:
        shape_errors.append("joint_pos and target_joint_pos shapes must match")
    if joint_pos.ndim == 2 and joint_pos.shape[0] < 3:
        shape_errors.append("joint_pos must contain at least 3 time steps")
    if target_joint_pos.ndim == 2 and target_joint_pos.shape[0] < 3:
        shape_errors.append("target_joint_pos must contain at least 3 time steps")
    if observation_state.ndim == 2 and target_state.ndim == 2 and observation_state.shape[0] != target_state.shape[0]:
        shape_errors.append("observation and tracking target time dimensions must match")
    if observation_state.ndim == 2 and joint_pos.ndim == 2 and observation_state.shape[0] != joint_pos.shape[0]:
        shape_errors.append("observation state and joint_pos time dimensions must match")
    if target_state.ndim == 2 and target_joint_pos.ndim == 2 and target_state.shape[0] != target_joint_pos.shape[0]:
        shape_errors.append("tracking target state and target_joint_pos time dimensions must match")
    if shape_errors:
        raise ValueError(
            "Invalid tracking metric shapes: "
            f"observation_state.shape={tuple(observation_state.shape)}, "
            f"tracking_target_state.shape={tuple(target_state.shape)}, "
            f"joint_pos.shape={tuple(joint_pos.shape)}, "
            f"target_joint_pos.shape={tuple(target_joint_pos.shape)}, "
            f"num_dofs={num_dofs}; " + "; ".join(shape_errors)
        )

    next_qpos = observation_state[..., :num_dofs]
    target_qpos = target_state[..., :num_dofs]
    dist_prox_res = distance_proximity(next_obs=next_qpos, tracking_target=target_qpos, prefix="obs_state_")
    metr.update(dist_prox_res)
    emd_res = emd_numpy(next_obs=next_qpos, tracking_target=target_qpos, prefix="obs_state_")
    metr.update(emd_res)
    metr.update(compute_joint_pos_metrics(joint_pos=joint_pos, target_joint_pos=target_joint_pos))
    for k, v in metr.items():
        if isinstance(v, torch.Tensor):
            metr[k] = v.tolist()
    metr["motion_id"] = ep["motion_id"]
    metr["motion_file"] = ep["motion_file"]
    return {ep["motion_file"]: metr}


def compute_joint_pos_metrics(joint_pos, target_joint_pos):
    """Joint-angle distances and per-frame differences; no physical time scaling."""
    stats = {}
    vel_gt = torch.diff(target_joint_pos, dim=-2)
    vel_pred = torch.diff(joint_pos, dim=-2)
    stats["joint_delta_l2_rad_per_frame"] = torch.norm(vel_pred - vel_gt, dim=-1).mean(-1)

    accel_gt = torch.diff(target_joint_pos, n=2, dim=-2)
    accel_pred = torch.diff(joint_pos, n=2, dim=-2)
    stats["joint_delta2_l2_rad_per_frame2"] = torch.norm(accel_pred - accel_gt, dim=-1).mean(-1)

    stats.update(distance_proximity(next_obs=joint_pos, tracking_target=target_joint_pos, prefix=""))
    stats.update(emd_numpy(next_obs=joint_pos, tracking_target=target_joint_pos, prefix=""))
    return stats
