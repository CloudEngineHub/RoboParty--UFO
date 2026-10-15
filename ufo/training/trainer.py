# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import os

from ufo.motion.reference import load_expert_trajectories_from_motion_lib
from ufo.training.config import AgentConfig, TrainConfig
from ufo.training.evaluation import (
    TrackingEvaluation,
    TrackingEvaluationConfig,
)
from ufo.training.logging import _assert_finite, _assert_model_finite, torso_contact_force_metrics

os.environ["OMP_NUM_THREADS"] = "1"

import json
import time
import typing as tp
import uuid
from pathlib import Path
from typing import Dict

import numpy as np
import torch  # better to use scoped import if we use processes
import wandb
from torch.utils._pytree import tree_map
from tqdm import tqdm

from ufo.agents.common.buffers.storage import replay_storage_dtype
from ufo.agents.common.buffers.trajectory import ReplayBuffer
from ufo.training.distributed import (
    all_ranks_true,
    average_metrics,
    barrier,
    broadcast_agent_state,
    broadcast_object,
    module_sync_report,
    set_seed_everywhere,
    sync_model_buffers,
)
from ufo.training.logging import CSVLogger

TRAIN_LOG_FILENAME = "train_log.txt"

CHECKPOINT_DIR_NAME = "checkpoint"


class EveryNStepsChecker:
    def __init__(self, current_step: int, every_n_steps: int, step_zero_should_trigger: bool = True):
        # if step_zero_should_trigger is True, `check` will return True for step=0
        # this is to be consistent with the original modulo logic (i.e. step % N == 0)
        self.step_zero_should_trigger = step_zero_should_trigger
        self.last_step = current_step
        self.every_n_steps = every_n_steps

    def check(self, step: int) -> bool:
        if (step - self.last_step) >= self.every_n_steps or (self.step_zero_should_trigger and step == 0):
            return True
        else:
            return False

    def update_last_step(self, step: int):
        self.last_step = step


class Trainer:
    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        torch.set_float32_matmul_precision("high")
        self.distributed_rank = int(self.cfg.distributed_rank)
        self.distributed_world_size = int(self.cfg.distributed_world_size)
        self._write_shared_artifacts = self.distributed_rank == 0

        # MJLab environments are created once and shared with evaluation.
        from omegaconf import OmegaConf

        self.train_env, self.train_env_info = cfg.env.build(num_envs=cfg.online_parallel_envs)
        self.obs_space = self.train_env.single_observation_space
        self.action_space = self.train_env.single_action_space

        assert "time" in self.obs_space.keys(), "Observation space must contain 'obs' and 'time' (TimeAwareObservation wrapper)"
        assert len(self.action_space.shape) == 1, "Only 1D action space is supported (first dim should be vector env)"
        # TODO for backwards consistency, we do not pass "time" to the agent, so we remove it from the obs_space we pass to the agent/model
        #      but would we need it at some point?
        del self.obs_space.spaces["time"]

        self.action_dim = self.action_space.shape[0]

        print(f"Workdir: {self.cfg.work_dir}")
        self.work_dir = Path(self.cfg.work_dir)
        self.work_dir.mkdir(exist_ok=True, parents=True)

        if self._write_shared_artifacts:
            with open(self.work_dir / "config.yaml", "w") as file:
                OmegaConf.save(self.train_env_info["task_config"], file)

        self.train_logger = CSVLogger(filename=self.work_dir / TRAIN_LOG_FILENAME) if self._write_shared_artifacts else None

        set_seed_everywhere(self.cfg.seed)

        self.agent, self._checkpoint_status = create_agent_or_load_checkpoint(
            self.work_dir, self.cfg, agent_build_kwargs=dict(obs_space=self.obs_space, action_dim=self.action_dim)
        )
        self._checkpoint_local_time = int(self._checkpoint_status["local_time"])
        self._checkpoint_global_time = int(self._checkpoint_status["global_time"])
        self._optimizer_steps = int(self._checkpoint_status["optimizer_steps"])
        self._replay_update_after = self.cfg.num_seed_steps
        self.agent.compile_network_forwards()
        if self.cfg.distributed_sync:
            broadcast_agent_state(self.agent, src=0)
        self.agent._model.train()

        self.evaluations: dict[str, TrackingEvaluation] = {cfg.name_in_logs: cfg.build() for cfg in self.cfg.evaluations}
        self.evaluate = len(self.evaluations) > 0

        self.eval_loggers = (
            {name: CSVLogger(filename=self.work_dir / f"{name}.csv") for name in self.evaluations.keys()}
            if self._write_shared_artifacts
            else {}
        )

        if self._write_shared_artifacts and self.cfg.use_wandb:
            init_wandb(self.cfg)

        if self._write_shared_artifacts:
            with (self.work_dir / "config.json").open("w") as f:
                f.write(self.cfg.model_dump_json(indent=4))
            self._save_effective_config()

        self.priorization_eval_name = None
        if self.cfg.prioritization:
            for name, evaluation in self.evaluations.items():
                if isinstance(evaluation.cfg, TrackingEvaluationConfig):
                    self.priorization_eval_name = name
                    break
            if self.priorization_eval_name is None:
                raise ValueError("Prioritization requires tracking evaluation to be enabled")

    def train(self) -> None:
        """Main path: prepare → save/evaluate when due → collect → update → log.

        Stage implementations follow this method in the same file. All schedule
        decisions use the original local/global counters and boundary order.
        """
        replay_buffer = self._prepare_replay_buffers()
        train_env = self.train_env

        print("Starting training")
        accounting = _step_accounting(self.cfg)
        global_step_scale = accounting["global_step_scale"]
        local_step_increment = self.cfg.online_parallel_envs
        global_step_increment = local_step_increment * global_step_scale
        max_local_time = (self.cfg.num_env_steps + global_step_scale - 1) // global_step_scale
        if self._write_shared_artifacts:
            print(
                "[INFO] Step accounting: "
                f"num_envs_per_rank={accounting['num_envs_per_rank']}, global_parallel_envs={accounting['global_parallel_envs']}, "
                f"local_step_increment={local_step_increment}, global_step_increment={global_step_increment}, "
                f"num_env_steps_global={self.cfg.num_env_steps}, num_seed_steps_local={self.cfg.num_seed_steps}, "
                f"update_agent_every_local={self.cfg.update_agent_every}, world_size={self.distributed_world_size}, "
                "gradient_sync=manual_128MiB, "
                f"loss_mode={accounting['loss_mode']}, effective_batch_size={accounting['effective_batch_size']}, "
                f"replay_capacity_per_rank={accounting['replay_capacity_per_rank']}, "
                f"effective_replay_capacity={accounting['effective_replay_capacity']}, "
                f"trajectory_steps_per_rank={accounting['trajectory_steps_per_rank']}, "
                "compile=CUDA_network_forwards"
            )
        progb = tqdm(
            total=self.cfg.num_env_steps,
            initial=min(self._checkpoint_global_time, self.cfg.num_env_steps),
            disable=self.cfg.disable_tqdm,
        )
        td, info = train_env.reset(to_numpy=False)
        if self.cfg.fail_on_nonfinite:
            self._check_finite(
                td, label="env.reset.obs", local_time=self._checkpoint_local_time, global_time=self._checkpoint_global_time
            )
        # see https://farama.org/Vector-Autoreset-Mode
        terminated = torch.zeros(self.cfg.online_parallel_envs, dtype=torch.bool, device=self.agent.device)
        truncated = torch.zeros(self.cfg.online_parallel_envs, dtype=torch.bool, device=self.agent.device)
        total_metrics, context = None, None
        metric_update_counts: dict[str, int] = {}
        start_time = time.time()
        fps_start_time = time.time()
        fps_start_step = self._checkpoint_global_time
        checkpoint_time_checker = EveryNStepsChecker(
            self._checkpoint_global_time, self.cfg.checkpoint_every_steps, step_zero_should_trigger=False
        )
        buffer_time_checker = EveryNStepsChecker(
            self._checkpoint_status.get("buffer_global_time", self._checkpoint_global_time), self.cfg.buffer_every_steps,
            step_zero_should_trigger=False,
        )
        eval_time_checker = EveryNStepsChecker(
            self._checkpoint_global_time, self.cfg.eval_every_steps, step_zero_should_trigger=self.cfg.eval_at_start
        )
        update_agent_time_checker = EveryNStepsChecker(self._checkpoint_local_time, self.cfg.update_agent_every)
        log_time_checker = EveryNStepsChecker(self._checkpoint_global_time, self.cfg.log_every_steps)

        for local_time in range(self._checkpoint_local_time, max_local_time + local_step_increment, local_step_increment):
            global_time = local_time * global_step_scale
            if global_time > self.cfg.num_env_steps:
                break
            finished = global_time + global_step_increment > self.cfg.num_env_steps
            model_due = checkpoint_time_checker.check(global_time)
            buffer_due = buffer_time_checker.check(global_time)
            if finished or model_due or buffer_due:
                stop_requested = (self.work_dir / "SAVE_AND_STOP").exists() if self.distributed_rank == 0 else False
                if self.cfg.distributed_sync:
                    stop_requested = broadcast_object(stop_requested, src=0)
                replay_mode = "all" if stop_requested else "rank0" if finished or buffer_due else "none"
                checkpoint_time_checker.update_last_step(global_time)
                if replay_mode != "none":
                    buffer_time_checker.update_last_step(global_time)
                self.save(
                    local_time=local_time, global_time=global_time, optimizer_steps=self._optimizer_steps,
                    replay_buffer=replay_buffer, replay_mode=replay_mode,
                )
                if stop_requested:
                    if self.distributed_rank == 0:
                        (self.work_dir / "SAVE_AND_STOP").unlink(missing_ok=True)
                        print(f"[INFO] Saved all replay shards and stopped at global_time={global_time}", flush=True)
                    break
                if finished:
                    break

            if self.evaluate and (global_time > 0 or self.cfg.eval_at_start) and (
                eval_time_checker.check(global_time) or global_time == self._checkpoint_global_time
            ):
                td, info, terminated, truncated = self._evaluate_training(
                    replay_buffer,
                    local_time,
                    global_time,
                    eval_time_checker,
                )

            new_td, new_info, new_terminated, new_truncated, context = self._collect_transition(
                td,
                info,
                context,
                terminated,
                truncated,
                replay_buffer,
                local_time,
                global_time,
                eval_time_checker,
            )

            if local_time > self._replay_update_after and update_agent_time_checker.check(local_time):
                if not self._replay_ready:
                    ready = replay_buffer["train"].can_sample()
                    self._replay_ready = all_ranks_true(ready, self.agent.device) if self.cfg.distributed_sync else ready
                if self._replay_ready:
                    update_agent_time_checker.update_last_step(local_time)
                    total_metrics, metric_update_counts = self._update_agent(
                        replay_buffer,
                        local_time,
                        global_time,
                        total_metrics,
                        metric_update_counts,
                    )

            if log_time_checker.check(global_time) and total_metrics is not None:
                fps_start_time = self._log_training(
                    total_metrics,
                    metric_update_counts,
                    local_time,
                    global_time,
                    start_time,
                    fps_start_time,
                    log_time_checker,
                    fps_start_step=fps_start_step,
                )
                fps_start_step = global_time + global_step_increment
                total_metrics, metric_update_counts = None, {}

            progb.update(global_step_increment)
            td = new_td
            # Simulator termination buffers may be reused by the next step.
            terminated = new_terminated.clone()
            truncated = new_truncated.clone()
            info = new_info
        train_env.close()

    def _prepare_replay_buffers(
        self,
    ):
        expert_buffer = load_expert_trajectories_from_motion_lib(self.train_env._env, self.cfg.agent, device=self.cfg.buffer_device)
        print("Allocating buffers")
        replay_buffer = {}
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        checkpoint_buffer_dir = self._checkpoint_buffer_path(checkpoint_dir)
        restore = _replay_matches_checkpoint(checkpoint_buffer_dir, self._checkpoint_status)
        if self.cfg.distributed_sync:
            restore = all_ranks_true(restore, self.agent.device)
        self._replay_ready = False
        self._replay_update_after = max(self.cfg.num_seed_steps, self._checkpoint_status.get("replay_update_after", 0))
        if restore:
            print("Loading checkpointed buffer")
            replay_buffer["train"] = ReplayBuffer.load(checkpoint_buffer_dir, device=self.cfg.buffer_device)
            # The simulator resets on resume; never join its new state to an old trajectory.
            buffer = replay_buffer["train"]
            buffer.output_key_t = _trajectory_output_keys(self.cfg.agent)
            unused = ["reward", "step_count"]
            if self.cfg.agent.name != "FBcprAuxAgent":
                unused += ["qpos", "qvel"]
            for key in unused:
                buffer.storage.pop(key, None)
                buffer.storage.get("next", {}).pop(key, None)
            if len(buffer):
                buffer.storage[buffer.end_key][(buffer._idx - 1) % buffer.capacity] = True
            print(f"Loaded buffer of size {len(replay_buffer['train'])}")
        else:
            replay_buffer["train"] = ReplayBuffer(
                capacity=self.cfg.buffer_size // self.cfg.online_parallel_envs,  # make sure to divide by num_envs
                device=self.cfg.buffer_device,
                n_dim=2,
                end_key="done",
                output_key_t=_trajectory_output_keys(self.cfg.agent),
                output_key_tp1=["observation", "terminated"],
            )
            self._replay_update_after = self._checkpoint_local_time + self.cfg.num_seed_steps
            if self._checkpoint_local_time and self.distributed_rank == 0:
                print(
                    "[INFO] No complete matching replay snapshot; all ranks collect fresh data "
                    f"before updates resume after local_time={self._replay_update_after}.", flush=True
                )
        replay_buffer["expert"] = expert_buffer

        return replay_buffer

    def _evaluate_training(
        self, replay_buffer, local_time, global_time, eval_time_checker
    ):
        train_env = self.train_env
        eval_metrics = {}
        distributed_tracking_eval = self.cfg.distributed_sync and self.distributed_world_size > 1
        run_eval_on_this_rank = distributed_tracking_eval or (not self.cfg.distributed_sync) or self.distributed_rank == 0
        if run_eval_on_this_rank:
            eval_metrics = self.eval(global_time, distributed_shard=distributed_tracking_eval)
        if self.cfg.distributed_sync:
            barrier()
        eval_time_checker.update_last_step(global_time)
        # reset if there is a ufo evaluation
        td, info = train_env.reset(to_numpy=False)
        if self.cfg.fail_on_nonfinite:
            self._check_finite(
                td, label="env.post_eval_reset.obs", local_time=local_time, global_time=global_time
            )
        terminated = torch.zeros(self.cfg.online_parallel_envs, dtype=torch.bool, device=self.agent.device)
        truncated = torch.zeros(self.cfg.online_parallel_envs, dtype=torch.bool, device=self.agent.device)

        if self.cfg.prioritization:
            # priorities
            priority_payload = None
            if (not self.cfg.distributed_sync) or self.distributed_rank == 0:
                assert len(eval_metrics[self.priorization_eval_name]) == len(replay_buffer["expert"].motion_ids), (
                    "Mismatch in number of motions returned by the eval"
                )
                index_in_buffer, name_in_buffer = {}, {}
                for i, motion_id in enumerate(replay_buffer["expert"].motion_ids):
                    index_in_buffer[motion_id] = i
                    if hasattr(replay_buffer["expert"], "file_names"):
                        name_in_buffer[motion_id] = replay_buffer["expert"].file_names[i]
                priorities, idxs = [], []
                for _, metr in eval_metrics[self.priorization_eval_name].items():
                    priorities.append(metr["emd"])
                    idxs.append(index_in_buffer[metr["motion_id"]])
                priorities = (
                    torch.clamp(
                        torch.tensor(priorities, dtype=torch.float32, device=self.agent.device),
                        min=self.cfg.prioritization_min_val,
                        max=self.cfg.prioritization_max_val,
                    )
                    * self.cfg.prioritization_scale
                )

                if self.cfg.prioritization_mode == "lin":
                    pass
                elif self.cfg.prioritization_mode == "exp":
                    priorities = 2**priorities
                elif self.cfg.prioritization_mode == "bin":
                    bins = torch.floor(priorities)
                    for i in range(int(bins.min().item()), int(bins.max().item()) + 1):
                        mask = bins == i
                        n = mask.sum().item()
                        if n > 0:
                            priorities[mask] = 1 / n
                else:
                    raise ValueError(f"Unsupported prioritization mode {self.cfg.prioritization_mode}")
                priority_payload = {
                    "priorities": priorities.detach().cpu(),
                    "idxs": idxs,
                    "file_name": name_in_buffer,
                }
            if self.cfg.distributed_sync:
                priority_payload = broadcast_object(priority_payload, src=0)
            if priority_payload is None:
                raise RuntimeError("Prioritization requires evaluation metrics, but no priority payload was produced.")
            priorities = priority_payload["priorities"].to(self.agent.device)
            idxs = priority_payload["idxs"]
            name_in_buffer = priority_payload["file_name"]

            train_env._env._motion_lib.update_sampling_weight_by_id(priorities=list(priorities), motions_id=idxs, file_name=name_in_buffer)

            replay_buffer["expert"].update_priorities(
                priorities=priorities.to(self.cfg.buffer_device), idxs=torch.tensor(np.array(idxs), device=self.cfg.buffer_device)
            )

        return td, info, terminated, truncated

    def _collect_transition(
        self, td, info, context, terminated, truncated, replay_buffer, local_time, global_time, eval_time_checker
    ):
        train_env = self.train_env
        with torch.no_grad():
            obs = tree_map(
                lambda x: torch.as_tensor(x, dtype=replay_storage_dtype(x.dtype), device=self.agent.device).clone(), td
            )
            # TODO consistency with obs_space: remove time assigned by TimeAwareObservationWrapper
            step_count = obs.pop("time")

            context = self.agent.maybe_update_rollout_context(z=context, step_count=step_count, replay_buffer=replay_buffer)
            if local_time < self.cfg.num_seed_steps:
                action = torch.as_tensor(train_env.action_space.sample(), dtype=torch.float32, device=self.agent.device)
            else:
                action = self.agent.act(obs=obs, z=context, mean=False)
            check_rollout_nonfinite = (
                self.cfg.fail_on_nonfinite
                and self.cfg.nonfinite_check_rollout_every_local_steps > 0
                and local_time % self.cfg.nonfinite_check_rollout_every_local_steps == 0
            )
            if check_rollout_nonfinite:
                self._check_finite(
                    obs, label="rollout.obs", local_time=local_time, global_time=global_time
                )
                self._check_finite(
                    context, label="rollout.context", local_time=local_time, global_time=global_time
                )
                self._check_finite(
                    action, label="rollout.action", local_time=local_time, global_time=global_time
                )
        new_td, new_reward, new_terminated, new_truncated, new_info = train_env.step(action, to_numpy=False)
        if check_rollout_nonfinite:
            self._check_finite(
                new_td, label="env.step.obs", local_time=local_time, global_time=global_time
            )
            self._check_finite(
                new_reward, label="env.step.reward", local_time=local_time, global_time=global_time
            )

        # we check if at the next iteration we will evaluate
        next_local_time = local_time + self.cfg.online_parallel_envs
        next_global_time = next_local_time * _global_step_scale(self.cfg)
        if (self.evaluate and eval_time_checker.check(next_global_time)) or (
            self.evaluate and next_global_time == self._checkpoint_global_time
        ):
            # Evaluation shares the environment, so end the trajectory before its reset.
            new_truncated = torch.ones_like(new_truncated, dtype=torch.bool)

        next_obs = {key: value for key, value in new_td.items() if key != "time"}
        final_ids = new_info.get("final_env_ids")
        if final_ids is not None:
            next_obs = tree_map(lambda value: value.clone(), next_obs)
            for key in next_obs:
                next_obs[key][final_ids] = new_info["final_observation"][key]

        data = {
            "observation": tree_map(lambda x: x[None, ...], obs),
            "action": action[None, ...],
            "done": (new_terminated | new_truncated)[None, ..., None],
            "next": {
                "observation": tree_map(lambda value: value[None, ...], next_obs),
                "terminated": new_terminated[None, ..., None],
                "truncated": new_truncated[None, ..., None],
            },
        }
        if context is not None:
            data["z"] = context[None, ...]
        if self.cfg.agent.name == "FBcprAuxAgent":
            for key in ("qpos", "qvel"):
                if key in info:
                    data[key] = info[key][None, ...]
                    value = new_info[key]
                    if final_ids is not None:
                        value = value.clone()
                        value[final_ids] = new_info[f"final_{key}"]
                    data["next"][key] = value[None, ...]
        if "aux_rewards" in new_info:
            data["aux_rewards"] = {k: v[None, ..., None] for k, v in new_info["aux_rewards"].items() if not k.startswith("_")}
        if check_rollout_nonfinite:
            self._check_finite(
                data, label="replay.extend.data", local_time=local_time, global_time=global_time
            )
        replay_buffer["train"].extend(data)

        return new_td, new_info, new_terminated, new_truncated, context

    def _update_agent(self, replay_buffer, local_time, global_time, total_metrics, metric_update_counts):
        for _ in range(self.cfg.num_agent_updates):
            metrics = self.agent.update(replay_buffer, local_time)
            self._optimizer_steps += 1
            if self.cfg.fail_on_nonfinite:
                self._check_finite(
                    metrics, label="agent.update.metrics", local_time=local_time, global_time=global_time
                )
                if (
                    self.cfg.nonfinite_check_model_every_updates > 0
                    and self._optimizer_steps % self.cfg.nonfinite_check_model_every_updates == 0
                ):
                    _assert_model_finite(
                        self.agent._model,
                        rank=self.distributed_rank,
                        local_time=local_time,
                        global_time=global_time,
                        optimizer_steps=self._optimizer_steps,
                    )
            if self.cfg.distributed_sync:
                sync_model_buffers(self.agent._model)
            total_metrics, metric_update_counts = _accumulate_metrics(
                total_metrics,
                metric_update_counts,
                metrics,
            )

        return total_metrics, metric_update_counts

    def _log_training(
        self, total_metrics, metric_update_counts, local_time, global_time, start_time, fps_start_time, log_time_checker,
        *, fps_start_step,
    ):
        # Logging is observational: reduce accumulated metrics once per
        # log interval, never once per optimizer update.
        if self.cfg.distributed_sync:
            total_metrics = average_metrics(total_metrics)
        log_time_checker.update_last_step(global_time)
        if not self._write_shared_artifacts:
            return time.time()
        m_dict = {k: np.round((total_metrics[k] / metric_update_counts[k]).mean().item(), 6) for k in sorted(total_metrics)}
        m_dict.update(torso_contact_force_metrics(
            self.train_env, enabled=self.cfg.log_torso_contact_forces, threshold=self.cfg.torso_contact_force_threshold
        ))
        now = time.time()
        completed_steps = global_time + self.cfg.online_parallel_envs * _global_step_scale(self.cfg)
        m_dict["duration [minutes]"] = (now - start_time) / 60
        m_dict["FPS"] = (completed_steps - fps_start_step) / (now - fps_start_time)
        m_dict["execution/compile_networks"] = int(self.agent.device == "cuda")
        m_dict["execution/fused_adam"] = int(self.agent.device == "cuda")
        m_dict["execution/shared_critic"] = int(getattr(self.agent.cfg.model.archi, "shared_critic", False))
        m_dict["execution/matmul_precision"] = "high"
        if self.cfg.distributed_sync and self.distributed_world_size > 1:
            m_dict["distributed/gradient_sync"] = "manual"
            m_dict["distributed/world_size"] = self.distributed_world_size
            m_dict.update({f"distributed/{k}": v for k, v in _step_accounting(self.cfg).items() if k != "global_step_scale"})
        m_dict["distributed/local_env_steps"] = int(local_time)
        m_dict["distributed/global_env_steps"] = int(global_time)
        m_dict["distributed/optimizer_steps"] = int(self._optimizer_steps)
        if self.cfg.use_wandb:
            wandb.log(
                {f"train/{k}": v for k, v in m_dict.items()},
                step=global_time,
            )
        print(m_dict)
        m_dict["timestep"] = global_time
        m_dict["local_timestep"] = local_time
        if self.train_logger is not None:
            self.train_logger.log(m_dict)

        # Start at the measurement boundary so the next interval includes logging I/O.
        return now

    def eval(self, t, *, distributed_shard: bool = False):
        print(f"Starting evaluation at time {t} on rank {self.distributed_rank}", flush=True)
        evaluation_results = {}

        for evaluation_name, evaluation in self.evaluations.items():
            logger = self.eval_loggers.get(evaluation_name)
            self.agent._model.train(False)

            evaluation_metrics, wandb_dict = evaluation.run(
                timestep=t,
                agent_or_model=self.agent,
                logger=logger,
                env=self.train_env,
                distributed=distributed_shard,
            )
            # For wandb dict, put it on wandb
            if self._write_shared_artifacts and self.cfg.use_wandb and wandb_dict is not None:
                wandb.log(
                    {f"eval/{evaluation_name}/{k}": v for k, v in wandb_dict.items()},
                    step=t,
                )

            evaluation_results[evaluation_name] = evaluation_metrics

        # ---------------------------------------------------------------
        # this is important, move back the agent to cuda and
        # restart the training
        self.agent._model.train()

        return evaluation_results

    def save(
        self, *, local_time: int, global_time: int, optimizer_steps: int,
        replay_buffer: Dict[str, tp.Any], replay_mode: tp.Literal["none", "rank0", "all"] = "none",
    ) -> None:
        checkpoint_dir = self.work_dir / CHECKPOINT_DIR_NAME
        sync_report = None
        if self.cfg.distributed_sync:
            barrier()
            sync_report = module_sync_report(self.agent._model, src=0)
            if sync_report["max_abs_diff_from_rank0"] > 1.0e-5:
                raise RuntimeError(f"Distributed model state diverged before checkpoint: {sync_report}")
        snapshot_id = uuid.uuid4().hex if self.distributed_rank == 0 else None
        if self.cfg.distributed_sync:
            snapshot_id = broadcast_object(snapshot_id, src=0)
        if self.distributed_rank == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            # A crash during an in-place save must never look like a valid checkpoint.
            (checkpoint_dir / ".saving").touch()
            print(f"Checkpointing at local_time={local_time} global_time={global_time} "
                  f"optimizer_steps={optimizer_steps} replay={replay_mode}", flush=True)
            self.agent.save(str(checkpoint_dir))
        if replay_mode == "all" or (replay_mode == "rank0" and self.distributed_rank == 0):
            buffer_path = self._checkpoint_buffer_path(checkpoint_dir)
            marker = buffer_path / "snapshot.json"
            marker.unlink(missing_ok=True)
            if len(replay_buffer["train"]):
                replay_buffer["train"].save(buffer_path)
                _write_json(marker, {"id": snapshot_id, "global_time": global_time, "world_size": self.distributed_world_size,
                                     "transition_format": 2})
        if self.cfg.distributed_sync:
            barrier()
        status = _make_train_status(
            self.cfg, local_time=local_time, global_time=global_time, optimizer_steps=optimizer_steps,
        )
        status["replay_id"] = snapshot_id if replay_mode == "all" or (replay_mode == "rank0" and self.distributed_world_size == 1) else None
        status["buffer_global_time"] = global_time if replay_mode != "none" else self._checkpoint_status.get("buffer_global_time", 0)
        status["replay_update_after"] = self._replay_update_after
        if self.distributed_rank == 0:
            if sync_report is not None:
                _write_json(checkpoint_dir / "distributed_sync.json", sync_report)
            _write_json(checkpoint_dir / "train_status.json", status)
            (checkpoint_dir / ".saving").unlink()
        if self.cfg.distributed_sync:
            barrier()
        self._checkpoint_status = status

    def _check_finite(self, value, *, label: str, local_time: int, global_time: int) -> None:
        _assert_finite(
            value,
            label=label,
            rank=self.distributed_rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=self._optimizer_steps,
        )

    def _save_effective_config(self):
        """One inspection artifact, including the actual agent on checkpoint resume."""
        from omegaconf import OmegaConf

        accounting = _step_accounting(self.cfg)
        manifest = {
            "training": self.cfg.model_dump(mode="json"),
            "agent": self.agent.cfg.model_dump(mode="json"),
            "task": OmegaConf.to_container(OmegaConf.create(self.train_env_info["task_config"]), resolve=True),
            "resume": self._checkpoint_status,
            "step_accounting": {
                **{
                    k: accounting[k]
                    for k in ("num_envs_per_rank", "global_parallel_envs", "effective_batch_size", "global_step_scale", "loss_mode")
                },
                "seed_steps_unit": "local_env_steps",
                "update_interval_unit": "local_env_steps",
                "budget_eval_checkpoint_log_unit": "global_env_steps",
            },
        }
        path = self.work_dir / "effective_config.json"
        path.write_text(json.dumps(manifest, indent=4))
        print(f"[INFO] Effective training configuration: {path}", flush=True)

    def _checkpoint_buffer_path(self, checkpoint_dir: Path) -> Path:
        return checkpoint_dir / "buffers" / f"train_rank_{self.distributed_rank}"



def _trajectory_output_keys(agent: AgentConfig) -> list[str]:
    keys = ["observation", "action", "z"]
    if getattr(agent, "aux_rewards", None):
        keys.append("aux_rewards")
    return keys


def _accumulate_metrics(
    total_metrics: dict[str, torch.Tensor] | None,
    metric_update_counts: dict[str, int],
    metrics: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    if total_metrics is None:
        total_metrics = {}
    keys, totals, values = [], [], []
    for key, value in metrics.items():
        value = value.float()
        if key in total_metrics:
            keys.append(key)
            totals.append(total_metrics[key])
            values.append(value)
        else:
            total_metrics[key] = value.clone()
        metric_update_counts[key] = metric_update_counts.get(key, 0) + 1
    if keys:
        total_metrics.update(zip(keys, torch._foreach_add(totals, values)))
    return total_metrics, metric_update_counts


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=4))
    temporary.replace(path)


def _replay_matches_checkpoint(buffer_path: Path, status: dict) -> bool:
    if not status.get("replay_id"):
        return False
    try:
        snapshot = json.loads((buffer_path / "snapshot.json").read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(snapshot, dict)
        and snapshot.get("transition_format") == 2
        and snapshot.get("id") == status["replay_id"]
        and snapshot.get("global_time") == status["global_time"]
        and snapshot.get("world_size") == status["world_size"]
        and (buffer_path / "buffer.hdf5").is_file()
        and (buffer_path / "config.json").is_file()
    )


def create_agent_or_load_checkpoint(work_dir: Path, cfg: TrainConfig, agent_build_kwargs: dict[str, tp.Any]):
    checkpoint_dir = work_dir / CHECKPOINT_DIR_NAME
    if (checkpoint_dir / ".saving").exists():
        raise RuntimeError(f"Incomplete checkpoint save: {checkpoint_dir}; restore a verified backup or use a fresh work_dir.")
    train_status_path = checkpoint_dir / "train_status.json"
    checkpoint_status = _initial_train_status(cfg)
    if train_status_path.exists():
        with train_status_path.open("r") as f:
            train_status = json.load(f)
        checkpoint_status = _normalize_train_status(train_status, cfg)

        print(
            f"Loading the agent at local_time={checkpoint_status['local_time']} "
            f"global_time={checkpoint_status['global_time']} optimizer_steps={checkpoint_status['optimizer_steps']}"
        )
        agent = cfg.agent.object_class.load(checkpoint_dir, device=cfg.agent.model.device)
    else:
        agent = cfg.agent.build(**agent_build_kwargs)
    return agent, checkpoint_status


def _global_step_scale(cfg: TrainConfig) -> int:
    if int(cfg.distributed_world_size) > 1:
        return int(cfg.distributed_world_size)
    return 1


def _step_accounting(cfg: TrainConfig) -> dict[str, int | str]:
    scale = _global_step_scale(cfg)
    num_envs = int(cfg.online_parallel_envs)
    replay_capacity = int(cfg.buffer_size)
    return {
        "loss_mode": "local_loss_average" if cfg.distributed_sync and int(cfg.distributed_world_size) > 1 else "single_rank",
        "effective_batch_size": int(cfg.agent.train.batch_size) * scale,
        "num_envs_per_rank": num_envs,
        "global_parallel_envs": num_envs * scale,
        "replay_capacity_per_rank": replay_capacity,
        "effective_replay_capacity": replay_capacity * scale,
        "trajectory_steps_per_rank": replay_capacity // max(num_envs, 1),
        "global_step_scale": scale,
    }


def _initial_train_status(cfg: TrainConfig) -> dict[str, tp.Any]:
    return _make_train_status(cfg, local_time=0, global_time=0, optimizer_steps=0)


def _normalize_train_status(train_status: dict[str, tp.Any], cfg: TrainConfig) -> dict[str, tp.Any]:
    initial = _initial_train_status(cfg)
    missing = initial.keys() - train_status.keys()
    if missing:
        raise ValueError(f"Checkpoint train status is missing required fields: {sorted(missing)}")
    current_world_size = int(cfg.distributed_world_size)
    checkpoint_world_size = int(train_status["world_size"])
    if checkpoint_world_size != current_world_size:
        raise RuntimeError(
            "Cannot resume checkpoint with a different distributed world_size: "
            f"checkpoint world_size={checkpoint_world_size}, current world_size={current_world_size}. "
            "Rank-local replay buffer shards cannot be automatically migrated; use a matching GPU count, "
            "start a fresh work_dir, or perform an explicit buffer migration."
        )
    return {
        **{key: train_status[key] for key in initial},
        **{key: train_status[key] for key in ("replay_id", "buffer_global_time", "replay_update_after") if key in train_status},
    }


def _make_train_status(cfg: TrainConfig, *, local_time: int, global_time: int, optimizer_steps: int) -> dict[str, tp.Any]:
    accounting = _step_accounting(cfg)
    return {
        "local_time": int(local_time),
        "global_time": int(global_time),
        "optimizer_steps": int(optimizer_steps),
        "world_size": int(cfg.distributed_world_size),
        "loss_mode": accounting["loss_mode"],
        "effective_batch_size": accounting["effective_batch_size"],
    }


def init_wandb(cfg: TrainConfig):
    from pathlib import Path

    exp_name = cfg.wandb_run_name if cfg.wandb_run_name else Path(cfg.work_dir).name
    wandb_config = cfg.model_dump()
    wandb.init(entity=cfg.wandb_ename, project=cfg.wandb_pname, group=cfg.wandb_gname, name=exp_name, config=wandb_config, dir="./_wandb")
