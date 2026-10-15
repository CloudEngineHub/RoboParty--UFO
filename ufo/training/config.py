# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""Training configuration and supported agent/evaluation types."""

from typing import List

import pydantic

from ufo.agents.common.base import BaseConfig
from ufo.agents.fb_cpr_aux.config import FBcprAuxAgentConfig
from ufo.agents.tech.config import TeCHAgentConfig
from ufo.envs.config import MJLabEnvConfig
from ufo.training.evaluation import TrackingEvaluationConfig

DEFAULT_CHECKPOINT_EVERY_STEPS = 3_200_000
DEFAULT_BUFFER_EVERY_STEPS = 16_000_000

AgentConfig = FBcprAuxAgentConfig | TeCHAgentConfig


class TrainConfig(BaseConfig):
    # The "pydantic.Field" field is used to explicitely tell which field is the discriminative
    # feature
    agent: AgentConfig = pydantic.Field(discriminator="name")

    env: MJLabEnvConfig = pydantic.Field(discriminator="name")

    work_dir: str

    seed: int
    online_parallel_envs: int
    # Logging, evaluation and total budget use global steps; update/seed intervals use local steps.
    log_every_steps: int
    num_env_steps: int
    update_agent_every: int
    num_seed_steps: int
    num_agent_updates: int
    # Model and replay snapshots have separate global environment-step intervals.
    checkpoint_every_steps: int = pydantic.Field(default=DEFAULT_CHECKPOINT_EVERY_STEPS, gt=0)
    buffer_every_steps: int = pydantic.Field(default=DEFAULT_BUFFER_EVERY_STEPS, gt=0)
    prioritization: bool = False
    prioritization_min_val: float = 0.5
    prioritization_max_val: float = 5
    prioritization_scale: float = 2
    prioritization_mode: str = "exp"  # ["bin", "exp", "lin"]

    # Buffer
    buffer_size: int

    # WANDB
    use_wandb: bool = False
    wandb_ename: str | None = None
    wandb_gname: str | None = None
    wandb_pname: str | None = None
    wandb_run_name: str | None = None

    # misc
    buffer_device: str
    # Default to True; otherwise you will spam the console with tqdm
    disable_tqdm: bool = True
    log_torso_contact_forces: bool = True
    torso_contact_force_threshold: float = 1.0
    distributed_rank: int = 0
    distributed_world_size: int = 1
    fail_on_nonfinite: bool = True
    nonfinite_check_model_every_updates: int = 0
    nonfinite_check_rollout_every_local_steps: int = 0

    evaluations: List[TrackingEvaluationConfig] = pydantic.Field(default_factory=lambda: [])
    eval_every_steps: int
    eval_at_start: bool = True

    tags: dict = pydantic.Field(default_factory=lambda: {})


    @property
    def distributed_sync(self) -> bool:
        return self.distributed_world_size > 1

    def model_post_init(self, context):
        if self.prioritization and not self.evaluations:
            raise ValueError("Prioritization requires tracking evaluation to be enabled")

        # Ensure all evaluations have unique log names
        log_names = set()
        for eval_cfg in self.evaluations:
            if eval_cfg.name_in_logs in log_names:
                raise ValueError(
                    f"Duplicate evaluation name_in_logs found: {eval_cfg.name}. These should be unique so we do not overwrite any logs"
                )
            log_names.add(eval_cfg.name_in_logs)

    def build(self):
        from ufo.training.trainer import Trainer

        return Trainer(self)
