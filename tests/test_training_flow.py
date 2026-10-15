"""Schedule and replay contracts for the default online training loop."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from ufo.agents.common.buffers.trajectory import ReplayBuffer, get_idxs
from ufo.train import build_mjlab_config
from ufo.training import trainer as current


def run_flow(module, path, *, resume=0, budget=24, world_size=1, buffer_interval=16, stop=False,
             eval_at_start=True, agent_family="fb"):
    events, rows, logs = [], [], []
    scale = world_size
    cfg = build_mjlab_config(
        device="cpu", work_dir=str(path), num_envs=2, num_env_steps=budget, seed=17, use_wandb=False, wandb_run_name=None,
        agent=agent_family,
    )
    cfg = cfg.model_copy(
        update={
            "num_seed_steps": 2,
            "update_agent_every": 2,
            "num_agent_updates": 2,
            "eval_every_steps": 8,
            "eval_at_start": eval_at_start,
            "checkpoint_every_steps": 8,
            "buffer_every_steps": buffer_interval,
            "log_every_steps": 4,
            "prioritization": True,
            "fail_on_nonfinite": True,
            "nonfinite_check_rollout_every_local_steps": 2,
            "disable_tqdm": True,
            "distributed_world_size": world_size,
        }
    )

    class Buffer:
        def __init__(self, **kwargs):
            pass

        def __len__(self):
            return len(rows)

        def can_sample(self):
            return len(rows) >= 2

        def extend(self, data):
            rows.append(copy.deepcopy(data))
            events.append(("collect", len(rows)))

    class Env:
        def __init__(self):
            self.tick = 0
            self.action_space = SimpleNamespace(sample=lambda: torch.rand(2, 1))
            self._env = SimpleNamespace(
                _motion_lib=SimpleNamespace(update_sampling_weight_by_id=lambda **kw: events.append(("motion_priorities",)))
            )

        def obs(self):
            return {"state": torch.full((2, 3), float(self.tick)), "time": torch.full((2, 1), self.tick)}

        def info(self):
            return {"qpos": torch.full((2, 8), float(self.tick)), "qvel": torch.full((2, 7), float(self.tick))}

        def reset(self, **kwargs):
            self.tick = 0
            events.append(("reset",))
            return self.obs(), self.info()

        def step(self, action, **kwargs):
            self.tick += 1
            return (
                self.obs(),
                action[:, 0],
                torch.tensor([self.tick % 3 == 0, False]),
                torch.tensor([False, self.tick % 2 == 0]),
                {**self.info(), "aux_rewards": {"term": action[:, 0]}},
            )

        def close(self):
            events.append(("close",))

    class Agent:
        device = "cpu"

        def __init__(self):
            self.cfg = cfg.agent
            self._model = torch.nn.Linear(3, 1)

        def maybe_update_rollout_context(self, **kwargs):
            return torch.rand(2, 2)

        def act(self, **kwargs):
            return torch.rand(2, 1)

        def update(self, buffer, step):
            events.append(("update", step))
            return {"loss": torch.rand(1)}

    expert = SimpleNamespace(
        motion_ids=["a", "b"], file_names=["a.pkl", "b.pkl"], update_priorities=lambda **kw: events.append(("expert_priorities",))
    )
    ws = module.Trainer.__new__(module.Trainer)
    ws.cfg = cfg
    ws.work_dir = Path(path)
    ws.work_dir.mkdir(parents=True, exist_ok=True)
    if stop:
        (ws.work_dir / "SAVE_AND_STOP").touch()
    ws.train_env = Env()
    ws.agent = Agent()
    ws.distributed_world_size = world_size
    ws.distributed_rank = 0
    ws._write_shared_artifacts = True
    ws._checkpoint_local_time = resume
    ws._checkpoint_global_time = resume * scale
    ws._optimizer_steps = 0
    ws._checkpoint_status = module._make_train_status(cfg, local_time=resume, global_time=resume * scale, optimizer_steps=0)
    ws.evaluations = {"tracking": module.TrackingEvaluation.__new__(module.TrackingEvaluation)}
    ws.evaluate = True
    ws.priorization_eval_name = "tracking"
    ws.train_logger = SimpleNamespace(log=lambda metrics: logs.append(copy.deepcopy(metrics)))

    def evaluate(t, **kwargs):
        events.append(("eval", t))
        return {"tracking": {"a": {"motion_id": "a", "emd": 1.0}, "b": {"motion_id": "b", "emd": 2.0}}}

    ws.eval = evaluate
    def save(**kwargs):
        status = module._make_train_status(
            cfg, local_time=kwargs["local_time"], global_time=kwargs["global_time"], optimizer_steps=kwargs["optimizer_steps"]
        )
        events.append(("save", kwargs["local_time"], kwargs["optimizer_steps"], status, kwargs["replay_mode"]))

    ws.save = save
    clock = iter(range(1, 10_000))
    with (
        patch.object(module, "load_expert_trajectories_from_motion_lib", return_value=expert),
        patch.object(module, "ReplayBuffer", Buffer),
        patch.object(module, "barrier"),
        patch.object(module, "broadcast_object", side_effect=lambda value, **kwargs: value),
        patch.object(module, "sync_model_buffers"),
        patch.object(module, "average_metrics", side_effect=lambda metrics: dict(metrics)),
        patch.object(module.time, "time", side_effect=lambda: float(next(clock))),
    ):
        torch.manual_seed(123)
        ws.train()
    return events, rows, torch.get_rng_state(), ws._optimizer_steps, logs


@pytest.mark.parametrize("family", ["fb", "tech"])
def test_training_replay_keeps_only_used_fields_and_fb_relabel_states(tmp_path, family):
    _, rows, _, _, _ = run_flow(current, tmp_path / "flow", agent_family=family)
    cfg = SimpleNamespace(aux_rewards=["term"])
    buffer = ReplayBuffer(capacity=16, n_dim=2, end_key="done",
                          output_key_t=current._trajectory_output_keys(cfg),
                          output_key_tp1=["observation", "terminated"])
    for row in rows:
        assert set(row) == {"observation", "action", "z", "done", "next", "aux_rewards"} | (
            {"qpos", "qvel"} if family == "fb" else set()
        )
        buffer.extend(row)
    buffer.save(tmp_path / "replay")
    loaded = ReplayBuffer.load(tmp_path / "replay")
    assert loaded.storage.keys() == rows[0].keys()
    loaded._get_idxs = get_idxs
    batch = loaded.sample(8)
    assert set(batch) == {"observation", "action", "z", "aux_rewards", "next"}
    assert set(batch["next"]) == {"observation", "terminated"}
    if family == "fb":
        loaded.output_key_tp1 += ["qpos", "qvel"]
        data = loaded.get_full_buffer()
        torch.testing.assert_close(data["next"]["qpos"][:, :1], data["next"]["observation"]["state"][:, :1])
        torch.testing.assert_close(data["next"]["qvel"][:, :1], data["next"]["observation"]["state"][:, :1])


def test_schedule_saves_before_eval_and_does_not_update_during_seed(tmp_path):
    events, rows, _, updates, _ = run_flow(current, tmp_path)
    assert [e[1] for e in events if e[0] == "eval"] == [0, 8, 16]
    assert [e[1] for e in events if e[0] == "save"] == [8, 16, 24]
    assert [e[1] for e in events if e[0] == "update"] == [t for t in range(4, 24, 2) for _ in range(2)]
    assert updates == 20
    assert len(rows) == 12
    for t in [8, 16]:
        assert next(i for i, e in enumerate(events) if e[:2] == ("save", t)) < events.index(("eval", t))
    assert events[-1] == ("close",)


def test_rp1_first_evaluation_at_interval(tmp_path):
    events, rows, _, updates, _ = run_flow(
        current, tmp_path, eval_at_start=False,
    )
    assert [event[1] for event in events if event[0] == "eval"] == [8, 16]
    assert len(rows) == 12 and updates == 20


def test_resume_and_incomplete_budget_do_not_collect_extra_samples(tmp_path):
    events, rows, _, _, _ = run_flow(current, tmp_path, resume=8, budget=15)
    assert len(rows) == 3
    assert [e[1] for e in events if e[0] == "eval"] == [8]
    assert [e[1] for e in events if e[0] == "save"] == [14]
    assert [e[-1] for e in events if e[0] == "save"] == ["rank0"]


@pytest.mark.parametrize("native_dict", [False, True])
def test_effective_config_records_restored_agent_and_resolved_task(tmp_path, native_dict):
    from omegaconf import OmegaConf

    cfg = build_mjlab_config(
        device="cpu",
        work_dir=str(tmp_path),
        num_envs=2,
        num_env_steps=24,
        seed=17,
        use_wandb=False,
        wandb_run_name=None,
    )
    restored = cfg.agent.model_copy(update={"train": cfg.agent.train.model_copy(update={"lr_actor": 0.0123})})
    ws = current.Trainer.__new__(current.Trainer)
    ws.cfg = cfg
    ws.agent = SimpleNamespace(cfg=restored)
    ws.work_dir = tmp_path
    ws.train_env_info = {"task_config": OmegaConf.create({"n": 2, "num_envs": "${n}"})}
    if native_dict:
        ws.train_env_info["task_config"] = {"n": 2, "num_envs": 2}
    ws._checkpoint_status = {"local_time": 8, "global_time": 8, "optimizer_steps": 2}
    ws._save_effective_config()
    result = json.loads((tmp_path / "effective_config.json").read_text())
    assert result["training"]["agent"]["train"]["lr_actor"] == cfg.agent.train.lr_actor
    assert result["agent"]["train"]["lr_actor"] == 0.0123
    assert result["task"]["num_envs"] == 2
    assert result["resume"]["optimizer_steps"] == 2
    assert result["step_accounting"]["num_envs_per_rank"] == 2


@pytest.mark.parametrize("world_size", [1, 2, 8])
def test_distributed_units_preserve_local_updates_and_global_budget(tmp_path, world_size):
    synchronized = world_size > 1
    scale = world_size
    # A resumed run with a non-divisible budget must stop before its next rollout.
    budget, resume = 24 * scale - 1, 8
    events, rows, _, updates, logs = run_flow(
        current, tmp_path, resume=resume, budget=budget, world_size=world_size
    )
    expected_local_steps = list(range(resume, budget // (2 * scale) * 2, 2))
    assert len(rows) == len(expected_local_steps)
    # No complete replay snapshot: all ranks rebuild replay before updating.
    expected_update_steps = [t for t in expected_local_steps if t > resume + 2]
    assert [e[1] for e in events if e[0] == "update"] == [t for t in expected_update_steps for _ in range(2)]
    assert updates == 2 * len(expected_update_steps)
    assert next(e[1] for e in events if e[0] == "eval") == resume * scale
    for event in (e for e in events if e[0] == "save"):
        status = event[3]
        assert status["global_time"] == event[1] * scale <= budget
        assert status["optimizer_steps"] == event[2]
        assert status["world_size"] == world_size
    assert logs
    for log in logs:
        assert log["timestep"] == log["distributed/global_env_steps"] == log["local_timestep"] * scale
        assert log["local_timestep"] == log["distributed/local_env_steps"]
        assert log["distributed/optimizer_steps"] <= updates
        if synchronized:
            assert log["distributed/loss_mode"] == "local_loss_average"
            assert log["distributed/global_parallel_envs"] == 2 * scale
    assert events[-1] == ("close",)


@pytest.mark.parametrize("world_size", [1, 8])
def test_effective_config_and_checkpoint_use_same_step_accounting(tmp_path, world_size):
    cfg = build_mjlab_config(
        device="cpu", work_dir=str(tmp_path), num_envs=2, num_env_steps=24, seed=17, use_wandb=False, wandb_run_name=None
    ).model_copy(update={"distributed_world_size": world_size})
    scale = world_size
    status = current._make_train_status(cfg, local_time=4, global_time=4 * scale, optimizer_steps=7)
    ws = current.Trainer.__new__(current.Trainer)
    ws.cfg, ws.work_dir = cfg, tmp_path
    ws.agent = SimpleNamespace(cfg=cfg.agent)
    ws.train_env_info = {"task_config": {"num_envs": 2}}
    ws._checkpoint_status = status
    ws._save_effective_config()
    accounting = json.loads((tmp_path / "effective_config.json").read_text())["step_accounting"]
    assert accounting["global_step_scale"] == scale
    assert accounting["num_envs_per_rank"] == 2
    assert accounting["global_parallel_envs"] == 2 * scale
    assert accounting["effective_batch_size"] == status["effective_batch_size"] == cfg.agent.train.batch_size * scale
    assert accounting["loss_mode"] == status["loss_mode"] == ("local_loss_average" if world_size > 1 else "single_rank")
    assert accounting["seed_steps_unit"] == accounting["update_interval_unit"] == "local_env_steps"
    assert accounting["budget_eval_checkpoint_log_unit"] == "global_env_steps"


def test_workspace_finite_errors_use_current_optimizer_count_and_original_path():
    ws = current.Trainer.__new__(current.Trainer)
    ws.distributed_rank = 3
    for optimizer_steps in (0, 7):
        ws._optimizer_steps = optimizer_steps
        with pytest.raises(FloatingPointError) as exc:
            ws._check_finite({"nested": [torch.tensor(float("nan"))]}, label="rollout.obs", local_time=4, global_time=32)
        message = str(exc.value)
        assert f"rank=3, local_time=4, global_time=32, optimizer_steps={optimizer_steps}" in message
        assert "rollout.obs.nested[0]" in message
        assert "bad_count=1" in message


def test_save_policy_separates_model_rank0_and_planned_stop(tmp_path):
    events, _, _, _, _ = run_flow(current, tmp_path / "normal", budget=25)
    assert [(e[1], e[-1]) for e in events if e[0] == "save"] == [(8, "none"), (16, "rank0"), (24, "rank0")]
    events, rows, _, _, _ = run_flow(current, tmp_path / "stop", stop=True)
    assert [(e[1], e[-1]) for e in events if e[0] == "save"] == [(8, "all")]
    assert len(rows) == 4
    assert not (tmp_path / "stop" / "SAVE_AND_STOP").exists()
    assert events[-1] == ("close",)


def test_buffer_interval_can_trigger_before_model_interval(tmp_path):
    events, _, _, _, _ = run_flow(current, tmp_path, budget=15, buffer_interval=6)
    assert [(e[1], e[-1]) for e in events if e[0] == "save"] == [(6, "rank0"), (12, "rank0"), (14, "rank0")]
