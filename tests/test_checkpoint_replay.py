"""Replay persistence and two-rank resume/gradient-entry contracts, without a simulator."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ufo.agents.common.buffers.trajectory import ReplayBuffer, get_idxs
from ufo.training import trainer


class TinyAgent:
    device = "cpu"

    def __init__(self):
        self._model = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self._model.weight.fill_(2)
            self._model.bias.fill_(3)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path / "tiny_model.pt")

    @classmethod
    def load(cls, path, *, device):
        result = cls()
        result._model.load_state_dict(torch.load(Path(path) / "tiny_model.pt", weights_only=True))
        return result


def make_workspace(path, rank=0, world_size=1, *, resume=0):
    ws = trainer.Trainer.__new__(trainer.Trainer)
    ws.work_dir = Path(path)
    ws.work_dir.mkdir(parents=True, exist_ok=True)
    ws.cfg = SimpleNamespace(
        agent=SimpleNamespace(
            name="FBcprAuxAgent",
            train=SimpleNamespace(batch_size=4), model=SimpleNamespace(device="cpu"),
            aux_rewards=[], object_class=TinyAgent, build=lambda **kwargs: TinyAgent(),
        ),
        buffer_device="cpu", buffer_size=16, online_parallel_envs=2,
        distributed_rank=rank, distributed_world_size=world_size,
        distributed_sync=world_size > 1,
        num_seed_steps=4, num_env_steps=72, update_agent_every=2,
        checkpoint_every_steps=100_000, buffer_every_steps=100_000,
        eval_every_steps=100_000, eval_at_start=True, log_every_steps=100_000,
        env=SimpleNamespace(task_overrides={}),
        disable_tqdm=True, fail_on_nonfinite=False,
    )
    ws.distributed_rank = rank
    ws.distributed_world_size = world_size
    ws._write_shared_artifacts = rank == 0
    ws._checkpoint_local_time = resume
    ws._checkpoint_global_time = resume * world_size
    ws._optimizer_steps = 3 if resume else 0
    ws._replay_update_after = ws.cfg.num_seed_steps
    ws._checkpoint_status = trainer._make_train_status(
        ws.cfg, local_time=resume, global_time=resume * world_size, optimizer_steps=ws._optimizer_steps,
    )
    ws.agent = TinyAgent()
    ws.train_env = SimpleNamespace(_env=object())
    ws.evaluate = False
    return ws


def row(value, *, truncated=False):
    scalar = torch.full((1, 2, 1), float(value))
    return {
        "observation": {"state": scalar}, "action": scalar, "z": scalar,
        "done": torch.full((1, 2, 1), truncated, dtype=torch.bool),
        "next": {"observation": {"state": scalar + 1},
                 "terminated": torch.zeros(1, 2, 1, dtype=torch.bool),
                 "truncated": torch.full((1, 2, 1), truncated, dtype=torch.bool),
                 "qpos": scalar + 1, "qvel": scalar + 1},
        "qpos": scalar, "qvel": scalar,
    }


def make_buffer(rank=0, *, rows=5, truncated=False):
    buffer = ReplayBuffer(
        capacity=8, n_dim=2, end_key="done", output_key_t=["observation", "action"],
        output_key_tp1=["observation", "terminated"],
    )
    for i in range(rows):
        buffer.extend(row(100 * rank + i, truncated=truncated))
    buffer._get_idxs = get_idxs
    return buffer


def reload_buffers(ws):
    ws.agent, ws._checkpoint_status = trainer.create_agent_or_load_checkpoint(ws.work_dir, ws.cfg, {})
    ws._checkpoint_local_time = ws._checkpoint_status["local_time"]
    ws._checkpoint_global_time = ws._checkpoint_status["global_time"]
    ws._optimizer_steps = ws._checkpoint_status["optimizer_steps"]
    with patch.object(trainer, "load_expert_trajectories_from_motion_lib", return_value=object()):
        return ws._prepare_replay_buffers()["train"]


@pytest.mark.parametrize("family", ["FBcprAuxAgent", "TeCHAgent"])
def test_resume_removes_unused_replay_fields_without_changing_samples(tmp_path, family):
    ws = make_workspace(tmp_path)
    ws.cfg.agent.name = family
    buffer = make_buffer()
    buffer.storage["reward"] = torch.zeros(8, 2, 1)
    buffer.storage["step_count"] = torch.zeros(8, 2, 1, dtype=torch.int64)
    buffer.output_key_t += ["reward", "step_count", "done"]
    save(ws, buffer, "all")
    restored = reload_buffers(ws)
    assert not {"reward", "step_count"}.intersection(restored.storage)
    assert ("qpos" in restored.storage) == (family == "FBcprAuxAgent")
    assert ("qvel" in restored.storage) == (family == "FBcprAuxAgent")
    assert restored.output_key_t == ["observation", "action", "z"]
    for key in ["action", "z"]:
        torch.testing.assert_close(restored.storage[key], buffer.storage[key], rtol=0, atol=0)
    assert restored._idx == buffer._idx
    restored._get_idxs = get_idxs
    assert "terminated" in restored.sample(4)["next"]


def save(ws, buffer, mode, *, local_time=20):
    ws.save(
        local_time=local_time, global_time=local_time * ws.distributed_world_size,
        optimizer_steps=3, replay_buffer={"train": buffer}, replay_mode=mode,
    )


def _assert_empty_resume(ws):
    buffer = reload_buffers(ws)
    assert len(buffer) == 0
    assert ws._replay_update_after == ws._checkpoint_local_time + ws.cfg.num_seed_steps
    assert ws._optimizer_steps == 3
    assert not ws._replay_ready


def _distributed_worker(rank, root, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        root = Path(root)
        for mode in ["none", "rank0", "all"]:
            ws = make_workspace(root / mode, rank, 2)
            buffer = make_buffer(rank, rows=9)  # Wrapped cursor exercises reset boundary at index 0, not len - 1.
            save(ws, buffer, mode)
            checkpoint = ws.work_dir / "checkpoint"
            status = json.loads((checkpoint / "train_status.json").read_text())
            assert not (checkpoint / ".saving").exists()
            assert (checkpoint / "tiny_model.pt").is_file()
            for shard_rank in range(2):
                shard = checkpoint / "buffers" / f"train_rank_{shard_rank}"
                expected = mode == "all" or (mode == "rank0" and shard_rank == 0)
                assert (shard / "buffer.hdf5").exists() is expected
                if expected:
                    snapshot = json.loads((shard / "snapshot.json").read_text())
                    assert snapshot["global_time"] == 40 and snapshot["world_size"] == 2
                    if mode == "all":
                        assert snapshot["id"] == status["replay_id"]
            if mode != "all":
                assert status.get("replay_id") is None
                _assert_empty_resume(ws)
            else:
                restored = reload_buffers(ws)
                assert len(restored) == len(buffer)
                assert restored._idx == buffer._idx == 1
                assert ws._replay_update_after == ws.cfg.num_seed_steps
                torch.testing.assert_close(restored.storage["observation"]["state"], buffer.storage["observation"]["state"])
                assert restored.storage["done"][0].all()
                assert not restored.storage["done"][1:].any()
                assert restored.can_sample()
            dist.barrier()

        for corruption in ["missing_shard", "missing_config", "missing_snapshot", "stale_id", "stale_step", "wrong_world", "model_advanced"]:
            ws = make_workspace(root / corruption, rank, 2)
            buffer = make_buffer(rank)
            save(ws, buffer, "all")
            shard = ws.work_dir / "checkpoint" / "buffers" / "train_rank_1"
            if corruption == "model_advanced":
                # Old complete shard directories must never override the new model-only checkpoint.
                before = {name: (shard / name).read_bytes() for name in ("buffer.hdf5", "snapshot.json")}
                save(ws, buffer, "none", local_time=22)
                assert before == {name: (shard / name).read_bytes() for name in before}
            elif rank == 0:
                snapshot_path = shard / "snapshot.json"
                if corruption == "missing_shard":
                    (shard / "buffer.hdf5").unlink()
                elif corruption == "missing_config":
                    (shard / "config.json").unlink()
                elif corruption == "missing_snapshot":
                    snapshot_path.unlink()
                else:
                    snapshot = json.loads(snapshot_path.read_text())
                    key, value = {
                        "stale_id": ("id", "different-save"),
                        "stale_step": ("global_time", 39),
                        "wrong_world": ("world_size", 8),
                    }[corruption]
                    snapshot[key] = value
                    snapshot_path.write_text(json.dumps(snapshot))
            dist.barrier()
            _assert_empty_resume(ws)
            dist.barrier()

        _distributed_train_gate(rank, root / "gate")
        _distributed_train_gate(rank, root / "planned_stop", planned_stop=True)
    finally:
        dist.destroy_process_group()


def _distributed_train_gate(rank, path, *, planned_stop=False):
    ws = make_workspace(path, rank, 2, resume=20)
    collected, updates = [], []
    if planned_stop:
        ws.cfg.checkpoint_every_steps = 16
        if rank == 0:
            (path / "SAVE_AND_STOP").touch()
        dist.barrier()

    def reset(**kwargs):
        return {"state": torch.zeros(2, 1)}, {}

    ws.train_env.reset = reset
    ws.train_env.close = lambda: None

    def collect(td, info, context, terminated, truncated, replay_buffer, local_time, global_time, eval_time_checker):
        # One-step episodes now have complete successors and are eligible on both ranks.
        replay_buffer["train"].extend(row(local_time, truncated=rank == 1 and local_time < 28))
        collected.append(local_time)
        return td, info, terminated, truncated, context

    def update(replay_buffer, local_time, global_time, total_metrics, counts):
        replay_buffer["train"]._get_idxs = get_idxs
        replay_buffer["train"].sample(4)
        assert local_time >= 26  # Updates resume once the warmup budget is collected.
        gradient = torch.tensor([float(rank + 1)])
        dist.all_reduce(gradient)
        assert gradient.item() == 3
        updates.append(local_time)
        ws._optimizer_steps += 1
        return {"loss": torch.ones(())}, {"loss": 1}

    ws._collect_transition = collect
    ws._update_agent = update
    with patch.object(trainer, "load_expert_trajectories_from_motion_lib", return_value=object()):
        ws.train()
    status = json.loads((path / "checkpoint" / "train_status.json").read_text())
    if planned_stop:
        assert collected == [20, 22, 24, 26] and updates == [26]
        assert status["global_time"] == 56 and status["optimizer_steps"] == 4
        assert not (path / "SAVE_AND_STOP").exists()
        assert len(reload_buffers(ws)) == 4
        assert ws._checkpoint_status["replay_id"]
    else:
        assert collected == list(range(20, 36, 2))
        assert updates == [26, 28, 30, 32, 34]
        assert ws._optimizer_steps == 8
        assert status["global_time"] == 72 and status["optimizer_steps"] == 8
    dist.barrier()


def test_two_rank_checkpoint_and_resume(tmp_path, monkeypatch):
    # These subprocesses deliberately exercise CPU collectives even on a GPU host.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    mp.spawn(_distributed_worker, args=(str(tmp_path), (tmp_path / "rendezvous").as_uri()), nprocs=2, join=True)


def test_single_rank_inference_snapshot_is_a_complete_resume(tmp_path):
    ws = make_workspace(tmp_path)
    buffer = make_buffer()
    save(ws, buffer, "rank0")
    restored = reload_buffers(ws)
    assert len(restored) == len(buffer)
    assert ws._checkpoint_status["replay_id"]
    assert ws._replay_update_after == ws.cfg.num_seed_steps
    assert restored.storage["done"][4].all()


@pytest.mark.parametrize("existing_checkpoint", [False, True])
def test_failed_model_write_cannot_be_loaded_as_a_checkpoint(tmp_path, existing_checkpoint):
    ws = make_workspace(tmp_path)
    status_path = tmp_path / "checkpoint" / "train_status.json"
    if existing_checkpoint:
        save(ws, make_buffer(), "all")
    old_status = status_path.read_bytes() if existing_checkpoint else None

    def fail(path):
        (Path(path) / "tiny_model.pt").write_bytes(b"partial checkpoint")
        raise OSError("injected disk failure")

    ws.agent.save = fail
    with pytest.raises(OSError, match="injected disk failure"):
        save(ws, make_buffer(), "none", local_time=22)
    checkpoint = tmp_path / "checkpoint"
    assert (checkpoint / ".saving").exists()
    assert (status_path.read_bytes() if status_path.exists() else None) == old_status
    with pytest.raises(RuntimeError, match="(?i)(incomplete|unfinished|interrupted|saving)"):
        trainer.create_agent_or_load_checkpoint(tmp_path, ws.cfg, {})


def test_readiness_requires_a_valid_transition_and_does_not_consume_rng():
    buffer = make_buffer(rows=5, truncated=True)
    rng = torch.get_rng_state()
    assert buffer.can_sample()
    buffer.extend(row(5))
    assert buffer.can_sample()
    buffer.extend(row(6))
    assert buffer.can_sample()
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    sample = buffer.sample(4)
    assert torch.equal(sample["next"]["observation"]["state"], sample["observation"]["state"] + 1)


def test_save_during_resume_warmup_preserves_remaining_warmup(tmp_path):
    ws = make_workspace(tmp_path, resume=20)
    with patch.object(trainer, "load_expert_trajectories_from_motion_lib", return_value=object()):
        ws._prepare_replay_buffers()
    assert ws._replay_update_after == 24
    save(ws, make_buffer(rows=2), "all", local_time=22)
    restored = reload_buffers(ws)
    assert len(restored) == 2
    assert ws._replay_update_after == 24


@pytest.mark.parametrize("during_load", [False, True])
def test_inference_rejects_incomplete_model_save(tmp_path, during_load):
    from ufo.agents.common.checkpoint import load_model_from_checkpoint_dir
    from ufo.agents.fb_cpr_aux.model import FBcprAuxModel

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"name": "FBcprAuxModel"}))

    def load(*args, **kwargs):
        (tmp_path / ".saving").touch()
        return object()

    if not during_load:
        (tmp_path / ".saving").touch()
    with patch.object(FBcprAuxModel, "load", side_effect=load):
        with pytest.raises(RuntimeError, match="(?i)(incomplete|save started)"):
            load_model_from_checkpoint_dir(tmp_path)
