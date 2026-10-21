from __future__ import annotations

import copy
import csv
import os
from datetime import timedelta
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ufo.envs.core import MJLabCore
from ufo.training import evaluation as tracking
from ufo.training.distributed import average_gradients, broadcast_object, sync_model_buffers
from ufo.training.logging import CSVLogger


def _metric(motion_id):
    return {"motion_id": motion_id, "emd": float(motion_id**2), "distance": float(motion_id)}


def _evaluation(num_motions, repetitions=1):
    cfg = tracking.TrackingEvaluationConfig(n_episodes_per_motion=repetitions)
    motion_lib = SimpleNamespace(_num_unique_motions=num_motions, load_motions_for_training=Mock())
    core = SimpleNamespace(_motion_lib=motion_lib, num_envs=2, reset_all=Mock(), is_evaluating=False)
    core.resample_motion = MethodType(MJLabCore.resample_motion, core)
    core.set_is_training = Mock(side_effect=MethodType(MJLabCore.set_is_training, core))
    return cfg.build(), SimpleNamespace(_env=core, close=Mock())


def _worker(motion_ids, agent, **kwargs):
    assert kwargs["env"]._env.is_evaluating
    return {f"motion_{motion_id}": _metric(motion_id) for motion_id in motion_ids}


@pytest.mark.parametrize("num_motions", [0, 3, 8, 17, 1025])
def test_eight_rank_partition_covers_each_motion_once(num_motions):
    shards = [tracking.distributed_motion_ids(num_motions, rank, 8) for rank in range(8)]
    assert sorted(motion_id for shard in shards for motion_id in shard) == list(range(num_motions))
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


@pytest.mark.parametrize("failure", ["missing", "duplicate", "wrong_rank", "extra", "repetition"])
def test_merge_rejects_incomplete_or_duplicate_results(failure):
    shards = [{"a": _metric(0)}, {"b": _metric(1)}]
    if failure == "missing":
        shards[1].clear()
    elif failure == "duplicate":
        shards[1] = {"a": _metric(1)}
    elif failure == "wrong_rank":
        shards.reverse()
    elif failure == "extra":
        shards[0]["c"] = _metric(2)
    else:
        shards[0]["a_again"] = _metric(0)
    with pytest.raises(RuntimeError):
        tracking.merge_tracking_metrics(
            shards,
            num_motions=2,
            n_episodes_per_motion=1,
            include_results_from_all_envs=False,
        )


def test_single_rank_preserves_metrics_and_restores_training_environment():
    evaluation, env = _evaluation(5)
    logger = Mock(fields=None)
    with patch.object(tracking, "_evaluate_motion_batch", side_effect=_worker):
        metrics, summary = evaluation.run(timestep=42, agent_or_model=None, env=env, logger=logger)
    assert len(metrics) == 5
    assert summary["emd"] == np.mean([i**2 for i in range(5)])
    assert summary["emd#std"] == np.std([i**2 for i in range(5)])
    logger.log.assert_called_once()
    assert logger.log.call_args.args[0]["motion_name"] == list(metrics)
    assert logger.log.call_args.args[0]["timestep"] == [42] * 5
    env._env._motion_lib.load_motions_for_training.assert_called_once()
    env._env.set_is_training.assert_called_once()
    env.close.assert_not_called()


@pytest.mark.parametrize("existing_fields", [False, True])
def test_evaluation_batches_csv_without_changing_rows_or_fields(tmp_path, existing_fields):
    reference = CSVLogger(tmp_path / "scalar.csv")
    logger = CSVLogger(tmp_path / "batch.csv")
    if existing_fields:
        initial = {
            "motion_id": -1, "distance": 1.0, "label": "initial", "optional": "", "motion_name": "initial", "timestep": 0
        }
        reference.log(initial)
        logger.log(initial)
    rounds = [
        {
            "clip,一": {"motion_id": 0, "distance": 0.125, "label": "walk\nleft", "optional": "present"},
            'clip"二': {"motion_id": 1, "distance": np.nan, "label": "run", "later_only": 42},
        },
        {
            "clip,一": {"motion_id": 0, "distance": 0.25, "label": "walk"},
            'clip"二': {"motion_id": 1, "distance": 0.5, "label": "run", "optional": "restored"},
        },
    ]
    evaluation, env = _evaluation(2)
    for timestep, metrics in enumerate(rounds, start=42):
        for name, metric in metrics.items():
            reference.log({**metric, "motion_name": name, "timestep": timestep})
        with patch.object(tracking, "_evaluate_motion_batch", return_value=copy.deepcopy(metrics)), patch.object(
            logger, "log", wraps=logger.log
        ) as log:
            result, _ = evaluation.run(timestep=timestep, agent_or_model=None, env=env, logger=logger)
        log.assert_called_once()
        assert list(result) == list(metrics)
        assert all(metric["motion_name"] == name and metric["timestep"] == timestep for name, metric in result.items())
    with reference.filename.open(newline="") as scalar, logger.filename.open(newline="") as batch:
        assert list(csv.reader(batch)) == list(csv.reader(scalar))
    assert logger.fields == reference.fields


def test_empty_evaluation_does_not_write_csv(tmp_path):
    evaluation, env = _evaluation(0)
    logger = CSVLogger(tmp_path / "empty.csv")
    metrics, summary = evaluation.run(timestep=42, agent_or_model=None, env=env, logger=logger)
    assert metrics == summary == {}
    assert logger.fields is None
    assert not logger.filename.exists()
    env._env._motion_lib.load_motions_for_training.assert_called_once()
    env._env.set_is_training.assert_called_once()


def _distributed_worker(rank, init_method, backend):
    if backend == "nccl":
        torch.cuda.set_device(rank)
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=8, timeout=timedelta(seconds=120))
    try:
        # Compare bucketed synchronization with the original per-tensor SUM /
        # world-size operation, including mixed types and noncontiguous grads.
        device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
        torch.manual_seed(123 + rank)
        parameters = []
        expected = []
        for shape, dtype in [
            ((3, 5), torch.float32),
            ((5, 3), torch.float32),
            ((2, 3), torch.float64),
            ((4096,), torch.float32),
            ((131072,), torch.float32),
            ((131072,), torch.float32),
        ]:
            param = torch.nn.Parameter(torch.empty(shape, dtype=dtype, device=device))
            gradient = torch.randn(shape, dtype=dtype, device=device)
            if len(shape) == 2:
                gradient = gradient.t().contiguous().t()
            param.grad = gradient
            reference = gradient.contiguous().clone()
            dist.all_reduce(reference)
            reference.mul_(1.0 / 8)
            parameters.append(param)
            expected.append(reference)
        unused = torch.nn.Parameter(torch.ones(1, device=device))
        original = [param.grad.clone() for param in parameters]
        cache = {}
        for bucket_size in [256, 128 * 1024 * 1024, 256]:
            for param, gradient in zip(parameters, original):
                param.grad.copy_(gradient)
            average_gradients([parameters[0], unused, *parameters[1:]], bucket_size_bytes=bucket_size, bucket_cache=cache)
            assert unused.grad is None
            for param, reference in zip(parameters, expected):
                torch.testing.assert_close(param.grad, reference, rtol=1e-6, atol=1e-7)

        # Packed model buffers must retain per-tensor SUM/world-size for floats
        # and rank-zero broadcast for counters, including noncontiguous views.
        module = torch.nn.Module()
        for i, (shape, dtype) in enumerate([((64,), torch.float32), ((463,), torch.float32), ((3, 2), torch.float64), ((), torch.int64), ((5,), torch.int64)]):
            value = torch.randn(shape, device=device).to(dtype) if dtype.is_floating_point else torch.full(shape, rank + 1, device=device, dtype=dtype)
            if len(shape) == 2:
                value = value.t().contiguous().t()
            module.register_buffer(f"buffer_{i}", value)
        addresses = [b.data_ptr() for b in module.buffers()]
        for _ in range(3):
            for value in module.buffers():
                value.add_(rank)
            references = [b.contiguous().clone() for b in module.buffers()]
            for value in references:
                if value.is_floating_point():
                    dist.all_reduce(value)
                    value.mul_(1.0 / 8)
                else:
                    dist.broadcast(value, src=0)
            sync_model_buffers(module)
            for value, expected in zip(module.buffers(), references):
                # Packing changes the collective's reduction order: allow only
                # eight-rank floating-point summation roundoff, never counter drift.
                tolerance = 8 * torch.finfo(value.dtype).eps if value.is_floating_point() else 0
                torch.testing.assert_close(value, expected, rtol=tolerance, atol=tolerance)
                reference_rank0 = value.contiguous().clone()
                dist.broadcast(reference_rank0, src=0)
                torch.testing.assert_close(value, reference_rank0, rtol=0, atol=0)
            assert [b.data_ptr() for b in module.buffers()] == addresses

        # Uneven shards, empty ranks, and repeated episodes all use the actual
        # process-group gather/broadcast, mocking only the simulator rollout.
        for num_motions, repetitions in [(17, 1), (3, 1), (17, 2)]:
            evaluation, env = _evaluation(num_motions, repetitions)
            logger = Mock(fields=None)
            with patch.object(tracking, "_evaluate_motion_batch", side_effect=_worker) as worker:
                metrics, summary = evaluation.run(
                    timestep=42,
                    agent_or_model=None,
                    env=env,
                    logger=logger,
                    distributed=True,
                )
            assigned = [motion_id for call in worker.call_args_list for motion_id in call.args[0]]
            assert sorted(assigned) == sorted(list(range(rank, num_motions, 8)) * repetitions)
            env._env._motion_lib.load_motions_for_training.assert_called_once()
            env._env.set_is_training.assert_called_once()
            env.close.assert_not_called()
            if rank == 0:
                assert len(metrics) == num_motions * repetitions
                expected = [i**2 for i in range(num_motions)] * repetitions
                assert summary["emd"] == np.mean(expected)
                assert summary["emd#std"] == np.std(expected)
                logger.log.assert_called_once()
                assert logger.log.call_args.args[0]["motion_name"] == list(metrics)
                assert logger.log.call_args.args[0]["timestep"] == [42] * len(metrics)
            else:
                assert metrics == {} and summary is None
                logger.log.assert_not_called()
            # Exercise the subsequent priority broadcast on the same group.
            priorities = broadcast_object([metrics[k]["emd"] for k in metrics] if rank == 0 else None, src=0)
            assert len(priorities) == num_motions * repetitions

        evaluation, env = _evaluation(17)

        def missing_worker(motion_ids, agent, **kwargs):
            return {} if rank == 0 else _worker(motion_ids, agent, **kwargs)

        with patch.object(tracking, "_evaluate_motion_batch", side_effect=missing_worker):
            with pytest.raises(RuntimeError, match="missed motion IDs"):
                evaluation.run(timestep=0, agent_or_model=None, env=env, logger=None, distributed=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_eight_process_evaluation(tmp_path):
    backend = os.environ.get("UFO_TEST_DISTRIBUTED_BACKEND", "gloo")
    mp.spawn(_distributed_worker, args=((tmp_path / "rendezvous").as_uri(), backend), nprocs=8, join=True)


def test_tracking_worker_uses_current_vector_env_without_legacy_context():
    core = SimpleNamespace(
        num_envs=2,
        num_dof=1,
        device="cpu",
        dof_pos=torch.zeros(2, 1),
        dt=0.02,
        mjlab_env=SimpleNamespace(cfg=SimpleNamespace(episode_length_s=10.0)),
        _motion_lib=SimpleNamespace(all_motions_loaded=True, curr_motion_keys=["motion"]),
    )

    class Env:
        include_last_action = False
        _env = core

        def reset(self, *, target_states, to_numpy):
            core.dof_pos.copy_(target_states["dof_states"][..., 0])
            return {"state": core.dof_pos.clone()}, {}

        def step(self, action, *, to_numpy):
            return {"state": core.dof_pos.clone()}, torch.zeros(2), torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}

    model = SimpleNamespace(tracking_inference=lambda obs: obs["state"])
    agent = SimpleNamespace(_model=model, act=lambda obs, context, mean: torch.zeros(2, 1))
    reference = {
        "dof_pos": torch.zeros(3, 1),
        "ref_dof_vel": torch.zeros(3, 1),
        "ref_body_pos": torch.zeros(3, 1, 3),
        "ref_body_rots": torch.zeros(3, 1, 4),
        "ref_body_vels": torch.zeros(3, 1, 3),
        "ref_body_angular_vels": torch.zeros(3, 1, 3),
    }
    with patch.object(tracking, "get_backward_observation", return_value=({"state": torch.zeros(3, 1)}, reference)):
        metrics = tracking._evaluate_motion_batch([0], agent, Env(), disable_tqdm=True)
    assert metrics["motion"]["motion_id"] == 0
    assert metrics["motion"]["emd"] == 0.0


def test_evaluation_restores_training_mode_after_failure():
    evaluation, env = _evaluation(1)
    def fail(*args, **kwargs):
        assert env._env.is_evaluating
        raise RuntimeError("injected evaluation failure")
    with patch.object(tracking, "_evaluate_motion_batch", side_effect=fail):
        with pytest.raises(RuntimeError, match="injected"):
            evaluation.run(timestep=0, agent_or_model=None, env=env, logger=None)
    assert not env._env.is_evaluating
    env._env.set_is_training.assert_called_once()


@pytest.mark.parametrize("unexpected_reset", [False, True])
def test_long_tracking_extends_timeout_and_rejects_reset_states(unexpected_reset):
    count = 551
    core = SimpleNamespace(num_envs=1, num_dof=1, device="cpu", dt=0.02, dof_pos=torch.zeros(1, 1),
        mjlab_env=SimpleNamespace(cfg=SimpleNamespace(episode_length_s=10.0)),
        _motion_lib=SimpleNamespace(all_motions_loaded=True, curr_motion_keys=["motion"]))
    class Env:
        include_last_action = False
        _env = core
        step_count = 0
        def reset(self, **kwargs):
            assert core.mjlab_env.cfg.episode_length_s > (count - 1) * core.dt
            return {"state": core.dof_pos.clone()}, {}
        def step(self, action, **kwargs):
            self.step_count += 1
            timeout = self.step_count * core.dt >= core.mjlab_env.cfg.episode_length_s
            done = timeout or (unexpected_reset and self.step_count == 500)
            core.dof_pos.fill_(-100 if done else self.step_count)
            return {"state": core.dof_pos.clone()}, torch.zeros(1), torch.tensor([False]), torch.tensor([done]), {}
    env = Env()
    agent = SimpleNamespace(_model=SimpleNamespace(tracking_inference=lambda obs: obs["state"]),
                            act=lambda *args, **kwargs: torch.zeros(1, 1))
    reference = {"dof_pos": torch.arange(count).float().view(count, 1), "ref_dof_vel": torch.zeros(count, 1),
                 "ref_body_pos": torch.zeros(count, 1, 3), "ref_body_rots": torch.zeros(count, 1, 4),
                 "ref_body_vels": torch.zeros(count, 1, 3), "ref_body_angular_vels": torch.zeros(count, 1, 3)}
    captured = []
    def metric(ep):
        captured.append(ep["joint_pos"])
        return {"motion": {"motion_id": 0, "emd": 0.0}}
    with patch.object(tracking, "get_backward_observation", return_value=({"state": reference["dof_pos"]}, reference)), \
         patch.object(tracking, "_calc_metrics", side_effect=metric):
        if unexpected_reset:
            with pytest.raises(RuntimeError, match="refusing invalid metrics"):
                tracking._evaluate_motion_batch([0], agent, env, disable_tqdm=True)
            assert not captured
        else:
            tracking._evaluate_motion_batch([0], agent, env, disable_tqdm=True)
            torch.testing.assert_close(captured[0], reference["dof_pos"])
    assert core.mjlab_env.cfg.episode_length_s == 10.0
