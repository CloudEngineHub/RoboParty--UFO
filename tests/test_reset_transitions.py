"""Same-step reset must preserve the successor, history and episode boundary."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils._pytree import tree_map

from ufo.agents.common.buffers.trajectory import ReplayBuffer, get_idxs
from ufo.envs.core import MJLabCore
from ufo.envs.observations import ObservationHistory
from ufo.envs.vector_env import MJLabVectorEnv
from ufo.training.trainer import Trainer


def make_env():
    core = MJLabCore.__new__(MJLabCore)
    core.config = OmegaConf.create({"obs": {"obs_scales": {}, "noise_scales": {},
                                           "obs_auxiliary": {"history_actor": {"dof_pos": 2}}}})
    core.history_handler = ObservationHistory(2, core.config.obs.obs_auxiliary, {"dof_pos": 1}, "cpu")
    core.is_evaluating = False
    core.add_noise_curriculum = False
    core._compute_privileged_obs = lambda: None
    core.actions = torch.zeros(2, 1)
    core.dof_pos = torch.tensor([[10.0], [20.0]])
    core.default_dof_pos = core.default_dof_pos_offset = torch.zeros(2, 1)
    core.dof_vel = torch.zeros(2, 1)
    core.base_ang_vel = core.projected_gravity = torch.zeros(2, 3)
    core._max_local_self = torch.zeros(2, 4)
    core.mjlab_env = SimpleNamespace(episode_length_buf=torch.zeros(2, dtype=torch.long), close=lambda: None)
    timeouts = torch.tensor([True, False])

    def advance(action):
        core.dof_pos.add_(1)
        core.actions.copy_(action)
        core.episode_length_buf.add_(1)
        return None, torch.tensor([3.0, 4.0]), timeouts.clone(), {
            "time_outs": timeouts.clone(), "aux_rewards": {"term": torch.tensor([5.0, 6.0])}}

    def reset(ids):
        core.dof_pos[ids] = 100
        core.actions[ids] = 0
        core.episode_length_buf[ids] = 0
        core.history_handler.reset(ids)
        timeouts.zero_()

    core.step, core.reset_finished = advance, reset
    env = MJLabVectorEnv.__new__(MJLabVectorEnv)
    env._env = core
    env.include_last_action = env.include_history_actor = True
    env._get_qpos_qvel = lambda **kwargs: (core.dof_pos.clone(), core.dof_pos.clone() + 1000)
    return env


@pytest.mark.parametrize("numpy", [False, True])
def test_partial_reset_keeps_final_state_and_updates_other_history_once(numpy):
    env = make_env()
    env.base_env.get_observation(to_numpy=False)
    obs, reward, terminated, truncated, info = env.step(torch.full((2, 1), 7.0), to_numpy=numpy)
    assert obs["state"][:, 0].tolist() == [100, 21]
    assert info["final_observation"]["state"][:, 0].tolist() == [11]
    assert info["final_observation"]["last_action"].tolist() == [[7]]
    assert info["final_observation"]["history_actor"].tolist() == [[10, 0]]
    assert obs["history_actor"].tolist() == [[0, 0], [20, 0]]
    assert info["final_qpos"].tolist() == [[11]]
    assert info["final_qvel"].tolist() == [[1011]]
    assert terminated.tolist() == [False, False] and truncated.tolist() == [True, False]
    following, _, _, _, _ = env.step(torch.ones(2, 1), to_numpy=False)
    assert following["history_actor"].tolist() == [[100, 0], [21, 20]]
    assert info["final_observation"]["state"][:, 0].tolist() == [11]


@pytest.mark.parametrize("true_termination", [False, True])
def test_collector_preserves_timeout_successor_and_first_reset_action(true_termination):
    env = make_env()
    if true_termination:
        advance = env.base_env.step
        def terminate(action):
            obs, reward, reset, info = advance(action)
            info["time_outs"].zero_()
            return obs, reward, reset, info
        env.base_env.step = terminate
    obs = env.base_env.get_observation(to_numpy=False)
    qpos, qvel = env._get_qpos_qvel()
    info = {"qpos": qpos, "qvel": qvel}
    replay = ReplayBuffer(capacity=8, n_dim=2, output_key_t=["observation", "action", "z", "aux_rewards"],
                          output_key_tp1=["observation", "terminated", "qpos", "qvel"])
    replay._get_idxs = get_idxs
    trainer = Trainer.__new__(Trainer)
    trainer.train_env = env
    trainer.agent = SimpleNamespace(device="cpu", maybe_update_rollout_context=lambda **kw: torch.ones(2, 1),
                                    act=lambda **kw: torch.full((2, 1), 7.0))
    trainer.cfg = SimpleNamespace(num_seed_steps=0, fail_on_nonfinite=False, online_parallel_envs=2,
                                  distributed_world_size=1, agent=SimpleNamespace(name="FBcprAuxAgent"))
    trainer.evaluate = False
    flags = torch.zeros(2, dtype=torch.bool)
    for step in range(2):
        obs, info, terminated, truncated, context = trainer._collect_transition(
            obs, info, None, flags, flags, {"train": replay}, step * 2, step * 2, None)
    full = replay.get_full_buffer()
    pairs = set(zip(full["observation"]["state"][:, 0].tolist(), full["next"]["observation"]["state"][:, 0].tolist()))
    assert pairs == {(10, 11), (100, 101), (20, 21), (21, 22)}
    assert replay.storage["done"][:2, :, 0].tolist() == [[True, False], [False, False]]
    assert replay.storage["next"]["truncated"][:2, :, 0].tolist() == [[not true_termination, False], [False, False]]
    assert replay.storage["next"]["terminated"][:2, :, 0].tolist() == [[true_termination, False], [False, False]]
    torch.testing.assert_close(full["next"]["qpos"][:, 0], full["next"]["observation"]["state"][:, 0])
    torch.testing.assert_close(full["next"]["qvel"][:, 0] - 1000, full["next"]["qpos"][:, 0])
    sampled = replay.sample(4096)
    assert set(zip(sampled["observation"]["state"][:, 0].tolist(), sampled["next"]["observation"]["state"][:, 0].tolist())) == pairs


@pytest.mark.parametrize("n_dim,lanes", [(1, 1), (2, 1), (2, 3)])
def test_explicit_successors_survive_wrap_sequences_and_save(tmp_path, n_dim, lanes):
    replay = ReplayBuffer(capacity=5, n_dim=n_dim, output_key_tp1=["observation", "terminated"])
    replay._get_idxs = get_idxs
    for step in range(8):
        value = torch.tensor([step * 10 + lane for lane in range(lanes)]).reshape(1, lanes, 1)
        flags = torch.full_like(value, step % 3 == 0, dtype=torch.bool)
        row = {"observation": value, "done": flags,
               "next": {"observation": value + 1, "terminated": flags}}
        if n_dim == 1:
            row = tree_map(lambda x: x[:, 0], row)
        replay.extend(row)
        assert replay.can_sample()
    full = replay.get_full_buffer()
    assert set(full["observation"].flatten().tolist()) == {step * 10 + lane for step in range(3, 8) for lane in range(lanes)}
    assert torch.equal(full["next"]["observation"], full["observation"] + 1)
    sampled = replay.sample(4096)
    assert set(sampled["observation"].flatten().tolist()) == set(full["observation"].flatten().tolist())
    sequence = replay.sample(4096, seq_length=2)
    values = sequence["observation"].reshape(-1, 2)
    assert ((values[:, 1] - values[:, 0]) == 10).all()
    assert not sequence["next"]["terminated"].reshape(-1, 2)[:, 0].any()
    replay.save(tmp_path)
    restored = ReplayBuffer.load(tmp_path)
    assert torch.equal(restored.get_full_buffer()["next"]["observation"], full["next"]["observation"])


def test_sampler_can_select_last_legal_start():
    from unittest.mock import patch
    kwargs = dict(seq_length=1, num_slices=1, lengths=torch.tensor([4]), start_idx=torch.tensor([[0]]), storage_length=8)
    with patch("torch.rand", return_value=torch.tensor([0.999])):
        assert get_idxs(**kwargs, offset=1).item() == 2
        assert get_idxs(**kwargs, offset=0).item() == 3


def test_real_mjlab_timeout_preserves_state_and_other_lane_history(tmp_path):
    import os
    if os.environ.get("UFO_TEST_RESET_GPU") != "1" or not torch.cuda.is_available():
        pytest.skip("requires an available GPU and UFO_TEST_RESET_GPU=1")
    from ufo.train import build_mjlab_config
    cfg = build_mjlab_config(device="cuda:0", work_dir=str(tmp_path), num_envs=4, num_env_steps=16,
                            seed=4728, use_wandb=False, wandb_run_name=None, disable_obs_noise=True,
                            data_path=os.environ.get("UFO_TEST_RESET_DATA", "ufo/data/lafan_29dof_10s-clipped.pkl"),
                            robot_config=os.environ.get("UFO_TEST_RESET_ROBOT", "configs/robots/g1_speed_pbfm_reward.yaml"))
    env, _ = cfg.env.build(num_envs=4)
    try:
        env.reset(to_numpy=False)
        core = env.base_env
        history = tree_map(lambda value: value.clone(), core.history_handler.history)
        captured = {}
        reset = core.reset_finished
        def capture(ids):
            captured["ids"] = ids.clone()
            captured["qpos"], captured["qvel"] = env._get_qpos_qvel(to_numpy=False)
            reset(ids)
        core.reset_finished = capture
        core.episode_length_buf.zero_()
        core.episode_length_buf[0] = 100000
        obs, _, terminated, truncated, info = env.step(torch.zeros_like(core.actions), to_numpy=False)
        assert truncated.tolist() == [True, False, False, False]
        assert not terminated.any()
        assert info["final_env_ids"].tolist() == [0]
        for key in ("qpos", "qvel"):
            torch.testing.assert_close(info[f"final_{key}"], captured[key][[0]])
        for key, value in core.history_handler.history.items():
            torch.testing.assert_close(value[1:, 1:], history[key][1:, :-1])
            assert not value[0, 1:].any()
        assert all(torch.isfinite(value).all() for value in obs.values())
        assert info["final_observation"]["time"].item() > 0 and obs["time"][0].item() == 0
        env.step(torch.zeros_like(core.actions), to_numpy=False)
        for key in ("qpos", "qvel"):
            torch.testing.assert_close(info[f"final_{key}"], captured[key][[0]])
    finally:
        env.close()
