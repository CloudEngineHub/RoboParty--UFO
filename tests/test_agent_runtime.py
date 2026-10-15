"""Exercise agent updates, optimizer state and checkpoint loading."""

import os
from datetime import timedelta
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ufo.agents.common.inputs import IdentityInputFilterConfig, IdentityNormalizerConfig, ObsNormalizerConfig
from ufo.agents.fb_cpr_aux.config import FBcprAuxAgentConfig
from ufo.agents.tech.config import TeCHAgentConfig
from ufo.training.distributed import broadcast_agent_state

OPTIMIZER_NAMES = {
    "fb": ("actor_optimizer", "backward_optimizer", "forward_optimizer", "critic_optimizer", "discriminator_optimizer", "aux_critic_optimizer"),
    "shared": ("actor_optimizer", "backward_optimizer", "forward_optimizer", "critic_optimizer", "discriminator_optimizer"),
    "tech": ("actor_optimizer", "goal_encoder_optimizer", "dual_lam_optimizer", "critic_optimizer", "discriminator_optimizer", "aux_critic_optimizer"),
}


def _small_agent(family, device):
    cfg = FBcprAuxAgentConfig() if family in ("fb", "shared") else TeCHAgentConfig()
    architectures = {"z_dim": 4}
    if family in ("fb", "shared"):
        architectures["shared_critic"] = family == "shared"
    for name in ("f", "b", "actor", "critic", "aux_critic", "discriminator", "goal_encoder"):
        architecture = getattr(cfg.model.archi, name, None)
        if architecture is not None:
            architectures[name] = architecture.model_copy(update={"hidden_dim": 16, "hidden_layers": 2, "input_filter": IdentityInputFilterConfig()})
    cfg = cfg.model_copy(
        update={
            "model": cfg.model.model_copy(
                update={
                    "device": device.type,
                    "seq_length": 1,
                    "obs_normalizer": ObsNormalizerConfig(normalizers=IdentityNormalizerConfig()),
                    "archi": cfg.model.archi.model_copy(update=architectures),
                }
            ),
            "train": cfg.train.model_copy(update={"batch_size": 8}),
            "aux_rewards": [],
            "aux_rewards_scaling": {},
        }
    )
    space = gym.spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
    return cfg.build(space, 2)


def _worker(rank, rendezvous, backend, world_size):
    torch.set_num_threads(1)
    if backend == "nccl":
        torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}" if backend == "nccl" else "cpu")
    dist.init_process_group(backend, init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=120))
    try:
        for family in ("fb", "shared", "tech"):
            torch.manual_seed(10)
            manual = _small_agent(family, device)
            torch.manual_seed(10)
            bucketed = _small_agent(family, device)

            # Use the original per-tensor SUM/world-size as an independent oracle.
            def reference_average(parameters):
                for parameter in parameters:
                    if parameter.grad is not None:
                        dist.all_reduce(parameter.grad)
                        parameter.grad.div_(world_size)

            manual._sync_gradients = reference_average
            keys = tuple(bucketed._model.state_dict())
            assert tuple(bucketed._model.state_dict()) == keys
            for iteration in range(3):
                torch.manual_seed(1000 + rank + iteration * world_size)
                obs, next_obs = (torch.randn(8, 6, device=device) for _ in range(2))
                z = manual._model.project_z(torch.randn(8, 4, device=device))
                action = torch.randn(8, 2, device=device).tanh()
                discount = torch.full((8, 1), 0.98, device=device)
                reward = torch.randn(8, 1, device=device)
                common = dict(obs=obs, action=action, discount=discount, next_obs=next_obs, z=z)
                stages = []
                if family in ("fb", "shared"):
                    stages.append(
                        (
                            "update_fb",
                            dict(**common, goal=next_obs, q_loss_coef=None, clip_grad_norm=1.0),
                            ("_forward_map", "_backward_map"),
                        )
                    )
                else:
                    stages.append(("update_tech_te", dict(train_obs_raw=obs, train_next_obs_raw=next_obs), ("_goal_encoder",)))
                stages.extend(
                    [
                        (
                            "update_discriminator",
                            dict(expert_obs=next_obs, expert_z=-z, train_obs=obs, train_z=z, grad_penalty=10.0),
                            ("_discriminator",),
                        ),
                        ("update_critic", common if family in ("fb", "shared") else dict(**common, reward=reward), ("_critic",)),
                        ("update_aux_critic", dict(**common, aux_reward=reward), ("_aux_critic",)),
                        (
                            "_run_actor_update",
                            dict(obs=obs, z=z, clip_grad_norm=1.0, **({"action": action} if family in ("fb", "shared") else {})),
                            ("_actor",),
                        ),
                    ]
                )
                if family == "shared":
                    stages[2:4] = [("update_shared_critic", dict(**common, aux_reward=reward), ("_critic",))]
                for index, (method, kwargs, modules) in enumerate(stages):
                    seed = 3000 + rank + iteration * 100 + index
                    torch.manual_seed(seed)
                    expected = getattr(manual, method)(**kwargs)
                    torch.manual_seed(seed)
                    actual = getattr(bucketed, method)(**kwargs)
                    for name in expected:
                        torch.testing.assert_close(
                            actual[name], expected[name], atol=2e-5, rtol=2e-5, msg=lambda m: f"{family}/{method}/{name}: {m}"
                        )
                    for module_name in modules:
                        for (name, a), b in zip(
                            getattr(bucketed._model, module_name).named_parameters(), getattr(manual._model, module_name).parameters()
                        ):
                            torch.testing.assert_close(
                                a.grad, b.grad, atol=2e-5, rtol=2e-5, msg=lambda m: f"{family}/{method}/{name} gradient: {m}"
                            )
                            torch.testing.assert_close(
                                a, b, atol=2e-5, rtol=2e-5, msg=lambda m: f"{family}/{method}/{name} parameter: {m}"
                            )
                for a, b in zip(bucketed._model.parameters(), manual._model.parameters()):
                    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
                    reference = a.detach().clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(a, reference, atol=0, rtol=0)
                if family == "tech":
                    torch.testing.assert_close(bucketed.dual_lam_log_param, manual.dual_lam_log_param)
                    reference = bucketed.dual_lam_log_param.detach().clone()
                    dist.broadcast(reference, src=0)
                    torch.testing.assert_close(bucketed.dual_lam_log_param, reference, atol=0, rtol=0)
            # A checkpoint written with bucketed reduction loads into the same unwrapped model.
            manual._model.load_state_dict(bucketed._model.state_dict(), strict=True)
            assert all(parameter.requires_grad for parameter in bucketed._model._actor.parameters())
            assert all(parameter.requires_grad for parameter in bucketed._model._critic.parameters())
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_bucketed_updates_match_per_tensor_average(tmp_path):
    backend = os.environ.get("UFO_TEST_DISTRIBUTED_BACKEND", "gloo")
    world_size = 8 if backend == "nccl" else 2
    mp.spawn(_worker, args=((tmp_path / "bucketed-rendezvous").as_uri(), backend, world_size), nprocs=world_size, join=True)


@pytest.mark.parametrize("family", ["fb", "shared", "tech"])
def test_adam_cpu_fallback(family):
    agent = _small_agent(family, torch.device("cpu"))
    for name in agent.optimizer_dict:
        optimizer = getattr(agent, name)
        assert optimizer.defaults["fused"] is None
        assert optimizer.defaults["foreach"] is None


def _initialize_optimizer_states(agent, steps=1):
    for optimizer in agent.optimizers.values():
        for _ in range(steps):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    parameter.grad = torch.full_like(parameter, steps)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


@pytest.mark.parametrize("family,frozen", [("fb", False), ("shared", False), ("tech", False), ("tech", True)])
def test_optimizer_mapping_checkpoint_keys_and_restored_states(tmp_path, family, frozen):
    torch.set_num_threads(1)
    agent = _small_agent(family, torch.device("cpu"))
    assert tuple(agent.optimizers) == OPTIMIZER_NAMES[family]
    assert all(optimizer is getattr(agent, name) for name, optimizer in agent.optimizers.items())
    _initialize_optimizer_states(agent)
    if frozen:
        agent._goal_encoder_frozen = True
        agent._model._goal_encoder.requires_grad_(False)
    agent.save(str(tmp_path))
    states = torch.load(tmp_path / "optimizers.pth", weights_only=True)
    assert tuple(states) == OPTIMIZER_NAMES[family]
    if family == "tech":
        # Existing checkpoints stored these same stages in a different order.
        previous_order = ("goal_encoder_optimizer", "actor_optimizer", "critic_optimizer", "discriminator_optimizer", "aux_critic_optimizer", "dual_lam_optimizer")
        torch.save({name: states[name] for name in previous_order}, tmp_path / "optimizers.pth")
    loaded = type(agent).load(str(tmp_path), device="cpu")
    assert tuple(loaded.optimizers) == OPTIMIZER_NAMES[family]
    for name, optimizer in loaded.optimizers.items():
        actual = optimizer.state_dict()
        assert actual["param_groups"] == states[name]["param_groups"]
        assert actual["state"].keys() == states[name]["state"].keys()
        for index, expected_state in states[name]["state"].items():
            for key, expected in expected_state.items():
                torch.testing.assert_close(actual["state"][index][key], expected, rtol=0, atol=0)
    if family == "tech":
        assert loaded._goal_encoder_frozen == frozen
        assert all(parameter.requires_grad != frozen for parameter in loaded._model._goal_encoder.parameters())
        torch.testing.assert_close(loaded.dual_lam_log_param, agent.dual_lam_log_param, rtol=0, atol=0)


def _optimizer_broadcast_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=120))
    try:
        for family, frozen in (("fb", False), ("shared", False), ("tech", False), ("tech", True)):
            torch.manual_seed(10)
            agent = _small_agent(family, torch.device("cpu"))
            assert tuple(agent.optimizers) == OPTIMIZER_NAMES[family]
            _initialize_optimizer_states(agent, steps=rank + 1)
            if frozen:
                agent._goal_encoder_frozen = True
                agent._model._goal_encoder.requires_grad_(False)
            expected = []
            for optimizer in agent.optimizers.values():
                assert optimizer.state
                for state in optimizer.state.values():
                    for value in state.values():
                        if torch.is_tensor(value):
                            reference = value.detach().clone()
                            dist.broadcast(reference, src=0)
                            if rank == 1:
                                assert not torch.equal(value, reference)
                            expected.append((value, reference))
            broadcast_agent_state(agent)
            for value, reference in expected:
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_broadcast_covers_every_optimizer_state(tmp_path):
    mp.spawn(_optimizer_broadcast_worker, args=((tmp_path / "optimizer-rendezvous").as_uri(),), nprocs=2, join=True)


def test_adam_cuda_selects_fused_without_a_config_switch():
    from unittest.mock import patch

    from ufo.agents.common.execution import make_adam

    with patch("torch.optim.Adam") as adam:
        make_adam([], "cuda", lr=0.01, weight_decay=0.0)
    assert adam.call_args.kwargs["fused"] is True


@pytest.mark.parametrize("dictionary", [False, True])
def test_wgan_penalty_preserves_single_draw_and_parameter_gradient(dictionary):
    from ufo.agents.common.execution import gradient_penalty_wgan

    class Discriminator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.4, -0.6, 0.9], dtype=torch.float64))

        def compute_logits(self, obs, z):
            state = obs["state"] if isinstance(obs, dict) else obs
            return (torch.cat([state, z], dim=1).square() * self.weight).sum(1, keepdim=True)

    real = torch.tensor([[0.2, 0.7], [-0.4, 0.3]], dtype=torch.float64)
    fake = -real
    real_z, fake_z = real[:, :1] + 0.1, fake[:, :1] - 0.2
    discriminator = Discriminator()
    torch.manual_seed(19)
    alpha = torch.rand(2, 1)
    expected_rng = torch.get_rng_state()
    features = torch.cat([alpha * real + (1 - alpha) * fake, alpha * real_z + (1 - alpha) * fake_z], dim=1)
    expected = ((2 * features * discriminator.weight).norm(2, dim=1) - 1).square().mean()
    if dictionary:
        real, fake = {"unused": real[:, :1], "state": real}, {"unused": fake[:, :1], "state": fake}
    torch.manual_seed(19)
    actual = gradient_penalty_wgan(discriminator, real, real_z, fake, fake_z)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual_grad = torch.autograd.grad(actual, discriminator.weight)[0]
    expected_grad = torch.autograd.grad(expected, discriminator.weight)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    if not dictionary:
        with pytest.raises(ValueError, match="Shape mismatch"):
            gradient_penalty_wgan(discriminator, real, real_z, fake[:, :1], fake_z)


def test_checkpoint_spaces_support_box_dict_and_reject_discrete():
    from ufo.agents.common.checkpoint import json_to_space, space_to_json

    space = gym.spaces.Dict({"state": gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)})
    assert json_to_space(space_to_json(space)) == space
    with pytest.raises(NotImplementedError, match="Discrete"):
        space_to_json(gym.spaces.Discrete(2))
    with pytest.raises(NotImplementedError, match="Discrete"):
        json_to_space({"type": "Discrete", "n": 2})


@pytest.mark.parametrize("family", ["fb", "shared", "tech"])
def test_inference_omits_targets_but_requires_all_policy_weights(tmp_path, family):
    import safetensors.torch

    from ufo.agents.common.checkpoint import load_model_from_checkpoint_dir

    agent = _small_agent(family, torch.device("cpu"))
    agent.save(str(tmp_path))
    model = load_model_from_checkpoint_dir(tmp_path)
    assert not any(key.startswith("_target_") for key in model.state_dict())
    expected = {key: value for key, value in agent._model.state_dict().items() if not key.startswith("_target_")}
    assert model.state_dict().keys() == expected.keys()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    obs = torch.randn(3, 6)
    z = agent._model.sample_z(3)
    model.eval()
    agent._model.eval()
    torch.testing.assert_close(model.act(obs, z), agent._model.act(obs, z), rtol=0, atol=0)
    torch.testing.assert_close(model.backward_map(obs), agent._model.backward_map(obs), rtol=0, atol=0)
    path = tmp_path / "model" / "model.safetensors"
    weights = safetensors.torch.load_file(path)
    del weights[next(key for key in weights if key.startswith("_actor."))]
    safetensors.torch.save_file(weights, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model_from_checkpoint_dir(tmp_path)


def test_tech_full_update_crosses_default_pretrain_boundary():
    from types import SimpleNamespace

    agent = _small_agent("tech", torch.device("cpu"))
    agent.cfg = agent.cfg.model_copy(update={"train": agent.cfg.train.model_copy(update={
        "use_tech_pretrain": True, "tech_pretrain_env_steps": 200000,
        "tech_te_during_rl": True, "freeze_goal_encoder_after_pretrain": False,
    })})
    batch = {"observation": torch.randn(8, 6), "action": torch.randn(8, 2), "z": agent._model.sample_z(8),
             "next": {"observation": torch.randn(8, 6), "terminated": torch.zeros(8, 1, dtype=torch.bool)}}
    class Replay:
        def __len__(self):
            return 8

        def sample(self, size):
            return batch

    replay = {"train": Replay(), "expert": SimpleNamespace(sample=lambda size: batch)}
    agent.update(replay, step=199999)
    assert not agent.actor_optimizer.state and not agent.critic_optimizer.state
    before = {key: value.clone() for key, value in agent._model._actor.state_dict().items()}
    with patch.object(agent._model._actor, "forward", wraps=agent._model._actor.forward) as forward:
        metrics = agent.update(replay, step=200000)
    assert forward.call_count == 2  # One target distribution, then the actor's own update.
    for name in ("actor_optimizer", "critic_optimizer", "aux_critic_optimizer", "discriminator_optimizer", "goal_encoder_optimizer"):
        assert getattr(agent, name).state, name
    assert any(not torch.equal(value, before[key]) for key, value in agent._model._actor.state_dict().items())
    assert all(torch.isfinite(value).all() for value in metrics.values())


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tech_cached_distribution_preserves_critic_updates_and_rng(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    reference = _small_agent("tech", torch.device(device))
    torch.manual_seed(42)
    actual = _small_agent("tech", torch.device(device))
    batch = {
        "obs": torch.randn(8, 6, device=device), "next_obs": torch.randn(8, 6, device=device),
        "action": torch.randn(8, 2, device=device), "z": actual._model.sample_z(8, device=device),
        "discount": torch.full((8, 1), .98, device=device),
    }
    reward = torch.randn(8, 1, device=device)
    for iteration in range(2):
        torch.manual_seed(100 + iteration)
        expected = [reference.update_critic(**batch, reward=reward), reference.update_aux_critic(**batch, aux_reward=-reward)]
        expected_rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
        torch.manual_seed(100 + iteration)
        with torch.no_grad():
            distribution = actual._model._actor(batch["next_obs"], batch["z"], actual._model.cfg.actor_std)
        metrics = [actual.update_critic(**batch, reward=reward, target_dist=distribution),
                   actual.update_aux_critic(**batch, aux_reward=-reward, target_dist=distribution)]
        torch.testing.assert_close(metrics, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual._model.state_dict(), reference._model.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(actual.optimizer_dict, reference.optimizer_dict, rtol=0, atol=0)
        rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
        assert torch.equal(rng, expected_rng)
