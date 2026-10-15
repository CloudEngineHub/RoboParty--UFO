"""Shared topology, precision boundaries and resumable agent checkpoints."""

import copy
import tempfile

import gymnasium as gym
import numpy as np
import pytest
import torch

from ufo.agents.fb_cpr_aux.agent import FBcprAuxAgent
from ufo.agents.fb_cpr_aux.config import build_fb_agent


def small_agent(shared=True, device="cpu"):
    cfg = build_fb_agent(device=device)
    arch = cfg.model.archi
    changes = {
        k: getattr(arch, k).model_copy(update={"hidden_dim": 32, "hidden_layers": 2})
        for k in ("actor", "f", "critic", "discriminator", "b")
    }
    arch = arch.model_copy(update=dict(changes, z_dim=8, shared_critic=shared))
    cfg = cfg.model_copy(
        update={
            "model": cfg.model.model_copy(update={"archi": arch, "seq_length": 4}),
            "train": cfg.train.model_copy(update={"batch_size": 16}),
        }
    )
    shapes = {"state": 4, "privileged_state": 3, "last_action": 2, "history_actor": 6}
    space = gym.spaces.Dict({k: gym.spaces.Box(-np.inf, np.inf, shape=(v,), dtype=np.float32) for k, v in shapes.items()})
    return cfg.build(space, 2), shapes


def fixture_batch(agent, shapes):
    device = agent.device
    obs = {k: torch.randn(16, v, device=device) for k, v in shapes.items()}
    batch = {
        "observation": obs,
        "action": torch.randn(16, 2, device=device),
        "z": torch.randn(16, 8, device=device),
        "aux_rewards": {k: torch.rand(16, 1, device=device) for k in agent.cfg.aux_rewards},
        "next": {"observation": obs, "terminated": torch.zeros(16, 1, device=device, dtype=torch.bool)},
    }

    class Buffer:
        def sample(self, n):
            return batch

    return batch, {"train": Buffer(), "expert": Buffer()}


@pytest.mark.parametrize("shared", [False, True])
def test_checkpoint_continues_same_update(shared):
    torch.set_num_threads(1)
    torch.manual_seed(21)
    agent, shapes = small_agent(shared)
    batch, replay = fixture_batch(agent, shapes)
    metrics = agent.update(replay, 0)
    assert all(torch.isfinite(v).all() for v in metrics.values())
    if shared:
        assert not hasattr(agent._model, "_aux_critic")
        assert "aux_critic_optimizer" not in agent.optimizer_dict
    assert agent._model._actor(batch["observation"], batch["z"], 0.05).mean.dtype == torch.float32
    with tempfile.TemporaryDirectory() as path:
        agent.save(path)
        loaded = FBcprAuxAgent.load(path, device="cpu")
    assert loaded.cfg.model.archi.shared_critic == shared
    for k, v in agent._model.state_dict().items():
        torch.testing.assert_close(v, loaded._model.state_dict()[k], rtol=0, atol=0)
    for candidate in (agent, loaded):
        torch.manual_seed(99)
        candidate.update(replay, 1)
    for k, v in agent._model.state_dict().items():
        torch.testing.assert_close(v, loaded._model.state_dict()[k], rtol=0, atol=0)
    for name in loaded.optimizer_dict:
        for state in getattr(loaded, name).state.values():
            for key in ("exp_avg", "exp_avg_sq"):
                assert state[key].dtype == torch.float32


def test_shared_gradient_sum_and_single_target_update():
    torch.set_num_threads(1)
    agent, shapes = small_agent()
    batch, replay = fixture_batch(agent, shapes)
    obs, z, action = batch["observation"], batch["z"], batch["action"]
    shared = agent._model._critic
    ref = copy.deepcopy(shared)
    targets = [torch.randn(2, 16, 1) for _ in range(2)]
    for head in range(2):
        (ref(obs, z, action)[head] - targets[head]).square().mean().backward()
    outputs = shared(obs, z, action)
    sum((q - target).square().mean() for q, target in zip(outputs, targets)).backward()
    for p, q in zip(shared.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    before = [p.detach().clone() for p in agent._model._target_critic.parameters()]
    actions = []
    hook = agent._model._target_critic.register_forward_pre_hook(lambda m, args: actions.append(args[2].detach().clone()))
    agent.update(replay, 0)
    hook.remove()
    assert len(actions) == 2 and not torch.equal(*actions)
    tau = agent.cfg.train.critic_target_tau
    for old, online, target in zip(before, shared.parameters(), agent._model._target_critic.parameters()):
        torch.testing.assert_close(target, old * (1 - tau) + online * tau)




@pytest.mark.parametrize("shared", [False, True])
def test_target_actor_reused_without_reusing_actions(shared):
    torch.manual_seed(19)
    agent, shapes = small_agent(shared)
    _, replay = fixture_batch(agent, shapes)
    distributions, actions = [], []
    def actor_output(module, args, output):
        if not torch.is_grad_enabled():
            distributions.append((output, torch.get_rng_state()))
    hooks = [agent._model._actor.register_forward_hook(actor_output)]
    targets = [agent._model._target_forward_map, agent._model._target_critic]
    if not shared:
        targets.append(agent._model._target_aux_critic)
    for target in targets:
        hooks.append(target.register_forward_pre_hook(lambda module, args: actions.append(args[2].detach().clone())))
    try:
        agent.update(replay, 0)
    finally:
        for hook in hooks:
            hook.remove()
    assert len(distributions) == 1 and len(actions) == 3
    distribution, rng = distributions[0]
    torch.set_rng_state(rng)
    for action in actions:
        torch.testing.assert_close(action, distribution.sample(clip=agent.cfg.train.stddev_clip), rtol=0, atol=0)
    assert not torch.equal(actions[0], actions[1]) and not torch.equal(actions[1], actions[2])


def test_two_critic_uncertainty_matches_pairwise_value_and_gradient():
    agent, _ = small_agent()
    values = torch.randn(2, 7, 3, dtype=torch.float64, requires_grad=True)
    expected = values.mean(0) - 0.5 * (values.unsqueeze(0) - values.unsqueeze(1)).abs().sum((0, 1)) / 2
    actual = agent.get_targets_uncertainty(values, 0.5)[2]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), values)[0], torch.autograd.grad(expected.sum(), values)[0])




@pytest.mark.parametrize("shared", [False, True])
def test_training_config_retains_critic_choice_and_width(shared):
    from ufo.train import build_mjlab_config
    from ufo.training.config import TrainConfig

    cfg = build_mjlab_config(
        device="cpu", work_dir="/tmp/ufo-config", num_envs=8, num_env_steps=32,
        seed=1, use_wandb=False, wandb_run_name=None, network_width=1024, shared_critic=shared,
    )
    assert TrainConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert cfg.agent.model.archi.shared_critic == shared
    assert all(getattr(cfg.agent.model.archi, key).hidden_dim == 1024 for key in ("actor", "f", "critic"))


def test_forward_compilation_preserves_parameter_keys_and_leaves_discriminator_eager():
    from unittest.mock import PropertyMock, patch

    agent, _ = small_agent()
    keys = tuple(agent._model.state_dict())
    actor = agent._model._actor.forward
    discriminator = agent._model._discriminator.compute_logits
    compiled = []
    def compile_forward(fn, **kwargs):
        assert kwargs == {"mode": "default"}
        compiled.append(fn)
        return fn
    with patch.object(type(agent), "device", new_callable=PropertyMock, return_value="cuda"), patch("torch.compile", side_effect=compile_forward):
        agent.compile_network_forwards()
        agent.compile_network_forwards()
    assert actor in compiled and discriminator not in compiled
    assert len(compiled) == len(set(compiled))
    assert tuple(agent._model.state_dict()) == keys
