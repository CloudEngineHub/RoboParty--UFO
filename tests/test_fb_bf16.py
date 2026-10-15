"""BF16 update forwards keep losses, master weights and rollout in FP32."""

from unittest.mock import patch

import pytest
import torch
from test_critic_topology import fixture_batch, small_agent

from ufo.agents.fb_cpr_aux.agent import FBcprAuxAgent
from ufo.agents.fb_cpr_aux.config import FBcprAuxAgentTrainConfig, build_fb_agent


def test_bf16_is_opt_in():
    assert not FBcprAuxAgentTrainConfig().use_bf16_amp
    assert not build_fb_agent().train.use_bf16_amp


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))])
def test_bf16_update_precision_and_checkpoint(shared, device, tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(67)
    agent, shapes = small_agent(shared, device=device)
    agent.cfg = agent.cfg.model_copy(update={"train": agent.cfg.train.model_copy(update={"use_bf16_amp": True})})
    batch, replay = fixture_batch(agent, shapes)
    model = agent._model
    names = ["_actor", "_forward_map", "_backward_map", "_target_forward_map", "_target_backward_map", "_critic", "_target_critic"]
    if not shared:
        names += ["_aux_critic", "_target_aux_critic"]
    observed, backward_dtypes = {}, []
    hooks = []

    def observe(name):
        def hook(module, args, result):
            assert torch.is_autocast_enabled(device), name
            tensors = result if isinstance(result, tuple) else (result,)
            observed.setdefault(name, []).extend(x.dtype if isinstance(x, torch.Tensor) else x.mean.dtype for x in tensors)
            if name in ("_forward_map", "_target_forward_map", "_critic", "_target_critic", "_aux_critic", "_target_aux_critic"):
                assert args[2].dtype == torch.float32
        return hook

    def backward_hook(grad):
        assert not torch.is_autocast_enabled(device)
        backward_dtypes.append(grad.dtype)

    for name in names:
        hooks.append(getattr(model, name).register_forward_hook(observe(name)))
    hooks.append(model._discriminator.trunk.register_forward_hook(observe("discriminator")))
    for parameter in model.parameters():
        if parameter.requires_grad:
            hooks.append(parameter.register_hook(backward_hook))
    try:
        metrics = agent.update(replay, 0)
    finally:
        for hook in hooks:
            hook.remove()

    assert set(observed) == set(names) | {"discriminator"}
    assert torch.bfloat16 in {dtype for values in observed.values() for dtype in values}
    assert len(observed["discriminator"]) >= 4  # Includes the WGAN-GP forward.
    assert backward_dtypes and set(backward_dtypes) == {torch.float32}
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in metrics.values())
    assert all(parameter.dtype == torch.float32 and torch.isfinite(parameter).all() for parameter in model.parameters())
    for optimizer in agent.optimizers.values():
        for state in optimizer.state.values():
            assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32

    assert agent.act(batch["observation"], batch["z"]).dtype == torch.float32
    agent.save(str(tmp_path))
    loaded = FBcprAuxAgent.load(str(tmp_path), device=device)
    assert loaded.cfg.train.use_bf16_amp
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded._model.state_dict()[name], rtol=0, atol=0)


def test_discriminator_reward_clamps_in_fp32():
    agent, shapes = small_agent()
    agent.cfg = agent.cfg.model_copy(update={"train": agent.cfg.train.model_copy(update={"use_bf16_amp": True})})
    batch, _ = fixture_batch(agent, shapes)
    logits = torch.tensor([[-50.0], [50.0]], dtype=torch.bfloat16)
    with patch.object(agent._model._discriminator, "compute_logits", return_value=logits):
        reward = agent._discriminator_reward(batch["observation"], batch["z"])
    assert reward.dtype == torch.float32
    assert torch.isfinite(reward).all()
    assert reward[0] < 0 < reward[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_bf16_compiled_update():
    torch.set_num_threads(1)
    torch.manual_seed(68)
    agent, shapes = small_agent(shared=True, device="cuda")
    agent.cfg = agent.cfg.model_copy(update={"train": agent.cfg.train.model_copy(update={"use_bf16_amp": True})})
    _, replay = fixture_batch(agent, shapes)
    state_keys = tuple(agent._model.state_dict())
    agent.compile_network_forwards()
    for step in range(2):
        metrics = agent.update(replay, step)
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in metrics.values())
        assert all(torch.isfinite(parameter).all() for parameter in agent._model.parameters())
    assert tuple(agent._model.state_dict()) == state_keys
