"""Behavioral coverage for native task configs and TeCH checkpoint continuation."""

import copy

import pytest
import torch

from ufo.envs.config import resolve_task_config
from ufo.train import build_mjlab_config


def config():
    return build_mjlab_config(
        device="cpu", work_dir="/tmp/ufo-config-test", num_envs=8, num_env_steps=32, seed=17, use_wandb=False, wandb_run_name=None
    )


def test_native_robot_controls_are_authoritative():
    env = config().env
    robot = copy.deepcopy(env.robot_training)
    robot["action_scale"] = 0.123
    robot["stiffness"] = {key: value * 1.1 for key, value in robot["stiffness"].items()}
    native = resolve_task_config(env.model_copy(update={"robot_training": robot}), 8)
    assert native.robot.control.action_scale == 0.123
    assert dict(native.robot.control.stiffness) == robot["stiffness"]


def test_reject_double_reset_ownership():
    with pytest.raises(ValueError, match="owns task reset"):
        config().env.model_copy(update={"auto_reset": True}).build()


def test_tech_checkpoint_restores_dual_and_continues_update(tmp_path):
    from test_agent_runtime import _small_agent

    torch.set_num_threads(1)
    torch.manual_seed(91)
    agent = _small_agent("tech", torch.device("cpu"))
    obs = torch.randn(8, 6)
    next_obs = torch.randn(8, 6)
    agent.update_tech_te(obs, next_obs)
    agent.save(tmp_path)
    restored = type(agent).load(tmp_path, device="cpu")
    torch.testing.assert_close(agent.dual_lam_log_param, restored.dual_lam_log_param, rtol=0, atol=0)
    for candidate in (agent, restored):
        torch.manual_seed(92)
        candidate.update_tech_te(obs, next_obs)
    for name, value in agent._model.state_dict().items():
        torch.testing.assert_close(value, restored._model.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(agent.dual_lam_log_param, restored.dual_lam_log_param, rtol=0, atol=0)


def test_tech_config_rejects_retired_fields():
    from ufo.agents.tech.config import TeCHAgentConfig
    from ufo.training.config import TrainConfig

    payload = config().model_dump()
    payload["agent"] = TeCHAgentConfig().model_dump()
    for key in ("use_gcr_pretrain", "gcr_loss_coef", "lr_actorr"):
        old_train = {**payload["agent"]["train"], key: True}
        with pytest.raises(ValueError, match=key):
            TrainConfig.model_validate({**payload, "agent": {**payload["agent"], "train": old_train}})



def test_model_checkpoint_roundtrip_and_strict_weights(tmp_path):
    import safetensors.torch
    from test_agent_runtime import _small_agent

    from ufo.agents.common.checkpoint import load_model, save_model

    model = _small_agent("tech", torch.device("cpu"))._model
    kwargs = {"obs_space": model.obs_space, "action_dim": model.action_dim}
    save_model(tmp_path, model, kwargs)
    assert kwargs["obs_space"] is model.obs_space
    # JSON is authoritative; old pickle files are not consulted.
    (tmp_path / "init_kwargs.pkl").write_bytes(b"unused")
    restored = load_model(tmp_path, "cpu", load_target_networks=True, config_class=type(model.cfg))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    inference = type(model).load(tmp_path, "cpu", load_target_networks=False)
    assert not any(name.startswith("_target_") for name in inference.state_dict())
    weights = safetensors.torch.load_file(tmp_path / "model.safetensors")
    del weights[next(name for name in weights if name.startswith("_actor."))]
    safetensors.torch.save_file(weights, tmp_path / "model.safetensors")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(tmp_path, "cpu", load_target_networks=False, config_class=type(model.cfg))
    (tmp_path / "init_kwargs.json").unlink()
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path, "cpu", load_target_networks=True, config_class=type(model.cfg))



@pytest.mark.parametrize("disabled", [False, True])
def test_native_task_respects_noise_and_randomization(disabled):
    env = config().env.model_copy(update={"disable_obs_noise": disabled, "disable_domain_randomization": disabled})
    task = resolve_task_config(env, 8)
    assert task.num_envs == 8
    assert task.robot.actions_dim == 29
    assert task.lie_down_init and task.lie_down_init_prob == 0.3
    assert task.domain_rand.randomize_link_mass is not disabled
    assert (task.obs.noise_scales.dof_pos == 0) is disabled


def test_current_train_counters_roundtrip_and_reject_legacy():
    from ufo.training.trainer import _make_train_status, _normalize_train_status

    cfg = config().model_copy(update={"distributed_world_size": 8})
    status = _make_train_status(cfg, local_time=16, global_time=128, optimizer_steps=3)
    assert _normalize_train_status(status, cfg) == status
    with pytest.raises(ValueError, match="missing required fields"):
        _normalize_train_status({"time": 128}, cfg)
    with pytest.raises(RuntimeError, match="different distributed world_size"):
        _normalize_train_status({**status, "world_size": 4}, cfg)


def test_network_config_has_one_actor_schema_and_only_batched_ensembles():
    from ufo.agents.common.networks import ActorArchiConfig, ForwardArchiConfig

    for name in ("simple", "residual"):
        with pytest.raises(ValueError, match="model"):
            ActorArchiConfig(model=name)
    for mode in ("seq", "vmap"):
        with pytest.raises(ValueError, match="ensemble_mode"):
            ForwardArchiConfig(ensemble_mode=mode)


@pytest.mark.parametrize("agent,updates", [("fb", 16), ("tech", 128)])
def test_runtime_overrides_do_not_mutate_algorithm_defaults(agent, updates):
    kwargs = dict(
        device="cpu", work_dir="/tmp/ufo-config-test", num_envs=8, num_env_steps=32,
        seed=17, use_wandb=False, wandb_run_name=None, agent=agent,
    )
    changed = build_mjlab_config(**kwargs, num_agent_updates=7, buffer_size=2048)
    default = build_mjlab_config(**kwargs)
    assert changed.num_agent_updates == 7 and changed.buffer_size == 2048
    assert default.num_agent_updates == updates and default.buffer_size == 5120000
