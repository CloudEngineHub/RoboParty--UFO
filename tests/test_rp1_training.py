import sys

import pytest

from ufo.train import build_mjlab_config, parse_args


def _config(**changes):
    return build_mjlab_config(**{
        "device": "cpu", "work_dir": "/tmp/rp1_config", "num_envs": 8,
        "num_env_steps": 32, "seed": 52, "use_wandb": False, "wandb_run_name": None,
        "robot_config": "configs/robots/rp1_3.yaml", **changes,
    })


def test_rp1_baseline_and_explicit_critic_override():
    config = _config()
    assert config.agent.model.archi.shared_critic
    assert config.agent.model.archi.f.hidden_dim == 1024
    assert config.agent.train.use_bf16_amp
    assert config.agent.train.lr_discriminator == 2e-6
    assert config.agent.train.reg_coeff == config.agent.train.reg_coeff_aux == .03
    assert len(config.agent.aux_rewards) == 13
    assert config.prioritization_scale == 1.5
    assert config.eval_every_steps == 400 * 1024 * 8
    other = _config(network_width=2048, shared_critic=False, lr_scale=.5)
    assert not other.agent.model.archi.shared_critic
    assert other.agent.model.archi.critic.hidden_dim == 2048
    assert other.agent.train.lr_discriminator == 1e-6
    assert other.agent.train.lr_actor == 1.5e-4


def test_rp1_cli_defaults_and_smoke_override(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--robot-config", "configs/robots/rp1_3.yaml"])
    args = parse_args()
    assert args.seed == 52 and args.num_env_steps == 384000000
    assert args.checkpoint_every_steps == 2000 * 1024 * 8
    monkeypatch.setattr(sys, "argv", sys.argv + ["--smoke", "--seed", "7", "--no-shared-critic"])
    args = parse_args()
    assert args.seed == 7 and args.num_env_steps == 2048
    assert args.shared_critic is False


def test_rp1_tech_uses_supported_physical_rewards_without_fb_hyperparameters():
    config = _config(agent="tech")
    assert len(config.agent.aux_rewards) == 13
    assert config.agent.aux_rewards_scaling["joint_energy"] == -.0001
    assert not hasattr(config.agent.train, "use_bf16_amp")


def test_rp1_rejects_conflicting_g1_reward_preset():
    with pytest.raises(ValueError, match="explicit FB reward baseline"):
        _config(cartwheel_aux_safe=True)
