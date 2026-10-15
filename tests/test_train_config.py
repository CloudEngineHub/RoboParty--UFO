from __future__ import annotations

import copy
import io
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from ufo.agents.fb_cpr_aux.config import FBcprAuxAgentConfig, FBcprAuxAgentTrainConfig, build_fb_agent
from ufo.agents.tech.config import TeCHAgentConfig, TeCHAgentTrainConfig, build_tech_agent
from ufo.envs.config import MJLabEnvConfig, resolve_task_config
from ufo.train import _default_update_z_every_step, _ensure_compile_cache, build_mjlab_config, canonical_agent_name, launch, parse_args
from ufo.training.config import TrainConfig
from ufo.training.trainer import _accumulate_metrics, _trajectory_output_keys


class TrainConfigTest(unittest.TestCase):
    def test_runtime_cache_preserves_explicit_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "cache"
            triton = Path(tmpdir) / "custom_triton"
            triton.mkdir()
            with patch.dict(os.environ, {"UFO_CACHE_DIR": str(root), "TRITON_CACHE_DIR": str(triton)}, clear=True):
                _ensure_compile_cache()
                self.assertEqual(os.environ["TRITON_CACHE_DIR"], str(triton))
                for key, directory in (("TMPDIR", "tmp"), ("TEMP", "tmp"), ("TMP", "tmp"),
                                       ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"), ("CUDA_CACHE_PATH", "cuda"),
                                       ("WARP_CACHE_PATH", "warp")):
                    self.assertEqual(os.environ[key], str(root / directory))
                    self.assertTrue(Path(os.environ[key]).is_dir())

    def _parse(self, *args: str):
        with patch.object(sys, "argv", ["train.py", *args]), patch("sys.stderr", io.StringIO()) as stderr:
            parsed = parse_args()
        self.assertEqual(stderr.getvalue(), "")
        return parsed

    def _config(self, **overrides):
        return build_mjlab_config(
            device="cpu", work_dir="/tmp/ufo_defaults_test", num_envs=1, num_env_steps=1,
            seed=1, use_wandb=False, wandb_run_name=None, smoke=True, **overrides,
        )

    def test_agent_specific_defaults_are_preserved(self) -> None:
        for name, schema, builder, expected in (
            ("fb", FBcprAuxAgentTrainConfig, build_fb_agent, 100),
            ("tech", TeCHAgentTrainConfig, build_tech_agent, 10),
        ):
            with self.subTest(agent=name):
                self.assertEqual(self._parse("--agent", name).update_z_every_step, expected)
                self.assertEqual(schema().update_z_every_step, expected)
                self.assertEqual(builder(device="cpu").train.update_z_every_step, expected)

    def test_checkpoint_defaults_agree_for_cli_builder_and_schema(self) -> None:
        args = self._parse()
        cfg = self._config()
        schema = TrainConfig(**{key: value for key, value in cfg.model_dump().items() if TrainConfig.model_fields[key].is_required()})
        for candidate in (args, cfg, schema):
            self.assertEqual(candidate.checkpoint_every_steps, 3_200_000)
            self.assertEqual(candidate.buffer_every_steps, 16_000_000)

    def test_explicit_intervals_survive_current_config_serialization(self) -> None:
        for name, interval, checkpoint_steps in (("fb", 150, 9_600_000), ("tech", 100, 5_000_000), ("fb", 37, 3_200_000)):
            with self.subTest(agent=name, interval=interval):
                buffer_steps = checkpoint_steps * 5
                args = self._parse(
                    "--agent", name, "--checkpoint-every-steps", str(checkpoint_steps),
                    "--buffer-every-steps", str(buffer_steps), "--update-z-every-step", str(interval),
                )
                cfg = self._config(
                    agent=name, checkpoint_every_steps=args.checkpoint_every_steps,
                    buffer_every_steps=args.buffer_every_steps, update_z_every_step=args.update_z_every_step,
                )
                loaded = TrainConfig.model_validate_json(cfg.model_dump_json())
                self.assertEqual(loaded.checkpoint_every_steps, checkpoint_steps)
                self.assertEqual(loaded.buffer_every_steps, buffer_steps)
                self.assertEqual(loaded.agent.train.update_z_every_step, interval)
                self.assertEqual(loaded, cfg)

    def test_incomplete_runtime_config_cannot_silently_use_other_defaults(self) -> None:
        payload = self._config().model_dump()
        for field in ("work_dir", "seed", "online_parallel_envs", "num_env_steps", "log_every_steps",
                      "update_agent_every", "num_seed_steps", "num_agent_updates", "buffer_size",
                      "buffer_device", "eval_every_steps"):
            with self.subTest(field=field):
                incomplete = {key: value for key, value in payload.items() if key != field}
                with self.assertRaisesRegex(ValueError, field):
                    TrainConfig.model_validate(incomplete)

    def test_save_intervals_reject_nonpositive_values(self) -> None:
        payload = self._config().model_dump()
        for field in ("checkpoint_every_steps", "buffer_every_steps"):
            for value in (0, -1):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, "must be positive"):
                        self._parse("--" + field.replace("_", "-"), str(value))
                    with self.assertRaisesRegex(ValueError, "greater than 0"):
                        TrainConfig.model_validate({**payload, field: value})
                    with self.assertRaisesRegex(ValueError, "greater than 0"):
                        self._config(**{field: value})

    def test_removed_checkpoint_switches_are_rejected(self) -> None:
        payload = self._config().model_dump()
        for field in ("checkpoint_buffer", "checkpoint_rank_buffers"):
            with self.subTest(field=field):
                self.assertNotIn(field, payload)
                with self.assertRaisesRegex(ValueError, "extra_forbidden"):
                    TrainConfig.model_validate({**payload, field: True})

    def test_tech_cli_value_reaches_agent_config(self) -> None:
        args = self._parse("--agent", "tech", "--update-z-every-step", "37")
        cfg = build_tech_agent(device="cpu", update_z_every_step=args.update_z_every_step)
        self.assertEqual(cfg.train.update_z_every_step, 37)

    def test_programmatic_tech_default_remains_ten(self) -> None:
        cfg = self._config(agent="tech")
        self.assertEqual(cfg.agent.train.update_z_every_step, 10)

    def test_tech_trajectory_buffer_keeps_aux_rewards(self) -> None:
        cfg = build_tech_agent(device="cpu", update_z_every_step=10)
        self.assertIn("aux_rewards", _trajectory_output_keys(cfg))

    def test_priority_mode_has_one_default_and_keeps_explicit_modes(self) -> None:
        cfg = self._config()
        self.assertEqual(cfg.prioritization_mode, "exp")
        payload = cfg.model_dump()
        del payload["prioritization_mode"]
        self.assertEqual(TrainConfig.model_validate(payload).prioritization_mode, cfg.prioritization_mode)
        for mode in ("lin", "bin"):
            self.assertEqual(TrainConfig.model_validate({**payload, "prioritization_mode": mode}).prioritization_mode, mode)

    def test_metric_accumulation_accepts_tech_phase_changes(self) -> None:
        totals, counts = _accumulate_metrics(
            None,
            {},
            {"tech_te_loss": torch.tensor(2.0)},
        )
        totals, counts = _accumulate_metrics(
            totals,
            counts,
            {
                "tech_te_loss": torch.tensor(4.0),
                "disc_wgan_gp_loss": torch.tensor(6.0),
            },
        )
        self.assertEqual(counts, {"tech_te_loss": 2, "disc_wgan_gp_loss": 1})
        self.assertEqual((totals["tech_te_loss"] / counts["tech_te_loss"]).item(), 3.0)
        self.assertEqual((totals["disc_wgan_gp_loss"] / counts["disc_wgan_gp_loss"]).item(), 6.0)


    def test_canonical_agent_names(self) -> None:
        self.assertEqual(canonical_agent_name("tech"), "tech")
        with self.assertRaises(ValueError):
            canonical_agent_name("tldr")
        self.assertEqual(canonical_agent_name("fb"), "fb")

    def test_agent_tech_parses(self) -> None:
        args = self._parse("--agent", "tech")
        self.assertEqual(args.agent, "tech")
        self.assertEqual(args.update_z_every_step, 10)

    def test_tech_default_update_z_every_step(self) -> None:
        self.assertEqual(_default_update_z_every_step("tech"), 10)

    def test_tech_build_config_uses_canonical_public_names(self) -> None:
        cfg = build_mjlab_config(
            device="cpu",
            work_dir="/tmp/ufo_agent_alias_test",
            num_envs=1,
            num_env_steps=1,
            seed=1,
            use_wandb=False,
            wandb_run_name=None,
            smoke=True,
            agent="tech",
        )
        self.assertEqual(cfg.agent.name, "TeCHAgent")
        self.assertEqual(cfg.agent.train.update_z_every_step, 10)
        self.assertEqual(cfg.wandb_gname, "ufo_tech")
        self.assertEqual(cfg.wandb_run_name, "ufo_tech")
        self.assertEqual(cfg.tags["agent"], "tech")

    def test_readme_exposes_tech_training(self) -> None:
        readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
        self.assertNotIn("TLDR Training", readme)
        self.assertIn("--robot g1  --agent tech", readme)
        self.assertIn("--robot rp1 --agent tech", readme)
        self.assertIn("--agent tech", readme)
        self.assertNotIn("  --agent tldr", readme)
        self.assertNotIn("ufo_tldr", readme)


    def test_removed_execution_flags_are_rejected(self):
        for flag in ("--bf16", "--bf16-network-forwards", "--fused-adam", "--compile-networks", "--gradient-sync", "--matmul-precision", "--ddp-bucket-cap-mb"):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                self._parse(flag)


def test_current_snapshot_roundtrips_and_rejects_retired_fields():
    cfg = build_mjlab_config(
        device="cpu", work_dir="/tmp/ufo-test", num_envs=2, num_env_steps=16, seed=1, use_wandb=False, wandb_run_name=None
    )
    payload = cfg.model_dump()
    assert TrainConfig.model_validate(payload) == cfg
    for key, value in {"motions": "", "infra": {}, "load_expert_data_from_motion_lib": True, "num_env_stepz": 16}.items():
        with pytest.raises(ValueError, match="extra_forbidden"):
            TrainConfig.model_validate({**payload, key: value})
    with pytest.raises(ValueError, match="extra_forbidden"):
        MJLabEnvConfig.model_validate({**payload["env"], "hydra_overrides": []})
    with pytest.raises(ValueError):
        MJLabEnvConfig.model_validate({**payload["env"], "task_overrides": None})
    with pytest.raises(ValueError, match="extra_forbidden"):
        TrainConfig.model_validate({**payload, "evaluations": [{**payload["evaluations"][0], "generate_videos": False}]})

@pytest.mark.parametrize("gpu_ids,visible,expected", [("all", "", "0,1"), ("0,2", "2,4,6", "2,6")])
def test_native_launcher_preserves_gpu_selection_and_resolved_args(tmp_path, monkeypatch, gpu_ids, visible, expected):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    data = tmp_path / "resolved.pkl"
    data.touch()
    args = Namespace(gpu_ids=gpu_ids, work_dir=str(tmp_path), data_path=[str(data)])
    with patch("torch.cuda.device_count", return_value=2), patch("torch.distributed.run.elastic_launch") as start:
        launch(args)
    config, worker = start.call_args.args
    assert config.nproc_per_node == 2 and config.rdzv_endpoint == "localhost:0"
    assert worker.__name__ == "run_train"
    start.return_value.assert_called_once_with(args, Path(tmp_path))
    assert args.gpu_ids == "single" and args.data_path == [str(data)]
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == expected


@pytest.mark.parametrize("schema,builder", [(FBcprAuxAgentConfig, build_fb_agent), (TeCHAgentConfig, build_tech_agent)])
def test_algorithm_schema_and_builder_have_one_default_config(schema, builder):
    assert schema().model_dump() == builder().model_dump()
    first, second = schema(), schema()
    first.aux_rewards.append("changed")
    first.aux_rewards_scaling["penalty_action_rate"] = 123.0
    first.model.archi.actor.input_filter.key.append("changed")
    assert second.model_dump() == builder().model_dump() == schema().model_dump()


@pytest.mark.parametrize("robot", ["g1_29dof", "rp1_3"])
@pytest.mark.parametrize("agent,width", [("fb", 1024), ("tech", 2048)])
def test_robot_launch_inherits_algorithm_topology_and_unscaled_angular_velocity(robot, agent, width):
    cfg = build_mjlab_config(
        device="cpu", work_dir="/tmp/ufo-defaults", num_envs=1, num_env_steps=1,
        seed=1, use_wandb=False, wandb_run_name=None,
        robot_config=f"configs/robots/{robot}.yaml", agent=agent,
    )
    defaults = (build_fb_agent if agent == "fb" else build_tech_agent)(device="cpu")
    assert cfg.agent.model.archi == defaults.model.archi
    for name in (("f", "actor", "critic") if agent == "fb" else ("actor", "critic", "aux_critic")):
        arch = getattr(cfg.agent.model.archi, name)
        assert arch.hidden_dim == width and arch.hidden_layers == 6
    if agent == "fb":
        assert cfg.agent.model.archi.shared_critic
    task = resolve_task_config(cfg.env, 1)
    assert task.obs.obs_scales.base_ang_vel == 1.0
    cfg.env.task_overrides.setdefault("obs", {}).setdefault("obs_scales", {})["base_ang_vel"] = 0.25
    assert resolve_task_config(cfg.env, 1).obs.obs_scales.base_ang_vel == 0.25


@pytest.mark.parametrize("shared", [None, False, True])
def test_fb_robot_baseline_and_cli_overrides_have_one_resolution(shared):
    baseline = {
        "model": {"archi": {"shared_critic": True, "actor": {"hidden_dim": 1536, "hidden_layers": 3}}},
        "train": {"lr_actor": 2e-4, "lr_discriminator": 2e-6, "clip_grad_norm": 1.0, "update_z_every_step": 73},
    }
    original = copy.deepcopy(baseline)
    cfg = build_fb_agent(device="cpu", overrides=baseline, network_width=1024, shared_critic=shared,
                         lr_scale=.5, update_z_every_step=37, clip_grad_norm=2.0)
    assert cfg.model.archi.shared_critic is (True if shared is None else shared)
    assert [getattr(cfg.model.archi, name).hidden_dim for name in ("actor", "f", "critic")] == [1024] * 3
    assert cfg.model.archi.actor.hidden_layers == 3
    assert cfg.train.lr_actor == 1e-4 and cfg.train.lr_discriminator == 1e-6
    assert cfg.train.lr_f == 1.5e-4 and cfg.train.lr_aux_critic == 1.5e-4
    # Preserve existing robot-baseline precedence for these fields.
    assert cfg.train.clip_grad_norm == 1.0 and cfg.train.update_z_every_step == 73
    assert baseline == original
    assert build_fb_agent().model_dump() == FBcprAuxAgentConfig().model_dump()


def test_fb_resolved_config_rejects_unknown_robot_fields_and_reward_conflicts():
    with pytest.raises(ValueError, match="extra_forbidden"):
        build_fb_agent(overrides={"train": {"lr_actorr": 1e-4}})
    with pytest.raises(ValueError, match="explicit FB reward baseline"):
        build_fb_agent(overrides={"aux_rewards_scaling": {"penalty_action_rate": -0.1}}, cartwheel_aux_safe=True)


@pytest.mark.parametrize("baseline", [
    {"train": {"lr_actor": "0.0001"}},
    {"model": {"archi": {"actor": {"hidden_dim": "invalid"}}}},
    {"model": {"archi": {"shared_critic": "yes"}}},
])
def test_fb_cli_override_does_not_hide_invalid_robot_baseline(baseline):
    with pytest.raises(ValueError):
        build_fb_agent(overrides=baseline, network_width=1024, shared_critic=False)


@pytest.mark.parametrize("paths", ["motions.pkl", ["first.pkl", "second.pkl"]])
def test_motion_data_name_roundtrip_rejects_retired_alias(paths):
    cfg = MJLabEnvConfig(motion_data_path=paths)
    assert MJLabEnvConfig.model_validate_json(cfg.model_dump_json()) == cfg
    assert cfg.model_dump()["motion_data_path"] == paths
    old = {**cfg.model_dump(), "lafan_tail_path": paths}
    del old["motion_data_path"]
    with pytest.raises(ValueError, match="extra_forbidden"):
        MJLabEnvConfig.model_validate(old)


if __name__ == "__main__":
    unittest.main()
