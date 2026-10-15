"""Public CLI selection and checkpoint-driven multi-robot inference contracts."""
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from ufo.envs.config import resolve_task_config
from ufo.inference.goal_inference import _find_goal_json
from ufo.inference.runtime import load_mjlab_env_cfg
from ufo.inference.tracking_inference import select_motion_ids
from ufo.tools.robot_config import builtin_robot_config
from ufo.train import build_mjlab_config, parse_args


def config(robot="g1", agent="fb"):
    return build_mjlab_config(device="cpu", work_dir="/tmp/entrypoint", num_envs=1,
                             num_env_steps=16, seed=1, use_wandb=False, wandb_run_name=None,
                             robot_config=builtin_robot_config(robot), agent=agent)


@pytest.mark.parametrize("robot", ["g1", "rp1"])
@pytest.mark.parametrize("agent", ["fb", "tech"])
def test_robot_algorithm_cli_defaults(monkeypatch, robot, agent):
    monkeypatch.setattr(sys, "argv", ["train", "--robot", robot, "--agent", agent])
    args = parse_args()
    assert args.robot_config == builtin_robot_config(robot)
    assert args.work_dir == f"runs/{robot}_{agent}"
    assert ("rp1_24dof" in args.data_path) == (robot == "rp1")
    assert args.num_env_steps == (384000000 if robot == "rp1" else 192000000)
    cfg = config(robot, agent)
    assert cfg.env.robot_training["robot"]["name"] == ("g1_29dof" if robot == "g1" else "rp1_24dof")


def test_cli_robot_sources_are_exclusive(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train", "--robot", "rp1", "--robot-config", str(builtin_robot_config("g1"))])
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("robot", ["g1", "rp1"])
def test_inference_restores_saved_environment_not_current_yaml(tmp_path, robot):
    cfg = config(robot)
    cfg.env.task_overrides.setdefault("obs", {}).setdefault("obs_scales", {})["base_ang_vel"] = .123
    cfg.env.robot_training["action_scale"] = .321
    cfg = cfg.model_copy(update={"env": cfg.env.model_copy(update={"motion_data_path": str(tmp_path / "saved_robot_motions.pkl")})})
    (tmp_path / "config.json").write_text(cfg.model_dump_json())
    restored, _ = load_mjlab_env_cfg(tmp_path, data_path=None, device="cpu",
                                    disable_dr=True, disable_obs_noise=True, max_episode_length_s=20)
    assert restored.motion_data_path == cfg.env.motion_data_path
    task = resolve_task_config(restored, 1)
    assert task.obs.obs_scales.base_ang_vel == .123
    assert task.robot.control.action_scale == .321
    assert restored.robot_config_path == str(builtin_robot_config(robot))
    other = builtin_robot_config("rp1" if robot == "g1" else "g1")
    with pytest.raises(ValueError, match="does not match checkpoint"):
        load_mjlab_env_cfg(tmp_path, data_path=None, robot_config=other, device="cpu",
                           disable_dr=True, disable_obs_noise=True, max_episode_length_s=20)


def test_motion_names_are_unambiguous_and_ordered():
    assert select_motion_ids(["dance.pkl", "fall.pkl"], ["fall", "dance"]) == [1, 0]
    for keys, name in [(["dance.pkl"], "missing"), (["a/dance.pkl", "b/dance.pkl"], "dance")]:
        with pytest.raises(ValueError, match="matches"):
            select_motion_ids(keys, [name])


def test_non_g1_with_29_joints_cannot_use_g1_goal_defaults():
    with pytest.raises(ValueError, match="requires --goal-json"):
        _find_goal_json(None, num_dof=29, robot_name="another_robot")


@pytest.mark.parametrize("robot", ["g1", "rp1"])
@pytest.mark.parametrize("agent", ["fb", "tech"])
def test_inference_model_dimensions_and_supported_latents(robot, agent):
    torch.set_num_threads(1)
    cfg = config(robot, agent)
    task = resolve_task_config(cfg.env, 1)
    n = task.robot.actions_dim
    dims = {"state": 2*n+6, "last_action": n, "history_actor": task.obs.obs_dims.history_actor,
            "privileged_state": task.obs.obs_dims.max_local_self + 1}
    space = gym.spaces.Dict({k: gym.spaces.Box(-np.inf, np.inf, (v,), dtype=np.float32) for k,v in dims.items()})
    arch = cfg.agent.model.archi
    small = {"z_dim": 8}
    for name in ("f", "b", "actor", "critic", "aux_critic", "discriminator", "goal_encoder"):
        network = getattr(arch, name, None)
        if network is not None:
            small[name] = network.model_copy(update={"hidden_dim": 16, "hidden_layers": 2})
    model_cfg = cfg.agent.model.model_copy(update={"device": "cpu", "archi": arch.model_copy(update=small)})
    model = model_cfg.build(space, n).eval()
    obs = {k: torch.randn(4, v) for k,v in dims.items()}
    for method in (model.tracking_inference, model.goal_inference):
        z = method(obs)
        actions = model.act(obs, z, mean=True)
        assert actions.shape == (4, n)
        assert torch.isfinite(actions).all()
    if agent == "fb":
        z = model.reward_wr_inference(obs, torch.ones(4, 1))
        assert torch.isfinite(z).all() and z.shape[-1] == 8
    else:
        assert not hasattr(model, "reward_wr_inference")
        torch.testing.assert_close(model.goal_inference(obs), model.tracking_inference(obs), rtol=0, atol=0)
        model.requires_grad_(True)
        assert not model.goal_inference(obs).requires_grad
        encoded = model.encode_goal(obs)
        assert encoded.requires_grad
        encoded.square().mean().backward()
        assert any(p.grad is not None for p in model._goal_encoder.parameters())


def test_robot_setup_checks_reviewed_config_and_builds_real_manifest(tmp_path):
    from ufo.tools.robot_setup import main
    config_path = tmp_path / "robot.yaml"
    config_path.write_text(builtin_robot_config("g1").read_text())
    # Preserve portable asset resolution when the robot YAML is placed elsewhere.
    from omegaconf import OmegaConf
    data = OmegaConf.load(config_path)
    root = builtin_robot_config("g1").parents[2]
    data.xml_path = str(root / data.xml_path)
    OmegaConf.save(data, config_path)
    csv = tmp_path / "motion.csv"
    values = np.zeros((60, 7+29))
    values[:,2] = .8
    values[:,6] = 1
    np.savetxt(csv, values, delimiter=",")
    main(["--name", "new_motion", "--robot-config", str(config_path), "--motion-source", str(csv), "--fps", "50"])
    report = json.loads((tmp_path / "robot_setup.json").read_text())
    assert Path(report["data_manifest"]).exists()
    assert "--smoke" in report["smoke_command"]
    assert report["robot"] == "g1_29dof"
    with pytest.raises(SystemExit):
        main(["--name", "new_motion", "--robot-config", str(config_path)])


@pytest.mark.parametrize("name,allowed", [("FBcprAuxModel", True), ("TeCHModel", False)])
def test_reward_rejects_tech_before_loading_model_or_replay(tmp_path, name, allowed):
    from ufo.inference.reward_inference import _require_fb_checkpoint
    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "checkpoint/config.json").write_text(json.dumps({"model": {"name": name}}))
    if allowed:
        _require_fb_checkpoint(tmp_path)
    else:
        with pytest.raises(ValueError, match="requires an FB checkpoint"):
            _require_fb_checkpoint(tmp_path)


def test_goal_metadata_rejects_same_dimension_wrong_robot_and_joint_order(tmp_path):
    from ufo.inference.goal_inference import load_and_validate_goal_json
    path = tmp_path / "goal.json"
    path.write_text(json.dumps({"robot_name": "custom", "control_joint_names": ["a", "b"],
                                "goals": [{"motion_id": 0, "motion_name": "dance", "frames": [0]}]}))
    assert len(load_and_validate_goal_json(path, num_dof=2, robot_name="custom", control_joint_names=["a", "b"])) == 1
    for robot, joints in [("other", ["a", "b"]), ("custom", ["b", "a"])]:
        with pytest.raises(ValueError, match="match"):
            load_and_validate_goal_json(path, num_dof=2, robot_name=robot, control_joint_names=joints)



def test_robot_setup_generates_a_draft_without_claiming_calibration(tmp_path):
    from ufo.tools.robot_config import load_robot_training_spec
    from ufo.tools.robot_setup import main
    source = load_robot_training_spec(builtin_robot_config("g1"))
    output = tmp_path / "new_robot.yaml"
    main(["--name", "new_robot", "--xml", source.robot.xml_path, "--output", str(output)])
    report = json.loads((tmp_path / "new_robot_setup.json").read_text())
    assert output.exists() and report["robot"] == "new_robot"
    assert "draft" in report["status"] and report["warnings"]
    assert report["smoke_command"] is None



def test_missing_data_reports_setup_before_launch(tmp_path):
    from argparse import Namespace

    from ufo.train import launch
    run = tmp_path / "run"
    with pytest.raises(FileNotFoundError, match="download_data.sh"):
        launch(Namespace(data_path=str(tmp_path / "missing.pkl"), work_dir=str(run)))
    assert not run.exists()



def test_inference_keeps_multisource_data_and_clears_weights_on_override(tmp_path):
    cfg = config("rp1")
    paths = [str(tmp_path / "first.pkl"), str(tmp_path / "second.pkl")]
    cfg = cfg.model_copy(update={"env": cfg.env.model_copy(update={"motion_data_path": paths, "data_mix_weights": [.8, .2]})})
    (tmp_path / "config.json").write_text(cfg.model_dump_json())
    kwargs = dict(device="cpu", disable_dr=True, disable_obs_noise=True, max_episode_length_s=20)
    saved, _ = load_mjlab_env_cfg(tmp_path, data_path=None, **kwargs)
    assert saved.motion_data_path == paths and saved.data_mix_weights == [.8, .2]
    replaced, _ = load_mjlab_env_cfg(tmp_path, data_path=tmp_path / "full.pkl", **kwargs)
    assert replaced.data_mix_weights is None


@pytest.mark.parametrize("robot", ["g1", "rp1"])
def test_inference_loads_robot_once_and_returns_matching_spec(tmp_path, monkeypatch, robot):
    from unittest.mock import Mock

    from ufo.inference import runtime

    cfg = config(robot)
    (tmp_path / "config.json").write_text(cfg.model_dump_json())
    load = Mock(wraps=runtime.load_robot_training_spec)
    monkeypatch.setattr(runtime, "load_robot_training_spec", load)
    restored, spec = runtime.load_mjlab_env_cfg(
        tmp_path, data_path=None, device="cpu", disable_dr=True, disable_obs_noise=True, max_episode_length_s=20,
    )
    load.assert_called_once()
    assert spec.robot.xml_path == restored.mjcf_path
    assert spec.robot.name == cfg.env.robot_training["robot"]["name"]


def test_tracking_mae_batches_unequal_motions_without_scalar_gpu_reads(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from ufo.tools import evaluate_mae as mae

    (tmp_path / "checkpoint").mkdir()
    data_path = tmp_path / "motions.pkl"
    data_path.touch()
    lengths = [3, 1, 2]
    targets = [torch.arange((n + 1) * 2, dtype=torch.float32).reshape(-1, 2) + 10 * i for i, n in enumerate(lengths)]
    library = SimpleNamespace(_num_unique_motions=3, load_all_motions=Mock())
    prep = SimpleNamespace(_env=SimpleNamespace(num_dof=2, _motion_lib=library), close=Mock())
    built = []
    contexts = []

    def build(num_envs):
        if not built:
            built.append(prep)
            return prep, {}
        start = 2 * (len(built) - 1)
        core = SimpleNamespace(dof_pos=torch.zeros(num_envs, 2))
        tick = 0

        def step(action, to_numpy):
            nonlocal tick
            tick += 1
            core.dof_pos = torch.tensor([[start + i + tick, start + i - tick] for i in range(num_envs)], dtype=torch.float32)
            return {}, 0, False, False, {}

        wrapped = SimpleNamespace(_env=core, reset=lambda **kw: ({}, {}), step=step, close=Mock())
        built.append(wrapped)
        return wrapped, {}

    cfg = SimpleNamespace(build=build, root_height_obs=False)
    load_config = Mock(return_value=(cfg, None))
    monkeypatch.setattr(mae, "load_mjlab_env_cfg", load_config)
    model = Mock()
    model.act.side_effect = lambda obs, z, mean: contexts.append(z.clone()) or torch.zeros(len(z), 2)
    monkeypatch.setattr(mae, "load_model_from_checkpoint_dir", lambda *a, **kw: model)
    monkeypatch.setattr(mae, "get_backward_observation", lambda env, mid, **kw: (
        {"state": torch.arange(lengths[mid] + 1, dtype=torch.float32).reshape(-1, 1) + mid * 10}, {"dof_pos": targets[mid]},
    ))
    model.tracking_inference.side_effect = lambda obs: obs["state"]
    monkeypatch.setattr(mae, "_target_states_from_obs", lambda *a, **kw: {
        "root_states": torch.zeros(1, 13), "dof_states": torch.zeros(1, 2, 2),
    })
    args = mae.parse_args(["tracking", "--model-folder", str(tmp_path), "--data-path", str(data_path),
                           "--device", "cpu", "--chunk-envs", "2"])
    with monkeypatch.context() as guarded:
        def scalar_read(*a, **kw):
            raise AssertionError("Tracking MAE must not read a GPU scalar for each motion/step")
        guarded.setattr(torch.Tensor, "item", scalar_read)
        result = json.loads(mae.evaluate_tracking_joint_mae(args).read_text())
    expected = []
    for i, length in enumerate(lengths):
        predictions = np.asarray([[i + step, i - step] for step in range(1, length + 1)])
        errors = np.abs(predictions - targets[i].numpy()[1:]).mean(axis=-1)
        expected.extend(errors)
        assert result["per_motion"][str(i)]["num_steps"] == length
        assert result["per_motion"][str(i)]["mean_mae"] == pytest.approx(errors.mean())
        assert result["per_motion"][str(i)]["std_mae"] == pytest.approx(errors.std())
    assert result["num_total_steps"] == 6
    assert result["global_mean"] == pytest.approx(np.mean(expected))
    assert result["global_std"] == pytest.approx(np.std(expected))
    torch.testing.assert_close(contexts[2], torch.tensor([[3.], [11.]]))
    load_config.assert_called_once()
    for env in built:
        env.close.assert_called_once()


@pytest.mark.parametrize("agent", ["fb", "tech"])
def test_tracking_latents_are_per_frame_independent_of_sequence_length(agent):
    from types import SimpleNamespace

    from ufo.agents.fb_cpr_aux.model import FBcprAuxModel
    from ufo.agents.tech.model import TeCHModel

    model_type = FBcprAuxModel if agent == "fb" else TeCHModel
    model = model_type.__new__(model_type)
    torch.nn.Module.__init__(model)
    model.cfg = SimpleNamespace(archi=SimpleNamespace(norm_z=False, z_geometry="sphere"), seq_length=2)
    model.backward_map = lambda obs: obs.clone()
    obs = torch.tensor([[1.0], [3.0], [5.0]])
    torch.testing.assert_close(model.tracking_inference(obs), obs, rtol=0, atol=0)


def test_goal_mae_checks_motion_identity_and_skips_invalid_frames(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from ufo.tools import evaluate_mae as mae

    (tmp_path / "checkpoint").mkdir()
    path = tmp_path / "goals.json"
    goal = {"motion_id": 0, "motion_name": "dance", "frames": [-1, 1, 4]}
    path.write_text(json.dumps([goal]))
    env = SimpleNamespace(num_dof=2, dof_pos=torch.zeros(1, 2), set_is_evaluating=Mock(),
                          _motion_lib=SimpleNamespace(_motion_data_keys=["dance"]))
    wrapped = SimpleNamespace(_env=env, close=Mock(), reset=Mock(return_value=({}, {})),
                              step=Mock(return_value=({}, 0, False, False, {})))
    cfg = SimpleNamespace(robot_training={"robot": {"name": "custom", "control_joint_names": ["a", "b"]}},
                          motion_data_path="motions.pkl", root_height_obs=False, build=lambda **kw: (wrapped, {}))
    model = Mock(goal_inference=Mock(return_value=torch.zeros(1, 2)))
    monkeypatch.setattr(mae, "load_mjlab_env_cfg", lambda *a, **kw: (cfg, False))
    monkeypatch.setattr(mae, "load_model_from_checkpoint_dir", lambda *a, **kw: model)
    monkeypatch.setattr(mae, "get_goal_observation", lambda *a, **kw: ({"state": torch.zeros(3, 2)}, {"dof_pos": torch.zeros(3, 2)}))
    monkeypatch.setattr(mae, "_target_states_from_obs", lambda *a, **kw: {})
    kwargs = dict(model_folder=tmp_path, data_path=None, goal_json=path, device="cpu", steps=1,
                  disable_dr=True, disable_obs_noise=True, max_episode_length_s=10, output=None)
    result = json.loads(mae.evaluate_goal_joint_mae(**kwargs).read_text())
    assert result["num_goals"] == 1
    assert result["per_goal"]["dance_-1"]["skipped"]
    assert result["per_goal"]["dance_4"]["skipped"]
    assert wrapped.step.call_count == 1
    goal["motion_name"] = "wrong_motion"
    path.write_text(json.dumps([goal]))
    with pytest.raises(ValueError, match="motion name"):
        mae.evaluate_goal_joint_mae(**kwargs)
    path.write_text(json.dumps({"robot_name": "wrong_robot", "control_joint_names": ["a", "b"], "goals": [goal]}))
    with pytest.raises(ValueError, match="robot_name"):
        mae.evaluate_goal_joint_mae(**kwargs)


def test_backward_export_preserves_cuda_device_selection(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from ufo.tools import export_onnx as tracking

    (tmp_path / "checkpoint").mkdir()
    (tmp_path / "config.json").write_text(config().model_dump_json())
    output = tmp_path / "backward_encoder.onnx"
    output.write_bytes(b"mock ONNX")
    select = Mock()
    load = Mock(return_value=object())
    monkeypatch.setattr(torch.cuda, "set_device", select)
    monkeypatch.setattr(tracking, "load_model_from_checkpoint_dir", load)
    monkeypatch.setattr(tracking, "export_backward_encoder_from_model", Mock(return_value=output))
    monkeypatch.setattr(tracking, "_deployment_contract", Mock(return_value={"history_oldest_first": False}))
    tracking.export_backward_encoder_from_checkpoint(tmp_path, device="cuda:3")
    select.assert_called_once_with(torch.device("cuda:3"))
    assert load.call_args.kwargs["device"] == "cuda"
    assert json.loads(output.with_suffix(".meta.json").read_text())["history_oldest_first"] is False


@pytest.mark.parametrize("keep_action", [False, True])
@pytest.mark.parametrize("mismatch", [False, True])
def test_backward_onnx_verification_uses_exported_inputs(tmp_path, monkeypatch, keep_action, mismatch):
    import sys
    from types import SimpleNamespace

    from ufo.tools.export_onnx import _verify_backward_encoder_onnx

    names = ["state", "privileged_state"] + (["last_action"] if keep_action else [])
    state, action, privileged = torch.ones(2, 3), torch.full((2, 3), 2.0), torch.full((2, 3), 3.0)

    def wrapper(state, action, privileged):
        return state + privileged + (action if keep_action else 0)

    def run(outputs, inputs):
        assert outputs == ["z"]
        assert set(inputs) == set(names)
        result = inputs["state"] + inputs["privileged_state"]
        if keep_action:
            result = result + inputs["last_action"]
        return [result + int(mismatch)]

    session = SimpleNamespace(get_inputs=lambda: [SimpleNamespace(name=name) for name in names], run=run)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=lambda *a, **kw: session))
    if mismatch:
        with pytest.raises(RuntimeError, match="ONNX verify failed"):
            _verify_backward_encoder_onnx(wrapper, tmp_path / "encoder.onnx", state, action, privileged, atol=1e-4, rtol=1e-4)
    else:
        _verify_backward_encoder_onnx(wrapper, tmp_path / "encoder.onnx", state, action, privileged, atol=1e-4, rtol=1e-4)
