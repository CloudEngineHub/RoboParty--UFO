"""Policy and FB backward-encoder ONNX export, shared by all inference modes."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ufo.agents.common.checkpoint import checkpoint_load_device, load_model_from_checkpoint_dir

_SUPPORTED_POLICY_ACTOR_INPUT_KEYS = ("state", "last_action", "history_actor")


def _as_actor_input_keys(raw_keys: Any) -> list[str]:
    if isinstance(raw_keys, str):
        keys = [raw_keys]
    else:
        try:
            keys = [str(key) for key in raw_keys]
        except TypeError as exc:
            raise TypeError("cfg.archi.actor.input_filter.key must be a string or sequence of strings") from exc
    if not keys:
        raise ValueError("cfg.archi.actor.input_filter.key must contain at least one key")
    unsupported = [key for key in keys if key not in _SUPPORTED_POLICY_ACTOR_INPUT_KEYS]
    if unsupported:
        raise ValueError(
            "Unsupported actor input key(s) for policy ONNX export: "
            f"{unsupported}. Supported keys: {list(_SUPPORTED_POLICY_ACTOR_INPUT_KEYS)}"
        )
    return keys


def _get_policy_actor_input_keys(inference_model: nn.Module) -> list[str]:
    try:
        raw_keys = inference_model.cfg.archi.actor.input_filter.key
    except AttributeError as exc:
        raise AttributeError("inference_model.cfg.archi.actor.input_filter.key is required for policy ONNX export") from exc
    return _as_actor_input_keys(raw_keys)


def _obs_space_mapping(inference_model: nn.Module) -> Any:
    if not hasattr(inference_model, "obs_space"):
        raise AttributeError("inference_model must expose obs_space for policy ONNX export")
    obs_space = inference_model.obs_space
    if hasattr(obs_space, "spaces"):
        return obs_space.spaces
    if isinstance(obs_space, dict):
        return obs_space
    raise TypeError(f"inference_model.obs_space must be a Dict-like space with a .spaces mapping or a dict, got {type(obs_space)}")


def _require_1d_box_like_space(space: Any, key: str) -> int:
    shape = getattr(space, "shape", None)
    if shape is None:
        raise TypeError(f"obs_space['{key}'] must be a 1D Box-like space with shape, got {type(space)}")
    shape = tuple(shape)
    if len(shape) != 1:
        raise ValueError(f"obs_space['{key}'] must be 1D, got shape={shape}")
    return int(shape[0])


def _actor_input_dims_from_obs_space(inference_model: nn.Module, actor_input_keys: list[str]) -> dict[str, int]:
    spaces = _obs_space_mapping(inference_model)
    dims: dict[str, int] = {}
    for key in actor_input_keys:
        if key not in spaces:
            raise KeyError(f"Actor input key '{key}' is missing from inference_model.obs_space")
        dims[key] = _require_1d_box_like_space(spaces[key], key)
    return dims


def _actor_filter_output_dim(inference_model: nn.Module) -> int:
    try:
        output_space = inference_model._actor.input_filter.output_space
    except AttributeError as exc:
        raise AttributeError("inference_model._actor.input_filter.output_space is required for policy ONNX export") from exc
    return _require_1d_box_like_space(output_space, "_actor.input_filter.output_space")


def _policy_output_action_dim(inference_model: nn.Module) -> int:
    if not hasattr(inference_model, "action_dim"):
        raise AttributeError("inference_model.action_dim is required for policy ONNX export metadata")
    return int(inference_model.action_dim)


def _infer_policy_onnx_export_spec(inference_model: nn.Module, z_dim: int) -> dict[str, Any]:
    actor_input_keys = _get_policy_actor_input_keys(inference_model)
    actor_input_dims = _actor_input_dims_from_obs_space(inference_model, actor_input_keys)
    actor_input_dim = sum(actor_input_dims.values())
    filter_output_dim = _actor_filter_output_dim(inference_model)
    if actor_input_dim != filter_output_dim:
        raise ValueError(
            "Actor input dims inferred from obs_space do not match _actor.input_filter.output_space: "
            f"sum(actor_input_dims)={actor_input_dim}, output_space_dim={filter_output_dim}, "
            f"actor_input_dims={actor_input_dims}"
        )
    return {
        "actor_input_keys": actor_input_keys,
        "actor_input_dims": actor_input_dims,
        "z_dim": int(z_dim),
        "actor_input_dim": actor_input_dim,
        "actor_obs_dim": actor_input_dim + int(z_dim),
        "output_action_dim": _policy_output_action_dim(inference_model),
    }


def export_meta_policy_as_onnx(
    inference_model,
    path,
    exported_policy_name,
    example_obs_dict: dict[str, torch.Tensor] | None = None,
    z_dim: int | None = None,
    *,
    batch_size: int = 1,
    opset_version: int = 13,
):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, exported_policy_name)
    if z_dim is None:
        try:
            z_dim = int(inference_model.cfg.archi.z_dim)
        except AttributeError as exc:
            raise AttributeError("z_dim must be provided or available at inference_model.cfg.archi.z_dim") from exc
    z_dim = int(z_dim)
    if z_dim <= 0:
        raise ValueError(f"z_dim must be positive, got {z_dim}")
    export_spec = _infer_policy_onnx_export_spec(inference_model, z_dim)

    class PolicyExportWrapper(nn.Module):
        def __init__(self, actor, actor_input_keys, actor_input_dims, z_dim):
            """
            model: The original PyTorch model.
            input_keys: List of input names as keys for the input dictionary.
            """
            super(PolicyExportWrapper, self).__init__()
            self.actor = copy.deepcopy(actor._actor).to("cpu").eval()
            self.obs_normalizer = copy.deepcopy(actor._obs_normalizer).to("cpu").eval()
            self.actor_std = actor.cfg.actor_std
            self.actor_input_keys = list(actor_input_keys)
            self.actor_input_dims = dict(actor_input_dims)
            self.z_dim = int(z_dim)

        def forward(self, actor_obs):
            """
            Dynamically creates a dictionary from the input keys and args.
            """
            actor_obs_without_z = actor_obs[:, : -self.z_dim]
            ctx = actor_obs[:, -self.z_dim :]
            actor_dict = {}
            cursor = 0
            for key in self.actor_input_keys:
                dim = self.actor_input_dims[key]
                actor_dict[key] = actor_obs_without_z[:, cursor : cursor + dim]
                cursor += dim

            return self.actor(self.obs_normalizer(actor_dict), ctx, self.actor_std).mean.float()

    wrapper = PolicyExportWrapper(
        inference_model,
        actor_input_keys=export_spec["actor_input_keys"],
        actor_input_dims=export_spec["actor_input_dims"],
        z_dim=export_spec["z_dim"],
    )
    if example_obs_dict is None:
        example_input_list = torch.randn(int(batch_size), export_spec["actor_obs_dim"], dtype=torch.float32)
    else:
        if "actor_obs" not in example_obs_dict:
            raise KeyError("example_obs_dict must contain 'actor_obs'")
        example_input_list = example_obs_dict["actor_obs"]
        if int(example_input_list.shape[-1]) != export_spec["actor_obs_dim"]:
            raise ValueError(
                "example_obs_dict['actor_obs'] has wrong last dimension: "
                f"expected {export_spec['actor_obs_dim']}, got {int(example_input_list.shape[-1])}"
            )
    torch.onnx.export(
        wrapper,
        example_input_list,  # Pass x1 and x2 as separate inputs
        path,
        verbose=False,
        input_names=["actor_obs"],  # Specify the input names
        output_names=["action"],  # Name the output
        dynamic_axes={
            "actor_obs": {0: "batch"},
            "action": {0: "batch"},
        },
        opset_version=int(opset_version),  # Specify the opset version, if needed
    )
    return export_spec


class UnsupportedBackwardEncoderExport(RuntimeError):
    """Raised when a loaded policy does not expose an FB backward encoder."""


def _require_1d_box(space: gym.spaces.Space, key: str) -> int:
    if not isinstance(space, gym.spaces.Box):
        raise TypeError(f"obs_space['{key}'] must be gym.spaces.Box, got {type(space)}")
    if len(space.shape) != 1:
        raise ValueError(f"obs_space['{key}'] must be 1D, got shape={space.shape}")
    return int(space.shape[0])


def _require_backward_encoder_model(model: nn.Module) -> None:
    missing = [name for name in ("_obs_normalizer", "_backward_map", "obs_space", "cfg") if not hasattr(model, name)]
    if missing:
        raise UnsupportedBackwardEncoderExport(
            f"Policy does not expose an FB backward encoder; missing attributes: {missing}. TeCH policies are skipped."
        )


class BackwardEncoderWrapper(nn.Module):
    """ONNX-exportable wrapper for backward latent inference."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.obs_normalizer = model._obs_normalizer
        self.backward_map = model._backward_map
        self.use_project_z = bool(model.cfg.archi.norm_z)

    def forward(
        self,
        state: torch.Tensor,
        last_action: torch.Tensor,
        privileged_state: torch.Tensor,
    ) -> torch.Tensor:
        obs = {
            "state": state,
            "last_action": last_action,
            "privileged_state": privileged_state,
        }
        normalized_obs = self.obs_normalizer(obs)
        z = self.backward_map(normalized_obs)
        if self.use_project_z:
            z = math.sqrt(z.shape[-1]) * F.normalize(z, dim=-1)
        return z


def export_backward_encoder_from_model(
    model: nn.Module,
    output_path: str | Path,
    *,
    batch_size: int = 1,
    opset: int = 13,
    verify: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> Path:
    """Export a loaded FB policy's backward encoder to ONNX."""

    _require_backward_encoder_model(model)

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy only encoder inputs and weights, leaving the caller's model untouched.
    wrapper = copy.deepcopy(BackwardEncoderWrapper(model)).to("cpu").eval()

    if not isinstance(model.obs_space, gym.spaces.Dict):
        raise TypeError(f"Expected dict obs space, got {type(model.obs_space)}")
    required_keys = ("state", "last_action", "privileged_state")
    missing = [k for k in required_keys if k not in model.obs_space.spaces]
    if missing:
        raise KeyError(f"Missing required obs keys for backward encoder export: {missing}")

    state_dim = _require_1d_box(model.obs_space["state"], "state")
    action_dim = _require_1d_box(model.obs_space["last_action"], "last_action")
    priv_dim = _require_1d_box(model.obs_space["privileged_state"], "privileged_state")

    batch = int(batch_size)
    if batch <= 0:
        raise ValueError(f"batch_size must be >=1, got {batch}")

    example_state = torch.randn(batch, state_dim, dtype=torch.float32)
    example_last_action = torch.randn(batch, action_dim, dtype=torch.float32)
    example_privileged_state = torch.randn(batch, priv_dim, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        (example_state, example_last_action, example_privileged_state),
        str(output_path),
        input_names=["state", "last_action", "privileged_state"],
        output_names=["z"],
        dynamic_axes={
            "state": {0: "batch"},
            "last_action": {0: "batch"},
            "privileged_state": {0: "batch"},
            "z": {0: "batch"},
        },
        opset_version=int(opset),
        do_constant_folding=True,
        verbose=False,
    )

    print(f"[INFO] Exported backward encoder ONNX: {output_path}")
    print(
        "[INFO] Backward encoder input dims: "
        f"state={state_dim}, last_action={action_dim}, privileged_state={priv_dim}; "
        f"z_dim={model.cfg.archi.z_dim}"
    )

    if verify:
        _verify_backward_encoder_onnx(
            wrapper,
            output_path,
            example_state,
            example_last_action,
            example_privileged_state,
            atol=atol,
            rtol=rtol,
        )
    return output_path


def export_backward_encoder_from_checkpoint(
    model_folder: Path,
    output: Path | None = None,
    *,
    checkpoint_subdir: str = "checkpoint",
    device: str = "cuda",
    batch_size: int = 1,
    opset: int = 13,
    verify: bool = True,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> Path:
    """Load a checkpoint and export its FB backward encoder to ONNX."""

    model_folder = Path(model_folder).expanduser().resolve()
    checkpoint_dir = model_folder / checkpoint_subdir
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")

    output_path = (model_folder / "exported" / "backward_encoder.onnx") if output is None else Path(output)
    model = load_model_from_checkpoint_dir(checkpoint_dir, device=checkpoint_load_device(device))
    from ufo.envs.config import MJLabEnvConfig

    saved = json.loads((model_folder / "config.json").read_text())
    metadata = _deployment_contract(model, MJLabEnvConfig.model_validate(saved["env"]))
    result = export_backward_encoder_from_model(
        model,
        output_path,
        batch_size=batch_size,
        opset=opset,
        verify=verify,
        atol=atol,
        rtol=rtol,
    )
    metadata["onnx_sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
    result.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return result


def _verify_backward_encoder_onnx(
    wrapper: nn.Module,
    output_path: Path,
    example_state: torch.Tensor,
    example_last_action: torch.Tensor,
    example_privileged_state: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> None:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError:
        print("[INFO] Skip backward encoder ONNX verify: onnxruntime is not installed.")
        return

    with torch.no_grad():
        torch_z = (
            wrapper(
                example_state,
                example_last_action,
                example_privileged_state,
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    ort_session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    inputs = {
        "state": example_state.numpy().astype(np.float32),
        "last_action": example_last_action.numpy().astype(np.float32),
        "privileged_state": example_privileged_state.numpy().astype(np.float32),
    }
    # ONNX removes inputs that the checkpoint's encoder does not use.
    ort_z = ort_session.run(["z"], {item.name: inputs[item.name] for item in ort_session.get_inputs()})[0]

    max_abs = float(np.max(np.abs(torch_z - ort_z)))
    if not np.allclose(torch_z, ort_z, atol=float(atol), rtol=float(rtol)):
        raise RuntimeError(f"Backward encoder ONNX verify failed: max_abs={max_abs:.6e}, atol={atol}, rtol={rtol}")
    print(f"[INFO] Backward encoder ONNX verify passed: max_abs={max_abs:.6e}")


def _deployment_contract(model, env_config) -> dict[str, Any]:
    from omegaconf import OmegaConf
    from ufo.envs.config import resolve_task_config, _action_target_scale_values, _match_joint_value

    task = resolve_task_config(env_config, 1)
    names = list(task.robot.dof_names)
    default = [float(task.robot.init_state.default_joint_angles[name]) for name in names]
    scales = OmegaConf.to_container(task.obs.obs_scales, resolve=True)
    observation = {}
    fields = {"dof_pos": "dof_pos_minus_default", "dof_vel": "dof_vel", "projected_gravity": "projected_gravity", "base_ang_vel": "base_ang_vel", "actions": "prev_actions"}
    for key in _get_policy_actor_input_keys(model):
        if key == "state":
            for field in ("dof_pos", "dof_vel", "projected_gravity", "base_ang_vel"):
                observation[fields[field]] = {"scale": float(scales[field])}
                if field == "dof_pos":
                    observation[fields[field]]["default_pos"] = default
        elif key == "last_action":
            observation["prev_actions"] = {"scale": float(scales["actions"])}
        elif key == "history_actor":
            for field, steps in sorted(task.obs.obs_auxiliary.history_actor.items()):
                entry = {"steps": int(steps), "scale": float(scales[field]) * float(scales["history_actor"])}
                if field == "dof_pos":
                    entry["default_pos"] = default
                observation[fields[field] + "_history"] = entry
    deployment = {
        "observation": {"policy": observation},
        "isaac_joint_names": names,
        "policy_joint_names": names,
        "default_joint_pos": dict(zip(names, default)),
        "action_scale": dict(zip(names, _action_target_scale_values(task))),
        "action_rescale": float(task.robot.control.normalize_action_to),
        "joint_kp": {name: _match_joint_value(name, task.robot.control.stiffness) for name in names},
        "joint_kd": {name: _match_joint_value(name, task.robot.control.damping) for name in names},
    }
    return {
        "deployment": deployment,
        "observation_order": list(observation),
        "history_includes_current": False,
        "history_oldest_first": bool(task.obs.history_oldest_first),
        "reference_observation_scales": scales,
        "reference_body_names": list(task.robot.body_names),
        "reference_motion_extensions": OmegaConf.to_container(task.robot.motion.extend_config, resolve=True),
        "root_height_obs": bool(task.obs.root_height_obs),
    }


def export_policy_model(model: torch.nn.Module, output_dir: Path, robot_training: Any, *, env_config=None) -> dict[str, Any]:
    contract = _deployment_contract(model, env_config) if env_config is not None else None
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = model.__class__.__name__
    output_name = f"{model_name}.onnx"
    control_joint_names = list(robot_training.robot.control_joint_names)
    num_dof = len(control_joint_names)
    export_metadata = export_meta_policy_as_onnx(
        model,
        output_dir,
        output_name,
        z_dim=model.cfg.archi.z_dim,
    )
    if int(export_metadata["output_action_dim"]) != num_dof:
        raise ValueError(
            "Policy action dim does not match robot control joint count: "
            f"output_action_dim={export_metadata['output_action_dim']}, num_dof={num_dof}"
        )
    export_metadata.update(
        {
            "robot_name": robot_training.robot.name,
            "robot_config_path": str(Path(robot_training.config_path).expanduser().resolve()),
            "xml_path": str(Path(robot_training.robot.xml_path).expanduser().resolve()),
            "num_dof": num_dof,
            "control_joint_names": control_joint_names,
        }
    )
    if env_config is not None:
        import yaml
        export_metadata.update(contract)
        deploy_path = output_dir / f"{model_name}.deploy.yaml"
        deploy_path.write_text(yaml.safe_dump(export_metadata["deployment"], sort_keys=False))
    model_path = output_dir / output_name
    if model_path.is_file():
        export_metadata["onnx_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata_path = output_dir / f"{model_name}.meta.json"
    metadata_path.write_text(json.dumps(export_metadata, indent=2, sort_keys=True) + "\n")
    print(f"[INFO] Exported model to {output_dir / output_name}")
    print(f"[INFO] Wrote policy ONNX metadata to {metadata_path}")
    return export_metadata


def export_tracking_onnx(model: torch.nn.Module, output_dir: Path, robot_training: Any, *, env_config=None) -> None:
    metadata = export_policy_model(model, output_dir, robot_training, env_config=env_config)
    try:
        export_backward_encoder_from_model(model, output_dir / "backward_encoder.onnx")
        if env_config is not None:
            metadata["onnx_sha256"] = hashlib.sha256((output_dir / "backward_encoder.onnx").read_bytes()).hexdigest()
            (output_dir / "backward_encoder.meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    except UnsupportedBackwardEncoderExport as exc:
        print(f"[INFO] Skip backward encoder ONNX export: {exc}")


if __name__ == "__main__":
    import tyro

    tyro.cli(export_backward_encoder_from_checkpoint)
