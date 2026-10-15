"""Checkpoint I/O shared by the explicit FB and TeCH training paths."""

import json
import typing as tp
from pathlib import Path

import gymnasium.spaces as gym_spaces
import numpy as np
import safetensors.torch
import torch

if tp.TYPE_CHECKING:
    from ufo.agents.common.base import BaseModel, BaseModelConfig

SUPPORTED_SPACES = tp.Union[gym_spaces.Box, gym_spaces.Dict]


def space_to_json(space: SUPPORTED_SPACES):
    """Convert a Gymnasium space to a JSON-serializable format."""
    if isinstance(space, gym_spaces.Box):
        state = {
            "low": space.low.tolist(),
            "high": space.high.tolist(),
            "shape": space.shape,
            "dtype": str(space.dtype),
            "type": "Box",
        }
        return state
    elif isinstance(space, gym_spaces.Dict):
        state = dict(type="Dict", spaces=dict())
        for name, sub_space in space.spaces.items():
            space_json = space_to_json(sub_space)
            state["spaces"][name] = space_json
        return state
    else:
        raise NotImplementedError(f"Space type {type(space)} is not supported for JSON serialization.")


def json_to_space(json_data: tp.Dict[str, tp.Any]) -> SUPPORTED_SPACES:
    """Convert JSON data back to a Gymnasium space."""
    if json_data["type"] == "Box":
        dtype = np.dtype(json_data["dtype"])
        return gym_spaces.Box(
            low=np.array(json_data["low"], dtype=dtype),
            high=np.array(json_data["high"], dtype=dtype),
            shape=json_data["shape"],
            dtype=dtype,
        )
    if json_data["type"] == "Dict":
        spaces = {}
        for name, space_json in json_data["spaces"].items():
            spaces[name] = json_to_space(space_json)
        dict_space = gym_spaces.Dict(spaces=spaces)
        return dict_space
    else:
        raise NotImplementedError(f"Space type {json_data['type']} is not supported for JSON deserialization.")


def save_build_kwargs(path, build_kwargs):
    values = dict(build_kwargs)
    if "obs_space" in values:
        values["obs_space"] = space_to_json(values["obs_space"])
    with (Path(path) / "init_kwargs.json").open("w") as stream:
        json.dump(values, stream, indent=4)


def load_build_kwargs(path):
    with (Path(path) / "init_kwargs.json").open() as stream:
        values = json.load(stream)
    if "obs_space" in values:
        values["obs_space"] = json_to_space(values["obs_space"])
    return values


def save_model(path: str, model: "BaseModel", build_kwargs: tp.Optional[tp.Dict[str, tp.Any]] = None) -> None:
    output_folder = Path(path)
    output_folder.mkdir(exist_ok=True, parents=True)
    safetensors.torch.save_model(model, output_folder / "model.safetensors")

    json_dump = model.cfg.model_dump()

    if build_kwargs is not None:
        save_build_kwargs(output_folder, build_kwargs)

    with (output_folder / "config.json").open("w+") as f:
        f.write(json.dumps(json_dump, indent=4))


def load_model(
    path: str, device: str | None, load_target_networks: bool, config_class: "BaseModelConfig"
) -> "BaseModel":
    model_dir = Path(path)
    with (model_dir / "config.json").open() as f:
        loaded_config = json.load(f)
    if device is not None:
        loaded_config["device"] = device

    build_kwargs = load_build_kwargs(model_dir)

    loaded_config = config_class(**loaded_config)
    loaded_model = loaded_config.build(**build_kwargs)

    # Read only requested tensors; inference must not allocate target weights on CUDA.
    with safetensors.safe_open(model_dir / "model.safetensors", framework="pt", device=device or "cpu") as saved:
        state_dict = {key: saved.get_tensor(key) for key in saved.keys() if load_target_networks or not key.startswith("_target_")}
    if load_target_networks and any(key.startswith("_target_") for key in state_dict):
        loaded_model._prepare_for_train()
    loaded_model.load_state_dict(state_dict, strict=True)
    return loaded_model


def checkpoint_load_device(device: str) -> str:
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
        return "cuda"
    if torch_device.type == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported inference device: {device}")


def load_model_from_checkpoint_dir(checkpoint_dir: str, device: str = "cpu") -> tp.Any:
    checkpoint_dir = Path(checkpoint_dir)
    if (checkpoint_dir / ".saving").exists():
        raise RuntimeError(f"Incomplete checkpoint save: {checkpoint_dir}")
    with (checkpoint_dir / "model" / "config.json").open("r") as f:
        config = json.load(f)

    # Model classes depend on checkpoint I/O through BaseModel.
    from ufo.agents.fb_cpr_aux.model import FBcprAuxModel
    from ufo.agents.tech.model import TeCHModel

    model_name_to_class = {"FBcprAuxModel": FBcprAuxModel, "TeCHModel": TeCHModel}
    model_name = config["name"]
    if model_name not in model_name_to_class:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(model_name_to_class.keys())}")
    model = model_name_to_class[model_name].load(checkpoint_dir / "model", device=device, load_target_networks=False)
    if (checkpoint_dir / ".saving").exists():
        raise RuntimeError(f"Checkpoint save started while loading: {checkpoint_dir}")
    return model


def save_agent(agent, path):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    with (path / "config.json").open("w") as stream:
        json.dump(agent.cfg.model_dump(), stream, indent=4)
    torch.save(agent.optimizer_dict, path / "optimizers.pth")
    agent._model.save(output_folder=str(path / "model"))
    save_build_kwargs(path, {"obs_space": agent.obs_space, "action_dim": agent.action_dim})


def load_agent(config_class, path, device=None):
    path = Path(path)
    with (path / "config.json").open() as stream:
        config = json.load(stream)
    if device is not None:
        config["model"]["device"] = device
    args = load_build_kwargs(path)
    agent = config_class(**config).build(args["obs_space"], args["action_dim"])
    optimizers = agent.optimizers
    states = torch.load(path / "optimizers.pth", weights_only=True, map_location=device)
    if states.keys() != optimizers.keys():
        raise ValueError("Checkpoint optimizer stages do not match the agent configuration")
    for name, state in states.items():
        optimizers[name].load_state_dict(state)
    safetensors.torch.load_model(agent._model, path / "model/model.safetensors", device=device, strict=True)
    agent._model.train()
    agent._model.requires_grad_(True)
    return agent
