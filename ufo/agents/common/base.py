from typing import Any, Literal

import pydantic
import torch
from torch import nn

from ufo.agents.common.checkpoint import load_model, save_model


class BaseConfig(pydantic.BaseModel):
    """Base class for model configurations."""

    # extra="forbid" -- prevent adding new fields (e.g., accidentally typo-ing argument model-dimm=256 )
    # strict=True -- be strict on types, do not try to cast e.g., "True" --> "1" (or vice versa)
    # use_enum_values=True -- fixes issues with serializing/deserializing enums
    # frozen=True -- make the model immutable
    model_config = pydantic.ConfigDict(extra="forbid", strict=True, use_enum_values=True, frozen=True)

    name: Literal["BaseConfig"] = "BaseConfig"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # We want to make sure `class_name` is always a valid name.
        # So we will add it to the list of Literals (if user set custom name), or create one.
        # We will default using user's name if one provided, otherwise the class name.
        if not hasattr(cls, "name") or cls.name == "BaseConfig":
            # Add field consistent with `name: Literal["BaseConfig"] = "BaseConfig"`
            cls.name = cls.__name__
            cls.__annotations__["name"] = Literal[cls.__name__]
        else:
            # Append the name to the list of Literals
            cls.__annotations__["name"] = Literal[(cls.__name__,) + cls.__annotations__["name"].__args__]

    def build(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"The object {self} did not have valid build function. Did you forget to define it?")


class BaseModelConfig(BaseConfig):
    device: Literal["cpu", "cuda"] = "cuda"


class BaseModel(nn.Module):
    config_class = BaseModelConfig

    def __init__(self, obs_space, action_dim, config: BaseModelConfig):
        super().__init__()
        self.obs_space = obs_space
        self.action_dim = action_dim
        self.cfg = config

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.device = device.type  # type: ignore
        return super().to(*args, **kwargs)

    @classmethod
    def load(cls, path: str, device: str | None = None, load_target_networks: bool = True):
        return load_model(path, device, load_target_networks=load_target_networks, config_class=cls.config_class)

    def save(self, output_folder: str) -> None:
        return save_model(output_folder, self, build_kwargs={"obs_space": self.obs_space, "action_dim": self.action_dim})
