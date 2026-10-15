# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import math
import typing as tp

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from torch import nn
from torch.distributions.utils import _standard_normal
from torch.utils._pytree import tree_flatten

from ufo.agents.common.base import BaseConfig
from ufo.agents.common.inputs import IdentityInputFilterConfig, NNFilter

##########################
# Initialization utils
##########################


# Initialization for parallel layers
def parallel_orthogonal_(tensor, gain=1):
    if tensor.ndimension() == 2:
        tensor = nn.init.orthogonal_(tensor, gain=gain)
        return tensor
    if tensor.ndimension() < 3:
        raise ValueError("Only tensors with 3 or more dimensions are supported")
    n_parallel = tensor.size(0)
    rows = tensor.size(1)
    cols = tensor.numel() // n_parallel // rows
    flattened = tensor.new(n_parallel, rows, cols).normal_(0, 1)

    qs = []
    for flat_tensor in torch.unbind(flattened, dim=0):
        if rows < cols:
            flat_tensor.t_()

        # Compute the qr factorization
        q, r = torch.linalg.qr(flat_tensor)
        # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph

        if rows < cols:
            q.t_()
        qs.append(q)

    qs = torch.stack(qs, dim=0)
    with torch.no_grad():
        tensor.view_as(qs).copy_(qs)
        tensor.mul_(gain)
    return tensor


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, DenseParallel):
        gain = nn.init.calculate_gain("relu")
        parallel_orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif hasattr(m, "reset_parameters"):
        m.reset_parameters()


##########################
# Update utils
##########################


def _soft_update_params(net_params: tp.Any, target_net_params: tp.Any, tau: float):
    torch._foreach_mul_(target_net_params, 1 - tau)
    torch._foreach_add_(target_net_params, net_params, alpha=tau)


class eval_mode:
    def __init__(self, *models) -> None:
        self.models = models
        self.prev_states = []

    def __enter__(self) -> None:
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args) -> None:
        for model, state in zip(self.models, self.prev_states):
            model.train(state)


##########################
# Creation utils
##########################


class ForwardArchiConfig(BaseConfig):
    name: tp.Literal["ForwardArchi"] = "ForwardArchi"
    hidden_dim: int = 1024
    hidden_layers: int = 2
    num_parallel: int = 2
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space, z_dim: int, action_dim, output_dim=None) -> torch.nn.Module:
        """Batched forward ensemble, also used for the critics."""
        return ResidualForwardMap(obs_space, z_dim, action_dim, self, output_dim=output_dim)


class ActorArchiConfig(BaseConfig):
    name: tp.Literal["actor"] = "actor"
    hidden_dim: int = 1024
    hidden_layers: int = 1
    embedding_layers: int = 2
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space, z_dim, action_dim):
        return ResidualActor(obs_space, z_dim, action_dim, self)


class DiscriminatorArchiConfig(BaseConfig):
    name: tp.Literal["DiscriminatorArchi"] = "DiscriminatorArchi"
    hidden_dim: int = 1024
    hidden_layers: int = 2
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space, z_dim) -> "Discriminator":
        return Discriminator(obs_space, z_dim, self)


def linear(input_dim, output_dim, num_parallel=1):
    if num_parallel > 1:
        return DenseParallel(input_dim, output_dim, n_parallel=num_parallel)
    return nn.Linear(input_dim, output_dim)


def layernorm(input_dim, num_parallel=1):
    if num_parallel > 1:
        return ParallelLayerNorm([input_dim], n_parallel=num_parallel)
    return nn.LayerNorm(input_dim)


##########################
# Simple MLP models
##########################


class BackwardArchiConfig(BaseConfig):
    name: tp.Literal["BackwardArchi"] = "BackwardArchi"
    hidden_dim: int = 256
    hidden_layers: int = 2
    norm: bool = True
    input_filter: NNFilter = IdentityInputFilterConfig()

    def build(self, obs_space, z_dim: int):
        return BackwardMap(obs_space, z_dim, self)


class BackwardMap(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: BackwardArchiConfig) -> None:
        super().__init__()
        self.cfg: BackwardArchiConfig = cfg

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        seq = [nn.Linear(filtered_space.shape[0], cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [nn.Linear(cfg.hidden_dim, z_dim)]
        if cfg.norm:
            seq += [Norm()]
        self.net = nn.Sequential(*seq)

    def forward(self, x: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.input_filter(x)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, obs_space, z_dim, cfg: DiscriminatorArchiConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        seq = [nn.Linear(obs_dim + z_dim, cfg.hidden_dim), nn.LayerNorm(cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.hidden_layers - 1):
            seq += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.ReLU()]
        seq += [nn.Linear(cfg.hidden_dim, 1)]
        self.trunk = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        s = self.compute_logits(obs, z)
        return torch.sigmoid(s)

    def compute_logits(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        obs = self.input_filter(obs)
        x = torch.cat([z, obs], dim=1)
        logits = self.trunk(x)
        return logits

    def compute_reward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        s = self.forward(obs, z)
        s = torch.clamp(s, eps, 1 - eps)
        reward = s.log() - (1 - s).log()
        return reward


##########################
# Residual models
##########################


class ResidualBlock(nn.Module):
    def __init__(self, dim, num_parallel: int = 1):
        super().__init__()
        ln = layernorm(dim, num_parallel)
        lin = linear(dim, dim, num_parallel)
        self.mlp = nn.Sequential(ln, lin, nn.Mish())

    def forward(self, x):
        return x + self.mlp(x)


class Block(nn.Module):
    def __init__(self, input_dim, output_dim, activation, num_parallel: int = 1):
        super().__init__()
        ln = layernorm(input_dim, num_parallel)
        lin = linear(input_dim, output_dim, num_parallel)
        seq = [ln, lin] + ([nn.Mish()] if activation else [])
        self.mlp = nn.Sequential(*seq)

    def forward(self, x):
        return self.mlp(x)


def residual_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
    assert hidden_layers >= 2, "must have at least 2 embedding layers"
    seq = [Block(input_dim, hidden_dim, True, num_parallel)]
    for _ in range(hidden_layers - 2):
        seq += [ResidualBlock(hidden_dim, num_parallel)]
    seq += [Block(hidden_dim, hidden_dim // 2, True, num_parallel)]
    return nn.Sequential(*seq)


class ResidualForwardMap(nn.Module):
    def __init__(
        self,
        obs_space,
        z_dim,
        action_dim,
        cfg: ForwardArchiConfig,
        output_dim=None,
    ) -> None:
        super().__init__()

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        self.z_dim = z_dim
        self.num_parallel = cfg.num_parallel
        self.hidden_dim = cfg.hidden_dim

        self.embed_z = residual_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.num_parallel)
        self.embed_sa = residual_embedding(obs_dim + action_dim, cfg.hidden_dim, cfg.hidden_layers, cfg.num_parallel)

        seq = [ResidualBlock(cfg.hidden_dim, cfg.num_parallel) for _ in range(cfg.hidden_layers)]
        seq += [Block(cfg.hidden_dim, output_dim if output_dim else z_dim, False, cfg.num_parallel)]
        self.Fs = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs = self.input_filter(obs)
        if self.num_parallel > 1:
            obs = obs.expand(self.num_parallel, -1, -1)
            z = z.expand(self.num_parallel, -1, -1)
            action = action.expand(self.num_parallel, -1, -1)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # num_parallel x bs x h_dim // 2
        sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1))  # num_parallel x bs x h_dim // 2
        return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))


class ResidualActor(nn.Module):
    def __init__(self, obs_space, z_dim, action_dim, cfg: ActorArchiConfig) -> None:
        super().__init__()

        self.input_filter = cfg.input_filter.build(obs_space)
        filtered_space = self.input_filter.output_space

        assert isinstance(filtered_space, gymnasium.spaces.Box), (
            f"filtered_space must be a Box space, got {type(filtered_space)}. Did you forget to set input_filter?"
        )
        assert len(filtered_space.shape) == 1, "filtered_space must have a 1D shape"
        obs_dim = filtered_space.shape[0]
        self.cfg: ActorArchiConfig = cfg
        self.embed_z = residual_embedding(obs_dim + z_dim, cfg.hidden_dim, cfg.embedding_layers)
        self.embed_s = residual_embedding(obs_dim, cfg.hidden_dim, cfg.embedding_layers)

        seq = [ResidualBlock(cfg.hidden_dim) for _ in range(cfg.hidden_layers)] + [Block(cfg.hidden_dim, action_dim, False)]
        self.policy = nn.Sequential(*seq)

    def forward(self, obs: torch.Tensor | dict[str, torch.Tensor], z, std):
        obs = self.input_filter(obs)
        z_embedding = self.embed_z(torch.cat([obs, z], dim=-1))  # bs x h_dim // 2
        s_embedding = self.embed_s(obs)  # bs x h_dim // 2
        embedding = torch.cat([s_embedding, z_embedding], dim=-1)
        mu = torch.tanh(self.policy(embedding))
        std = torch.ones_like(mu) * std
        dist = TruncatedNormal(mu, std)
        return dist


##########################
# Helper modules
##########################


class DenseParallel(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_parallel: int, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert n_parallel > 1, "Use nn.Linear for a single network"
        self.in_features = in_features
        self.out_features = out_features
        self.n_parallel = n_parallel
        self.weight = nn.Parameter(torch.empty((n_parallel, in_features, out_features), **factory_kwargs))
        self.bias = nn.Parameter(torch.empty((n_parallel, 1, out_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return torch.baddbmm(self.bias, input, self.weight)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, n_parallel={self.n_parallel}, bias=True"


class ParallelLayerNorm(nn.Module):
    def __init__(self, normalized_shape, n_parallel, eps=1e-5, device=None, dtype=None) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        assert n_parallel > 1, "Use nn.LayerNorm for a single network"
        assert len(normalized_shape) == 1
        self.n_parallel = n_parallel
        self.normalized_shape = list(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        self.bias = nn.Parameter(torch.empty([n_parallel, 1, *self.normalized_shape], **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input):
        norm_input = F.layer_norm(input, self.normalized_shape, None, None, self.eps)
        return (norm_input * self.weight) + self.bias

    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine=True"


class TruncatedNormal(pyd.Normal):
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6) -> None:
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x) -> torch.Tensor:
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()) -> torch.Tensor:  # type: ignore
        shape = self._extended_shape(sample_shape)
        eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        eps *= self.scale
        if clip is not None:
            eps = torch.clamp(eps, -clip, clip)
        x = self.loc + eps
        return self._clamp(x)


class Norm(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return math.sqrt(x.shape[-1]) * F.normalize(x, dim=-1)


class EMA(nn.Module):
    """exponential moving average"""

    def __init__(self, tau=0.99, epsilon=1e-8, shape=(1,), translate=False, scale=False) -> None:
        super().__init__()
        self.tau = tau
        self.epsilon = epsilon
        self.register_buffer("mean", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("mean_square", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("counter", torch.LongTensor([0]))
        self.translate = translate
        self.scale = scale

    def forward(self, x):
        m = x.mean()
        sm = x.pow(2).mean()
        self.mean.data = self.tau * self.mean + (1 - self.tau) * m
        self.mean_square.data = self.tau * self.mean_square + (1 - self.tau) * sm
        self.counter += 1  # type: ignore
        norm = 1 - self.tau**self.counter
        ema_mean = self.mean / norm
        ema_mean_square = self.mean_square / norm
        var = torch.clamp(ema_mean_square - ema_mean**2, min=self.epsilon)

        translate_mean = ema_mean if self.translate else 0
        scale_std = torch.sqrt(var) if self.scale else 1
        return (x - translate_mean) / scale_std

class RewardNormalizerConfig(BaseConfig):
    name: tp.Literal["RewardNormalizer"] = "RewardNormalizer"
    translate: bool = False
    scale: bool = False

    def build(self) -> nn.Module:
        return EMA(translate=self.translate, scale=self.scale)


def tree_get_batch_size(pytree):
    tensors, _ = tree_flatten(pytree)
    batch_sizes = [t.shape[0] for t in tensors]
    assert all(bs == batch_sizes[0] for bs in batch_sizes), f"All tensors must have the same batch size {batch_sizes[0]}, got {batch_sizes}"
    return batch_sizes[0]
