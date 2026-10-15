# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


from __future__ import annotations

import os
import random
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank() -> int:
    return dist.get_rank() if is_distributed() else int(os.environ.get("RANK", "0"))


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else int(os.environ.get("WORLD_SIZE", "1"))


def barrier() -> None:
    if is_distributed():
        if torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def all_ranks_true(value: bool, device: str | torch.device) -> bool:
    """Agree on readiness before entering gradient collectives."""
    if not is_distributed():
        return value
    flag = torch.tensor(int(value), device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


@torch.compiler.disable
def average_gradients(
    parameters: Iterable[torch.nn.Parameter], *, bucket_size_bytes: int = 128 * 1024 * 1024, bucket_cache: dict | None = None
) -> None:
    """Launch bucket reductions together, then average before the optimizer step."""
    if not is_distributed():
        return
    if bucket_size_bytes <= 0:
        raise ValueError("Gradient bucket size must be positive")
    parameters = tuple(p for p in parameters if p.grad is not None)
    key = (bucket_size_bytes, tuple((p, p.grad.device, p.grad.dtype, p.grad.layout) for p in parameters))
    buckets = bucket_cache.get(key) if bucket_cache is not None else None
    if buckets is None:
        buckets, bucket, size = [], [], 0
        for param in parameters:
            grad = param.grad
            nbytes = grad.numel() * grad.element_size()
            if bucket and (
                size + nbytes > bucket_size_bytes
                or grad.device != bucket[0].grad.device
                or grad.dtype != bucket[0].grad.dtype
                or grad.is_sparse
                or bucket[0].grad.is_sparse
            ):
                buckets.append(tuple(bucket))
                bucket, size = [], 0
            bucket.append(param)
            size += nbytes
        if bucket:
            buckets.append(tuple(bucket))
        if bucket_cache is not None:
            bucket_cache[key] = buckets

    pending = []
    for bucket in buckets:
        grads = [p.grad for p in bucket]
        direct = len(grads) == 1 and (grads[0].is_sparse or grads[0].is_contiguous())
        packed = grads[0] if direct else torch.cat([g.reshape(-1) for g in grads])
        work = dist.all_reduce(packed, op=dist.ReduceOp.SUM, async_op=True)
        pending.append((work, packed, grads, direct))
    scale = 1.0 / dist.get_world_size()
    for work, packed, grads, direct in pending:
        work.wait()
        packed.mul_(scale)
        if not direct:
            views = [part.view_as(g) for part, g in zip(packed.split([g.numel() for g in grads]), grads)]
            torch._foreach_copy_(grads, views)


@torch.compiler.disable
def broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=src)


@torch.compiler.disable
def sync_model_buffers(module: torch.nn.Module, src: int = 0) -> None:
    if not is_distributed():
        return
    groups = {}
    for buffer in module.buffers():
        groups.setdefault((buffer.device, buffer.dtype), []).append(buffer.data)
    for buffers in groups.values():
        packed = torch.cat([buffer.reshape(-1) for buffer in buffers])
        if packed.dtype.is_floating_point or packed.dtype.is_complex:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            packed.mul_(1.0 / dist.get_world_size())
        else:
            dist.broadcast(packed, src=src)
        parts = packed.split([buffer.numel() for buffer in buffers])
        torch._foreach_copy_(buffers, [part.view_as(buffer) for part, buffer in zip(parts, buffers)])


@torch.compiler.disable
def broadcast_optimizer_state(optimizer: torch.optim.Optimizer, src: int = 0) -> None:
    if not is_distributed():
        return
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                dist.broadcast(value.data, src=src)


def broadcast_agent_state(agent: Any, src: int = 0) -> None:
    if not is_distributed():
        return
    broadcast_module_state(agent._model, src=src)
    for optimizer in agent.optimizers.values():
        broadcast_optimizer_state(optimizer, src=src)


@torch.compiler.disable
def broadcast_object(value: Any, src: int = 0) -> Any:
    if not is_distributed():
        return value
    objects = [value if dist.get_rank() == src else None]
    dist.broadcast_object_list(objects, src=src)
    return objects[0]


@torch.compiler.disable
def module_sync_report(module: torch.nn.Module, src: int = 0) -> dict[str, Any]:
    if not is_distributed():
        return {"world_size": 1, "rank": 0, "max_abs_diff_from_rank0": 0.0}

    tensors = list(module.parameters()) + [
        buffer for buffer in module.buffers() if buffer.dtype.is_floating_point or buffer.dtype.is_complex
    ]
    if tensors:
        device = tensors[0].device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_summary = torch.zeros(3, dtype=torch.float64, device=device)
    max_abs_diff = torch.zeros((), dtype=torch.float32, device=device)

    for tensor in tensors:
        detached = tensor.detach()
        local_summary[0] += detached.double().sum()
        local_summary[1] += detached.double().square().sum()
        local_summary[2] += detached.numel()
        reference = detached.clone()
        dist.broadcast(reference, src=src)
        if detached.numel() > 0:
            diff = (detached - reference).abs().max().float()
            max_abs_diff = torch.maximum(max_abs_diff, diff)

    dist.all_reduce(max_abs_diff, op=dist.ReduceOp.MAX)
    gathered = [torch.empty_like(local_summary) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_summary)
    summaries = torch.stack(gathered).cpu()
    return {
        "world_size": dist.get_world_size(),
        "rank": dist.get_rank(),
        "max_abs_diff_from_rank0": float(max_abs_diff.cpu().item()),
        "rank_param_buffer_sum": [float(x) for x in summaries[:, 0].tolist()],
        "rank_param_buffer_sqsum": [float(x) for x in summaries[:, 1].tolist()],
        "rank_param_buffer_numel": [int(x) for x in summaries[:, 2].tolist()],
    }


@torch.compiler.disable
def average_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    if not is_distributed():
        return dict(metrics)
    reduced: dict[str, Any] = {}
    scale = 1.0 / float(dist.get_world_size())
    groups: dict[tuple, list[tuple[str, torch.Tensor]]] = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            tensor = value.detach().clone()
            if not (tensor.dtype.is_floating_point or tensor.dtype.is_complex):
                tensor = tensor.float()
            groups.setdefault((tensor.device, tensor.dtype), []).append((key, tensor))
        else:
            reduced[key] = value
    for values in groups.values():
        packed = torch.cat([tensor.reshape(-1) for _, tensor in values])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        packed.mul_(scale)
        offset = 0
        for key, tensor in values:
            reduced[key] = packed[offset : offset + tensor.numel()].view_as(tensor)
            offset += tensor.numel()
    return reduced


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
