# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


"""Training CSV output, finite-value checks and contact metrics."""

import csv
import dataclasses
import tempfile
import typing as tp
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch


@dataclasses.dataclass
class CSVLogger:
    filename: Union[str, Path]
    fields: Optional[List[str]] = None

    def log(self, log_data: Dict[str, Any]) -> None:
        islist = [isinstance(v, Iterable) and not isinstance(v, str) for v in log_data.values()]
        if all(islist):
            df = pd.DataFrame(log_data)
        elif not any(islist):
            df = pd.DataFrame([log_data])
        else:
            raise RuntimeError("Fields should all be a numbers, a string or iterable objects. We don't support mixed types.")
        path = Path(self.filename)
        has_header = path.exists() and path.stat().st_size > 0
        if self.fields is None and has_header:
            with path.open(newline="") as stream:
                self.fields = next(csv.reader(stream))
        fields = sorted(set(self.fields or []).union(log_data))
        if has_header and fields != self.fields:
            # A new training phase may introduce metrics; preserve old rows verbatim.
            temporary = None
            try:
                with path.open(newline="") as source, tempfile.NamedTemporaryFile(
                    mode="w", newline="", dir=path.parent, prefix=f".{path.name}.", delete=False,
                ) as target:
                    temporary = Path(target.name)
                    writer = csv.DictWriter(target, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(csv.DictReader(source))
                temporary.chmod(path.stat().st_mode & 0o777)
                temporary.replace(path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        self.fields = fields
        df.reindex(columns=fields, fill_value="").to_csv(path, mode="a", header=not has_header, index=False)


def _tensor_nonfinite_summary(value: tp.Any) -> str | None:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or not torch.is_floating_point(value):
            return None
        finite = torch.isfinite(value)
        if bool(finite.all().item()):
            return None
        bad = ~finite
        bad_count = int(bad.sum().item())
        with torch.no_grad():
            finite_values = value[finite]
            finite_min = float(finite_values.min().item()) if finite_values.numel() > 0 else None
            finite_max = float(finite_values.max().item()) if finite_values.numel() > 0 else None
            bad_indices = bad.nonzero(as_tuple=False)[:5].detach().cpu().tolist()
            bad_values = value[bad][:5].detach().cpu().tolist()
        return (
            f"type=torch shape={tuple(value.shape)} dtype={value.dtype} bad_count={bad_count} "
            f"finite_min={finite_min} finite_max={finite_max} bad_indices={bad_indices} bad_values={bad_values}"
        )
    if isinstance(value, np.ndarray):
        if value.size == 0 or not np.issubdtype(value.dtype, np.floating):
            return None
        finite = np.isfinite(value)
        if bool(finite.all()):
            return None
        bad = ~finite
        bad_count = int(bad.sum())
        finite_values = value[finite]
        finite_min = float(finite_values.min()) if finite_values.size > 0 else None
        finite_max = float(finite_values.max()) if finite_values.size > 0 else None
        bad_indices = np.argwhere(bad)[:5].tolist()
        bad_values = value[bad][:5].tolist()
        return (
            f"type=numpy shape={value.shape} dtype={value.dtype} bad_count={bad_count} "
            f"finite_min={finite_min} finite_max={finite_max} bad_indices={bad_indices} bad_values={bad_values}"
        )
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value):
            return None
        return f"type=scalar value={value}"
    return None


def _iter_nonfinite(value: tp.Any, prefix: str = "") -> tp.Iterator[tuple[str, str]]:
    summary = _tensor_nonfinite_summary(value)
    if summary is not None:
        yield prefix or "<root>", summary
        return
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_nonfinite(item, child_prefix)
    elif isinstance(value, (list, tuple)):
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            yield from _iter_nonfinite(item, child_prefix)


def _assert_finite(value: tp.Any, *, label: str, rank: int, local_time: int, global_time: int, optimizer_steps: int) -> None:
    # The common path needs one fused check per device/dtype, not one per
    # logged metric. Detailed field inspection is only needed after a failure.
    tensors = {}
    others = []

    def collect(item):
        if isinstance(item, torch.Tensor):
            if item.numel() and torch.is_floating_point(item):
                tensors.setdefault((item.device, item.dtype), []).append(item.detach().reshape(-1))
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)
        else:
            others.append(item)

    collect(value)
    finite = all(bool(torch.isfinite(group[0] if len(group) == 1 else torch.cat(group)).all().item()) for group in tensors.values())
    if finite and all(_tensor_nonfinite_summary(item) is None for item in others):
        return
    bad = list(_iter_nonfinite(value, label))
    if not bad:
        return
    details = "\n".join(f"  - {path}: {summary}" for path, summary in bad[:20])
    more = "" if len(bad) <= 20 else f"\n  ... {len(bad) - 20} more non-finite fields"
    raise FloatingPointError(
        "Non-finite value detected "
        f"(rank={rank}, local_time={local_time}, global_time={global_time}, optimizer_steps={optimizer_steps}):\n"
        f"{details}{more}"
    )


def _assert_model_finite(module: torch.nn.Module, *, rank: int, local_time: int, global_time: int, optimizer_steps: int) -> None:
    for name, param in module.named_parameters():
        _assert_finite(
            param,
            label=f"model.parameters.{name}",
            rank=rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
        )
    for name, buffer in module.named_buffers():
        _assert_finite(
            buffer,
            label=f"model.buffers.{name}",
            rank=rank,
            local_time=local_time,
            global_time=global_time,
            optimizer_steps=optimizer_steps,
        )


def torso_contact_force_metrics(train_env, *, enabled: bool, threshold: float) -> dict[str, float]:
    if not enabled:
        return {}

    raw_env = getattr(train_env, "_env", train_env)
    torso_index = getattr(raw_env, "torso_index", None)
    if not hasattr(raw_env, "contact_forces") or torso_index is None:
        return {}

    with torch.no_grad():
        torso_force = raw_env.contact_forces[:, torso_index, :].float()
        torso_norm = torch.linalg.vector_norm(torso_force, dim=-1)
        torso_mean = torso_force.mean(dim=0)
        torso_force_z = torso_force[:, 2]
        torso_z = raw_env.body_pos[:, torso_index, 2].float()
        contact_mask = torso_norm > threshold
        contact_count = contact_mask.sum()
        contact_count_value = int(contact_count.item())
        num_envs = max(torso_norm.numel(), 1)

        metrics = {
            "torso_contact_force_mean": np.round(torso_norm.mean().item(), 6),
            "torso_contact_force_max": np.round(torso_norm.max().item(), 6),
            "torso_contact_force_p95": np.round(torch.quantile(torso_norm, 0.95).item(), 6),
            "torso_contact_force_contact_count": contact_count_value,
            "torso_contact_force_contact_frac": np.round((contact_count.float() / num_envs).item(), 6),
            "torso_contact_force_mean_x": np.round(torso_mean[0].item(), 6),
            "torso_contact_force_mean_y": np.round(torso_mean[1].item(), 6),
            "torso_contact_force_mean_z": np.round(torso_mean[2].item(), 6),
            "torso_contact_force_z_mean": np.round(torso_force_z.mean().item(), 6),
            "torso_contact_force_z_max": np.round(torso_force_z.max().item(), 6),
            "torso_contact_force_z_p95": np.round(torch.quantile(torso_force_z, 0.95).item(), 6),
            "torso_z_mean": np.round(torso_z.mean().item(), 6),
            "torso_z_min": np.round(torso_z.min().item(), 6),
            "torso_z_p05": np.round(torch.quantile(torso_z, 0.05).item(), 6),
        }
        if contact_count_value > 0:
            torso_z_contact = torso_z[contact_mask]
            metrics.update(
                {
                    "torso_z_contact_mean": np.round(torso_z_contact.mean().item(), 6),
                    "torso_z_contact_min": np.round(torso_z_contact.min().item(), 6),
                    "torso_z_contact_p05": np.round(torch.quantile(torso_z_contact, 0.05).item(), 6),
                }
            )
        else:
            metrics.update(
                {
                    "torso_z_contact_mean": 0.0,
                    "torso_z_contact_min": 0.0,
                    "torso_z_contact_p05": 0.0,
                }
            )
        return metrics
