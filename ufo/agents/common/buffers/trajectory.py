import dataclasses
import typing
from collections.abc import Mapping
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils._pytree import tree_map

from ufo.agents.common.buffers.storage import DictBuffer, _to_torch

INT_CLASSES_TYPING = Union[int, np.integer]
INT_CLASSES = typing.get_args(INT_CLASSES_TYPING)


def _is_int(index):
    if isinstance(index, INT_CLASSES):
        return True
    if isinstance(index, (np.ndarray, torch.Tensor)):
        return index.ndim == 0
    return False


def tree_concat(list_of_pytree_of_tensors, dim=0):
    return tree_map(lambda *leaves: torch.cat(leaves, dim=dim), *list_of_pytree_of_tensors)


def _gather_transitions(buffer, indices):
    indices = indices.to(torch.long).unbind(-1)
    output = {"next": {}} if buffer.output_key_tp1 else {}
    output.update({key: tree_map(lambda value: value[indices], buffer.storage[key]) for key in buffer.output_key_t})
    if buffer.output_key_tp1:
        if "next" in buffer.storage:
            source = buffer.storage["next"]
        else:
            source = buffer.storage
            indices = ((indices[0] + 1) % buffer.capacity, *indices[1:])
        output["next"] = {key: tree_map(lambda value: value[indices], source[key]) for key in buffer.output_key_tp1}
    return output


class ExpertTrajectoryBuffer:
    def __init__(
        self,
        episodes: List[dict],
        device: str = "cpu",
        seq_length: int = 1,
        output_key_t: List[str] = ["observation"],
        output_key_tp1: List[str] = ["observation"],
        end_key: Tuple[str] | str = "done",
        motion_id_key: Tuple[str] | str = "motion_id",
    ) -> None:
        self._is_full = True
        self._sampling_trajectories = {}
        self.output_key_t = output_key_t
        self.output_key_tp1 = output_key_tp1
        self.seq_length = seq_length
        self.device = device
        self.end_key = end_key
        self.motion_id_key = motion_id_key

        def add_new_data(data, storage):
            for k, v in data.items():
                if isinstance(v, Mapping):
                    # If the value is a dictionary, recursively call the function
                    if k not in storage:
                        storage[k] = {}
                    add_new_data(v, storage=storage[k])
                else:
                    if k not in storage:
                        storage[k] = []
                    storage[k].append(_to_torch(v, device=self.device))

        def concat_dict(storage):
            for k, v in storage.items():
                if isinstance(v, Mapping):
                    concat_dict(storage[k])
                else:
                    storage[k] = torch.cat(v, axis=0)

        self.storage = {}
        set_key(self.storage, self.end_key, [])
        self.motion_ids = []
        for ep in episodes:
            _k = [k for k, v in ep.items() if not isinstance(v, Mapping)][0]
            # first append all the values to a list
            add_new_data(ep, self.storage)
            # add termination of each episode
            if not key_exists(ep, self.end_key):
                episode_start = torch.zeros((ep[_k].shape[0], 1), dtype=torch.bool, device=self.device)
                episode_start[-1] = True
                get_key(self.storage, self.end_key).append(episode_start)
            self.motion_ids.append(get_key(ep, self.motion_id_key)[0].item())
        # then concatenate them into a single tensor
        concat_dict(self.storage)
        assert get_key(self.storage, self.end_key).dtype == torch.bool
        done = get_key(self.storage, self.end_key)
        # cursor is none since we don't wrap data
        self.start_idx, _, self.lengths = find_start_stop_traj(
            done.squeeze()[: len(self)], at_capacity=self._is_full, cursor=None
        )
        # set priorities to match the number of trajectories
        self.priorities = torch.ones(len(self.lengths), device=self.device, dtype=torch.float32) / len(self.lengths)
        self._get_idxs = torch.compile(get_idxs, mode="reduce-overhead")

    def sample(self, batch_size: int = 1, seq_length: int | None = None):
        seq_length = seq_length or self.seq_length
        if batch_size < seq_length:
            raise ValueError(
                f"The batch-size must be bigger than the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )

        if batch_size % seq_length != 0:
            raise ValueError(
                f"The batch-size must be divisible by the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )
        offset = bool(self.output_key_tp1)
        num_slices = batch_size // seq_length
        lengths, starts, traj_idx = _sampling_trajectories(self, seq_length + offset)
        idxs = self._get_idxs(
            seq_length=seq_length,
            num_slices=num_slices,
            lengths=lengths,
            start_idx=starts,
            storage_length=self.capacity,
            priorities=self.priorities[traj_idx],
            offset=int(offset),
        )
        return _gather_transitions(self, idxs)

    def update_priorities(self, priorities: torch.Tensor, idxs: torch.Tensor) -> None:
        """update priorities of trajectories"""
        assert len(priorities) == len(self.priorities)
        self.priorities[idxs] = priorities
        self.priorities = self.priorities / torch.sum(self.priorities)

    @property
    def capacity(self):
        return get_key(self.storage, self.end_key).shape[0]

    def __len__(self) -> int:
        return self.capacity

    def empty(self) -> bool:
        return len(self) == 0


@dataclasses.dataclass(kw_only=True)
class ReplayBuffer(DictBuffer):
    # DictBuffer that supports multiple dimensions + sampling consecutive subsequences. Can be used to store whole episodes
    n_dim: int = 1  # use n_dim 2 to support multi-dimensional data
    seq_length: int = 1
    output_key_t: List[str] = dataclasses.field(default_factory=lambda: ["observation"])
    output_key_tp1: List[str] = dataclasses.field(default_factory=lambda: ["observation"])
    end_key: Tuple[str] | str = "done"

    def __post_init__(self) -> None:
        self.storage = None
        self._idx = 0
        self._is_full = False
        self._recompute_start_stop = True
        self._sampling_trajectories = {}
        self._get_idxs = torch.compile(get_idxs, mode="reduce-overhead", fullgraph=True)
        assert self.n_dim == 1 or self.n_dim == 2, "n_dim must be either 1 or 2 for ReplayBuffer"

    def _ndim(self) -> int:
        return self.n_dim

    def size(self):
        return len(self) * self.storage[self.end_key].shape[1]

    @torch.no_grad
    def extend(self, data: Dict) -> None:
        """Extend the buffer with new data.
        We use a dictionary representation for the storage."""
        super().extend(data)
        self._recompute_start_stop = True

    def _update_trajectories(self) -> None:
        if self._recompute_start_stop:
            done = get_key(self.storage, self.end_key)
            self.start_idx, _, self.lengths = find_start_stop_traj(
                done.reshape(done.shape[: self.n_dim])[: len(self)], at_capacity=self._is_full, cursor=self._idx - 1
            )
            self._recompute_start_stop = False
            self._sampling_trajectories.clear()

    def can_sample(self, seq_length: int | None = None) -> bool:
        """Check transition eligibility without drawing samples or advancing RNG."""
        if self.empty():
            return False
        self._update_trajectories()
        seq_length = seq_length or self.seq_length
        offset = bool(self.output_key_tp1) and "next" not in self.storage
        lengths, _, _ = _sampling_trajectories(self, seq_length + offset)
        return lengths.numel() > 0

    def sample(self, batch_size: int = 1, seq_length: int | None = None):
        seq_length = seq_length or self.seq_length
        if batch_size < seq_length:
            raise ValueError(
                f"The batch-size must be bigger than the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )

        if batch_size % seq_length != 0:
            raise ValueError(
                f"The batch-size must be divisible by the sequence length, got batch_size={batch_size} and seq_length={seq_length}."
            )

        self._update_trajectories()

        offset = bool(self.output_key_tp1) and "next" not in self.storage
        num_slices = batch_size // seq_length
        lengths, starts, traj_idx = _sampling_trajectories(self, seq_length + offset)
        if lengths.numel() == 0:
            raise ValueError(f"No trajectories with length >= {seq_length + offset} to sample {num_slices} slices.")
        idxs = self._get_idxs(
            seq_length=seq_length,
            num_slices=num_slices,
            lengths=lengths,
            start_idx=starts,
            storage_length=self.capacity,
            priorities=None,
            offset=int(offset),
        )
        return _gather_transitions(self, idxs)

    def get_full_buffer(self) -> Dict:
        """We assume to return transition based"""

        self._update_trajectories()

        offset = bool(self.output_key_tp1) and "next" not in self.storage
        slices = []
        for start, length in zip(self.start_idx, self.lengths):
            indices = start.expand(int(length) - offset, -1).clone()
            indices[:, 0] = (start[0] + torch.arange(len(indices), device=self.device)) % self.capacity
            slices.append(_gather_transitions(self, indices))
        return tree_concat(slices)


def _sampling_trajectories(buffer, min_length):
    # Boundaries stay fixed between extends; priorities are read live in sample().
    if min_length not in buffer._sampling_trajectories:
        indices = (buffer.lengths >= min_length).nonzero(as_tuple=True)[0]
        buffer._sampling_trajectories[min_length] = (
            buffer.lengths[indices], buffer.start_idx[indices], indices
        )
    return buffer._sampling_trajectories[min_length]


def get_idxs(seq_length, num_slices, lengths, start_idx, storage_length, priorities: torch.Tensor | None = None, offset=1):
    if priorities is not None:
        traj_idx = torch.multinomial(priorities, num_slices, replacement=True)
    else:
        traj_idx = torch.randint(lengths.shape[0], (num_slices,), device=lengths.device)
    end_point = lengths[traj_idx] - seq_length - offset + 1
    relative_starts = (torch.rand(num_slices, device=lengths.device) * end_point).floor().to(start_idx.dtype)
    starts = torch.cat(
        [
            (start_idx[traj_idx, 0] + relative_starts).unsqueeze(1),
            start_idx[traj_idx, 1:],
        ],
        1,
    )
    idxs = _tensor_slices_from_startend(seq_length, starts, storage_length=storage_length)
    return idxs


def find_start_stop_traj(end: torch.Tensor, at_capacity: bool = True, cursor=None | int):
    length = end.shape[0]
    if not at_capacity:
        end = torch.index_fill(
            end,
            index=torch.tensor(length - 1, device=end.device, dtype=torch.long),
            dim=0,
            value=1,
        )
    else:
        if cursor is not None:
            if not _is_int(cursor):
                raise ValueError("cursor must be an int")
            end = torch.index_fill(
                end,
                index=torch.tensor(cursor, device=end.device, dtype=torch.long),
                dim=0,
                value=1,
            )
        if not end.any(0).all():
            mask = ~end.any(0, True)
            mask = torch.cat([torch.zeros_like(end[:-1]), mask])
            end = torch.masked_fill(mask, end, 1)
    return _end_to_start_stop(length=length, end=end)


def _end_to_start_stop(end, length):
    # Using transpose ensures the start and stop are sorted the same way
    stop_idx = end.transpose(0, -1).nonzero()
    stop_idx[:, [0, -1]] = stop_idx[:, [-1, 0]].clone()
    # First build the start indices as the stop + 1, we'll shift it later
    start_idx = stop_idx.clone()
    start_idx[:, 0] += 1
    start_idx[:, 0] %= end.shape[0]
    # shift start: to do this, we check when the non-first dim indices are identical
    # and get a mask like [False, True, True, False, True, ...] where False means
    # that there's a switch from one dim to another (ie, a switch from one element of the batch
    # to another). We roll this one step along the time dimension and these two
    # masks provide us with the indices of the permutation matrix we need
    # to apply to start_idx.
    if start_idx.shape[0] > 1:
        start_idx_mask = (start_idx[1:, 1:] == start_idx[:-1, 1:]).all(-1)
        m1 = torch.cat([torch.zeros_like(start_idx_mask[:1]), start_idx_mask])
        m2 = torch.cat([start_idx_mask, torch.zeros_like(start_idx_mask[:1])])
        start_idx_replace = torch.empty_like(start_idx)
        start_idx_replace[m1] = start_idx[m2]
        start_idx_replace[~m1] = start_idx[~m2]
        start_idx = start_idx_replace
    else:
        # In this case we have only one start and stop has already been set
        pass
    lengths = stop_idx[:, 0] - start_idx[:, 0] + 1
    lengths[lengths <= 0] = lengths[lengths <= 0] + length
    return start_idx, stop_idx, lengths


def _tensor_slices_from_startend(seq_length, starts, storage_length):
    arange = torch.arange(seq_length, device=starts.device, dtype=starts.dtype)
    ndims = starts.shape[-1] - 1 if (starts.ndim - 1) else 0
    if ndims:
        arange_reshaped = torch.empty(list(arange.shape) + [ndims + 1], device=starts.device, dtype=starts.dtype)
        arange_reshaped[..., 0] = arange
        arange_reshaped[..., 1:] = 0
    else:
        arange_reshaped = arange.unsqueeze(-1)
    arange_expanded = arange_reshaped.expand([starts.shape[0]] + list(arange_reshaped.shape))
    n_missing_dims = arange_expanded.dim() - starts.dim()
    start_expanded = starts[(slice(None),) + (None,) * n_missing_dims].expand_as(arange_expanded)
    result = (start_expanded + arange_expanded).flatten(0, 1)
    result[:, 0] = result[:, 0] % storage_length
    return result


def key_exists(data, key):
    if isinstance(key, str):
        return key in data.keys()
    else:
        if len(key) == 1:
            return key[0] in data.keys()
        if key[0] not in data.keys():
            return False
        return key_exists(data[key[0]], key[1:])


def set_key(data, key, value):
    if isinstance(key, str):
        data[key] = value
    elif len(key) == 1:
        data[key[0]] = value
    else:
        set_key(data, key[1:], value)


def get_key(data, key):
    if isinstance(key, str):
        return data[key]
    elif len(key) == 1:
        return data[key[0]]
    else:
        return get_key(data, key[1:])
