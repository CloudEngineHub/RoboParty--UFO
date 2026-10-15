import gc
import glob
import os.path as osp
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from easydict import EasyDict
from loguru import logger
from rich.progress import track
from scipy.spatial.transform import Rotation as sRot

from ufo.math import (
    calc_heading_quat_inv,
    slerp,
)
from ufo.motion.kinematics import HumanoidKinematics


def to_torch(tensor):
    if torch.is_tensor(tensor):
        return tensor
    else:
        return torch.from_numpy(tensor)


def _raw_dof_pos_from_motion_file(curr_file, start, end, dtype):
    if "dof_pos" not in curr_file:
        return None
    dof_pos = to_torch(curr_file["dof_pos"]).clone()[start:end].to(dtype=dtype)
    if dof_pos.ndim != 2:
        raise ValueError(f"dof_pos must have shape [T, num_dof], got {tuple(dof_pos.shape)}")
    return dof_pos


def _dof_vel_from_dof_pos(dof_pos, dt):
    if dof_pos.shape[0] <= 1:
        return torch.zeros_like(dof_pos)
    dof_vel = (dof_pos[1:] - dof_pos[:-1]) / dt
    return torch.cat([dof_vel, dof_vel[-1:]], dim=0)


class MotionLib:
    def __init__(self, motion_lib_cfg, num_envs, device):
        if "fix_height" in motion_lib_cfg:
            raise ValueError("fix_height is retired; preprocess motion heights before training")
        self.m_cfg = motion_lib_cfg
        self.all_motions_loaded = False

        self.num_envs = num_envs
        self._device = device
        logger.info(f"Loading motion data from {self.m_cfg.motion_file}...")
        self.load_data(self.m_cfg.motion_file)
        self.setup_constants()

        self.kinematics = HumanoidKinematics(motion_lib_cfg)

    def load_data(self, motion_file):
        self._motion_source_ids_np = None
        self._motion_source_names = None
        self._motion_source_weights = None

        is_mixed_motion_file = isinstance(motion_file, (list, tuple))
        if not is_mixed_motion_file and not isinstance(motion_file, (str, bytes, Path)):
            try:
                motion_file = list(motion_file)
                is_mixed_motion_file = True
            except TypeError:
                is_mixed_motion_file = False

        if is_mixed_motion_file:
            data_values = []
            data_keys = []
            source_ids = []
            source_names = []
            for source_idx, source_path in enumerate(motion_file):
                source_path = str(source_path)
                source_names.append(Path(source_path).stem or source_path)
                if osp.isfile(source_path):
                    source_data = joblib.load(source_path)
                    if not isinstance(source_data, dict):
                        raise ValueError(f"Mixed motion pkl must contain a dict: {source_path}")
                    items = list(source_data.items())
                    data_keys.extend([k for k, _ in items])
                    data_values.extend([v for _, v in items])
                    source_ids.extend([source_idx] * len(items))
                else:
                    source_files = sorted(glob.glob(osp.join(source_path, "*.pkl")))
                    data_keys.extend(source_files)
                    data_values.extend(source_files)
                    source_ids.extend([source_idx] * len(source_files))

            weights = self.m_cfg.get("motion_file_weights", None)
            if weights is None:
                weights = [1.0 / len(motion_file)] * len(motion_file)
            else:
                weights = [float(w) for w in weights]
            if len(weights) != len(motion_file):
                raise ValueError(f"motion_file_weights length ({len(weights)}) must match motion_file length ({len(motion_file)})")
            if sum(weights) <= 0:
                raise ValueError("motion_file_weights must sum to a positive value")

            self._motion_data_list = np.array(data_values, dtype=object)
            self._motion_data_keys = np.array(data_keys)
            self._motion_source_ids_np = np.array(source_ids, dtype=np.int64)
            self._motion_source_names = source_names
            self._motion_source_weights = torch.tensor(weights, device=self._device, dtype=torch.float32)
            self._motion_source_weights = self._motion_source_weights / self._motion_source_weights.sum()
            self._num_unique_motions = len(self._motion_data_list)
            logger.info(f"Loaded {self._num_unique_motions} motions")
            return

        is_file = osp.isfile(motion_file)
        if is_file:
            source_data = joblib.load(motion_file)
            if not isinstance(source_data, dict):
                raise ValueError(f"Motion pkl must contain a dict: {motion_file}")
            self._motion_data_list = np.array(list(source_data.values()))
            self._motion_data_keys = np.array(list(source_data.keys()))
        else:
            source_data = glob.glob(osp.join(motion_file, "*.pkl"))
            self._motion_data_list = np.array(source_data)
            self._motion_data_keys = np.array(source_data)

        self._num_unique_motions = len(self._motion_data_list)
        self._motion_source_ids_np = np.zeros(self._num_unique_motions, dtype=np.int64)
        self._motion_source_names = [Path(str(motion_file)).stem or str(motion_file)]
        self._motion_source_weights = torch.ones(1, device=self._device, dtype=torch.float32)
        logger.info(f"Loaded {self._num_unique_motions} motions")

    def setup_constants(self):
        if self._num_unique_motions == 0:
            raise ValueError(f"No motions found in {self.m_cfg.motion_file}")
        self._curr_motion_ids = None
        self._motion_source_ids = torch.tensor(self._motion_source_ids_np, device=self._device, dtype=torch.long)
        self._sampling_prob = self._source_mixed_prob(torch.ones(self._num_unique_motions, device=self._device))
        mix_summary = ", ".join(
            f"{name}:{float(weight):.4f}/n={int((self._motion_source_ids == idx).sum().item())}"
            for idx, (name, weight) in enumerate(zip(self._motion_source_names, self._motion_source_weights))
        )
        logger.info(f"Motion source mix: {mix_summary}")

    def _source_counts(self, n, weights, exact_counts):
        if exact_counts:
            expected = weights * int(n)
            counts = torch.floor(expected).to(torch.long)
            remainder = int(n) - int(counts.sum().item())
            if remainder > 0:
                order = torch.argsort(expected - counts.to(expected.dtype), descending=True)
                counts[order[:remainder]] += 1
            return counts
        source_samples = torch.multinomial(weights, num_samples=int(n), replacement=True)
        return torch.bincount(source_samples, minlength=len(weights))

    def _source_mixed_prob(self, priorities):
        priorities = torch.as_tensor(priorities, device=self._device, dtype=torch.float32)
        priorities = torch.where(torch.isfinite(priorities), priorities, torch.zeros_like(priorities))
        priorities = priorities.clamp_min(0.0)
        if priorities.numel() != self._num_unique_motions:
            raise ValueError(f"Expected {self._num_unique_motions} priorities, got {priorities.numel()}")

        probs = torch.zeros(self._num_unique_motions, device=self._device, dtype=torch.float32)
        for source_idx, source_weight in enumerate(self._motion_source_weights):
            mask = self._motion_source_ids == source_idx
            if not bool(mask.any()) or float(source_weight) <= 0.0:
                continue
            source_priorities = priorities[mask]
            source_total = source_priorities.sum()
            if not torch.isfinite(source_total) or float(source_total) <= 0.0:
                source_priorities = torch.ones_like(source_priorities)
                source_total = source_priorities.sum()
            probs[mask] = source_priorities / source_total * source_weight

        total = probs.sum()
        if not torch.isfinite(total) or float(total) <= 0.0:
            probs = torch.ones(self._num_unique_motions, device=self._device, dtype=torch.float32)
            total = probs.sum()
        return probs / total

    def _refresh_sampling_batch_prob(self):
        if self._curr_motion_ids is None:
            return
        batch_prob = self._sampling_prob[self._curr_motion_ids]
        total = batch_prob.sum()
        if not torch.isfinite(total) or float(total) <= 0.0:
            batch_prob = torch.ones_like(batch_prob)
            total = batch_prob.sum()
        self._sampling_batch_prob = batch_prob / total

    def _sample_indices_with_source_mix(self, probabilities, source_ids, n, exact_counts):
        probabilities = torch.as_tensor(probabilities, device=self._device, dtype=torch.float32)
        probabilities = torch.where(torch.isfinite(probabilities), probabilities, torch.zeros_like(probabilities)).clamp_min(0.0)
        source_ids = torch.as_tensor(source_ids, device=self._device, dtype=torch.long)

        available_weights = self._motion_source_weights.clone()
        for source_idx in range(len(available_weights)):
            if not bool((source_ids == source_idx).any()):
                available_weights[source_idx] = 0.0
        if float(available_weights.sum()) <= 0.0:
            total = probabilities.sum()
            if not torch.isfinite(total) or float(total) <= 0.0:
                probabilities = torch.ones_like(probabilities)
                total = probabilities.sum()
            return torch.multinomial(probabilities / total, num_samples=int(n), replacement=True).to(self._device)

        available_weights = available_weights / available_weights.sum()
        counts = self._source_counts(n, available_weights, exact_counts=exact_counts)
        sampled = []
        for source_idx, count in enumerate(counts.tolist()):
            if count <= 0:
                continue
            candidate_ids = (source_ids == source_idx).nonzero(as_tuple=False).flatten()
            source_prob = probabilities[candidate_ids]
            source_total = source_prob.sum()
            if not torch.isfinite(source_total) or float(source_total) <= 0.0:
                source_prob = torch.ones_like(source_prob)
                source_total = source_prob.sum()
            local_samples = torch.multinomial(source_prob / source_total, num_samples=count, replacement=True)
            sampled.append(candidate_ids[local_samples])

        sample_idxes = torch.cat(sampled) if sampled else torch.empty(0, device=self._device, dtype=torch.long)
        if sample_idxes.numel() != int(n):
            raise RuntimeError(f"Source-stratified sampler produced {sample_idxes.numel()} samples, expected {n}")
        return sample_idxes[torch.randperm(sample_idxes.numel(), device=self._device)]

    def update_sampling_weight_by_id(self, priorities: list, motions_id: list, file_name: list | None = None) -> None:
        """
        Update sampling probabilities (_sampling_prob) given motion keys and their priorities.
        This replaces the distribution with source-stratified values. If multiple motion files
        were configured, importance sampling is only allowed to rebalance motions within each
        file; the total probability mass of each file remains fixed.
        """
        assert len(motions_id) == len(priorities), "motions_id and priorities must have the same length"
        priority_values = torch.as_tensor(priorities, device=self._device, dtype=torch.float32)
        assert torch.isfinite(priority_values).all(), "Priorities should be finite"
        priority_values = priority_values.clamp_min(0.0)
        assert float(priority_values.sum()) > 0, "Sum of priorities must be greater than zero"

        # Get all motion keys
        all_keys = self._motion_data_keys.tolist()
        new_sampling_priorities = torch.zeros(self._num_unique_motions, device=self._device)

        for offset, (m, p) in enumerate(zip(motions_id, priority_values)):
            m = int(m)
            new_sampling_priorities[m] = p
            if file_name is not None:
                expected_file = None
                if isinstance(file_name, dict):
                    expected_file = file_name.get(m)
                elif len(file_name) == self._num_unique_motions:
                    expected_file = file_name[m]
                elif offset < len(file_name):
                    expected_file = file_name[offset]
                if expected_file is not None and all_keys[m] != expected_file:
                    print(
                        f"[Warning] Motion ID {m} expected file {expected_file}, got {all_keys[m]}; continuing without file-name assertion."
                    )

        if new_sampling_priorities.sum() == 0:
            print("[Warning] No valid motions updated. Falling back to uniform sampling.")
            new_sampling_priorities = torch.ones(self._num_unique_motions, device=self._device)

        self._sampling_prob = self._source_mixed_prob(new_sampling_priorities)
        self._refresh_sampling_batch_prob()

    def get_motion_state(self, motion_ids, motion_times, offset=None):
        motion_len = self._motion_lengths[motion_ids]
        num_frames = self._motion_num_frames[motion_ids]
        dt = self._motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        dof_pos0 = self.dof_pos[f0l]
        dof_pos1 = self.dof_pos[f1l]

        body_vel0 = self.gvs[f0l]
        body_vel1 = self.gvs[f1l]

        body_ang_vel0 = self.gavs[f0l]
        body_ang_vel1 = self.gavs[f1l]

        rg_pos0 = self.gts[f0l, :]
        rg_pos1 = self.gts[f1l, :]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        vals = [dof_pos0, dof_pos1, body_vel0, body_vel1, body_ang_vel0, body_ang_vel1, rg_pos0, rg_pos1, dof_vel0, dof_vel1]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        blend_exp = blend.unsqueeze(-1)

        if offset is None:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1  # ZL: apply offset
        else:
            rg_pos = (1.0 - blend_exp) * rg_pos0 + blend_exp * rg_pos1 + offset[..., None, :]  # ZL: apply offset

        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1

        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]
        rb_rot = slerp(rb_rot0, rb_rot1, blend_exp)
        return_dict = {}

        if "gts_t" in self.__dict__:
            rg_pos_t0 = self.gts_t[f0l]
            rg_pos_t1 = self.gts_t[f1l]

            rg_rot_t0 = self.grs_t[f0l]
            rg_rot_t1 = self.grs_t[f1l]

            body_vel_t0 = self.gvs_t[f0l]
            body_vel_t1 = self.gvs_t[f1l]

            body_ang_vel_t0 = self.gavs_t[f0l]
            body_ang_vel_t1 = self.gavs_t[f1l]
            if offset is None:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1
            else:
                rg_pos_t = (1.0 - blend_exp) * rg_pos_t0 + blend_exp * rg_pos_t1 + offset[..., None, :]
            rg_rot_t = slerp(rg_rot_t0, rg_rot_t1, blend_exp)
            body_vel_t = (1.0 - blend_exp) * body_vel_t0 + blend_exp * body_vel_t1
            body_ang_vel_t = (1.0 - blend_exp) * body_ang_vel_t0 + blend_exp * body_ang_vel_t1
        else:
            rg_pos_t = rg_pos
            rg_rot_t = rb_rot
            body_vel_t = body_vel
            body_ang_vel_t = body_ang_vel

        return_dict.update(
            {
                "root_pos": rg_pos[..., 0, :].clone(),
                "root_rot": rb_rot[..., 0, :].clone(),
                "dof_pos": dof_pos.clone(),
                "root_vel": body_vel[..., 0, :].clone(),
                "root_ang_vel": body_ang_vel[..., 0, :].clone(),
                "dof_vel": dof_vel.view(dof_vel.shape[0], -1).clone(),
                "rg_pos": rg_pos.clone(),
                "rb_rot": rb_rot.clone(),
                "body_vel": body_vel.clone(),
                "body_ang_vel": body_ang_vel.clone(),
                "rg_pos_t": rg_pos_t.clone(),
                "rg_rot_t": rg_rot_t.clone(),
                "body_vel_t": body_vel_t.clone(),
                "body_ang_vel_t": body_ang_vel_t.clone(),
            }
        )
        return return_dict

    def load_motions_for_training(self, max_num_seqs=None):
        if max_num_seqs is not None:
            assert max_num_seqs <= self.num_envs

        if self.all_motions_loaded:
            print("All motions already loaded!!! No need to resample.")
            return

        if self.m_cfg.get("override_num_motions_to_load", None) is not None:
            max_num_seqs = self.m_cfg.override_num_motions_to_load
        if max_num_seqs is None:  # if not specified, load all motions, can OOM if the dataset is too large.
            max_num_seqs = self._num_unique_motions
            self.all_motions_loaded = True
            self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)
        else:  #
            if (
                max_num_seqs > self._num_unique_motions
            ):  # if specified but more than the number of unique motions, load all motions as well.
                self.all_motions_loaded = True
                self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)
            else:  # if there are more motions than specified, then randomly sample the requested number of motions.
                self.all_motions_loaded = False
                self.load_motions(random_sample=True, num_motions_to_load=max_num_seqs)

    def load_motions_for_evaluation(self, start_idx=0):
        if self.all_motions_loaded:
            print("All motions already loaded!!! No need to resample.")
            return

        if (
            self._num_unique_motions > self.num_envs
        ):  # if number of motions is more than number of envs, then we should only partially load the motions.
            self.all_motions_loaded = False
            self.load_motions(random_sample=False, num_motions_to_load=self.num_envs, start_idx=start_idx)
        else:
            self.all_motions_loaded = True
            self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions, start_idx=start_idx)

    def load_all_motions(self):
        self.all_motions_loaded = True
        self.load_motions(random_sample=False, num_motions_to_load=self._num_unique_motions)

    def load_motions(self, random_sample=True, start_idx=0, max_len=-1, target_heading=None, num_motions_to_load=None):
        if "gts" in self.__dict__:
            del self.gts, self.grs, self.gavs, self.gvs, self.dvs, self.dof_pos
            if "gts_t" in self.__dict__:
                del self.gts_t, self.grs_t, self.gvs_t, self.gavs_t

        motions = []
        _motion_lengths = []
        _motion_dt = []
        _motion_num_frames = []

        total_len = 0.0
        if num_motions_to_load is None:
            num_motion_to_load = self.num_envs
        else:
            num_motion_to_load = num_motions_to_load

        if random_sample:
            sample_idxes = self._sample_indices_with_source_mix(
                self._sampling_prob, self._motion_source_ids, num_motion_to_load, exact_counts=True
            )
        else:  # start_idx only used for non-random sampling.
            sample_idxes = torch.clamp(torch.arange(num_motion_to_load) + start_idx, max=self._num_unique_motions - 1).to(self._device)

        self._curr_motion_ids = sample_idxes
        self.curr_motion_keys = (
            [self._motion_data_keys[sample_idxes.cpu()]]
            if sample_idxes.numel() == 1
            else self._motion_data_keys[sample_idxes.cpu()].tolist()
        )
        self._refresh_sampling_batch_prob()

        logger.info(f"Loading {num_motion_to_load} motions...")
        logger.info(f"Sampling motion: {sample_idxes[:10]}, ....")
        logger.info(f"Current motion keys: {self.curr_motion_keys[:10]}, ....")

        motion_data_list = self._motion_data_list[sample_idxes.cpu().numpy()]
        torch.set_num_threads(1)
        res_acc = self.load_motion_with_skeleton(motion_data_list, target_heading, max_len)

        for f in track(range(len(res_acc)), description="Processing motions..."):
            curr_motion = res_acc[f]
            motion_fps = curr_motion.fps
            curr_dt = 1.0 / motion_fps
            num_frames = curr_motion.global_rotation.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            _motion_dt.append(curr_dt)
            _motion_num_frames.append(num_frames)
            motions.append(curr_motion)
            _motion_lengths.append(curr_len)
            del curr_motion

        self._motion_lengths = torch.tensor(_motion_lengths, device=self._device, dtype=torch.float32)
        self._motion_dt = torch.tensor(_motion_dt, device=self._device, dtype=torch.float32)
        self._motion_num_frames = torch.tensor(_motion_num_frames, device=self._device)

        self._num_motions = len(motions)

        self.gts = torch.cat([m.global_translation for m in motions], dim=0).float().to(self._device)
        self.grs = torch.cat([m.global_rotation for m in motions], dim=0).float().to(self._device)
        self.gavs = torch.cat([m.global_angular_velocity for m in motions], dim=0).float().to(self._device)
        self.gvs = torch.cat([m.global_velocity for m in motions], dim=0).float().to(self._device)
        self.dvs = torch.cat([m.dof_vels for m in motions], dim=0).float().to(self._device)
        if "global_translation_extend" in motions[0].__dict__:
            self.gts_t = torch.cat([m.global_translation_extend for m in motions], dim=0).float().to(self._device)
            self.grs_t = torch.cat([m.global_rotation_extend for m in motions], dim=0).float().to(self._device)
            self.gvs_t = torch.cat([m.global_velocity_extend for m in motions], dim=0).float().to(self._device)
            self.gavs_t = torch.cat([m.global_angular_velocity_extend for m in motions], dim=0).float().to(self._device)

        self.dof_pos = torch.cat([m.dof_pos for m in motions], dim=0).float().to(self._device)

        lengths = self._motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.length_starts = lengths_shifted.cumsum(0)

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        logger.info(f"Loaded {num_motions:d} motions with a total length of {total_len:.3f}s and {self.gts.shape[0]} frames.")

        del motions, _motion_lengths, _motion_dt, _motion_num_frames
        torch.cuda.empty_cache()
        gc.collect()

    def load_motion_with_skeleton(self, motion_data_list, target_heading, max_len):
        # loading motion with the specified skeleton. Perfoming forward kinematics to get the joint positions
        res = {}

        for f in track(range(len(motion_data_list)), description="Loading motions..."):
            curr_file = motion_data_list[f]
            if not isinstance(curr_file, dict) and osp.isfile(curr_file):
                key = motion_data_list[f].split("/")[-1].split(".")[0]
                motion_path = curr_file
                source_data = joblib.load(motion_path)
                if not isinstance(source_data, dict) or key not in source_data:
                    raise ValueError(f"Motion pkl must contain key {key!r}: {motion_path}")
                curr_file = source_data[key]

            seq_len = curr_file["root_trans_offset"].shape[0]
            if max_len == -1 or seq_len < max_len:
                start, end = 0, seq_len
            else:
                start = random.randint(0, seq_len - max_len)
                end = start + max_len

            trans = to_torch(curr_file["root_trans_offset"]).clone()[start:end]
            pose_aa = to_torch(curr_file["pose_aa"][start:end]).clone()

            dt = 1 / curr_file["fps"]

            B, J, N = pose_aa.shape

            if target_heading is not None:
                start_root_rot = sRot.from_rotvec(pose_aa[0, 0])
                heading_inv_rot = sRot.from_quat(calc_heading_quat_inv(torch.from_numpy(start_root_rot.as_quat()[None,]), w_last=True))
                heading_delta = sRot.from_quat(target_heading) * heading_inv_rot
                pose_aa[:, 0] = torch.tensor((heading_delta * sRot.from_rotvec(pose_aa[:, 0])).as_rotvec())

                trans = torch.matmul(trans, torch.from_numpy(heading_delta.as_matrix().squeeze().T).to(trans))

            curr_motion = self.kinematics.fk_batch(pose_aa[None,], trans[None,], dt=dt)
            curr_motion = EasyDict({k: v.squeeze() if torch.is_tensor(v) else v for k, v in curr_motion.items()})
            raw_dof_pos = _raw_dof_pos_from_motion_file(curr_file, start, end, pose_aa.dtype)
            if raw_dof_pos is not None:
                curr_motion.dof_pos = raw_dof_pos
                curr_motion.dof_vels = _dof_vel_from_dof_pos(raw_dof_pos, dt)
            # Imported derivatives preserve source differences and clip boundaries; body velocities use link origins in world axes.
            for field, target, shape in (
                ("dof_vel", "dof_vels", (seq_len, curr_motion.dof_pos.shape[-1])),
                ("body_vel", "global_velocity", (seq_len, self.kinematics.num_bodies, 3)),
                ("body_ang_vel", "global_angular_velocity", (seq_len, self.kinematics.num_bodies, 3)),
            ):
                if field not in curr_file:
                    continue
                velocity = to_torch(curr_file[field]).to(dtype=pose_aa.dtype)
                if tuple(velocity.shape) != shape or not torch.isfinite(velocity).all():
                    raise ValueError(f"{field} must be finite with shape {shape}, got {tuple(velocity.shape)}")
                velocity = velocity[start:end].clone()
                if field != "dof_vel":
                    if self.kinematics.cfg.extend_config:
                        raise ValueError("Stored body velocities require a robot without motion_extensions")
                    if target_heading is not None:
                        velocity = velocity @ torch.from_numpy(heading_delta.as_matrix().squeeze().T).to(velocity)
                curr_motion[target] = velocity
            res[f] = curr_motion

        return res

    def num_motions(self):
        return self._num_motions

    def get_total_length(self):
        return sum(self._motion_lengths)

    def sample_time(self, motion_ids, truncate_time=None):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time

        motion_time = phase * motion_len
        return motion_time.to(self._device)

    def sample_motions(self, n):
        source_ids = self._motion_source_ids[self._curr_motion_ids]
        motion_ids = self._sample_indices_with_source_mix(self._sampling_batch_prob, source_ids, n, exact_counts=False).to(self._device)

        return motion_ids

    def _calc_frame_blend(self, time, len, num_frames, dt):
        time = time.clone()
        phase = time / len
        phase = torch.clip(phase, 0.0, 1.0)  # clip time to be within motion length.
        time[time < 0] = 0

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = torch.clip((time - frame_idx0 * dt) / dt, 0.0, 1.0)  # clip blend to be within 0 and 1

        return frame_idx0, frame_idx1, blend
