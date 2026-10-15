"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


# Old quat_rotate_inverse removed - use the version with w_last parameter below


@torch.jit.script
def quat_unit(a):
    return normalize(a)



@torch.jit.script
def wrap_to_pi(angles):
    """Wrap angles to [-pi, pi] range."""
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles


@torch.jit.script
def quat_conjugate(a: Tensor, w_last: bool) -> Tensor:
    """Compute quaternion conjugate."""
    shape = a.shape
    a = a.reshape(-1, 4)
    if w_last:
        return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)
    else:
        return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


@torch.jit.script
def quat_rotate_inverse(q: Tensor, v: Tensor, w_last: bool) -> Tensor:
    """Rotate vector by inverse quaternion."""
    shape = q.shape
    if w_last:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    else:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_angle_axis(x: Tensor, w_last: bool) -> Tuple[Tensor, Tensor]:
    """Convert quaternion to angle-axis representation."""
    if w_last:
        w = x[..., -1]
        axis = x[..., :3]
    else:
        w = x[..., 0]
        axis = x[..., 1:]
    s = 2 * (w**2) - 1
    angle = s.clamp(-1, 1).arccos()
    axis /= axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return angle, axis


@torch.jit.script
def quat_from_angle_axis(angle: Tensor, axis: Tensor, w_last: bool) -> Tensor:
    """Convert angle-axis to quaternion."""
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if w_last:
        return quat_unit(torch.cat([xyz, w], dim=-1))
    else:
        return quat_unit(torch.cat([w, xyz], dim=-1))



@torch.jit.script
def slerp(q0, q1, t):
    """Spherical linear interpolation between two quaternions."""
    cos_half_theta = torch.sum(q0 * q1, dim=-1)

    neg_mask = cos_half_theta < 0
    q1 = q1.clone()
    q1[neg_mask] = -q1[neg_mask]
    cos_half_theta = torch.abs(cos_half_theta)
    cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

    half_theta = torch.acos(cos_half_theta)
    sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

    ratioA = torch.sin((1 - t) * half_theta) / sin_half_theta
    ratioB = torch.sin(t * half_theta) / sin_half_theta

    new_q = ratioA * q0 + ratioB * q1

    new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
    new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)

    return new_q



@torch.jit.script
def my_quat_rotate(q, v):
    """Rotate vector by quaternion (assumes w_last=True format)."""
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_to_tan_norm(q: Tensor, w_last: bool) -> Tensor:
    """Represent rotation using tangent and normal vectors."""
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    if w_last:
        tan = my_quat_rotate(q, ref_tan)
    else:
        raise NotImplementedError

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    if w_last:
        norm = my_quat_rotate(q, ref_norm)
    else:
        raise NotImplementedError

    norm_tan = torch.cat([tan, norm], dim=len(tan.shape) - 1)
    return norm_tan


@torch.jit.script
def calc_heading(q):
    """Calculate heading direction from quaternion."""
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = my_quat_rotate(q, ref_dir)

    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading



@torch.jit.script
def calc_heading_quat_inv(q, w_last: bool):
    # type: (Tensor, bool) -> Tensor
    """Calculate inverse heading rotation from quaternion."""
    heading = calc_heading(q)
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = quat_from_angle_axis(-heading, axis, w_last=w_last)
    return heading_q


@torch.jit.script
def quat_inverse(x, w_last: bool):
    # type: (Tensor, bool) -> Tensor
    """Compute inverse quaternion."""
    return quat_conjugate(x, w_last=w_last)


@torch.jit.script
def quat_mul(a, b, w_last: bool):
    # type: (Tensor, Tensor, bool) -> Tensor
    """Multiply two quaternions."""
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if w_last:
        x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    else:
        w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
        w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if w_last:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)
    else:
        quat = torch.stack([w, x, y, z], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_mul_norm(x, y, w_last: bool):
    # type: (Tensor, Tensor, bool) -> Tensor
    """Multiply and normalize quaternions."""
    return quat_unit(quat_mul(x, y, w_last))


@torch.jit.script
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices (assumes wxyz format)."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


@torch.jit.script
def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to quaternion (wxyz format)."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    sin_half_angles_over_angles[small_angles] = 0.5 - (angles[small_angles] * angles[small_angles]) / 48
    quaternions = torch.cat([torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1)
    return quaternions


# @torch.jit.script
def wxyz_to_xyzw(quat):
    """Convert quaternion from wxyz to xyzw format."""
    return quat[..., [1, 2, 3, 0]]


# @torch.jit.script
def xyzw_to_wxyz(quat):
    """Convert quaternion from xyzw to wxyz format."""
    return quat[..., [3, 0, 1, 2]]


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (wxyz format)."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    return quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Returns torch.sqrt(torch.max(0, x)) but with a zero subgradient where x is 0."""
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret
