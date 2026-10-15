"""Read, validate, convert, and clip motion data in supported UFO formats."""

from __future__ import annotations

import csv
import glob
import os
from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from loguru import logger

from ufo.tools.robot_config import RobotSpec

REQUIRED_MOTION_FIELDS = ("root_trans_offset", "pose_aa", "fps")


def _fail(source_name: str, motion_key: str | None, message: str) -> None:
    key_text = f", motion_key={motion_key}" if motion_key is not None else ""
    raise ValueError(f"Invalid UFO motion data source={source_name}{key_text}: {message}")


def _coerce_fps(fps: Any, source_name: str, motion_key: str) -> float:
    fps_arr = np.asarray(fps)
    if fps_arr.size != 1:
        _fail(source_name, motion_key, f"fps must be a scalar, got shape {fps_arr.shape}")
    try:
        fps_value = float(fps_arr.reshape(-1)[0])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid UFO motion data source={source_name}, motion_key={motion_key}: fps is not numeric") from exc
    if not np.isfinite(fps_value) or fps_value <= 0.0:
        _fail(source_name, motion_key, f"fps must be > 0, got {fps_value}")
    return fps_value


def validate_ufo_motion_dict(data: dict[str, dict[str, Any]] | Mapping[str, Mapping[str, Any]], source_name: str) -> dict[str, Any]:
    """Validate and return a standard UFO/UFO motion dictionary.

    The validator intentionally checks only the canonical fields required to
    enter MotionLib: root translation, axis-angle pose, and per-motion fps.
    It does not resample fps or retarget skeletons.
    """

    if not isinstance(data, Mapping):
        raise ValueError(f"Invalid UFO motion data source={source_name}: expected dict[str, dict], got {type(data).__name__}")
    if len(data) == 0:
        raise ValueError(f"Invalid UFO motion data source={source_name}: motion dictionary is empty")

    validated: dict[str, Any] = {}
    for key, motion in data.items():
        motion_key = str(key)
        if not isinstance(motion, Mapping):
            _fail(source_name, motion_key, f"motion record must be a dict, got {type(motion).__name__}")

        for field in REQUIRED_MOTION_FIELDS:
            if field not in motion:
                _fail(source_name, motion_key, f"missing required field '{field}'")

        root_trans_offset = np.asarray(motion["root_trans_offset"])
        pose_aa = np.asarray(motion["pose_aa"])
        _coerce_fps(motion["fps"], source_name, motion_key)

        if root_trans_offset.ndim != 2 or root_trans_offset.shape[1] != 3:
            _fail(source_name, motion_key, f"root_trans_offset must have shape [T, 3], got {root_trans_offset.shape}")
        if pose_aa.ndim != 3 or pose_aa.shape[2] != 3:
            _fail(source_name, motion_key, f"pose_aa must have shape [T, J, 3], got {pose_aa.shape}")
        if root_trans_offset.shape[0] <= 0:
            _fail(source_name, motion_key, "motion must contain at least one frame")
        if root_trans_offset.shape[0] != pose_aa.shape[0]:
            _fail(
                source_name,
                motion_key,
                f"root_trans_offset and pose_aa must share T, got {root_trans_offset.shape[0]} and {pose_aa.shape[0]}",
            )

        validated[motion_key] = motion

    return validated


def motion_fps_values(data: Mapping[str, Mapping[str, Any]], source_name: str) -> list[float]:
    return [_coerce_fps(motion["fps"], source_name, str(key)) for key, motion in data.items()]


def format_fps_distribution(data: Mapping[str, Mapping[str, Any]], source_name: str) -> str:
    fps_values = motion_fps_values(data, source_name)
    counts = Counter(fps_values)
    parts: list[str] = []
    for fps_value in sorted(counts):
        if float(fps_value).is_integer():
            fps_text = str(int(fps_value))
        else:
            fps_text = f"{fps_value:.6g}"
        parts.append(f"{fps_text}: {counts[fps_value]}")
    return "{" + ", ".join(parts) + "}"


def _slice_motion_record(record: dict[str, Any], start: int, end: int, total_frames: int) -> dict[str, Any]:
    clipped: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == total_frames:
            clipped[key] = value[start:end].copy()
        else:
            clipped[key] = deepcopy(value)
    return clipped


def motion_clip_windows(total_frames, fps, *, clip_seconds, stride_seconds, keep_short, min_clip_seconds):
    if clip_seconds <= 0.0:
        raise ValueError(f"clip_seconds must be > 0, got {clip_seconds}")
    if stride_seconds <= 0.0:
        raise ValueError(f"stride_seconds must be > 0, got {stride_seconds}")
    if min_clip_seconds < 0.0:
        raise ValueError(f"min_clip_seconds must be >= 0, got {min_clip_seconds}")

    clip_frames = max(1, int(round(clip_seconds * fps)))
    stride_frames = max(1, int(round(stride_seconds * fps)))
    min_frames = max(1, int(round(min_clip_seconds * fps))) if min_clip_seconds > 0.0 else 1

    windows: list[tuple[int, int]] = []
    if total_frames >= clip_frames:
        start = 0
        while start + clip_frames <= total_frames:
            windows.append((start, start + clip_frames))
            start += stride_frames
        if keep_short and start < total_frames and total_frames - start >= min_frames:
            windows.append((start, total_frames))
    elif keep_short and total_frames >= min_frames:
        windows.append((0, total_frames))

    return windows


def clip_ufo_motion_dict(
    data: dict[str, dict[str, Any]],
    clip_seconds: float = 10.0,
    stride_seconds: float = 10.0,
    keep_short: bool = True,
    min_clip_seconds: float = 1.0,
    source_name: str = "",
) -> dict[str, Any]:
    """Clip each motion while preserving all synchronized time-series fields.

    No fps resampling is performed. Each motion uses its own fps to convert
    seconds into frame counts.
    """

    validated = validate_ufo_motion_dict(data, source_name or "clip")
    clipped_data: dict[str, Any] = {}
    for motion_key, record in validated.items():
        total_frames = int(np.asarray(record["root_trans_offset"]).shape[0])
        fps = float(np.asarray(record["fps"]).reshape(-1)[0])
        windows = motion_clip_windows(total_frames, fps, clip_seconds=clip_seconds, stride_seconds=stride_seconds,
                                      keep_short=keep_short, min_clip_seconds=min_clip_seconds)

        for clip_idx, (start, end) in enumerate(windows):
            new_key = f"{motion_key}__clip{clip_idx:03d}"
            if new_key in clipped_data:
                raise ValueError(f"Duplicate clipped motion key: {new_key}")
            clipped = _slice_motion_record(record, start, end, total_frames)
            clipped["motion_key"] = new_key
            metadata = dict(clipped.get("metadata") or {})
            metadata.update(
                {
                    "source_motion_key": motion_key,
                    "clip_start_frame": start,
                    "clip_end_frame": end,
                    "clip_seconds": clip_seconds,
                    "stride_seconds": stride_seconds,
                }
            )
            clipped["metadata"] = metadata
            clipped_data[new_key] = clipped

    if not clipped_data:
        raise ValueError(
            f"No motion clips were generated for source={source_name!r}. Try keep_short=true or min_clip_seconds <= shortest motion length."
        )
    return validate_ufo_motion_dict(clipped_data, source_name or "clip")


@dataclass(frozen=True)
class RobotStateMotion:
    motion_key: str
    root_pos: np.ndarray
    root_quat: np.ndarray
    dof_pos: np.ndarray
    fps: float
    joint_names: list[str] | None = None
    source: str | None = None
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)


def _context(source_name: str, motion_key: str | None = None) -> str:
    return f"{source_name}:{motion_key}" if motion_key else source_name


def _as_float_matrix(value: Any, field_name: str, width: int, context: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{context} {field_name} must have shape [T, {width}], got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{context} {field_name} contains non-finite values")
    return arr


def reorder_dof_by_joint_names(
    dof_pos: np.ndarray,
    input_joint_names: list[str],
    target_joint_names: list[str],
    source_name: str,
) -> np.ndarray:
    dof = np.asarray(dof_pos, dtype=np.float32)
    if dof.ndim != 2:
        raise ValueError(f"{source_name} dof_pos must have shape [T, N], got {dof.shape}")
    if len(input_joint_names) != dof.shape[1]:
        raise ValueError(
            f"{source_name} joint_names length must match dof_pos width, got {len(input_joint_names)} names for width {dof.shape[1]}"
        )
    if len(set(input_joint_names)) != len(input_joint_names):
        raise ValueError(f"{source_name} joint_names contains duplicates")
    extra = [joint for joint in input_joint_names if joint not in target_joint_names]
    if extra:
        raise ValueError(f"{source_name} unexpected joint_names entries: {extra}")
    joint_to_idx = {name: idx for idx, name in enumerate(input_joint_names)}
    missing = [joint for joint in target_joint_names if joint not in joint_to_idx]
    if missing:
        raise ValueError(f"{source_name} missing joint_names entries: {missing}")
    return dof[:, [joint_to_idx[joint] for joint in target_joint_names]].copy()


def validate_robot_state_motion(
    motion: RobotStateMotion,
    robot_spec: RobotSpec,
    source_name: str,
) -> RobotStateMotion:
    context = _context(source_name, motion.motion_key)
    root = _as_float_matrix(motion.root_pos, "root_pos", 3, context)
    quat = _as_float_matrix(motion.root_quat, "root_quat", 4, context)
    dof = np.asarray(motion.dof_pos, dtype=np.float32)
    if dof.ndim != 2:
        raise ValueError(f"{context} dof_pos must have shape [T, N], got {dof.shape}")
    if not np.all(np.isfinite(dof)):
        raise ValueError(f"{context} dof_pos contains non-finite values")
    if root.shape[0] != quat.shape[0] or root.shape[0] != dof.shape[0]:
        raise ValueError(f"{context} root_pos/root_quat/dof_pos must share T, got {root.shape[0]}, {quat.shape[0]}, {dof.shape[0]}")

    fps = float(motion.fps)
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{context} fps must be > 0, got {motion.fps}")

    quat_norm = np.linalg.norm(quat, axis=1)
    if np.any(quat_norm <= 0.0) or np.any(~np.isfinite(quat_norm)):
        raise ValueError(f"{context} root_quat contains zero or non-finite quaternions")

    target_joint_names = list(robot_spec.control_joint_names)
    if motion.joint_names is not None:
        dof = reorder_dof_by_joint_names(dof, list(motion.joint_names), target_joint_names, context)
        joint_names: list[str] | None = target_joint_names
    else:
        expected_dofs = len(target_joint_names)
        if dof.shape[1] != expected_dofs:
            raise ValueError(f"{context} dof_pos must have shape [T, {expected_dofs}], got {dof.shape}")
        joint_names = None

    return RobotStateMotion(
        motion_key=str(motion.motion_key),
        root_pos=root,
        root_quat=quat,
        dof_pos=dof,
        fps=fps,
        joint_names=joint_names,
        source=motion.source or source_name,
        metadata=dict(motion.metadata or {}),
    )


def validate_robot_state_dict(
    data: dict[str, RobotStateMotion],
    robot_spec: RobotSpec,
    source_name: str,
) -> dict[str, RobotStateMotion]:
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{source_name} RobotState data must be a non-empty dict")
    validated: dict[str, RobotStateMotion] = {}
    for key, motion in data.items():
        if not isinstance(motion, RobotStateMotion):
            raise ValueError(f"{source_name}:{key} must be a RobotStateMotion")
        canonical = validate_robot_state_motion(motion, robot_spec, source_name)
        motion_key = str(canonical.motion_key)
        if motion_key in validated:
            raise ValueError(f"Duplicate RobotState motion_key={motion_key} in source={source_name}")
        if motion_key != str(key):
            raise ValueError(f"RobotState dict key={key!r} does not match motion_key={motion_key!r}")
        validated[motion_key] = canonical
    return validated


def _root_quat_to_xyzw(root_quat: np.ndarray, order: str) -> np.ndarray:
    quat = np.asarray(root_quat, dtype=np.float32)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"root_quat must have shape [T, 4], got {quat.shape}")
    if order == "xyzw":
        xyzw = quat.copy()
    elif order == "wxyz":
        xyzw = quat[:, [1, 2, 3, 0]].copy()
    else:
        raise ValueError(f"Unsupported root_quat_order={order!r}")
    norm = np.linalg.norm(xyzw, axis=1, keepdims=True)
    if np.any(norm <= 0.0) or np.any(~np.isfinite(norm)):
        raise ValueError("root_quat contains zero or non-finite quaternions")
    return xyzw / norm


def _quat_xyzw_to_axis_angle(root_quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(root_quat_xyzw, dtype=np.float64)
    vec = quat[:, :3]
    w = np.clip(quat[:, 3], -1.0, 1.0)
    vec_norm = np.linalg.norm(vec, axis=1)
    angle = 2.0 * np.arctan2(vec_norm, w)
    axis = np.zeros_like(vec)
    valid = vec_norm > 1e-8
    axis[valid] = vec[valid] / vec_norm[valid, None]
    return (axis * angle[:, None]).astype(np.float32)


def robot_state_to_ufo_motion(
    motion: RobotStateMotion,
    robot_spec: RobotSpec,
    source_name: str | None = None,
) -> dict[str, Any]:
    """Convert one validated RobotStateMotion into the UFO pose_aa schema."""

    effective_source = source_name or motion.source or "robot_state"
    motion = validate_robot_state_motion(motion, robot_spec, effective_source)
    dof = motion.dof_pos.astype(np.float32)
    if robot_spec.dof_unit == "deg":
        dof = np.deg2rad(dof)
    elif robot_spec.dof_unit != "rad":
        raise ValueError(f"Unsupported dof_unit={robot_spec.dof_unit!r}; expected 'rad' or 'deg'")

    pose_aa = np.zeros((motion.root_pos.shape[0], len(robot_spec.body_names), 3), dtype=np.float32)
    body_to_idx = {name: idx for idx, name in enumerate(robot_spec.body_names)}
    if robot_spec.base_body not in body_to_idx:
        raise ValueError(f"RobotSpec base_body={robot_spec.base_body!r} is not present in body_names")
    pose_aa[:, body_to_idx[robot_spec.base_body]] = _quat_xyzw_to_axis_angle(
        _root_quat_to_xyzw(motion.root_quat, robot_spec.root_quat_order)
    )

    for dof_idx, joint_name in enumerate(robot_spec.control_joint_names):
        joint_type = robot_spec.joint_types[joint_name]
        if joint_type != "hinge":
            raise ValueError(f"robot_state_to_ufo_motion currently supports hinge control joints only, got {joint_name}:{joint_type}")
        body_name = robot_spec.joint_body_names[joint_name]
        if body_name not in body_to_idx:
            raise ValueError(f"Control joint {joint_name} attaches to body {body_name}, which is not in RobotSpec.body_names")
        axis = np.asarray(robot_spec.joint_axes[joint_name], dtype=np.float32)
        pose_aa[:, body_to_idx[body_name]] = dof[:, dof_idx : dof_idx + 1] * axis[None, :]

    metadata = dict(motion.metadata or {})
    metadata.update(
        {
            "source_name": effective_source,
            "motion_key": motion.motion_key,
            "robot_name": robot_spec.name,
            "xml_path": robot_spec.xml_path,
            "root_quat_order": robot_spec.root_quat_order,
            "coordinate_system": robot_spec.coordinate_system,
            "dof_unit": robot_spec.dof_unit,
        }
    )
    return {
        "root_trans_offset": motion.root_pos.astype(np.float32),
        "pose_aa": pose_aa,
        "fps": float(motion.fps),
        "dof_pos": dof.astype(np.float32),
        "root_quat": motion.root_quat.astype(np.float32),
        "joint_names": list(robot_spec.control_joint_names),
        "body_names": list(robot_spec.body_names),
        "robot_name": robot_spec.name,
        "motion_key": motion.motion_key,
        "source": motion.source or effective_source,
        "metadata": metadata,
    }


def robot_state_dict_to_ufo_motion_dict(
    data: dict[str, RobotStateMotion],
    robot_spec: RobotSpec,
    source_name: str,
) -> dict[str, Any]:
    validated = validate_robot_state_dict(data, robot_spec, source_name)
    converted = {motion_key: robot_state_to_ufo_motion(motion, robot_spec, source_name) for motion_key, motion in validated.items()}
    return validate_ufo_motion_dict(converted, source_name)


def _as_path_list(path_spec: str | os.PathLike[str] | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(path_spec, (list, tuple)):
        return [str(item) for item in path_spec]
    return [str(path_spec)]


def _candidate_patterns(raw_path: str, base_dir: Path | None) -> list[Path]:
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    path = Path(expanded)
    if path.is_absolute():
        return [path]
    candidates: list[Path] = []
    if base_dir is not None:
        candidates.append(base_dir / path)
    candidates.append(Path.cwd() / path)
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def expand_motion_paths(
    path_spec: str | os.PathLike[str] | list[str] | tuple[str, ...],
    *,
    base_dir: Path | None = None,
    suffix: str,
) -> list[Path]:
    """Resolve a path, directory, glob, or list of those into concrete files."""

    resolved: list[Path] = []
    missing_patterns: list[str] = []
    for raw in _as_path_list(path_spec):
        found_for_raw: list[Path] = []
        for candidate in _candidate_patterns(raw, base_dir):
            candidate_str = str(candidate)
            if glob.has_magic(candidate_str):
                found_for_raw = [Path(item) for item in sorted(glob.glob(candidate_str))]
            elif candidate.is_dir():
                found_for_raw = sorted(candidate.glob(f"*{suffix}"))
            elif candidate.exists():
                found_for_raw = [candidate]
            if found_for_raw:
                break
        if not found_for_raw:
            missing_patterns.append(raw)
        resolved.extend(found_for_raw)

    filtered = [path.expanduser().resolve() for path in resolved if path.suffix == suffix]
    if not filtered:
        raise FileNotFoundError(f"No {suffix} motion files matched: {missing_patterns or path_spec}")
    return filtered


def _scalar_from_npz(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return value


def _motion_key_from_npz(value: Any) -> str:
    scalar = _scalar_from_npz(value)
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8")
    return str(scalar)


DEFAULT_ROOT_POS_COLUMNS = ["root_pos_x", "root_pos_y", "root_pos_z"]


DEFAULT_ROOT_QUAT_COLUMNS = ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"]


def _is_numeric_row(row: list[str]) -> bool:
    if not row:
        return False
    try:
        for value in row:
            float(value)
    except ValueError:
        return False
    return True


def _headerless_csv_fieldnames(path: Path, width: int, dof_count: int) -> list[str]:
    dof_columns = [f"dof_{idx}" for idx in range(dof_count)]
    no_time_width = len(DEFAULT_ROOT_POS_COLUMNS) + len(DEFAULT_ROOT_QUAT_COLUMNS) + dof_count
    with_time_width = 1 + no_time_width
    if width == no_time_width:
        return [*DEFAULT_ROOT_POS_COLUMNS, *DEFAULT_ROOT_QUAT_COLUMNS, *dof_columns]
    if width == with_time_width:
        return ["time", *DEFAULT_ROOT_POS_COLUMNS, *DEFAULT_ROOT_QUAT_COLUMNS, *dof_columns]
    raise ValueError(
        f"CSV file={path} has no header and {width} columns; expected {no_time_width} "
        f"columns without time or {with_time_width} columns with a leading time column for {dof_count} DOFs"
    )


def _dict_rows(fieldnames: list[str], raw_rows: list[list[str]], path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    expected = len(fieldnames)
    for line_no, row in enumerate(raw_rows, start=1):
        if len(row) != expected:
            raise ValueError(f"CSV file={path} line {line_no} has {len(row)} columns, expected {expected}")
        rows.append(dict(zip(fieldnames, row)))
    return rows


def _read_csv_rows(path: Path, robot_spec: RobotSpec) -> tuple[list[str], list[dict[str, str]], bool]:
    with path.open("r", newline="") as f:
        raw_rows = [row for row in csv.reader(f) if any(cell.strip() for cell in row)]
    if not raw_rows:
        raise ValueError(f"CSV file is empty: {path}")
    first = raw_rows[0]
    if _is_numeric_row(first):
        fieldnames = _headerless_csv_fieldnames(path, len(first), len(robot_spec.control_joint_names))
        return fieldnames, _dict_rows(fieldnames, raw_rows, path), True
    fieldnames = [str(value) for value in first]
    rows = _dict_rows(fieldnames, raw_rows[1:], path)
    if not rows:
        raise ValueError(f"CSV file is empty: {path}")
    return fieldnames, rows, False


def _column_array(rows: list[dict[str, str]], column: str, path: Path) -> np.ndarray:
    try:
        return np.asarray([float(row[column]) for row in rows], dtype=np.float32)
    except KeyError as exc:
        raise ValueError(f"CSV file={path} missing required column '{column}'") from exc
    except ValueError as exc:
        raise ValueError(f"CSV file={path} column '{column}' contains non-numeric values") from exc


def _infer_fps_from_time(rows: list[dict[str, str]], path: Path) -> float:
    try:
        times = np.asarray([float(row["time"]) for row in rows], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"CSV file={path} missing required column 'time'") from exc
    except ValueError as exc:
        raise ValueError(f"CSV file={path} column 'time' contains non-numeric values") from exc
    if len(times) < 2:
        raise ValueError(f"CSV file={path} needs at least two time samples to infer fps")
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        raise ValueError(f"CSV file={path} time column must be strictly increasing")
    median_dt = float(np.median(dt))
    if median_dt <= 0.0 or not np.isfinite(median_dt):
        raise ValueError(f"CSV file={path} has invalid median dt={median_dt}")
    rel_jitter = float(np.max(np.abs(dt - median_dt)) / median_dt)
    if rel_jitter > 0.05:
        logger.warning(f"CSV file={path} has non-uniform time intervals; max relative jitter={rel_jitter:.3f}")
    return 1.0 / median_dt


def _fps_from_rows(fieldnames: list[str], rows: list[dict[str, str]], path: Path, fps: float | int | None) -> float:
    if fps is not None:
        motion_fps = float(fps)
        if not np.isfinite(motion_fps) or motion_fps <= 0.0:
            raise ValueError(f"CSV file={path} has invalid manifest fps={fps}")
        return motion_fps
    if "time" in fieldnames:
        return _infer_fps_from_time(rows, path)
    raise ValueError(f"CSV file={path} requires a time column or manifest fps")


def _columns_matrix(rows: list[dict[str, str]], columns: list[str], path: Path) -> np.ndarray:
    return np.stack([_column_array(rows, column, path) for column in columns], axis=1).astype(np.float32)


def _robot_state_columns(columns: dict[str, Any] | None) -> tuple[list[str], list[str], Any]:
    config = dict(columns or {})
    root_pos_columns = config.get("root_pos", ["root_pos_x", "root_pos_y", "root_pos_z"])
    root_quat_columns = config.get("root_quat", ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"])
    dof_spec = config.get("dof_pos", "auto_by_joint_name")
    if not isinstance(root_pos_columns, list) or len(root_pos_columns) != 3:
        raise ValueError("robot_state_csv columns.root_pos must be a list of three column names")
    if not isinstance(root_quat_columns, list) or len(root_quat_columns) != 4:
        raise ValueError("robot_state_csv columns.root_quat must be a list of four column names")
    return [str(v) for v in root_pos_columns], [str(v) for v in root_quat_columns], dof_spec


def read_robot_state_csv(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
    columns: dict[str, Any] | None = None,
) -> dict[str, RobotStateMotion]:
    data: dict[str, RobotStateMotion] = {}
    configured_root_pos_columns, configured_root_quat_columns, configured_dof_spec = _robot_state_columns(columns)
    for path in expand_motion_paths(path_spec, base_dir=base_dir, suffix=".csv"):
        fieldnames, rows, headerless = _read_csv_rows(path, robot_spec)
        root_pos_columns = configured_root_pos_columns
        root_quat_columns = configured_root_quat_columns
        dof_spec = "xml_order" if headerless and columns is None else configured_dof_spec
        root_pos = _columns_matrix(rows, root_pos_columns, path)
        root_quat = _columns_matrix(rows, root_quat_columns, path)
        joint_names: list[str] | None = None
        if dof_spec == "auto_by_joint_name":
            missing = [joint for joint in robot_spec.control_joint_names if joint not in fieldnames]
            if missing:
                raise ValueError(f"robot_state_csv file={path} missing control joint columns: {missing}")
            joint_names = list(robot_spec.control_joint_names)
            dof_pos = _columns_matrix(rows, joint_names, path)
        elif dof_spec == "xml_order":
            dof_columns = [f"dof_{idx}" for idx in range(len(robot_spec.control_joint_names))]
            missing = [column for column in dof_columns if column not in fieldnames]
            if missing:
                raise ValueError(f"robot_state_csv file={path} missing xml_order dof columns: {missing}")
            dof_pos = _columns_matrix(rows, dof_columns, path)
        elif isinstance(dof_spec, list):
            if len(dof_spec) != len(robot_spec.control_joint_names):
                raise ValueError(
                    f"robot_state_csv columns.dof_pos list length must match control joints "
                    f"({len(robot_spec.control_joint_names)}), got {len(dof_spec)}"
                )
            dof_pos = _columns_matrix(rows, [str(column) for column in dof_spec], path)
        else:
            raise ValueError("robot_state_csv columns.dof_pos must be auto_by_joint_name, xml_order, or a column list")

        motion_key = path.stem
        if motion_key in data:
            raise ValueError(f"Duplicate motion_key={motion_key} while reading robot_state_csv source={source_name}")
        data[motion_key] = RobotStateMotion(
            motion_key=motion_key,
            root_pos=root_pos,
            root_quat=root_quat,
            dof_pos=dof_pos,
            fps=_fps_from_rows(fieldnames, rows, path, fps),
            joint_names=joint_names,
            source=source_name,
            metadata={
                "path": str(path),
                "reader": "robot_state_csv",
                "columns": {
                    "root_pos": root_pos_columns,
                    "root_quat": root_quat_columns,
                    "dof_pos": dof_spec,
                },
            },
        )
    return validate_robot_state_dict(data, robot_spec, source_name)


def _string_list_from_npz(value: Any) -> list[str]:
    arr = np.asarray(value)
    values: list[str] = []
    for item in arr.reshape(-1):
        if isinstance(item, bytes):
            values.append(item.decode("utf-8"))
        else:
            values.append(str(item))
    return values


def _fps_from_npz(npz: Any, path: Path, fps: float | int | None) -> float:
    if fps is not None:
        motion_fps = float(fps)
    elif "fps" in npz:
        motion_fps = float(_scalar_from_npz(npz["fps"]))
    elif "time" in npz:
        times = np.asarray(npz["time"], dtype=np.float64)
        if times.ndim != 1 or times.shape[0] < 2:
            raise ValueError(f"robot_state_npz file={path} time must have shape [T] with at least two samples")
        dt = np.diff(times)
        if np.any(dt <= 0.0):
            raise ValueError(f"robot_state_npz file={path} time must be strictly increasing")
        motion_fps = 1.0 / float(np.median(dt))
    else:
        raise ValueError(f"robot_state_npz file={path} requires fps, time, or manifest fps")
    if not np.isfinite(motion_fps) or motion_fps <= 0.0:
        raise ValueError(f"robot_state_npz file={path} has invalid fps={motion_fps}")
    return motion_fps


def read_robot_state_npz(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
) -> dict[str, RobotStateMotion]:
    data: dict[str, RobotStateMotion] = {}
    for path in expand_motion_paths(path_spec, base_dir=base_dir, suffix=".npz"):
        with np.load(path, allow_pickle=True) as npz:
            missing = [field for field in ("root_pos", "root_quat", "dof_pos") if field not in npz]
            if missing:
                raise ValueError(f"robot_state_npz source={source_name}, file={path}: missing fields {missing}")
            joint_names = _string_list_from_npz(npz["joint_names"]) if "joint_names" in npz else None
            if joint_names is None:
                logger.warning(
                    f"robot_state_npz file={path} has no joint_names; assuming dof_pos is ordered as RobotSpec.control_joint_names"
                )
            motion_key = _motion_key_from_npz(npz["motion_key"]) if "motion_key" in npz else path.stem
            if motion_key in data:
                raise ValueError(f"Duplicate motion_key={motion_key} while reading robot_state_npz source={source_name}")
            data[motion_key] = RobotStateMotion(
                motion_key=motion_key,
                root_pos=np.asarray(npz["root_pos"], dtype=np.float32),
                root_quat=np.asarray(npz["root_quat"], dtype=np.float32),
                dof_pos=np.asarray(npz["dof_pos"], dtype=np.float32),
                fps=_fps_from_npz(npz, path, fps),
                joint_names=joint_names,
                source=source_name,
                metadata={"path": str(path), "reader": "robot_state_npz"},
            )
    return validate_robot_state_dict(data, robot_spec, source_name)


SUPPORTED_FORMATS = {"ufo_pkl", "robot_state_csv", "robot_state_npz"}


def _merge_motion_dicts(sources: list[tuple[Path, dict[str, Any]]], source_name: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for path, data in sources:
        for motion_key, motion in data.items():
            if motion_key in merged:
                raise ValueError(f"Duplicate motion_key={motion_key} while merging source={source_name} from {path}")
            merged[motion_key] = motion
    return validate_ufo_motion_dict(merged, source_name)


def load_ufo_pkl(path_spec: str | os.PathLike[str] | list[str], *, source_name: str, base_dir: Path | None = None) -> dict[str, Any]:
    sources: list[tuple[Path, dict[str, Any]]] = []
    for path in expand_motion_paths(path_spec, base_dir=base_dir, suffix=".pkl"):
        data = joblib.load(path)
        validated = validate_ufo_motion_dict(data, f"{source_name}:{path.name}")
        sources.append((path, validated))
    return _merge_motion_dicts(sources, source_name)


def _require_robot_spec(fmt: str, robot_spec: RobotSpec | None) -> RobotSpec:
    if robot_spec is None:
        raise ValueError(f"{fmt} requires a robot_config/RobotSpec")
    return robot_spec


def load_robot_state_csv(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
    columns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    robot_state = read_robot_state_csv(
        path_spec,
        source_name=source_name,
        robot_spec=robot_spec,
        base_dir=base_dir,
        fps=fps,
        columns=columns,
    )
    return robot_state_dict_to_ufo_motion_dict(robot_state, robot_spec, source_name)


def load_robot_state_npz(
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    robot_spec: RobotSpec,
    base_dir: Path | None = None,
    fps: float | int | None = None,
) -> dict[str, Any]:
    robot_state = read_robot_state_npz(
        path_spec,
        source_name=source_name,
        robot_spec=robot_spec,
        base_dir=base_dir,
        fps=fps,
    )
    return robot_state_dict_to_ufo_motion_dict(robot_state, robot_spec, source_name)


def load_motion_data_by_format(
    fmt: str,
    path_spec: str | os.PathLike[str] | list[str],
    *,
    source_name: str,
    base_dir: Path | None = None,
    fps: float | int | None = None,
    robot_spec: RobotSpec | None = None,
    columns: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fmt = str(fmt)
    if fmt == "ufo_pkl":
        return load_ufo_pkl(path_spec, source_name=source_name, base_dir=base_dir)
    if fmt == "robot_state_csv":
        return load_robot_state_csv(
            path_spec,
            source_name=source_name,
            robot_spec=_require_robot_spec(fmt, robot_spec),
            base_dir=base_dir,
            fps=fps,
            columns=columns,
        )
    if fmt == "robot_state_npz":
        return load_robot_state_npz(
            path_spec,
            source_name=source_name,
            robot_spec=_require_robot_spec(fmt, robot_spec),
            base_dir=base_dir,
            fps=fps,
        )
    raise ValueError(f"Unsupported motion data format '{fmt}'. Supported formats: {sorted(SUPPORTED_FORMATS)}")


def dump_ufo_pkl(data: dict[str, Any], output_path: Path, source_name: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_ufo_motion_dict(data, source_name)
    joblib.dump(validated, output_path)
    return output_path
