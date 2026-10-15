"""Inspect RobotState CSV/NPZ data and build UFO motion manifests and caches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ufo.motion.formats import (
    _read_csv_rows,
    expand_motion_paths,
    motion_clip_windows,
    read_robot_state_csv,
    read_robot_state_npz,
)
from ufo.motion.manifest import prepare_motion_manifest
from ufo.tools.robot_config import RobotSpec, load_robot_spec

ROOT_POS_ALIASES = [
    ["root_pos_x", "root_pos_y", "root_pos_z"],
    ["root_x", "root_y", "root_z"],
    ["base_pos_x", "base_pos_y", "base_pos_z"],
    ["base_x", "base_y", "base_z"],
    ["pelvis_pos_x", "pelvis_pos_y", "pelvis_pos_z"],
]

ROOT_QUAT_ALIASES = [
    ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"],
    ["root_qx", "root_qy", "root_qz", "root_qw"],
    ["base_quat_x", "base_quat_y", "base_quat_z", "base_quat_w"],
    ["base_qx", "base_qy", "base_qz", "base_qw"],
    ["pelvis_quat_x", "pelvis_quat_y", "pelvis_quat_z", "pelvis_quat_w"],
]


@dataclass(frozen=True)
class FileInspection:
    path: Path
    frames: int
    fps: float
    duration_seconds: float
    estimated_clip_count: int
    root_pos_columns: list[str] | None = None
    root_quat_columns: list[str] | None = None
    dof_pos_mode: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DataInspection:
    format: str
    source_paths: list[Path]
    files: list[FileInspection]
    root_pos_columns: list[str] | None
    root_quat_columns: list[str] | None
    dof_pos_mode: str | None
    manifest_columns: dict[str, Any]
    suggested_manifest: dict[str, Any]

    @property
    def fps_values(self) -> list[float]:
        return [item.fps for item in self.files]

    @property
    def duration_range(self) -> tuple[float, float]:
        durations = [item.duration_seconds for item in self.files]
        return min(durations), max(durations)

    @property
    def estimated_clip_count(self) -> int:
        return sum(item.estimated_clip_count for item in self.files)


def _read_header_and_rows(path: Path, robot_spec: RobotSpec) -> tuple[list[str], list[dict[str, str]], list[str]]:
    fieldnames, rows, headerless = _read_csv_rows(path, robot_spec)
    notes = []
    if headerless:
        notes.append("No CSV header detected; interpreting columns as "
                     + ("time, " if fieldnames[0] == "time" else "")
                     + "root_pos xyz, root_quat xyzw, and dof_0..dof_N in RobotSpec control joint order.")
    return fieldnames, rows, notes


def _first_alias(fieldnames: list[str], aliases: list[list[str]]) -> list[str] | None:
    names = set(fieldnames)
    for candidate in aliases:
        if all(column in names for column in candidate):
            return candidate
    return None


def estimate_clip_count(
    frames: int,
    fps: float,
    *,
    clip_seconds: float = 10.0,
    stride_seconds: float | None = None,
    keep_short: bool = True,
    min_clip_seconds: float = 1.0,
) -> int:
    return len(motion_clip_windows(
        frames, fps, clip_seconds=clip_seconds, stride_seconds=clip_seconds if stride_seconds is None else stride_seconds,
        keep_short=keep_short, min_clip_seconds=min_clip_seconds,
    ))


def detect_csv_columns(fieldnames: list[str], robot_spec: RobotSpec) -> tuple[dict[str, Any], list[str], list[str]]:
    missing: list[str] = []
    notes: list[str] = []
    root_pos = _first_alias(fieldnames, ROOT_POS_ALIASES)
    root_quat = _first_alias(fieldnames, ROOT_QUAT_ALIASES)
    if root_pos is None:
        missing.append("root_pos aliases: " + " or ".join("/".join(alias) for alias in ROOT_POS_ALIASES))
    if root_quat is None:
        missing.append("root_quat aliases: " + " or ".join("/".join(alias) for alias in ROOT_QUAT_ALIASES))

    names = set(fieldnames)
    missing_named = [joint for joint in robot_spec.control_joint_names if joint not in names]
    dof_columns = [f"dof_{idx}" for idx in range(len(robot_spec.control_joint_names))]
    missing_dof = [column for column in dof_columns if column not in names]

    has_named = not missing_named
    has_xml_order = not missing_dof
    if has_named:
        dof_mode = "auto_by_joint_name"
        if has_xml_order:
            notes.append("Both named joint columns and dof_0..dof_N columns exist; using auto_by_joint_name.")
    elif has_xml_order:
        dof_mode = "xml_order"
    else:
        dof_mode = None
        missing.append("missing named joints: " + ", ".join(missing_named))
        missing.append("missing xml_order dof columns: " + ", ".join(missing_dof))

    columns: dict[str, Any] = {}
    if root_pos is not None:
        columns["root_pos"] = root_pos
    if root_quat is not None:
        columns["root_quat"] = root_quat
    if dof_mode is not None:
        columns["dof_pos"] = dof_mode
    return columns, missing, notes


def _inspect_csv_file(
    path: Path,
    robot_spec: RobotSpec,
    *,
    fps: float | None,
    clip_seconds: float,
    stride_seconds: float | None,
    keep_short: bool,
    min_clip_seconds: float,
) -> FileInspection:
    fieldnames, rows, reader_notes = _read_header_and_rows(path, robot_spec)
    columns, missing, notes = detect_csv_columns(fieldnames, robot_spec)
    if missing:
        raise ValueError(f"Motion source inspection failed for {path}: " + "; ".join(missing))
    del rows
    notes = [*reader_notes, *notes]
    motion = next(iter(read_robot_state_csv(
        path, source_name=str(path), robot_spec=robot_spec, fps=fps, columns=columns,
    ).values()))
    motion_fps = motion.fps
    frames = len(motion.root_pos)
    return FileInspection(
        path=path,
        frames=frames,
        fps=motion_fps,
        duration_seconds=frames / motion_fps,
        estimated_clip_count=estimate_clip_count(
            frames,
            motion_fps,
            clip_seconds=clip_seconds,
            stride_seconds=stride_seconds,
            keep_short=keep_short,
            min_clip_seconds=min_clip_seconds,
        ),
        root_pos_columns=columns.get("root_pos"),
        root_quat_columns=columns.get("root_quat"),
        dof_pos_mode=columns.get("dof_pos"),
        notes=notes,
    )



def _inspect_npz_file(
    path: Path,
    robot_spec: RobotSpec,
    *,
    fps: float | None,
    clip_seconds: float,
    stride_seconds: float | None,
    keep_short: bool,
    min_clip_seconds: float,
) -> FileInspection:
    motion = next(iter(read_robot_state_npz(path, source_name=str(path), robot_spec=robot_spec, fps=fps).values()))
    notes = [
        "joint_names found; dof_pos will be reordered into RobotSpec.control_joint_names."
        if motion.joint_names is not None else
        "No joint_names found; assuming dof_pos already follows RobotSpec.control_joint_names."
    ]
    motion_fps = motion.fps
    frames = len(motion.root_pos)
    return FileInspection(
        path=path,
        frames=frames,
        fps=motion_fps,
        duration_seconds=frames / motion_fps,
        estimated_clip_count=estimate_clip_count(
            frames,
            motion_fps,
            clip_seconds=clip_seconds,
            stride_seconds=stride_seconds,
            keep_short=keep_short,
            min_clip_seconds=min_clip_seconds,
        ),
        dof_pos_mode="npz",
        notes=notes,
    )


def _infer_format(source: str, fmt: str) -> str:
    if fmt != "auto":
        return fmt
    suffix = Path(source).suffix.lower()
    if suffix == ".csv":
        return "robot_state_csv"
    if suffix == ".npz":
        return "robot_state_npz"
    raise ValueError(f"Cannot infer format from source={source!r}; pass --format robot_state_csv or robot_state_npz")


def inspect_data_source(
    *,
    robot_config: str | Path,
    source: str | list[str],
    fmt: str = "auto",
    fps: float | None = None,
    clip_seconds: float = 10.0,
    stride_seconds: float | None = None,
    keep_short: bool = True,
    min_clip_seconds: float = 1.0,
    dataset_name: str = "dataset",
    weight: float = 1.0,
) -> DataInspection:
    robot_spec = load_robot_spec(robot_config)
    source_for_format = source[0] if isinstance(source, list) else source
    resolved_format = _infer_format(str(source_for_format), fmt)
    suffix = ".csv" if resolved_format == "robot_state_csv" else ".npz" if resolved_format == "robot_state_npz" else ""
    if not suffix:
        raise ValueError(f"Data inspection supports robot_state_csv and robot_state_npz, got {resolved_format}")
    paths = expand_motion_paths(source, suffix=suffix)

    files: list[FileInspection] = []
    manifest_columns: dict[str, Any] = {}
    if resolved_format == "robot_state_csv":
        for path in paths:
            files.append(
                _inspect_csv_file(
                    path,
                    robot_spec,
                    fps=fps,
                    clip_seconds=clip_seconds,
                    stride_seconds=stride_seconds,
                    keep_short=keep_short,
                    min_clip_seconds=min_clip_seconds,
                )
            )
        first = files[0]
        if first.root_pos_columns is not None:
            manifest_columns["root_pos"] = first.root_pos_columns
        if first.root_quat_columns is not None:
            manifest_columns["root_quat"] = first.root_quat_columns
        if first.dof_pos_mode is not None:
            manifest_columns["dof_pos"] = first.dof_pos_mode
    else:
        for path in paths:
            files.append(
                _inspect_npz_file(
                    path,
                    robot_spec,
                    fps=fps,
                    clip_seconds=clip_seconds,
                    stride_seconds=stride_seconds,
                    keep_short=keep_short,
                    min_clip_seconds=min_clip_seconds,
                )
            )

    source_path: str | list[str] = source
    suggested_dataset: dict[str, Any] = {
        "name": dataset_name,
        "format": resolved_format,
        "source_path": source_path,
        "weight": float(weight),
        "auto_build": {
            "train_clip_seconds": float(clip_seconds),
            "clip_stride_seconds": float(stride_seconds if stride_seconds is not None else clip_seconds),
            "keep_short": bool(keep_short),
            "min_clip_seconds": float(min_clip_seconds),
        },
    }
    if fps is not None:
        suggested_dataset["fps"] = float(fps)
    if manifest_columns:
        suggested_dataset["columns"] = manifest_columns
    suggested = {"robot_config": str(robot_config), "datasets": [suggested_dataset]}
    return DataInspection(
        format=resolved_format,
        source_paths=paths,
        files=files,
        root_pos_columns=manifest_columns.get("root_pos"),
        root_quat_columns=manifest_columns.get("root_quat"),
        dof_pos_mode=manifest_columns.get("dof_pos") or ("npz" if resolved_format == "robot_state_npz" else None),
        manifest_columns=manifest_columns,
        suggested_manifest=suggested,
    )


def _print_report(result: DataInspection) -> None:
    duration_min, duration_max = result.duration_range
    fps_values = result.fps_values
    print("Detected RobotState schema")
    print(f"  reader: {result.format}")
    print(f"  files: {len(result.source_paths)}")
    print(f"  fps range: {min(fps_values):.6g} - {max(fps_values):.6g}")
    print(f"  duration range: {duration_min:.3f}s - {duration_max:.3f}s")
    print(f"  estimated training clips: {result.estimated_clip_count}")
    if result.root_pos_columns:
        print(f"  root_pos columns: {result.root_pos_columns}")
    if result.root_quat_columns:
        print(f"  root_quat columns: {result.root_quat_columns}")
    if result.dof_pos_mode:
        print(f"  dof_pos mode: {result.dof_pos_mode}")
    notes = [note for item in result.files for note in item.notes]
    for note in sorted(set(notes)):
        print(f"  note: {note}")
    print("Suggested manifest snippet:")
    print(OmegaConf.to_yaml(OmegaConf.create(result.suggested_manifest)).rstrip())


def write_manifest(config: dict[str, Any], out_path: str | Path, *, force: bool = False) -> Path:
    path = Path(out_path).expanduser()
    if path.exists() and not force:
        raise FileExistsError(f"Data manifest already exists: {path}. Use --force to overwrite it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(config), path)
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--robot", required=True, help="Robot YAML path")
    common.add_argument("--source", required=True, nargs="+", help="CSV/NPZ source path, directory, or glob")
    common.add_argument("--format", default="auto", choices=["auto", "robot_state_csv", "robot_state_npz"])
    common.add_argument("--fps", type=float, default=None)
    common.add_argument("--clip-seconds", type=float, default=10.0)
    common.add_argument("--stride-seconds", type=float, default=None)
    common.add_argument("--keep-short", dest="keep_short", action="store_true", default=True)
    common.add_argument("--drop-short", dest="keep_short", action="store_false")
    common.add_argument("--min-clip-seconds", type=float, default=1.0)
    commands.add_parser("inspect", parents=[common], help="Inspect data and print a suggested manifest")
    build = commands.add_parser("build", parents=[common], help="Write a manifest and build motion caches")
    build.add_argument("--name", required=True, help="Dataset name")
    build.add_argument("--weight", type=float, default=1.0)
    build.add_argument("--out", required=True, help="Output data manifest path")
    build.add_argument("--rebuild-cache", action="store_true", help="Force cache rebuild when building")
    build.add_argument("--no-build", action="store_true", help="Only write the manifest")
    build.add_argument("--force", action="store_true", help="Overwrite existing manifest")
    args = parser.parse_args(argv)

    options = {"dataset_name": args.name, "weight": args.weight} if args.command == "build" else {}
    result = inspect_data_source(
        robot_config=args.robot,
        source=args.source[0] if len(args.source) == 1 else args.source,
        fmt=args.format,
        fps=args.fps,
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        keep_short=args.keep_short,
        min_clip_seconds=args.min_clip_seconds,
        **options,
    )
    if args.command == "inspect":
        _print_report(result)
        return
    print(f"Detected RobotState schema via {result.format} reader")
    manifest_path = write_manifest(result.suggested_manifest, args.out, force=args.force)
    print(f"Wrote data manifest: {manifest_path}")
    if args.no_build:
        return
    dataset = prepare_motion_manifest(manifest_path, rebuild_cache=args.rebuild_cache)
    print(f"manifest path: {dataset.manifest_path}")
    print(f"train_data_paths: {dataset.train_data_paths}")
    print(f"train_data_weights: {dataset.train_data_weights}")
    print(f"inference_paths: {dataset.inference_paths}")


if __name__ == "__main__":
    main()
