from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pytest
from omegaconf import OmegaConf

from ufo.motion.formats import read_robot_state_csv, read_robot_state_npz
from ufo.motion.manifest import prepare_motion_manifest
from ufo.tools.motion_data import inspect_data_source, write_manifest
from ufo.tools.robot_config import load_robot_spec
from ufo.tools.robot_inspect import infer_robot_semantics, inspect_mujoco_xml, write_robot_yaml


@pytest.mark.parametrize("fmt", ["csv", "npz"])
@pytest.mark.parametrize("fps", [None, 100])
def test_inspection_and_built_cache_agree_on_fps_and_clip_count(tmp_path, fmt, fps):
    xml = _write_import_robot(tmp_path)
    robot = tmp_path / "robot.yaml"
    info = inspect_mujoco_xml(xml)
    write_robot_yaml(robot, info, infer_robot_semantics(info))
    source = tmp_path / f"motion.{fmt}"
    if fmt == "csv":
        _write_robot_state_csv(source, named_joints=True, frames=1250)
    else:
        _write_robot_state_npz(source, frames=1250)
    inspection = inspect_data_source(robot_config=robot, source=str(source), fps=fps, dataset_name="motion")
    expected_fps = 50 if fps is None else fps
    assert inspection.fps_values == pytest.approx([expected_fps])
    assert inspection.duration_range == pytest.approx((1250 / expected_fps, 1250 / expected_fps))
    manifest = write_manifest(inspection.suggested_manifest, tmp_path / "motion.yaml")
    built = prepare_motion_manifest(manifest, cache_root=tmp_path / "cache")
    full = joblib.load(built.inference_paths["motion"])
    clips = joblib.load(built.train_data_paths[0])
    assert all(motion["fps"] == inspection.fps_values[0] for motion in full.values())
    assert len(clips) == inspection.estimated_clip_count == (3 if fps is None else 2)


@pytest.mark.parametrize("fmt", ["csv", "npz"])
@pytest.mark.parametrize("fps", [0, -1, float("nan")])
def test_inspection_rejects_invalid_explicit_fps_even_with_source_timing(tmp_path, fmt, fps):
    xml = _write_import_robot(tmp_path)
    robot = tmp_path / "robot.yaml"
    info = inspect_mujoco_xml(xml)
    write_robot_yaml(robot, info, infer_robot_semantics(info))
    source = tmp_path / f"motion.{fmt}"
    if fmt == "csv":
        _write_robot_state_csv(source, named_joints=True, frames=3)
    else:
        _write_robot_state_npz(source, frames=3)
    with pytest.raises(ValueError, match="invalid"):
        inspect_data_source(robot_config=robot, source=str(source), fps=fps)


@pytest.mark.parametrize("fmt,field,value", [
    ("csv", "root_pos_x", "nan"),
    ("csv", "root_quat_w", "0"),
    ("csv", "joint1", "invalid"),
    ("npz", "root_pos", np.full((3, 3), np.nan)),
    ("npz", "root_pos", np.array(0.0)),
    ("npz", "root_quat", np.zeros((3, 4))),
    ("npz", "dof_pos", np.full((3, 2), np.inf)),
    ("npz", "dof_pos", np.zeros((2, 2))),
    ("npz", "joint_names", np.array(["joint1", "joint1"])),
])
def test_inspection_rejects_the_same_invalid_motion_as_loading(tmp_path, fmt, field, value):
    xml = _write_import_robot(tmp_path)
    robot = tmp_path / "robot.yaml"
    info = inspect_mujoco_xml(xml)
    write_robot_yaml(robot, info, infer_robot_semantics(info))
    source = tmp_path / f"motion.{fmt}"
    if fmt == "csv":
        _write_robot_state_csv(source, named_joints=True, frames=3)
        with source.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fieldnames, rows = reader.fieldnames, list(reader)
        rows[0][field] = value
        with source.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        load = read_robot_state_csv
    else:
        _write_robot_state_npz(source, frames=3)
        with np.load(source) as data:
            arrays = {key: data[key] for key in data.files}
        arrays[field] = value
        np.savez(source, **arrays)
        load = read_robot_state_npz
    with pytest.raises(ValueError) as loaded:
        load(source, source_name=str(source), robot_spec=load_robot_spec(robot))
    with pytest.raises(ValueError) as inspected:
        inspect_data_source(robot_config=robot, source=str(source))
    assert str(inspected.value) == str(loaded.value)


def _write_import_robot(root: Path) -> Path:
    xml_path = root / "tiny_import.xml"
    xml_path.write_text(
        """
<mujoco model="tiny_import">
  <worldbody>
    <body name="base" pos="0 0 1">
      <freejoint name="root"/>
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="torso" pos="0 0 0.1">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 0.2" mass="0.1"/>
        <body name="left_foot" pos="0 0.1 -0.2">
          <joint name="joint2" type="hinge" axis="0 1 0" range="-2 2"/>
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
        <body name="right_foot" pos="0 -0.1 -0.2">
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
        <body name="left_hand" pos="0 0.2 0.1">
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
        <body name="right_hand" pos="0 -0.2 0.1">
          <geom type="sphere" size="0.03" mass="0.1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="joint1_motor" joint="joint1"/>
    <motor name="joint2_motor" joint="joint2"/>
  </actuator>
</mujoco>
""".strip()
    )
    return xml_path


def _write_robot_state_csv(path: Path, *, named_joints: bool, frames: int) -> None:
    if named_joints:
        fieldnames = [
            "time",
            "root_pos_x",
            "root_pos_y",
            "root_pos_z",
            "root_quat_x",
            "root_quat_y",
            "root_quat_z",
            "root_quat_w",
            "joint1",
            "joint2",
        ]
    else:
        fieldnames = [
            "time",
            "root_pos_x",
            "root_pos_y",
            "root_pos_z",
            "root_quat_x",
            "root_quat_y",
            "root_quat_z",
            "root_quat_w",
            "dof_0",
            "dof_1",
        ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(frames):
            row = {
                "time": idx * 0.02,
                "root_pos_x": 0.0,
                "root_pos_y": 0.0,
                "root_pos_z": 1.0,
                "root_quat_x": 0.0,
                "root_quat_y": 0.0,
                "root_quat_z": 0.0,
                "root_quat_w": 1.0,
            }
            if named_joints:
                row.update({"joint1": 0.1 * idx, "joint2": 0.2 * idx})
            else:
                row.update({"dof_0": 0.1 * idx, "dof_1": 0.2 * idx})
            writer.writerow(row)


def _write_robot_state_npz(path: Path, *, frames: int) -> None:
    idx = np.arange(frames, dtype=np.float32)
    np.savez(
        path,
        root_pos=np.zeros((frames, 3), dtype=np.float32),
        root_quat=np.tile(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (frames, 1)),
        dof_pos=np.stack([0.1 * idx, 0.2 * idx], axis=1).astype(np.float32),
        joint_names=np.asarray(["joint1", "joint2"]),
        fps=np.asarray(50),
    )


class ImportToolsTest(unittest.TestCase):
    def test_robot_inspect_writes_valid_robot_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_import_robot(root)
            out_path = root / "tiny_import.yaml"

            inspection = inspect_mujoco_xml(xml_path, name="tiny_import")
            semantics = infer_robot_semantics(inspection)
            write_robot_yaml(out_path, inspection, semantics)
            spec = load_robot_spec(out_path)

            self.assertEqual(spec.base_body, "base")
            self.assertEqual(spec.free_joint, "root")
            self.assertEqual(spec.control_joint_names, ["joint1", "joint2"])
            self.assertIn("base", spec.key_bodies)
            self.assertIn("torso", spec.key_bodies)
            self.assertIn("left_foot", spec.feet)
            self.assertIn("right_foot", spec.feet)

    def test_data_inspect_csv_xml_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_import_robot(root)
            robot_yaml = root / "tiny_import.yaml"
            inspection = inspect_mujoco_xml(xml_path, name="tiny_import")
            write_robot_yaml(robot_yaml, inspection, infer_robot_semantics(inspection))
            csv_path = root / "state_xml_order.csv"
            _write_robot_state_csv(csv_path, named_joints=False, frames=1250)

            result = inspect_data_source(
                robot_config=robot_yaml,
                source=str(csv_path),
                fmt="robot_state_csv",
                fps=50,
                clip_seconds=10,
            )
            self.assertEqual(result.root_pos_columns, ["root_pos_x", "root_pos_y", "root_pos_z"])
            self.assertEqual(result.root_quat_columns, ["root_quat_x", "root_quat_y", "root_quat_z", "root_quat_w"])
            self.assertEqual(result.dof_pos_mode, "xml_order")
            self.assertEqual(result.estimated_clip_count, 3)

    def test_data_inspect_named_joint_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_import_robot(root)
            robot_yaml = root / "tiny_import.yaml"
            inspection = inspect_mujoco_xml(xml_path, name="tiny_import")
            write_robot_yaml(robot_yaml, inspection, infer_robot_semantics(inspection))
            csv_path = root / "state_named.csv"
            _write_robot_state_csv(csv_path, named_joints=True, frames=60)

            result = inspect_data_source(robot_config=robot_yaml, source=str(csv_path), fmt="robot_state_csv")
            self.assertEqual(result.dof_pos_mode, "auto_by_joint_name")
            self.assertEqual(result.manifest_columns["dof_pos"], "auto_by_joint_name")

    def test_data_build_manifest_builds_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_import_robot(root)
            robot_yaml = root / "tiny_import.yaml"
            inspection = inspect_mujoco_xml(xml_path, name="tiny_import")
            write_robot_yaml(robot_yaml, inspection, infer_robot_semantics(inspection))
            csv_path = root / "state.csv"
            _write_robot_state_csv(csv_path, named_joints=False, frames=60)

            inspection = inspect_data_source(
                robot_config=str(robot_yaml),
                source=str(csv_path),
                fmt="robot_state_csv",
                dataset_name="tiny_motion",
                weight=1.0,
                fps=50,
                clip_seconds=10,
            )
            manifest_path = write_manifest(inspection.suggested_manifest, root / "tiny_motion.yaml")
            result = prepare_motion_manifest(manifest_path, cache_root=root / "cache", rebuild_cache=True)

            self.assertEqual(len(result.train_data_paths), 1)
            train_path = Path(result.train_data_paths[0])
            full_path = Path(result.inference_paths["tiny_motion"])
            self.assertTrue(train_path.exists())
            self.assertTrue(full_path.exists())
            self.assertTrue(train_path.name.endswith("_train_ufo.pkl"))
            self.assertTrue(full_path.name.endswith("_full_ufo.pkl"))

    def test_data_build_manifest_builds_npz_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml_path = _write_import_robot(root)
            robot_yaml = root / "tiny_import.yaml"
            inspection = inspect_mujoco_xml(xml_path, name="tiny_import")
            write_robot_yaml(robot_yaml, inspection, infer_robot_semantics(inspection))
            npz_path = root / "state.npz"
            _write_robot_state_npz(npz_path, frames=60)

            inspection = inspect_data_source(
                robot_config=str(robot_yaml),
                source=str(npz_path),
                fmt="robot_state_npz",
                dataset_name="tiny_motion_npz",
                weight=1.0,
                fps=50,
                clip_seconds=10,
            )
            manifest_path = write_manifest(inspection.suggested_manifest, root / "tiny_motion_npz.yaml")
            result = prepare_motion_manifest(manifest_path, cache_root=root / "cache", rebuild_cache=True)

            self.assertEqual(len(result.train_data_paths), 1)
            self.assertTrue(Path(result.train_data_paths[0]).exists())
            self.assertTrue(Path(result.inference_paths["tiny_motion_npz"]).exists())


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("fmt", ["csv", "npz"])
def test_data_cli_inspect_build_and_overwrite_protection(tmp_path, capsys, monkeypatch, fmt):
    from ufo.tools import motion_data as data

    xml = _write_import_robot(tmp_path)
    robot = tmp_path / "robot.yaml"
    inspection = inspect_mujoco_xml(xml)
    write_robot_yaml(robot, inspection, infer_robot_semantics(inspection))
    source = tmp_path / f"motion.{fmt}"
    if fmt == "csv":
        _write_robot_state_csv(source, named_joints=True, frames=60)
    else:
        _write_robot_state_npz(source, frames=60)
    common = ["--robot", str(robot), "--source", str(source), "--clip-seconds", "0.5",
              "--stride-seconds", "0.25", "--min-clip-seconds", "0.1", "--drop-short"]
    before = set(tmp_path.iterdir())
    data.main(["inspect", *common])
    assert "Suggested manifest snippet" in capsys.readouterr().out
    assert set(tmp_path.iterdir()) == before

    built = []
    prepare = data.prepare_motion_manifest

    def build(path, **kwargs):
        result = prepare(path, cache_root=tmp_path / "cache", **kwargs)
        built.append((result, kwargs))
        return result

    monkeypatch.setattr(data, "prepare_motion_manifest", build)
    output = tmp_path / "motions.yaml"
    command = ["build", *common, "--name", "motion", "--weight", "2", "--out", str(output)]
    data.main([*command, "--no-build"])
    assert not built and not (tmp_path / "cache").exists()
    manifest = OmegaConf.load(output)
    dataset = manifest.datasets[0]
    assert dataset.weight == 2 and dataset.auto_build.keep_short is False
    assert dataset.auto_build.clip_stride_seconds == 0.25
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        data.main(command)
    assert output.read_bytes() == original and not built
    data.main([*command, "--force", "--rebuild-cache"])
    assert built[0][1] == {"rebuild_cache": True}
    assert Path(built[0][0].train_data_paths[0]).exists()
    assert Path(built[0][0].inference_paths["motion"]).exists()
    with pytest.raises(SystemExit):
        data.main(["build", *common])
