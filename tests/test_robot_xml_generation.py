from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import pytest
from omegaconf import OmegaConf

from ufo.tools.robot_config import load_robot_spec, load_robot_training_spec
from ufo.tools.robot_inspect import infer_control_joints_from_xml, infer_robot_semantics, inspect_mujoco_xml, write_robot_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_xml_effort_limits_and_fallback_warnings(tmp_path):
    from ufo.tools.robot_inspect import infer_actuator_limits_from_xml

    model = mujoco.MjModel.from_xml_path(str(_write_xml(tmp_path)))
    joints, _ = infer_control_joints_from_xml(model)
    efforts, velocities, actuators, warnings = infer_actuator_limits_from_xml(model, joints, effort_limit_default=17.0)
    assert efforts == [120.0, 90.0, 17.0, 120.0, 90.0, 17.0]
    assert velocities == [20.0] * 6
    assert actuators["left_hip_joint"]["armature"] == .02
    assert actuators["left_hip_joint"]["friction"] == .001
    assert sum("using ctrlrange" in warning for warning in warnings) == 2
    assert sum("effort_limit_default=17.0" in warning for warning in warnings) == 2


@pytest.mark.parametrize("feet", [None, "right_ankle_roll_link,left_ankle_roll_link", ""])
def test_training_draft_uses_the_same_semantics_as_inspection(tmp_path, feet):
    from ufo.tools.robot_inspect import main

    xml = _write_xml(tmp_path)
    out = tmp_path / "robot.yaml"
    options = ["--xml", str(xml), "--name", "test", "--out", str(out), "--with-training"]
    override = None if feet is None else [value for value in feet.split(",") if value]
    if feet is not None:
        options += ["--feet", feet]
    expected = infer_robot_semantics(inspect_mujoco_xml(xml), feet=override)
    main(options)
    result = OmegaConf.to_container(OmegaConf.load(out), resolve=True)
    assert result["feet"] == expected["feet"]
    assert result["training"]["semantics"]["contact_bodies"] == expected["feet"]
    assert result["hands"] == expected["hands"]
    assert result["key_bodies"] == expected["key_bodies"]
    assert result["base_body"] == expected["base_body"]


def _write_xml(root: Path, *, actuators: bool = True, freejoint: bool = True) -> Path:
    freejoint_xml = '<freejoint name="root"/>' if freejoint else ""
    qpos = "0 0 0.9 1 0 0 0 0.1 -0.2 0.05 0.1 -0.2 0.05" if freejoint else "0.1 -0.2 0.05 0.1 -0.2 0.05"
    actuator_xml = (
        """
  <actuator>
    <motor name="left_hip_motor" joint="left_hip_joint" forcerange="-120 120"/>
    <motor name="left_knee_motor" joint="left_knee_joint" ctrlrange="-90 90"/>
    <motor name="left_ankle_motor" joint="left_ankle_joint"/>
    <motor name="right_hip_motor" joint="right_hip_joint" forcerange="-120 120"/>
    <motor name="right_knee_motor" joint="right_knee_joint" ctrlrange="-90 90"/>
    <motor name="right_ankle_motor" joint="right_ankle_joint"/>
  </actuator>
"""
        if actuators
        else ""
    )
    xml = root / "tiny_hydra.xml"
    xml.write_text(
        f"""
<mujoco model="tiny_hydra">
  <compiler autolimits="true"/>
  <worldbody>
    <body name="pelvis" pos="0 0 0.8">
      {freejoint_xml}
      <geom type="sphere" size="0.05" mass="1"/>
      <body name="torso_link" pos="0 0 0.2">
        <geom type="sphere" size="0.05" mass="0.3"/>
      </body>
      <body name="left_hip_link" pos="0 0.1 -0.1">
        <joint name="left_hip_joint" type="hinge" axis="0 1 0" range="-1 1" armature="0.02" frictionloss="0.001"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
        <body name="left_knee_link" pos="0 0 -0.2">
          <joint name="left_knee_joint" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
          <body name="left_ankle_roll_link" pos="0 0 -0.2">
            <joint name="left_ankle_joint" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.03" mass="0.1"/>
          </body>
        </body>
      </body>
      <body name="right_hip_link" pos="0 -0.1 -0.1">
        <joint name="right_hip_joint" type="hinge" axis="0 1 0" range="-1 1"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
        <body name="right_knee_link" pos="0 0 -0.2">
          <joint name="right_knee_joint" type="hinge" axis="0 1 0" range="-2 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.2" mass="0.1"/>
          <body name="right_ankle_roll_link" pos="0 0 -0.2">
            <joint name="right_ankle_joint" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.03" mass="0.1"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key name="stand" qpos="{qpos}"/>
  </keyframe>
  {actuator_xml}
</mujoco>
""".strip()
    )
    return xml


def _run_robot_inspect(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ufo.tools.robot_inspect", *args],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


class RobotXmlGenerationTest(unittest.TestCase):
    def test_legacy_robot_inspect_behavior_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml = _write_xml(root)
            out = root / "legacy.yaml"
            inspection = inspect_mujoco_xml(xml, name="tiny_hydra")
            write_robot_yaml(out, inspection, infer_robot_semantics(inspection))
            config = OmegaConf.to_container(OmegaConf.load(out), resolve=True)
            self.assertNotIn("training", config)
            spec = load_robot_spec(out)
            self.assertEqual(spec.name, "tiny_hydra")

    def test_with_training_generates_loadable_native_draft(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml = _write_xml(root)
            robot_yaml = root / "tiny_hydra.yaml"
            result = _run_robot_inspect(
                [
                    "--xml",
                    str(xml),
                    "--name",
                    "tiny_hydra",
                    "--out",
                    str(robot_yaml),
                    "--with-training",
                    "--force",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertTrue(robot_yaml.exists())
            robot_config = OmegaConf.to_container(OmegaConf.load(robot_yaml), resolve=True)
            spec = load_robot_training_spec(robot_yaml)
            model = mujoco.MjModel.from_xml_path(str(xml))
            actuator_joints, _warnings = infer_control_joints_from_xml(model)
            self.assertEqual(len(spec.robot.control_joint_names), len(actuator_joints))
            self.assertEqual(len(spec.effort_limits), len(spec.robot.control_joint_names))
            self.assertEqual(len(spec.velocity_limits), len(spec.robot.control_joint_names))
            self.assertEqual(set(spec.default_joint_angles), set(spec.robot.control_joint_names))
            self.assertEqual(spec.actuator["source"], "yaml")
            for joint in spec.robot.control_joint_names:
                params = spec.actuator["joints"][joint]
                for key in ("effort_limit", "velocity_limit", "armature", "friction"):
                    self.assertIn(key, params)
            self.assertEqual(robot_config["metadata"]["review_status"], "draft")
            self.assertTrue(robot_config["metadata"]["warnings"])

    def test_missing_actuator_has_clear_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml = _write_xml(root, actuators=False)
            result = _run_robot_inspect(
                [
                    "--xml",
                    str(xml),
                    "--name",
                    "tiny_hydra",
                    "--out",
                    str(root / "tiny_hydra.yaml"),
                    "--with-training",
                ]
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("no joint-transmission actuators", result.stderr + result.stdout)

    def test_missing_freejoint_records_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            xml = _write_xml(root, freejoint=False)
            robot_yaml = root / "tiny_no_freejoint.yaml"
            result = _run_robot_inspect(
                [
                    "--xml",
                    str(xml),
                    "--name",
                    "tiny_no_freejoint",
                    "--out",
                    str(robot_yaml),
                    "--with-training",
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            robot_config = OmegaConf.to_container(OmegaConf.load(robot_yaml), resolve=True)
            self.assertTrue(any("No freejoint" in item for item in robot_config["metadata"]["warnings"]))

if __name__ == "__main__":
    unittest.main()


def test_robot_inspection_keeps_raw_actuators_and_training_warnings(tmp_path):
    from ufo.tools.robot_inspect import infer_freejoint_base_body

    xml = tmp_path / "actuator_order.xml"
    xml.write_text("""<mujoco>
      <worldbody><body name="base"><freejoint name="root"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <body name="link"><joint name="hinge"/>
          <geom type="sphere" size="0.1" mass="1"/><site name="tip"/>
        </body></body></worldbody>
      <actuator><motor name="first" joint="hinge"/><motor name="duplicate" joint="hinge"/>
        <general name="site_force" site="tip" gear="1 0 0 0 0 0"/>
      </actuator></mujoco>""")
    inspection = inspect_mujoco_xml(xml)
    assert inspection["actuated_joint_names"] == ["hinge", "hinge"]
    assert inspection["body_names"] == ["base", "link"]
    joints, warnings = infer_control_joints_from_xml(inspection["model"])
    assert joints == ["hinge"] and len(warnings) == 2
    assert "site_force" in warnings[0]
    assert infer_freejoint_base_body(inspection["model"]) == ("base", "root", [])

    xml.write_text(xml.read_text().replace('<freejoint name="root"/>', ''))
    inspection = inspect_mujoco_xml(xml)
    assert inspection["free_joint"] is None and inspection["free_body"] is None
    base, joint, warnings = infer_freejoint_base_body(inspection["model"])
    assert base == "base" and joint is None and "No freejoint" in warnings[0]
